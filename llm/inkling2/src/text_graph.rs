//! Native F32 chronological text graph. Operator and first-layer gates do not
//! establish this complete graph's quality; no engine registration is provided.
use crate::{
    config::MiniConfig,
    device::DeviceWeights,
    ops::Ops,
    plan::{Attention, ExecutionPlan, Operation, Visit},
    primitives::Primitives,
    quant::Quant,
    session::Cursor,
    store::HostStore,
    Result,
};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::GType;
use std::{collections::BTreeMap, path::Path, sync::Arc};
const H: usize = 2304;
pub struct TextGraph {
    gpu: Arc<Gpu>,
    store: HostStore,
    weights: DeviceWeights,
    plan: ExecutionPlan,
    cursor: Cursor,
    primitive: Primitives,
    quant: Quant,
    ops: Ops,
    work: BTreeMap<&'static str, DevBuf>,
}
impl TextGraph {
    pub fn load(
        model: &Path,
        config: &MiniConfig,
        gpu: &Arc<Gpu>,
        context: usize,
        host_available: u64,
        mut cancel: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        Self::load_with_expert_mode(
            model,
            config,
            gpu,
            context,
            host_available,
            crate::expert_cache::ExpertMode::Stream,
            &mut cancel,
        )
    }
    pub fn load_with_expert_mode(
        model: &Path,
        config: &MiniConfig,
        gpu: &Arc<Gpu>,
        context: usize,
        host_available: u64,
        mode: crate::expert_cache::ExpertMode,
        mut cancel: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        if context == 0 || context > 65536 {
            return Err("initial text graph context must be1..65536".into());
        }
        let primitive = Primitives::load(gpu)?;
        let quant = Quant::load(gpu)?;
        let ops = Ops::load(gpu)?;
        let plan = ExecutionPlan::new(config, context)?;
        let cursor = Cursor::new(plan.clone());
        let mut store = HostStore::open(model, host_available)?;
        let weights = DeviceWeights::load_with_expert_mode(
            &mut store,
            gpu,
            &plan,
            host_available,
            mode,
            &mut cancel,
        )?;
        let sizes = [
            ("hidden", H),
            ("input", H),
            ("norms", 3 * H),
            ("attnmix", 3 * H),
            ("controlled", H),
            ("qraw", 3072),
            ("kraw", 1024),
            ("vraw", 1024),
            ("rraw", 384),
            ("kc", 1024),
            ("vc", 1024),
            ("q", 3072),
            ("k", 1024),
            ("v", 1024),
            ("r", 384),
            ("sink", 24),
            ("dynk", 32),
            ("dynv", 32),
            ("attn", 3072),
            ("gate", 3072),
            ("gated", 3072),
            ("projected", H),
            ("attnconv", H),
            ("ffinput", H),
            ("states", 5 * H),
            ("blocks", 4 * H),
            ("prelude_blocks", 4 * H),
            ("prelude", H),
            ("bank", 2 * H),
            ("messages", 4 * H),
            ("message_pair", 2 * H),
            ("control_concat", 2 * H),
            ("rank", 64),
            ("gain", H),
            ("forget", H),
            ("mixed", H),
            ("ffnorm", H),
            ("router", 128),
            ("ids", 8),
            ("routing", 8),
            ("gu", 2304),
            ("situ", 1152),
            ("down", H),
            ("sum", H),
            ("ffconv", H),
            ("scores", 24 * context),
            ("logits", 201024),
        ];
        let bytes = sizes.iter().try_fold(0usize, |sum, (_, n)| {
            sum.checked_add(n * 4).ok_or("activation size overflow")
        })?;
        if bytes as u64 > weights.estimate.activation_reserve {
            return Err("text work exceeds reserved activation budget".into());
        }
        let mut work = BTreeMap::new();
        for (name, n) in sizes {
            cancel()?;
            work.insert(name, DevBuf::new(gpu, n * 4).map_err(|e| e.0)?);
        }
        Ok(Self {
            gpu: gpu.clone(),
            store,
            weights,
            plan,
            cursor,
            primitive,
            quant,
            ops,
            work,
        })
    }
    pub fn expert_stats(&self) -> crate::expert_cache::ExpertStats {
        self.weights.expert_stats()
    }
    pub fn expert_mode(&self) -> crate::expert_cache::ExpertMode {
        self.weights.expert_mode()
    }
    fn b(&self, n: &str) -> &DevBuf {
        &self.work[n]
    }
    fn view(&self, n: &str, offset: usize, elements: usize) -> Result<DevBuf> {
        self.b(n).view(offset * 4, elements * 4).map_err(|e| e.0)
    }
    fn copy(&self, dst: &str, dst_at: usize, src: &str, src_at: usize, n: usize) -> Result<()> {
        self.b(dst)
            .copy_within(dst_at * 4, self.b(src), src_at * 4, n * 4)
            .map_err(|e| e.0)
    }
    fn linear(&self, name: &str, x: &DevBuf, out: &DevBuf) -> Result<()> {
        let t = self
            .store
            .file()
            .tensor(name)
            .ok_or("missing matrix descriptor")?;
        if t.shape.len() != 2 {
            return Err("linear rank mismatch".into());
        }
        self.quant.matvec(
            self.weights.tensor(name)?,
            t.ty,
            t.shape[0] as usize,
            t.shape[1] as usize,
            x,
            None,
            out,
        )
    }
    /// Populate original GGML depth ordering: completed blocks reversed, then cur.
    fn pack_depth(&self, count: usize, current: &DevBuf) -> Result<()> {
        if count > 4 {
            return Err("too many residual depth blocks".into());
        }
        for i in 0..count {
            self.copy("states", i * H, "blocks", (count - 1 - i) * H, H)?;
        }
        self.b("states")
            .copy_within(count * H * 4, current, 0, H * 4)
            .map_err(|e| e.0)
    }
    fn control(
        &self,
        layer: usize,
        module: &str,
        current: &DevBuf,
        message: &DevBuf,
        out: &DevBuf,
    ) -> Result<()> {
        let left = self.view("control_concat", 0, H)?;
        let right = self.view("control_concat", H, H)?;
        self.primitive.rms(current, None, 1, H, &left)?;
        self.primitive.rms(message, None, 1, H, &right)?;
        self.linear(
            &format!("blk.{layer}.loop_{module}_down.weight"),
            self.b("control_concat"),
            self.b("rank"),
        )?;
        self.linear(
            &format!("blk.{layer}.loop_{module}_gain.weight"),
            self.b("rank"),
            self.b("gain"),
        )?;
        self.linear(
            &format!("blk.{layer}.loop_{module}_forget.weight"),
            self.b("rank"),
            self.b("forget"),
        )?;
        self.ops
            .control(current, message, self.b("gain"), self.b("forget"), out)
    }
    fn messages(&self, layer: usize) -> Result<()> {
        let offset = 2 * (layer - 6) * H * 4;
        let aq = self
            .weights
            .tensor("loop_res_query.weight")?
            .view(offset, 2 * H * 4)
            .map_err(|e| e.0)?;
        let kq = self
            .weights
            .tensor("loop_kv_res_query.weight")?
            .view(offset, 2 * H * 4)
            .map_err(|e| e.0)?;
        self.ops
            .mix(self.b("bank"), &aq, 2, 2, self.b("message_pair"))?;
        self.copy("messages", 0, "message_pair", 0, H)?;
        self.copy("messages", 3 * H, "message_pair", H, H)?;
        self.ops
            .mix(self.b("bank"), &kq, 2, 2, self.b("message_pair"))?;
        self.copy("messages", H, "message_pair", 0, 2 * H)
    }
    /// Caller provides real token IDs from the separately validated tokenizer.
    /// Any partial execution failure poisons the cursor and requires a fresh graph.
    /// Observe receives one completed visit output; it must not retain buffer views.
    pub fn token(
        &mut self,
        token: u32,
        mut cancel: impl FnMut() -> Result<()>,
        mut observe: impl FnMut(&Visit, &DevBuf) -> Result<()>,
    ) -> Result<Vec<f32>> {
        if token >= 200058 {
            return Err("token outside unpadded vocabulary".into());
        }
        let pos = self.cursor.begin()?;
        let result = self.run(token, pos, &mut cancel, &mut observe);
        match result {
            Ok(logits) => {
                if let Err(e) = self.cursor.commit() {
                    self.cursor.abort();
                    return Err(e);
                }
                Ok(logits)
            }
            Err(e) => {
                self.cursor.abort();
                Err(e)
            }
        }
    }
    fn run(
        &mut self,
        token: u32,
        pos: usize,
        cancel: &mut impl FnMut() -> Result<()>,
        observe: &mut impl FnMut(&Visit, &DevBuf) -> Result<()>,
    ) -> Result<Vec<f32>> {
        cancel()?;
        let row = self
            .weights
            .tensor("token_embd.weight")?
            .view(token as usize * 2448, 2448)
            .map_err(|e| e.0)?;
        self.quant
            .decode(&row, GType::Q8_0, 1, H, self.b("hidden"))?;
        self.primitive.rms(
            self.b("hidden"),
            Some(self.weights.tensor("token_embd_norm.weight")?),
            1,
            H,
            self.b("hidden"),
        )?;
        let (mut count, mut prelude_count) = (0usize, 0usize);
        while let Some(op) = self.cursor.current()?.cloned() {
            cancel()?;
            match op {
                Operation::SavePrelude => {
                    self.copy("prelude", 0, "hidden", 0, H)?;
                    self.copy("prelude_blocks", 0, "blocks", 0, count * H)?;
                    prelude_count = count;
                }
                Operation::ResetToPrelude { message_values, .. } => {
                    if message_values != 2 || count != prelude_count + 2 {
                        return Err("loop bank layout mismatch".into());
                    }
                    // Exclude prelude and the core's saved input: retain completed core block1
                    // and partial block2 exactly as merged graph bank.assign(begin+prelude+1).
                    self.copy("bank", 0, "blocks", (prelude_count + 1) * H, H)?;
                    self.copy("bank", H, "hidden", 0, H)?;
                    self.copy("hidden", 0, "prelude", 0, H)?;
                    self.copy("blocks", 0, "prelude_blocks", 0, prelude_count * H)?;
                    count = prelude_count;
                }
                Operation::Layer(v) => {
                    self.layer(v, pos, &mut count, cancel)?;
                    self.gpu.sync().map_err(|e| e.0)?;
                    observe(&v, self.b("hidden"))?;
                }
                Operation::FinalResidualMixAndNorm => {
                    self.pack_depth(count, self.b("hidden"))?;
                    self.ops.mix(
                        self.b("states"),
                        self.weights.tensor("output_res_query.weight")?,
                        count + 1,
                        1,
                        self.b("mixed"),
                    )?;
                    self.primitive.rms(
                        self.b("mixed"),
                        Some(self.weights.tensor("output_norm.weight")?),
                        1,
                        H,
                        self.b("hidden"),
                    )?;
                    self.linear("output.weight", self.b("hidden"), self.b("logits"))?;
                    self.primitive
                        .softcap(self.b("logits"), 200058, self.b("logits"))?;
                }
            }
            self.cursor.complete_operation()?;
        }
        let mut bytes = vec![0u8; 200058 * 4];
        self.b("logits").read(0, &mut bytes).map_err(|e| e.0)?;
        let logits: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if logits.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite text logits".into());
        }
        cancel()?;
        Ok(logits)
    }
    fn layer(
        &mut self,
        v: Visit,
        pos: usize,
        count: &mut usize,
        cancel: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let l = v.physical_layer;
        let local = v.attention == Attention::Sliding;
        let name = |suffix: &str| format!("blk.{l}.{suffix}.weight");
        self.copy("input", 0, "hidden", 0, H)?;
        if v.previous_pass_messages {
            self.messages(l)?;
        }
        if l == 0 {
            self.copy("attnmix", 0, "hidden", 0, H)?;
        } else {
            self.pack_depth(*count, self.b("hidden"))?;
            self.ops.mix(
                self.b("states"),
                self.weights.tensor(&name("attn_res_query"))?,
                *count + 1,
                if local { 1 } else { 3 },
                self.b("attnmix"),
            )?;
        }
        for j in 0..3 {
            let raw = self.view("attnmix", if local { 0 } else { j * H }, H)?;
            let norm = self.view("norms", j * H, H)?;
            let input = if v.previous_pass_messages {
                let message = self.view("messages", j * H, H)?;
                self.control(
                    l,
                    ["attn", "k", "v"][j],
                    &raw,
                    &message,
                    self.b("controlled"),
                )?;
                self.b("controlled")
            } else {
                &raw
            };
            let norm_name = if local {
                "attn_norm"
            } else {
                ["attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm"][j]
            };
            self.primitive.rms(
                input,
                Some(self.weights.tensor(&name(norm_name))?),
                1,
                H,
                &norm,
            )?;
        }
        let (qin, kin, vin) = (
            self.view("norms", 0, H)?,
            self.view("norms", H, H)?,
            self.view("norms", 2 * H, H)?,
        );
        self.linear(&name("attn_q"), &qin, self.b("qraw"))?;
        self.linear(&name("attn_k"), &kin, self.b("kraw"))?;
        self.linear(&name("attn_v"), &vin, self.b("vraw"))?;
        self.linear(&name("attn_rel"), &qin, self.b("rraw"))?;
        if !local {
            self.linear(&name("attn_k_conv_proj"), &kin, self.b("dynk"))?;
            self.linear(&name("attn_v_conv_proj"), &vin, self.b("dynv"))?;
        }
        let slot = self.weights.state.layout.slots[v.ordinal].clone();
        let history = |offset: usize, n: usize| {
            self.weights
                .state
                .history
                .view((slot.history_offset + offset * 3) * 4, n * 3 * 4)
                .map_err(|e| e.0)
        };
        self.ops.conv(
            self.b("kraw"),
            self.weights.tensor(&name("attn_k_conv"))?,
            if local { None } else { Some(self.b("dynk")) },
            8,
            1024,
            &history(0, 1024)?,
            self.b("kc"),
        )?;
        self.ops.conv(
            self.b("vraw"),
            self.weights.tensor(&name("attn_v_conv"))?,
            if local { None } else { Some(self.b("dynv")) },
            8,
            1024,
            &history(1024, 1024)?,
            self.b("vc"),
        )?;
        self.primitive.clip(
            self.b("qraw"),
            Some(self.weights.tensor(&name("attn_q_clip"))?),
            24,
            128,
            22.627416997969522f32,
            self.b("q"),
        )?;
        self.primitive.clip(
            self.b("kc"),
            None,
            8,
            128,
            22.627416997969522f32,
            self.b("k"),
        )?;
        self.primitive
            .rms(self.b("vc"), None, 8, 128, self.b("v"))?;
        self.primitive.clip(
            self.b("rraw"),
            Some(self.weights.tensor(&name("attn_rel_clip"))?),
            24,
            16,
            8.0,
            self.b("r"),
        )?;
        if !local {
            // Explicit F32 source formula; transcendental order still needs full-graph gate.
            let ratio = ((pos + 1) as f32 / 8192.0f32).max(1.0);
            let tau = 1.0f32 + 0.1f32 * ratio.ln();
            self.ops.scale(self.b("q"), 3072, tau, self.b("q"))?;
            self.ops.scale(self.b("r"), 384, tau, self.b("r"))?;
        }
        let keys = self
            .weights
            .state
            .kv
            .view(slot.key_offset * 4, slot.capacity * 1024 * 4)
            .map_err(|e| e.0)?;
        let vals = self
            .weights
            .state
            .kv
            .view(slot.value_offset * 4, slot.capacity * 1024 * 4)
            .map_err(|e| e.0)?;
        let row = slot.write_row(pos, self.plan.context())?;
        keys.copy_within(row * 4096, self.b("k"), 0, 4096)
            .map_err(|e| e.0)?;
        vals.copy_within(row * 4096, self.b("v"), 0, 4096)
            .map_err(|e| e.0)?;
        self.linear(&name("attn_sink_proj"), &qin, self.b("sink"))?;
        self.ops.attention(
            self.b("q"),
            &keys,
            &vals,
            self.b("r"),
            self.weights.tensor(&name("attn_rel_bias"))?,
            self.b("sink"),
            pos,
            slot.capacity,
            local,
            self.b("attn"),
            self.b("scores"),
        )?;
        self.linear(&name("attn_gate"), &qin, self.b("gate"))?;
        self.primitive
            .gate(self.b("attn"), self.b("gate"), 3072, self.b("gated"))?;
        self.linear(&name("attn_output"), self.b("gated"), self.b("projected"))?;
        self.ops.conv(
            self.b("projected"),
            self.weights.tensor(&name("attn_post_conv"))?,
            None,
            1,
            H,
            &history(2048, H)?,
            self.b("attnconv"),
        )?;
        if v.append_input_residual_block {
            if *count >= 4 {
                return Err("residual block overflow".into());
            }
            self.copy("blocks", *count * H, "input", 0, H)?;
            *count += 1;
            self.copy("ffinput", 0, "attnconv", 0, H)?;
        } else {
            self.ops
                .add(self.b("input"), self.b("attnconv"), H, self.b("ffinput"))?;
        }
        self.pack_depth(*count, self.b("ffinput"))?;
        self.ops.mix(
            self.b("states"),
            self.weights.tensor(&name("ffn_res_query"))?,
            *count + 1,
            1,
            self.b("mixed"),
        )?;
        let input = if v.previous_pass_messages {
            let message = self.view("messages", 3 * H, H)?;
            self.control(l, "mlp", self.b("mixed"), &message, self.b("controlled"))?;
            self.b("controlled")
        } else {
            self.b("mixed")
        };
        self.primitive.rms(
            input,
            Some(self.weights.tensor(&name("ffn_norm"))?),
            1,
            H,
            self.b("ffnorm"),
        )?;
        self.linear(&name("ffn_gate_inp"), self.b("ffnorm"), self.b("router"))?;
        self.ops.router(
            self.b("router"),
            self.weights.tensor(&format!("blk.{l}.exp_probs_b.bias"))?,
            self.b("ids"),
            self.b("routing"),
        )?;
        let mut raw = [0u8; 32];
        self.b("ids").read(0, &mut raw).map_err(|e| e.0)?;
        let ids: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut unique = std::collections::BTreeSet::new();
        if ids
            .iter()
            .any(|&id| !(0..128).contains(&id) || !unique.insert(id))
        {
            return Err("invalid router IDs".into());
        }
        self.b("sum").fill(0).map_err(|e| e.0)?;
        let work = &self.work;
        let quant = &self.quant;
        let primitive = &self.primitive;
        let ops = &self.ops;
        let (gu_type, down_type) = if l < 20 {
            (GType::NVFP4, GType::NVFP4)
        } else {
            (GType::Q5K, GType::Q8_0)
        };
        for (i, &expert) in ids.iter().enumerate() {
            cancel()?;
            self.weights.with_expert(
                &mut self.store,
                &v,
                expert as usize,
                &mut *cancel,
                |pair| {
                    quant.matvec(
                        &pair.gate_up,
                        gu_type,
                        2304,
                        H,
                        &work["ffnorm"],
                        pair.gate_scale.as_ref(),
                        &work["gu"],
                    )?;
                    let gate = work["gu"].view(0, 1152 * 4).map_err(|e| e.0)?;
                    let up = work["gu"].view(1152 * 4, 1152 * 4).map_err(|e| e.0)?;
                    primitive.situ(&gate, &up, 1152, &work["situ"])?;
                    quant.matvec(
                        &pair.down,
                        down_type,
                        H,
                        1152,
                        &work["situ"],
                        pair.down_scale.as_ref(),
                        &work["down"],
                    )?;
                    let weight = work["routing"].view(i * 4, 4).map_err(|e| e.0)?;
                    ops.axpy(&work["down"], &weight, H, &work["sum"])
                },
            )?;
        }
        let history = self
            .weights
            .state
            .history
            .view((slot.history_offset + (2048 + H) * 3) * 4, H * 3 * 4)
            .map_err(|e| e.0)?;
        self.ops.conv(
            self.b("sum"),
            self.weights.tensor(&name("ffn_post_conv"))?,
            None,
            1,
            H,
            &history,
            self.b("ffconv"),
        )?;
        self.ops
            .add(self.b("ffinput"), self.b("ffconv"), H, self.b("hidden"))?;
        Ok(())
    }
}

#[path = "graph_state.rs"]
mod graph_state;
