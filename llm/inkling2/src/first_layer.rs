//! Diagnostic native physical-layer0 graph, chronological single-token execution.
//! This is not the 48-visit language model. Captured outputs require CPU validation.
use crate::{
    config::MiniConfig, device::DeviceWeights, ops::Ops, plan::ExecutionPlan,
    primitives::Primitives, quant::Quant, store::HostStore, Result,
};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::GType;
use std::{collections::BTreeMap, path::Path, sync::Arc};
const H: usize = 2304;
pub struct FirstLayer {
    gpu: Arc<Gpu>,
    store: HostStore,
    weights: DeviceWeights,
    plan: ExecutionPlan,
    primitive: Primitives,
    quant: Quant,
    ops: Ops,
    work: BTreeMap<&'static str, DevBuf>,
    position: usize,
    poisoned: bool,
}
impl FirstLayer {
    pub fn load(
        model: &Path,
        config: &MiniConfig,
        gpu: &Arc<Gpu>,
        context: usize,
        host_available: u64,
        mut cancel: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        if context == 0 || context > 1024 {
            return Err("first-layer diagnostic context must be1..1024".into());
        }
        let primitive = Primitives::load(gpu)?;
        let quant = Quant::load(gpu)?;
        let ops = Ops::load(gpu)?;
        let plan = ExecutionPlan::new(config, context)?;
        let mut store = HostStore::open(model, host_available)?;
        let weights = DeviceWeights::load(&mut store, gpu, &plan, host_available, &mut cancel)?;
        let sizes = [
            ("hidden", H),
            ("norm", H),
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
            ("attn", 3072),
            ("gate", 3072),
            ("gated", 3072),
            ("projected", H),
            ("attnconv", H),
            ("states", 2 * H),
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
            ("output", H),
            ("scores", 24 * context),
        ];
        let bytes = sizes.iter().try_fold(0usize, |sum, (_, n)| {
            sum.checked_add(n * 4).ok_or("work bytes overflow")
        })?;
        if bytes as u64 > weights.estimate.activation_reserve {
            return Err("work buffers exceed reserved activation budget".into());
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
            primitive,
            quant,
            ops,
            work,
            position: 0,
            poisoned: false,
        })
    }
    fn b(&self, name: &str) -> &DevBuf {
        &self.work[name]
    }
    fn linear(&self, name: &str, x: &DevBuf, out: &DevBuf) -> Result<()> {
        let t = self
            .store
            .file()
            .tensor(name)
            .ok_or("missing linear descriptor")?;
        if t.shape.len() != 2 {
            return Err("linear descriptor rank".into());
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
    /// A failed/cancelled token poisons this graph because history may be changed.
    /// `capture` runs after synchronized native output; it performs no model math.
    pub fn token(
        &mut self,
        token: u32,
        mut cancel: impl FnMut() -> Result<()>,
        mut capture: impl FnMut(&str, &DevBuf) -> Result<()>,
    ) -> Result<()> {
        if self.poisoned || self.position >= self.plan.context() || token >= 200058 {
            return Err("invalid token/poisoned/exhausted diagnostic".into());
        }
        let result = self.run(token, &mut cancel, &mut capture);
        match result {
            Ok(()) => {
                self.position += 1;
                Ok(())
            }
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }
    fn run(
        &mut self,
        token: u32,
        cancel: &mut impl FnMut() -> Result<()>,
        capture: &mut impl FnMut(&str, &DevBuf) -> Result<()>,
    ) -> Result<()> {
        cancel()?;
        let tap = |name: &str,
                   b: &DevBuf,
                   capture: &mut dyn FnMut(&str, &DevBuf) -> Result<()>|
         -> Result<()> {
            self.gpu.sync().map_err(|e| e.0)?;
            capture(name, b)
        };
        let row = self
            .weights
            .tensor("token_embd.weight")?
            .view(token as usize * 2448, 2448)
            .map_err(|e| e.0)?;
        self.quant
            .decode(&row, GType::Q8_0, 1, H, self.b("hidden"))?;
        tap("embedding_decoded", self.b("hidden"), capture)?;
        self.primitive.rms(
            self.b("hidden"),
            Some(self.weights.tensor("token_embd_norm.weight")?),
            1,
            H,
            self.b("hidden"),
        )?;
        tap("embedding_normalized", self.b("hidden"), capture)?;
        self.primitive.rms(
            self.b("hidden"),
            Some(self.weights.tensor("blk.0.attn_norm.weight")?),
            1,
            H,
            self.b("norm"),
        )?;
        tap("attention_input_norm", self.b("norm"), capture)?;
        for (weight, out, label) in [
            ("attn_q", "qraw", "Q_raw"),
            ("attn_k", "kraw", "K_raw"),
            ("attn_v", "vraw", "V_raw"),
            ("attn_rel", "rraw", "R_raw"),
        ] {
            self.linear(
                &format!("blk.0.{weight}.weight"),
                self.b("norm"),
                self.b(out),
            )?;
            tap(label, self.b(out), capture)?;
        }
        let history = |offset: usize, n: usize| {
            self.weights
                .state
                .history
                .view(offset * 3 * 4, n * 3 * 4)
                .map_err(|e| e.0)
        };
        self.ops.conv(
            self.b("kraw"),
            self.weights.tensor("blk.0.attn_k_conv.weight")?,
            None,
            8,
            1024,
            &history(0, 1024)?,
            self.b("kc"),
        )?;
        self.ops.conv(
            self.b("vraw"),
            self.weights.tensor("blk.0.attn_v_conv.weight")?,
            None,
            8,
            1024,
            &history(1024, 1024)?,
            self.b("vc"),
        )?;
        tap("K_shortconv", self.b("kc"), capture)?;
        tap("V_shortconv", self.b("vc"), capture)?;
        self.primitive.clip(
            self.b("qraw"),
            Some(self.weights.tensor("blk.0.attn_q_clip.weight")?),
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
            Some(self.weights.tensor("blk.0.attn_rel_clip.weight")?),
            24,
            16,
            8.0,
            self.b("r"),
        )?;
        for (label, name) in [
            ("Q_clipped", "q"),
            ("K_clipped", "k"),
            ("V_normalized", "v"),
            ("R_clipped", "r"),
        ] {
            tap(label, self.b(name), capture)?;
        }
        let slot = &self.weights.state.layout.slots[0];
        let row = slot.write_row(self.position, self.plan.context())?;
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
        keys.copy_within(row * 1024 * 4, self.b("k"), 0, 1024 * 4)
            .map_err(|e| e.0)?;
        vals.copy_within(row * 1024 * 4, self.b("v"), 0, 1024 * 4)
            .map_err(|e| e.0)?;
        self.linear(
            "blk.0.attn_sink_proj.weight",
            self.b("norm"),
            self.b("sink"),
        )?;
        tap("sink_logits", self.b("sink"), capture)?;
        self.ops.attention(
            self.b("q"),
            &keys,
            &vals,
            self.b("r"),
            self.weights.tensor("blk.0.attn_rel_bias.weight")?,
            self.b("sink"),
            self.position,
            slot.capacity,
            true,
            self.b("attn"),
            self.b("scores"),
        )?;
        tap("attention_output", self.b("attn"), capture)?;
        self.linear("blk.0.attn_gate.weight", self.b("norm"), self.b("gate"))?;
        tap("attention_gate_logits", self.b("gate"), capture)?;
        self.primitive
            .gate(self.b("attn"), self.b("gate"), 3072, self.b("gated"))?;
        tap("attention_gated", self.b("gated"), capture)?;
        self.linear(
            "blk.0.attn_output.weight",
            self.b("gated"),
            self.b("projected"),
        )?;
        tap("attention_projected", self.b("projected"), capture)?;
        self.ops.conv(
            self.b("projected"),
            self.weights.tensor("blk.0.attn_post_conv.weight")?,
            None,
            1,
            H,
            &history(2048, H)?,
            self.b("attnconv"),
        )?;
        tap("attention_postconv", self.b("attnconv"), capture)?;
        tap("ffn_input", self.b("attnconv"), capture)?;
        // Block0 starts a new residual stream: saved embedding is a depth state,
        // not added directly to the attention output.
        self.b("states")
            .copy_within(0, self.b("hidden"), 0, H * 4)
            .map_err(|e| e.0)?;
        self.b("states")
            .copy_within(H * 4, self.b("attnconv"), 0, H * 4)
            .map_err(|e| e.0)?;
        self.ops.mix(
            self.b("states"),
            self.weights.tensor("blk.0.ffn_res_query.weight")?,
            2,
            1,
            self.b("mixed"),
        )?;
        tap("ffn_depthmix", self.b("mixed"), capture)?;
        self.primitive.rms(
            self.b("mixed"),
            Some(self.weights.tensor("blk.0.ffn_norm.weight")?),
            1,
            H,
            self.b("ffnorm"),
        )?;
        tap("ffn_norm", self.b("ffnorm"), capture)?;
        self.linear(
            "blk.0.ffn_gate_inp.weight",
            self.b("ffnorm"),
            self.b("router"),
        )?;
        tap("router_logits", self.b("router"), capture)?;
        self.ops.router(
            self.b("router"),
            self.weights.tensor("blk.0.exp_probs_b.bias")?,
            self.b("ids"),
            self.b("routing"),
        )?;
        tap("router_ids", self.b("ids"), capture)?;
        tap("router_weights", self.b("routing"), capture)?;
        let mut raw = [0u8; 32];
        self.b("ids").read(0, &mut raw).map_err(|e| e.0)?;
        let ids: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut unique = std::collections::BTreeSet::new();
        if ids
            .iter()
            .any(|&i| !(0..128).contains(&i) || !unique.insert(i))
        {
            return Err("invalid native router IDs".into());
        }
        self.b("sum").fill(0).map_err(|e| e.0)?;
        // Split fields to keep pair views inside the synchronous upload callback.
        let work = &self.work;
        let q = &self.quant;
        let p = &self.primitive;
        let ops = &self.ops;
        let gpu = &self.gpu;
        let visit = self.plan.visits()[0];
        for (i, &expert) in ids.iter().enumerate() {
            cancel()?;
            self.weights.with_expert(
                &mut self.store,
                &visit,
                expert as usize,
                &mut *cancel,
                |pair| {
                    q.matvec(
                        &pair.gate_up,
                        GType::NVFP4,
                        2304,
                        H,
                        &work["ffnorm"],
                        pair.gate_scale.as_ref(),
                        &work["gu"],
                    )?;
                    gpu.sync().map_err(|e| e.0)?;
                    capture(&format!("expert{i}_gate_up_scaled"), &work["gu"])?;
                    let gate = work["gu"].view(0, 1152 * 4).map_err(|e| e.0)?;
                    let up = work["gu"].view(1152 * 4, 1152 * 4).map_err(|e| e.0)?;
                    p.situ(&gate, &up, 1152, &work["situ"])?;
                    gpu.sync().map_err(|e| e.0)?;
                    capture(&format!("expert{i}_situ"), &work["situ"])?;
                    q.matvec(
                        &pair.down,
                        GType::NVFP4,
                        H,
                        1152,
                        &work["situ"],
                        pair.down_scale.as_ref(),
                        &work["down"],
                    )?;
                    gpu.sync().map_err(|e| e.0)?;
                    capture(&format!("expert{i}_down_scaled"), &work["down"])?;
                    let weight = work["routing"].view(i * 4, 4).map_err(|e| e.0)?;
                    ops.axpy(&work["down"], &weight, H, &work["sum"])
                },
            )?;
        }
        self.gpu.sync().map_err(|e| e.0)?;
        capture("expert_weighted_sum", self.b("sum"))?;
        let history = self
            .weights
            .state
            .history
            .view((2048 + H) * 3 * 4, H * 3 * 4)
            .map_err(|e| e.0)?;
        self.ops.conv(
            self.b("sum"),
            self.weights.tensor("blk.0.ffn_post_conv.weight")?,
            None,
            1,
            H,
            &history,
            self.b("ffconv"),
        )?;
        self.gpu.sync().map_err(|e| e.0)?;
        capture("ffn_postconv", self.b("ffconv"))?;
        self.ops
            .add(self.b("attnconv"), self.b("ffconv"), H, self.b("output"))?;
        self.gpu.sync().map_err(|e| e.0)?;
        capture("block_output", self.b("output"))?;
        cancel()?;
        Ok(())
    }
}
