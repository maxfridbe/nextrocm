//! Text-only F32 Qwen3-VL, adapted from nextsycl98124afb and the retained Comfy
//! CPU reference. Private and unregistered; all36 blocks are streamed one at a
//! time. No vision tower, LM head, final norm, generation, or edit support.
mod ffi;
mod weights;
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_tok::Tokenizer;
use std::{path::Path, sync::Arc};
use weights::{Matrix, Weights, BLOCK_BYTES, FFN, HIDDEN, LAYERS};
pub const SYSTEM: &str =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
/// Explicit private experiment settings, copied at load; no ambient environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadOptions {
    pub probability_first: bool,
    pub ordered_headnorm: bool,
    pub ordered_rms: bool,
    pub score_division: bool,
}
pub struct Budget {
    pub allowance_bytes: usize,
    pub driver_margin_bytes: usize,
}
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    pub tracked_bytes: usize,
    pub with_margin_bytes: usize,
    pub block_bytes: usize,
}
impl Plan {
    pub fn new(tokens: usize, budget: &Budget) -> Result<Self> {
        if tokens == 0
            || tokens > 4096
            || budget.allowance_bytes >= 8_000_000_000
            || budget.driver_margin_bytes < 512 * 1024 * 1024
        {
            return Err(Error(
                "TE requires1..4096tokens, explicit <8GB allowance and >=512MiB driver margin"
                    .into(),
            ));
        }
        // All working activations + max linear scratch + conservative full-size
        // cropped output + one weight block. Output allocation overlaps final x.
        let tracked_bytes = tokens
            .checked_mul(414_212)
            .and_then(|b| b.checked_add(4 + BLOCK_BYTES))
            .ok_or_else(|| Error("TE allocation overflow".into()))?;
        let with_margin_bytes = tracked_bytes
            .checked_add(budget.driver_margin_bytes)
            .ok_or_else(|| Error("TE margin overflow".into()))?;
        if with_margin_bytes > budget.allowance_bytes {
            return Err(Error(format!(
                "TE planned {with_margin_bytes} bytes exceeds allowance{}",
                budget.allowance_bytes
            )));
        }
        Ok(Self {
            tracked_bytes,
            with_margin_bytes,
            block_bytes: BLOCK_BYTES,
        })
    }
}
pub struct Encoded {
    /// Cropped[S,4096] F32 after layer36 and before final norm.
    pub embeddings: DevBuf,
    pub ids: Vec<u32>,
    pub drop: usize,
    pub image_pad: Vec<bool>,
    pub plan: Plan,
}
pub struct TextEncoder {
    weights: Weights,
    tok: Tokenizer,
    gpu: Arc<Gpu>,
    probability_first: bool,
    ordered_headnorm: bool,
    ordered_rms: bool,
    score_division: bool,
    attention_tree32: bool,
    attention_source32: bool,
    attention_order32: bool,
    silu_source: bool,
}
/// Diagnostic-only readback hook. Byte counts exclude unused scratch capacity.
pub type LayerTrace<'a> = dyn FnMut(&str, &DevBuf, usize) -> Result<()> + 'a;
fn capture(
    trace: &mut Option<&mut LayerTrace<'_>>,
    gpu: &Gpu,
    name: &str,
    buf: &DevBuf,
    bytes: usize,
) -> Result<()> {
    if let Some(callback) = trace.as_deref_mut() {
        gpu.sync()?;
        callback(name, buf, bytes)?;
    }
    Ok(())
}
struct Scratch {
    rot: DevBuf,
    q: DevBuf,
    rs: DevBuf,
    acc: DevBuf,
    status: DevBuf,
}
impl Scratch {
    fn capture(
        &self,
        trace: &mut Option<&mut LayerTrace<'_>>,
        gpu: &Gpu,
        name: &str,
        s: usize,
        w: &Matrix,
        out: &DevBuf,
    ) -> Result<()> {
        capture(
            trace,
            gpu,
            &format!("{name}.rotated"),
            &self.rot,
            s * w.k * 4,
        )?;
        capture(trace, gpu, &format!("{name}.quantized"), &self.q, s * w.k)?;
        capture(trace, gpu, &format!("{name}.row_scale"), &self.rs, s * 4)?;
        capture(trace, gpu, &format!("{name}.accum"), &self.acc, s * w.n * 4)?;
        capture(trace, gpu, &format!("{name}.raw"), out, s * w.n * 4)
    }
    fn new(gpu: &Arc<Gpu>, s: usize) -> Result<Self> {
        Ok(Self {
            rot: DevBuf::f32(gpu, s * FFN)?,
            q: DevBuf::new(gpu, s * FFN)?,
            rs: DevBuf::f32(gpu, s)?,
            acc: DevBuf::new(gpu, s * 2 * FFN * 4)?,
            status: DevBuf::new(gpu, 4)?,
        })
    }
    fn linear(&self, gpu: &Arc<Gpu>, x: &DevBuf, s: usize, w: &Matrix, out: &DevBuf) -> Result<()> {
        let args = ffi::Int8Args {
            x: x.ptr(),
            weight: w.w.ptr().cast(),
            weight_scale: w.scales.fp(),
            bias: std::ptr::null(),
            out: out.ptr(),
            rotated: self.rot.fp(),
            quantized: self.q.ptr().cast(),
            row_scale: self.rs.fp(),
            accum: self.acc.ptr().cast(),
            status: self.status.ptr().cast(),
            m: s as i64,
            n: w.n as i64,
            k: w.k as i64,
            scales: w.n as i64,
            dtype: 0,
        };
        // SAFETY: private ABI matchesreprC; all owned buffers cover checked geometry
        // and stay live until the following stream sync/status read.
        ffi::check(
            unsafe { (ffi::api(gpu)?.linear)(gpu.raw(), &args) },
            "TE INT8 linear",
        )?;
        gpu.sync()?;
        let mut status = [0u8; 4];
        self.status.read(0, &mut status)?;
        let status = u32::from_le_bytes(status);
        if status != 0 {
            return Err(Error(format!(
                "TE linear numerical status0x{status:x}; output not consumed"
            )));
        }
        Ok(())
    }
}
impl TextEncoder {
    /// Metadata/tokenizer only; no device weights allocated until encode.
    pub fn load(file: &Path, tokenizer: &Path, config: &Path, gpu: Arc<Gpu>) -> Result<Self> {
        Self::load_with_attention(file, tokenizer, config, gpu, false)
    }
    /// Explicit private experiment, defaultoff; no ambient environment selection.
    pub fn load_with_attention(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        probability_first: bool,
    ) -> Result<Self> {
        Self::load_with_controls(file, tokenizer, config, gpu, probability_first, false)
    }
    /// Independent default-off head-normalization and probability controls.
    pub fn load_with_controls(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        probability_first: bool,
        ordered_headnorm: bool,
    ) -> Result<Self> {
        Self::load_with_options(
            file,
            tokenizer,
            config,
            gpu,
            LoadOptions {
                probability_first,
                ordered_headnorm,
                ..LoadOptions::default()
            },
        )
    }
    /// Independent candidate controls. Existing load methods preserve both new defaults off.
    pub fn load_with_options(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        options: LoadOptions,
    ) -> Result<Self> {
        Self::load_with_tree32(file, tokenizer, config, gpu, options, false)
    }
    /// Private exact32-token experiment; existing load APIs keep this off.
    pub fn load_with_tree32(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        options: LoadOptions,
        attention_tree32: bool,
    ) -> Result<Self> {
        Self::load_with_source32(
            file,
            tokenizer,
            config,
            gpu,
            options,
            attention_tree32,
            false,
        )
    }
    /// Complete source-softmax experiment, immutable and default-off. Exact32 only;
    /// other lengths retain the selected prior attention path. Takes precedence over tree32.
    pub fn load_with_source32(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        options: LoadOptions,
        attention_tree32: bool,
        attention_source32: bool,
    ) -> Result<Self> {
        Self::load_with_source_silu(
            file,
            tokenizer,
            config,
            gpu,
            options,
            attention_tree32,
            attention_source32,
            false,
        )
    }
    /// Independent immutable source-SiLU option. Existing APIs keep it off.
    /// Above the private operator's count cap, retain the prior shared SwiGLU path.
    pub fn load_with_source_silu(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        options: LoadOptions,
        attention_tree32: bool,
        attention_source32: bool,
        silu_source: bool,
    ) -> Result<Self> {
        Self::load_with_order32(
            file,
            tokenizer,
            config,
            gpu,
            options,
            attention_tree32,
            attention_source32,
            silu_source,
            false,
        )
    }
    /// Independent fixed-S32 ordered attention; previous APIs keep it disabled.
    /// Other sequence lengths retain the previously selected attention path.
    pub fn load_with_order32(
        file: &Path,
        tokenizer: &Path,
        config: &Path,
        gpu: Arc<Gpu>,
        options: LoadOptions,
        attention_tree32: bool,
        attention_source32: bool,
        silu_source: bool,
        attention_order32: bool,
    ) -> Result<Self> {
        if attention_order32 && (!options.probability_first || !options.score_division) {
            return Err(Error(
                "ordered attention requires probability-first and score-division controls".into(),
            ));
        }
        if attention_source32 && (!options.probability_first || !options.score_division) {
            return Err(Error(
                "source32 requires probability-first and score-division controls".into(),
            ));
        }
        if attention_tree32 && (!options.probability_first || !options.score_division) {
            return Err(Error(
                "tree32 requires probability-first and score-division controls".into(),
            ));
        }
        let LoadOptions {
            probability_first,
            ordered_headnorm,
            ordered_rms,
            score_division,
        } = options;
        if score_division && !probability_first {
            return Err(Error(
                "score division requires explicit probability-first control".into(),
            ));
        }
        host_memory_preflight(512 * 1024 * 1024)?;
        validate_config(config)?;
        let api = ffi::api(&gpu)?; // Reject missing private symbols before any model allocation.
        if probability_first && api.attention_prob.is_none() {
            return Err(Error(
                "requested private probability-first ABI missing".into(),
            ));
        }
        if ordered_headnorm && api.ordered_rope.is_none() {
            return Err(Error("requested ordered headnorm ABI missing".into()));
        }
        if ordered_rms && api.rms4096.is_none() {
            return Err(Error("requested source RMS4096 ABI missing".into()));
        }
        if score_division && api.attention_divide.is_none() {
            return Err(Error(
                "requested attention score division ABI missing".into(),
            ));
        }
        if attention_tree32 && api.attention_tree32.is_none() {
            return Err(Error("requested tree32 attention ABI missing".into()));
        }
        if attention_source32 && api.attention_source32.is_none() {
            return Err(Error("requested source32 attention ABI missing".into()));
        }
        if attention_order32 && api.attention_order32.is_none() {
            return Err(Error("requested ordered attention ABI missing".into()));
        }
        if silu_source && api.silu_source.is_none() {
            return Err(Error("requested source SiLU ABI missing".into()));
        }
        let tok = Tokenizer::from_hf_json(tokenizer).map_err(Error)?;
        for token in ["<|im_start|>", "<|im_end|>", "<|image_pad|>"] {
            if tok.id(token).is_none() {
                return Err(Error(format!("TE missing special token{token}")));
            }
        }
        Ok(Self {
            attention_order32,
            silu_source,
            attention_source32,
            attention_tree32,
            weights: Weights::open(file)?,
            tok,
            gpu,
            probability_first,
            ordered_headnorm,
            ordered_rms,
            score_division,
        })
    }
    pub fn tokenize(&self, prompt: &str) -> Result<(Vec<u32>, usize)> {
        prompt_ids(&self.tok, prompt)
    }
    /// Stage callback observes embeddings0/layer1..36 for independent CPU capture
    /// and cancellation. It runs after synchronization; its Err aborts safely.
    pub fn encode(
        &self,
        prompt: &str,
        budget: &Budget,
        stage: &mut dyn FnMut(usize, &DevBuf) -> Result<()>,
    ) -> Result<Encoded> {
        self.encode_inner(prompt, budget, stage, None, 0..LAYERS, None)
    }
    /// First-layer diagnostics only; cannot return a purported full TE result.
    pub fn trace_layer0(
        &self,
        prompt: &str,
        budget: &Budget,
        trace: &mut LayerTrace<'_>,
    ) -> Result<()> {
        self.encode_inner(prompt, budget, &mut |_, _| Ok(()), Some(trace), 0..1, None)?;
        Ok(())
    }
    /// Diagnostic block1 replay on caller-supplied full F32 hidden state.
    /// Does not return a full encoder result or change normal encoding.
    pub fn trace_block1(
        &self,
        prompt: &str,
        budget: &Budget,
        input: &[f32],
        trace: &mut LayerTrace<'_>,
    ) -> Result<()> {
        let rows = self.tokenize(prompt)?.0.len();
        if input.len() != rows * HIDDEN || input.iter().any(|x| !x.is_finite()) {
            return Err(Error("diagnostic block1 input shape/finite check".into()));
        }
        self.encode_inner(
            prompt,
            budget,
            &mut |_, _| Ok(()),
            Some(trace),
            1..2,
            Some(input),
        )?;
        Ok(())
    }
    /// Diagnostic block2 replay on caller-supplied full F32 hidden state.
    /// Does not return a full encoder result or change normal encoding.
    pub fn trace_block2(
        &self,
        prompt: &str,
        budget: &Budget,
        input: &[f32],
        trace: &mut LayerTrace<'_>,
    ) -> Result<()> {
        let rows = self.tokenize(prompt)?.0.len();
        if input.len() != rows * HIDDEN || input.iter().any(|x| !x.is_finite()) {
            return Err(Error("diagnostic block2 input shape/finite check".into()));
        }
        self.encode_inner(
            prompt,
            budget,
            &mut |_, _| Ok(()),
            Some(trace),
            2..3,
            Some(input),
        )?;
        Ok(())
    }
    /// Diagnostic block3 replay on caller-supplied full F32 hidden state.
    /// Does not return a full encoder result or change normal encoding.
    pub fn trace_block3(
        &self,
        prompt: &str,
        budget: &Budget,
        input: &[f32],
        trace: &mut LayerTrace<'_>,
    ) -> Result<()> {
        let rows = self.tokenize(prompt)?.0.len();
        if input.len() != rows * HIDDEN || input.iter().any(|x| !x.is_finite()) {
            return Err(Error("diagnostic block3 input shape/finite check".into()));
        }
        self.encode_inner(
            prompt,
            budget,
            &mut |_, _| Ok(()),
            Some(trace),
            3..4,
            Some(input),
        )?;
        Ok(())
    }
    fn encode_inner(
        &self,
        prompt: &str,
        budget: &Budget,
        stage: &mut dyn FnMut(usize, &DevBuf) -> Result<()>,
        mut trace: Option<&mut LayerTrace<'_>>,
        layers: std::ops::Range<usize>,
        input: Option<&[f32]>,
    ) -> Result<Encoded> {
        let (ids, drop) = self.tokenize(prompt)?;
        let s = ids.len();
        let plan = Plan::new(s, budget)?;
        host_memory_preflight(512 * 1024 * 1024 + s * HIDDEN * 4)?;
        let (_, free) = self.gpu.memory()?;
        if free.is_none_or(|f| f < plan.with_margin_bytes as u64) {
            return Err(Error("TE insufficient/unknown free device memory".into()));
        }
        let gpu = &self.gpu;
        let api = ffi::api(gpu)?;
        let rope = if self.ordered_headnorm {
            api.ordered_rope
                .ok_or_else(|| Error("ordered headnorm ABI missing".into()))?
        } else {
            api.rope
        };
        let x = match input {
            Some(values) => DevBuf::from_f32(gpu, values)?,
            None => DevBuf::from_f32(gpu, &self.weights.embeddings(&ids)?)?,
        };
        let hb = DevBuf::f32(gpu, s * HIDDEN)?;
        let qkv = DevBuf::f32(gpu, s * 6144)?;
        let att = DevBuf::f32(gpu, s * HIDDEN)?;
        let out = DevBuf::f32(gpu, s * HIDDEN)?;
        let gu = DevBuf::f32(gpu, s * 2 * FFN)?;
        let act = DevBuf::f32(gpu, s * FFN)?;
        let cs = DevBuf::from_f32(gpu, &rotations(s))?;
        let scratch = Scratch::new(gpu, s)?;
        capture(
            &mut trace,
            gpu,
            if input.is_some() {
                "block_input"
            } else {
                "embed"
            },
            &x,
            s * HIDDEN * 4,
        )?;
        capture(&mut trace, gpu, "cos_sin", &cs, s * 128 * 4)?;
        stage(0, &x)?;
        for layer in layers {
            host_memory_preflight(128 * 1024 * 1024)?;
            let w = self.weights.block(gpu, layer, plan.block_bytes)?;
            capture(&mut trace, gpu, "input_norm.weight", &w.input, HIDDEN * 4)?;
            capture(&mut trace, gpu, "post_norm.weight", &w.post, HIDDEN * 4)?;
            capture(&mut trace, gpu, "qnorm.weight", &w.qnorm, 128 * 4)?;
            capture(&mut trace, gpu, "knorm.weight", &w.knorm, 128 * 4)?;
            let work = (|| -> Result<()> {
                // SAFETY: all pointers belong to live F32 buffers with the model's checked
                // sizes; existing shared routines are called only with supported geometry.
                if self.ordered_rms {
                    let rms = api
                        .rms4096
                        .ok_or_else(|| Error("private source RMS ABI missing".into()))?;
                    // SAFETY: checked rows, disjoint F32 [rows,4096] buffers and 4096 weights.
                    ffi::check(
                        unsafe { rms(gpu.raw(), x.fp(), s as i64, w.input.fp(), hb.fp()) },
                        "TE source input RMS",
                    )?;
                } else {
                    unsafe {
                        ffi::check(
                            (api.rms)(
                                gpu.raw(),
                                x.ptr(),
                                0,
                                s as i64,
                                4096,
                                w.input.fp(),
                                1e-6,
                                std::ptr::null(),
                                std::ptr::null(),
                                std::ptr::null(),
                                hb.ptr(),
                                0,
                            ),
                            "TE input RMS",
                        )?;
                    }
                }
                capture(&mut trace, gpu, "input_norm", &hb, s * HIDDEN * 4)?;
                scratch.linear(gpu, &hb, s, &w.qkv, &qkv)?;
                scratch.capture(&mut trace, gpu, "qkv", s, &w.qkv, &qkv)?;
                unsafe {
                    ffi::check(
                        rope(
                            gpu.raw(),
                            qkv.fp(),
                            s as i64,
                            32,
                            6144,
                            w.qnorm.fp(),
                            cs.fp(),
                        ),
                        "TE Q RMSRoPE",
                    )?;
                    ffi::check(
                        rope(
                            gpu.raw(),
                            qkv.fp().add(4096),
                            s as i64,
                            8,
                            6144,
                            w.knorm.fp(),
                            cs.fp(),
                        ),
                        "TE K RMSRoPE",
                    )?;
                }
                capture(&mut trace, gpu, "qkv.rope", &qkv, s * 6144 * 4)?;
                if (self.attention_order32 || self.attention_source32) && s == 32 {
                    let attention = if self.attention_order32 {
                        api.attention_order32
                    } else {
                        api.attention_source32
                    }
                    .ok_or_else(|| Error("private selected source attention ABI missing".into()))?;
                    // SAFETY: exact32 packed Q32/KV8/D128, disjoint owned output/status.
                    // The ABI resets status on the same stream; check it before any consumer
                    // or following linear can reset it. Scratch allocation accounting is unchanged.
                    ffi::check(
                        unsafe {
                            attention(
                                gpu.raw(),
                                qkv.fp(),
                                s as i64,
                                att.fp(),
                                scratch.status.ptr().cast(),
                            )
                        },
                        "TE source32 GQA",
                    )?;
                    gpu.sync()?;
                    let mut status = [0u8; 4];
                    scratch.status.read(0, &mut status)?;
                    let status = u32::from_le_bytes(status);
                    if status != 0 {
                        return Err(Error(format!(
                            "TE source32 numerical status0x{status:x}; output not consumed"
                        )));
                    }
                } else if self.probability_first {
                    let attention = if self.attention_tree32 && s == 32 {
                        api.attention_tree32
                    } else if self.score_division {
                        api.attention_divide
                    } else {
                        api.attention_prob
                    }
                    .ok_or_else(|| Error("private attention ABI missing".into()))?;
                    // SAFETY: fixed Q32/KV8/D128 packed QKV and disjoint F32 output.
                    ffi::check(
                        unsafe { attention(gpu.raw(), qkv.fp(), s as i64, att.fp()) },
                        "TE probability-first GQA",
                    )?;
                } else {
                    unsafe {
                        ffi::check(
                            (api.attention)(
                                gpu.raw(),
                                qkv.ptr(),
                                qkv.fp().add(4096).cast(),
                                qkv.fp().add(5120).cast(),
                                0,
                                0,
                                s as i64,
                                32,
                                8,
                                128,
                                6144,
                                6144,
                                att.ptr(),
                            ),
                            "TE causal GQA",
                        )?;
                    }
                }
                capture(&mut trace, gpu, "attention", &att, s * HIDDEN * 4)?;
                scratch.linear(gpu, &att, s, &w.out, &out)?;
                scratch.capture(&mut trace, gpu, "out", s, &w.out, &out)?;
                unsafe {
                    ffi::check(
                        (api.add)(
                            gpu.raw(),
                            x.ptr(),
                            0,
                            s as i64,
                            4096,
                            out.ptr(),
                            0,
                            std::ptr::null(),
                            std::ptr::null(),
                        ),
                        "TE attention residual",
                    )?;
                }
                capture(&mut trace, gpu, "attention_residual", &x, s * HIDDEN * 4)?;
                if self.ordered_rms {
                    let rms = api
                        .rms4096
                        .ok_or_else(|| Error("private source RMS ABI missing".into()))?;
                    // SAFETY: checked rows, disjoint F32 [rows,4096] buffers and 4096 weights.
                    ffi::check(
                        unsafe { rms(gpu.raw(), x.fp(), s as i64, w.post.fp(), hb.fp()) },
                        "TE source post RMS",
                    )?;
                } else {
                    unsafe {
                        ffi::check(
                            (api.rms)(
                                gpu.raw(),
                                x.ptr(),
                                0,
                                s as i64,
                                4096,
                                w.post.fp(),
                                1e-6,
                                std::ptr::null(),
                                std::ptr::null(),
                                std::ptr::null(),
                                hb.ptr(),
                                0,
                            ),
                            "TE post RMS",
                        )?;
                    }
                }
                capture(&mut trace, gpu, "post_norm", &hb, s * HIDDEN * 4)?;
                scratch.linear(gpu, &hb, s, &w.gu, &gu)?;
                scratch.capture(&mut trace, gpu, "gu", s, &w.gu, &gu)?;
                if self.silu_source && s <= 4_194_304 / FFN {
                    let silu = api
                        .silu_source
                        .ok_or_else(|| Error("private source SiLU ABI missing".into()))?;
                    // SAFETY: checked positive rows, fixed FFN12288, count cap, disjoint
                    // owned F32 gate/up and activation buffers live through stream sync.
                    // IEEE outputs are consumed by the unchanged finite-checking INT8 path.
                    ffi::check(
                        unsafe { silu(gpu.raw(), gu.fp(), s as i64, FFN as i64, act.fp()) },
                        "TE source SiLU",
                    )?;
                } else {
                    unsafe {
                        ffi::check(
                            (api.swiglu)(
                                gpu.raw(),
                                gu.ptr(),
                                0,
                                s as i64,
                                FFN as i64,
                                act.ptr(),
                                0,
                            ),
                            "TE SwiGLU",
                        )?;
                    }
                }
                capture(&mut trace, gpu, "swiglu", &act, s * FFN * 4)?;
                scratch.linear(gpu, &act, s, &w.down, &out)?;
                scratch.capture(&mut trace, gpu, "down", s, &w.down, &out)?;
                unsafe {
                    ffi::check(
                        (api.add)(
                            gpu.raw(),
                            x.ptr(),
                            0,
                            s as i64,
                            4096,
                            out.ptr(),
                            0,
                            std::ptr::null(),
                            std::ptr::null(),
                        ),
                        "TE FFN residual",
                    )?;
                }
                Ok(())
            })();
            // Sync even after an error, before any block buffer can be freed/reused.
            let sync = gpu.sync();
            work?;
            sync?;
            capture(
                &mut trace,
                gpu,
                if input.is_some() {
                    "block_output"
                } else {
                    "layer0"
                },
                &x,
                s * HIDDEN * 4,
            )?;
            stage(layer + 1, &x)?;
            drop_block(w);
        }
        let embeddings = DevBuf::f32(gpu, (s - drop) * HIDDEN)?;
        embeddings.copy_within(0, &x, drop * HIDDEN * 4, (s - drop) * HIDDEN * 4)?;
        gpu.sync()?;
        let image_pad = vec![false; s - drop];
        Ok(Encoded {
            embeddings,
            ids,
            drop,
            image_pad,
            plan,
        })
    }
}
fn drop_block(block: weights::Block) {
    drop(block);
}
fn host_memory_preflight(required: usize) -> Result<()> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| Error(e.to_string()))?;
    let kb = text
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")
                .and_then(|v| v.split_whitespace().next())
                .and_then(|v| v.parse::<usize>().ok())
        })
        .ok_or_else(|| Error("host MemAvailable unavailable".into()))?;
    if kb.checked_mul(1024).is_none_or(|bytes| bytes < required) {
        return Err(Error(format!(
            "TE host staging needs {required} available bytes"
        )));
    }
    Ok(())
}
pub fn prompt_ids(tok: &Tokenizer, prompt: &str) -> Result<(Vec<u32>, usize)> {
    let ids = tok.encode(&format!(
        "{SYSTEM}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ));
    let drop = tok.encode(SYSTEM).len();
    if ids.len() <= drop || ids.len() > 4096 {
        return Err(Error("TE token bound; no truncation".into()));
    }
    for special in ["<|image_pad|>", "<|video_pad|>"] {
        if tok.id(special).is_some_and(|id| ids.contains(&id)) {
            return Err(Error(
                "text-only TE cannot consume image/video tokens".into(),
            ));
        }
    }
    Ok((ids, drop))
}
pub fn rotations(s: usize) -> Vec<f32> {
    let mut cs = vec![0.; s * 128];
    for row in 0..s {
        for i in 0..64 {
            let angle = row as f64 / 5_000_000f64.powf(2. * i as f64 / 128.);
            cs[row * 128 + 2 * i] = angle.cos() as f32;
            cs[row * 128 + 2 * i + 1] = angle.sin() as f32;
        }
    }
    cs
}
fn validate_config(path: &Path) -> Result<()> {
    let bytes = std::fs::read(path).map_err(|e| Error(e.to_string()))?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| Error(e.to_string()))?;
    let t = v.get("text_config").unwrap_or(&v);
    for (key, n) in [
        ("num_hidden_layers", 36),
        ("hidden_size", 4096),
        ("num_attention_heads", 32),
        ("num_key_value_heads", 8),
        ("head_dim", 128),
        ("intermediate_size", 12288),
        ("vocab_size", 151936),
    ] {
        if t[key].as_u64() != Some(n) {
            return Err(Error(format!("TE config{key} differs")));
        }
    }
    if t["rms_norm_eps"].as_f64() != Some(1e-6)
        || t["rope_theta"].as_f64() != Some(5_000_000.)
        || t["hidden_act"] != "silu"
        || t["attention_bias"] != false
    {
        return Err(Error("TE norm/rope/activation/bias config differs".into()));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preallocation_bounds() {
        let b = Budget {
            allowance_bytes: 3_000_000_000,
            driver_margin_bytes: 1024 * 1024 * 1024,
        };
        assert_eq!(
            Plan::new(1024, &b).unwrap().tracked_bytes,
            414_212 * 1024 + 4 + BLOCK_BYTES
        );
        assert!(Plan::new(0, &b).is_err());
        assert!(Plan::new(4097, &b).is_err());
        assert!(Plan::new(
            1024,
            &Budget {
                allowance_bytes: 500_000_000,
                driver_margin_bytes: 512 * 1024 * 1024
            }
        )
        .is_err());
    }
    #[test]
    fn text_positions() {
        let cs = rotations(3);
        for i in 0..64 {
            assert_eq!(cs[2 * i], 1.);
            assert_eq!(cs[2 * i + 1], 0.);
        }
        assert_eq!(cs[128], 1f64.cos() as f32);
    }
    #[test]
    fn regular_hadamard() {
        for i in 0..256 {
            for j in 0..256 {
                let mut dot = 0.;
                for k in 0..256 {
                    dot += weights::hadamard(i, k) * weights::hadamard(j, k);
                }
                assert_eq!(dot, if i == j { 1. } else { 0. });
            }
        }
    }
}
