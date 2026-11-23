//! Native, single-GPU StableLM 3B. Contract/dispatch structure follows nextcuda
//! llm/qwen35; quantized products follow nextsycl's common GGUF readers.
//! Deliberately sequential prefill/batching; no draft model or subprocess.
mod geometry;
mod kernels;
mod state;
pub use geometry::Geometry;
use kernels::Kernels;
use nextrocm_gguf::Gguf;
use nextrocm_llm::*;
use nextrocm_tok::{Effort, Message};
use state::{Ck, Dec, Sess};
use std::{
    any::Any,
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Instant,
};

pub const ARCH: &str = "stablelm";
const CHUNK: usize = 32;
// Reserve enough space for runtime work and avoid exhausting the device on load/session creation.
const MARGIN: u64 = 256 * 1024 * 1024;
fn error(s: impl Into<String>) -> Error {
    Error(s.into())
}

/// Registry entry for StableLM Zephyr. Other family members use the same load
/// function but require their own template selection by the existing front door.
pub fn kind() -> EngineKind {
    EngineKind {
        archs: &[ARCH],
        name: "StableLM 3B (native HIP)",
        load,
        info,
        kernels: None,
        chat: nextrocm_llm::ChatTemplate::Legacy(zephyr_chat),
        options: &[],
        prepare: None,
    }
}
/// Select the actual inventory model's prompt format without changing EngineKind.
/// Old GGUFs call themselves `source`; their template/vocabulary is authoritative.
pub fn chat_for(f: &Gguf) -> std::result::Result<ChatFn, String> {
    Geometry::read(f).map_err(|e| e.0)?;
    match f.meta("tokenizer.chat_template").and_then(|v| v.as_str()) {
        Some(t) if t.contains("<|im_start|>") && t.contains("<|im_end|>") => Ok(code_instruct_chat),
        Some(t) if t.contains("<|user|>") && t.contains("<|assistant|>") => Ok(zephyr_chat),
        Some(t) if !t.is_empty() => Err("unrecognized StableLM chat template".into()),
        _ => {
            let tokens = f
                .meta("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .ok_or("missing tokenizer vocabulary")?;
            if tokens.iter().any(|v| v.as_str() == Some("<fim_prefix>"))
                && !tokens.iter().any(|v| v.as_str() == Some("<|im_start|>"))
            {
                Ok(code_completion)
            } else {
                Err(
                    "StableLM GGUF has no recognized chat template or Code completion vocabulary"
                        .into(),
                )
            }
        }
    }
}
/// Stable Code base is raw continuation, as in development-host's `mode: complete`:
/// only the latest user prompt is supplied, without chat delimiters or history.
pub fn code_completion(messages: &[Message], _: Effort, _: &[String]) -> String {
    messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map_or_else(String::new, |m| m.content.to_owned())
}
/// Checked geometry, without loading a GPU.
pub fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = Geometry::read(f).map_err(|e| e.0)?;
    Ok(format!("stablelm: {} layers, width {}, FF {}, {} heads of {}, rotary {}, theta {}, vocabulary {}, context {}; fp16 KV {} bytes/token",g.layers,g.width,g.ff,g.heads,g.head,g.rotary,g.theta,g.vocab,g.context,g.cache_bytes(1).map_err(|e|e.0)?))
}
/// Zephyr's text-only template; tools and reasoning have no model-specific path.
pub fn zephyr_chat(messages: &[Message], _: Effort, _: &[String]) -> String {
    let mut out = String::new();
    for m in messages {
        if !matches!(m.role, "user" | "system" | "assistant") {
            continue;
        }
        out.push_str("<|");
        out.push_str(m.role);
        out.push_str("|>\n");
        out.push_str(m.content);
        out.push_str("<|endoftext|>\n");
    }
    if !messages.is_empty() {
        out.push_str("<|assistant|>\n");
    }
    out
}
/// Stable Code Instruct's ChatML template (including its default system text).
pub fn code_instruct_chat(messages: &[Message], _: Effort, _: &[String]) -> String {
    let (system, rest) = match messages.first() {
        Some(m) if m.role == "system" => (m.content, &messages[1..]),
        _ => ("You are a helpful assistant.", messages),
    };
    let mut out = String::new();
    if !rest.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    for m in rest {
        out.push_str("<|im_start|>");
        out.push_str(m.role);
        out.push('\n');
        out.push_str(m.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

struct Weight {
    buf: DevBuf,
    rows: usize,
    cols: usize,
    ty: i32,
}
struct Scratch {
    x: DevBuf,
    norm: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    attn: DevBuf,
    proj: DevBuf,
    gate: DevBuf,
    up: DevBuf,
    logits: DevBuf,
}
impl Scratch {
    fn new(gpu: &Arc<Gpu>, g: &Geometry) -> Result<Self> {
        let d = || DevBuf::f32(gpu, g.width);
        Ok(Self {
            x: d()?,
            norm: d()?,
            q: d()?,
            k: d()?,
            v: d()?,
            attn: d()?,
            proj: d()?,
            gate: DevBuf::f32(gpu, g.ff)?,
            up: DevBuf::f32(gpu, g.ff)?,
            logits: DevBuf::f32(gpu, g.vocab)?,
        })
    }
}
struct StableLm {
    geom: Geometry,
    gpu: Arc<Gpu>,
    weights: BTreeMap<String, Weight>,
    kernels: Kernels,
    scratch: Mutex<Scratch>,
    owner: Arc<()>,
    seconds: f64,
    bytes: u64,
}
/// Load the bounded family on exactly one GPU. Rejects unsupported tensor layouts
/// before allocating weights; no fallback to host execution.
pub fn load<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn Engine + 'g>> {
    let start = Instant::now();
    let geom = Geometry::read(f)?;
    if gpus.len() != 1 {
        return Err(error("stablelm currently requires exactly one GPU"));
    }
    if o.draft {
        return Err(error("stablelm has no draft model"));
    }
    let gpu = Arc::clone(&gpus[0]);
    if gpu.raw().is_null() {
        return Err(error("stablelm requires a native HIP GPU"));
    }
    let kernels = Kernels::load(&gpu)?;
    let bytes = f
        .tensors
        .iter()
        .try_fold(0u64, |s, t| s.checked_add(t.bytes))
        .ok_or_else(|| error("weight size overflow"))?;
    if bytes >= 8_000_000_000 {
        return Err(error("stablelm weights exceed the 8 GB admission limit"));
    }
    let reserved = geom.cache_bytes(o.kv.0)? as u64;
    ensure_memory(
        &gpu,
        bytes
            .checked_add(reserved)
            .ok_or_else(|| error("load size overflow"))?,
    )?;
    let mut weights = BTreeMap::new();
    for (name, shape) in geom.specs() {
        let t = f
            .tensor(&name)
            .ok_or_else(|| error(format!("missing {name}")))?;
        let data = f.read(t).map_err(|e| error(e.to_string()))?;
        let buf = DevBuf::new(&gpu, data.len())?;
        buf.write(0, &data)?;
        let (rows, cols) = if shape.len() == 2 {
            (shape[0] as usize, shape[1] as usize)
        } else {
            (1, shape[0] as usize)
        };
        weights.insert(
            name,
            Weight {
                buf,
                rows,
                cols,
                ty: t.ty.code() as i32,
            },
        );
    }
    let scratch = Mutex::new(Scratch::new(&gpu, &geom)?);
    gpu.sync()?;
    log(format!(
        "stablelm native HIP: {bytes} weight bytes; fp16 KV {} bytes/token; sequential prefill",
        geom.cache_bytes(1)?
    ));
    Ok(Box::new(StableLm {
        geom,
        gpu,
        weights,
        kernels,
        scratch,
        owner: Arc::new(()),
        seconds: start.elapsed().as_secs_f64(),
        bytes,
    }))
}
fn ensure_memory(gpu: &Gpu, bytes: u64) -> Result<()> {
    let (_, free) = gpu.memory()?;
    if let Some(free) = free {
        if bytes.checked_add(MARGIN).is_none_or(|n| n > free) {
            return Err(error(format!(
                "stablelm needs {bytes} bytes plus {MARGIN} margin; {free} free"
            )));
        }
    }
    Ok(())
}
impl StableLm {
    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .ok_or_else(|| error(format!("missing loaded {name}")))
    }
    fn sess<'a>(&self, s: &'a Session) -> Result<&'a Sess> {
        s.get::<Sess>()
            .filter(|s| Arc::ptr_eq(&s.owner, &self.owner))
            .ok_or_else(|| error("session belongs to another model instance"))
    }
    fn sess_mut<'a>(&self, s: &'a mut Session) -> Result<&'a mut Sess> {
        s.get_mut::<Sess>()
            .filter(|s| Arc::ptr_eq(&s.owner, &self.owner))
            .ok_or_else(|| error("session belongs to another model instance"))
    }
    fn validate_pass(&self, s: &Session, tokens: &[u32]) -> Result<()> {
        let s = self.sess(s)?;
        if s.poisoned {
            return Err(error("session had a failed GPU pass; reset or restore it"));
        }
        if tokens.is_empty() {
            return Err(error("empty StableLM pass"));
        }
        if s.pos
            .checked_add(tokens.len())
            .is_none_or(|n| n > s.max_ctx)
        {
            return Err(error("StableLM context exhausted"));
        }
        if tokens.iter().any(|t| *t as usize >= self.geom.vocab) {
            return Err(error("token outside vocabulary"));
        }
        Ok(())
    }
    fn token(&self, s: &mut Sess, token: u32, a: &Scratch, tap: Tap) -> Result<Vec<f32>> {
        let g = &self.geom;
        let gpu = &self.gpu;
        let k = &self.kernels;
        k.embed(gpu, self.weight("token_embd.weight")?, token, &a.x)?;
        tap("inp_embd", &a.x)?;
        for l in 0..g.layers {
            let w = |n: &str| self.weight(&format!("blk.{l}.{n}"));
            k.norm(
                gpu,
                &a.x,
                &w("attn_norm.weight")?.buf,
                &w("attn_norm.bias")?.buf,
                g.width,
                g.eps,
                &a.norm,
            )?;
            tap(&format!("attn_norm-{l}"), &a.norm)?;
            k.mv(gpu, w("attn_q.weight")?, &a.norm, &a.q)?;
            k.mv(gpu, w("attn_k.weight")?, &a.norm, &a.k)?;
            k.mv(gpu, w("attn_v.weight")?, &a.norm, &a.v)?;
            k.rope(gpu, &a.q, g, s.pos)?;
            k.rope(gpu, &a.k, g, s.pos)?;
            tap(&format!("Qcur-{l}"), &a.q)?;
            tap(&format!("Kcur-{l}"), &a.k)?;
            tap(&format!("Vcur-{l}"), &a.v)?;
            k.attn(
                gpu, &a.q, &a.k, &a.v, &s.kv[l].0, &s.kv[l].1, g, s.pos, &a.attn,
            )?;
            k.mv(gpu, w("attn_output.weight")?, &a.attn, &a.proj)?;
            k.add(gpu, &a.x, &a.proj, g.width)?;
            tap(&format!("ffn_inp-{l}"), &a.x)?;
            k.norm(
                gpu,
                &a.x,
                &w("ffn_norm.weight")?.buf,
                &w("ffn_norm.bias")?.buf,
                g.width,
                g.eps,
                &a.norm,
            )?;
            tap(&format!("ffn_norm-{l}"), &a.norm)?;
            k.mv(gpu, w("ffn_gate.weight")?, &a.norm, &a.gate)?;
            k.mv(gpu, w("ffn_up.weight")?, &a.norm, &a.up)?;
            k.swiglu(gpu, &a.gate, &a.up, g.ff)?;
            k.mv(gpu, w("ffn_down.weight")?, &a.gate, &a.proj)?;
            k.add(gpu, &a.x, &a.proj, g.width)?;
            tap(&format!("l_out-{l}"), &a.x)?;
        }
        k.norm(
            gpu,
            &a.x,
            &self.weight("output_norm.weight")?.buf,
            &self.weight("output_norm.bias")?.buf,
            g.width,
            g.eps,
            &a.norm,
        )?;
        tap("result_norm", &a.norm)?;
        k.mv(gpu, self.weight("output.weight")?, &a.norm, &a.logits)?;
        tap("result_output", &a.logits)?;
        let logits = a.logits.to_f32()?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(error("StableLM produced non-finite logits"));
        }
        s.pos += 1;
        Ok(logits)
    }
}
impl Engine for StableLm {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn load_seconds(&self) -> f64 {
        self.seconds
    }
    fn load_bytes(&self) -> u64 {
        self.bytes
    }
    fn has_draft(&self) -> bool {
        false
    }
    fn prefill_chunk(&self) -> usize {
        CHUNK
    }
    fn max_verify(&self) -> usize {
        CHUNK
    }
    fn cache_fingerprint(&self) -> String {
        format!(
            "stablelm-3b-f16kv-v1-theta{}-eps{}",
            self.geom.theta, self.geom.eps
        )
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        if max_ctx == 0 || max_ctx > self.geom.context {
            return Err(error("session context exceeds trained context or is zero"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("StableLM scratch lock poisoned"))?;
        ensure_memory(&self.gpu, self.geom.cache_bytes(max_ctx)? as u64)?;
        Ok(Session::new(Sess::new(
            &self.gpu,
            &self.geom,
            max_ctx,
            Arc::clone(&self.owner),
        )?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("StableLM scratch lock poisoned"))?;
        let s = self.sess_mut(s)?;
        self.gpu.sync()?;
        s.pos = 0;
        s.last = None;
        s.poisoned = false;
        Ok(())
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let ck = self.save(src)?;
        self.restore(dst, &ck)
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(s, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(
        &self,
        s: &mut Session,
        tokens: &[u32],
        stop: &(dyn Fn(usize) -> bool + Sync),
        tap: Tap,
    ) -> Result<(usize, Vec<f32>)> {
        self.validate_pass(s, tokens)?;
        let mut read = 0;
        let mut logits = Vec::new();
        for chunk in tokens.chunks(CHUNK) {
            if stop(read) {
                break;
            }
            logits = self.forward(s, chunk, tap)?;
            read += chunk.len();
        }
        Ok((read, logits))
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        self.forward_rows(s, tokens, 1, tap)?
            .pop()
            .ok_or_else(|| error("missing final logits"))
    }
    fn forward_rows(
        &self,
        s: &mut Session,
        tokens: &[u32],
        n_out: usize,
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_pass(s, tokens)?;
        if tokens.len() > CHUNK || n_out > tokens.len() {
            return Err(error("invalid StableLM pass/row count"));
        }
        let a = self
            .scratch
            .lock()
            .map_err(|_| error("StableLM scratch lock poisoned"))?;
        let s = self.sess_mut(s)?;
        let start = s.pos;
        s.last = None;
        let mut out = Vec::with_capacity(n_out);
        for (i, t) in tokens.iter().enumerate() {
            match self.token(s, *t, &a, tap) {
                Ok(row) => {
                    if i >= tokens.len() - n_out {
                        out.push(row);
                    }
                }
                Err(e) => {
                    let _ = self.gpu.sync();
                    s.pos = start;
                    s.poisoned = true;
                    return Err(e);
                }
            }
        }
        s.last = Some((start, tokens.len()));
        Ok(out)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let s = self.sess_mut(s)?;
        if s.poisoned {
            return Err(error("cannot roll back a failed pass"));
        }
        let (start, count) = s
            .last
            .ok_or_else(|| error("no pass available for rollback"))?;
        if keep > count {
            return Err(error("rollback exceeds last pass"));
        }
        s.pos = start + keep;
        s.last = None;
        Ok(())
    }
    fn forward_batch(
        &self,
        sessions: &mut [&mut Session],
        tokens: &[u32],
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        if sessions.len() != tokens.len() {
            return Err(error("batch session/token count mismatch"));
        }
        for (s, t) in sessions.iter().zip(tokens) {
            self.validate_pass(s, &[*t])?;
        }
        sessions
            .iter_mut()
            .zip(tokens)
            .map(|(s, t)| self.forward(s, &[*t], tap))
            .collect()
    }
    fn decoder(&self, logits: Vec<f32>, _: bool) -> Decoder {
        Decoder::new(Dec {
            owner: Arc::clone(&self.owner),
            logits: Some(logits),
            next: None,
        })
    }
    fn decoder_after(&self, next: u32, _: bool) -> Decoder {
        Decoder::new(Dec {
            owner: Arc::clone(&self.owner),
            logits: None,
            next: Some(next),
        })
    }
    fn step(
        &self,
        s: &mut Session,
        d: &mut Decoder,
        smp: &mut dyn Sampler,
        tap: Tap,
    ) -> Result<Vec<u32>> {
        self.sess(s)?;
        let d = d
            .get_mut::<Dec>()
            .filter(|d| Arc::ptr_eq(&d.owner, &self.owner))
            .ok_or_else(|| error("decoder belongs to another model"))?;
        if let Some(t) = d.next {
            let logits = self.forward(s, &[t], tap)?;
            d.next = None;
            d.logits = Some(logits);
        }
        let row = d
            .logits
            .as_ref()
            .ok_or_else(|| error("decoder pending token was already handed off"))?;
        if row.len() != self.geom.vocab || row.iter().any(|x| !x.is_finite()) {
            return Err(error("invalid decoder logits"));
        }
        let t = smp.sample(row);
        if t as usize >= self.geom.vocab {
            return Err(error("sampler returned out-of-vocabulary token"));
        }
        d.logits = None;
        d.next = Some(t);
        Ok(vec![t])
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        let s = self.sess(s)?;
        if s.poisoned {
            return Err(error("cannot save failed session"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("StableLM scratch lock poisoned"))?;
        self.gpu.sync()?;
        let bytes = s.pos * self.geom.width * 2;
        let mut data = Vec::with_capacity(self.geom.cache_bytes(s.pos)?);
        for (k, v) in &s.kv {
            for b in [k, v] {
                let start = data.len();
                data.resize(start + bytes, 0);
                b.read(0, &mut data[start..])?;
            }
        }
        Ok(Checkpoint::new(Ck {
            owner: Arc::clone(&self.owner),
            pos: s.pos,
            data,
        }))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck
            .get::<Ck>()
            .filter(|c| Arc::ptr_eq(&c.owner, &self.owner))
            .ok_or_else(|| error("checkpoint belongs to another model instance"))?;
        let s = self.sess_mut(s)?;
        if ck.pos > s.max_ctx || ck.data.len() != self.geom.cache_bytes(ck.pos)? {
            return Err(error("checkpoint exceeds session or has invalid length"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("StableLM scratch lock poisoned"))?;
        s.poisoned = true;
        s.last = None;
        let bytes = ck.pos * self.geom.width * 2;
        let mut at = 0;
        for (k, v) in &s.kv {
            for b in [k, v] {
                b.write(0, &ck.data[at..at + bytes])?;
                at += bytes;
            }
        }
        self.gpu.sync()?;
        s.pos = ck.pos;
        s.poisoned = false;
        Ok(())
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(Ck::read(
            r,
            &self.geom,
            Arc::clone(&self.owner),
        )?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        vec![GpuInfo {
            index: self.gpu.index,
            name: self.gpu.name.clone(),
            pci: self.gpu.pci.clone(),
            total,
            free,
            layers: (0, self.geom.layers as u64),
            expert_slots: 0,
            host_slots: 0,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn templates_preserve_family_delimiters() {
        let messages = [Message {
            role: "user",
            content: "hello",
            ..Default::default()
        }];
        assert_eq!(
            zephyr_chat(&messages, Effort::Default, &[]),
            "<|user|>\nhello<|endoftext|>\n<|assistant|>\n"
        );
        assert_eq!(code_instruct_chat(&messages,Effort::Default,&[]),"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n");
        assert!(zephyr_chat(&[], Effort::Default, &[]).is_empty());
    }
}
