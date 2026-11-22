//! Private single-session Engine adapter. No registry entry or admission claim.
//! Sampling uses the existing caller Sampler/RNG; all forward math stays in TextGraph.
use crate::{config::MiniConfig, text_graph::TextGraph, weights::Weights};
use nextrocm_gguf::Gguf;
use nextrocm_llm::*;
use serde_json::Value;
use std::{
    any::Any,
    io::{self, Read, Write},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex, Weak,
    },
    time::Instant,
};
const MAX_CONTEXT: usize = 4096;
const HOST_RESERVE: u64 = 4 * 1024 * 1024 * 1024;
const MAGIC: &[u8; 8] = b"INKF3201";
fn err(s: impl Into<String>) -> Error {
    Error(s.into())
}
fn available() -> Result<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").map_err(|e| err(e.to_string()))?;
    s.lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|s| s.parse::<u64>().ok())
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| err("MemAvailable unavailable"))
}
fn reserve_host(bytes: usize) -> Result<()> {
    if (bytes as u64)
        .checked_add(HOST_RESERVE)
        .is_none_or(|n| n > available().unwrap_or(0))
    {
        return Err(err("checkpoint would violate host reserve"));
    }
    Ok(())
}
/// Request validation must run before the shared infallible ChatFn front door.
/// Images, tools and embedded control spellings are deliberately unsupported.
pub fn chat_checked(
    messages: &[Value],
    effort: Option<&Value>,
    tools: &[Value],
) -> crate::Result<String> {
    if !tools.is_empty() || messages.is_empty() {
        return Err("text-only nonempty conversation required; tools unsupported".into());
    }
    for m in messages {
        if !matches!(m["role"].as_str(), Some("user" | "assistant" | "system"))
            || m.get("tool_calls").is_some()
        {
            return Err("unsupported message role or tools".into());
        }
        for key in ["content", "reasoning_content"] {
            if key == "content" || m.get(key).is_some() {
                let s = m[key]
                    .as_str()
                    .ok_or("only text strings supported; media unsupported")?;
                if s.contains("<|") {
                    return Err("literal model control spellings unsupported in text".into());
                }
            }
        }
    }
    nextrocm_tok::inkling2_chat(messages, effort, tools)
}
struct Live {
    graph: TextGraph,
    lease: Weak<()>,
    epoch: u64,
}
struct Model {
    gpu: Arc<Gpu>,
    live: Mutex<Live>,
    owner: Arc<()>,
    context: usize,
    feed_chunk: usize,
    bytes: u64,
    seconds: f64,
    state_bytes: usize,
    cancel: Arc<AtomicBool>,
}
struct Sess {
    owner: Arc<()>,
    lease: Arc<()>,
    pos: usize,
    max: usize,
    poisoned: bool,
    last: bool,
}
impl SessionState for Sess {
    fn pos(&self) -> usize {
        self.pos
    }
    fn max_ctx(&self) -> usize {
        self.max
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
struct Dec {
    owner: Arc<()>,
    epoch: Option<u64>,
    logits: Option<Vec<f32>>,
    next: Option<u32>,
}
impl DecoderState for Dec {
    fn drafts(&self) -> (u64, u64) {
        (0, 0)
    }
    fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
struct Ck {
    owner: Arc<()>,
    context: usize,
    pos: usize,
    data: Vec<u8>,
}
impl CheckpointState for Ck {
    fn pos(&self) -> usize {
        self.pos
    }
    fn bytes(&self) -> usize {
        self.data.len()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        for n in [self.context, self.pos, self.data.len()] {
            w.write_all(&(n as u64).to_le_bytes())?;
        }
        w.write_all(&self.data)
    }
}
/// Strict selected checkpoint, one real GPU and one physical session, F32 state.
/// No registration: caller still must wire checked requests and Inkling output parser.
pub fn load<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn Engine + 'g>> {
    load_with_cancel(f, gpus, o, log, Arc::new(AtomicBool::new(false)))
}
/// Request owner must set this flag on cancellation, and clear only before a new
/// request after resetting/restoring the session. The shared Engine trait has no
/// per-decode cancellation argument; registration must retain this handle.
pub fn load_with_cancel<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
) -> Result<Box<dyn Engine + 'g>> {
    load_impl(
        f,
        gpus,
        o,
        log,
        cancel,
        configured_expert_mode(o.expert_bytes)?,
    )
}
/// Explicit diagnostic staging policy; never overrides an options-level policy.
pub fn load_with_expert_mode<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
    mode: crate::expert_cache::ExpertMode,
) -> Result<Box<dyn Engine + 'g>> {
    if o.expert_bytes.is_some() {
        return Err(err(
            "explicit expert mode conflicts with options expert_bytes",
        ));
    }
    load_impl(f, gpus, o, log, cancel, mode)
}
fn configured_expert_mode(bytes: Option<usize>) -> Result<crate::expert_cache::ExpertMode> {
    use crate::expert_cache::{ExpertMode, MAX_CACHE_BYTES};
    match bytes {
        None => Ok(ExpertMode::DeviceLru {
            budget_bytes: MAX_CACHE_BYTES,
        }),
        Some(0) => Ok(ExpertMode::Stream),
        Some(n) => {
            let budget_bytes = u64::try_from(n).map_err(|_| err("expert cache size overflow"))?;
            if budget_bytes > MAX_CACHE_BYTES {
                return Err(err("Inkling expert cache budget exceeds 1GiB"));
            }
            Ok(ExpertMode::DeviceLru { budget_bytes })
        }
    }
}
fn load_impl<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
    mode: crate::expert_cache::ExpertMode,
) -> Result<Box<dyn Engine + 'g>> {
    load_impl_with_embedding(
        f,
        gpus,
        o,
        log,
        cancel,
        mode,
        crate::device::EmbeddingMode::Resident,
    )
}
/// Explicit isolated diagnostic policy. Defaults and registry loading stay Resident.
pub fn load_with_policies<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
    mode: crate::expert_cache::ExpertMode,
    embedding: crate::device::EmbeddingMode,
) -> Result<Box<dyn Engine + 'g>> {
    if o.expert_bytes.is_some() {
        return Err(err(
            "explicit expert mode conflicts with options expert_bytes",
        ));
    }
    load_impl_with_embedding(f, gpus, o, log, cancel, mode, embedding)
}
fn load_impl_with_embedding<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
    mode: crate::expert_cache::ExpertMode,
    embedding: crate::device::EmbeddingMode,
) -> Result<Box<dyn Engine + 'g>> {
    load_prefill_impl(f, gpus, o, log, cancel, mode, embedding, 1)
}
/// Explicit plain-feed chunk policy. feed_until retains serial stop semantics.
pub fn load_with_prefill_policies<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
    mode: crate::expert_cache::ExpertMode,
    embedding: crate::device::EmbeddingMode,
    feed_chunk: usize,
) -> Result<Box<dyn Engine + 'g>> {
    if !matches!(feed_chunk, 1 | 2 | 4) || o.expert_bytes.is_some() {
        return Err(err(
            "explicit prefill requires chunk1/2/4 and no conflicting expert_bytes",
        ));
    }
    load_prefill_impl(f, gpus, o, log, cancel, mode, embedding, feed_chunk)
}
fn load_prefill_impl<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
    cancel: Arc<AtomicBool>,
    mode: crate::expert_cache::ExpertMode,
    embedding: crate::device::EmbeddingMode,
    feed_chunk: usize,
) -> Result<Box<dyn Engine + 'g>> {
    let start = Instant::now();
    if cancel.load(Ordering::Relaxed) {
        return Err(err("Inkling load cancelled"));
    }
    if gpus.len() != 1 || o.draft || o.mirror_bytes.is_some() || o.kv.1 > 1 {
        return Err(err(
            "Inkling requires one GPU/session, no draft or alternate mirror policy",
        ));
    }
    let context = if o.kv.0 == 0 { MAX_CONTEXT } else { o.kv.0 };
    if context == 0 || context > MAX_CONTEXT || feed_chunk > context || f.paths.len() != 1 {
        return Err(err("Inkling supports one GGUF and context1..4096"));
    }
    let w = Weights::read(f).map_err(err)?;
    // Check the selected checkpoint's actual largest pair before any GPU allocation.
    // Insufficient cache/free/UMA memory is an error, never an implicit fallback.
    mode.extra_bytes(w.largest_expert_pair_bytes())
        .map_err(err)?;
    let bytes = match embedding {
        crate::device::EmbeddingMode::Resident => w.non_expert_bytes(),
        crate::device::EmbeddingMode::HostRow => w
            .non_expert_bytes()
            .checked_sub(492_106_752)
            .ok_or_else(|| err("embedding budget underflow"))?,
    };
    let cfg = MiniConfig::from_official_json(include_bytes!("../assets/mini-config.json"))
        .map_err(err)?;
    let gpu = gpus[0].clone();
    let graph = TextGraph::load_with_dense_prefill_policies(
        &f.paths[0],
        &cfg,
        &gpu,
        context,
        available()?,
        mode,
        embedding,
        feed_chunk,
        feed_chunk > 1,
        feed_chunk > 1,
        || {
            if cancel.load(Ordering::Relaxed) {
                Err("Inkling load cancelled".into())
            } else {
                Ok(())
            }
        },
    )
    .map_err(err)?;
    let state_bytes = graph.state_bytes();
    log(format!("Inkling native F32 text: one session, context{context}, packed resident{bytes}, expert policy {mode:?}, embedding {embedding:?}, plain-feed chunk{feed_chunk}, expert batch{}, dense batch{} (feed_until serial); no vision/tools/draft", feed_chunk > 1, feed_chunk > 1));
    Ok(Box::new(Model {
        gpu,
        live: Mutex::new(Live {
            graph,
            lease: Weak::new(),
            epoch: 0,
        }),
        owner: Arc::new(()),
        context,
        feed_chunk,
        bytes,
        seconds: start.elapsed().as_secs_f64(),
        state_bytes,
        cancel,
    }))
}
impl Model {
    fn decoder_epoch(&self) -> Option<u64> {
        self.live
            .lock()
            .ok()
            .and_then(|l| l.lease.upgrade().map(|_| l.epoch))
    }
    fn invalidate_decoders(l: &mut Live) -> Result<()> {
        l.epoch = l
            .epoch
            .checked_add(1)
            .ok_or_else(|| err("Inkling decoder epoch exhausted"))?;
        Ok(())
    }
    fn sess<'a>(&self, s: &'a Session) -> Result<&'a Sess> {
        s.get::<Sess>()
            .filter(|s| Arc::ptr_eq(&s.owner, &self.owner))
            .ok_or_else(|| err("foreign Inkling session"))
    }
    fn sess_mut<'a>(&self, s: &'a mut Session) -> Result<&'a mut Sess> {
        s.get_mut::<Sess>()
            .filter(|s| Arc::ptr_eq(&s.owner, &self.owner))
            .ok_or_else(|| err("foreign Inkling session"))
    }
    fn live(&self) -> Result<std::sync::MutexGuard<'_, Live>> {
        self.live
            .lock()
            .map_err(|_| err("Inkling model mutex poisoned"))
    }
    fn lease(l: &Live, s: &Sess) -> Result<()> {
        if !l.lease.upgrade().is_some_and(|v| Arc::ptr_eq(&v, &s.lease)) {
            return Err(err("inactive Inkling session"));
        }
        Ok(())
    }
    fn validate(&self, s: &Session, tokens: &[u32]) -> Result<()> {
        let s = self.sess(s)?;
        if s.poisoned
            || tokens.is_empty()
            || s.pos.checked_add(tokens.len()).is_none_or(|n| n > s.max)
            || tokens.iter().any(|&t| t >= 200058 || t == 200054)
        {
            return Err(err(
                "poisoned session, exhausted context or invalid/media tokens",
            ));
        }
        Ok(())
    }
    fn one(&self, s: &mut Session, t: u32, stop: &dyn Fn() -> bool, tap: Tap) -> Result<Vec<f32>> {
        self.validate(s, &[t])?;
        let s = self.sess_mut(s)?;
        let mut l = self.live()?;
        Self::lease(&l, s)?;
        s.last = false;
        let result = l.graph.token(
            t,
            || {
                if stop() || self.cancel.load(Ordering::Relaxed) {
                    Err("request cancelled during token; reset required".into())
                } else {
                    Ok(())
                }
            },
            |v, b| tap(&format!("l_out-{}", v.ordinal), b).map_err(|e| e.0),
        );
        match result {
            Ok(x) => {
                s.pos += 1;
                s.last = true;
                Ok(x)
            }
            Err(e) => {
                s.poisoned = true;
                Err(err(e))
            }
        }
    }
}
impl Engine for Model {
    fn max_sessions(&self) -> usize {
        1
    }
    fn output_format(&self) -> OutputFormat {
        OutputFormat::Inkling2
    }
    fn sampling_policy(&self) -> SamplingPolicy {
        SamplingPolicy::FullVocabulary
    }
    fn default_top_p(&self) -> f32 {
        1.0
    }
    fn arch(&self) -> &'static str {
        "inkling2"
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
        1
    }
    fn max_verify(&self) -> usize {
        1
    }
    fn cache_fingerprint(&self) -> String {
        format!(
            "inkling2-mini-root883286-f32-complete-state-v1-ctx{}",
            self.context
        )
    }
    fn session(&self, max: usize) -> Result<Session> {
        if max == 0 || max > self.context {
            return Err(err("session context out of bounds"));
        }
        let mut l = self.live()?;
        if l.lease.upgrade().is_some() {
            return Err(err("Inkling supports one live session"));
        }
        Self::invalidate_decoders(&mut l)?;
        l.graph.reset_state().map_err(err)?;
        let lease = Arc::new(());
        l.lease = Arc::downgrade(&lease);
        Ok(Session::new(Sess {
            owner: self.owner.clone(),
            lease,
            pos: 0,
            max,
            poisoned: false,
            last: false,
        }))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        let s = self.sess_mut(s)?;
        let mut l = self.live()?;
        Self::lease(&l, s)?;
        s.poisoned = true;
        s.last = false;
        Self::invalidate_decoders(&mut l)?;
        l.graph.reset_state().map_err(err)?;
        s.pos = 0;
        s.poisoned = false;
        Ok(())
    }
    fn copy_session(&self, _: &mut Session, _: &Session) -> Result<()> {
        Err(err(
            "multiple Inkling sessions unsupported; use checkpoint restore",
        ))
    }
    fn feed(&self, s: &mut Session, t: &[u32], tap: Tap) -> Result<Vec<f32>> {
        if self.feed_chunk == 1 {
            return Ok(self.feed_until(s, t, &|_| false, tap)?.1);
        }
        self.validate(s, t)?;
        let mut last = Vec::new();
        for ids in t.chunks(self.feed_chunk) {
            let state = self.sess_mut(s)?;
            let mut live = self.live()?;
            Self::lease(&live, state)?;
            state.last = false;
            let result = live.graph.chunk(
                ids,
                || {
                    if self.cancel.load(Ordering::Relaxed) {
                        Err("request cancelled during chunk; reset required".into())
                    } else {
                        Ok(())
                    }
                },
                |_, v, b| tap(&format!("l_out-{}", v.ordinal), b).map_err(|e| e.0),
            );
            match result {
                Ok(mut rows) => {
                    if rows.len() != ids.len() || rows.is_empty() {
                        state.poisoned = true;
                        return Err(err("chunk output row count mismatch; reset required"));
                    }
                    state.pos += ids.len();
                    state.last = true;
                    last = rows
                        .pop()
                        .ok_or_else(|| err("chunk returned no output rows"))?;
                }
                Err(e) => {
                    state.poisoned = true;
                    return Err(err(e));
                }
            }
        }
        Ok(last)
    }
    fn feed_until(
        &self,
        s: &mut Session,
        t: &[u32],
        stop: &(dyn Fn(usize) -> bool + Sync),
        tap: Tap,
    ) -> Result<(usize, Vec<f32>)> {
        self.validate(s, t)?;
        let mut row = Vec::new();
        let mut n = 0;
        for &id in t {
            if stop(n) {
                break;
            }
            row = self.one(s, id, &|| stop(n), tap)?;
            n += 1;
        }
        Ok((n, row))
    }
    fn forward(&self, s: &mut Session, t: &[u32], tap: Tap) -> Result<Vec<f32>> {
        if t.len() != 1 {
            return Err(err(
                "Inkling forward accepts exactly one token; use feed for prompts",
            ));
        }
        self.one(s, t[0], &|| false, tap)
    }
    fn forward_rows(
        &self,
        s: &mut Session,
        t: &[u32],
        n: usize,
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        if n > 1 {
            return Err(err("Inkling supports at most one output row"));
        }
        let row = self.forward(s, t, tap)?;
        Ok(if n == 1 { vec![row] } else { Vec::new() })
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let s = self.sess_mut(s)?;
        if s.poisoned || !s.last || keep != 1 {
            return Err(err(
                "Inkling rollback requires full checkpoint for shortened pass; no draft support",
            ));
        }
        s.last = false;
        Ok(())
    }
    fn forward_batch(&self, s: &mut [&mut Session], t: &[u32], tap: Tap) -> Result<Vec<Vec<f32>>> {
        if s.len() != 1 || t.len() != 1 {
            return Err(err("Inkling batching unsupported"));
        }
        Ok(vec![self.forward(s[0], t, tap)?])
    }
    fn decoder(&self, logits: Vec<f32>, _: bool) -> Decoder {
        Decoder::new(Dec {
            owner: self.owner.clone(),
            epoch: self.decoder_epoch(),
            logits: Some(logits),
            next: None,
        })
    }
    fn decoder_after(&self, next: u32, _: bool) -> Decoder {
        Decoder::new(Dec {
            owner: self.owner.clone(),
            epoch: self.decoder_epoch(),
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
        self.step_until(s, d, smp, &|| false, tap)
    }
    fn step_until(
        &self,
        s: &mut Session,
        d: &mut Decoder,
        smp: &mut dyn Sampler,
        stop: &(dyn Fn() -> bool + Sync),
        tap: Tap,
    ) -> Result<Vec<u32>> {
        if stop() || self.cancel.load(Ordering::Relaxed) {
            return Err(err("Inkling generation cancelled"));
        }
        let ss = self.sess(s)?;
        if ss.poisoned {
            return Err(err("poisoned Inkling session"));
        }
        let d = d
            .get_mut::<Dec>()
            .filter(|d| Arc::ptr_eq(&d.owner, &self.owner))
            .ok_or_else(|| err("foreign Inkling decoder"))?;
        {
            let live = self.live()?;
            Self::lease(&live, ss)?;
            if d.epoch != Some(live.epoch) {
                return Err(err("stale Inkling decoder after reset/restore/new session"));
            }
        }
        if let Some(t) = d.next {
            let row = self.one(s, t, stop, tap)?;
            d.next = None;
            d.logits = Some(row);
        }
        let row = d
            .logits
            .as_ref()
            .ok_or_else(|| err("decoder token already handed off"))?;
        if row.len() != 200058 || row.iter().any(|x| !x.is_finite()) {
            return Err(err("invalid Inkling logits"));
        }
        if stop() || self.cancel.load(Ordering::Relaxed) {
            return Err(err("Inkling generation cancelled"));
        }
        let t = smp.try_sample(row)?;
        if t >= 200058 || t == 200054 {
            return Err(err("sampler selected invalid or unsupported image token"));
        }
        d.logits = None;
        d.next = Some(t);
        Ok(vec![t])
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        let s = self.sess(s)?;
        if s.poisoned {
            return Err(err("cannot save poisoned state"));
        }
        reserve_host(self.state_bytes)?;
        let l = self.live()?;
        Self::lease(&l, s)?;
        let (pos, data) = l.graph.snapshot().map_err(err)?;
        if pos != s.pos {
            return Err(err("session cursor mismatch"));
        }
        Ok(Checkpoint::new(Ck {
            owner: self.owner.clone(),
            context: self.context,
            pos,
            data,
        }))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck
            .get::<Ck>()
            .filter(|c| {
                Arc::ptr_eq(&c.owner, &self.owner)
                    && c.context == self.context
                    && c.data.len() == self.state_bytes
            })
            .ok_or_else(|| err("foreign/invalid Inkling checkpoint"))?;
        let s = self.sess_mut(s)?;
        if ck.pos > s.max {
            return Err(err("checkpoint exceeds session context"));
        }
        let mut l = self.live()?;
        Self::lease(&l, s)?;
        s.poisoned = true;
        s.last = false;
        Self::invalidate_decoders(&mut l)?;
        l.graph.restore_state(ck.pos, &ck.data).map_err(err)?;
        s.pos = ck.pos;
        s.poisoned = false;
        Ok(())
    }
    fn read_checkpoint(&self, r: &mut dyn Read) -> io::Result<Checkpoint> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid Inkling complete-state checkpoint",
            )
        };
        let mut h = [0u8; 32];
        r.read_exact(&mut h)?;
        if &h[..8] != MAGIC {
            return Err(invalid());
        }
        let val = |i: usize| u64::from_le_bytes(h[i..i + 8].try_into().expect("eight bytes"));
        if val(8) != self.context as u64
            || val(16) > self.context as u64
            || val(24) != self.state_bytes as u64
        {
            return Err(invalid());
        }
        reserve_host(self.state_bytes).map_err(|e| io::Error::other(e.0))?;
        let mut data = Vec::new();
        data.try_reserve_exact(self.state_bytes)
            .map_err(io::Error::other)?;
        data.resize(self.state_bytes, 0);
        r.read_exact(&mut data)?;
        if data
            .chunks_exact(4)
            .any(|b| !f32::from_le_bytes(b.try_into().expect("four bytes")).is_finite())
        {
            return Err(invalid());
        }
        Ok(Checkpoint::new(Ck {
            owner: self.owner.clone(),
            context: self.context,
            pos: val(16) as usize,
            data,
        }))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        vec![GpuInfo {
            index: self.gpu.index,
            name: self.gpu.name.clone(),
            pci: self.gpu.pci.clone(),
            total,
            free,
            layers: (0, 48),
            expert_slots: 1,
            host_slots: 0,
        }]
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_chat_rejects_unsupported_before_model_work() {
        assert!(chat_checked(
            &[serde_json::json!({"role":"user","content":"Hi"})],
            None,
            &[]
        )
        .unwrap()
        .contains("Hi"));
        for m in [
            serde_json::json!({"role":"tool","content":"x"}),
            serde_json::json!({"role":"user","content":[{"type":"image_url"}]}),
            serde_json::json!({"role":"user","content":"<|content_image|>"}),
            serde_json::json!({"role":"assistant","content":"a","tool_calls":[]}),
        ] {
            assert!(chat_checked(&[m], None, &[]).is_err());
        }
        assert!(chat_checked(
            &[serde_json::json!({"role":"user","content":"x"})],
            None,
            &[serde_json::json!({})]
        )
        .is_err());
    }
}

#[cfg(test)]
mod staging_policy_tests {
    use super::*;
    use crate::expert_cache::{ExpertMode, MAX_CACHE_BYTES};
    #[test]
    fn default_cache_and_explicit_stream_policy() {
        assert_eq!(
            configured_expert_mode(None).unwrap(),
            ExpertMode::DeviceLru {
                budget_bytes: MAX_CACHE_BYTES
            }
        );
        assert_eq!(configured_expert_mode(Some(0)).unwrap(), ExpertMode::Stream);
        assert_eq!(
            configured_expert_mode(Some(6469632)).unwrap(),
            ExpertMode::DeviceLru {
                budget_bytes: 6469632
            }
        );
        assert!(configured_expert_mode(Some(MAX_CACHE_BYTES as usize + 1)).is_err());
        // Syntactically bounded budgets still must hold the actual selected pair.
        assert!(configured_expert_mode(Some(1))
            .unwrap()
            .extra_bytes(6469632)
            .is_err());
    }
}
