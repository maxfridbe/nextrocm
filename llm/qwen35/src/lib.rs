//! Native Qwen35/Qwen35MoE text backend. Default serial and opt-in layer-major
//! prefill with bounded streamed experts follow the original Engine contract.
mod batch;
mod contract;
mod expert_cache;
mod geometry;
mod host_mirror;
mod kernels;
mod prefetch;
mod profile;
mod route_profile;
mod session_access;
mod state;
mod store;
pub use geometry::Geometry;
use kernels::{check, Kernels};
use nextrocm_gguf::Gguf;
use nextrocm_llm::*;
use state::{Ck, Dec, Sess};
use std::{
    any::Any,
    collections::BTreeMap,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
    time::Instant,
};
use store::{Reservation, Store, Weight, TILE};
const CHUNK: usize = 1;
const MARGIN: u64 = 256 * 1024 * 1024;
fn error(s: impl Into<String>) -> Error {
    Error(s.into())
}
/// One original registry entry for dense and MoE GGUF architectures.
pub fn kind() -> EngineKind {
    EngineKind {
        archs: &["qwen35", "qwen35moe"],
        name: "Qwen3.5 / Qwen3.6 native HIP (resident weights, streamed experts)",
        load,
        info,
        kernels: None,
        chat: nextrocm_llm::ChatTemplate::Legacy(nextrocm_tok::qwen36_chat),
        options: &[EngineOption {
            name: "chunk",
            env: "NS_Q35_CHUNK",
            value: "1|8|32|64",
            help: "prompt rows per layer-major pass (1: serial default; batch requires batch.hip)",
            at: At::Load,
        }, EngineOption {
            name: "scalar",
            env: "NS_Q35_SCALAR",
            value: "0|1|2|3",
            help: "scalar matvec variant (0: original default; 1: specialize; 2: shuffle tail; 3: combined)",
            at: At::Load,
        }, EngineOption {
            name: "host-mib", env: "NS_Q35_HOST_MIB", value: "0..2048",
            help: "bounded packed expert RAM mirror MiB (default 0)", at: At::Load,
        }, EngineOption {
            name: "expert-profile", env: "NS_Q35_EXPERT_PROFILE", value: "PATH",
            help: "model-matched expert frequency profile for RAM preloading", at: At::Load,
        }, EngineOption {
            name: "prefetch-mib", env: "NS_Q35_PREFETCH_MIB", value: "0..32",
            help: "bounded selected-expert read-ahead MiB (default 0)", at: At::Load,
        }, EngineOption {
            name: "record-experts", env: "NS_Q35_RECORD_EXPERTS", value: "0|1",
            help: "record actual selected routes across mixed PP/TG (default 0)", at: At::Load,
        }],
        prepare: None,
    }
}
pub fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = Geometry::read(f).map_err(|e| e.0)?;
    Ok(format!("{}: {} layers, width {}, {} full / {} KV heads, DeltaNet {}/{} heads, {} experts top {}, resident packed non-experts, streaming16MiB experts, fp16 KV",g.arch,g.layers,g.width,g.heads,g.kv_heads,g.k_heads,g.v_heads,g.experts,g.used))
}
struct Scratch {
    x: DevBuf,
    norm: DevBuf,
    qf: DevBuf,
    q: DevBuf,
    k: DevBuf,
    v: DevBuf,
    attn: DevBuf,
    proj: DevBuf,
    qkv: DevBuf,
    co: DevBuf,
    z: DevBuf,
    alpha: DevBuf,
    beta: DevBuf,
    delta: DevBuf,
    gate: DevBuf,
    up: DevBuf,
    logits: DevBuf,
    router: DevBuf,
    shared_gate: DevBuf,
    stage: DevBuf,
    route_ids: DevBuf,
    route_weights: DevBuf,
    ffn_sum: DevBuf,
}
impl Scratch {
    fn sizes(g: &Geometry) -> Vec<usize> {
        vec![
            g.width,
            g.width,
            2 * g.heads * g.head,
            g.heads * g.head,
            g.kv_heads * g.head,
            g.kv_heads * g.head,
            g.inner().max(g.heads * g.head),
            g.width,
            g.channels(),
            g.channels(),
            g.inner(),
            g.v_heads,
            g.v_heads,
            g.inner(),
            g.ff.max(g.shared),
            g.ff.max(g.shared),
            g.vocab,
            g.experts.max(1),
            1,
            g.used.max(1),
            g.used.max(1),
            g.width,
        ]
    }
    fn bytes(g: &Geometry) -> usize {
        Self::sizes(g).iter().sum::<usize>() * 4 + TILE
    }
    fn new(gpu: &Arc<Gpu>, g: &Geometry) -> Result<Self> {
        let mut b = Self::sizes(g)
            .into_iter()
            .map(|n| DevBuf::f32(gpu, n))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut next = || b.next().ok_or_else(|| error("scratch layout mismatch"));
        Ok(Self {
            x: next()?,
            norm: next()?,
            qf: next()?,
            q: next()?,
            k: next()?,
            v: next()?,
            attn: next()?,
            proj: next()?,
            qkv: next()?,
            co: next()?,
            z: next()?,
            alpha: next()?,
            beta: next()?,
            delta: next()?,
            gate: next()?,
            up: next()?,
            logits: next()?,
            router: next()?,
            shared_gate: next()?,
            route_ids: next()?,
            route_weights: next()?,
            ffn_sum: next()?,
            stage: DevBuf::new(gpu, TILE)?,
        })
    }
}
struct Qwen35 {
    geom: Geometry,
    gpu: Arc<Gpu>,
    weights: BTreeMap<String, Weight>,
    store: Store,
    profile: profile::Profile,
    route_profile: route_profile::RouteProfile,
    kernels: Kernels,
    scratch: Mutex<Scratch>,
    batch: Option<Mutex<batch::BatchScratch>>,
    chunk: usize,
    scalar: i32,
    expert_cache: expert_cache::ExpertCache,
    expert_cache_enabled: bool,
    owner: Arc<()>,
    seconds: f64,
    bytes: u64,
    model: u64,
    _reservation: Reservation,
}
fn ensure_memory(gpu: &Gpu, bytes: u64) -> Result<()> {
    let (_, free) = gpu.memory()?;
    if free.is_some_and(|free| bytes.checked_add(MARGIN).is_none_or(|n| n > free)) {
        return Err(error(format!(
            "Qwen35 needs {bytes} bytes plus {MARGIN} margin; free {free:?}"
        )));
    }
    Ok(())
}
/// Loads packed non-expert weights; selected experts stream from disk. Model files
/// must be immutable while loaded (the same requirement as other mmap readers).
pub fn load<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn Engine + 'g>> {
    let start = Instant::now();
    let geom = Geometry::read(f)?;
    if gpus.len() != 1 || gpus[0].raw().is_null() {
        return Err(error("Qwen35 requires exactly one native HIP GPU"));
    }
    if o.draft {
        return Err(error("Qwen35 baseline does not implement MTP drafting"));
    }
    let gpu = Arc::clone(&gpus[0]);
    let kernels = Kernels::load()?;
    let scalar = scalar_setting(
        std::env::var("NS_Q35_SCALAR").ok().as_deref(),
        kernels.mv_fast.is_some(),
    )?;
    let chunk = batch::chunk_setting(std::env::var("NS_Q35_CHUNK").ok().as_deref())?;
    if chunk > 1 && kernels.mv_batch.is_none() {
        return Err(error(
            "NS_Q35_CHUNK > 1 requires nq35_mv_batch from batch.hip; use 1 with an older runtime",
        ));
    }
    let host_budget = host_budget(
        o.mirror_bytes,
        std::env::var("NS_Q35_HOST_MIB").ok().as_deref(),
    )?;
    let record_routes = match std::env::var("NS_Q35_RECORD_EXPERTS") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(v) if v == "0" => false,
        Ok(v) if v == "1" => true,
        _ => return Err(error("NS_Q35_RECORD_EXPERTS must be 0 or 1")),
    };
    let mut store = Store::open(f, &gpu)?;
    let (weights, bytes) = weight_plan(f, &geom)?;
    let bytes = bytes
        .checked_add(if chunk > 1 {
            batch::BatchScratch::bytes(&geom, chunk)?
        } else {
            0
        })
        .ok_or_else(|| error("batch load reservation overflow"))?;
    // Reserve all resident weights plus scratch before the first device allocation.
    // Keep a driver/runtime margin within the global admission cap as well.
    let reserved = bytes
        .checked_add(MARGIN as usize)
        .ok_or_else(|| error("load reservation overflow"))?;
    let reservation = Reservation::device(reserved)?;
    ensure_memory(&gpu, bytes as u64)?;
    // On partial-load failure, drop allocations before releasing their reservation.
    let mut weights = weights;
    for (name, w) in &mut weights {
        if !w.streamed() {
            let b = DevBuf::new(&gpu, w.resident_bytes()?)?;
            // Map at most TILE bytes at a time, copying directly into the final
            // packed allocation. Never materialize a whole matrix on the host.
            for at in (0..b.len).step_by(TILE) {
                let n = TILE.min(b.len - at);
                let dst = b.view(at, n)?;
                store.copy(w, at, n, &dst)?;
            }
            if w.tensor.shape.len() == 1 || name.ends_with("ssm_conv1d.weight") {
                // Only small F32 parameters are read back for validation.
                // Decay is already -exp(A_log), as in GGUF conversion.
                let values = b.to_f32()?;
                if values.iter().any(|v| !v.is_finite())
                    || (name.ends_with("ssm_a") && values.iter().any(|v| *v >= 0.0))
                {
                    return Err(error(format!("invalid f32 parameter {name}")));
                }
            }
            w.resident = Some(b);
        }
    }
    let scratch = Scratch::new(&gpu, &geom)?;
    let batch = if chunk > 1 {
        Some(Mutex::new(batch::BatchScratch::new(
            &gpu,
            &geom,
            chunk,
            &scratch.stage,
        )?))
    } else {
        None
    };
    let scratch = Mutex::new(scratch);
    gpu.sync()?;
    // Fingerprint binds disk checkpoints to this model directory and tensor table.
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    f.paths.hash(&mut hash);
    format!("{:?}", geom).hash(&mut hash);
    for t in &f.tensors {
        t.name.hash(&mut hash);
        t.shape.hash(&mut hash);
        t.ty.code().hash(&mut hash);
        t.offset.hash(&mut hash);
        t.bytes.hash(&mut hash);
    }
    let model = hash.finish();
    let model_bytes = f.paths.iter().try_fold(0u64, |n, p| {
        let len = std::fs::metadata(p)
            .map_err(|e| error(format!("model size: {e}")))?
            .len();
        n.checked_add(len)
            .ok_or_else(|| error("model file size overflow"))
    })?;
    let route_profile = route_profile::RouteProfile::new(
        record_routes,
        geom.layers,
        geom.experts,
        model_bytes,
        model,
    )?;
    // KV request is the aggregate token budget; every session also owns Delta
    // rollback snapshots. Reserve these future allocations before RAM preloading.
    let future_device = future_device_budget(
        geom.cache_bytes(o.kv.0)?,
        geom.cache_bytes(0)?,
        o.kv.1,
        o.expert_bytes,
    )?;
    store.configure(&gpu, f, &weights, model, host_budget, future_device as u64)?;
    log(format!("{} native HIP: {bytes} tracked load bytes; packed non-expert weights resident, 16MiB mapped expert staging, prefill chunk {chunk}",geom.arch));
    let profile = profile::Profile::new(&gpu);
    Ok(Box::new(Qwen35 {
        geom,
        gpu,
        weights,
        store,
        profile,
        route_profile,
        kernels,
        scratch,
        batch,
        chunk,
        scalar,
        expert_cache: expert_cache::ExpertCache::new(o.expert_bytes.unwrap_or(0)),
        expert_cache_enabled: o.expert_bytes.unwrap_or(0) > 0,
        owner: Arc::new(()),
        seconds: start.elapsed().as_secs_f64(),
        bytes: bytes as u64,
        model,
        _reservation: reservation,
    }))
}
/// Build and size the complete allocation plan without touching the GPU.
fn weight_plan(f: &Gguf, geom: &Geometry) -> Result<(BTreeMap<String, Weight>, usize)> {
    let mut weights = BTreeMap::new();
    let mut bytes = Scratch::bytes(geom);
    for (name, _) in geom.specs() {
        // A tied output uses the embedding allocation directly, without a second
        // allocation or a non-owning view stored beyond its owner's lifetime.
        if name == "output.weight" && f.tensor(&name).is_none() {
            continue;
        }
        let t = f
            .tensor(&name)
            .ok_or_else(|| error(format!("missing {name}")))?;
        let w = Weight::new(t)?;
        bytes = bytes
            .checked_add(w.resident_bytes()?)
            .ok_or_else(|| error("resident size overflow"))?;
        weights.insert(name, w);
    }
    Ok((weights, bytes))
}
impl Qwen35 {
    fn weight(&self, name: &str) -> Result<&Weight> {
        self.weights
            .get(name)
            .or_else(|| {
                (name == "output.weight")
                    .then(|| self.weights.get("token_embd.weight"))
                    .flatten()
            })
            .ok_or_else(|| error(format!("missing weight {name}")))
    }
    fn param(&self, name: &str) -> Result<&DevBuf> {
        self.weight(name)?
            .resident
            .as_ref()
            .ok_or_else(|| error(format!("not a resident f32 parameter: {name}")))
    }
    fn packed(
        &self,
        w: &Weight,
        offset: usize,
        len: usize,
        stage: &DevBuf,
    ) -> Result<(DevBuf, Option<Arc<expert_cache::Cached>>)> {
        if let Some(resident) = &w.resident {
            return Ok((resident.view(offset, len)?, None));
        }
        if !w.streamed() {
            return Err(error("non-expert weight is not resident"));
        }
        // Cache entries are immutable allocations. Keep the lease through dispatch;
        // eviction's owning DevBuf synchronizes before freeing, unlike mutable staging.
        let lookup_started = if self.expert_cache_enabled {
            self.profile.start()?
        } else {
            None
        };
        let cached = self
            .expert_cache
            .get(&self.gpu, &self.store, w, offset, len)?;
        // Disjoint from staging below: a cache miss can fill inside get().
        // Bytes count the requested extent, including hits, not actual H2D.
        self.profile
            .record("expert_cache_lookup_and_fill", lookup_started, len)?;
        if let Some(entry) = cached {
            return Ok((entry.buffer.view(0, len)?, Some(entry)));
        }
        let started = self.profile.start()?;
        self.store.copy(w, offset, len, stage)?;
        self.profile.record("expert_transfer", started, len)?;
        Ok((stage.view(0, len)?, None))
    }
    fn mv(&self, name: &str, expert: usize, x: &DevBuf, y: &DevBuf, a: &Scratch) -> Result<()> {
        let w = self.weight(name)?;
        if expert >= w.experts {
            return Err(error("expert id outside packed tensor"));
        }
        x.bounds(0, w.cols * 4)?;
        y.bounds(0, w.rows * 4)?;
        let chunk = (TILE / w.row_bytes)
            .min((i32::MAX as usize) / w.cols)
            .max(1);
        for first in (0..w.rows).step_by(chunk) {
            let rows = chunk.min(w.rows - first);
            let offset = (expert * w.rows + first) * w.row_bytes;
            let (packed, _lease) = self.packed(w, offset, rows * w.row_bytes, &a.stage)?;
            let out = y.view(first * 4, rows * 4)?;
            let started = self.profile.start()?;
            // SAFETY: validated packed row extent; x/y bounds above, staging synchronized and all buffers on this GPU.
            check(unsafe {
                if self.scalar == 0 {
                    (self.kernels.mv)(
                        self.gpu.raw(),
                        packed.ptr(),
                        w.tensor.ty.code() as i32,
                        rows as i32,
                        w.cols as i32,
                        x.fp(),
                        out.fp(),
                    )
                } else {
                    let fast = self
                        .kernels
                        .mv_fast
                        .ok_or_else(|| error("missing nq35_mv_fast"))?;
                    fast(
                        self.gpu.raw(),
                        packed.ptr(),
                        w.tensor.ty.code() as i32,
                        rows as i32,
                        w.cols as i32,
                        x.fp(),
                        out.fp(),
                        self.scalar,
                    )
                }
            })?;
            self.profile.record(
                if w.streamed() {
                    "expert_matvec"
                } else {
                    "resident_matvec"
                },
                started,
                rows * w.row_bytes,
            )?;
        }
        Ok(())
    }
    fn norm(&self, x: &DevBuf, name: &str, y: &DevBuf) -> Result<()> {
        let n = self.geom.width;
        let w = self.param(name)?;
        x.bounds(0, n * 4)?;
        y.bounds(0, n * 4)?;
        // SAFETY: geometry and parameter shape validated; activation extents checked above.
        check(unsafe {
            (self.kernels.norm)(
                self.gpu.raw(),
                x.fp(),
                w.fp(),
                n as i32,
                self.geom.eps,
                y.fp(),
            )
        })
    }
    fn add(&self, x: &DevBuf, y: &DevBuf, n: usize, scale: f32) -> Result<()> {
        x.bounds(0, n * 4)?;
        y.bounds(0, n * 4)?;
        // SAFETY: vector extents checked; same GPU and ordered stream.
        check(unsafe { (self.kernels.axpy)(self.gpu.raw(), x.fp(), y.fp(), n as i32, scale) })
    }
    fn ffn(&self, l: usize, suffix: &str, expert: usize, n: usize, a: &Scratch) -> Result<()> {
        self.mv(
            &format!("blk.{l}.ffn_gate{suffix}.weight"),
            expert,
            &a.norm,
            &a.gate,
            a,
        )?;
        self.mv(
            &format!("blk.{l}.ffn_up{suffix}.weight"),
            expert,
            &a.norm,
            &a.up,
            a,
        )?;
        a.gate.bounds(0, n * 4)?;
        a.up.bounds(0, n * 4)?;
        // SAFETY: validated feed-forward dimensions and above bounds.
        check(unsafe { (self.kernels.swiglu)(self.gpu.raw(), a.gate.fp(), a.up.fp(), n as i32) })?;
        self.mv(
            &format!("blk.{l}.ffn_down{suffix}.weight"),
            expert,
            &a.gate,
            &a.proj,
            a,
        )
    }
    fn token(
        &self,
        s: &mut Sess,
        token: u32,
        a: &Scratch,
        tap: Tap,
        want_logits: bool,
    ) -> Result<Vec<f32>> {
        let g = &self.geom;
        let k = &self.kernels;
        let raw = self.gpu.raw();
        let w = self.weight("token_embd.weight")?;
        if token as usize >= g.vocab {
            return Err(error("token outside vocabulary"));
        }
        let row = w
            .resident
            .as_ref()
            .ok_or_else(|| error("embedding is not resident"))?
            .view(token as usize * w.row_bytes, w.row_bytes)?;
        // SAFETY: packed embedding row and scratch width validated at load; allocation lives for the model.
        check(unsafe {
            (k.embed)(
                raw,
                row.ptr(),
                w.tensor.ty.code() as i32,
                g.width as i32,
                a.x.fp(),
            )
        })?;
        tap("inp_embd", &a.x)?;
        for l in 0..g.layers {
            let layer_started = self.profile.start_layer()?;
            let name = |n: &str| format!("blk.{l}.{n}");
            self.norm(&a.x, &name("attn_norm.weight"), &a.norm)?;
            tap(&format!("attn_norm-{l}"), &a.norm)?;
            let cache = &s.cache[l];
            if g.full(l) {
                self.mv(&name("attn_q.weight"), 0, &a.norm, &a.qf, a)?;
                self.mv(&name("attn_k.weight"), 0, &a.norm, &a.k, a)?;
                self.mv(&name("attn_v.weight"), 0, &a.norm, &a.v, a)?;
                let qw = self.param(&name("attn_q_norm.weight"))?;
                let kw = self.param(&name("attn_k_norm.weight"))?;
                // SAFETY: Scratch and session caches derive from this validated geometry, pos<max_ctx checked by validate_pass.
                // qk supports in-place key normalization through a shared-memory row; query output is distinct.
                unsafe {
                    check((k.qk)(
                        raw,
                        a.qf.fp(),
                        qw.fp(),
                        g.heads as i32,
                        g.head as i32,
                        (2 * g.head) as i32,
                        g.rotary as i32,
                        s.pos as i32,
                        g.theta,
                        g.eps,
                        a.q.fp(),
                    ))?;
                    check((k.qk)(
                        raw,
                        a.k.fp(),
                        kw.fp(),
                        g.kv_heads as i32,
                        g.head as i32,
                        g.head as i32,
                        g.rotary as i32,
                        s.pos as i32,
                        g.theta,
                        g.eps,
                        a.k.fp(),
                    ))?;
                }
                tap(&format!("Qcur-{l}"), &a.q)?;
                tap(&format!("Kcur-{l}"), &a.k)?;
                tap(&format!("Vcur-{l}"), &a.v)?;
                // SAFETY: query/key/value head extents and cache capacity validated at creation; same stream, no aliasing with cache.
                unsafe {
                    check((k.attn)(
                        raw,
                        a.q.fp(),
                        a.k.fp(),
                        a.v.fp(),
                        cache.a.ptr(),
                        cache.b.ptr(),
                        g.heads as i32,
                        g.kv_heads as i32,
                        g.head as i32,
                        s.pos as i32,
                        a.attn.fp(),
                    ))?;
                    check((k.gate)(
                        raw,
                        a.attn.fp(),
                        a.qf.fp(),
                        g.heads as i32,
                        g.head as i32,
                    ))?;
                }
                self.mv(&name("attn_output.weight"), 0, &a.attn, &a.proj, a)?;
            } else {
                self.mv(&name("attn_qkv.weight"), 0, &a.norm, &a.qkv, a)?;
                self.mv(&name("attn_gate.weight"), 0, &a.norm, &a.z, a)?;
                self.mv(&name("ssm_alpha.weight"), 0, &a.norm, &a.alpha, a)?;
                self.mv(&name("ssm_beta.weight"), 0, &a.norm, &a.beta, a)?;
                let conv = self.param(&name("ssm_conv1d.weight"))?;
                let dt = self.param(&name("ssm_dt.bias"))?;
                let decay = self.param(&name("ssm_a"))?;
                let norm = self.param(&name("ssm_norm.weight"))?;
                // SAFETY: all matrix outputs, resident parameters, history and recurrent extents checked against geometry at load/session creation.
                unsafe {
                    check((k.conv)(
                        raw,
                        a.qkv.fp(),
                        conv.fp(),
                        cache.b.fp(),
                        g.channels() as i32,
                        g.conv as i32,
                        a.co.fp(),
                    ))?;
                    check((k.delta)(
                        raw,
                        a.co.fp(),
                        a.alpha.fp(),
                        a.beta.fp(),
                        dt.fp(),
                        decay.fp(),
                        cache.a.fp(),
                        g.k_heads as i32,
                        g.v_heads as i32,
                        g.state as i32,
                        g.eps,
                        a.delta.fp(),
                    ))?;
                    check((k.gated_norm)(
                        raw,
                        a.delta.fp(),
                        a.z.fp(),
                        norm.fp(),
                        g.v_heads as i32,
                        g.state as i32,
                        g.eps,
                        a.attn.fp(),
                    ))?;
                }
                tap(&format!("delta_state-{l}"), &cache.a)?;
                self.mv(&name("ssm_out.weight"), 0, &a.attn, &a.proj, a)?;
            }
            self.add(&a.x, &a.proj, g.width, 1.)?;
            tap(&format!("ffn_inp-{l}"), &a.x)?;
            self.norm(&a.x, &name("post_attention_norm.weight"), &a.norm)?;
            tap(&format!("ffn_norm-{l}"), &a.norm)?;
            if g.experts == 0 {
                self.ffn(l, "", 0, g.ff, a)?;
                self.add(&a.x, &a.proj, g.width, 1.)?;
            } else {
                self.mv(&name("ffn_gate_inp.weight"), 0, &a.norm, &a.router, a)?;
                tap(&format!("router_logits-{l}"), &a.router)?;
                let routing = store::route(&a.router.to_f32()?, g.used)?;
                self.route_profile.record(l, &routing)?;
                self.prefetch_experts(l, routing.iter().map(|&(id, _)| id))?;
                let mut route_ids = Vec::with_capacity(routing.len() * 4);
                let mut route_weights = Vec::with_capacity(routing.len() * 4);
                for &(id, weight) in &routing {
                    route_ids.extend_from_slice(&(id as f32).to_le_bytes());
                    route_weights.extend_from_slice(&weight.to_le_bytes());
                }
                a.route_ids.write(0, &route_ids)?;
                a.route_weights.write(0, &route_weights)?;
                tap(&format!("router_ids-{l}"), &a.route_ids)?;
                tap(&format!("router_weights-{l}"), &a.route_weights)?;
                // Compute the shared gate before overwriting feed-forward scratch; same normalized input as the router.
                self.mv(
                    &name("ffn_gate_inp_shexp.weight"),
                    0,
                    &a.norm,
                    &a.shared_gate,
                    a,
                )?;
                let shared = a.shared_gate.to_f32()?[0];
                if !shared.is_finite() {
                    return Err(error("non-finite shared expert gate"));
                }
                a.ffn_sum.fill(0)?;
                for (expert, weight) in routing {
                    self.ffn(l, "_exps", expert, g.ff, a)?;
                    self.add(&a.ffn_sum, &a.proj, g.width, weight)?;
                }
                self.ffn(l, "_shexp", 0, g.shared, a)?;
                self.add(&a.ffn_sum, &a.proj, g.width, 1. / (1. + (-shared).exp()))?;
                self.add(&a.x, &a.ffn_sum, g.width, 1.)?;
            }
            tap(&format!("l_out-{l}"), &a.x)?;
            self.profile.record(
                if g.full(l) {
                    "full_layer_attention"
                } else {
                    "full_layer_delta"
                },
                layer_started,
                0,
            )?;
        }
        // Like the reference engine's prompt passes, only materialize requested
        // output rows. Intermediate tokens still update every layer's state.
        if !want_logits {
            s.pos += 1;
            return Ok(Vec::new());
        }
        self.norm(&a.x, "output_norm.weight", &a.norm)?;
        tap("result_norm", &a.norm)?;
        self.mv("output.weight", 0, &a.norm, &a.logits, a)?;
        tap("result_output", &a.logits)?;
        let logits = a.logits.to_f32()?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(error("Qwen35 produced non-finite logits"));
        }
        s.pos += 1;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests;

fn scalar_setting(value: Option<&str>, available: bool) -> Result<i32> {
    let variant = match value.unwrap_or("0") {
        "0" => 0,
        "1" => 1,
        "2" => 2,
        "3" => 3,
        _ => return Err(error("NS_Q35_SCALAR must be 0, 1, 2 or 3")),
    };
    if variant != 0 && !available {
        return Err(error("NS_Q35_SCALAR requires nq35_mv_fast: integrate kernels/llm/qwen35/scalar_fast.hip or use 0"));
    }
    Ok(variant)
}

fn future_device_budget(
    kv_bytes: usize,
    state_bytes: usize,
    slots: usize,
    expert_bytes: Option<usize>,
) -> Result<usize> {
    slots
        .max(1)
        .checked_mul(2)
        .and_then(|n| n.checked_sub(1))
        .and_then(|n| state_bytes.checked_mul(n))
        .and_then(|n| kv_bytes.checked_add(n))
        // The lazy device expert cache still consumes shared physical memory
        // later: leave room for its full configured capacity before RAM preload.
        .and_then(|n| n.checked_add(expert_bytes.unwrap_or(0)))
        .ok_or_else(|| error("future cache size overflow"))
}

fn host_budget(explicit: Option<usize>, env: Option<&str>) -> Result<usize> {
    const MAX: usize = 2048 * 1024 * 1024;
    let bytes = match explicit {
        Some(n) => n,
        None => env
            .unwrap_or("0")
            .parse::<usize>()
            .ok()
            .and_then(|n| n.checked_mul(1024 * 1024))
            .ok_or_else(|| error("NS_Q35_HOST_MIB must be an integer in 0..2048"))?,
    };
    if bytes > MAX {
        return Err(error("Qwen35 host mirror exceeds 2048 MiB"));
    }
    Ok(bytes)
}

impl Qwen35 {
    fn prefetch_experts(&self, layer: usize, ids: impl IntoIterator<Item = usize>) -> Result<()> {
        if !self.store.prefetch_enabled() {
            return Ok(());
        }
        let weights = ["gate", "up", "down"]
            .map(|role| self.weight(&format!("blk.{layer}.ffn_{role}_exps.weight")));
        let [gate, up, down] = weights;
        let weights = [gate?, up?, down?];
        let mut requests = Vec::new();
        for id in ids {
            for w in weights {
                if id >= w.experts {
                    return Err(error("prefetch expert outside tensor"));
                }
                let len = w
                    .rows
                    .checked_mul(w.row_bytes)
                    .ok_or_else(|| error("prefetch size overflow"))?;
                let offset = id
                    .checked_mul(len)
                    .ok_or_else(|| error("prefetch offset overflow"))?;
                if offset
                    .checked_add(len)
                    .is_none_or(|end| end as u64 > w.tensor.bytes)
                {
                    return Err(error("prefetch extent outside tensor"));
                }
                requests.push((w, offset, len));
            }
        }
        self.store.prefetch(&requests);
        Ok(())
    }
}

#[cfg(test)]
mod scalar_option_tests {
    use super::*;

    #[test]
    fn future_device_budget_includes_expert_cache_and_checks_overflow() {
        assert_eq!(future_device_budget(100, 10, 1, None).unwrap(), 110);
        assert_eq!(future_device_budget(100, 10, 1, Some(0)).unwrap(), 110);
        assert_eq!(future_device_budget(100, 10, 2, Some(50)).unwrap(), 180);
        assert!(future_device_budget(100, 10, 1, Some(usize::MAX)).is_err());
        assert!(future_device_budget(usize::MAX, 1, 1, None).is_err());
        assert!(future_device_budget(0, usize::MAX, 2, None).is_err());
        assert!(future_device_budget(0, 0, usize::MAX, None).is_err());
    }

    #[test]
    fn host_budget_explicit_zero_overrides_environment_and_bounds_sizes() {
        assert_eq!(host_budget(None, None).unwrap(), 0);
        assert_eq!(host_budget(Some(0), Some("2048")).unwrap(), 0);
        assert_eq!(host_budget(Some(1), Some("invalid")).unwrap(), 1);
        assert_eq!(host_budget(None, Some("2048")).unwrap(), 2048 * 1024 * 1024);
        for value in ["-1", "2049", "x", "18446744073709551615"] {
            assert!(host_budget(None, Some(value)).is_err());
        }
        assert!(host_budget(Some(usize::MAX), None).is_err());
    }

    #[test]
    fn default_keeps_old_runtime_and_requested_variants_require_symbol() {
        for available in [false, true] {
            assert_eq!(scalar_setting(None, available).unwrap(), 0);
            assert_eq!(scalar_setting(Some("0"), available).unwrap(), 0);
        }
        for (value, expected) in [("1", 1), ("2", 2), ("3", 3)] {
            assert_eq!(scalar_setting(Some(value), true).unwrap(), expected);
            assert!(scalar_setting(Some(value), false).is_err());
        }
        for value in ["", "-1", "4", "01", "true", " 1"] {
            assert!(scalar_setting(Some(value), true).is_err());
        }
    }
}
