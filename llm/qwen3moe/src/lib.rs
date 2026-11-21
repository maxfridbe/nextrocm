//! Native Qwen3-MoE text backend. Sequential one-token prefill and
//! bounded file-window streaming follow the original Engine contract.
mod chat;
mod contract;
mod geometry;
mod kernels;
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
/// Registry entry for the Qwen3-MoE architecture.
pub fn kind() -> EngineKind {
    EngineKind {
        archs: &["qwen3moe"],
        name: "Qwen3-Coder MoE native HIP (resident weights, streamed experts)",
        load,
        info,
        kernels: None,
        chat: nextrocm_llm::ChatTemplate::Legacy(chat::chat),
        options: &[],
        prepare: None,
    }
}
pub fn info(f: &Gguf) -> std::result::Result<String, String> {
    let g = Geometry::read(f).map_err(|e| e.0)?;
    Ok(format!("{}: {} full-attention layers, width {}, {} / {} heads of {}, {} experts top {}, packed resident non-experts, 16MiB streamed experts", g.arch,g.layers,g.width,g.heads,g.kv_heads,g.head,g.experts,g.used))
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
    gate: DevBuf,
    up: DevBuf,
    logits: DevBuf,
    router: DevBuf,
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
            g.heads * g.head,
            g.heads * g.head,
            g.kv_heads * g.head,
            g.kv_heads * g.head,
            g.heads * g.head,
            g.width,
            g.ff,
            g.ff,
            g.vocab,
            g.experts,
            g.used,
            g.used,
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
            gate: next()?,
            up: next()?,
            logits: next()?,
            router: next()?,
            route_ids: next()?,
            route_weights: next()?,
            ffn_sum: next()?,
            stage: DevBuf::new(gpu, TILE)?,
        })
    }
}
struct Qwen3Moe {
    geom: Geometry,
    gpu: Arc<Gpu>,
    weights: BTreeMap<String, Weight>,
    store: Store,
    kernels: Kernels,
    scratch: Mutex<Scratch>,
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
            "Qwen3Moe needs {bytes} bytes plus {MARGIN} margin; free {free:?}"
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
        return Err(error("Qwen3Moe requires exactly one native HIP GPU"));
    }
    if o.draft {
        return Err(error("Qwen3Moe baseline does not implement MTP drafting"));
    }
    let gpu = Arc::clone(&gpus[0]);
    let kernels = Kernels::load()?;
    let store = Store::open(f)?;
    let (weights, bytes) = weight_plan(f, &geom)?;
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
            if w.tensor.shape.len() == 1 {
                // Only small F32 parameters are read back for validation.
                let values = b.to_f32()?;
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(error(format!("invalid f32 parameter {name}")));
                }
            }
            w.resident = Some(b);
        }
    }
    let scratch = Mutex::new(Scratch::new(&gpu, &geom)?);
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
    log(format!("{} native HIP: {bytes} tracked load bytes; packed non-expert weights resident, 16MiB mapped expert staging, one-token prefill",geom.arch));
    Ok(Box::new(Qwen3Moe {
        geom,
        gpu,
        weights,
        store,
        kernels,
        scratch,
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
impl Qwen3Moe {
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
            let packed = if let Some(resident) = &w.resident {
                resident.view(offset, rows * w.row_bytes)?
            } else {
                if !w.streamed() {
                    return Err(error("non-expert weight is not resident"));
                }
                self.store.copy(w, offset, rows * w.row_bytes, &a.stage)?;
                a.stage.view(0, rows * w.row_bytes)?
            };
            let out = y.view(first * 4, rows * 4)?;
            // SAFETY: validated packed row extent; x/y bounds above, staging synchronized and all buffers on this GPU.
            check(unsafe {
                (self.kernels.mv)(
                    self.gpu.raw(),
                    packed.ptr(),
                    w.tensor.ty.code() as i32,
                    rows as i32,
                    w.cols as i32,
                    x.fp(),
                    out.fp(),
                )
            })?;
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
    fn token(&self, s: &mut Sess, token: u32, a: &Scratch, tap: Tap) -> Result<Vec<f32>> {
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
            let name = |n: &str| format!("blk.{l}.{n}");
            self.norm(&a.x, &name("attn_norm.weight"), &a.norm)?;
            tap(&format!("attn_norm-{l}"), &a.norm)?;
            let cache = &s.cache[l];
            {
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
                        g.head as i32,
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
                }
                self.mv(&name("attn_output.weight"), 0, &a.attn, &a.proj, a)?;
            }
            self.add(&a.x, &a.proj, g.width, 1.)?;
            tap(&format!("ffn_inp-{l}"), &a.x)?;
            self.norm(&a.x, &name("ffn_norm.weight"), &a.norm)?;
            tap(&format!("ffn_norm-{l}"), &a.norm)?;
            {
                self.mv(&name("ffn_gate_inp.weight"), 0, &a.norm, &a.router, a)?;
                tap(&format!("router_logits-{l}"), &a.router)?;
                let mut routing = store::route(&a.router.to_f32()?, g.used)?;
                for (slot, &(id, weight)) in routing.iter().enumerate() {
                    a.route_ids.write(slot * 4, &(id as f32).to_le_bytes())?;
                    a.route_weights.write(slot * 4, &weight.to_le_bytes())?;
                }
                tap(&format!("router_ids-{l}"), &a.route_ids)?;
                tap(&format!("router_weights-{l}"), &a.route_weights)?;
                routing.sort_by_key(|&(id, _)| id);
                a.ffn_sum.fill(0)?;
                for (expert, weight) in routing {
                    self.ffn(l, "_exps", expert, g.ff, a)?;
                    self.add(&a.ffn_sum, &a.proj, g.width, weight)?;
                }
                self.add(&a.x, &a.ffn_sum, g.width, 1.)?;
            }
            tap(&format!("l_out-{l}"), &a.x)?;
        }
        self.norm(&a.x, "output_norm.weight", &a.norm)?;
        tap("result_norm", &a.norm)?;
        self.mv("output.weight", 0, &a.norm, &a.logits, a)?;
        tap("result_output", &a.logits)?;
        let logits = a.logits.to_f32()?;
        if logits.iter().any(|v| !v.is_finite()) {
            return Err(error("Qwen3Moe produced non-finite logits"));
        }
        s.pos += 1;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests;
