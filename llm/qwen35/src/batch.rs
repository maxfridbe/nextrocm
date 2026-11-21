//! Opt-in layer-major prefill; scalar state updates preserve the established cache layout.
use super::*;

pub(crate) fn chunk_setting(value: Option<&str>) -> Result<usize> {
    let n = value
        .unwrap_or("1")
        .parse::<usize>()
        .map_err(|_| error("NS_Q35_CHUNK must be 1, 8, 32, or 64"))?;
    if !matches!(n, 1 | 8 | 32 | 64) {
        return Err(error("NS_Q35_CHUNK must be 1, 8, 32, or 64"));
    }
    Ok(n)
}
fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, d| {
        n.checked_mul(*d)
            .ok_or_else(|| error("batch size overflow"))
    })
}

pub(crate) struct BatchScratch {
    rows: Vec<Scratch>,
    all: Scratch,
    gather: DevBuf,
    slots: DevBuf,
}
impl BatchScratch {
    fn upload_routes(&self, rows: &[Vec<(usize, f32)>]) -> Result<()> {
        let ids: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|(id, _)| (*id as f32).to_le_bytes())
            .collect();
        let weights: Vec<u8> = rows
            .iter()
            .flatten()
            .flat_map(|(_, weight)| weight.to_le_bytes())
            .collect();
        self.all.route_ids.write(0, &ids)?;
        self.all.route_weights.write(0, &weights)
    }

    pub fn bytes(g: &Geometry, n: usize) -> Result<usize> {
        let floats = Scratch::sizes(g).iter().try_fold(0usize, |sum, &v| {
            sum.checked_add(v)
                .ok_or_else(|| error("batch size overflow"))
        })?;
        let extra = g
            .width
            .checked_add(product(&[g.used.max(1), g.width])?)
            .ok_or_else(|| error("batch size overflow"))?;
        product(&[
            floats
                .checked_add(extra)
                .ok_or_else(|| error("batch size overflow"))?,
            n,
            4,
        ])
    }
    pub fn new(gpu: &Arc<Gpu>, g: &Geometry, n: usize, stage: &DevBuf) -> Result<Self> {
        let mut buffers = Scratch::sizes(g)
            .into_iter()
            .map(|v| DevBuf::f32(gpu, product(&[v, n])?))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        let mut next = || {
            buffers
                .next()
                .ok_or_else(|| error("batch scratch layout mismatch"))
        };
        let all = Scratch {
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
            stage: stage.view(0, stage.len)?,
        };
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            rows.push(Scratch {
                x: all.x.view(i * (all.x.len / n), all.x.len / n)?,
                norm: all.norm.view(i * (all.norm.len / n), all.norm.len / n)?,
                qf: all.qf.view(i * (all.qf.len / n), all.qf.len / n)?,
                q: all.q.view(i * (all.q.len / n), all.q.len / n)?,
                k: all.k.view(i * (all.k.len / n), all.k.len / n)?,
                v: all.v.view(i * (all.v.len / n), all.v.len / n)?,
                attn: all.attn.view(i * (all.attn.len / n), all.attn.len / n)?,
                proj: all.proj.view(i * (all.proj.len / n), all.proj.len / n)?,
                qkv: all.qkv.view(i * (all.qkv.len / n), all.qkv.len / n)?,
                co: all.co.view(i * (all.co.len / n), all.co.len / n)?,
                z: all.z.view(i * (all.z.len / n), all.z.len / n)?,
                alpha: all.alpha.view(i * (all.alpha.len / n), all.alpha.len / n)?,
                beta: all.beta.view(i * (all.beta.len / n), all.beta.len / n)?,
                delta: all.delta.view(i * (all.delta.len / n), all.delta.len / n)?,
                gate: all.gate.view(i * (all.gate.len / n), all.gate.len / n)?,
                up: all.up.view(i * (all.up.len / n), all.up.len / n)?,
                logits: all
                    .logits
                    .view(i * (all.logits.len / n), all.logits.len / n)?,
                router: all
                    .router
                    .view(i * (all.router.len / n), all.router.len / n)?,
                shared_gate: all
                    .shared_gate
                    .view(i * (all.shared_gate.len / n), all.shared_gate.len / n)?,
                route_ids: all
                    .route_ids
                    .view(i * (all.route_ids.len / n), all.route_ids.len / n)?,
                route_weights: all
                    .route_weights
                    .view(i * (all.route_weights.len / n), all.route_weights.len / n)?,
                ffn_sum: all
                    .ffn_sum
                    .view(i * (all.ffn_sum.len / n), all.ffn_sum.len / n)?,
                stage: stage.view(0, stage.len)?,
            });
        }
        Ok(Self {
            rows,
            all,
            gather: DevBuf::f32(gpu, product(&[n, g.width])?)?,
            slots: DevBuf::f32(gpu, product(&[n, g.used.max(1), g.width])?)?,
        })
    }
}

impl Qwen35 {
    pub(crate) fn batch_pass(
        &self,
        session: &mut Session,
        tokens: &[u32],
        n_out: usize,
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_pass(session, tokens)?;
        if tokens.len() > self.chunk || n_out > 1 || n_out > tokens.len() {
            return Err(error("invalid batch pass"));
        }
        let _serial = self
            .scratch
            .lock()
            .map_err(|_| error("scratch lock poisoned"))?;
        let batch = self
            .batch
            .as_ref()
            .ok_or_else(|| error("batch scratch unavailable"))?
            .lock()
            .map_err(|_| error("batch scratch lock poisoned"))?;
        let s = self.sess_mut(session)?;
        let p0 = s.pos;
        s.last = None;
        s.poisoned = true;
        let result = self
            .run_batch(s, tokens, n_out, &batch, tap)
            .and_then(|rows| {
                self.gpu.sync()?;
                Ok(rows)
            });
        match result {
            Ok(rows) => {
                s.pos = p0 + tokens.len();
                // Preserve max_verify=1: rollback(0) removes ONLY the final token.
                s.last = Some(s.pos - 1);
                s.poisoned = false;
                Ok(rows)
            }
            Err(err) => {
                let _ = self.gpu.sync();
                s.pos = p0;
                // No partial layer-major state is visible as a valid checkpoint.
                Err(err)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_mm(
        &self,
        name: &str,
        expert: usize,
        x: &DevBuf,
        xs: usize,
        y: &DevBuf,
        ys: usize,
        n: usize,
        stage: &DevBuf,
    ) -> Result<()> {
        let w = self.weight(name)?;
        if n == 0 || n > self.chunk || expert >= w.experts || xs < w.cols || ys < w.rows {
            return Err(error("invalid batched matrix extents"));
        }
        let extent = |stride: usize, width: usize| -> Result<usize> {
            product(&[n - 1, stride])?
                .checked_add(width)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| error("batch extent overflow"))
        };
        x.bounds(0, extent(xs, w.cols)?)?;
        y.bounds(0, extent(ys, w.rows)?)?;
        let kernel = self
            .kernels
            .mv_batch
            .ok_or_else(|| error("batch matrix kernel unavailable"))?;
        let tile = (TILE / w.row_bytes)
            .min((i32::MAX as usize) / w.cols)
            .max(1);
        for first in (0..w.rows).step_by(tile) {
            let rows = tile.min(w.rows - first);
            let offset = product(&[expert, w.rows])?
                .checked_add(first)
                .and_then(|v| v.checked_mul(w.row_bytes))
                .ok_or_else(|| error("packed offset overflow"))?;
            let (packed, _lease) = self.packed(w, offset, product(&[rows, w.row_bytes])?, stage)?;
            let out = y.view(first * 4, y.len - first * 4)?;
            let started = self.profile.start()?;
            // SAFETY: batch.h row/byte strides and full strided extents checked above;
            // immutable packed allocation/lease remains live through queued dispatch.
            check(unsafe {
                kernel(
                    self.gpu.raw(),
                    packed.ptr(),
                    w.tensor.ty.code() as i32,
                    rows as i32,
                    w.cols as i32,
                    w.row_bytes as u64,
                    x.fp(),
                    n as i32,
                    xs as u64,
                    out.fp(),
                    ys as u64,
                )
            })?;
            self.profile.record(
                if w.streamed() {
                    "expert_matmul_batch"
                } else {
                    "resident_matmul_batch"
                },
                started,
                rows * w.row_bytes,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_ffn(
        &self,
        l: usize,
        suffix: &str,
        expert: usize,
        x: &DevBuf,
        n: usize,
        width: usize,
        b: &BatchScratch,
    ) -> Result<()> {
        let g = &self.geom;
        let stride = g.ff.max(g.shared);
        let a = &b.all;
        self.batch_mm(
            &format!("blk.{l}.ffn_gate{suffix}.weight"),
            expert,
            x,
            g.width,
            &a.gate,
            stride,
            n,
            &a.stage,
        )?;
        self.batch_mm(
            &format!("blk.{l}.ffn_up{suffix}.weight"),
            expert,
            x,
            g.width,
            &a.up,
            stride,
            n,
            &a.stage,
        )?;
        for row in b.rows.iter().take(n) {
            // SAFETY: each row has at least width floats; exact scalar SwiGLU arithmetic.
            check(unsafe {
                (self.kernels.swiglu)(self.gpu.raw(), row.gate.fp(), row.up.fp(), width as i32)
            })?;
        }
        self.batch_mm(
            &format!("blk.{l}.ffn_down{suffix}.weight"),
            expert,
            &a.gate,
            stride,
            &a.proj,
            g.width,
            n,
            &a.stage,
        )
    }

    fn batch_moe(&self, l: usize, b: &BatchScratch, n: usize, tap: Tap) -> Result<()> {
        let g = &self.geom;
        let a = &b.all;
        let name = |s: &str| format!("blk.{l}.{s}");
        self.batch_mm(
            &name("ffn_gate_inp.weight"),
            0,
            &a.norm,
            g.width,
            &a.router,
            g.experts,
            n,
            &a.stage,
        )?;
        let logits = a.router.view(0, n * g.experts * 4)?.to_f32()?;
        let rows = logits
            .chunks_exact(g.experts)
            .map(|v| store::route(v, g.used))
            .collect::<Result<Vec<_>>>()?;
        let groups = group_routes(&rows);
        for row in &rows {
            self.route_profile.record(l, row)?;
        }
        self.prefetch_experts(l, groups.keys().copied())?;
        self.batch_mm(
            &name("ffn_gate_inp_shexp.weight"),
            0,
            &a.norm,
            g.width,
            &a.shared_gate,
            1,
            n,
            &a.stage,
        )?;
        let shared = a.shared_gate.view(0, n * 4)?.to_f32()?;
        if shared.iter().any(|v| !v.is_finite()) {
            return Err(error("nonfinite shared gate"));
        }
        b.upload_routes(&rows)?;
        for row in b.rows.iter().take(n) {
            tap(&format!("router_logits-{l}"), &row.router)?;
            tap(&format!("router_ids-{l}"), &row.route_ids)?;
            tap(&format!("router_weights-{l}"), &row.route_weights)?;
        }
        for (expert, entries) in groups {
            // One expert is staged once per matrix for all rows selecting it.
            for (j, &(row, _)) in entries.iter().enumerate() {
                b.gather
                    .copy_within(j * g.width * 4, &a.norm, row * g.width * 4, g.width * 4)?;
            }
            self.batch_ffn(l, "_exps", expert, &b.gather, entries.len(), g.ff, b)?;
            for (j, &(row, rank)) in entries.iter().enumerate() {
                b.slots.copy_within(
                    (row * g.used + rank) * g.width * 4,
                    &a.proj,
                    j * g.width * 4,
                    g.width * 4,
                )?;
            }
        }
        // Preserve ORIGINAL per-token rank accumulation order, not expert-ID order.
        for (i, row) in b.rows.iter().take(n).enumerate() {
            row.ffn_sum.fill(0)?;
            for (rank, &(_, weight)) in rows[i].iter().enumerate() {
                let value = b
                    .slots
                    .view((i * g.used + rank) * g.width * 4, g.width * 4)?;
                self.add(&row.ffn_sum, &value, g.width, weight)?;
            }
        }
        self.batch_ffn(l, "_shexp", 0, &a.norm, n, g.shared, b)?;
        for (i, row) in b.rows.iter().take(n).enumerate() {
            self.add(
                &row.ffn_sum,
                &row.proj,
                g.width,
                1. / (1. + (-shared[i]).exp()),
            )?;
            self.add(&row.x, &row.ffn_sum, g.width, 1.)?;
        }
        Ok(())
    }

    fn run_batch(
        &self,
        s: &Sess,
        tokens: &[u32],
        n_out: usize,
        b: &BatchScratch,
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        let g = &self.geom;
        let k = &self.kernels;
        let raw = self.gpu.raw();
        let n = tokens.len();
        let p0 = s.pos;
        let all = &b.all;
        for (i, &token) in tokens.iter().enumerate() {
            let a = &b.rows[i];
            let w = self.weight("token_embd.weight")?;
            let row = w
                .resident
                .as_ref()
                .ok_or_else(|| error("embedding is not resident"))?
                .view(token as usize * w.row_bytes, w.row_bytes)?;
            // SAFETY: validated token ID and packed embedding row; output is one model-width row.
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
        }
        for l in 0..g.layers {
            let layer_started = self.profile.start_layer()?;
            let name = |s: &str| format!("blk.{l}.{s}");
            for a in b.rows.iter().take(n) {
                self.norm(&a.x, &name("attn_norm.weight"), &a.norm)?;
                tap(&format!("attn_norm-{l}"), &a.norm)?;
            }
            let cache = &s.cache[l];
            if g.full(l) {
                self.batch_mm(
                    &name("attn_q.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.qf,
                    2 * g.heads * g.head,
                    n,
                    &all.stage,
                )?;
                self.batch_mm(
                    &name("attn_k.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.k,
                    g.kv_heads * g.head,
                    n,
                    &all.stage,
                )?;
                self.batch_mm(
                    &name("attn_v.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.v,
                    g.kv_heads * g.head,
                    n,
                    &all.stage,
                )?;
                for (i, a) in b.rows.iter().take(n).enumerate() {
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
                            (p0 + i) as i32,
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
                            (p0 + i) as i32,
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
                            (p0 + i) as i32,
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
                }
                self.batch_mm(
                    &name("attn_output.weight"),
                    0,
                    &all.attn,
                    g.inner().max(g.heads * g.head),
                    &all.proj,
                    g.width,
                    n,
                    &all.stage,
                )?;
            } else {
                self.batch_mm(
                    &name("attn_qkv.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.qkv,
                    g.channels(),
                    n,
                    &all.stage,
                )?;
                self.batch_mm(
                    &name("attn_gate.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.z,
                    g.inner(),
                    n,
                    &all.stage,
                )?;
                self.batch_mm(
                    &name("ssm_alpha.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.alpha,
                    g.v_heads,
                    n,
                    &all.stage,
                )?;
                self.batch_mm(
                    &name("ssm_beta.weight"),
                    0,
                    &all.norm,
                    g.width,
                    &all.beta,
                    g.v_heads,
                    n,
                    &all.stage,
                )?;
                for (i, a) in b.rows.iter().take(n).enumerate() {
                    if i + 1 == n {
                        let (before_state, before_conv) = cache
                            .before
                            .as_ref()
                            .ok_or_else(|| error("missing recurrent rollback snapshot"))?;
                        before_state.copy_within(0, &cache.a, 0, cache.a.len)?;
                        before_conv.copy_within(0, &cache.b, 0, cache.b.len)?;
                    }
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
                }
                self.batch_mm(
                    &name("ssm_out.weight"),
                    0,
                    &all.attn,
                    g.inner().max(g.heads * g.head),
                    &all.proj,
                    g.width,
                    n,
                    &all.stage,
                )?;
            }
            for a in b.rows.iter().take(n) {
                self.add(&a.x, &a.proj, g.width, 1.)?;
                tap(&format!("ffn_inp-{l}"), &a.x)?;
                self.norm(&a.x, &name("post_attention_norm.weight"), &a.norm)?;
                tap(&format!("ffn_norm-{l}"), &a.norm)?;
            }
            if g.experts == 0 {
                self.batch_ffn(l, "", 0, &all.norm, n, g.ff, b)?;
                for a in b.rows.iter().take(n) {
                    self.add(&a.x, &a.proj, g.width, 1.)?;
                }
            } else {
                self.batch_moe(l, b, n, tap)?;
            }
            for a in b.rows.iter().take(n) {
                tap(&format!("l_out-{l}"), &a.x)?;
            }
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
        let mut out = Vec::with_capacity(n_out);
        for a in b.rows.iter().take(n).skip(n - n_out) {
            self.norm(&a.x, "output_norm.weight", &a.norm)?;
            tap("result_norm", &a.norm)?;
            self.mv("output.weight", 0, &a.norm, &a.logits, a)?;
            tap("result_output", &a.logits)?;
            let logits = a.logits.to_f32()?;
            if logits.iter().any(|v| !v.is_finite()) {
                return Err(error("nonfinite batched logits"));
            }
            out.push(logits);
        }
        Ok(out)
    }
}

fn group_routes(rows: &[Vec<(usize, f32)>]) -> BTreeMap<usize, Vec<(usize, usize)>> {
    let mut groups: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for (row, selected) in rows.iter().enumerate() {
        for (rank, &(expert, _)) in selected.iter().enumerate() {
            groups.entry(expert).or_default().push((row, rank));
        }
    }
    groups
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn batch_views_are_disjoint_and_reservation_covers_owned_buffers() {
        let g = Geometry {
            arch: "qwen35moe",
            layers: 4,
            width: 8,
            ff: 4,
            heads: 2,
            kv_heads: 1,
            head: 4,
            rotary: 2,
            k_heads: 1,
            v_heads: 4,
            state: 4,
            conv: 4,
            interval: 4,
            vocab: 16,
            context: 128,
            experts: 4,
            used: 2,
            shared: 2,
            eps: 1e-6,
            theta: 10000.,
        };
        let gpu = Gpu::detached("batch layout host test");
        let stage = DevBuf::new(&gpu, TILE).unwrap();
        for n in [8, 32, 64] {
            let b = BatchScratch::new(&gpu, &g, n, &stage).unwrap();
            for (i, row) in b.rows.iter().enumerate() {
                let data: Vec<u8> = vec![i as f32; g.width]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                row.x.write(0, &data).unwrap();
                assert_eq!(row.stage.ptr(), stage.ptr());
                assert_eq!(row.attn.len, g.inner().max(g.heads * g.head) * 4);
            }
            let actual = b.all.x.to_f32().unwrap();
            for (i, row) in actual.chunks_exact(g.width).enumerate() {
                assert!(row.iter().all(|v| *v == i as f32));
            }
            // Full batches and short tails retain each tap's row/rank layout.
            for active in [n, 1, n - 1] {
                let routes: Vec<_> = (0..active)
                    .map(|i| vec![(i % 4, 0.75), ((i + 1) % 4, 0.25)])
                    .collect();
                b.upload_routes(&routes).unwrap();
                for (row, selected) in b.rows.iter().zip(&routes) {
                    assert_eq!(
                        row.route_ids.to_f32().unwrap(),
                        selected
                            .iter()
                            .map(|(id, _)| *id as f32)
                            .collect::<Vec<_>>()
                    );
                    assert_eq!(row.route_weights.to_f32().unwrap(), vec![0.75, 0.25]);
                }
            }
            let expected =
                Scratch::sizes(&g).iter().sum::<usize>() * n * 4 + b.gather.len + b.slots.len;
            assert_eq!(BatchScratch::bytes(&g, n).unwrap(), expected);
        }
    }
    #[test]
    fn batch_setting_is_bounded_and_serial_by_default() {
        assert_eq!(chunk_setting(None).unwrap(), 1);
        for n in ["1", "8", "32", "64"] {
            assert!(chunk_setting(Some(n)).is_ok());
        }
        for n in ["0", "2", "65", "-1", "x"] {
            assert!(chunk_setting(Some(n)).is_err());
        }
    }
    #[test]
    fn grouping_retains_original_row_and_rank() {
        let groups = group_routes(&[vec![(7, 0.7), (2, 0.3)], vec![(2, 0.6), (7, 0.4)]]);
        assert_eq!(groups[&2], vec![(0, 1), (1, 0)]);
        assert_eq!(groups[&7], vec![(0, 0), (1, 1)]);
        assert!(product(&[usize::MAX, 4]).is_err());
    }
}
