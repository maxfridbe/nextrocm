//! Bounded text-only DiT stage, ported from nextsycl 98124afb `dit.rs`.
//! No engine registration: editing, LoRA and the full resolution envelope still
//! need their original contracts and separate admission gates.
use crate::{
    auxiliary::Auxiliary,
    block::Block,
    ffi::{self, AttentionKernel, OutputType, PackedKernel},
    plan::{self, DitPlan, FFN, HEADS, HEAD_DIM as H, LATENT, LAYERS, WIDTH as D},
};
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_gguf::Gguf;
use std::{
    ffi::c_void,
    sync::{Arc, OnceLock},
};

// Compile-time private hooks: normal builds have no event/timer calls.
macro_rules! gpu_span {
    ($g:expr, $label:expr, $shape:expr, $body:expr) => {{
        #[cfg(feature = "operator-profile")]
        {
            crate::profiler::mark($g, $label, $shape, || $body)
        }
        #[cfg(not(feature = "operator-profile"))]
        {
            $body
        }
    }};
}
macro_rules! host_span {
    ($label:expr, $shape:expr, $body:expr) => {{
        #[cfg(feature = "operator-profile")]
        {
            let start = std::time::Instant::now();
            let result = $body;
            crate::profiler::record_host($label, $shape, start.elapsed().as_secs_f64())?;
            result
        }
        #[cfg(not(feature = "operator-profile"))]
        {
            $body
        }
    }};
}

const EPS: f32 = 1e-6;
const BF16_EXPANSION: u64 = 67_633_152;
const TRANSIENT_RESERVE: u64 = 8 * 1024 * 1024;

/// Aggregate preflight includes every resident block, expanded F32 auxiliaries,
/// prefix caches, both phases' peak activations, text projection temporaries,
/// modulation/host-transfer staging and explicit external residency/margin.
#[derive(Clone, Debug)]
pub struct MemoryPlan {
    pub resident: u64,
    pub tracked: u64,
    pub with_margin: u64,
    pub attention_scratch: u64,
    pub packed_scratch:u64,
    pub image_rows: usize,
    pub text_rows: usize,
}
impl MemoryPlan {
    pub fn new_with_attention(
        weights: &DitPlan,
        image_rows: usize,
        text_rows: usize,
        other_live: u64,
        margin: u64,
        attention: AttentionKernel,
    ) -> Result<Self> {
        attention.validate_image_geometry(image_rows, text_rows)?;
        let mut plan = Self::new(weights, image_rows, text_rows, other_live, margin)?;
        let scratch = attention.scratch_bytes(image_rows, text_rows)?;
        plan.tracked = plan
            .tracked
            .checked_add(scratch)
            .ok_or_else(|| Error("attention tracked overflow".into()))?;
        plan.with_margin = plan
            .with_margin
            .checked_add(scratch)
            .ok_or_else(|| Error("attention margin overflow".into()))?;
        if plan.with_margin >= 8_000_000_000 {
            return Err(Error("batched DiT aggregate reaches8GB".into()));
        }
        plan.attention_scratch = scratch;
        Ok(plan)
    }
    pub fn new(
        weights: &DitPlan,
        image_rows: usize,
        text_rows: usize,
        other_live: u64,
        margin: u64,
    ) -> Result<Self> {
        if ![256, 1024, 4096].contains(&image_rows)
            || text_rows == 0
            || text_rows > 512
            || image_rows + text_rows > 4608
            || margin < 512 * 1024 * 1024
        {
            return Err(Error(
                "bounded DiT stage requires 256/512/1024 pixel image, 1..512 text rows and >=512MiB margin"
                    .into(),
            ));
        }
        let resident = weights
            .packed_bytes
            .checked_add(BF16_EXPANSION)
            .ok_or_else(|| Error("resident overflow".into()))?;
        let tracked = resident
            .checked_add(plan::activation_bytes(image_rows as u64, text_rows as u64)?)
            .and_then(|v| v.checked_add(text_rows as u64 * D as u64 * 20))
            .and_then(|v| v.checked_add(image_rows as u64 * LATENT as u64 * 8))
            .and_then(|v| v.checked_add(TRANSIENT_RESERVE))
            .ok_or_else(|| Error("aggregate memory overflow".into()))?;
        let with_margin = tracked
            .checked_add(other_live)
            .and_then(|v| v.checked_add(margin))
            .ok_or_else(|| Error("memory margin overflow".into()))?;
        if with_margin >= 8_000_000_000 {
            return Err(Error("aggregate DiT device budget reaches 8GB".into()));
        }
        Ok(Self {
            resident,
            tracked,
            with_margin,
            attention_scratch: 0,
            packed_scratch:0,
            image_rows,
            text_rows,
        })
    }
}

struct Extra {
    to_half: unsafe extern "C" fn(*mut c_void, *const f32, i64, *mut c_void) -> i32,
    to_float: unsafe extern "C" fn(*mut c_void, *const c_void, i64, *mut f32) -> i32,
    gelu: unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i64, i32) -> i32,
}
fn extra() -> Result<&'static Extra> {
    static API: OnceLock<std::result::Result<Extra, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextrocm_engine_core::api().map_err(|e| e.0)?;
        macro_rules! bind {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("missing {}", $name));
                }
                // SAFETY: exact convert.h / speech.hip ABI; library remains loaded.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Extra {
            to_half: bind!("nrqi_f32_to_f16"),
            to_float: bind!("nrqi_f16_to_f32"),
            gelu: bind!("nsd_gelu"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

pub struct Denoiser {
    gpu: Arc<Gpu>,
    blocks: Vec<Block>,
    attention: AttentionKernel,
    img: Auxiliary,
    txt1: Auxiliary,
    txt2: Auxiliary,
    t1: Auxiliary,
    t2: Auxiliary,
    modulation: Auxiliary,
    norm_out: Auxiliary,
    projection: Auxiliary,
    txt_norm: DevBuf,
    plan: MemoryPlan,
    owner: Arc<()>,
}
pub struct Prefix {
    kv: Vec<DevBuf>,
    first_output: Option<Vec<f32>>,
    owner: Arc<()>,
    tokens: usize,
}
struct Work {
    h: DevBuf,
    qkv: DevBuf,
    att: DevBuf,
    o: DevBuf,
    gu: DevBuf,
    act: DevBuf,
    attention_scratch: Option<DevBuf>,
    packed_scratch:Option<DevBuf>,
}
impl Work {
    fn new(g: &Arc<Gpu>, n: usize, prefix: usize, scratch: u64, packed_scratch:u64) -> Result<Self> {
        Ok(Self {
            packed_scratch:if packed_scratch==0{None}else{Some(DevBuf::new(g,usize::try_from(packed_scratch).map_err(|_|Error("packed scratch usize".into()))?)?)},
            h: DevBuf::new(g, n * D * 2)?,
            qkv: DevBuf::new(g, (n + prefix) * 3 * D * 2)?,
            att: DevBuf::new(g, n * D * 2)?,
            o: DevBuf::new(g, n * D * 2)?,
            gu: DevBuf::new(g, n * 2 * FFN * 2)?,
            act: DevBuf::new(g, n * FFN * 2)?,
            attention_scratch: if scratch == 0 {
                None
            } else {
                Some(DevBuf::new(
                    g,
                    usize::try_from(scratch).map_err(|_| Error("scratch usize overflow".into()))?,
                )?)
            },
        })
    }
}
fn half_at(b: &DevBuf, n: usize) -> *mut c_void {
    b.ptr().cast::<u16>().wrapping_add(n).cast()
}
fn finite(v: &[f32]) -> Result<()> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(Error("nonfinite DiT values".into()))
    }
}

impl Denoiser {
    pub fn plan(&self) -> &MemoryPlan {
        &self.plan
    }

    /// No device weights are allocated before the complete aggregate preflight.
    pub fn load(
        file: &Gguf,
        gpu: &Arc<Gpu>,
        image_rows: usize,
        text_rows: usize,
        kernel: PackedKernel,
        other_live: u64,
        margin: u64,
    ) -> Result<Self> {
        Self::load_with_attention(
            file,
            gpu,
            image_rows,
            text_rows,
            kernel,
            other_live,
            margin,
            AttentionKernel::Baseline,
        )
    }

    pub fn load_with_attention(
        file: &Gguf,
        gpu: &Arc<Gpu>,
        image_rows: usize,
        text_rows: usize,
        kernel: PackedKernel,
        other_live: u64,
        margin: u64,
        attention: AttentionKernel,
    ) -> Result<Self> {
        let weights = DitPlan::validate(file)?;
        let mut plan = MemoryPlan::new_with_attention(
            &weights, image_rows, text_rows, other_live, margin, attention,
        )?;
        let packed_scratch=kernel.expanded_scratch(image_rows)?;
        plan.tracked=plan.tracked.checked_add(packed_scratch).ok_or_else(||Error("packed scratch budget overflow".into()))?;
        plan.with_margin=plan.with_margin.checked_add(packed_scratch).ok_or_else(||Error("packed scratch margin overflow".into()))?;
        if plan.with_margin>=8_000_000_000{return Err(Error("expanded aggregate budget reaches8GB".into()))}
        plan.packed_scratch=packed_scratch;
        attention.validate_image_geometry(image_rows, text_rows)?;
        ffi::api()?.require_kernel(kernel)?;
        ffi::api()?.require_attention(attention)?;
        extra()?;
        let (total, free) = gpu.memory()?;
        if free.ok_or_else(|| Error("GPU free memory unavailable".into()))? < plan.with_margin {
            return Err(Error(
                format!("insufficient free device memory for complete DiT stage: total={total} free={free:?} required_with_margin={} resident={} tracked={}", plan.with_margin, plan.resident, plan.tracked),
            ));
        }
        let mut remaining =
            usize::try_from(plan.resident).map_err(|_| Error("resident usize overflow".into()))?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let b = Block::load_with_kernel(file, gpu, i, remaining, kernel)?;
            remaining = remaining
                .checked_sub(b.packed_bytes)
                .ok_or_else(|| Error("block reservation underflow".into()))?;
            blocks.push(b);
        }
        let mut load = |name: &str| -> Result<Auxiliary> {
            let a = Auxiliary::load(file, gpu, name, remaining)?;
            remaining = remaining
                .checked_sub(a.bytes)
                .ok_or_else(|| Error("auxiliary reservation underflow".into()))?;
            Ok(a)
        };
        let img = load("img_in.weight")?;
        let txt1 = load("txt_in.in_layer.weight")?;
        let txt2 = load("txt_in.out_layer.weight")?;
        let t1 = load("time_text_embed.timestep_embedder.linear_1.weight")?;
        let t2 = load("time_text_embed.timestep_embedder.linear_2.weight")?;
        let modulation = load("modulation.1.weight")?;
        let norm_out = load("norm_out.linear.weight")?;
        let projection = load("proj_out.weight")?;
        if remaining != D * 4 {
            return Err(Error("aggregate resident accounting differs".into()));
        }
        let t = file
            .tensor("txt_in.text_norm.weight")
            .ok_or_else(|| Error("missing text norm".into()))?;
        let raw = file.read(t).map_err(|e| Error(e.0))?;
        let norm: Vec<_> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()) + 1.)
            .collect();
        finite(&norm)?;
        let txt_norm = DevBuf::from_f32(gpu, &norm)?;
        gpu.sync()?;
        Ok(Self {
            gpu: gpu.clone(),
            blocks,
            attention,
            img,
            txt1,
            txt2,
            t1,
            t2,
            modulation,
            norm_out,
            projection,
            txt_norm,
            plan,
            owner: Arc::new(()),
        })
    }

    /// Recorded scheduler timestep divided by 1000 in F32, as in the oracle.
    fn modulation(&self, sigma: f32) -> Result<Vec<DevBuf>> {
        if !sigma.is_finite() || !(0.0..=1.0).contains(&sigma) {
            return Err(Error("invalid recorded timestep".into()));
        }
        let g = &self.gpu;
        let mut emb = vec![0f32; 512];
        host_span!("modulation.sinusoid_host", &[2, 256], {
            for (r, t) in [sigma, 0.].iter().enumerate() {
                for i in 0..128 {
                    let a = 1000. * f64::from(*t) * (-10000f64.ln() * i as f64 / 128.).exp();
                    emb[r * 256 + i] = a.cos() as f32;
                    emb[r * 256 + 128 + i] = a.sin() as f32;
                }
            }
        });
        let e = host_span!(
            "modulation.embed_upload",
            &[2, 256],
            DevBuf::from_f32(g, &emb)
        )?;
        let h = DevBuf::f32(g, 2 * D)?;
        gpu_span!(
            g,
            "aux.timestep1",
            &[2, 256, D, 12],
            self.t1.apply(g, &e, 2, &h)
        )?;
        let silu = |b: &DevBuf| -> Result<DevBuf> {
            let mut v = host_span!("modulation.silu_download", &[2, D], b.to_f32())?;
            host_span!("modulation.silu_host", &[2, D], {
                for x in &mut v {
                    *x /= 1. + (-*x).exp();
                }
                finite(&v)
            })?;
            host_span!("modulation.silu_upload", &[2, D], DevBuf::from_f32(g, &v))
        };
        let sh = silu(&h)?;
        let t = DevBuf::f32(g, 2 * D)?;
        gpu_span!(
            g,
            "aux.timestep2",
            &[2, D, D, 12],
            self.t2.apply(g, &sh, 2, &t)
        )?;
        let st = silu(&t)?;
        let m = DevBuf::f32(g, 2 * 4 * D)?;
        gpu_span!(
            g,
            "aux.modulation",
            &[2, D, 4 * D, 12],
            self.modulation.apply(g, &st, 2, &m)
        )?;
        let o = DevBuf::f32(g, 2 * D)?;
        gpu_span!(
            g,
            "aux.norm_scale",
            &[2, D, D, 12],
            self.norm_out.apply(g, &st, 2, &o)
        )?;
        let mv = host_span!("modulation.split_download", &[2, 4 * D], m.to_f32())?;
        finite(&mv)?;
        let mut md = Vec::with_capacity(5);
        for part in 0..4 {
            let v = host_span!("modulation.split_tanh_host", &[2, D], {
                let mut v = Vec::with_capacity(2 * D);
                for row in 0..2 {
                    v.extend(
                        mv[(row * 4 + part) * D..(row * 4 + part + 1) * D]
                            .iter()
                            .map(|&x| if part % 2 == 1 { x.tanh() } else { x }),
                    );
                }
                v
            });
            md.push(host_span!(
                "modulation.part_upload",
                &[2, D],
                DevBuf::from_f32(g, &v)
            )?);
        }
        md.push(o);
        Ok(md)
    }

    /// One block on text (causal) or target image (prefix plus all target keys).
    /// Cache is copied before subsequent layers overwrite the workspace.
    fn block(
        &self,
        b: &Block,
        x: &DevBuf,
        n: usize,
        cs: &DevBuf,
        md: &[DevBuf],
        row: usize,
        w: &Work,
        pre: Option<&DevBuf>,
    ) -> Result<()> {
        let a = ffi::api()?;
        let g = &self.gpu;
        let p = pre.map_or(0, |b| b.len / (3 * D * 2));
        let norm = |part: usize| -> Result<()> {
            // SAFETY: private callers preflight n rows and allocate all complete
            // workspace spans; row selects one of two WIDTH-element scales.
            ffi::check(
                unsafe {
                    (a.ln_mod)(
                        g.raw(),
                        x.fp(),
                        n as i64,
                        D as i64,
                        EPS,
                        md[part].fp().wrapping_add(row * D),
                        w.h.ptr(),
                    )
                },
                "DiT norm",
            )
        };
        let gate = |part: usize| -> Result<()> {
            // SAFETY: owned disjoint residual/output, broadcast scale row.
            ffi::check(
                unsafe {
                    (a.gate_add)(
                        g.raw(),
                        x.fp(),
                        w.o.ptr(),
                        n as i64,
                        D as i64,
                        md[part].fp().wrapping_add(row * D),
                    )
                },
                "DiT residual",
            )
        };
        gpu_span!(g, "block.norm1", &[n, D], norm(0))?;
        // Compute target qkv into a temporary view of the suffix. The view's
        // lifetime is strictly inside the owning workspace lifetime.
        let qkv = w.qkv.view(p * 3 * D * 2, n * 3 * D * 2)?;
        gpu_span!(
            g,
            "block.qkv",
            &[n, D, 3 * D, 12, 12, 14],
            a.packed_matrix_using_workspace(b.kernel, g, &b.qkv, &w.h, n, &qkv, OutputType::Half,w.packed_scratch.as_ref())
        )?;
        if let Some(pre) = pre {
            gpu_span!(
                g,
                "block.prefix_copy",
                &[p, 3 * D],
                w.qkv.copy_within(0, pre, 0, pre.len)
            )?;
        }
        for (off, norm) in [(0, &b.norm_q), (D, &b.norm_k)] {
            // SAFETY: strided heads cover only the target suffix, cs n*H floats.
            gpu_span!(
                g,
                if off == 0 {
                    "block.rms_q"
                } else {
                    "block.rms_k"
                },
                &[n, HEADS, H, 3 * D],
                ffi::check(
                    unsafe {
                        (a.rms_rope)(
                            g.raw(),
                            half_at(&w.qkv, p * 3 * D + off),
                            n as i64,
                            HEADS as i64,
                            H as i64,
                            (3 * D) as i64,
                            norm.fp(),
                            EPS,
                            cs.fp(),
                            H as i32,
                        )
                    },
                    "DiT RMS rotary",
                )
            )?;
        }
        // SAFETY: prefix is immutable cached qkv; bounded n+p<=4608; values
        // above4096 require the explicit private1024 kernel, checked before load.
        // All output
        // attention rows occupy exactly n*D half values.
        if pre.is_some() {
            if self
                .attention
                .uses_batched_1024(n, p + n, HEADS, H, 3 * D, 3 * D)
            {
                let function = a
                    .image_attention_batched
                    .ok_or_else(|| Error("requested batched attention ABI missing".into()))?;
                let scratch = w
                    .attention_scratch
                    .as_ref()
                    .ok_or_else(|| Error("batched attention workspace missing".into()))?;
                if scratch.len as u64 != self.plan.attention_scratch {
                    return Err(Error(
                        "batched attention workspace length differs from plan".into(),
                    ));
                }
                // SAFETY: owned disjoint scratch/output and complete same-context half Q/K/V views;
                // required B256 score/denominator bytes preflighted and allocated once per velocity.
                gpu_span!(
                    g,
                    "block.attention",
                    &[n, p + n, HEADS, H],
                    ffi::check(
                        unsafe {
                            function(
                                g.raw(),
                                half_at(&w.qkv, p * 3 * D),
                                half_at(&w.qkv, D),
                                half_at(&w.qkv, 2 * D),
                                n as i64,
                                (p + n) as i64,
                                HEADS as i64,
                                H as i64,
                                (3 * D) as i64,
                                (3 * D) as i64,
                                w.att.ptr(),
                                scratch.ptr(),
                                scratch.len as i64,
                            )
                        },
                        "DiT batched image attention"
                    )
                )?;
            } else if self.attention.uses_tiled(n, p + n, HEADS, H, 3 * D, 3 * D)
                || self
                    .attention
                    .uses_tiled_1024(n, p + n, HEADS, H, 3 * D, 3 * D)
            {
                let function = (if self.attention == AttentionKernel::Tiled1024 {
                    a.image_attention_1024
                } else {
                    a.image_attention_tiled
                })
                .ok_or_else(|| Error("requested private attention ABI missing".into()))?;
                // SAFETY: exact admitted operator geometry, same owned QKV
                // prefix/suffix layout and disjoint contiguous half output.
                gpu_span!(
                    g,
                    "block.attention",
                    &[n, p + n, HEADS, H],
                    ffi::check(
                        unsafe {
                            function(
                                g.raw(),
                                half_at(&w.qkv, p * 3 * D),
                                half_at(&w.qkv, D),
                                half_at(&w.qkv, 2 * D),
                                n as i64,
                                (p + n) as i64,
                                HEADS as i64,
                                H as i64,
                                (3 * D) as i64,
                                (3 * D) as i64,
                                w.att.ptr(),
                            )
                        },
                        "DiT tiled image attention"
                    )
                )?;
            } else {
                gpu_span!(
                    g,
                    "block.attention",
                    &[n, p + n, HEADS, H],
                    ffi::check(
                        unsafe {
                            (a.attention_qk)(
                                g.raw(),
                                half_at(&w.qkv, p * 3 * D),
                                n as i64,
                                (3 * D) as i64,
                                half_at(&w.qkv, D),
                                half_at(&w.qkv, 2 * D),
                                (p + n) as i64,
                                (3 * D) as i64,
                                1,
                                HEADS as i64,
                                H as i64,
                                w.att.ptr(),
                                1,
                            )
                        },
                        "DiT image attention",
                    )
                )?;
            }
        } else {
            gpu_span!(
                g,
                "prefix.attention",
                &[n, HEADS, H],
                ffi::check(
                    unsafe {
                        (a.attention_causal_rows)(
                            g.raw(),
                            w.qkv.ptr(),
                            half_at(&w.qkv, D),
                            half_at(&w.qkv, 2 * D),
                            1,
                            0,
                            n as i64,
                            HEADS as i64,
                            HEADS as i64,
                            H as i64,
                            (3 * D) as i64,
                            (3 * D) as i64,
                            w.att.ptr(),
                        )
                    },
                    "DiT prefix attention",
                )
            )?;
        }
        gpu_span!(
            g,
            "block.out",
            &[n, D, D, 12],
            a.packed_matrix_using_workspace(b.kernel, g, &b.out, &w.att, n, &w.o, OutputType::Half,w.packed_scratch.as_ref())
        )?;
        gpu_span!(g, "block.gate1", &[n, D], gate(1))?;
        gpu_span!(g, "block.norm2", &[n, D], norm(2))?;
        gpu_span!(
            g,
            "block.gate_up",
            &[n, D, 2 * FFN, 12, 12],
            a.packed_matrix_using_workspace(b.kernel, g, &b.gate_up, &w.h, n, &w.gu, OutputType::Half,w.packed_scratch.as_ref())
        )?;
        // SAFETY: n rows of concatenated gate/up and n rows of FFN halves.
        gpu_span!(
            g,
            "block.swiglu",
            &[n, FFN],
            ffi::check(
                unsafe { (a.swiglu)(g.raw(), w.gu.ptr(), 1, n as i64, FFN as i64, w.act.ptr(), 1) },
                "DiT SwiGLU",
            )
        )?;
        gpu_span!(
            g,
            "block.down",
            &[n, FFN, D, 12],
            a.packed_matrix_using_workspace(b.kernel, g, &b.down, &w.act, n, &w.o, OutputType::Half,w.packed_scratch.as_ref())
        )?;
        gpu_span!(g, "block.gate2", &[n, D], gate(3))
    }

    pub fn prefix(&self, embeds: &[f32], capture_first: bool) -> Result<Prefix> {
        let p = self.plan.text_rows;
        if embeds.len() != p * D {
            return Err(Error("text embedding shape differs from preflight".into()));
        }
        finite(embeds)?;
        let g = &self.gpu;
        let a = ffi::api()?;
        let ex = extra()?;
        let e = DevBuf::from_f32(g, embeds)?;
        let norm = DevBuf::f32(g, p * D)?;
        // SAFETY: complete p*D F32 buffers and D norm weights.
        ffi::check(
            unsafe {
                (a.text_rms)(
                    g.raw(),
                    e.fp(),
                    p as i64,
                    D as i64,
                    self.txt_norm.fp(),
                    EPS,
                    norm.fp(),
                )
            },
            "text RMS",
        )?;
        let h = DevBuf::f32(g, p * D)?;
        self.txt1.apply(g, &norm, p, &h)?;
        let hh = DevBuf::new(g, p * D * 2)?;
        // SAFETY: exact same p*D count, disjoint conversion buffers; GELU in place.
        ffi::check(
            unsafe { (ex.to_half)(g.raw(), h.fp(), (p * D) as i64, hh.ptr()) },
            "text to half",
        )?;
        ffi::check(
            unsafe { (ex.gelu)(g.raw(), hh.ptr(), 1, (p * D) as i64, 1) },
            "text GELU tanh",
        )?;
        ffi::check(
            unsafe { (ex.to_float)(g.raw(), hh.ptr(), (p * D) as i64, h.fp()) },
            "text to float",
        )?;
        let x = DevBuf::f32(g, p * D)?;
        self.txt2.apply(g, &h, p, &x)?;
        drop((e, norm, h, hh));
        let md = self.modulation(0.)?;
        let cs = DevBuf::from_f32(g, &rope(&(0..p).map(|i| [i as i64; 3]).collect::<Vec<_>>()))?;
        let w = Work::new(g, p, 0, 0, 0)?;
        let mut kv = Vec::with_capacity(LAYERS);
        let mut first_output = None;
        for (i, b) in self.blocks.iter().enumerate() {
            self.block(b, &x, p, &cs, &md, 1, &w, None)?;
            let keep = DevBuf::new(g, p * 3 * D * 2)?;
            keep.copy_within(0, &w.qkv, 0, keep.len)?;
            kv.push(keep);
            if i == 0 && capture_first {
                let v = x.to_f32()?;
                finite(&v)?;
                first_output = Some(v);
            }
        }
        g.sync()?;
        Ok(Prefix {
            kv,
            first_output,
            tokens: p,
            owner: self.owner.clone(),
        })
    }

    /// Returns velocity; optional callback captures the full joint first-block
    /// output so it can be compared to the independent extraction-mode oracle.
    pub fn velocity(
        &self,
        pre: &Prefix,
        latents: &DevBuf,
        hw: (usize, usize),
        recorded_timestep: f32,
        mut block0: Option<&mut dyn FnMut(&[f32]) -> Result<()>>,
    ) -> Result<DevBuf> {
        let n =
            hw.0.checked_mul(hw.1)
                .ok_or_else(|| Error("image geometry overflow".into()))?;
        if n != self.plan.image_rows
            || hw.0 != hw.1
            || latents.len != n * LATENT * 4
            || !Arc::ptr_eq(latents.gpu(), &self.gpu)
            || !Arc::ptr_eq(&pre.owner, &self.owner)
            || pre.tokens != self.plan.text_rows
            || pre.kv.len() != LAYERS
        {
            return Err(Error("DiT request differs from preflight".into()));
        }
        let g = &self.gpu;
        let md = self.modulation(recorded_timestep / 1000.)?;
        let x = DevBuf::f32(g, n * D)?;
        gpu_span!(
            g,
            "aux.img_in_f32",
            &[n, LATENT, D],
            self.img.apply(g, latents, n, &x)
        )?;
        let idx: Vec<_> = (0..n)
            .map(|i| {
                [
                    pre.tokens as i64,
                    (i / hw.1) as i64 - (hw.0 - hw.0 / 2) as i64,
                    (i % hw.1) as i64 - (hw.1 - hw.1 / 2) as i64,
                ]
            })
            .collect();
        let rotary = host_span!("velocity.rope_host", &[n, H], rope(&idx));
        let cs = host_span!(
            "velocity.rope_upload",
            &[n, H],
            DevBuf::from_f32(g, &rotary)
        )?;
        let w = host_span!(
            "velocity.workspace_allocate",
            &[n, pre.tokens],
            Work::new(g, n, pre.tokens, self.plan.attention_scratch,self.plan.packed_scratch)
        )?;
        for (i, b) in self.blocks.iter().enumerate() {
            self.block(b, &x, n, &cs, &md, 0, &w, Some(&pre.kv[i]))?;
            if i == 0 {
                if let Some(hook) = block0.as_mut() {
                    let mut all = pre
                        .first_output
                        .as_ref()
                        .ok_or_else(|| Error("prefix was not captured".into()))?
                        .clone();
                    all.extend(x.to_f32()?);
                    finite(&all)?;
                    hook(&all)?;
                }
            }
        }
        let xh = DevBuf::new(g, n * D * 2)?;
        let xn = DevBuf::f32(g, n * D)?;
        // SAFETY: n complete rows, one scale row and disjoint cast destination.
        gpu_span!(
            g,
            "output.norm",
            &[n, D],
            ffi::check(
                unsafe {
                    (ffi::api()?.ln_mod)(
                        g.raw(),
                        x.fp(),
                        n as i64,
                        D as i64,
                        EPS,
                        md[4].fp(),
                        xh.ptr(),
                    )
                },
                "output norm",
            )
        )?;
        gpu_span!(
            g,
            "output.to_float",
            &[n, D],
            ffi::check(
                unsafe { (extra()?.to_float)(g.raw(), xh.ptr(), (n * D) as i64, xn.fp()) },
                "output to float",
            )
        )?;
        let out = DevBuf::f32(g, n * LATENT)?;
        gpu_span!(
            g,
            "aux.proj_out",
            &[n, D, LATENT, 12],
            self.projection.apply(g, &xn, n, &out)
        )?;
        g.sync()?;
        host_span!(
            "velocity.workspace_free",
            &[n, pre.tokens],
            drop((w, xh, xn, x, cs, md))
        );
        Ok(out)
    }
}

fn rope(idx: &[[i64; 3]]) -> Vec<f32> {
    let mut cs = Vec::with_capacity(idx.len() * H);
    for p in idx {
        for (axis, dim) in [16, 56, 56].iter().enumerate() {
            for i in 0..dim / 2 {
                let inv = 1.0 / 10000f64.powf(2. * i as f64 / f64::from(*dim));
                let a = p[axis] as f64 * inv;
                cs.push(a.cos() as f32);
                cs.push(a.sin() as f32);
            }
        }
    }
    cs
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aggregate_cap_and_geometry() {
        let w = DitPlan {
            packed_bytes: 4_189_323_264,
            block_bytes: 0,
            auxiliary_bytes: 0,
        };
        let p = MemoryPlan::new(&w, 256, 18, 0, 1024 * 1024 * 1024).unwrap();
        assert_eq!(p.resident, 4_256_956_416);
        assert!(p.with_margin < 8_000_000_000);
        assert!(MemoryPlan::new(&w, 1024, 512, 3_000_000_000, 1024 * 1024 * 1024).is_err());
        let large = MemoryPlan::new(&w, 4096, 18, 0, 1024 * 1024 * 1024).unwrap();
        assert_eq!(large.tracked, 4_957_765_632);
        assert_eq!(large.with_margin, 6_031_507_456);
        assert!(MemoryPlan::new(&w, 4096, 513, 0, 1024 * 1024 * 1024).is_err());
        assert!(MemoryPlan::new(&w, 4096, 18, 2_000_000_000, 1024 * 1024 * 1024).is_err());
        assert!(MemoryPlan::new(&w, 256, 18, 0, 0).is_err());
    }
    #[test]
    fn batched_scratch_is_in_aggregate_preflight() {
        let w = DitPlan {
            packed_bytes: 4_189_323_264,
            block_bytes: 0,
            auxiliary_bytes: 0,
        };
        let p = MemoryPlan::new_with_attention(
            &w,
            4096,
            18,
            0,
            1024 * 1024 * 1024,
            AttentionKernel::Batched1024,
        )
        .unwrap();
        assert_eq!(p.attention_scratch, 134_840_320);
        assert_eq!(p.tracked, 5_092_605_952);
        assert_eq!(p.with_margin, 6_166_347_776);
        assert!(MemoryPlan::new_with_attention(
            &w,
            4096,
            18,
            1_900_000_000,
            1024 * 1024 * 1024,
            AttentionKernel::Batched1024
        )
        .is_err());
        assert!(MemoryPlan::new_with_attention(
            &w,
            4096,
            18,
            0,
            1024 * 1024 * 1024,
            AttentionKernel::Baseline
        )
        .is_err());
    }
    #[test]
    fn rotary_positions_original_axes() {
        let v = rope(&[[0, 0, 0], [1, -8, 7]]);
        assert_eq!(v.len(), 256);
        assert!(v[..128].chunks_exact(2).all(|x| x == [1., 0.]));
        assert_eq!(v[128], 1f64.cos() as f32);
        assert_eq!(v[144], (-8f64).cos() as f32);
        assert_eq!(v[200], 7f64.cos() as f32);
    }
}
