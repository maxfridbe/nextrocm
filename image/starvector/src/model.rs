//! StarVector-1B's three parts on one GPU, from the checkpoint's two safetensors shards: the image encoder (CLIP's
//! ViT-L/14 at 224: the 14 x 14 patches as one product, the class token, learned positions, ln_pre, 23 pre-norm blocks
//! of attention and a QuickGELU MLP, ln_vision), the adapter (Linear, Swish, Linear, then BatchNorm1d over the 257
//! positions with its running statistics) and StarCoder-1B (GPTBigCode: learned positions, 24 pre-norm blocks of
//! multi-query attention - 16 query heads over one key / value head of 128 - and a tanh-GELU MLP, ln_f, the output
//! tied to the token embedding).
//!
//! Weights half on the card (StarCoder's float32 halved as it loads, as the platform loads it in float16); the
//! residual streams float32, the norms writing half for the products, which accumulate in float32. The shared
//! kernels (nsd) do the products, norms, attention and GELU; the upscaler's (kernels/image/upscale) the sigmoid and the
//! products that make QuickGELU and Swish. The adapter's BatchNorm (one affine a position, once a picture) is done on
//! the host. Each beam has its own key / value cache; a reorder copies the rows a beam takes over.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Arc;

use crate::kernels::{none, Dt, Nsd};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::safetensors::{f16_to_f32, SafeTensors, StTensor};
use nextrocm_image::{Error, Result};

use crate::pre::SIZE;

type P = *const c_void;
macro_rules! profile_model {
    ($gpu:expr, $label:expr, $body:expr) => {{
        #[cfg(feature = "operator-profile")]
        {
            crate::profile::measure(&$gpu.clone(), $label, || $body)
        }
        #[cfg(not(feature = "operator-profile"))]
        {
            $body
        }
    }};
}
type M = *mut c_void;

pub const PATCH: usize = 14;
pub const GRID: usize = SIZE / PATCH;
/// the image's rows: the class token and 16 x 16 patches
pub const QUERY: usize = GRID * GRID + 1;
const V_D: usize = 1024;
const V_HEADS: usize = 16;
const V_LAYERS: usize = 23;
const V_FF: usize = 4096;
pub const D: usize = 2048;
const HEADS: usize = 16;
const HD: usize = 128;
const LAYERS: usize = 24;
const FF: usize = 8192;
pub const POSITIONS: usize = 8192;
const EPS: f32 = 1e-5;

// Opt-in control of the earliest measured precision loss. All subsequent
// vision/adapter/language operations retain their existing precision.
#[cfg(feature = "patch-f32")]
type PatchValue = f32;
#[cfg(not(feature = "patch-f32"))]
type PatchValue = u16;
const PATCH_DT: Dt = if cfg!(feature = "patch-f32") {
    Dt::F32
} else {
    Dt::F16
};

const VISION_DT: Dt = if cfg!(feature = "vision-f32") {
    Dt::F32
} else {
    Dt::F16
};
const VISION_BYTES: usize = if cfg!(feature = "vision-f32") { 4 } else { 2 };

/// float32 to IEEE half (round to nearest even)
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 255 {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 31 {
        return s | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = 1u32 << (shift - 1);
        let r = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let r = if rem > half || (rem == half && r & 1 == 1) {
            r + 1
        } else {
            r
        };
        return s | r as u16;
    }
    let r = m >> 13;
    let rem = m & 0x1fff;
    let mut v = ((e2 as u32) << 10) | r;
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        v += 1;
    }
    s | v as u16
}

/// A pointer `bytes` into a buffer (may point before it for a kernel that indexes rows from an offset)
fn at(b: &DevBuf, bytes: isize) -> M {
    b.ptr().cast::<u8>().wrapping_offset(bytes).cast()
}

/// The checkpoint's shards, a tensor looked up in each
pub struct Shards(Vec<SafeTensors>);

impl Shards {
    pub fn open(paths: &[std::path::PathBuf]) -> Result<Shards> {
        paths
            .iter()
            .map(|p| SafeTensors::open(p).map_err(|e| Error(e.0)))
            .collect::<Result<Vec<_>>>()
            .map(Shards)
    }
    fn find(&self, name: &str) -> Result<(&SafeTensors, &StTensor)> {
        self.0
            .iter()
            .find_map(|s| s.tensor(name).map(|t| (s, t)))
            .ok_or_else(|| Error(format!("the checkpoint has no {name}")))
    }
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (s, _) = self.find(name)?;
        s.f32(name).map_err(|e| Error(e.0))
    }
    pub fn shape(&self, name: &str) -> Result<Vec<u64>> {
        Ok(self.find(name)?.1.shape.clone())
    }

    /// Validate the fixed 1B geometry before any device allocation or indexing.
    fn validate(&self) -> Result<()> {
        let mut names = std::collections::HashSet::new();
        for shard in &self.0 {
            for t in &shard.tensors {
                if !names.insert(t.name.as_str()) {
                    return Err(Error(format!("duplicate tensor {}", t.name)));
                }
            }
        }
        let tensor = |name: &str, dims: &[u64]| -> Result<()> {
            let (_, t) = self.find(name)?;
            let width = match t.dtype.as_str() {
                "F16" => 2,
                "F32" => 4,
                _ => return Err(Error(format!("{name}: unsupported {}", t.dtype))),
            };
            if t.shape != dims || t.bytes != dims.iter().product::<u64>() * width {
                return Err(Error(format!(
                    "{name}: expected {dims:?}, got {:?}/{} bytes",
                    t.shape, t.bytes
                )));
            }
            Ok(())
        };
        let norm = |name: &str, dim: u64| -> Result<()> {
            tensor(&format!("{name}.weight"), &[dim])?;
            tensor(&format!("{name}.bias"), &[dim])
        };
        let linear = |name: &str, n: u64, k: u64| -> Result<()> {
            tensor(&format!("{name}.weight"), &[n, k])?;
            tensor(&format!("{name}.bias"), &[n])
        };
        let e = "model.image_encoder.visual_encoder";
        tensor(&format!("{e}.conv1.weight"), &[1024, 3, 14, 14])?;
        tensor(&format!("{e}.class_embedding"), &[1024])?;
        tensor(&format!("{e}.positional_embedding"), &[257, 1024])?;
        norm(&format!("{e}.ln_pre"), 1024)?;
        for i in 0..V_LAYERS {
            let p = format!("{e}.transformer.resblocks.{i}");
            norm(&format!("{p}.ln_1"), 1024)?;
            norm(&format!("{p}.ln_2"), 1024)?;
            tensor(&format!("{p}.attn.in_proj_weight"), &[3072, 1024])?;
            tensor(&format!("{p}.attn.in_proj_bias"), &[3072])?;
            linear(&format!("{p}.attn.out_proj"), 1024, 1024)?;
            linear(&format!("{p}.mlp.c_fc"), 4096, 1024)?;
            linear(&format!("{p}.mlp.c_proj"), 1024, 4096)?;
        }
        norm("model.image_encoder.ln_vision", 1024)?;
        linear("model.image_projection.c_fc", 2048, 1024)?;
        linear("model.image_projection.c_proj", 2048, 2048)?;
        for part in ["weight", "bias", "running_mean", "running_var"] {
            tensor(&format!("model.image_projection.norm.{part}"), &[257])?;
        }
        let t = "model.svg_transformer.transformer.transformer";
        for i in 0..LAYERS {
            let p = format!("{t}.h.{i}");
            norm(&format!("{p}.ln_1"), 2048)?;
            norm(&format!("{p}.ln_2"), 2048)?;
            linear(&format!("{p}.attn.c_attn"), 2304, 2048)?;
            linear(&format!("{p}.attn.c_proj"), 2048, 2048)?;
            linear(&format!("{p}.mlp.c_fc"), 8192, 2048)?;
            linear(&format!("{p}.mlp.c_proj"), 2048, 8192)?;
        }
        norm(&format!("{t}.ln_f"), 2048)?;
        tensor(&format!("{t}.wte.weight"), &[49156, 2048])?;
        tensor(&format!("{t}.wpe.weight"), &[8192, 2048])
    }
}

struct Lin {
    w: DevBuf,
    b: Option<DevBuf>,
    n: usize,
    k: usize,
}

struct Ln {
    w: DevBuf,
    b: DevBuf,
}

struct VBlock {
    ln1: Ln,
    qkv: Lin,
    o: Lin,
    ln2: Ln,
    fc: Lin,
    proj: Lin,
}

struct Block {
    ln1: Ln,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: Ln,
    fc: Lin,
    proj: Lin,
}

struct Loader<'a> {
    st: &'a Shards,
    gpu: &'a Arc<Gpu>,
}

impl Loader<'_> {
    fn half_of(&self, v: &[f32]) -> Result<DevBuf> {
        let bytes: Vec<u8> = v
            .iter()
            .flat_map(|x| f32_to_f16(*x).to_le_bytes())
            .collect();
        let b = DevBuf::new(self.gpu, bytes.len())?;
        b.write(0, &bytes)?;
        Ok(b)
    }
    fn float(&self, name: &str) -> Result<DevBuf> {
        DevBuf::from_f32(self.gpu, &self.st.f32(name)?)
    }
    fn lin(&self, p: &str) -> Result<Lin> {
        self.lin_storage(p, false)
    }
    fn lin_storage(&self, p: &str, float: bool) -> Result<Lin> {
        let shape = self.st.shape(&format!("{p}.weight"))?;
        let (n, k) = (
            shape[0] as usize,
            shape[1..].iter().product::<u64>() as usize,
        );
        Ok(Lin {
            w: if float {
                self.float(&format!("{p}.weight"))?
            } else {
                self.half_of(&self.st.f32(&format!("{p}.weight"))?)?
            },
            b: self
                .st
                .find(&format!("{p}.bias"))
                .ok()
                .map(|_| self.float(&format!("{p}.bias")))
                .transpose()?,
            n,
            k,
        })
    }
    /// rows [r0, r1) of a [n, k] weight (and its bias): a slice of a fused product
    fn lin_rows(&self, p: &str, r0: usize, r1: usize) -> Result<Lin> {
        let shape = self.st.shape(&format!("{p}.weight"))?;
        let k = shape[1] as usize;
        let w = self.st.f32(&format!("{p}.weight"))?;
        let b = self.st.f32(&format!("{p}.bias"))?;
        Ok(Lin {
            w: self.half_of(&w[r0 * k..r1 * k])?,
            b: Some(DevBuf::from_f32(self.gpu, &b[r0..r1])?),
            n: r1 - r0,
            k,
        })
    }
    fn ln(&self, p: &str) -> Result<Ln> {
        Ok(Ln {
            w: self.float(&format!("{p}.weight"))?,
            b: self.float(&format!("{p}.bias"))?,
        })
    }
}

/// One beam's key / value cache: a buffer a layer, [POSITIONS, 128] half each
struct Cache {
    k: Vec<DevBuf>,
    v: Vec<DevBuf>,
}

pub struct Model {
    pub nsd: Nsd,
    pub gpu: Arc<Gpu>,
    up: &'static crate::ffi::Api,
    // the encoder
    patch: Lin,
    class: Vec<f32>,
    vpos: DevBuf,
    ln_pre: Ln,
    vis: Vec<VBlock>,
    ln_vision: Ln,
    // the adapter
    a_fc: Lin,
    a_proj: Lin,
    bn: [Vec<f32>; 4],
    // the language model
    blocks: Vec<Block>,
    ln_f: Ln,
    wte: DevBuf,
    wte_host: Vec<u16>,
    wpe_host: Vec<f32>,
    pub vocab: usize,
    caches: Vec<Cache>,
    // work
    px: DevBuf,
    vx: DevBuf,
    vx2: DevBuf,
    vh: DevBuf,
    vqkv: DevBuf,
    vatt: DevBuf,
    vy: DevBuf,
    vff: DevBuf,
    #[cfg(not(feature = "vision-f32"))]
    vtmp: DevBuf,
    enc_f: DevBuf,
    enc_h: DevBuf,
    aff: DevBuf,
    atmp: DevBuf,
    aout: DevBuf,
    x: DevBuf,
    h: DevBuf,
    q: DevBuf,
    kb: DevBuf,
    vb: DevBuf,
    att: DevBuf,
    y: DevBuf,
    ff: DevBuf,
    logits: DevBuf,
    /// rows the work buffers hold (the prefix: the picture's 257 and the prompt)
    rows: usize,
}

impl Model {
    pub fn load(
        files: &[std::path::PathBuf],
        gpu: &Arc<Gpu>,
        max_rows: usize,
        ffn_small_m: bool,
        log: &mut dyn FnMut(String),
    ) -> Result<Model> {
        // Resolve explicit opt-in ABI before allocating/loading model weights.
        let nsd = Nsd::new(gpu, ffn_small_m)?;
        if ffn_small_m {
            log("starvector: private small-M FFN opt-in enabled for this engine".into());
        }
        let st = Shards::open(files)?;
        st.validate()?;
        if max_rows > 512 {
            return Err(Error("StarVector prefix work limit is 512 rows".into()));
        }
        // Conservative admission estimate: half weights, eight full KV beams,
        // work buffers and driver margin. Actual tracked/device peaks remain a gate.
        let (_, free) = gpu.memory()?;
        if free.is_none_or(|bytes| bytes < 4_000_000_000) {
            return Err(Error(
                "StarVector needs 4 GB available device memory for the bounded plan".into(),
            ));
        }
        let l = Loader { st: &st, gpu };
        let e = "model.image_encoder.visual_encoder";
        #[cfg(not(feature = "patch-f32"))]
        let patch = l.lin(&format!("{e}.conv1"))?;
        #[cfg(feature = "patch-f32")]
        let patch = {
            log("starvector: opt-in patch-f32; only patch input/projection uses F32".into());
            Lin {
                w: l.float(&format!("{e}.conv1.weight"))?,
                b: None,
                n: V_D,
                k: 3 * PATCH * PATCH,
            }
        };
        if patch.k != 3 * PATCH * PATCH || patch.n != V_D {
            return Err(Error(format!(
                "the image encoder's patch embedding is {}x{}, not CLIP ViT-L/14's",
                patch.n, patch.k
            )));
        }
        let class = st.f32(&format!("{e}.class_embedding"))?;
        let vpos = l.float(&format!("{e}.positional_embedding"))?;
        let ln_pre = l.ln(&format!("{e}.ln_pre"))?;
        if cfg!(feature = "vision-f32") {
            log(
                "starvector: opt-in vision-f32; CLIP F32, adapter and LM retain half intermediates"
                    .into(),
            );
        }
        let mut vis = Vec::new();
        for i in 0..V_LAYERS {
            let p = format!("{e}.transformer.resblocks.{i}");
            let qkv_w = st.f32(&format!("{p}.attn.in_proj_weight"))?;
            let qkv = Lin {
                w: if cfg!(feature = "vision-f32") {
                    DevBuf::from_f32(gpu, &qkv_w)?
                } else {
                    l.half_of(&qkv_w)?
                },
                b: Some(l.float(&format!("{p}.attn.in_proj_bias"))?),
                n: 3 * V_D,
                k: V_D,
            };
            vis.push(VBlock {
                ln1: l.ln(&format!("{p}.ln_1"))?,
                qkv,
                o: l.lin_storage(&format!("{p}.attn.out_proj"), cfg!(feature = "vision-f32"))?,
                ln2: l.ln(&format!("{p}.ln_2"))?,
                fc: l.lin_storage(&format!("{p}.mlp.c_fc"), cfg!(feature = "vision-f32"))?,
                proj: l.lin_storage(&format!("{p}.mlp.c_proj"), cfg!(feature = "vision-f32"))?,
            });
        }
        let ln_vision = l.ln("model.image_encoder.ln_vision")?;
        let a = "model.image_projection";
        let bn = [
            st.f32(&format!("{a}.norm.weight"))?,
            st.f32(&format!("{a}.norm.bias"))?,
            st.f32(&format!("{a}.norm.running_mean"))?,
            st.f32(&format!("{a}.norm.running_var"))?,
        ];
        let a_fc = l.lin(&format!("{a}.c_fc"))?;
        let a_proj = l.lin(&format!("{a}.c_proj"))?;
        log(format!(
            "starvector: the image encoder ({V_LAYERS} layers) and the adapter on {}",
            gpu.name
        ));
        let t = "model.svg_transformer.transformer.transformer";
        let mut blocks = Vec::new();
        for i in 0..LAYERS {
            let p = format!("{t}.h.{i}");
            blocks.push(Block {
                ln1: l.ln(&format!("{p}.ln_1"))?,
                q: l.lin_rows(&format!("{p}.attn.c_attn"), 0, D)?,
                k: l.lin_rows(&format!("{p}.attn.c_attn"), D, D + HD)?,
                v: l.lin_rows(&format!("{p}.attn.c_attn"), D + HD, D + 2 * HD)?,
                o: l.lin(&format!("{p}.attn.c_proj"))?,
                ln2: l.ln(&format!("{p}.ln_2"))?,
                fc: l.lin(&format!("{p}.mlp.c_fc"))?,
                proj: l.lin(&format!("{p}.mlp.c_proj"))?,
            });
        }
        let ln_f = l.ln(&format!("{t}.ln_f"))?;
        let wte_f = st.f32(&format!("{t}.wte.weight"))?;
        let vocab = wte_f.len() / D;
        let wte_host: Vec<u16> = wte_f.iter().map(|x| f32_to_f16(*x)).collect();
        drop(wte_f);
        let wte = DevBuf::new(gpu, wte_host.len() * 2)?;
        // SAFETY: plain halves as bytes.
        wte.write(0, unsafe {
            std::slice::from_raw_parts(wte_host.as_ptr().cast::<u8>(), wte_host.len() * 2)
        })?;
        let wpe_host = st.f32(&format!("{t}.wpe.weight"))?;
        log(format!(
            "starvector: StarCoder-1B ({LAYERS} layers, a vocabulary of {vocab}) on {}",
            gpu.name
        ));
        let rows = max_rows.max(QUERY + 8);
        let h16 = |n: usize| DevBuf::new(gpu, n * 2);
        let f32b = |n: usize| DevBuf::f32(gpu, n);
        let vision_buf = |n: usize| DevBuf::new(gpu, n * VISION_BYTES);
        Ok(Model {
            nsd,
            gpu: gpu.clone(),
            up: crate::ffi::api(gpu).map_err(|e| Error(e.0))?,
            patch,
            class,
            vpos,
            ln_pre,
            vis,
            ln_vision,
            a_fc,
            a_proj,
            bn,
            blocks,
            ln_f,
            wte,
            wte_host,
            wpe_host,
            vocab,
            caches: Vec::new(),
            px: DevBuf::new(
                gpu,
                GRID * GRID * 3 * PATCH * PATCH * std::mem::size_of::<PatchValue>(),
            )?,
            vx: f32b(QUERY * V_D)?,
            vx2: f32b(QUERY * V_D)?,
            vh: vision_buf(QUERY * V_D)?,
            vqkv: vision_buf(QUERY * 3 * V_D)?,
            vatt: vision_buf(QUERY * V_D)?,
            vy: vision_buf(QUERY * V_D)?,
            vff: vision_buf(QUERY * V_FF)?,
            #[cfg(not(feature = "vision-f32"))]
            vtmp: vision_buf(QUERY * V_FF)?,
            enc_f: f32b(QUERY * V_D)?,
            enc_h: h16(QUERY * V_D)?,
            aff: h16(QUERY * 2 * V_D)?,
            atmp: h16(QUERY * 2 * V_D)?,
            aout: f32b(QUERY * D)?,
            x: f32b(rows * D)?,
            h: h16(rows * D)?,
            q: h16(rows * D)?,
            kb: h16(rows * HD)?,
            vb: h16(rows * HD)?,
            att: h16(rows * D)?,
            y: h16(rows * D)?,
            ff: h16(rows * FF)?,
            logits: f32b(8 * vocab)?,
            rows,
        })
    }

    /// bytes of weights on the card
    pub fn weight_bytes(vocab: usize) -> u64 {
        let vis = V_LAYERS * (4 * V_D * V_D + 2 * V_D * V_FF) + V_D * 3 * PATCH * PATCH;
        let ad = V_D * 2 * V_D + 2 * V_D * D;
        let lm = LAYERS * (D * (D + 2 * HD) + D * D + 2 * D * FF) + vocab * D;
        let extra_patch = if cfg!(feature = "patch-f32") {
            2 * V_D * 3 * PATCH * PATCH
        } else {
            0
        };
        let extra_vision = if cfg!(feature = "vision-f32") {
            2 * V_LAYERS * (4 * V_D * V_D + 2 * V_D * V_FF)
        } else {
            0
        };
        (2 * (vis + ad + lm) + extra_patch + extra_vision) as u64
    }

    fn linear(&self, x: P, m: usize, w: &Lin, out: M, out_dt: Dt) -> Result<()> {
        self.nsd.linear(
            x,
            Dt::F16,
            m,
            w.k,
            w.w.ptr(),
            w.n,
            w.b.as_ref().map_or(none(), |b| b.ptr() as P),
            out,
            out_dt,
        )
    }

    fn vision_linear(&self, x: P, w: &Lin, out: M) -> Result<()> {
        self.nsd.linear(
            x,
            VISION_DT,
            QUERY,
            w.k,
            w.w.ptr(),
            w.n,
            w.b.as_ref().map_or(none(), |b| b.ptr() as P),
            out,
            VISION_DT,
        )
    }

    fn ln(&self, x: P, m: usize, c: usize, n: &Ln, out: M, dt: Dt) -> Result<()> {
        self.nsd
            .layer_norm(x, Dt::F32, m, c, n.w.ptr(), n.b.ptr(), EPS, out, dt)
    }

    fn add(&self, x: M, y: P, m: usize, c: usize) -> Result<()> {
        self.nsd
            .gate_add(x, Dt::F32, m, c, y, Dt::F16, none(), none())
    }

    fn up_check(&self, rc: std::ffi::c_int, what: &str) -> Result<()> {
        if rc == 0 {
            return Ok(());
        }
        let why = nextrocm_engine_core::api()
            .map(|a| a.error())
            .unwrap_or_default();
        Err(Error(format!("{what}: {why}")))
    }

    /// x (half, n) = x * sigmoid(s x): QuickGELU (s 1.702) or Swish (s 1); `tmp` holds n halves
    fn gated(&self, x: M, n: usize, s: f32, tmp: M) -> Result<()> {
        profile_model!(self.gpu, format!("gated:n={n}"), {
            let g = self.gpu.raw();
            // SAFETY: x and tmp hold n halves.
            unsafe {
                self.up_check((self.up.add)(g, x, x, n as i64, 0, s, 0.0, tmp), "scale")?;
                self.up_check(
                    (self.up.act)(g, tmp, n as i64, 1, 4, 0.0, std::ptr::null(), 0),
                    "sigmoid",
                )?;
                self.up_check(
                    (self.up.add)(g, x, tmp, n as i64, 1, 1.0, 1.0, x),
                    "product",
                )?;
            }
            Ok(())
        })
    }

    /// The image encoder (its output after ln_vision, [257, 1024] floats) and the adapter's (the language model's
    /// first 257 rows, [257, 2048] floats) from the picture's pixels [3, 224, 224]
    pub fn encode(&self, pixels: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if pixels.len() != 3 * SIZE * SIZE || pixels.iter().any(|v| !v.is_finite()) {
            return Err(Error(
                "StarVector encoder requires finite [3,224,224] pixels".into(),
            ));
        }
        #[cfg(feature = "vision-taps")]
        let mut diagnostic = crate::vision_taps::Taps::open()?;
        macro_rules! tap {
            ($name:expr, $buf:expr, $offset:expr, $rows:expr, $cols:expr, $half:expr) => {
                #[cfg(feature = "vision-taps")]
                if let Some(taps) = diagnostic.as_mut() {
                    taps.device($name, $buf, $offset, $rows, $cols, $half)?;
                }
            };
        }
        let n = &self.nsd;
        let np = GRID * GRID;
        let kk = 3 * PATCH * PATCH;
        // the patches as rows (channel, row, column within the patch: conv1's weight order)
        let mut cols = vec![PatchValue::default(); np * kk];
        for gy in 0..GRID {
            for gx in 0..GRID {
                let r = gy * GRID + gx;
                for c in 0..3 {
                    for ky in 0..PATCH {
                        for kx in 0..PATCH {
                            let v = pixels
                                [c * SIZE * SIZE + (gy * PATCH + ky) * SIZE + gx * PATCH + kx];
                            let index = r * kk + (c * PATCH + ky) * PATCH + kx;
                            #[cfg(feature = "patch-f32")]
                            {
                                cols[index] = v;
                            }
                            #[cfg(not(feature = "patch-f32"))]
                            {
                                cols[index] = f32_to_f16(v);
                            }
                        }
                    }
                }
            }
        }
        // SAFETY: contiguous initialized PatchValue elements, F16 bits or F32.
        self.px.write(0, unsafe {
            std::slice::from_raw_parts(
                cols.as_ptr().cast::<u8>(),
                cols.len() * std::mem::size_of::<PatchValue>(),
            )
        })?;
        // SAFETY: plain floats as bytes.
        self.vx.write(0, unsafe {
            std::slice::from_raw_parts(self.class.as_ptr().cast::<u8>(), V_D * 4)
        })?;
        n.linear(
            self.px.ptr(),
            PATCH_DT,
            np,
            kk,
            self.patch.w.ptr(),
            V_D,
            none(),
            at(&self.vx, (V_D * 4) as isize),
            Dt::F32,
        )?;
        tap!("patch", &self.vx, V_D * 4, np, V_D, false);
        n.gate_add(
            self.vx.ptr(),
            Dt::F32,
            QUERY,
            V_D,
            self.vpos.ptr(),
            Dt::F32,
            none(),
            none(),
        )?;
        tap!("positions", &self.vx, 0, QUERY, V_D, false);
        self.ln(
            self.vx.ptr(),
            QUERY,
            V_D,
            &self.ln_pre,
            self.vx2.ptr(),
            Dt::F32,
        )?;
        tap!("ln_pre", &self.vx2, 0, QUERY, V_D, false);
        let x = self.vx2.ptr();
        for (_block, b) in self.vis.iter().enumerate() {
            self.ln(x, QUERY, V_D, &b.ln1, self.vh.ptr(), VISION_DT)?;
            if _block == 0 {
                tap!("block00_ln1", &self.vh, 0, QUERY, V_D, VISION_DT == Dt::F16);
            }
            self.vision_linear(self.vh.ptr(), &b.qkv, self.vqkv.ptr())?;
            let q = self.vqkv.ptr() as P;
            n.attention(
                q,
                at(&self.vqkv, (V_D * VISION_BYTES) as isize),
                at(&self.vqkv, (2 * V_D * VISION_BYTES) as isize),
                VISION_DT,
                QUERY,
                V_HEADS,
                V_D / V_HEADS,
                3 * V_D,
                self.vatt.ptr(),
                VISION_DT,
            )?;
            self.vision_linear(self.vatt.ptr(), &b.o, self.vy.ptr())?;
            if _block == 0 {
                tap!(
                    "block00_att_out",
                    &self.vy,
                    0,
                    QUERY,
                    V_D,
                    VISION_DT == Dt::F16
                );
            }
            n.gate_add(
                x,
                Dt::F32,
                QUERY,
                V_D,
                self.vy.ptr(),
                VISION_DT,
                none(),
                none(),
            )?;
            if _block == 0 {
                tap!("block00_residual_att", &self.vx2, 0, QUERY, V_D, false);
            }
            self.ln(x, QUERY, V_D, &b.ln2, self.vh.ptr(), VISION_DT)?;
            if _block == 0 {
                tap!("block00_ln2", &self.vh, 0, QUERY, V_D, VISION_DT == Dt::F16);
            }
            self.vision_linear(self.vh.ptr(), &b.fc, self.vff.ptr())?;
            if _block == 0 {
                tap!(
                    "block00_fc",
                    &self.vff,
                    0,
                    QUERY,
                    V_FF,
                    VISION_DT == Dt::F16
                );
            }
            #[cfg(not(feature = "vision-f32"))]
            self.gated(self.vff.ptr(), QUERY * V_FF, 1.702, self.vtmp.ptr())?;
            #[cfg(feature = "vision-f32")]
            n.quick_gelu_f32(self.vff.ptr(), QUERY * V_FF)?;
            if _block == 0 {
                tap!(
                    "block00_gelu",
                    &self.vff,
                    0,
                    QUERY,
                    V_FF,
                    VISION_DT == Dt::F16
                );
            }
            self.vision_linear(self.vff.ptr(), &b.proj, self.vy.ptr())?;
            if _block == 0 {
                tap!(
                    "block00_proj",
                    &self.vy,
                    0,
                    QUERY,
                    V_D,
                    VISION_DT == Dt::F16
                );
            }
            n.gate_add(
                x,
                Dt::F32,
                QUERY,
                V_D,
                self.vy.ptr(),
                VISION_DT,
                none(),
                none(),
            )?;
            tap!(
                &format!("block{_block:02}"),
                &self.vx2,
                0,
                QUERY,
                V_D,
                false
            );
        }
        self.ln(x, QUERY, V_D, &self.ln_vision, self.enc_f.ptr(), Dt::F32)?;
        tap!("ln_vision", &self.enc_f, 0, QUERY, V_D, false);
        self.ln(x, QUERY, V_D, &self.ln_vision, self.enc_h.ptr(), Dt::F16)?;
        self.linear(self.enc_h.ptr(), QUERY, &self.a_fc, self.aff.ptr(), Dt::F16)?;
        self.gated(self.aff.ptr(), QUERY * 2 * V_D, 1.0, self.atmp.ptr())?;
        self.linear(
            self.aff.ptr(),
            QUERY,
            &self.a_proj,
            self.aout.ptr(),
            Dt::F32,
        )?;
        let mut enc = vec![0f32; QUERY * V_D];
        let mut emb = vec![0f32; QUERY * D];
        // SAFETY: plain floats as bytes.
        unsafe {
            self.enc_f.read(
                0,
                std::slice::from_raw_parts_mut(enc.as_mut_ptr().cast::<u8>(), enc.len() * 4),
            )?;
            self.aout.read(
                0,
                std::slice::from_raw_parts_mut(emb.as_mut_ptr().cast::<u8>(), emb.len() * 4),
            )?;
        }
        #[cfg(feature = "vision-taps")]
        if let Some(taps) = diagnostic.as_mut() {
            taps.host("adapter_pre_bn", &emb, QUERY, D)?;
        }
        // BatchNorm1d(257), eval: one affine a position
        let [w, b, mean, var] = &self.bn;
        for r in 0..QUERY {
            let s = w[r] / (var[r] + EPS).sqrt();
            for v in &mut emb[r * D..(r + 1) * D] {
                *v = (*v - mean[r]) * s + b[r];
            }
        }
        #[cfg(feature = "vision-taps")]
        if let Some(taps) = diagnostic.as_mut() {
            taps.host("adapter", &emb, QUERY, D)?;
        }
        Ok((enc, emb))
    }

    /// A token's embedding (half, as the card has it) as floats
    pub fn token_row(&self, t: u32) -> Vec<f32> {
        self.wte_host[t as usize * D..(t as usize + 1) * D]
            .iter()
            .map(|h| f16_to_f32(*h))
            .collect()
    }

    fn ensure_caches(&mut self, beams: usize) -> Result<()> {
        if !(1..=8).contains(&beams) {
            return Err(Error("StarVector supports 1..8 beams".into()));
        }
        while self.caches.len() < beams {
            let mk = || {
                (0..LAYERS)
                    .map(|_| DevBuf::new(&self.gpu, POSITIONS * HD * 2))
                    .collect::<nextrocm_engine_core::Result<Vec<_>>>()
            };
            self.caches.push(Cache { k: mk()?, v: mk()? });
        }
        Ok(())
    }

    /// The prefix (`embeds` [rows, 2048]: the adapter's rows and the prompt's tokens) through the language model into
    /// cache 0: the last row's logits
    pub fn prefill(&mut self, embeds: &[f32], beams: usize) -> Result<Vec<f32>> {
        if embeds.is_empty()
            || embeds.len() % D != 0
            || embeds.len() / D > self.rows
            || embeds.iter().any(|v| !v.is_finite())
        {
            return Err(Error("invalid StarVector prefix embeddings".into()));
        }
        self.ensure_caches(beams.max(1))?;
        let m = embeds.len() / D;
        if m > self.rows {
            return Err(Error(format!(
                "a prefix of {m} rows, past the {} the work buffers hold",
                self.rows
            )));
        }
        let mut x = embeds.to_vec();
        for (i, v) in x.iter_mut().enumerate() {
            *v += self.wpe_host[i];
        }
        // SAFETY: plain floats as bytes.
        profile_model!(
            self.gpu,
            "input_upload".into(),
            self.x.write(0, unsafe {
                std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len() * 4)
            })
        )?;
        self.layers(m, &[0], 0)?;
        self.head(m - 1, 1)
    }

    /// `from[i]`'s cache becomes beam i's (`len` rows of it), then beam i is fed tokens[i] at position `len`: the
    /// logits [beams, vocab]
    pub fn step(&mut self, from: &[usize], tokens: &[u32], len: usize) -> Result<Vec<f32>> {
        let nb = tokens.len();
        if nb == 0
            || nb > 8
            || from.len() != nb
            || from.iter().any(|&i| i >= self.caches.len())
            || tokens.iter().any(|&t| t as usize >= self.vocab)
        {
            return Err(Error("invalid StarVector beam source/token".into()));
        }
        if len >= POSITIONS {
            return Err(Error(format!(
                "position {len}: past StarCoder's {POSITIONS}"
            )));
        }
        self.ensure_caches(nb)?;
        // the new beams' caches: a beam keeps a slot no other took, the others get a copy in the slots left over
        let mut slot_of = vec![usize::MAX; nb];
        let mut taken = vec![false; self.caches.len()];
        for (i, &f) in from.iter().enumerate() {
            if !taken[f] {
                taken[f] = true;
                slot_of[i] = f;
            }
        }
        let mut copies = Vec::new();
        for i in 0..nb {
            if slot_of[i] == usize::MAX {
                let free = (0..taken.len())
                    .find(|&s| !taken[s])
                    .ok_or_else(|| Error("no free cache for a beam".into()))?;
                taken[free] = true;
                slot_of[i] = free;
                copies.push((from[i], free));
            }
        }
        for (src, dst) in copies {
            for l in 0..LAYERS {
                profile_model!(self.gpu, format!("kv_reorder:keys={len}"), {
                    self.caches[dst].k[l].copy_within(
                        0,
                        &self.caches[src].k[l],
                        0,
                        len * HD * 2,
                    )?;
                    self.caches[dst].v[l].copy_within(0, &self.caches[src].v[l], 0, len * HD * 2)
                })?;
            }
        }
        // rename: beam i lives in caches[i]
        let mut taken_caches: Vec<Option<Cache>> = std::mem::take(&mut self.caches)
            .into_iter()
            .map(Some)
            .collect();
        let mut next: Vec<Cache> = slot_of
            .iter()
            .map(|&s| taken_caches[s].take().expect("each slot once"))
            .collect();
        next.extend(taken_caches.into_iter().flatten());
        self.caches = next;
        let mut x = vec![0f32; nb * D];
        for (i, &t) in tokens.iter().enumerate() {
            let row = self.token_row(t);
            for j in 0..D {
                x[i * D + j] = row[j] + self.wpe_host[len * D + j];
            }
        }
        // SAFETY: plain floats as bytes.
        profile_model!(
            self.gpu,
            "input_upload".into(),
            self.x.write(0, unsafe {
                std::slice::from_raw_parts(x.as_ptr().cast::<u8>(), x.len() * 4)
            })
        )?;
        let beams: Vec<usize> = (0..nb).collect();
        self.layers(nb, &beams, len)?;
        self.head(0, nb)
    }

    /// The decoder layers over `m` rows in `self.x`: a prefix of beam 0 at positions [0, m) (beams = [0]), or one row
    /// a beam at position `pos`
    fn layers(&self, m: usize, beams: &[usize], pos: usize) -> Result<()> {
        let n = &self.nsd;
        let prefix = beams.len() == 1 && m > 1;
        for (li, b) in self.blocks.iter().enumerate() {
            self.ln(self.x.ptr(), m, D, &b.ln1, self.h.ptr(), Dt::F16)?;
            self.linear(self.h.ptr(), m, &b.q, self.q.ptr(), Dt::F16)?;
            if prefix {
                let c = &self.caches[0];
                self.linear(self.h.ptr(), m, &b.k, c.k[li].ptr(), Dt::F16)?;
                self.linear(self.h.ptr(), m, &b.v, c.v[li].ptr(), Dt::F16)?;
                n.attention_causal_rows(
                    self.q.ptr(),
                    c.k[li].ptr(),
                    c.v[li].ptr(),
                    Dt::F16,
                    0,
                    m,
                    HEADS,
                    1,
                    HD,
                    D,
                    HD,
                    self.att.ptr(),
                )?;
            } else {
                self.linear(self.h.ptr(), m, &b.k, self.kb.ptr(), Dt::F16)?;
                self.linear(self.h.ptr(), m, &b.v, self.vb.ptr(), Dt::F16)?;
                for (i, &bi) in beams.iter().enumerate() {
                    let c = &self.caches[bi];
                    profile_model!(self.gpu, "kv_append".into(), {
                        c.k[li].copy_within(pos * HD * 2, &self.kb, i * HD * 2, HD * 2)?;
                        c.v[li].copy_within(pos * HD * 2, &self.vb, i * HD * 2, HD * 2)
                    })?;
                    // Private HIP ABI takes the actual query row pointer; pos only offsets the causal mask.
                    let q = at(&self.q, (i * D * 2) as isize);
                    n.attention_causal_rows(
                        q,
                        c.k[li].ptr(),
                        c.v[li].ptr(),
                        Dt::F16,
                        pos,
                        1,
                        HEADS,
                        1,
                        HD,
                        D,
                        HD,
                        at(&self.att, (i * D * 2) as isize),
                    )?;
                }
            }
            self.linear(self.att.ptr(), m, &b.o, self.y.ptr(), Dt::F16)?;
            self.add(self.x.ptr(), self.y.ptr(), m, D)?;
            self.ln(self.x.ptr(), m, D, &b.ln2, self.h.ptr(), Dt::F16)?;
            self.linear(self.h.ptr(), m, &b.fc, self.ff.ptr(), Dt::F16)?;
            n.gelu(self.ff.ptr(), Dt::F16, m * FF, true)?;
            self.linear(self.ff.ptr(), m, &b.proj, self.y.ptr(), Dt::F16)?;
            self.add(self.x.ptr(), self.y.ptr(), m, D)?;
        }
        Ok(())
    }

    /// ln_f and the logits of rows [from, from + r) of `self.x`
    fn head(&self, from: usize, r: usize) -> Result<Vec<f32>> {
        let x0 = at(&self.x, (from * D * 4) as isize);
        self.nsd.layer_norm(
            x0,
            Dt::F32,
            r,
            D,
            self.ln_f.w.ptr(),
            self.ln_f.b.ptr(),
            EPS,
            self.h.ptr(),
            Dt::F16,
        )?;
        self.nsd.linear(
            self.h.ptr(),
            Dt::F16,
            r,
            D,
            self.wte.ptr(),
            self.vocab,
            none(),
            self.logits.ptr(),
            Dt::F32,
        )?;
        let mut out = vec![0f32; r * self.vocab];
        // SAFETY: plain floats as bytes.
        profile_model!(
            self.gpu,
            "logits_readback".into(),
            self.logits.read(0, unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), out.len() * 4)
            })
        )?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_nearest_even() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(65520.0), 0x7c00);
        for x in [0.5f32, -3.25, 1e-4, 1234.5] {
            assert!((f16_to_f32(f32_to_f16(x)) - x).abs() <= x.abs() * 1e-3);
        }
    }

    #[test]
    fn the_sizes_are_starvector_1bs() {
        assert_eq!(QUERY, 257);
        let gib = Model::weight_bytes(49156) as f64 / (1u64 << 30) as f64;
        let range = if cfg!(feature = "vision-f32") {
            3.0..3.5
        } else {
            2.5..2.9
        };
        assert!(range.contains(&gib), "{gib}");
    }
}
