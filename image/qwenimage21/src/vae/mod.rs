//! The decoder: Qwen-Image 2.1's VAE (diffusers' AutoencoderKLQwenImage21, a Wan 2.2-style decoder with 2-D kernels
//! and 64 latent channels, 16x down) from latents to RGBA. Everything is channels-last half on the GPU; one picture is
//! one frame, so the causal video parts reduce to their first chunk (the time convolutions are skipped, each
//! duplicate-upsample shortcut keeps its last time copy).
//!
//! ```text
//!   z [h, w, 64] (latents * std + mean) -> 1x1 conv -> 3x3 conv (1152)
//!   middle: resnet, attention (one head over all pixels), resnet
//!   5 up blocks (1152, 1152, 576, 288, 144): 3 resnets, then (blocks 0-3) nearest 2x + 3x3 conv, plus the block
//!     input's duplicate-upsample shortcut
//!   rms norm, silu, 3x3 conv -> 4 channels (RGBA) in [-1, 1]
//! resnet: x + conv(silu(norm(conv(silu(norm(x)))))), a 1x1 conv on the shortcut when the width changes
//! ```
//!
//! The file names its tensors as Wan / ComfyUI do (decoder.middle.0.residual.2, decoder.upsamples.3.upsamples.0, ...).

use std::path::Path;
use std::sync::Arc;

use backend::{none, Backend as Nsd, Dt};
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_gguf::safetensors::SafeTensors;

// Diagnostic-only leaf timing. Normal builds execute the unchanged expression.
macro_rules! vae_gpu_span {
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

mod backend;
mod plan;
use backend as ffi;
pub use plan::{validate_dimensions, VaePlan};

/// Explicit diagnostic selection; ordinary load retains original convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvKernel {
    Baseline,
    Register64Measured1024,
}
impl ConvKernel {
    pub fn uses_register64(self, h: usize, w: usize, ci: usize, co: usize, k: usize) -> bool {
        self == Self::Register64Measured1024
            && matches!(
                (h, w, ci, co, k),
                (64, 64, 64, 64, 1)
                    | (64, 64, 64, 1152, 3)
                    | (64, 64, 1152, 1152, 3)
                    | (128, 128, 1152, 1152, 3)
                    | (256, 256, 576, 576, 3)
                    | (256, 256, 1152, 576, 1)
                    | (256, 256, 1152, 576, 3)
                    | (256, 256, 1152, 1152, 3)
                    | (512, 512, 288, 288, 3)
                    | (512, 512, 576, 288, 1)
                    | (512, 512, 576, 288, 3)
                    | (512, 512, 576, 576, 3)
                    | (1024, 1024, 144, 4, 3)
                    | (1024, 1024, 144, 144, 3)
                    | (1024, 1024, 288, 144, 1)
                    | (1024, 1024, 288, 144, 3)
                    | (1024, 1024, 288, 288, 3)
            )
    }
}

/// F.normalize's floor, as an epsilon on the mean square: never reached by real activations
const EPS: f32 = 1e-24;

/// Kept free on the card beside a decode (the box spills to host memory and stalls past it)
pub const MARGIN: u64 = 3 << 29;

/// Conservative reserve retained from the original stage planner. Native private
/// convolution uses bounded LDS and no global im2col; this is not a measured ROCm scratch claim.
pub const SLACK: u64 = 1 << 30;

/// The fewest latent rows a band of the up path covers (its context rows besides)
const MIN_BAND: usize = 4;

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// A convolution: half weights [Co, Ci, k, k], float32 bias
struct Conv {
    w: DevBuf,
    b: DevBuf,
    ci: usize,
    co: usize,
    k: usize,
}

struct Resnet {
    norm1: DevBuf,
    conv1: Conv,
    norm2: DevBuf,
    conv2: Conv,
    shortcut: Option<Conv>,
}

struct Up {
    resnets: Vec<Resnet>,
    /// the 2x upsample's conv (blocks 0-3)
    resample: Option<Conv>,
    /// the shortcut's time copies (2: a block that also upsamples in time; 1: space only), when it upsamples
    ft: Option<i32>,
    ci: usize,
}

/// The middle's attention block: one head of every channel
struct Att {
    norm: DevBuf,
    qkv: DevBuf,
    qkv_b: DevBuf,
    proj: DevBuf,
    proj_b: DevBuf,
}

/// An encoder down block: resnets, then (blocks 0-3) the zero-padded stride-2 conv; its input's average shortcut
/// (2x2 in space, and (blocks 1-3) a zero frame in time) added
struct Down {
    resnets: Vec<Resnet>,
    resample: Option<Conv>,
    ft: i32,
    ci: usize,
    co: usize,
}

/// The encoder (edits: the condition pictures to latents): RGBA [H, W, 4] in [-1, 1] -> conv_in (96) -> 5 down blocks
/// (96, 192, 384, 768, 768; 16x down) -> middle (resnet, attention, resnet) -> rms norm, silu, 3x3 conv (128) ->
/// 1x1 conv; the mean is the first 64 channels
struct Encoder {
    conv_in: Conv,
    downs: Vec<Down>,
    mid: [Resnet; 2],
    att: Att,
    out_norm: DevBuf,
    head: Conv,
    quant: Conv,
}

pub struct Vae {
    backend: Nsd,
    allowance: u64,
    pub resident_bytes: u64,
    gpu: Arc<Gpu>,
    /// an encode running (the memory probes' phase: NS_QI_MEM)
    encoding: std::sync::atomic::AtomicBool,
    post_quant: Conv,
    conv_in: Conv,
    mid: [Resnet; 2],
    att: Att,
    enc: Option<Encoder>,
    ups: Vec<Up>,
    out_norm: DevBuf,
    conv_out: Conv,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub z: usize,
}

fn ge(e: nextrocm_gguf::Error) -> Error {
    Error(e.0)
}

/// Wan 2.2's latent statistics for the 64-channel VAE (diffusers' vae/config.json: latents_mean, latents_std), read
/// from the model's config when it is beside the file, else these
fn stats(file: &Path) -> Option<(Vec<f32>, Vec<f32>)> {
    let dir = file.parent()?;
    for name in ["config.json", "vae_config.json"] {
        if let Ok(s) = std::fs::read_to_string(dir.join(name)) {
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            let f = |k: &str| {
                v[k].as_array().map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_f64())
                        .map(|x| x as f32)
                        .collect::<Vec<f32>>()
                })
            };
            if let (Some(m), Some(s)) = (f("latents_mean"), f("latents_std")) {
                return Some((m, s));
            }
        }
    }
    None
}

impl Vae {
    /// The decoder from its file; `stats`: the latents' per-channel (mean, std)
    pub fn load(file: &Path, gpu: &Arc<Gpu>, device_allowance: usize) -> Result<Vae> {
        Self::load_with_conv(file, gpu, device_allowance, ConvKernel::Baseline)
    }

    pub fn load_with_conv(
        file: &Path,
        gpu: &Arc<Gpu>,
        device_allowance: usize,
        kernel: ConvKernel,
    ) -> Result<Vae> {
        let nsd = Nsd::new(gpu, kernel)?;
        let plan = VaePlan::read(file, device_allowance)?;
        let latent_stats = plan::default_stats()?;
        let t0 = std::time::Instant::now();
        let gpu = nsd.gpu.clone();
        let st = SafeTensors::open(file).map_err(ge)?;
        let half = |name: &str| -> Result<(DevBuf, Vec<u64>)> {
            let t = st.need(name).map_err(ge)?;
            let raw = st.f32(name).map_err(ge)?;
            if raw.iter().any(|v| !v.is_finite() || v.abs() > 65504.0) {
                return Err(Error(format!(
                    "{name}: weights cannot be represented as finite half"
                )));
            }
            let n = raw.len();
            let src = DevBuf::from_f32(&gpu, &raw)?;
            let out = DevBuf::new(&gpu, n * 2)?;
            // SAFETY: source and output allocations cover n F32 and F16 elements.
            ffi::check(
                unsafe { (ffi::api(&gpu)?.to_half)(gpu.raw(), src.fp(), n as i64, out.ptr()) },
                "VAE weight cast",
            )?;
            nsd.wait()?;
            Ok((out, t.shape.clone()))
        };
        let vec = |name: &str| -> Result<DevBuf> {
            let values = st.f32(name).map_err(ge)?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(Error(format!("{name}: nonfinite VAE vector")));
            }
            DevBuf::from_f32(&gpu, &values)
        };
        let conv = |p: &str| -> Result<Conv> {
            let (w, shape) = half(&format!("{p}.weight"))?;
            let (co, ci, k) = (
                shape[0] as usize,
                shape[1] as usize,
                *shape.last().unwrap_or(&1) as usize,
            );
            Ok(Conv {
                w,
                b: vec(&format!("{p}.bias"))?,
                ci,
                co,
                k,
            })
        };
        let resnet = |p: &str| -> Result<Resnet> {
            Ok(Resnet {
                norm1: vec(&format!("{p}.residual.0.gamma"))?,
                conv1: conv(&format!("{p}.residual.2"))?,
                norm2: vec(&format!("{p}.residual.3.gamma"))?,
                conv2: conv(&format!("{p}.residual.6"))?,
                shortcut: if st.tensor(&format!("{p}.shortcut.weight")).is_some() {
                    Some(conv(&format!("{p}.shortcut"))?)
                } else {
                    None
                },
            })
        };
        let mut ups = Vec::new();
        for i in 0.. {
            let p = format!("decoder.upsamples.{i}.upsamples");
            if st.tensor(&format!("{p}.0.residual.0.gamma")).is_none() {
                break;
            }
            let resnets = (0..3)
                .map(|j| resnet(&format!("{p}.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let ci = resnets[0].conv1.ci;
            let up = if st.tensor(&format!("{p}.3.resample.1.weight")).is_some() {
                let time = st.tensor(&format!("{p}.3.time_conv.weight")).is_some();
                Up {
                    resnets,
                    resample: Some(conv(&format!("{p}.3.resample.1"))?),
                    ft: Some(if time { 2 } else { 1 }),
                    ci,
                }
            } else {
                Up {
                    resnets,
                    resample: None,
                    ft: None,
                    ci,
                }
            };
            ups.push(up);
        }
        let att = |p: &str| -> Result<Att> {
            Ok(Att {
                norm: vec(&format!("{p}.norm.gamma"))?,
                qkv: half(&format!("{p}.to_qkv.weight"))?.0,
                qkv_b: vec(&format!("{p}.to_qkv.bias"))?,
                proj: half(&format!("{p}.proj.weight"))?.0,
                proj_b: vec(&format!("{p}.proj.bias"))?,
            })
        };
        // the encoder, when the file has it
        let enc = if st.tensor("encoder.conv1.weight").is_some() {
            let mut downs = Vec::new();
            for i in 0.. {
                let p = format!("encoder.downsamples.{i}.downsamples");
                if st.tensor(&format!("{p}.0.residual.0.gamma")).is_none() {
                    break;
                }
                let resnets = (0..2)
                    .map(|j| resnet(&format!("{p}.{j}")))
                    .collect::<Result<Vec<_>>>()?;
                let (ci, co) = (resnets[0].conv1.ci, resnets[1].conv2.co);
                let resample = if st.tensor(&format!("{p}.2.resample.1.weight")).is_some() {
                    Some(conv(&format!("{p}.2.resample.1"))?)
                } else {
                    None
                };
                let ft = if st.tensor(&format!("{p}.2.time_conv.weight")).is_some() {
                    2
                } else {
                    1
                };
                downs.push(Down {
                    resnets,
                    resample,
                    ft,
                    ci,
                    co,
                });
            }
            Some(Encoder {
                conv_in: conv("encoder.conv1")?,
                downs,
                mid: [resnet("encoder.middle.0")?, resnet("encoder.middle.2")?],
                att: att("encoder.middle.1")?,
                out_norm: vec("encoder.head.0.gamma")?,
                head: conv("encoder.head.2")?,
                quant: conv("conv1")?,
            })
        } else {
            None
        };
        let post_quant = conv("conv2")?;
        let z = post_quant.ci;
        let (mean, std) = stats(file).unwrap_or(latent_stats);
        if mean.len() != z
            || std.len() != z
            || mean.iter().any(|v| !v.is_finite())
            || std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error(format!(
                "the VAE takes {z} latent channels; the statistics have {} / {}",
                mean.len(),
                std.len()
            )));
        }
        let vae = Vae {
            gpu: gpu.clone(),
            allowance: device_allowance as u64,
            resident_bytes: plan.resident_bytes,
            encoding: std::sync::atomic::AtomicBool::new(false),
            post_quant,
            conv_in: conv("decoder.conv1")?,
            mid: [resnet("decoder.middle.0")?, resnet("decoder.middle.2")?],
            att: att("decoder.middle.1")?,
            enc,
            ups,
            out_norm: vec("decoder.head.0.gamma")?,
            conv_out: conv("decoder.head.2")?,
            mean,
            std,
            z,
            backend: nsd,
        };
        let _ = t0;
        Ok(vae)
    }

    fn conv(&self, nsd: &Nsd, c: &Conv, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, h * w * c.co * 2)?;
        self.conv_into(nsd, c, x, h, w, &out)?;
        Ok(out)
    }

    /// The convolution into `out` (h x w x co halves at least)
    fn conv_into(
        &self,
        nsd: &Nsd,
        c: &Conv,
        x: &DevBuf,
        h: usize,
        w: usize,
        out: &DevBuf,
    ) -> Result<()> {
        nsd.conv2d(
            x.ptr(),
            Dt::F16,
            1,
            h,
            w,
            c.ci,
            c.w.ptr(),
            c.co,
            c.k,
            c.b.ptr(),
            out.ptr(),
        )?;
        Ok(())
    }

    /// silu(rms norm(x) * gamma), a new buffer
    fn norm_silu(
        &self,
        nsd: &Nsd,
        x: &DevBuf,
        px: usize,
        c: usize,
        gamma: &DevBuf,
    ) -> Result<DevBuf> {
        let out = DevBuf::new(&self.gpu, px * c * 2)?;
        self.norm_silu_into(nsd, x, px, c, gamma, &out)?;
        Ok(out)
    }

    /// silu(rms norm(x) * gamma) into `out` (px x c halves at least)
    fn norm_silu_into(
        &self,
        nsd: &Nsd,
        x: &DevBuf,
        px: usize,
        c: usize,
        gamma: &DevBuf,
        out: &DevBuf,
    ) -> Result<()> {
        let k = ffi::api(&self.gpu)?;
        nsd.rms_norm_mod(
            x.ptr(),
            Dt::F16,
            px,
            c,
            gamma.ptr(),
            EPS,
            none(),
            none(),
            none(),
            out.ptr(),
            Dt::F16,
        )?;
        // SAFETY: px x c halves.
        vae_gpu_span!(
            &self.gpu,
            "vae.silu",
            &[px, c],
            ffi::check(
                unsafe { (k.silu)(self.gpu.raw(), out.ptr(), (px * c) as i64) },
                "silu",
            )
        )?;
        Ok(())
    }

    /// x + conv(silu(norm(conv(silu(norm(x)))))) in a new buffer, with one more beside it: `a` holds each
    /// convolution's input (and the shortcut's output), the result each one's output
    fn resnet(&self, nsd: &Nsd, r: &Resnet, x: &DevBuf, h: usize, w: usize) -> Result<DevBuf> {
        let px = h * w;
        let (ci, cm, co) = (r.conv1.ci, r.conv1.co, r.conv2.co);
        let a = DevBuf::new(&self.gpu, px * ci.max(cm).max(co) * 2)?;
        let out = DevBuf::new(&self.gpu, px * cm.max(co) * 2)?;
        self.norm_silu_into(nsd, x, px, ci, &r.norm1, &a)?;
        self.conv_into(nsd, &r.conv1, &a, h, w, &out)?;
        self.norm_silu_into(nsd, &out, px, cm, &r.norm2, &a)?;
        self.conv_into(nsd, &r.conv2, &a, h, w, &out)?;
        let short = match &r.shortcut {
            Some(c) => {
                self.conv_into(nsd, c, x, h, w, &a)?;
                &a
            }
            None => x,
        };
        nsd.gate_add(
            out.ptr(),
            Dt::F16,
            px,
            co,
            short.ptr(),
            Dt::F16,
            none(),
            none(),
        )?;
        // `a` is freed on return (and the caller may free x): the work on them first (a free does not wait for the queue)
        nsd.wait()?;
        Ok(out)
    }

    /// The bytes the up blocks and the head hold at most (beside the middle's output) over `h` x `w` latents: each
    /// resnet's input and two buffers, an upsampling block's input kept for its shortcut, the 2x copy and the
    /// convolution's output, the head's norm and output
    pub fn up_bytes(&self, h: usize, w: usize) -> u64 {
        if h == 0 || w == 0 || h > 256 || w > 256 {
            return u64::MAX;
        }
        let (mut h, mut w) = (h as u64, w as u64);
        let mut peak = 0u64;
        for up in &self.ups {
            let px = h * w;
            let kept = if up.ft.is_some() {
                px * up.ci as u64 * 2
            } else {
                0
            };
            for (i, r) in up.resnets.iter().enumerate() {
                let (ci, cm, co) = (r.conv1.ci as u64, r.conv1.co as u64, r.conv2.co as u64);
                // the first resnet's input is the block's input
                let input = if i == 0 && kept > 0 { 0 } else { px * ci * 2 };
                peak = peak.max(kept + input + px * 2 * (ci.max(cm).max(co) + cm.max(co)));
            }
            if let Some(c) = &up.resample {
                peak = peak.max(kept + 4 * px * 2 * (c.ci as u64 + c.co as u64));
                h *= 2;
                w *= 2;
            }
        }
        let c = self.conv_out.ci as u64;
        peak.max(h * w * 2 * (2 * c + self.conv_out.co as u64))
    }

    /// Latent rows of context a band of the up path needs each side to come out as the whole would: every 3 x 3
    /// convolution after the middle reaches a row further at its own scale (a 2x upsample doubles what is reached)
    pub fn halo(&self) -> usize {
        let mut rows = 0usize;
        let mut scale = 1usize;
        for up in &self.ups {
            rows += up
                .resnets
                .iter()
                .map(|r| (r.conv1.k / 2) + (r.conv2.k / 2))
                .sum::<usize>();
            if let Some(c) = &up.resample {
                rows = 2 * rows + c.k / 2;
                scale *= 2;
            }
        }
        rows += self.conv_out.k / 2;
        rows.div_ceil(scale)
    }

    /// The up blocks and the head over `x` [h, w, ci] (the middle's output, or a band of its rows): the picture as
    /// halves [h', w', 4] in [-1, 1] and its size
    fn up_path(&self, nsd: &Nsd, x: DevBuf, h: usize, w: usize) -> Result<(DevBuf, usize, usize)> {
        let gpu = &self.gpu;
        let k = ffi::api(&self.gpu)?;
        let (mut x, mut h, mut w) = (x, h, w);
        for up in &self.ups {
            // the block's input, kept for its shortcut when it upsamples (the first resnet reads it, not a copy)
            let inp = x;
            x = self.resnet(nsd, &up.resnets[0], &inp, h, w)?;
            let inp = up.ft.map(|_| inp);
            for r in &up.resnets[1..] {
                x = self.resnet(nsd, r, &x, h, w)?;
            }
            if let (Some(c), Some(ft), Some(inp)) = (&up.resample, up.ft, inp) {
                let co = c.co;
                let u = DevBuf::new(gpu, 4 * h * w * c.ci * 2)?;
                // SAFETY: x holds h x w x ci halves, u four times that.
                vae_gpu_span!(
                    gpu,
                    "vae.upsample",
                    &[h, w, c.ci],
                    ffi::check(
                        unsafe {
                            (k.up2)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, u.ptr())
                        },
                        "upsample",
                    )
                )?;
                // x is freed before the convolution's output is made: the copy first
                nsd.wait()?;
                drop(x);
                x = self.conv(nsd, c, &u, 2 * h, 2 * w)?;
                // SAFETY: inp holds h x w x ci halves, x 2h x 2w x co.
                vae_gpu_span!(
                    gpu,
                    "vae.upsample_shortcut",
                    &[h, w, up.ci, co],
                    ffi::check(
                        unsafe {
                            (k.dupup_add)(
                                gpu.raw(),
                                inp.ptr(),
                                h as i64,
                                w as i64,
                                up.ci as i64,
                                co as i64,
                                ft,
                                x.ptr(),
                            )
                        },
                        "shortcut",
                    )
                )?;
                nsd.wait()?;
                h *= 2;
                w *= 2;
            }
        }
        let c = self.conv_out.ci;
        let a = self.norm_silu(nsd, &x, h * w, c, &self.out_norm)?;
        let img = self.conv(nsd, &self.conv_out, &a, h, w)?;
        nsd.wait()?;
        Ok((img, h, w))
    }

    /// The least a decode of `h` x `w` latents holds at once: the middle's buffers (its attention's at the most), the
    /// picture's bytes, and the up path in bands of the fewest rows
    pub fn min_bytes(&self, h: usize, w: usize) -> u64 {
        if h == 0 || w == 0 || h > 256 || w > 256 {
            return u64::MAX;
        }
        let px = (h * w) as u64;
        let c = self.conv_in.co as u64;
        let rows = ((256u64 << 20) / (px * 4)).clamp(1, px);
        let mid = 6 * px * c * 2 + rows * px * 6;
        let band = self.up_bytes((MIN_BAND + 2 * self.halo()).min(h), w);
        mid.max(px * c * 2 + band) + px * 256 * 4 + SLACK
    }

    /// The latent rows a band of the up path covers when the card has `room` bytes for it over `h` x `w` latents: all
    /// of them when the whole fits (one pass), else the most whose band, context rows included, fits; None when not
    /// even a band of a few rows does
    pub fn band_rows(&self, h: usize, w: usize, room: u64) -> Option<usize> {
        if h == 0 || w == 0 || h > 256 || w > 256 {
            return None;
        }
        if self.up_bytes(h, w) + SLACK <= room {
            return Some(h);
        }
        let halo = self.halo();
        (MIN_BAND..h)
            .rev()
            .find(|r| self.up_bytes((r + 2 * halo).min(h), w) + SLACK <= room)
    }

    /// RGBA [H, W, 4] bytes from latents float32 [h * w, z] (as the denoiser leaves them, before the statistics). The
    /// up path runs over the whole when the explicit stage allowance and free memory permit, else
    /// a band of latent rows at a time with the rows its convolutions reach each side (the same picture, less memory)
    pub fn decode(&self, lat: &[f32], h: usize, w: usize) -> Result<(Vec<u8>, usize, usize)> {
        let (rgba, _, w, h) = self.decode_inner(lat, h, w, false)?;
        Ok((rgba, w, h))
    }
    /// Raw NHWC decoder output before clamp/byte conversion, for independent comparison.
    pub fn decode_f32(&self, lat: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let (_, raw, w, h) = self.decode_inner(lat, h, w, true)?;
        Ok((raw, w, h))
    }
    fn read_half(&self, img: &DevBuf, offset: usize, count: usize) -> Result<Vec<f32>> {
        let out = DevBuf::f32(&self.gpu, count)?;
        let src = img.view(offset * 2, count * 2)?;
        // SAFETY: views and output cover count half and float elements.
        vae_gpu_span!(
            &self.gpu,
            "vae.capture_cast",
            &[count],
            ffi::check(
                unsafe {
                    (ffi::api(&self.gpu)?.to_float)(self.gpu.raw(), src.ptr(), count as i64, out.fp())
                },
                "VAE raw float capture",
            )
        )?;
        self.gpu.sync()?;
        #[cfg(feature = "operator-profile")]
        let started = std::time::Instant::now();
        let result = out.to_f32();
        #[cfg(feature = "operator-profile")]
        crate::profiler::record_host(
            "vae.raw_readback",
            &[count],
            started.elapsed().as_secs_f64(),
        )?;
        result
    }
    fn decode_inner(
        &self,
        lat: &[f32],
        h: usize,
        w: usize,
        capture: bool,
    ) -> Result<(Vec<u8>, Vec<f32>, usize, usize)> {
        if h == 0 || w == 0 || h > 256 || w > 256 {
            return Err(Error(
                "VAE latent dimensions outside16..4096 image bound".into(),
            ));
        }
        validate_dimensions(h * 16, w * 16)?;
        if lat.len() != h * w * self.z || lat.iter().any(|v| !v.is_finite()) {
            return Err(Error("invalid VAE latents".into()));
        }
        if self.resident_bytes
            + self.min_bytes(h, w)
            + if capture {
                (h * w * 256 * 16) as u64
            } else {
                0
            }
            > self.allowance
        {
            return Err(Error("VAE decode exceeds device allowance".into()));
        }
        let nsd = &self.backend;
        let gpu = &self.gpu;
        let k = ffi::api(&self.gpu)?;
        let z = self.z;
        let zl: Vec<f32> = lat
            .iter()
            .enumerate()
            .map(|(i, v)| v * self.std[i % z] + self.mean[i % z])
            .collect();
        if zl.iter().any(|v| !v.is_finite() || v.abs() > 65504.0) {
            return Err(Error(
                "VAE denormalized latents outside finite half range".into(),
            ));
        }
        let zf = DevBuf::from_f32(gpu, &zl)?;
        let mut x = DevBuf::new(gpu, zl.len() * 2)?;
        // SAFETY: h x w x z values each way.
        vae_gpu_span!(
            gpu,
            "vae.latent_cast",
            &[h, w, z],
            ffi::check(
                unsafe { (k.to_half)(gpu.raw(), zf.fp(), zl.len() as i64, x.ptr()) },
                "to half",
            )
        )?;
        x = self.conv(nsd, &self.post_quant, &x, h, w)?;
        x = self.conv(nsd, &self.conv_in, &x, h, w)?;
        x = self.resnet(nsd, &self.mid[0], &x, h, w)?;
        x = self.attention(nsd, &self.att, x, h, w, self.conv_in.co)?;
        x = self.resnet(nsd, &self.mid[1], &x, h, w)?;
        drop(zf);
        let (hh, ww) = (h * 16, w * 16);
        let rgba = DevBuf::new(gpu, hh * ww * 4)?;
        let mut raw = if capture {
            vec![0f32; hh * ww * 4]
        } else {
            Vec::new()
        };
        // Caller reserves this stage alone. Also honor current free device memory;
        // leave the original safety margin and account for retained middle/RGBA.
        let (total, free) = gpu.memory()?;
        let retained = (h * w * self.conv_in.co * 2
            + hh * ww * 4
            + if capture { hh * ww * 16 } else { 0 }) as u64;
        let room = self
            .allowance
            .saturating_sub(self.resident_bytes + retained)
            .min(free.unwrap_or(total).saturating_sub(MARGIN));
        let rows = self.band_rows(h, w, room).ok_or_else(|| {
            Error(format!(
                "qwen-image-2.1: decoding {ww}x{hh} needs {:.1} GiB more on {}",
                gib(
                    (self.up_bytes((MIN_BAND + 2 * self.halo()).min(h), w) + SLACK)
                        .saturating_sub(room)
                ),
                gpu.name
            ))
        })?;
        if rows >= h {
            let (img, ph, pw) = self.up_path(nsd, x, h, w)?;
            if capture {
                raw = self.read_half(&img, 0, ph * pw * 4)?;
            }
            // SAFETY: ph x pw x 4 halves in, as many bytes out.
            vae_gpu_span!(
                gpu,
                "vae.rgba",
                &[ph, pw, 4],
                ffi::check(
                    unsafe {
                        (k.to_rgba8)(
                            gpu.raw(),
                            img.ptr(),
                            ph as i64,
                            pw as i64,
                            rgba.ptr().cast(),
                        )
                    },
                    "to rgba",
                )
            )?;
            nsd.wait()?;
        } else {
            let halo = self.halo();
            let row = w * self.conv_in.co * 2;
            let mut r0 = 0;
            while r0 < h {
                let r1 = (r0 + rows).min(h);
                let (a, b) = (r0.saturating_sub(halo), (r1 + halo).min(h));
                let (img, _, pw) = self.up_path(nsd, x.view(a * row, (b - a) * row)?, b - a, w)?;
                // the band's own rows of the picture
                let s = (r0 - a) * 16 * pw * 4;
                if capture {
                    let count = (r1 - r0) * 16 * pw * 4;
                    let start = r0 * 16 * pw * 4;
                    raw[start..start + count].copy_from_slice(&self.read_half(&img, s, count)?);
                }
                let src: *const std::ffi::c_void = img.ptr().cast::<u16>().wrapping_add(s).cast();
                let dst: *mut u8 = rgba.ptr().cast::<u8>().wrapping_add(r0 * 16 * pw * 4);
                // SAFETY: the band's rows r0..r1 (16 picture rows each) of img, as many of rgba.
                vae_gpu_span!(
                    gpu,
                    "vae.rgba",
                    &[(r1 - r0) * 16, pw, 4],
                    ffi::check(
                        unsafe {
                            (k.to_rgba8)(gpu.raw(), src, ((r1 - r0) * 16) as i64, pw as i64, dst)
                        },
                        "to rgba",
                    )
                )?;
                nsd.wait()?;
                r0 = r1;
            }
        }
        gpu.sync()?;
        let mut out = vec![0u8; hh * ww * 4];
        #[cfg(feature = "operator-profile")]
        let started = std::time::Instant::now();
        rgba.read(0, &mut out)?;
        #[cfg(feature = "operator-profile")]
        crate::profiler::record_host(
            "vae.rgba_readback",
            &[hh, ww, 4],
            started.elapsed().as_secs_f64(),
        )?;
        Ok((out, raw, ww, hh))
    }

    /// The middle's attention: x + proj(attention(qkv(norm(x)))), one head of every channel over every pixel - as two
    /// GEMMs and a softmax (one head of 1,152 features is past the fused kernel), a block of query rows at a time.
    /// v is made transposed (W_v . norm(x)^T): the second GEMM reads it as its weights; its bias is added after
    /// (each row of weights sums to 1).
    fn attention(
        &self,
        nsd: &Nsd,
        a: &Att,
        x: DevBuf,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<DevBuf> {
        let gpu = &self.gpu;
        let k = ffi::api(&self.gpu)?;
        let px = h * w;
        let n = DevBuf::new(gpu, px * c * 2)?;
        nsd.rms_norm_mod(
            x.ptr(),
            Dt::F16,
            px,
            c,
            a.norm.ptr(),
            EPS,
            none(),
            none(),
            none(),
            n.ptr(),
            Dt::F16,
        )?;
        let wpart = |i: usize| -> *const std::ffi::c_void {
            a.qkv.ptr().cast::<u16>().wrapping_add(i * c * c).cast()
        };
        let bpart =
            |i: usize| -> *const std::ffi::c_void { a.qkv_b.fp().wrapping_add(i * c).cast() };
        let (q, kk, vt) = (
            DevBuf::new(gpu, px * c * 2)?,
            DevBuf::new(gpu, px * c * 2)?,
            DevBuf::new(gpu, c * px * 2)?,
        );
        nsd.linear(
            n.ptr(),
            Dt::F16,
            px,
            c,
            wpart(0),
            c,
            bpart(0),
            q.ptr(),
            Dt::F16,
        )?;
        nsd.linear(
            n.ptr(),
            Dt::F16,
            px,
            c,
            wpart(1),
            c,
            bpart(1),
            kk.ptr(),
            Dt::F16,
        )?;
        nsd.linear(
            wpart(2),
            Dt::F16,
            c,
            c,
            n.ptr(),
            px,
            none(),
            vt.ptr(),
            Dt::F16,
        )?;
        // query rows a block at a time: the block's scores float32 [rows, px] under ~256 MiB
        let rows_max = ((256usize << 20) / (px * 4)).clamp(1, px);
        let scores = DevBuf::f32(gpu, rows_max * px)?;
        let probs = DevBuf::new(gpu, rows_max * px * 2)?;
        let o = DevBuf::new(gpu, px * c * 2)?;
        let scale = 1.0 / (c as f32).sqrt();
        let mut r0 = 0;
        while r0 < px {
            let rows = rows_max.min(px - r0);
            let qa: *const std::ffi::c_void = q.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(
                qa,
                Dt::F16,
                rows,
                c,
                kk.ptr(),
                px,
                none(),
                scores.ptr(),
                Dt::F32,
            )?;
            // SAFETY: scores holds rows x px floats, probs as many halves.
            vae_gpu_span!(
                gpu,
                "vae.softmax",
                &[rows, px],
                ffi::check(
                    unsafe {
                        (k.softmax_rows)(
                            gpu.raw(),
                            scores.fp(),
                            rows as i64,
                            px as i64,
                            scale,
                            probs.ptr(),
                        )
                    },
                    "softmax",
                )
            )?;
            let oa: *mut std::ffi::c_void = o.ptr().cast::<u16>().wrapping_add(r0 * c).cast();
            nsd.linear(
                probs.ptr(),
                Dt::F16,
                rows,
                px,
                vt.ptr(),
                c,
                bpart(2),
                oa,
                Dt::F16,
            )?;
            r0 += rows;
        }
        nsd.linear(
            o.ptr(),
            Dt::F16,
            px,
            c,
            a.proj.ptr(),
            c,
            a.proj_b.ptr(),
            n.ptr(),
            Dt::F16,
        )?;
        nsd.gate_add(x.ptr(), Dt::F16, px, c, n.ptr(), Dt::F16, none(), none())?;
        nsd.wait()?;
        Ok(x)
    }

    /// The bytes an encode of an `hh` x `ww` picture holds at most: each down block's input (kept for its shortcut,
    /// the first resnet reads it), each resnet's input and two buffers, the stride-2 convolution's padded input and
    /// output; the middle's buffers (its attention's at the most); the latents out
    pub fn encode_bytes(&self, hh: usize, ww: usize) -> u64 {
        if validate_dimensions(hh, ww).is_err() {
            return u64::MAX;
        }
        let Some(e) = self.enc.as_ref() else { return 0 };
        let (mut h, mut w) = (hh as u64, ww as u64);
        let mut peak = h * w * 4 * 6 + h * w * e.conv_in.co as u64 * 2; // F32 upload + half input + conv output
        for d in &e.downs {
            let px = h * w;
            let kept = px * d.ci as u64 * 2;
            for (i, r) in d.resnets.iter().enumerate() {
                let (ci, cm, co) = (r.conv1.ci as u64, r.conv1.co as u64, r.conv2.co as u64);
                let input = if i == 0 { 0 } else { px * ci * 2 };
                peak = peak.max(kept + input + px * 2 * (ci.max(cm).max(co) + cm.max(co)));
            }
            if let Some(c) = &d.resample {
                peak = peak.max(
                    kept + px * c.ci as u64 * 2
                        + (h + 1) * (w + 1) * c.ci as u64 * 2
                        + (px / 4) * c.co as u64 * 2,
                );
                h /= 2;
                w /= 2;
            }
        }
        let (px, c) = (h * w, e.mid[0].conv2.co as u64);
        let rows = ((256u64 << 20) / (px * 4)).clamp(1, px);
        peak.max(6 * px * c * 2 + rows * px * 6) + px * e.quant.co as u64 * 4 + SLACK
    }

    /// Normalized latents float32 [h * w, z] (h = H / 16) of a picture: RGBA float32 [H, W, 4] in [-1, 1]
    pub fn encode(&self, rgba: &[f32], hh: usize, ww: usize) -> Result<Vec<f32>> {
        validate_dimensions(hh, ww)?;
        if rgba.len() != hh * ww * 4 || rgba.iter().any(|v| !v.is_finite() || *v < -1.0 || *v > 1.0)
        {
            return Err(Error("VAE encode requires finite RGBA in [-1,1]".into()));
        }
        if self.resident_bytes + self.encode_bytes(hh, ww) > self.allowance {
            return Err(Error("VAE encode exceeds device allowance".into()));
        }
        let nsd = &self.backend;
        use std::sync::atomic::Ordering;
        self.encoding.store(true, Ordering::Relaxed);
        let r = self.encode_inner(nsd, rgba, hh, ww);
        self.encoding.store(false, Ordering::Relaxed);
        r
    }

    fn encode_inner(&self, nsd: &Nsd, rgba: &[f32], hh: usize, ww: usize) -> Result<Vec<f32>> {
        let e = self
            .enc
            .as_ref()
            .ok_or_else(|| Error("this VAE file has no encoder (edits need it)".into()))?;
        let gpu = &self.gpu;
        let k = ffi::api(&self.gpu)?;
        let mut x = {
            let xf = DevBuf::from_f32(gpu, rgba)?;
            let xh = DevBuf::new(gpu, rgba.len() * 2)?;
            // SAFETY: H x W x 4 values each way.
            ffi::check(
                unsafe { (k.to_half)(gpu.raw(), xf.fp(), rgba.len() as i64, xh.ptr()) },
                "to half",
            )?;
            let x = self.conv(nsd, &e.conv_in, &xh, hh, ww)?;
            // the picture's buffers are freed here: the work on them first
            nsd.wait()?;
            x
        };
        let (mut h, mut w) = (hh, ww);
        for d in &e.downs {
            // the block's input, kept for its shortcut (the first resnet reads it and leaves it as it is: no copy)
            let keep = x;
            x = self.resnet(nsd, &d.resnets[0], &keep, h, w)?;
            for r in &d.resnets[1..] {
                x = self.resnet(nsd, r, &x, h, w)?;
            }
            match &d.resample {
                Some(c) => {
                    let p = DevBuf::new(gpu, (h + 1) * (w + 1) * c.ci * 2)?;
                    // SAFETY: x holds h x w x ci halves, p (h + 1) x (w + 1) x ci.
                    ffi::check(
                        unsafe {
                            (k.pad_br)(gpu.raw(), x.ptr(), h as i64, w as i64, c.ci as i64, p.ptr())
                        },
                        "pad",
                    )?;
                    let out = DevBuf::new(gpu, (h / 2) * (w / 2) * c.co * 2)?;
                    nsd.conv3d_ex(
                        p.ptr(),
                        Dt::F16,
                        (1, h + 1, w + 1),
                        c.ci,
                        c.w.ptr(),
                        c.co,
                        (1, 3, 3),
                        (1, 2, 2),
                        c.b.ptr(),
                        out.ptr(),
                    )?;
                    // SAFETY: keep holds h x w x ci halves, out (h / 2) x (w / 2) x co.
                    ffi::check(
                        unsafe {
                            (k.avg_down_add)(
                                gpu.raw(),
                                keep.ptr(),
                                h as i64,
                                w as i64,
                                d.ci as i64,
                                d.co as i64,
                                d.ft,
                                out.ptr(),
                            )
                        },
                        "shortcut",
                    )?;
                    nsd.wait()?;
                    x = out;
                    h /= 2;
                    w /= 2;
                }
                None => {
                    // the shortcut at one-to-one: the block's input
                    nsd.gate_add(
                        x.ptr(),
                        Dt::F16,
                        h * w,
                        d.co,
                        keep.ptr(),
                        Dt::F16,
                        none(),
                        none(),
                    )?;
                    nsd.wait()?;
                }
            }
        }
        x = self.resnet(nsd, &e.mid[0], &x, h, w)?;
        x = self.attention(nsd, &e.att, x, h, w, e.mid[0].conv2.co)?;
        x = self.resnet(nsd, &e.mid[1], &x, h, w)?;
        let a = self.norm_silu(nsd, &x, h * w, e.head.ci, &e.out_norm)?;
        x = self.conv(nsd, &e.head, &a, h, w)?;
        x = self.conv(nsd, &e.quant, &x, h, w)?;
        let zc = e.quant.co;
        let f = DevBuf::f32(gpu, h * w * zc)?;
        // SAFETY: h x w x zc values each way.
        ffi::check(
            unsafe { (k.to_float)(gpu.raw(), x.ptr(), (h * w * zc) as i64, f.fp()) },
            "to float",
        )?;
        gpu.sync()?;
        let all = f.to_f32()?;
        let z = self.z;
        Ok((0..h * w)
            .flat_map(|p| (0..z).map(move |c| (p, c)))
            .map(|(p, c)| (all[p * zc + c] - self.mean[c]) / self.std[c])
            .collect())
    }
}

#[cfg(test)]
mod conv64_route_tests {
    use super::ConvKernel;
    #[test]
    fn explicit_profile_shapes_and_fallback() {
        let k = ConvKernel::Register64Measured1024;
        for (h, w, ci, co, ks) in [
            (64, 64, 64, 64, 1),
            (64, 64, 64, 1152, 3),
            (64, 64, 1152, 1152, 3),
            (128, 128, 1152, 1152, 3),
            (256, 256, 576, 576, 3),
            (256, 256, 1152, 576, 1),
            (256, 256, 1152, 576, 3),
            (256, 256, 1152, 1152, 3),
            (512, 512, 288, 288, 3),
            (512, 512, 576, 288, 1),
            (512, 512, 576, 288, 3),
            (512, 512, 576, 576, 3),
            (1024, 1024, 144, 4, 3),
            (1024, 1024, 144, 144, 3),
            (1024, 1024, 288, 144, 1),
            (1024, 1024, 288, 144, 3),
            (1024, 1024, 288, 288, 3),
        ] {
            assert!(k.uses_register64(h, w, ci, co, ks));
            assert!(!ConvKernel::Baseline.uses_register64(h, w, ci, co, ks));
            assert!(!k.uses_register64(h - 1, w, ci, co, ks));
            assert!(!k.uses_register64(h, w, ci, co, 2));
        }
    }
}
