//! The engine's kernels (ffi.rs) as checked calls on device pointers, and its matrices: half or int8 (a scale per
//! row), read from the checkpoint's safetensors (bf16 / f32) and converted on the GPU. (The MiniMax Music engine's
//! ops, as Qwen3-TTS's, with this engine's kernels beside them.)
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]

use std::ffi::c_void;
use std::path::PathBuf;
use std::sync::Arc;

use nextrocm_audio::{Error, LoadOptions, Result};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::safetensors::{SafeTensors, StTensor};
use nextrocm_ops::kernels::{Dt, Nsd, none};

use crate::ffi;
#[path = "conv_up.rs"]
mod conv_up;
#[path = "linear_f16.rs"]
mod linear_f16;
#[path = "profile.rs"]
mod profile;

type P = *const c_void;
type M = *mut c_void;

/// A float pointer `at` floats into a buffer
pub fn fp(b: &DevBuf, at: usize) -> *mut f32 {
    b.fp().wrapping_add(at)
}

/// The engine's kernels on one GPU
pub struct Ops {
    pub gpu: Arc<Gpu>,
    k: &'static ffi::Api,
    profile: Option<profile::Profile>,
    f32_rows: bool,
    conv_up_tiled: bool,
    f32_kv: bool,
    rope_bf16_cache: bool,
    linear_f16_tiled: bool,
    flow_f16_tiled: bool,
}

impl Ops {
    pub fn new(gpu: &Arc<Gpu>, options: &LoadOptions) -> Result<Ops> {
        Ok(Ops {
            gpu: gpu.clone(),
            k: ffi::api()?,
            profile: profile::Profile::enabled(),
            rope_bf16_cache: load_control(options, "NEXTROCM_VCP_ROPE_BF16_CACHE")?,
            linear_f16_tiled: load_control(options, "NS_VCP_LINEAR_F16_TILED")?,
            flow_f16_tiled: load_control(options, "NS_VCP_FLOW_F16_TILED")?,
            f32_kv: load_control(options, "NEXTROCM_VCP_F32_KV")?,
            conv_up_tiled: load_control(options, "NS_VCP_CONV_UP_TILED")?,
            f32_rows: load_control(options, "NEXTROCM_VCP_F32_ROWS")?,
        })
    }

    pub(crate) fn controls_report(&self) -> String {
        format!(
            "VoxCPM2 effective controls: F32_ROWS={} F32_KV={} ROPE_BF16_CACHE={} CONV_UP_TILED={} LINEAR_F16_TILED={} FLOW_F16_TILED={}",
            u8::from(self.f32_rows),
            u8::from(self.f32_kv),
            u8::from(self.rope_bf16_cache),
            u8::from(self.conv_up_tiled),
            u8::from(self.linear_f16_tiled),
            u8::from(self.flow_f16_tiled)
        )
    }

    /// KV allocation width follows the immutable per-engine precision setting.
    pub fn kv_element_bytes(&self) -> usize {
        if self.f32_kv { 4 } else { 2 }
    }

    pub fn stage(&self, name: &'static str) -> Option<profile::Stage> {
        self.profile.as_ref().map(|_| profile::Stage::enter(name))
    }
    pub fn profiled<T>(
        &self,
        op: &'static str,
        shape: &[usize],
        run: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        match &self.profile {
            Some(p) => p.measure(&self.gpu, op, shape, run),
            None => run(),
        }
    }
    fn g(&self) -> M {
        self.gpu.raw()
    }

    /// out[m] (+)= x[m] . W^T (* scale + bias), m < M <= 8, out rows `ldo` apart
    pub fn gemv(
        &self,
        x: *const f32,
        m: usize,
        k: usize,
        w: &Mat,
        bias: P,
        out: *mut f32,
        ldo: usize,
        acc: bool,
    ) -> Result<()> {
        self.profiled(
            "gemv",
            &[m, w.n, k, usize::from(w.scale.is_some()), ldo, acc as usize],
            || {
                let (wt, sc) = match &w.scale {
                    None => (0, std::ptr::null()),
                    Some(s) => (1, s.fp() as *const f32),
                };
                // SAFETY: the caller's buffers hold the sizes named.
                ffi::check(
                    unsafe {
                        (self.k.gemv)(
                            self.g(),
                            x,
                            m as i64,
                            k as i64,
                            w.w.ptr(),
                            wt,
                            sc,
                            w.n as i64,
                            bias.cast(),
                            out,
                            ldo as i64,
                            acc as i32,
                        )
                    },
                    "gemv",
                )
            },
        )
    }

    /// Floats of scratch `attn` needs
    pub fn attn_scratch(&self, b: usize, s: usize, hq: usize, d: usize, p0: usize) -> usize {
        // SAFETY: arithmetic only.
        unsafe {
            (self.k.attn_scratch)(b as i64, s as i64, hq as i64, d as i64, p0 as i64) as usize
        }
    }

    pub fn attn(
        &self,
        q: *const f32,
        qs: usize,
        kc: &DevBuf,
        vc: &DevBuf,
        b: usize,
        s: usize,
        hq: usize,
        hkv: usize,
        d: usize,
        t: usize,
        p0: usize,
        out: *mut f32,
        part: *mut f32,
    ) -> Result<()> {
        let kernel = if self.f32_kv {
            self.k
                .attn_f32
                .ok_or_else(|| Error("F32 KV requires rebuilt audio attention kernel".into()))?
        } else {
            self.k.attn
        };
        self.profiled(
            if self.f32_kv { "attn_f32_kv" } else { "attn" },
            &[b, s, hq, hkv, d, t, p0],
            || {
                // SAFETY: caller buffers hold the named sizes; cache [b,hkv,t,d] uses
                // kv_element_bytes(), matching the selected kernel for this engine.
                ffi::check(
                    unsafe {
                        kernel(
                            self.g(),
                            q,
                            qs as i64,
                            kc.ptr(),
                            vc.ptr(),
                            b as i64,
                            s as i64,
                            hq as i64,
                            hkv as i64,
                            d as i64,
                            t as i64,
                            p0 as i64,
                            out,
                            part,
                        )
                    },
                    "attention",
                )
            },
        )
    }

    pub fn kv_store(
        &self,
        k: *const f32,
        v: *const f32,
        ks: usize,
        b: usize,
        s: usize,
        hkv: usize,
        d: usize,
        t: usize,
        p0: usize,
        kc: &DevBuf,
        vc: &DevBuf,
    ) -> Result<()> {
        let kernel = if self.f32_kv {
            self.k
                .kv_store_f32
                .ok_or_else(|| Error("F32 KV requires rebuilt audio store kernel".into()))?
        } else {
            self.k.kv_store
        };
        self.profiled(
            if self.f32_kv {
                "kv_store_f32"
            } else {
                "kv_store"
            },
            &[b, s, hkv, d, t, p0],
            || {
                // SAFETY: as attn.
                ffi::check(
                    unsafe {
                        kernel(
                            self.g(),
                            k,
                            v,
                            ks as i64,
                            b as i64,
                            s as i64,
                            hkv as i64,
                            d as i64,
                            t as i64,
                            p0 as i64,
                            kc.ptr(),
                            vc.ptr(),
                        )
                    },
                    "kv store",
                )
            },
        )
    }

    pub fn to_half(&self, x: *const f32, out: M, n: usize) -> Result<()> {
        self.profiled("to_half", &[n], || {
            // SAFETY: n values each.
            ffi::check(
                unsafe { (self.k.to_half)(self.g(), x, out, n as i64) },
                "to half",
            )
        })
    }

    pub fn bf16_to_float(&self, x: P, out: *mut f32, n: usize) -> Result<()> {
        self.profiled("bf16_to_float", &[n], || {
            // SAFETY: n values each.
            ffi::check(
                unsafe { (self.k.bf16_to_float)(self.g(), x.cast(), out, n as i64) },
                "bf16 to float",
            )
        })
    }

    pub fn quant_rows(
        &self,
        w: *const f32,
        n: usize,
        k: usize,
        q: M,
        scale: *mut f32,
    ) -> Result<()> {
        self.profiled("quant_rows", &[n, k], || {
            // SAFETY: w [n, k] floats, q [n, k] bytes, scale n floats.
            ffi::check(
                unsafe { (self.k.quant_rows)(self.g(), w, n as i64, k as i64, q.cast(), scale) },
                "int8 rows",
            )
        })
    }

    pub fn dequant_rows(&self, q: P, scale: *const f32, n: usize, k: usize, out: M) -> Result<()> {
        self.profiled("dequant_rows", &[n, k], || {
            // SAFETY: q [n, k] bytes, scale n floats, out [n, k] halfs.
            ffi::check(
                unsafe {
                    (self.k.dequant_rows)(self.g(), q.cast(), scale, n as i64, k as i64, out)
                },
                "int8 rows to half",
            )
        })
    }

    pub fn add(&self, x: *mut f32, y: *const f32, n: usize) -> Result<()> {
        self.profiled("add", &[n], || {
            // SAFETY: n floats each.
            ffi::check(unsafe { (self.k.add)(self.g(), x, y, n as i64) }, "add")
        })
    }

    pub fn transpose(&self, x: *const f32, r: usize, c: usize, out: *mut f32) -> Result<()> {
        self.profiled("transpose", &[r, c], || {
            // SAFETY: r x c floats each.
            ffi::check(
                unsafe { (self.k.transpose)(self.g(), x, r as i64, c as i64, out) },
                "transpose",
            )
        })
    }

    /// A causal 1-D convolution on native HIP: x [b, ci, l] -> out [b, co, l], w [co, ci, k] ((k - 1) * dil zeros ahead;
    /// the weight kept alive: its reordered copy is cached by its pointer)
    pub fn conv_causal(
        &self,
        x: *const f32,
        b: usize,
        ci: usize,
        l: usize,
        w: &DevBuf,
        co: usize,
        k: usize,
        bias: *const f32,
        dil: usize,
        out: *mut f32,
    ) -> Result<()> {
        self.profiled("conv_causal", &[b, ci, l, co, k, dil], || {
            // SAFETY: the caller's buffers hold the sizes named.
            ffi::check(
                unsafe {
                    (self.k.conv1d_lr)(
                        self.g(),
                        0,
                        x,
                        b as i64,
                        ci as i64,
                        l as i64,
                        w.fp(),
                        co as i64,
                        k as i64,
                        bias,
                        1,
                        dil as i64,
                        ((k - 1) * dil) as i64,
                        0,
                        out,
                        l as i64,
                    )
                },
                "conv1d",
            )
        })
    }

    /// A transposed one, its end trimmed to l * stride: w [ci, co, k]
    pub fn conv_up(
        &self,
        x: *const f32,
        b: usize,
        ci: usize,
        l: usize,
        w: &DevBuf,
        co: usize,
        k: usize,
        bias: *const f32,
        stride: usize,
        out: *mut f32,
    ) -> Result<()> {
        if self.conv_up_tiled && conv_up::supported(b, ci, l, co, k, stride) {
            let kernel = self.k.conv_up_tiled.ok_or_else(|| {
                Error("NS_VCP_CONV_UP_TILED=1 requires the rebuilt audio kernel library".into())
            })?;
            return self.profiled("conv_up_tiled", &[b, ci, l, co, k, stride], || {
                // SAFETY: same contiguous F32 buffers as the baseline below;
                // supported bounds all dimensions/spans before i64 casts. The
                // wrapper also checks pointers, alignment and source overlap.
                ffi::check(
                    unsafe {
                        kernel(
                            self.g(),
                            x,
                            b as i64,
                            ci as i64,
                            l as i64,
                            w.fp(),
                            co as i64,
                            k as i64,
                            bias,
                            stride as i64,
                            out,
                        )
                    },
                    "conv_transpose1d_tiled",
                )
            });
        }
        self.profiled("conv_up", &[b, ci, l, co, k, stride], || {
            // SAFETY: as conv_causal.
            ffi::check(
                unsafe {
                    (self.k.conv1d_lr)(
                        self.g(),
                        1,
                        x,
                        b as i64,
                        ci as i64,
                        l as i64,
                        w.fp(),
                        co as i64,
                        k as i64,
                        bias,
                        stride as i64,
                        1,
                        0,
                        (k - stride) as i64,
                        out,
                        (l * stride) as i64,
                    )
                },
                "conv_transpose1d",
            )
        })
    }

    pub fn tanh(&self, x: *mut f32, n: usize) -> Result<()> {
        self.profiled("tanh", &[n], || {
            // SAFETY: n floats.
            ffi::check(unsafe { (self.k.tanh)(self.g(), x, n as i64) }, "tanh")
        })
    }

    /// Rotary positions from a frequency table: row r at p0 + r % s
    pub fn rope(
        &self,
        x: *mut f32,
        xs: usize,
        rows: usize,
        s: usize,
        h: usize,
        d: usize,
        inv_freq: &DevBuf,
        p0: usize,
    ) -> Result<()> {
        let kernel = if self.rope_bf16_cache {
            self.k
                .rope_bf16_cache
                .ok_or_else(|| Error("BF16 rotary cache requires rebuilt audio kernel".into()))?
        } else {
            self.k.rope
        };
        self.profiled("rope", &[rows, s, h, d, p0, xs], || {
            // SAFETY: rows of h x d floats xs apart; inv_freq d / 2 floats.
            ffi::check(
                unsafe {
                    kernel(
                        self.g(),
                        x,
                        xs as i64,
                        rows as i64,
                        s as i64,
                        h as i64,
                        d as i64,
                        inv_freq.fp(),
                        p0 as i64,
                    )
                },
                "rope",
            )
        })
    }

    /// Attention within b groups of s rows (none masked)
    pub fn local_attn(
        &self,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        b: usize,
        s: usize,
        hq: usize,
        hkv: usize,
        d: usize,
        qs: usize,
        kvs: usize,
        out: *mut f32,
    ) -> Result<()> {
        self.profiled("local_attn", &[b, s, hq, hkv, d], || {
            // SAFETY: the rows the caller names.
            ffi::check(
                unsafe {
                    (self.k.local_attn)(
                        self.g(),
                        q,
                        k,
                        v,
                        b as i64,
                        s as i64,
                        hq as i64,
                        hkv as i64,
                        d as i64,
                        qs as i64,
                        kvs as i64,
                        out,
                    )
                },
                "local attention",
            )
        })
    }

    pub fn snake(
        &self,
        x: *const f32,
        c: usize,
        l: usize,
        alpha: &DevBuf,
        out: *mut f32,
    ) -> Result<()> {
        self.profiled("snake", &[c, l], || {
            // SAFETY: x and out [c, l], alpha c floats.
            ffi::check(
                unsafe { (self.k.snake)(self.g(), x, c as i64, l as i64, alpha.fp(), out) },
                "snake",
            )
        })
    }

    /// A causal depthwise convolution, dilation `dil`
    pub fn dwconv(
        &self,
        x: *const f32,
        c: usize,
        l: usize,
        w: &DevBuf,
        bias: *const f32,
        k: usize,
        dil: usize,
        out: *mut f32,
    ) -> Result<()> {
        self.profiled("dwconv", &[c, l, k, dil], || {
            // SAFETY: x and out [c, l], w [c, k], bias c floats or null.
            ffi::check(
                unsafe {
                    (self.k.dwconv)(
                        self.g(),
                        x,
                        c as i64,
                        l as i64,
                        w.fp(),
                        bias,
                        k as i64,
                        dil as i64,
                        out,
                    )
                },
                "depthwise conv",
            )
        })
    }

    pub fn affine(
        &self,
        x: *mut f32,
        c: usize,
        l: usize,
        scale: *const f32,
        bias: *const f32,
    ) -> Result<()> {
        self.profiled("affine", &[c, l], || {
            // SAFETY: x [c, l], scale and bias c floats.
            ffi::check(
                unsafe { (self.k.affine)(self.g(), x, c as i64, l as i64, scale, bias) },
                "channel affine",
            )
        })
    }

    pub fn fsq(&self, x: *mut f32, n: usize, s: f32) -> Result<()> {
        self.profiled("fsq", &[n], || {
            // SAFETY: n floats.
            ffi::check(
                unsafe { (self.k.fsq)(self.g(), x, n as i64, s) },
                "scalar quantize",
            )
        })
    }

    pub fn silu(&self, x: *mut f32, n: usize) -> Result<()> {
        self.profiled("silu", &[n], || {
            // SAFETY: n floats.
            ffi::check(unsafe { (self.k.silu)(self.g(), x, n as i64) }, "silu")
        })
    }

    /// A strided causal convolution as Mimi's: (k - stride) zeros ahead, behind as many as the last window needs;
    /// out [b, co, ceil(l / stride)]; its length returned
    pub fn conv_strided(
        &self,
        x: *const f32,
        b: usize,
        ci: usize,
        l: usize,
        w: &DevBuf,
        co: usize,
        k: usize,
        bias: *const f32,
        stride: usize,
        out: *mut f32,
    ) -> Result<usize> {
        self.profiled("conv_strided", &[b, ci, l, co, k, stride], || {
            let pt = k - stride;
            let frames = (l + pt).saturating_sub(k).div_ceil(stride);
            let extra = frames * stride + k - pt - l;
            let lo = frames + 1;
            // SAFETY: as conv_causal.
            ffi::check(
                unsafe {
                    (self.k.conv1d_lr)(
                        self.g(),
                        0,
                        x,
                        b as i64,
                        ci as i64,
                        l as i64,
                        w.fp(),
                        co as i64,
                        k as i64,
                        bias,
                        stride as i64,
                        1,
                        pt as i64,
                        extra as i64,
                        out,
                        lo as i64,
                    )
                },
                "conv1d",
            )?;
            Ok(lo)
        })
    }
}

/// A matrix [n, k] on the GPU: half, or int8 with a scale per row
pub struct Mat {
    pub w: DevBuf,
    pub scale: Option<DevBuf>,
    pub n: usize,
    pub k: usize,
}

/// Explicit call-site scope; only the DiT transformer opts into Flow.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowScope {
    Original,
    Flow,
}

impl Mat {
    /// out [m, n] f32 = x [m, k] f32 . W^T + bias: a few rows by the engine's own product (bound by reading W once,
    /// int8 weights against float activations), more by the shared half GEMM (x converted into `xh`, a half buffer of
    /// m * k; an int8 matrix first expanded to half in `wh`, n * k halfs - quantizing the activations too, oneDNN's
    /// int8 path, loses Qwen3's outlier features)
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        ops: &Ops,
        nsd: &Nsd,
        x: *const f32,
        m: usize,
        xh: Option<&DevBuf>,
        wh: Option<&DevBuf>,
        bias: P,
        out: *mut f32,
    ) -> Result<()> {
        self.apply_scoped(ops, nsd, x, m, xh, wh, bias, out, RowScope::Original)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_scoped(
        &self,
        ops: &Ops,
        nsd: &Nsd,
        x: *const f32,
        m: usize,
        xh: Option<&DevBuf>,
        wh: Option<&DevBuf>,
        bias: P,
        out: *mut f32,
        scope: RowScope,
    ) -> Result<()> {
        if m <= 8 {
            return ops.gemv(x, m, self.k, self, bias, out, self.n, false);
        }
        // Diagnostic precision control: preserve F32 activations and avoid
        // rounding dequantized int8 weights into half staging. Same stored
        // weights, bias and row order; each existing kernel takes <=8 rows.
        // Opt-in only until the frozen gates and unprofiled timing are measured.
        // Only supported flow shapes may bypass F32 rows. All other call
        // sites, small GEMVs and unsupported shapes keep their original policy.
        let flow_tiled = scope == RowScope::Flow
            && ops.flow_f16_tiled
            && linear_f16::supported(m, self.k, self.n);
        if ops.f32_rows && !flow_tiled {
            for row in (0..m).step_by(8) {
                ops.gemv(
                    x.wrapping_add(row * self.k),
                    (m - row).min(8),
                    self.k,
                    self,
                    bias,
                    out.wrapping_add(row * self.n),
                    self.n,
                    false,
                )?;
            }
            return Ok(());
        }
        let xh = xh.ok_or_else(|| Error("a half staging buffer for a wide product".into()))?;
        ops.to_half(x, xh.ptr(), m * self.k)?;
        let w = match &self.scale {
            None => self.w.ptr(),
            Some(s) => {
                let wh =
                    wh.ok_or_else(|| Error("a half matrix buffer for a wide int8 product".into()))?;
                ops.dequant_rows(self.w.ptr(), s.fp(), self.n, self.k, wh.ptr())?;
                wh.ptr()
            }
        };
        if (ops.linear_f16_tiled || flow_tiled) && linear_f16::supported(m, self.k, self.n) {
            let kernel = ops.k.linear_f16_tiled.ok_or_else(|| {
                Error("selected F16 tiled linear requires rebuilt audio kernel".into())
            })?;
            return ops.profiled("linear_f16_tiled", &[m, self.n, self.k], || {
                // SAFETY: original half staging above supplies contiguous F16
                // operands; supported bounds spans and the wrapper checks aliasing.
                ffi::check(
                    unsafe {
                        kernel(
                            ops.g(),
                            xh.ptr(),
                            m as i64,
                            self.k as i64,
                            w,
                            self.n as i64,
                            bias.cast(),
                            out,
                        )
                    },
                    "linear_f16_tiled",
                )
            });
        }
        ops.profiled("linear_f16", &[m, self.n, self.k], || {
            nsd.linear(
                xh.ptr(),
                Dt::F16,
                m,
                self.k,
                w,
                self.n,
                bias,
                out.cast(),
                Dt::F32,
            )
        })
    }

    pub fn bytes(&self) -> usize {
        self.w.len + self.scale.as_ref().map_or(0, |s| s.len)
    }
}

/// A model's safetensors shards, its tensors found by name across them
pub struct Shards {
    files: Vec<SafeTensors>,
}

impl Shards {
    pub fn open(paths: &[PathBuf]) -> Result<Shards> {
        if paths.is_empty() {
            return Err(Error("no files".into()));
        }
        let files = paths
            .iter()
            .map(|p| SafeTensors::open(p).map_err(|e| Error(e.0)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Shards { files })
    }

    /// Conservative resident bytes plus the largest load/conversion workspace.
    /// Embedding rows remain on disk; only base/residual LM matrices use int8.
    pub fn load_upper_bound(&self, int8: bool) -> Result<usize> {
        let mut resident = 0usize;
        let mut temporary = 0usize;
        for f in &self.files {
            for t in &f.tensors {
                elem(t)?;
                if t.name == "base_lm.embed_tokens.weight" {
                    continue;
                }
                let dims = t
                    .shape
                    .iter()
                    .map(|d| {
                        usize::try_from(*d).map_err(|_| Error("tensor dimension overflow".into()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let n = crate::budget::product(&dims)?;
                let quant = int8
                    && (t.name.starts_with("base_lm.layers.")
                        || t.name.starts_with("residual_lm.layers."))
                    && t.name.ends_with(".weight")
                    && dims.len() == 2;
                // Precisely the Mat/Proj banks in lm.rs/model.rs; Lin, norms,
                // biases and unknown tensors remain conservatively float32.
                let half = dims.len() == 2
                    && t.name.ends_with(".weight")
                    && (t.name.contains(".layers.")
                        || matches!(
                            t.name.as_str(),
                            "enc_to_lm_proj.weight"
                                | "lm_to_dit_proj.weight"
                                | "res_to_dit_proj.weight"
                                | "fusion_concat_proj.weight"
                                | "fsq_layer.in_proj.weight"
                                | "fsq_layer.out_proj.weight"
                                | "stop_proj.weight"
                                | "stop_head.weight"
                                | "feat_decoder.estimator.out_proj.weight"
                                | "feat_decoder.estimator.time_mlp.linear_1.weight"
                                | "feat_decoder.estimator.time_mlp.linear_2.weight"
                                | "feat_decoder.estimator.delta_time_mlp.linear_1.weight"
                                | "feat_decoder.estimator.delta_time_mlp.linear_2.weight"
                        ));
                let bytes = if quant {
                    crate::budget::sum(&[n, crate::budget::product(&[dims[0], 4])?])?
                } else {
                    crate::budget::product(&[n, if half { 2 } else { 4 }])?
                };
                resident = crate::budget::sum(&[resident, bytes])?;
                temporary = temporary.max(crate::budget::product(&[n, 8])?);
            }
        }
        crate::budget::sum(&[resident, temporary])
    }

    pub fn find(&self, name: &str) -> Result<(&SafeTensors, &StTensor)> {
        self.files
            .iter()
            .find_map(|f| f.tensor(name).map(|t| (f, t)))
            .ok_or_else(|| {
                Error(format!(
                    "{}: no tensor {name}",
                    self.files[0]
                        .path
                        .parent()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ))
            })
    }

    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self
            .find(name)?
            .1
            .shape
            .iter()
            .map(|d| *d as usize)
            .collect())
    }

    /// A tensor as float32 on the host
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (f, _) = self.find(name)?;
        f.f32(name).map_err(|e| Error(e.0))
    }

    /// Rows [r0, r1) of a 2-D (or more: rows of the rest) bf16 / f32 tensor as float32 on the host
    pub fn rows_f32(&self, name: &str, r0: usize, r1: usize) -> Result<Vec<f32>> {
        let (f, t) = self.find(name)?;
        let row: usize = t.shape.iter().skip(1).product::<u64>() as usize;
        let es = elem(t)?;
        let mut b = vec![0u8; (r1 - r0) * row * es];
        f.read_into(t, (r0 * row * es) as u64, &mut b)
            .map_err(|e| Error(e.0))?;
        Ok(match t.dtype.as_str() {
            "BF16" => b
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            _ => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }

    /// Rows [r0, r1) of a tensor onto the GPU as float32 at `dst` (f32 offset `at`), through `stage` (bytes, at least
    /// the rows' size)
    fn rows_to(
        &self,
        ops: &Ops,
        name: &str,
        r0: usize,
        r1: usize,
        stage: &DevBuf,
        dst: &DevBuf,
        at: usize,
    ) -> Result<usize> {
        let (f, t) = self.find(name)?;
        let row: usize = t.shape.iter().skip(1).product::<u64>() as usize;
        let es = elem(t)?;
        let n = (r1 - r0) * row;
        let mut b = vec![0u8; n * es];
        f.read_into(t, (r0 * row * es) as u64, &mut b)
            .map_err(|e| Error(e.0))?;
        if es == 4 {
            dst.write(at * 4, &b)?;
        } else {
            stage.write(0, &b)?;
            ops.bf16_to_float(stage.ptr(), fp(dst, at), n)?;
            ops.gpu.sync()?;
        }
        Ok(n)
    }

    /// A tensor as float32 on the GPU
    pub fn dev_f32(&self, ops: &Ops, name: &str) -> Result<DevBuf> {
        let (_, t) = self.find(name)?;
        let n = t.elements() as usize;
        let out = DevBuf::f32(&ops.gpu, n)?;
        let stage = DevBuf::new(&ops.gpu, n * 2)?;
        self.rows_to(
            ops,
            name,
            0,
            t.shape.first().copied().unwrap_or(1) as usize,
            &stage,
            &out,
            0,
        )?;
        Ok(out)
    }

    /// A matrix from row ranges of tensors, stacked: [(name, r0, r1)], every part `k` wide; half, or int8 rows
    pub fn mat(&self, ops: &Ops, parts: &[(&str, usize, usize)], int8: bool) -> Result<Mat> {
        let mut k = 0;
        let mut n = 0;
        let mut biggest = 0;
        for (name, r0, r1) in parts {
            let s = self.shape(name)?;
            let kk: usize = s.iter().skip(1).product();
            if k != 0 && kk != k {
                return Err(Error(format!("{name}: {kk} wide, the matrix {k}")));
            }
            if *r1 > s[0] || r0 >= r1 {
                return Err(Error(format!("{name}: rows {r0}..{r1} of {}", s[0])));
            }
            k = kk;
            n += r1 - r0;
            biggest = biggest.max((r1 - r0) * kk);
        }
        let gpu = &ops.gpu;
        let w = DevBuf::new(gpu, n * k * if int8 { 1 } else { 2 })?;
        let scale = if int8 {
            Some(DevBuf::f32(gpu, n)?)
        } else {
            None
        };
        let stage = DevBuf::new(gpu, biggest * 2)?;
        let f = DevBuf::f32(gpu, biggest)?;
        let mut row = 0;
        for (name, r0, r1) in parts {
            let cnt = self.rows_to(ops, name, *r0, *r1, &stage, &f, 0)?;
            match &scale {
                None => ops.to_half(f.fp(), w.ptr().wrapping_byte_add(row * k * 2), cnt)?,
                Some(s) => ops.quant_rows(
                    f.fp(),
                    r1 - r0,
                    k,
                    w.ptr().wrapping_byte_add(row * k),
                    fp(s, row),
                )?,
            }
            ops.gpu.sync()?;
            row += r1 - r0;
        }
        Ok(Mat { w, scale, n, k })
    }

    /// A whole tensor as a matrix
    pub fn mat1(&self, ops: &Ops, name: &str, int8: bool) -> Result<Mat> {
        let r = self.shape(name)?[0];
        self.mat(ops, &[(name, 0, r)], int8)
    }
}

fn elem(t: &StTensor) -> Result<usize> {
    match t.dtype.as_str() {
        "BF16" => Ok(2),
        "F32" => Ok(4),
        d => Err(Error(format!("{}: {d}, not bf16 or f32", t.name))),
    }
}

/// A null pointer (an absent bias)
pub fn nul() -> P {
    none()
}

// Registry/card settings take precedence over process environment, using the
// original audio contract. No process-wide environment mutation at model load.
fn load_control(options: &LoadOptions, name: &str) -> Result<bool> {
    match options.setting(name).as_deref() {
        Some("1") => Ok(true),
        Some("0") | None => Ok(false),
        Some(_) => Err(Error(format!("{name} must be 0 or 1"))),
    }
}

#[cfg(test)]
mod control_tests {
    use super::*;
    #[test]
    fn card_precision_controls_parse_without_gpu_or_environment_mutation() {
        let mut options = LoadOptions::default();
        for name in [
            "NEXTROCM_VCP_F32_ROWS",
            "NEXTROCM_VCP_F32_KV",
            "NEXTROCM_VCP_ROPE_BF16_CACHE",
            "NS_VCP_CONV_UP_TILED",
            "NS_VCP_LINEAR_F16_TILED",
            "NS_VCP_FLOW_F16_TILED",
        ] {
            options.settings.insert(name.into(), "1".into());
            assert!(load_control(&options, name).unwrap());
            options.settings.insert(name.into(), "0".into());
            assert!(!load_control(&options, name).unwrap());
            options.settings.insert(name.into(), "invalid".into());
            assert!(load_control(&options, name).is_err());
        }
    }
}
