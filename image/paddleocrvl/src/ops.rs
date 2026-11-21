//! Bind only the existing shared operations used by this engine; keep missing
//! BF16/RoPE and larger-page attention in the private PaddleOCR ABI.
#![allow(clippy::too_many_arguments, clippy::not_unsafe_ptr_arg_deref)]
use crate::ffi;
use nextrocm_engine_core::{Error, Gpu, Result};
pub use nextrocm_ops::kernels::{none, Dt};
use std::{ffi::c_void, ops::Deref, sync::Arc};
type P = *const c_void;
type M = *mut c_void;
pub struct Nsd(
    nextrocm_ops::kernels::Nsd,
    pub(crate) crate::settings::RuntimeSettings,
);
impl Deref for Nsd {
    type Target = nextrocm_ops::kernels::Nsd;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Nsd {
    /// Decoder-only precision candidate; the shared vision Linear is unchanged.
    pub fn decoder_linear(
        &self,
        x: P,
        dt: Dt,
        rows: usize,
        width: usize,
        weight: P,
        outputs: usize,
        bias: P,
        out: M,
        out_dt: Dt,
    ) -> Result<()> {
        if !self.1.linear8 {
            return self
                .0
                .linear(x, dt, rows, width, weight, outputs, bias, out, out_dt);
        }
        if dt != Dt::F16 || out_dt != Dt::F32 || !bias.is_null() {
            return Err(Error(
                "PaddleOCR decoder Linear requires unbiased F16 to F32".into(),
            ));
        }
        // SAFETY: callers own the checked dense spans; the private ABI rejects
        // invalid geometry before launching on this GPU's stream.
        ffi::check(
            unsafe {
                (ffi::api()?.linear)(
                    self.gpu.raw(),
                    x,
                    rows as i64,
                    width as i64,
                    weight,
                    outputs as i64,
                    out.cast(),
                )
            },
            "PaddleOCR decoder Linear",
        )
    }
    pub fn new(g: &Arc<Gpu>) -> Result<Self> {
        let settings =
            crate::settings::RuntimeSettings::from_options(&nextrocm_image::LoadOptions::default())
                .map_err(|e| Error(e.0))?;
        Self::with_settings(g, settings)
    }
    pub(crate) fn with_settings(
        g: &Arc<Gpu>,
        settings: crate::settings::RuntimeSettings,
    ) -> Result<Self> {
        if settings.vision_attention_tiled && ffi::api()?.vision_tiled.is_none() {
            return Err(Error("NS_POCR_VISION_TILED requires the private attention_tiled runtime symbol".into()));
        }
        ffi::api()?;
        Ok(Self(nextrocm_ops::kernels::Nsd::new(g)?, settings))
    }
    pub fn dequant(&self, src: P, ty: u32, n: usize, out: M, dt: Dt) -> Result<()> {
        // SAFETY: the caller owns n encoded elements and the destination span.
        ffi::check(
            unsafe {
                (ffi::api()?.dequant)(self.gpu.raw(), src, ty as i32, n as i64, out, dt as i32)
            },
            "PaddleOCR dequant",
        )
    }
    pub fn rope(
        &self,
        x: M,
        dt: Dt,
        rows: usize,
        heads: usize,
        dim: usize,
        stride: usize,
        cs: P,
        rot: usize,
    ) -> Result<()> {
        if dt != Dt::F32 || rot != dim {
            return Err(Error("PaddleOCR RoPE requires F32 full heads".into()));
        }
        // SAFETY: caller owns the strided rows and rows*dim interleaved cos/sin values.
        ffi::check(
            unsafe {
                (ffi::api()?.rope)(
                    self.gpu.raw(),
                    x.cast(),
                    rows as i64,
                    heads as i64,
                    dim as i64,
                    stride as i64,
                    cs.cast(),
                )
            },
            "PaddleOCR RoPE",
        )
    }
    pub fn attention(
        &self,
        q: P,
        k: P,
        v: P,
        dt: Dt,
        rows: usize,
        heads: usize,
        dim: usize,
        stride: usize,
        out: M,
        out_dt: Dt,
    ) -> Result<()> {
        if dt != Dt::F32 || out_dt != Dt::F32 {
            return Err(Error("PaddleOCR attention requires F32".into()));
        }
        if self.1.vision_attention_tiled && tiled_vision_shape(rows, heads, dim, stride) {
            let call = ffi::api()?.vision_tiled.ok_or_else(|| Error("private vision attention symbol missing".into()))?;
            // SAFETY: caller owns the F32 strided input/output spans; the private ABI
            // validates geometry, alignment, address wrap and disjoint output.
            return ffi::check(unsafe {
                call(self.gpu.raw(), q.cast(), k.cast(), v.cast(), rows as i64,
                     heads as i64, dim as i64, stride as i64, out.cast())
            }, "PaddleOCR tiled vision attention");
        }
        self.att(q, k, v, rows, heads, heads, dim, stride, stride, out, false)
    }
    pub fn attention_causal(
        &self,
        q: P,
        k: P,
        v: P,
        dt: Dt,
        rows: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        qs: usize,
        ks: usize,
        out: M,
    ) -> Result<()> {
        if dt != Dt::F32 {
            return Err(Error("PaddleOCR prefill requires F32".into()));
        }
        self.att(q, k, v, rows, heads, kv, dim, qs, ks, out, true)
    }
    fn att(
        &self,
        q: P,
        k: P,
        v: P,
        rows: usize,
        heads: usize,
        kv: usize,
        dim: usize,
        qs: usize,
        ks: usize,
        out: M,
        causal: bool,
    ) -> Result<()> {
        let mode = i32::from(causal)
            | if causal && self.1.attention_probability_first {
                2
            } else {
                0
            };
        // SAFETY: caller owns the Q/K/V spans and contiguous rows*heads*dim output.
        ffi::check(
            unsafe {
                (ffi::api()?.attention)(
                    self.gpu.raw(),
                    q.cast(),
                    k.cast(),
                    v.cast(),
                    rows as i64,
                    heads as i64,
                    kv as i64,
                    dim as i64,
                    qs as i64,
                    ks as i64,
                    out.cast(),
                    mode,
                )
            },
            "PaddleOCR attention",
        )
    }
}

// Larger grids deliberately retain the measured baseline; Q1 regressed.
fn tiled_vision_shape(rows: usize, heads: usize, dim: usize, stride: usize) -> bool {
    (1..=5120).contains(&rows) && (1..=16).contains(&heads) && dim == 72 && stride == 3 * heads * 72
}
#[cfg(test)]
mod tiled_tests {
    use super::tiled_vision_shape;
    #[test]
    fn larger_and_nonvision_shapes_keep_baseline() {
        assert!(tiled_vision_shape(5120, 16, 72, 3456));
        assert!(tiled_vision_shape(1, 1, 72, 216));
        for n in [0, 5121, 8192, usize::MAX] { assert!(!tiled_vision_shape(n, 16, 72, 3456)); }
        assert!(!tiled_vision_shape(576, usize::MAX, 72, 3456));
        assert!(!tiled_vision_shape(576, 16, 64, 3072));
        assert!(!tiled_vision_shape(576, 16, 72, 1152));
    }
}
