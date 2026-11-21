//! Checked private quant.h ABI. Scalar F32 matvec is not original BF16/MMQ parity.
use crate::Result;
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::GType;
use std::{
    ffi::{c_int, c_void},
    sync::Arc,
};
type Decode = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    c_int,
    c_int,
    c_int,
    i64,
    *mut f32,
    i64,
) -> c_int;
type Matvec = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    c_int,
    c_int,
    c_int,
    i64,
    *const f32,
    i64,
    *const f32,
    *mut f32,
    i64,
) -> c_int;
pub struct Quant {
    gpu: Arc<Gpu>,
    decode: Decode,
    matvec: Matvec,
}
impl Quant {
    pub fn load(gpu: &Arc<Gpu>) -> Result<Self> {
        if gpu.raw().is_null() {
            return Err("Inkling quant requires a real device".into());
        }
        let lib = nextrocm_engine_core::api().map_err(|e| e.0)?;
        let d = lib.symbol("ni2_quant_decode");
        let m = lib.symbol("ni2_quant_matvec");
        if d.is_null() || m.is_null() {
            return Err("missing required Inkling quant ABI".into());
        }
        // SAFETY: exact kernels/llm/inkling2/quant.h declarations; process owns library.
        Ok(Self {
            gpu: gpu.clone(),
            decode: unsafe { std::mem::transmute::<*mut c_void, Decode>(d) },
            matvec: unsafe { std::mem::transmute::<*mut c_void, Matvec>(m) },
        })
    }
    fn buffer(&self, b: &DevBuf, bytes: usize) -> Result<()> {
        if !Arc::ptr_eq(b.gpu(), &self.gpu) || bytes == 0 || b.len < bytes {
            return Err("quant buffer extent/device mismatch".into());
        }
        Ok(())
    }
    fn finish(&self, rc: c_int) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(format!(
                "Inkling quant: {}",
                nextrocm_engine_core::api().map_err(|e| e.0)?.error()
            ))
        }
    }
    /// Contiguous packed rows and contiguous F32 output. No external scales.
    pub fn decode(
        &self,
        w: &DevBuf,
        ty: GType,
        rows: usize,
        cols: usize,
        out: &DevBuf,
    ) -> Result<()> {
        let stride = geometry(ty, rows, cols)?;
        self.buffer(w, rows * stride)?;
        self.buffer(out, rows * cols * 4)?;
        // SAFETY: bounded geometry/storage and same device; C rejects read/write overlap.
        self.finish(unsafe {
            (self.decode)(
                self.gpu.raw(),
                w.ptr(),
                ty.code() as c_int,
                rows as c_int,
                cols as c_int,
                stride as i64,
                out.ptr().cast(),
                cols as i64,
            )
        })
    }
    /// Contiguous one-vector dot, then separate external expert scalar multiply.
    pub fn matvec(
        &self,
        w: &DevBuf,
        ty: GType,
        rows: usize,
        cols: usize,
        x: &DevBuf,
        scale: Option<&DevBuf>,
        out: &DevBuf,
    ) -> Result<()> {
        let stride = geometry(ty, rows, cols)?;
        self.buffer(w, rows * stride)?;
        self.buffer(x, cols * 4)?;
        self.buffer(out, rows * 4)?;
        if let Some(s) = scale {
            self.buffer(s, 4)?;
        }
        // SAFETY: exact ABI, checked extents, scalar device scale; C validates aliasing.
        self.finish(unsafe {
            (self.matvec)(
                self.gpu.raw(),
                w.ptr(),
                ty.code() as c_int,
                rows as c_int,
                cols as c_int,
                stride as i64,
                x.ptr().cast(),
                1,
                scale.map_or(std::ptr::null(), |s| s.ptr().cast()),
                out.ptr().cast(),
                1,
            )
        })
    }
}
fn geometry(ty: GType, rows: usize, cols: usize) -> Result<usize> {
    if !matches!(
        ty,
        GType::F32 | GType::Q8_0 | GType::Q5K | GType::BF16 | GType::NVFP4
    ) || rows == 0
        || rows > 201024
        || cols == 0
        || cols > 8192
        || rows.checked_mul(cols).is_none_or(|n| n > 512 * 1024 * 1024)
    {
        return Err("unsupported quant geometry/type".into());
    }
    let (block, bytes) = ty.block();
    if cols % block != 0 {
        return Err("quant row must contain complete blocks".into());
    }
    Ok(cols / block * bytes)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn storage_and_geometry() {
        assert_eq!(geometry(GType::Q8_0, 1, 2304).unwrap(), 2448);
        assert_eq!(geometry(GType::NVFP4, 2304, 1152).unwrap(), 648);
        assert!(geometry(GType::NVFP4, 1, 63).is_err());
        assert!(geometry(GType::F16, 1, 128).is_err());
        assert!(geometry(GType::F32, usize::MAX, 2).is_err());
        assert!(Quant::load(&Gpu::detached("test")).is_err());
    }
}
