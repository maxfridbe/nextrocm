//! Checked F32 bindings to the private primitives.h ABI. No inference/admission claim.
use crate::Result;
use nextrocm_engine_core::{DevBuf, Gpu};
use std::{
    ffi::{c_int, c_void},
    sync::Arc,
};

type Rows =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, c_int, c_int, f32, *mut f32) -> c_int;
type Situ =
    unsafe extern "C" fn(*mut c_void, *const f32, *const f32, c_int, f32, f32, *mut f32) -> c_int;
type Gate = unsafe extern "C" fn(*mut c_void, *const f32, *const f32, c_int, *mut f32) -> c_int;
type Softcap = unsafe extern "C" fn(*mut c_void, *const f32, c_int, f32, *mut f32) -> c_int;
const MAX_ELEMENTS: usize = 16 * 1024 * 1024;

pub struct Primitives {
    gpu: Arc<Gpu>,
    rms: Rows,
    clip: Rows,
    situ: Situ,
    gate: Gate,
    softcap: Softcap,
}
impl Primitives {
    /// Resolve every primitive before any model allocation. The process runtime
    /// retains its library; holding the Arc keeps the device/stream alive.
    pub fn load(gpu: &Arc<Gpu>) -> Result<Self> {
        if gpu.raw().is_null() {
            return Err("Inkling primitives require a real device".into());
        }
        let lib = nextrocm_engine_core::api().map_err(|e| e.0)?;
        macro_rules! bind {
            ($name:literal, $ty:ty) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("missing required Inkling symbol {}", $name));
                }
                // SAFETY: exact signatures in kernels/llm/inkling2/primitives.h;
                // the process-wide runtime retains the library permanently.
                unsafe { std::mem::transmute::<*mut c_void, $ty>(p) }
            }};
        }
        Ok(Self {
            gpu: gpu.clone(),
            rms: bind!("ni2_rms", Rows),
            clip: bind!("ni2_clip", Rows),
            situ: bind!("ni2_situ", Situ),
            gate: bind!("ni2_gate", Gate),
            softcap: bind!("ni2_softcap", Softcap),
        })
    }
    fn buffer(&self, b: &DevBuf, elements: usize) -> Result<()> {
        if !Arc::ptr_eq(b.gpu(), &self.gpu) {
            return Err("primitive buffer belongs to another device handle".into());
        }
        if elements == 0
            || elements > MAX_ELEMENTS
            || b.len < elements.checked_mul(4).ok_or("F32 byte overflow")?
        {
            return Err("primitive buffer/element bounds".into());
        }
        Ok(())
    }
    fn finish(&self, rc: c_int) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(format!(
                "Inkling primitive: {}",
                nextrocm_engine_core::api().map_err(|e| e.0)?.error()
            ))
        }
    }
    pub fn rms(
        &self,
        x: &DevBuf,
        gamma: Option<&DevBuf>,
        rows: usize,
        width: usize,
        out: &DevBuf,
    ) -> Result<()> {
        let n = extent(rows, width)?;
        self.buffer(x, n)?;
        self.buffer(out, n)?;
        if let Some(g) = gamma {
            self.buffer(g, width)?;
        }
        // SAFETY: same-device buffer extents checked; C validates overlap before
        // launch. Buffers own allocations; queued work uses the device's stream.
        self.finish(unsafe {
            (self.rms)(
                self.gpu.raw(),
                x.ptr().cast(),
                gamma.map_or(std::ptr::null(), |g| g.ptr().cast()),
                rows as c_int,
                width as c_int,
                1e-6,
                out.ptr().cast(),
            )
        })
    }
    pub fn clip(
        &self,
        x: &DevBuf,
        scale: Option<&DevBuf>,
        rows: usize,
        width: usize,
        threshold: f32,
        out: &DevBuf,
    ) -> Result<()> {
        let n = extent(rows, width)?;
        if !threshold.is_finite() || threshold <= 0.0 {
            return Err("invalid clip threshold".into());
        }
        self.buffer(x, n)?;
        self.buffer(out, n)?;
        // Unlike RMS gamma, clip scales every element, not a broadcast width.
        if let Some(s) = scale {
            self.buffer(s, n)?;
        }
        // SAFETY: checked extents/device/threshold; C rejects forbidden aliasing.
        self.finish(unsafe {
            (self.clip)(
                self.gpu.raw(),
                x.ptr().cast(),
                scale.map_or(std::ptr::null(), |s| s.ptr().cast()),
                rows as c_int,
                width as c_int,
                threshold,
                out.ptr().cast(),
            )
        })
    }
    pub fn situ(&self, gate: &DevBuf, up: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        self.buffer(gate, n)?;
        self.buffer(up, n)?;
        self.buffer(out, n)?;
        // SAFETY: checked same-device extents; exact selected Mini constants.
        self.finish(unsafe {
            (self.situ)(
                self.gpu.raw(),
                gate.ptr().cast(),
                up.ptr().cast(),
                n as c_int,
                4.0,
                25.0,
                out.ptr().cast(),
            )
        })
    }
    pub fn gate(&self, x: &DevBuf, gate: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        self.buffer(x, n)?;
        self.buffer(gate, n)?;
        self.buffer(out, n)?;
        // SAFETY: checked same-device extents, C validates overlapping spans.
        self.finish(unsafe {
            (self.gate)(
                self.gpu.raw(),
                x.ptr().cast(),
                gate.ptr().cast(),
                n as c_int,
                out.ptr().cast(),
            )
        })
    }
    pub fn softcap(&self, x: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        self.buffer(x, n)?;
        self.buffer(out, n)?;
        // SAFETY: checked same-device extents; selected Mini logit soft cap.
        self.finish(unsafe {
            (self.softcap)(
                self.gpu.raw(),
                x.ptr().cast(),
                n as c_int,
                30.0,
                out.ptr().cast(),
            )
        })
    }
}
fn extent(rows: usize, width: usize) -> Result<usize> {
    let n = rows.checked_mul(width).ok_or("primitive extent overflow")?;
    if rows == 0 || width == 0 || width > 8192 || n > MAX_ELEMENTS {
        return Err("primitive extent unsupported".into());
    }
    Ok(n)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_and_extent_rejections_precede_launch() {
        assert!(Primitives::load(&Gpu::detached("test")).is_err());
        assert_eq!(extent(24, 128).unwrap(), 3072);
        assert!(extent(usize::MAX, 2).is_err());
        assert!(extent(1, 8193).is_err());
        assert!(extent(0, 2304).is_err());
        assert!(extent(MAX_ELEMENTS, 2).is_err());
    }
}
