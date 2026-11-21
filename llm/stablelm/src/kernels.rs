//! Private checked family bindings using the upstream runtime's symbol loader.
use nextrocm_engine_core::{api, DevBuf, Error, Gpu, Result};
use std::ffi::c_void;
type P = *mut c_void;
type F = *mut f32;
pub(crate) struct Kernels {
    embed: unsafe extern "C" fn(P, P, i32, i32, i32, F) -> i32,
    mv: unsafe extern "C" fn(P, P, i32, i32, i32, F, F) -> i32,
    norm: unsafe extern "C" fn(P, F, F, F, i32, f32, F) -> i32,
    add: unsafe extern "C" fn(P, F, F, i32) -> i32,
    swiglu: unsafe extern "C" fn(P, F, F, i32) -> i32,
    rope: unsafe extern "C" fn(P, F, i32, i32, i32, i32, f32) -> i32,
    attn: unsafe extern "C" fn(P, F, F, F, P, P, i32, i32, i32, F) -> i32,
}
fn check(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!("stablelm kernel: {}", api()?.error())))
    }
}
fn i(n: usize) -> Result<i32> {
    i32::try_from(n).map_err(|_| Error("kernel dimension exceeds i32".into()))
}
fn floats(b: &DevBuf, n: usize) -> Result<()> {
    b.bounds(
        0,
        n.checked_mul(4)
            .ok_or_else(|| Error("buffer size overflow".into()))?,
    )
}
impl Kernels {
    pub fn load() -> Result<Self> {
        let a = api()?;
        macro_rules! sym {
            ($s:literal) => {{
                let p = a.symbol($s);
                if p.is_null() {
                    return Err(Error(format!(
                        "missing {}: build kernels/llm/stablelm/stablelm.hip into runtime",
                        $s
                    )));
                }
                // SAFETY: signatures match stablelm.h; runtime library lives for the process.
                #[allow(clippy::missing_transmute_annotations)] // destination type is the ABI field
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Self {
            embed: sym!("nslm_embed"),
            mv: sym!("nslm_mv"),
            norm: sym!("nslm_norm"),
            add: sym!("nslm_add"),
            swiglu: sym!("nslm_swiglu"),
            rope: sym!("nslm_rope"),
            attn: sym!("nslm_attn"),
        })
    }
    pub fn embed(&self, g: &Gpu, w: &super::Weight, token: u32, out: &DevBuf) -> Result<()> {
        if token as usize >= w.rows {
            return Err(Error("token outside vocabulary".into()));
        }
        floats(out, w.cols)?;
        // SAFETY: validated weight storage and output width; token checked above; engine owns all buffers on g.
        check(unsafe {
            (self.embed)(
                g.raw(),
                w.buf.ptr(),
                w.ty,
                i(w.cols)?,
                token as i32,
                out.fp(),
            )
        })
    }
    pub fn mv(&self, g: &Gpu, w: &super::Weight, x: &DevBuf, y: &DevBuf) -> Result<()> {
        floats(x, w.cols)?;
        floats(y, w.rows)?;
        // SAFETY: matrix and vector extents checked at load and here; buffers belong to g.
        check(unsafe {
            (self.mv)(
                g.raw(),
                w.buf.ptr(),
                w.ty,
                i(w.rows)?,
                i(w.cols)?,
                x.fp(),
                y.fp(),
            )
        })
    }
    #[allow(clippy::too_many_arguments)] // mirrors the private C ABI
    pub fn norm(
        &self,
        g: &Gpu,
        x: &DevBuf,
        w: &DevBuf,
        b: &DevBuf,
        n: usize,
        eps: f32,
        y: &DevBuf,
    ) -> Result<()> {
        for v in [x, w, b, y] {
            floats(v, n)?;
        }
        // SAFETY: all four vectors hold n floats on g; output is distinct from input.
        check(unsafe { (self.norm)(g.raw(), x.fp(), w.fp(), b.fp(), i(n)?, eps, y.fp()) })
    }
    pub fn add(&self, g: &Gpu, x: &DevBuf, y: &DevBuf, n: usize) -> Result<()> {
        floats(x, n)?;
        floats(y, n)?;
        // SAFETY: both buffers hold n floats on g.
        check(unsafe { (self.add)(g.raw(), x.fp(), y.fp(), i(n)?) })
    }
    pub fn swiglu(&self, g: &Gpu, x: &DevBuf, y: &DevBuf, n: usize) -> Result<()> {
        floats(x, n)?;
        floats(y, n)?;
        // SAFETY: both buffers hold n floats on g.
        check(unsafe { (self.swiglu)(g.raw(), x.fp(), y.fp(), i(n)?) })
    }
    pub fn rope(&self, g: &Gpu, x: &DevBuf, geom: &super::Geometry, pos: usize) -> Result<()> {
        floats(x, geom.width)?;
        // SAFETY: geometry fixes valid heads, head and rotary extents; x belongs to g.
        check(unsafe {
            (self.rope)(
                g.raw(),
                x.fp(),
                i(geom.heads)?,
                i(geom.head)?,
                i(geom.rotary)?,
                i(pos)?,
                geom.theta,
            )
        })
    }
    #[allow(clippy::too_many_arguments)] // mirrors the private C ABI
    pub fn attn(
        &self,
        g: &Gpu,
        q: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        kc: &DevBuf,
        vc: &DevBuf,
        geom: &super::Geometry,
        pos: usize,
        out: &DevBuf,
    ) -> Result<()> {
        for b in [q, k, v, out] {
            floats(b, geom.width)?;
        }
        let bytes = (pos + 1)
            .checked_mul(geom.width * 2)
            .ok_or_else(|| Error("cache extent overflow".into()))?;
        kc.bounds(0, bytes)?;
        vc.bounds(0, bytes)?;
        // SAFETY: all vectors and cache extents checked; cache is fp16 and position is in range.
        check(unsafe {
            (self.attn)(
                g.raw(),
                q.fp(),
                k.fp(),
                v.fp(),
                kc.ptr(),
                vc.ptr(),
                i(geom.heads)?,
                i(geom.head)?,
                i(pos)?,
                out.fp(),
            )
        })
    }
}
