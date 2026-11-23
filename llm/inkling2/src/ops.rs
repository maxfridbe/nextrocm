//! Private single-token state/depth operators; checked buffers, no allocation.
use crate::Result;
use nextrocm_engine_core::{DevBuf, Gpu};
use std::{
    ffi::{c_int, c_void},
    sync::Arc,
};
type M = *mut c_void;
type P = *const f32;
type O = *mut f32;
type Conv = unsafe extern "C" fn(M, P, P, P, c_int, c_int, O, O) -> c_int;
type Attn = unsafe extern "C" fn(M, P, P, P, P, P, P, i64, c_int, c_int, O, O, i64) -> c_int;
type Mix = unsafe extern "C" fn(M, P, P, c_int, c_int, c_int, O) -> c_int;
type Router = unsafe extern "C" fn(M, P, P, *mut i32, O) -> c_int;
type Control = unsafe extern "C" fn(M, P, P, P, P, c_int, O) -> c_int;
type Scale = unsafe extern "C" fn(M, P, c_int, f32, O) -> c_int;
type Binary = unsafe extern "C" fn(M, P, P, c_int, O) -> c_int;
pub(crate) struct Ops {
    gpu: Arc<Gpu>,
    conv: Conv,
    attn: Attn,
    mix: Mix,
    router: Router,
    control: Control,
    scale: Scale,
    add: Binary,
    axpy: Binary,
}
impl Ops {
    pub fn load(gpu: &Arc<Gpu>) -> Result<Self> {
        if gpu.raw().is_null() {
            return Err("state operations require a real GPU".into());
        }
        let lib = nextrocm_engine_core::bindings::Library::new("inkling2", gpu).map_err(|e| e.0)?;
        macro_rules! bind {
            ($name:literal,$ty:ty) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("missing Inkling graph symbol {}", $name));
                }
                // SAFETY: exact state.h/mix.h signatures; runtime library lives for process.
                unsafe { std::mem::transmute::<M, $ty>(p) }
            }};
        }
        Ok(Self {
            gpu: gpu.clone(),
            conv: bind!("ni2_shortconv", Conv),
            attn: bind!("ni2_attention", Attn),
            mix: bind!("ni2_depthmix", Mix),
            router: bind!("ni2_router", Router),
            control: bind!("ni2_loop_control", Control),
            scale: bind!("ni2_scale", Scale),
            add: bind!("ni2_add", Binary),
            axpy: bind!("ni2_axpy", Binary),
        })
    }
    fn buffers(&self, bs: &[(&DevBuf, usize)]) -> Result<()> {
        for (b, n) in bs {
            if !Arc::ptr_eq(b.gpu(), &self.gpu)
                || *n == 0
                || n.checked_mul(4).is_none_or(|bytes| bytes > b.len)
            {
                return Err("state operation buffer extent/device mismatch".into());
            }
        }
        Ok(())
    }
    fn done(&self, rc: c_int) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(format!(
                "Inkling state operation: {}",
                nextrocm_engine_core::api().map_err(|e| e.0)?.error()
            ))
        }
    }
    pub fn conv(
        &self,
        x: &DevBuf,
        w: &DevBuf,
        d: Option<&DevBuf>,
        heads: usize,
        c: usize,
        h: &DevBuf,
        out: &DevBuf,
    ) -> Result<()> {
        if c == 0 || c > 8192 || heads == 0 || heads > c || c % heads != 0 {
            return Err("shortconv geometry".into());
        }
        self.buffers(&[(x, c), (w, c * 4), (h, c * 3), (out, c)])?;
        if let Some(d) = d {
            self.buffers(&[(d, heads * 4)])?;
        }
        // SAFETY: checked geometry/device/extents; C checks all write overlap.
        self.done(unsafe {
            (self.conv)(
                self.gpu.raw(),
                x.ptr().cast(),
                w.ptr().cast(),
                d.map_or(std::ptr::null(), |d| d.ptr().cast()),
                heads as c_int,
                c as c_int,
                h.ptr().cast(),
                out.ptr().cast(),
            )
        })
    }
    pub fn attention(
        &self,
        q: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        r: &DevBuf,
        w: &DevBuf,
        sink: &DevBuf,
        pos: usize,
        cap: usize,
        local: bool,
        out: &DevBuf,
        scratch: &DevBuf,
    ) -> Result<()> {
        if pos > i32::MAX as usize || cap == 0 || cap > 65536 {
            return Err("attention position/capacity".into());
        }
        let keys = if local { (pos + 1).min(1024) } else { pos + 1 };
        if keys > cap {
            return Err("attention keys exceed capacity".into());
        }
        self.buffers(&[
            (q, 3072),
            (k, cap * 1024),
            (v, cap * 1024),
            (r, 384),
            (w, 16384),
            (sink, 24),
            (out, 3072),
            (scratch, keys * 24),
        ])?;
        // SAFETY: checked F32 buffers and fixed Mini geometry; C validates overlap.
        self.done(unsafe {
            (self.attn)(
                self.gpu.raw(),
                q.ptr().cast(),
                k.ptr().cast(),
                v.ptr().cast(),
                r.ptr().cast(),
                w.ptr().cast(),
                sink.ptr().cast(),
                pos as i64,
                cap as c_int,
                if local { 1024 } else { 0 },
                out.ptr().cast(),
                scratch.ptr().cast(),
                (keys * 24) as i64,
            )
        })
    }
    pub fn mix(&self, s: &DevBuf, q: &DevBuf, count: usize, nq: usize, out: &DevBuf) -> Result<()> {
        if !(1..=5).contains(&count) || !(1..=4).contains(&nq) {
            return Err("depth mix geometry".into());
        }
        self.buffers(&[(s, count * 2304), (q, nq * 2304), (out, nq * 2304)])?;
        // SAFETY: fixed width and bounded matrix spans, C rejects read/write overlap.
        self.done(unsafe {
            (self.mix)(
                self.gpu.raw(),
                s.ptr().cast(),
                q.ptr().cast(),
                count as c_int,
                nq as c_int,
                2304,
                out.ptr().cast(),
            )
        })
    }
    pub fn router(&self, x: &DevBuf, b: &DevBuf, ids: &DevBuf, weights: &DevBuf) -> Result<()> {
        self.buffers(&[(x, 128), (b, 128), (ids, 8), (weights, 8)])?;
        // SAFETY: ids is an equally sized I32 buffer, all inputs same device.
        self.done(unsafe {
            (self.router)(
                self.gpu.raw(),
                x.ptr().cast(),
                b.ptr().cast(),
                ids.ptr().cast(),
                weights.ptr().cast(),
            )
        })
    }
    pub fn control(
        &self,
        x: &DevBuf,
        m: &DevBuf,
        g: &DevBuf,
        f: &DevBuf,
        out: &DevBuf,
    ) -> Result<()> {
        self.buffers(&[(x, 2304), (m, 2304), (g, 2304), (f, 2304), (out, 2304)])?;
        // SAFETY: checked same-device full Mini hidden vectors, C alias guards.
        self.done(unsafe {
            (self.control)(
                self.gpu.raw(),
                x.ptr().cast(),
                m.ptr().cast(),
                g.ptr().cast(),
                f.ptr().cast(),
                2304,
                out.ptr().cast(),
            )
        })
    }
    pub fn scale(&self, x: &DevBuf, n: usize, f: f32, out: &DevBuf) -> Result<()> {
        extent(n)?;
        if !f.is_finite() {
            return Err("nonfinite scale".into());
        }
        self.buffers(&[(x, n), (out, n)])?;
        // SAFETY: bounded same-device spans/finite scalar; C alias guards.
        self.done(unsafe {
            (self.scale)(
                self.gpu.raw(),
                x.ptr().cast(),
                n as c_int,
                f,
                out.ptr().cast(),
            )
        })
    }
    pub fn add(&self, a: &DevBuf, b: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        extent(n)?;
        self.buffers(&[(a, n), (b, n), (out, n)])?;
        // SAFETY: checked same-device spans and bounded cast; C alias guards.
        self.done(unsafe {
            (self.add)(
                self.gpu.raw(),
                a.ptr().cast(),
                b.ptr().cast(),
                n as c_int,
                out.ptr().cast(),
            )
        })
    }
    pub fn axpy(&self, x: &DevBuf, w: &DevBuf, n: usize, out: &DevBuf) -> Result<()> {
        extent(n)?;
        self.buffers(&[(x, n), (w, 1), (out, n)])?;
        // SAFETY: weight is one device F32 scalar; C enforces output disjointness.
        self.done(unsafe {
            (self.axpy)(
                self.gpu.raw(),
                x.ptr().cast(),
                w.ptr().cast(),
                n as c_int,
                out.ptr().cast(),
            )
        })
    }
}
fn extent(n: usize) -> Result<()> {
    if n == 0 || n > 16 * 1024 * 1024 {
        Err("vector extent".into())
    } else {
        Ok(())
    }
}
