//! Model-private SD1.5 HIP bindings; see kernels/image/sd15/sd15.h.
//! Calls enqueue on the existing GPU stream. Callers retain buffers (and owners
//! of DevBuf views) until completion. No allocation, host fallback, or conversion
//! of the weight bank. Shapes and strides are in elements. Activations are f32.
#![allow(clippy::too_many_arguments)]
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum WeightType {
    F32 = 0,
    F16 = 1,
}
#[derive(Clone, Copy)]
pub struct Weight<'a> {
    buf: &'a DevBuf,
    dtype: WeightType,
}
impl<'a> Weight<'a> {
    pub fn new(buf: &'a DevBuf, dtype: WeightType) -> Self {
        Self { buf, dtype }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Activation {
    Silu = 0,
    Gelu = 1,
    QuickGelu = 2,
    GeluTanh = 3,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Linear {
    pub rows: i64,
    pub input: i64,
    pub output: i64,
    pub x_stride: i64,
    pub out_stride: i64,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Conv2d {
    pub batch: i64,
    pub ci: i64,
    pub height: i64,
    pub width: i64,
    pub co: i64,
    pub kh: i64,
    pub kw: i64,
    pub sh: i64,
    pub sw: i64,
    pub ph: i64,
    pub pw: i64,
    pub oh: i64,
    pub ow: i64,
}
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Attention {
    pub batch: i64,
    pub queries: i64,
    pub keys: i64,
    pub heads: i64,
    pub dim: i64,
    pub qb: i64,
    pub qt: i64,
    pub qh: i64,
    pub kb: i64,
    pub kt: i64,
    pub kh: i64,
    pub vb: i64,
    pub vt: i64,
    pub vh: i64,
    pub ob: i64,
    pub ot: i64,
    pub oh: i64,
    pub scale: f32,
    pub causal: i32,
}
#[derive(Clone, Copy)]
#[repr(C)]
struct RawWeight {
    data: *const c_void,
    dtype: i32,
}
type G = *mut c_void;
type P = *const f32;
type M = *mut f32;
struct Api {
    linear: unsafe extern "C" fn(G, P, RawWeight, RawWeight, M, Linear) -> i32,
    conv2d: unsafe extern "C" fn(G, P, RawWeight, RawWeight, M, Conv2d) -> i32,
    group_norm:
        unsafe extern "C" fn(G, P, RawWeight, RawWeight, M, i64, i64, i64, i64, i64, f32) -> i32,
    layer_norm: unsafe extern "C" fn(G, P, RawWeight, RawWeight, M, i64, i64, f32) -> i32,
    activation: unsafe extern "C" fn(G, P, M, i64, i32) -> i32,
    add: unsafe extern "C" fn(G, P, P, M, i64, f32, f32) -> i32,
    multiply: unsafe extern "C" fn(G, P, P, M, i64) -> i32,
    geglu: unsafe extern "C" fn(G, P, M, i64, i64) -> i32,
    time_emb: unsafe extern "C" fn(G, P, P, M, i64, i64, i64, i64) -> i32,
    layout: unsafe extern "C" fn(G, P, M, i64, i64, i64, i64, i32) -> i32,
    up2: unsafe extern "C" fn(G, P, M, i64, i64, i64, i64) -> i32,
    concat: unsafe extern "C" fn(G, P, P, M, i64, i64, i64, i64, i64) -> i32,
    attention: unsafe extern "C" fn(G, P, P, P, M, Attention) -> i32,
    last_error: unsafe extern "C" fn() -> *const c_char,
}
fn api(gpu: &nextrocm_engine_core::Gpu) -> Result<&'static Api> {
    static API: nextrocm_engine_core::bindings::PerGpu<Api> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("sd15", gpu).map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("missing SD1.5 kernel {}", $name));
                }
                // SAFETY: signatures exactly match sd15.h; core retains the library.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            linear: sym!("nrsd_linear"),
            conv2d: sym!("nrsd_conv2d"),
            group_norm: sym!("nrsd_group_norm"),
            layer_norm: sym!("nrsd_layer_norm"),
            activation: sym!("nrsd_activation"),
            add: sym!("nrsd_add"),
            multiply: sym!("nrsd_multiply"),
            geglu: sym!("nrsd_geglu"),
            time_emb: sym!("nrsd_time_emb"),
            layout: sym!("nrsd_layout"),
            up2: sym!("nrsd_up2"),
            concat: sym!("nrsd_concat"),
            attention: sym!("nrsd_attention"),
            last_error: sym!("ns_last_error"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}
fn invalid(message: &str) -> Error {
    Error(format!("SD1.5: {message}"))
}
fn count(dims: &[i64]) -> Result<usize> {
    dims.iter().try_fold(1usize, |a, &v| {
        let v = usize::try_from(v).map_err(|_| invalid("negative dimension"))?;
        if v == 0 {
            return Err(invalid("zero dimension"));
        }
        a.checked_mul(v)
            .filter(|&n| n <= i64::MAX as usize / 4)
            .ok_or_else(|| invalid("shape overflow"))
    })
}
fn span(b: i64, t: i64, h: i64, d: i64, bs: i64, ts: i64, hs: i64) -> Result<usize> {
    count(&[b, t, h, d])?;
    let mut n = count(&[d])?;
    for (len, stride) in [(b, bs), (t, ts), (h, hs)] {
        let stride = count(&[stride])?;
        n = n
            .checked_add(
                (len as usize - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| invalid("stride overflow"))?,
            )
            .ok_or_else(|| invalid("span overflow"))?;
    }
    Ok(n)
}
/// Checks dimensions, byte spans, alignment, GPU ownership, and output overlap.
/// Native calls additionally validate operation-specific shapes and launch bounds.
pub struct Kernels {
    gpu: Arc<Gpu>,
    api: &'static Api,
}
impl Kernels {
    pub fn new(gpu: Arc<Gpu>) -> Result<Self> {
        if gpu.raw().is_null() {
            return Err(invalid("native GPU required"));
        }
        let api = api(&gpu)?;
        Ok(Self { gpu, api })
    }
    fn buffer(&self, b: &DevBuf, n: usize, width: usize) -> Result<()> {
        if !Arc::ptr_eq(b.gpu(), &self.gpu) {
            return Err(invalid("buffer belongs to another GPU"));
        }
        let bytes = n
            .checked_mul(width)
            .filter(|&x| x <= isize::MAX as usize)
            .ok_or_else(|| invalid("byte span overflow"))?;
        if b.ptr().is_null() || (b.ptr() as usize) % width != 0 {
            return Err(invalid("null or unaligned buffer"));
        }
        b.bounds(0, bytes)
    }
    fn f32(&self, b: &DevBuf, n: usize) -> Result<()> {
        self.buffer(b, n, 4)
    }
    fn weight(&self, w: Option<Weight<'_>>, n: usize, out: &DevBuf) -> Result<RawWeight> {
        match w {
            None => Ok(RawWeight {
                data: std::ptr::null(),
                dtype: 0,
            }),
            Some(w) => {
                self.buffer(w.buf, n, if w.dtype == WeightType::F16 { 2 } else { 4 })?;
                self.disjoint(w.buf, out, false)?;
                Ok(RawWeight {
                    data: w.buf.ptr(),
                    dtype: w.dtype as i32,
                })
            }
        }
    }
    fn disjoint(&self, a: &DevBuf, b: &DevBuf, in_place: bool) -> Result<()> {
        let (x, y) = (a.ptr() as usize, b.ptr() as usize);
        if in_place && x == y {
            return Ok(());
        }
        let xe = x
            .checked_add(a.len)
            .ok_or_else(|| invalid("address overflow"))?;
        let ye = y
            .checked_add(b.len)
            .ok_or_else(|| invalid("address overflow"))?;
        if x < ye && y < xe {
            return Err(invalid("overlapping output and input"));
        }
        Ok(())
    }
    fn pair(&self, x: &DevBuf, out: &DevBuf, n: usize, m: usize, in_place: bool) -> Result<()> {
        self.f32(x, n)?;
        self.f32(out, m)?;
        self.disjoint(x, out, in_place)
    }
    fn ok(&self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // SAFETY: runtime owns a thread-local NUL-terminated error string.
        let p = unsafe { (self.api.last_error)() };
        let message = if p.is_null() {
            "unknown HIP error".to_owned()
        } else {
            // SAFETY: checked non-null runtime error pointer.
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        };
        Err(invalid(&message))
    }
    pub fn linear(
        &self,
        x: &DevBuf,
        w: Weight<'_>,
        b: Option<Weight<'_>>,
        out: &DevBuf,
        s: Linear,
    ) -> Result<()> {
        self.pair(
            x,
            out,
            count(&[s.rows, s.x_stride])?,
            count(&[s.rows, s.out_stride])?,
            false,
        )?;
        let w = self.weight(Some(w), count(&[s.output, s.input])?, out)?;
        let b = self.weight(b, count(&[s.output])?, out)?;
        // SAFETY: checked buffers/weights; native validates leading dimensions.
        self.ok(unsafe { (self.api.linear)(self.gpu.raw(), x.fp(), w, b, out.fp(), s) })
    }
    pub fn conv2d(
        &self,
        x: &DevBuf,
        w: Weight<'_>,
        b: Option<Weight<'_>>,
        out: &DevBuf,
        s: Conv2d,
    ) -> Result<()> {
        self.pair(
            x,
            out,
            count(&[s.batch, s.ci, s.height, s.width])?,
            count(&[s.batch, s.co, s.oh, s.ow])?,
            false,
        )?;
        let w = self.weight(Some(w), count(&[s.co, s.ci, s.kh, s.kw])?, out)?;
        let b = self.weight(b, count(&[s.co])?, out)?;
        // SAFETY: checked contiguous NCHW/OIHW buffers, native validates geometry.
        self.ok(unsafe { (self.api.conv2d)(self.gpu.raw(), x.fp(), w, b, out.fp(), s) })
    }
    pub fn group_norm(
        &self,
        x: &DevBuf,
        w: Option<Weight<'_>>,
        b: Option<Weight<'_>>,
        out: &DevBuf,
        n: i64,
        c: i64,
        h: i64,
        width: i64,
        groups: i64,
        eps: f32,
    ) -> Result<()> {
        let size = count(&[n, c, h, width])?;
        self.pair(x, out, size, size, true)?;
        let w = self.weight(w, count(&[c])?, out)?;
        let b = self.weight(b, count(&[c])?, out)?;
        // SAFETY: checked activation/affine spans, native validates groups/epsilon.
        self.ok(unsafe {
            (self.api.group_norm)(
                self.gpu.raw(),
                x.fp(),
                w,
                b,
                out.fp(),
                n,
                c,
                h,
                width,
                groups,
                eps,
            )
        })
    }
    pub fn layer_norm(
        &self,
        x: &DevBuf,
        w: Option<Weight<'_>>,
        b: Option<Weight<'_>>,
        out: &DevBuf,
        rows: i64,
        c: i64,
        eps: f32,
    ) -> Result<()> {
        let size = count(&[rows, c])?;
        self.pair(x, out, size, size, true)?;
        let w = self.weight(w, count(&[c])?, out)?;
        let b = self.weight(b, count(&[c])?, out)?;
        // SAFETY: checked activation/affine spans, native validates epsilon.
        self.ok(unsafe {
            (self.api.layer_norm)(self.gpu.raw(), x.fp(), w, b, out.fp(), rows, c, eps)
        })
    }
    pub fn activation(&self, x: &DevBuf, out: &DevBuf, n: i64, kind: Activation) -> Result<()> {
        let size = count(&[n])?;
        self.pair(x, out, size, size, true)?;
        // SAFETY: checked f32 spans; exact in-place is supported.
        self.ok(unsafe { (self.api.activation)(self.gpu.raw(), x.fp(), out.fp(), n, kind as i32) })
    }
    pub fn add(
        &self,
        x: &DevBuf,
        y: Option<&DevBuf>,
        out: &DevBuf,
        n: i64,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let size = count(&[n])?;
        self.pair(x, out, size, size, true)?;
        if let Some(y) = y {
            self.f32(y, size)?;
            self.disjoint(y, out, true)?;
        }
        // SAFETY: checked equal f32 spans, null y denotes scale only.
        self.ok(unsafe {
            (self.api.add)(
                self.gpu.raw(),
                x.fp(),
                y.map_or(std::ptr::null(), |b| b.fp()),
                out.fp(),
                n,
                alpha,
                beta,
            )
        })
    }
    pub fn multiply(&self, x: &DevBuf, y: &DevBuf, out: &DevBuf, n: i64) -> Result<()> {
        let size = count(&[n])?;
        self.pair(x, out, size, size, true)?;
        self.f32(y, size)?;
        self.disjoint(y, out, true)?;
        // SAFETY: checked equal f32 spans; exact in-place is supported.
        self.ok(unsafe { (self.api.multiply)(self.gpu.raw(), x.fp(), y.fp(), out.fp(), n) })
    }
    /// Packed [rows, 2C] value * tanh-GELU(gate), matching pinned development-host ggml.
    /// Explicit erf-GELU remains available through `Activation::Gelu`.
    pub fn geglu(&self, x: &DevBuf, out: &DevBuf, rows: i64, c: i64) -> Result<()> {
        self.pair(x, out, count(&[rows, c, 2])?, count(&[rows, c])?, false)?;
        // SAFETY: checked packed value/gate input and disjoint output.
        self.ok(unsafe { (self.api.geglu)(self.gpu.raw(), x.fp(), out.fp(), rows, c) })
    }
    pub fn time_emb(
        &self,
        x: &DevBuf,
        e: &DevBuf,
        out: &DevBuf,
        n: i64,
        c: i64,
        h: i64,
        w: i64,
    ) -> Result<()> {
        let size = count(&[n, c, h, w])?;
        self.pair(x, out, size, size, true)?;
        self.f32(e, count(&[n, c])?)?;
        self.disjoint(e, out, false)?;
        // SAFETY: checked NCHW and embedding spans.
        self.ok(unsafe {
            (self.api.time_emb)(self.gpu.raw(), x.fp(), e.fp(), out.fp(), n, c, h, w)
        })
    }
    pub fn layout(
        &self,
        x: &DevBuf,
        out: &DevBuf,
        n: i64,
        c: i64,
        h: i64,
        w: i64,
        to_nhwc: bool,
    ) -> Result<()> {
        let size = count(&[n, c, h, w])?;
        self.pair(x, out, size, size, false)?;
        // SAFETY: checked disjoint layout buffers of equal size.
        self.ok(unsafe {
            (self.api.layout)(
                self.gpu.raw(),
                x.fp(),
                out.fp(),
                n,
                c,
                h,
                w,
                i32::from(to_nhwc),
            )
        })
    }
    pub fn up2(&self, x: &DevBuf, out: &DevBuf, n: i64, c: i64, h: i64, w: i64) -> Result<()> {
        self.pair(
            x,
            out,
            count(&[n, c, h, w])?,
            count(&[n, c, h, w, 4])?,
            false,
        )?;
        // SAFETY: checked disjoint NCHW input and 2x spatial output.
        self.ok(unsafe { (self.api.up2)(self.gpu.raw(), x.fp(), out.fp(), n, c, h, w) })
    }
    pub fn concat(
        &self,
        x: &DevBuf,
        y: &DevBuf,
        out: &DevBuf,
        n: i64,
        ca: i64,
        cb: i64,
        h: i64,
        w: i64,
    ) -> Result<()> {
        let c = ca
            .checked_add(cb)
            .ok_or_else(|| invalid("channel overflow"))?;
        self.pair(x, out, count(&[n, ca, h, w])?, count(&[n, c, h, w])?, false)?;
        self.f32(y, count(&[n, cb, h, w])?)?;
        self.disjoint(y, out, false)?;
        // SAFETY: checked disjoint NCHW channel inputs and output.
        self.ok(unsafe {
            (self.api.concat)(self.gpu.raw(), x.fp(), y.fp(), out.fp(), n, ca, cb, h, w)
        })
    }
    pub fn attention(
        &self,
        q: &DevBuf,
        k: &DevBuf,
        v: &DevBuf,
        out: &DevBuf,
        s: Attention,
    ) -> Result<()> {
        self.f32(
            out,
            span(s.batch, s.queries, s.heads, s.dim, s.ob, s.ot, s.oh)?,
        )?;
        for (buf, t, bs, ts, hs) in [
            (q, s.queries, s.qb, s.qt, s.qh),
            (k, s.keys, s.kb, s.kt, s.kh),
            (v, s.keys, s.vb, s.vt, s.vh),
        ] {
            self.f32(buf, span(s.batch, t, s.heads, s.dim, bs, ts, hs)?)?;
            self.disjoint(buf, out, false)?;
        }
        // SAFETY: checked strided spans; native validates non-overlap of rows/heads.
        self.ok(unsafe {
            (self.api.attention)(self.gpu.raw(), q.fp(), k.fp(), v.fp(), out.fp(), s)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checked_shapes() {
        assert_eq!(count(&[2, 320, 64, 64]).unwrap(), 2 * 320 * 64 * 64);
        assert!(count(&[0]).is_err());
        assert!(count(&[-1]).is_err());
        assert!(count(&[i64::MAX, 2]).is_err());
    }
    #[test]
    fn token_and_head_major_spans() {
        assert_eq!(
            span(2, 77, 12, 64, 77 * 768, 768, 64).unwrap(),
            2 * 77 * 768
        );
        assert_eq!(
            span(2, 77, 12, 64, 77 * 768, 64, 77 * 64).unwrap(),
            2 * 77 * 768
        );
        assert!(span(2, 77, 12, 64, i64::MAX, i64::MAX, 64).is_err());
    }
    #[test]
    fn c_layout() {
        assert_eq!(std::mem::size_of::<Linear>(), 40);
        assert_eq!(std::mem::size_of::<Conv2d>(), 104);
        assert_eq!(std::mem::size_of::<Attention>(), 144);
        assert_eq!(std::mem::size_of::<RawWeight>(), 16);
    }
}
