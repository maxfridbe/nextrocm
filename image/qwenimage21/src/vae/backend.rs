//! Private VAE subset binder; no shared Nsd or cross-model exports.
use nextrocm_engine_core::{Error, Gpu, Result};
use std::ffi::{c_int, c_void};
use std::sync::{Arc, OnceLock};
type P = *const c_void;
type M = *mut c_void;
type I = i64;
type ConvFn = unsafe extern "C" fn(M, P, I, I, I, P, I, c_int, c_int, *const f32, M) -> c_int;
#[derive(Clone, Copy, PartialEq)]
pub enum Dt {
    F32 = 0,
    F16 = 1,
}
pub fn none() -> P {
    std::ptr::null()
}
#[allow(clippy::type_complexity)]
pub struct Api {
    pub to_half: unsafe extern "C" fn(M, *const f32, I, M) -> c_int,
    pub to_float: unsafe extern "C" fn(M, P, I, *mut f32) -> c_int,
    pub silu: unsafe extern "C" fn(M, M, I) -> c_int,
    pub up2: unsafe extern "C" fn(M, P, I, I, I, M) -> c_int,
    pub dupup_add: unsafe extern "C" fn(M, P, I, I, I, I, c_int, M) -> c_int,
    pub pad_br: unsafe extern "C" fn(M, P, I, I, I, M) -> c_int,
    pub avg_down_add: unsafe extern "C" fn(M, P, I, I, I, I, c_int, M) -> c_int,
    pub to_rgba8: unsafe extern "C" fn(M, P, I, I, *mut u8) -> c_int,
    pub softmax_rows: unsafe extern "C" fn(M, *const f32, I, I, f32, M) -> c_int,
    conv: ConvFn,
    conv64: Option<ConvFn>,
    linear: unsafe extern "C" fn(M, P, c_int, I, I, P, I, *const f32, M, c_int) -> c_int,
    rms: unsafe extern "C" fn(
        M,
        P,
        c_int,
        I,
        I,
        *const f32,
        f32,
        *const i32,
        *const f32,
        *const f32,
        M,
        c_int,
    ) -> c_int,
    add: unsafe extern "C" fn(M, M, c_int, I, I, P, c_int, *const i32, *const f32) -> c_int,
}
pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(||{let lib=nextrocm_engine_core::api().map_err(|e|e.0)?;
 macro_rules! sym {($name:literal)=>{{let p=lib.symbol($name);if p.is_null(){return Err(format!("missing VAE symbol {}",$name));}
 // SAFETY: matching private vae/convert headers or existing speech.hip ABI; library stays live.
 #[allow(clippy::missing_transmute_annotations)]
 unsafe{std::mem::transmute::<M,_>(p)}}};}
 let p=lib.symbol("nrqi_vae_conv64");
 // SAFETY: optional independently validated export has the exact ConvFn ABI.
 let conv64=if p.is_null(){None}else{Some(unsafe{std::mem::transmute::<M,ConvFn>(p)})};
 Ok(Api{conv64,to_half:sym!("nrqi_f32_to_f16"),to_float:sym!("nrqi_f16_to_f32"),silu:sym!("nrqi_vae_silu"),up2:sym!("nrqi_vae_up2"),dupup_add:sym!("nrqi_vae_dupup_add"),pad_br:sym!("nrqi_vae_pad_br"),avg_down_add:sym!("nrqi_vae_avg_down_add"),to_rgba8:sym!("nrqi_vae_rgba8"),softmax_rows:sym!("nrqi_vae_softmax"),conv:sym!("nrqi_vae_conv"),linear:sym!("nsd_linear"),rms:sym!("nsd_rms_norm_mod"),add:sym!("nsd_gate_add")})
 }).as_ref().map_err(|e|Error(e.clone()))
}
pub fn check(code: c_int, op: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        let detail = nextrocm_engine_core::api()
            .map(|a| a.error())
            .unwrap_or_default();
        Err(Error(format!("{op}: {detail} (code {code})")))
    }
}
pub struct Backend {
    pub gpu: Arc<Gpu>,
    kernel: super::ConvKernel,
}
impl Backend {
    pub fn new(gpu: &Arc<Gpu>, kernel: super::ConvKernel) -> Result<Self> {
        let a = api()?;
        if kernel != super::ConvKernel::Baseline && a.conv64.is_none() {
            return Err(Error("explicit VAE conv64 ABI unavailable".into()));
        }
        Ok(Self {
            gpu: gpu.clone(),
            kernel,
        })
    }
    pub fn wait(&self) -> Result<()> {
        self.gpu.sync()
    }
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &self,
        x: P,
        dt: Dt,
        n: usize,
        h: usize,
        w: usize,
        ci: usize,
        weight: P,
        co: usize,
        k: usize,
        bias: P,
        out: M,
    ) -> Result<()> {
        if dt != Dt::F16 || n != 1 {
            return Err(Error("VAE convolution expects single-frame half".into()));
        }
        let a = api()?;
        let function = if self.kernel.uses_register64(h, w, ci, co, k) {
            a.conv64
                .ok_or_else(|| Error("VAE conv64 ABI unavailable".into()))?
        } else {
            a.conv
        };
        // SAFETY: VAE graph allocates validated NHWC/OIHW storage; private launcher validates geometry/spans.
        vae_gpu_span!(
            &self.gpu,
            "vae.conv",
            &[h, w, ci, co, k, 1],
            check(
                unsafe {
                    (function)(
                        self.gpu.raw(),
                        x,
                        h as I,
                        w as I,
                        ci as I,
                        weight,
                        co as I,
                        k as c_int,
                        1,
                        bias.cast(),
                        out,
                    )
                },
                "VAE convolution",
            )
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn conv3d_ex(
        &self,
        x: P,
        dt: Dt,
        thw: (usize, usize, usize),
        ci: usize,
        weight: P,
        co: usize,
        k: (usize, usize, usize),
        s: (usize, usize, usize),
        bias: P,
        out: M,
    ) -> Result<()> {
        if dt != Dt::F16 || thw.0 != 1 || k != (1, 3, 3) || s != (1, 2, 2) {
            return Err(Error("VAE stride2 single-frame geometry".into()));
        }
        // SAFETY: same graph invariant as conv2d, valid-padding stride2 output.
        vae_gpu_span!(
            &self.gpu,
            "vae.conv",
            &[thw.1, thw.2, ci, co, 3, 2],
            check(
                unsafe {
                    (api()?.conv)(
                        self.gpu.raw(),
                        x,
                        thw.1 as I,
                        thw.2 as I,
                        ci as I,
                        weight,
                        co as I,
                        3,
                        2,
                        bias.cast(),
                        out,
                    )
                },
                "VAE down convolution",
            )
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn linear(
        &self,
        x: P,
        dt: Dt,
        m: usize,
        k: usize,
        w: P,
        n: usize,
        bias: P,
        out: M,
        odt: Dt,
    ) -> Result<()> {
        // SAFETY: private attention graph owns matrices in row-major layout and output allocation.
        vae_gpu_span!(
            &self.gpu,
            "vae.linear",
            &[m, k, n],
            check(
                unsafe {
                    (api()?.linear)(
                        self.gpu.raw(),
                        x,
                        dt as c_int,
                        m as I,
                        k as I,
                        w,
                        n as I,
                        bias.cast(),
                        out,
                        odt as c_int,
                    )
                },
                "VAE linear",
            )
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_mod(
        &self,
        x: P,
        dt: Dt,
        m: usize,
        c: usize,
        w: P,
        eps: f32,
        rows: P,
        scale: P,
        shift: P,
        out: M,
        odt: Dt,
    ) -> Result<()> {
        // SAFETY: graph owns m*c input/output and c gamma entries, null modulation pointers.
        vae_gpu_span!(
            &self.gpu,
            "vae.rms",
            &[m, c],
            check(
                unsafe {
                    (api()?.rms)(
                        self.gpu.raw(),
                        x,
                        dt as c_int,
                        m as I,
                        c as I,
                        w.cast(),
                        eps,
                        rows.cast(),
                        scale.cast(),
                        shift.cast(),
                        out,
                        odt as c_int,
                    )
                },
                "VAE RMS",
            )
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn gate_add(
        &self,
        x: M,
        dt: Dt,
        m: usize,
        c: usize,
        y: P,
        ydt: Dt,
        rows: P,
        gate: P,
    ) -> Result<()> {
        // SAFETY: both graph buffers cover m*c entries; null gate gives original residual addition.
        vae_gpu_span!(
            &self.gpu,
            "vae.residual",
            &[m, c],
            check(
                unsafe {
                    (api()?.add)(
                        self.gpu.raw(),
                        x,
                        dt as c_int,
                        m as I,
                        c as I,
                        y,
                        ydt as c_int,
                        rows.cast(),
                        gate.cast(),
                    )
                },
                "VAE residual",
            )
        )
    }
}
