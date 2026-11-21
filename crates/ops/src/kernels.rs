//! The shared diffusion kernels (`kernels/diffusion/nsd.h`, linked into the image library), bound by name. `Nsd` is
//! one context on one GPU - around that GPU's own stream, so the buffers nextcuda-core allocates there are the kernels'
//! to read - with checked calls on raw device pointers (`DevBuf::ptr`). The calls are nextsycl's, so its engines'
//! Rust code is what runs on them.
//!
//! The pointers are device memory the host never dereferences: the kernels read them on the GPU, so the calls are not
//! `unsafe` fns; what they rest on is the sizes the caller names (each call's doc says them).
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::{Arc, OnceLock};

use nextrocm_engine_core::{Error, Gpu, Result};

/// A buffer's element type, as the kernels name it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Dt {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}

type P = *const c_void;
type M = *mut c_void;

#[allow(clippy::type_complexity)]
struct Api {
    swiglu: unsafe extern "C" fn(M, P, c_int, i64, i64, M, c_int) -> c_int,
    rms_norm_mod: unsafe extern "C" fn(
        M,
        P,
        c_int,
        i64,
        i64,
        *const f32,
        f32,
        *const i32,
        *const f32,
        *const f32,
        M,
        c_int,
    ) -> c_int,
    last_error: unsafe extern "C" fn() -> *const c_char,
    create: unsafe extern "C" fn(M) -> M,
    destroy: unsafe extern "C" fn(M),
    wait: unsafe extern "C" fn(M) -> c_int,
    linear: unsafe extern "C" fn(M, P, c_int, i64, i64, P, i64, *const f32, M, c_int) -> c_int,
    layer_norm:
        unsafe extern "C" fn(M, P, c_int, i64, i64, *const f32, *const f32, f32, M, c_int) -> c_int,
    gelu: unsafe extern "C" fn(M, M, c_int, i64, c_int) -> c_int,
    gate_add:
        unsafe extern "C" fn(M, M, c_int, i64, i64, P, c_int, *const i32, *const f32) -> c_int,
    attention_causal_rows:
        unsafe extern "C" fn(M, P, P, P, c_int, i64, i64, i64, i64, i64, i64, i64, M) -> c_int,
    attention_qk:
        unsafe extern "C" fn(M, P, i64, i64, P, P, i64, i64, c_int, i64, i64, M, c_int) -> c_int,
    attention: unsafe extern "C" fn(M, P, P, P, c_int, i64, i64, i64, i64, M, c_int) -> c_int,
    conv1d: unsafe extern "C" fn(
        M,
        *const f32,
        i64,
        i64,
        i64,
        *const f32,
        i64,
        i64,
        *const f32,
        i64,
        i64,
        i64,
        *mut f32,
        i64,
    ) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextrocm_engine_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (an image library, built with kernels/diffusion)", $name));
                }
                // SAFETY: the symbol is declared in nsd.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            swiglu: sym!("nsd_swiglu"),
            rms_norm_mod: sym!("nsd_rms_norm_mod"),
            last_error: sym!("nsd_last_error"),
            create: sym!("nsd_create"),
            destroy: sym!("nsd_destroy"),
            wait: sym!("nsd_wait"),
            linear: sym!("nsd_linear"),
            layer_norm: sym!("nsd_layer_norm"),
            gelu: sym!("nsd_gelu"),
            gate_add: sym!("nsd_gate_add"),
            attention_causal_rows: sym!("nsd_attention_causal_rows"),
            attention_qk: sym!("nsd_attention_qk"),
            attention: sym!("nsd_attention"),
            conv1d: sym!("nsd_conv1d"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

/// The kernels' context on one GPU
pub struct Nsd {
    ctx: M,
    k: &'static Api,
    pub gpu: Arc<Gpu>,
}

// SAFETY: the context is used through &self from one thread at a time by its engine (the stream is in order).
unsafe impl Send for Nsd {}
unsafe impl Sync for Nsd {}

impl Drop for Nsd {
    fn drop(&mut self) {
        // SAFETY: made by nsd_create, destroyed once.
        unsafe { (self.k.destroy)(self.ctx) };
    }
}

fn last(k: &Api) -> String {
    // SAFETY: a C string owned by the library, valid until the next call on this thread.
    unsafe { CStr::from_ptr((k.last_error)()) }
        .to_string_lossy()
        .into_owned()
}

/// A pointer for "none" (a null bias, rows, scale)
pub fn none() -> P {
    std::ptr::null()
}

impl Nsd {
    /// A context on `gpu`'s own stream
    pub fn new(gpu: &Arc<Gpu>) -> Result<Nsd> {
        let k = api()?;
        // SAFETY: the GPU's handle, alive as long as the GPU (held here).
        let ctx = unsafe { (k.create)(gpu.raw()) };
        if ctx.is_null() {
            return Err(Error(format!(
                "the diffusion kernels' context on {}: {}",
                gpu.name,
                last(k)
            )));
        }
        Ok(Nsd {
            ctx,
            k,
            gpu: gpu.clone(),
        })
    }

    fn ok(&self, rc: c_int, what: &str) -> Result<()> {
        if rc == 0 {
            Ok(())
        } else {
            Err(Error(format!("{what}: {}", last(self.k))))
        }
    }

    /// The context itself, for a kernel of `nsd.h` this crate does not bind (an engine binds it by name)
    pub fn raw(&self) -> M {
        self.ctx
    }

    pub fn wait(&self) -> Result<()> {
        // SAFETY: a live context.
        self.ok(unsafe { (self.k.wait)(self.ctx) }, "waiting for the GPU")
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
        out_dt: Dt,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.linear)(
                    self.ctx,
                    x,
                    dt as c_int,
                    m as i64,
                    k as i64,
                    w,
                    n as i64,
                    bias.cast(),
                    out,
                    out_dt as c_int,
                )
            },
            "linear",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layer_norm(
        &self,
        x: P,
        x_dt: Dt,
        m: usize,
        c: usize,
        weight: P,
        bias: P,
        eps: f32,
        out: M,
        out_dt: Dt,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.layer_norm)(
                    self.ctx,
                    x,
                    x_dt as c_int,
                    m as i64,
                    c as i64,
                    weight.cast(),
                    bias.cast(),
                    eps,
                    out,
                    out_dt as c_int,
                )
            },
            "layer norm",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gelu(&self, x: M, dt: Dt, n: usize, tanh: bool) -> Result<()> {
        // SAFETY: x holds n values.
        self.ok(
            unsafe { (self.k.gelu)(self.ctx, x, dt as c_int, n as i64, tanh as c_int) },
            "gelu",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gate_add(
        &self,
        x: M,
        x_dt: Dt,
        m: usize,
        c: usize,
        other: P,
        other_dt: Dt,
        rows: P,
        gate: P,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.gate_add)(
                    self.ctx,
                    x,
                    x_dt as c_int,
                    m as i64,
                    c as i64,
                    other,
                    other_dt as c_int,
                    rows.cast(),
                    gate.cast(),
                )
            },
            "gated add",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_causal_rows(
        &self,
        q: P,
        k: P,
        v: P,
        dt: Dt,
        row0: usize,
        rows: usize,
        hq: usize,
        hkv: usize,
        d: usize,
        qs: usize,
        kvs: usize,
        out: M,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.attention_causal_rows)(
                    self.ctx,
                    q,
                    k,
                    v,
                    dt as c_int,
                    row0 as i64,
                    rows as i64,
                    hq as i64,
                    hkv as i64,
                    d as i64,
                    qs as i64,
                    kvs as i64,
                    out,
                )
            },
            "causal attention (rows)",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_qk(
        &self,
        q: P,
        sq: usize,
        qs: usize,
        k: P,
        v: P,
        skv: usize,
        kvs: usize,
        dt: Dt,
        h: usize,
        d: usize,
        out: M,
        out_dt: Dt,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.attention_qk)(
                    self.ctx,
                    q,
                    sq as i64,
                    qs as i64,
                    k,
                    v,
                    skv as i64,
                    kvs as i64,
                    dt as c_int,
                    h as i64,
                    d as i64,
                    out,
                    out_dt as c_int,
                )
            },
            "attention",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: P,
        k: P,
        v: P,
        dt: Dt,
        s: usize,
        h: usize,
        d: usize,
        stride: usize,
        out: M,
        out_dt: Dt,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.attention)(
                    self.ctx,
                    q,
                    k,
                    v,
                    dt as c_int,
                    s as i64,
                    h as i64,
                    d as i64,
                    stride as i64,
                    out,
                    out_dt as c_int,
                )
            },
            "attention",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv1d(
        &self,
        x: P,
        b: usize,
        ci: usize,
        l: usize,
        w: P,
        co: usize,
        k: usize,
        bias: P,
        stride: usize,
        dil: usize,
        pad: usize,
        out: M,
        lo: usize,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.conv1d)(
                    self.ctx,
                    x.cast(),
                    b as i64,
                    ci as i64,
                    l as i64,
                    w.cast(),
                    co as i64,
                    k as i64,
                    bias.cast(),
                    stride as i64,
                    dil as i64,
                    pad as i64,
                    out.cast(),
                    lo as i64,
                )
            },
            "conv1d",
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_mod(
        &self,
        x: P,
        x_dt: Dt,
        m: usize,
        c: usize,
        weight: P,
        eps: f32,
        rows: P,
        scale: P,
        shift: P,
        out: M,
        out_dt: Dt,
    ) -> Result<()> {
        // SAFETY: the caller's buffers hold the sizes named.
        self.ok(
            unsafe {
                (self.k.rms_norm_mod)(
                    self.ctx,
                    x,
                    x_dt as c_int,
                    m as i64,
                    c as i64,
                    weight.cast(),
                    eps,
                    rows.cast(),
                    scale.cast(),
                    shift.cast(),
                    out,
                    out_dt as c_int,
                )
            },
            "rms norm",
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn swiglu(&self, x: P, x_dt: Dt, m: usize, c: usize, out: M, out_dt: Dt) -> Result<()> {
        // SAFETY: x holds M x 2C values, out M x C.
        self.ok(
            unsafe {
                (self.k.swiglu)(
                    self.ctx,
                    x,
                    x_dt as c_int,
                    m as i64,
                    c as i64,
                    out,
                    out_dt as c_int,
                )
            },
            "swiglu",
        )
    }
}
