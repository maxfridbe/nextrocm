//! Private narrow binding: only symbols used by StarVector. No shared Nsd eager-load dependency.
#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::too_many_arguments)]
use nextrocm_engine_core::{Error, Gpu, Result};
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::Arc;
type P = *const c_void;
type M = *mut c_void;
type FfnSmallM = unsafe extern "C" fn(M, P, c_int, i64, i64, P, i64, *const f32, M, c_int) -> c_int;
fn resolve_ffn(enabled: bool, symbol: M) -> Result<Option<FfnSmallM>> {
    if !enabled {
        return Ok(None);
    }
    if symbol.is_null() {
        return Err(Error(
            "NR_STARVECTOR_FFN_SMALL_M=1 requires optional nrsv_ffn_small_m_apply ABI".into(),
        ));
    }
    // SAFETY: exact private starvector.h signature; the runtime stays loaded.
    Ok(Some(unsafe { std::mem::transmute::<M, FfnSmallM>(symbol) }))
}
fn ffn_shape(dt: Dt, m: usize, k: usize, n: usize, odt: Dt) -> bool {
    dt == Dt::F16
        && odt == Dt::F16
        && matches!(m, 1 | 2 | 8)
        && matches!((k, n), (2048, 8192) | (8192, 2048))
}

macro_rules! profile_op {
    ($this:expr, $label:expr, $body:expr) => {{
        #[cfg(feature = "operator-profile")]
        {
            crate::profile::measure(&$this.gpu, $label, || $body)
        }
        #[cfg(not(feature = "operator-profile"))]
        {
            $body
        }
    }};
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Dt {
    F32 = 0,
    F16 = 1,
}
struct Api {
    #[cfg(feature = "vision-f32")]
    quick_gelu_f32: unsafe extern "C" fn(M, *mut f32, i64) -> c_int,
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
    attention: unsafe extern "C" fn(M, P, P, P, c_int, i64, i64, i64, i64, M, c_int) -> c_int,
}
fn api(gpu: &nextrocm_engine_core::Gpu) -> Result<&'static Api> {
    static API: nextrocm_engine_core::bindings::PerGpu<Api> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("starvector", gpu).map_err(|e| e.0)?;
        macro_rules! sym {($name:literal)=>{{
   let p=lib.symbol($name);
   if p.is_null(){return Err(format!("missing StarVector dependency {}",$name));}
   // SAFETY: symbol signatures are copied from nsd.h / private starvector.h; library stays loaded.
   #[allow(clippy::missing_transmute_annotations)]
   unsafe {std::mem::transmute::<*mut c_void,_>(p)}
  }};}
        Ok(Api {
            #[cfg(feature = "vision-f32")]
            quick_gelu_f32: sym!("nrsv_quick_gelu_f32"),
            last_error: sym!("nsd_last_error"),
            create: sym!("nsd_create"),
            destroy: sym!("nsd_destroy"),
            wait: sym!("nsd_wait"),
            linear: sym!("nsd_linear"),
            layer_norm: sym!("nsd_layer_norm"),
            gelu: sym!("nsd_gelu"),
            gate_add: sym!("nsd_gate_add"),
            attention_causal_rows: sym!("nrsv_attention_causal_rows"),
            attention: sym!("nrsv_attention"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}
/// The kernels' context on one GPU
pub struct Nsd {
    ctx: M,
    k: &'static Api,
    // Immutable per-engine opt-in. Missing optional ABI is harmless only when off.
    ffn_small_m: Option<FfnSmallM>,
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
    #[cfg(feature = "vision-f32")]
    pub fn quick_gelu_f32(&self, x: M, count: usize) -> Result<()> {
        profile_op!(self, format!("quick_gelu_f32:n={count}"), {
            // SAFETY: caller supplies count initialized F32 elements on this GPU.
            self.ok(
                unsafe { (self.k.quick_gelu_f32)(self.gpu.raw(), x.cast(), count as i64) },
                "F32 QuickGELU",
            )
        })
    }
    /// A context on `gpu`'s own stream
    pub fn new(gpu: &Arc<Gpu>, ffn_enabled: bool) -> Result<Nsd> {
        let k = api(gpu)?;
        let symbol = if ffn_enabled {
            nextrocm_engine_core::bindings::Library::new("starvector", gpu)?.symbol("nrsv_ffn_small_m_apply")
        } else {
            std::ptr::null_mut()
        };
        let ffn_small_m = resolve_ffn(ffn_enabled, symbol)?;
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
            ffn_small_m,
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
    pub fn ffn_small_m_enabled(&self) -> bool {
        self.ffn_small_m.is_some()
    }

    pub fn raw(&self) -> M {
        self.ctx
    }

    pub fn wait(&self) -> Result<()> {
        // SAFETY: a live context.
        self.ok(unsafe { (self.k.wait)(self.ctx) }, "waiting for the GPU")
    }

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
        profile_op!(
            self,
            format!("linear:m={m}:k={k}:n={n}:dt={dt:?}:out={out_dt:?}"),
            {
                if let Some(ffn) = self.ffn_small_m.filter(|_| ffn_shape(dt, m, k, n, out_dt)) {
                    // SAFETY: same validated linear buffers, exact guarded half FFN geometry.
                    let rc = unsafe {
                        ffn(
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
                    };
                    // Only unsupported status 1 falls back; errors must propagate.
                    if rc != 1 {
                        return self.ok(rc, "private small-M FFN");
                    }
                }
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
        )
    }
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
        profile_op!(self, format!("norm:m={m}:c={c}:out={out_dt:?}"), {
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
        })
    }
    pub fn gelu(&self, x: M, dt: Dt, n: usize, tanh: bool) -> Result<()> {
        profile_op!(self, format!("gelu:n={n}:dt={dt:?}"), {
            // SAFETY: x holds n values.
            self.ok(
                unsafe { (self.k.gelu)(self.ctx, x, dt as c_int, n as i64, tanh as c_int) },
                "gelu",
            )
        })
    }
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
        profile_op!(self, format!("residual:m={m}:c={c}"), {
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
        })
    }
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
        profile_op!(
            self,
            format!(
                "causal_attention:rows={rows}:keys={}:heads={hq}:dim={d}",
                row0 + rows
            ),
            {
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
        )
    }
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
        profile_op!(self, format!("attention:rows={s}:heads={h}:dim={d}"), {
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
        })
    }
}

#[cfg(test)]
mod ffn_tests {
    use super::*;
    #[test]
    fn missing_optional_abi_is_error_only_when_requested() {
        assert!(resolve_ffn(false, std::ptr::null_mut()).unwrap().is_none());
        assert!(resolve_ffn(true, std::ptr::null_mut()).is_err());
    }
    #[test]
    fn only_actual_half_ffn_shapes_dispatch() {
        for m in [1, 2, 8] {
            assert!(ffn_shape(Dt::F16, m, 2048, 8192, Dt::F16));
            assert!(ffn_shape(Dt::F16, m, 8192, 2048, Dt::F16));
        }
        for (m, k, n) in [
            (3, 2048, 8192),
            (257, 2048, 8192),
            (259, 8192, 2048),
            (1, 2048, 2048),
            (2, 2048, 128),
            (1, 2048, 49156),
            (8, 2049, 8192),
        ] {
            assert!(!ffn_shape(Dt::F16, m, k, n, Dt::F16));
        }
        assert!(!ffn_shape(Dt::F32, 1, 2048, 8192, Dt::F16));
        assert!(!ffn_shape(Dt::F16, 1, 2048, 8192, Dt::F32));
    }
}
