//! Private symbols; engine validates geometry, matrix extents and buffer views before dispatch.
use nextrocm_engine_core::{api, Error, Result};
use std::ffi::c_void;
type P = *mut c_void;
type F = *mut f32;
pub(crate) type MvFast = unsafe extern "C" fn(P, P, i32, i32, i32, F, F, i32) -> i32;
pub(crate) type MvBatch =
    unsafe extern "C" fn(P, P, i32, i32, i32, u64, F, i32, u64, F, u64) -> i32;
pub(crate) struct Kernels {
    pub mv_fast: Option<MvFast>,
    pub mv_batch: Option<MvBatch>,
    pub embed: unsafe extern "C" fn(P, P, i32, i32, F) -> i32,
    pub mv: unsafe extern "C" fn(P, P, i32, i32, i32, F, F) -> i32,
    pub norm: unsafe extern "C" fn(P, F, F, i32, f32, F) -> i32,
    pub axpy: unsafe extern "C" fn(P, F, F, i32, f32) -> i32,
    pub swiglu: unsafe extern "C" fn(P, F, F, i32) -> i32,
    pub qk: unsafe extern "C" fn(P, F, F, i32, i32, i32, i32, i32, f32, f32, F) -> i32,
    pub attn: unsafe extern "C" fn(P, F, F, F, P, P, i32, i32, i32, i32, F) -> i32,
    pub gate: unsafe extern "C" fn(P, F, F, i32, i32) -> i32,
    pub conv: unsafe extern "C" fn(P, F, F, F, i32, i32, F) -> i32,
    pub delta: unsafe extern "C" fn(P, F, F, F, F, F, F, i32, i32, i32, f32, F) -> i32,
    pub gated_norm: unsafe extern "C" fn(P, F, F, F, i32, i32, f32, F) -> i32,
}
impl Kernels {
    pub fn load() -> Result<Self> {
        let a = api()?;
        macro_rules! sym {
            ($s:literal) => {{
                let p = a.symbol($s);
                if p.is_null() {
                    return Err(Error(format!(
                        "missing {}: integrate kernels/llm/qwen35/qwen35.hip",
                        $s
                    )));
                }
                // SAFETY: exact signatures generated from qwen35.h, library lives for process.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        let batch = a.symbol("nq35_mv_batch");
        let fast = a.symbol("nq35_mv_fast");
        let mv_fast = if fast.is_null() {
            None
        } else {
            // SAFETY: exact private scalar_fast.h ABI; library retained for process lifetime.
            Some(unsafe { std::mem::transmute::<*mut c_void, MvFast>(fast) })
        };
        let mv_batch = if batch.is_null() {
            None
        } else {
            // SAFETY: exact private batch.h ABI; library retained for process lifetime.
            Some(unsafe { std::mem::transmute::<*mut c_void, MvBatch>(batch) })
        };
        Ok(Self {
            mv_fast,
            mv_batch,
            embed: sym!("nq35_embed"),
            mv: sym!("nq35_mv"),
            norm: sym!("nq35_norm"),
            axpy: sym!("nq35_axpy"),
            swiglu: sym!("nq35_swiglu"),
            qk: sym!("nq35_qk"),
            attn: sym!("nq35_attn"),
            gate: sym!("nq35_gate"),
            conv: sym!("nq35_conv"),
            delta: sym!("nq35_delta"),
            gated_norm: sym!("nq35_gated_norm"),
        })
    }
}
pub(crate) fn check(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!("qwen35 HIP: {}", api()?.error())))
    }
}
