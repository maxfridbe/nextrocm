//! Private symbols; engine validates geometry, matrix extents and buffer views before dispatch.
use nextrocm_engine_core::{api, Error, Result};
use std::ffi::c_void;
type P = *mut c_void;
type F = *mut f32;
pub(crate) struct Kernels {
    pub embed: unsafe extern "C" fn(P, P, i32, i32, F) -> i32,
    pub mv: unsafe extern "C" fn(P, P, i32, i32, i32, F, F) -> i32,
    pub norm: unsafe extern "C" fn(P, F, F, i32, f32, F) -> i32,
    pub axpy: unsafe extern "C" fn(P, F, F, i32, f32) -> i32,
    pub swiglu: unsafe extern "C" fn(P, F, F, i32) -> i32,
    pub qk: unsafe extern "C" fn(P, F, F, i32, i32, i32, i32, i32, f32, f32, F) -> i32,
    pub attn: unsafe extern "C" fn(P, F, F, F, P, P, i32, i32, i32, i32, F) -> i32,
}
impl Kernels {
    pub fn load() -> Result<Self> {
        let a = api()?;
        macro_rules! sym {
            ($s:literal) => {{
                let p = a.symbol($s);
                if p.is_null() {
                    return Err(Error(format!(
                        "missing {}: integrate kernels/llm/qwen3moe/qwen3moe.hip",
                        $s
                    )));
                }
                // SAFETY: exact signatures generated from qwen3moe.h, library lives for process.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Self {
            embed: sym!("nq3m_embed"),
            mv: sym!("nq3m_mv"),
            norm: sym!("nq3m_norm"),
            axpy: sym!("nq3m_axpy"),
            swiglu: sym!("nq3m_swiglu"),
            qk: sym!("nq3m_qk"),
            attn: sym!("nq3m_attn"),
        })
    }
}
pub(crate) fn check(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!("qwen3moe HIP: {}", api()?.error())))
    }
}
