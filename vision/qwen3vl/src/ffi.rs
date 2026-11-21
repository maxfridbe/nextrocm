use nextrocm_engine_core::{Error, Result};
use std::ffi::{c_int, c_void};
use std::sync::OnceLock;
type G = *mut c_void;
type P = *const c_void;
type F = *const f32;
pub type Rope = unsafe extern "C" fn(G, *mut f32, i64, i64, i64, F, F) -> c_int;
pub type Rms4096 = unsafe extern "C" fn(G, F, i64, F, *mut f32) -> c_int;
pub type SiluSource = unsafe extern "C" fn(G, F, i64, i64, *mut f32) -> c_int;
pub type AttentionSource32 = unsafe extern "C" fn(G, F, i64, *mut f32, *mut u32) -> c_int;
pub type AttentionProb = unsafe extern "C" fn(G, F, i64, *mut f32) -> c_int;
#[repr(C)]
pub struct Int8Args {
    pub x: P,
    pub weight: *const i8,
    pub weight_scale: F,
    pub bias: P,
    pub out: G,
    pub rotated: *mut f32,
    pub quantized: *mut i8,
    pub row_scale: *mut f32,
    pub accum: *mut i32,
    pub status: *mut u32,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    pub scales: i64,
    pub dtype: c_int,
}
pub struct Api {
    pub attention_order32: Option<AttentionSource32>,
    pub silu_source: Option<SiluSource>,
    pub attention_source32: Option<AttentionSource32>,
    pub rms4096: Option<Rms4096>,
    pub attention_divide: Option<AttentionProb>,
    pub attention_tree32: Option<AttentionProb>,
    pub attention_prob: Option<AttentionProb>,
    pub ordered_rope: Option<Rope>,
    pub linear: unsafe extern "C" fn(G, *const Int8Args) -> c_int,
    pub rope: unsafe extern "C" fn(G, *mut f32, i64, i64, i64, F, F) -> c_int,
    pub rms:
        unsafe extern "C" fn(G, P, c_int, i64, i64, F, f32, *const i32, F, F, G, c_int) -> c_int,
    pub attention:
        unsafe extern "C" fn(G, P, P, P, c_int, i64, i64, i64, i64, i64, i64, i64, G) -> c_int,
    pub swiglu: unsafe extern "C" fn(G, P, c_int, i64, i64, G, c_int) -> c_int,
    pub add: unsafe extern "C" fn(G, G, c_int, i64, i64, P, c_int, *const i32, F) -> c_int,
}
pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextrocm_engine_core::api().map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("missing private TE dependency {}", $name));
                }
                // SAFETY: exact private qv.h or existing speech.hip declaration.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        let probability = lib.symbol("nrqv_attention_prob_f32");
        // SAFETY: optional symbol has the exact qv.h signature.
        let attention_prob = if probability.is_null() {
            None
        } else {
            Some(unsafe { std::mem::transmute::<*mut c_void, AttentionProb>(probability) })
        };
        let ordered = lib.symbol("nrqv_rms_rope_ordered_f32");
        let ordered_rope = if ordered.is_null() {
            None
        } else {
            // SAFETY: optional symbol has the exact private qv.h Rope signature.
            Some(unsafe { std::mem::transmute::<*mut c_void, Rope>(ordered) })
        };
        let rms = lib.symbol("nrqv_rms4096_strict_f32");
        let rms4096 = if rms.is_null() {
            None
        } else {
            // SAFETY: optional symbol has the exact private qv.h signature.
            Some(unsafe { std::mem::transmute::<*mut c_void, Rms4096>(rms) })
        };
        let divide = lib.symbol("nrqv_attention_divide_f32");
        let attention_divide = if divide.is_null() {
            None
        } else {
            // SAFETY: optional symbol has the exact private qv.h signature.
            Some(unsafe { std::mem::transmute::<*mut c_void, AttentionProb>(divide) })
        };
        let tree = lib.symbol("nrqv_attention_tree32_f32");
        let attention_tree32 = if tree.is_null() {
            None
        } else {
            // SAFETY: exact optional attention_tree32.h F32 ABI.
            Some(unsafe { std::mem::transmute::<*mut c_void, AttentionProb>(tree) })
        };
        let source = lib.symbol("nrqv_attention_source32_f32");
        let attention_source32 = if source.is_null() {
            None
        } else {
            // SAFETY: exact optional attention_source32.h ABI, including caller-owned status.
            Some(unsafe { std::mem::transmute::<*mut c_void, AttentionSource32>(source) })
        };
        let ordered = lib.symbol("nrqv_attention_order32_f32");
        let attention_order32 = if ordered.is_null() {
            None
        } else {
            // SAFETY: exact optional attention_order32.h ABI, including caller-owned status.
            Some(unsafe { std::mem::transmute::<*mut c_void, AttentionSource32>(ordered) })
        };
        let silu = lib.symbol("nrqv_silu_source_f32");
        let silu_source = if silu.is_null() {
            None
        } else {
            // SAFETY: exact optional silu_source.h ABI; IEEE outputs have no status buffer.
            Some(unsafe { std::mem::transmute::<*mut c_void, SiluSource>(silu) })
        };
        Ok(Api {
            attention_order32,
            silu_source,
            attention_source32,
            attention_tree32,
            rms4096,
            attention_divide,
            ordered_rope,
            attention_prob,
            linear: sym!("nrqv_int8_convrot_linear"),
            rope: sym!("nrqv_rms_rope_f32"),
            rms: sym!("nsd_rms_norm_mod"),
            attention: sym!("nsd_attention_causal_rows"),
            swiglu: sym!("nsd_swiglu"),
            add: sym!("nsd_gate_add"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}
pub fn check(rc: c_int, stage: &str) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(Error(format!(
            "{stage}: {}",
            nextrocm_engine_core::api()?.error()
        )))
    }
}
