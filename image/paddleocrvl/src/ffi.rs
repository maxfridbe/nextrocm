//! This engine's own C ABI (`kernels/image/paddleocrvl/pocr.h`), bound from the image kernel library by name.

use std::ffi::{c_int, c_void};

type G = *mut c_void;
type F = *const f32;
type FM = *mut f32;

pub type VisionTiled = unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, FM) -> c_int;

pub struct Api {
    pub vision_tiled: Option<VisionTiled>,
    pub linear: unsafe extern "C" fn(G, *const c_void, i64, i64, *const c_void, i64, FM) -> c_int,
    pub dequant: unsafe extern "C" fn(G, *const c_void, c_int, i64, *mut c_void, c_int) -> c_int,
    pub rope: unsafe extern "C" fn(G, FM, i64, i64, i64, i64, F) -> c_int,
    pub attention:
        unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, i64, i64, FM, c_int) -> c_int,
    pub to_half: unsafe extern "C" fn(G, F, *mut c_void, i64) -> c_int,
    /// (src, src stride, dst, dst stride, rows, cols)
    pub copy_rows: unsafe extern "C" fn(G, F, i64, FM, i64, i64, i64) -> c_int,
    /// (x, H, W, C, out)
    pub merge2: unsafe extern "C" fn(G, F, i64, i64, i64, FM) -> c_int,
    /// (n, Hq, D): the scratch attn1 needs, in floats
    pub attn1_scratch: unsafe extern "C" fn(i64, i64, i64) -> i64,
    /// (q, k, v, n, kv stride, Hq, Hkv, D, scratch, out)
    pub attn1: unsafe extern "C" fn(G, F, F, F, i64, i64, i64, i64, i64, FM, FM) -> c_int,
}

/// The table, bound once (an error when the library lacks a symbol: an older build)
pub fn api(gpu: &nextrocm_engine_core::Gpu) -> nextrocm_engine_core::Result<&'static Api> {
    static API: nextrocm_engine_core::bindings::PerGpu<Api> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("paddleocrvl", gpu).map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!(
                        "{} is not in the kernel library (rebuild it: ./build.sh kernels)",
                        $name
                    ));
                }
                // SAFETY: the symbol is declared in pocr.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        let tiled = lib.symbol("ns_image_pocr_attention_tiled");
        // SAFETY: optional private symbol has the exact signature in pocr.h.
        let vision_tiled = if tiled.is_null() { None } else {
            Some(unsafe { std::mem::transmute::<*mut c_void, VisionTiled>(tiled) })
        };
        Ok(Api {
            vision_tiled,
            linear: sym!("ns_image_pocr_linear"),
            dequant: sym!("ns_image_pocr_dequant"),
            rope: sym!("ns_image_pocr_rope"),
            attention: sym!("ns_image_pocr_attention"),
            to_half: sym!("ns_image_pocr_to_half"),
            copy_rows: sym!("ns_image_pocr_copy_rows"),
            merge2: sym!("ns_image_pocr_merge2"),
            attn1_scratch: sym!("ns_image_pocr_attn1_scratch"),
            attn1: sym!("ns_image_pocr_attn1"),
        })
    })
    .as_ref()
    .map_err(|e| nextrocm_engine_core::Error(e.clone()))
}

/// A call's return code as a result (the reason from the library)
pub fn check(rc: c_int, what: &str) -> nextrocm_engine_core::Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextrocm_engine_core::api()
        .map(|a| a.error())
        .unwrap_or_default();
    Err(nextrocm_engine_core::Error(format!("{what}: {why}")))
}
