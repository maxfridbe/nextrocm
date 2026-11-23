//! The kernels this engine runs, bound from the audio kernel library by name: the language models' decode on the
//! MiniMax Music engine's (`kernels/audio/minimaxmusic3/mm3.h`: small-batch products, cached attention, conversions,
//! the oneDNN convolutions), the local transformers' attention, the long RoPE and the AudioVAE's own on this engine's
//! (`kernels/audio/voxcpm2/vcp.h`); SiLU from Qwen3-TTS's (`q3t.h`).

use std::ffi::{c_int, c_void};

type G = *mut c_void;
type M = *mut c_void;
type P = *const c_void;

pub type LinearF16Tiled =
    unsafe extern "C" fn(G, P, i64, i64, P, i64, *const f32, *mut f32) -> c_int;
pub type Rope =
    unsafe extern "C" fn(G, *mut f32, i64, i64, i64, i64, i64, *const f32, i64) -> c_int;

pub type ConvUpTiled = unsafe extern "C" fn(
    G,
    *const f32,
    i64,
    i64,
    i64,
    *const f32,
    i64,
    i64,
    *const f32,
    i64,
    *mut f32,
) -> c_int;

pub type CachedAttention = unsafe extern "C" fn(
    G,
    *const f32,
    i64,
    P,
    P,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    *mut f32,
    *mut f32,
) -> c_int;

pub type KvStore = unsafe extern "C" fn(
    G,
    *const f32,
    *const f32,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    i64,
    M,
    M,
) -> c_int;

#[allow(clippy::type_complexity)]
pub struct Api {
    pub linear_f16_tiled: Option<LinearF16Tiled>,
    pub rope_bf16_cache: Option<Rope>,
    // Optional so the default path still loads older baseline audio libraries.
    pub conv_up_tiled: Option<ConvUpTiled>,
    pub gemv: unsafe extern "C" fn(
        G,
        *const f32,
        i64,
        i64,
        P,
        c_int,
        *const f32,
        i64,
        *const f32,
        *mut f32,
        i64,
        c_int,
    ) -> c_int,
    pub attn_scratch: unsafe extern "C" fn(i64, i64, i64, i64, i64) -> i64,
    pub attn: CachedAttention,
    pub attn_f32: Option<CachedAttention>,
    pub kv_store: KvStore,
    pub kv_store_f32: Option<KvStore>,
    pub to_half: unsafe extern "C" fn(G, *const f32, M, i64) -> c_int,
    pub bf16_to_float: unsafe extern "C" fn(G, *const u16, *mut f32, i64) -> c_int,
    pub quant_rows: unsafe extern "C" fn(G, *const f32, i64, i64, *mut i8, *mut f32) -> c_int,
    pub dequant_rows: unsafe extern "C" fn(G, *const i8, *const f32, i64, i64, M) -> c_int,
    pub add: unsafe extern "C" fn(G, *mut f32, *const f32, i64) -> c_int,
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub conv1d_lr: unsafe extern "C" fn(
        G,
        c_int,
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
        i64,
        *mut f32,
        i64,
    ) -> c_int,
    pub tanh: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub silu: unsafe extern "C" fn(G, *mut f32, i64) -> c_int,
    pub rope: Rope,
    pub local_attn: unsafe extern "C" fn(
        G,
        *const f32,
        *const f32,
        *const f32,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
        *mut f32,
    ) -> c_int,
    pub snake: unsafe extern "C" fn(G, *const f32, i64, i64, *const f32, *mut f32) -> c_int,
    pub dwconv: unsafe extern "C" fn(
        G,
        *const f32,
        i64,
        i64,
        *const f32,
        *const f32,
        i64,
        i64,
        *mut f32,
    ) -> c_int,
    pub affine: unsafe extern "C" fn(G, *mut f32, i64, i64, *const f32, *const f32) -> c_int,
    pub fsq: unsafe extern "C" fn(G, *mut f32, i64, f32) -> c_int,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api(gpu: &nextrocm_engine_core::Gpu) -> nextrocm_engine_core::Result<&'static Api> {
    static API: nextrocm_engine_core::bindings::PerGpu<Api> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("voxcpm2", gpu).map_err(|e| e.0)?;
        macro_rules! sym {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!("{} is not in the kernel library (the audio library, rebuilt: ./build.sh kernels)", $name));
                }
                // SAFETY: the symbol is declared in mm3.h / q3t.h / vcp.h with the field's signature.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        Ok(Api {
            linear_f16_tiled: {
                let p = lib.symbol("ns_audio_vcp_linear_f16_tiled");
                if p.is_null() { None } else {
                    // SAFETY: vcp.h declares LinearF16Tiled; library stays loaded.
                    Some(unsafe { std::mem::transmute::<*mut c_void, LinearF16Tiled>(p) })
                }
            },
            rope_bf16_cache: {
                let p = lib.symbol("ns_audio_vcp_rope_bf16_cache");
                if p.is_null() { None } else {
                    // SAFETY: vcp.h declares Rope; library stays loaded.
                    Some(unsafe { std::mem::transmute::<*mut c_void, Rope>(p) })
                }
            },
            conv_up_tiled: {
                let p = lib.symbol("ns_audio_vcp_conv_up_tiled");
                if p.is_null() {
                    None
                } else {
                    // SAFETY: vcp.h declares exactly ConvUpTiled; library stays loaded.
                    Some(unsafe { std::mem::transmute::<*mut c_void, ConvUpTiled>(p) })
                }
            },
            gemv: sym!("ns_audio_mm3_gemv"),
            attn_scratch: sym!("ns_audio_mm3_attn_scratch"),
            attn_f32: {
                let p = lib.symbol("ns_audio_vcp_attn_f32");
                if p.is_null() { None } else {
                    // SAFETY: vcp.h declares CachedAttention; library stays loaded.
                    Some(unsafe { std::mem::transmute::<*mut c_void, CachedAttention>(p) })
                }
            },
            attn: sym!("ns_audio_mm3_attn"),
            kv_store_f32: {
                let p = lib.symbol("ns_audio_vcp_kv_store_f32");
                if p.is_null() { None } else {
                    // SAFETY: vcp.h declares KvStore; library stays loaded.
                    Some(unsafe { std::mem::transmute::<*mut c_void, KvStore>(p) })
                }
            },
            kv_store: sym!("ns_audio_mm3_kv_store"),
            to_half: sym!("ns_audio_mm3_to_half"),
            bf16_to_float: sym!("ns_audio_mm3_bf16_to_float"),
            quant_rows: sym!("ns_audio_mm3_quant_rows"),
            dequant_rows: sym!("ns_audio_mm3_dequant_rows"),
            add: sym!("ns_audio_mm3_add"),
            transpose: sym!("ns_audio_mm3_transpose"),
            conv1d_lr: sym!("ns_audio_mm3_conv1d_lr"),
            tanh: sym!("ns_audio_mm3_tanh"),
            silu: sym!("ns_audio_q3t_silu"),
            rope: sym!("ns_audio_vcp_rope"),
            local_attn: sym!("ns_audio_vcp_local_attn"),
            snake: sym!("ns_audio_vcp_snake"),
            dwconv: sym!("ns_audio_vcp_dwconv"),
            affine: sym!("ns_audio_vcp_affine"),
            fsq: sym!("ns_audio_vcp_fsq"),
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
