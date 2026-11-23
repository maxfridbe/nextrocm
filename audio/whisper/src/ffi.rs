//! The one kernel of the audio library this engine needs beyond the shared diffusion ones (nsd: products, norms,
//! GELU, attention, the 1-D convolution): the MiniMax Music engine's float32 transpose (`kernels/audio/minimaxmusic3/
//! mm3.h`), which turns the convolutions' channels-first output into the transformer's rows.

use std::ffi::{c_int, c_void};

type G = *mut c_void;
pub type AttentionTiled = unsafe extern "C" fn(
    G,
    *const c_void,
    *const c_void,
    *const c_void,
    i64,
    i64,
    i64,
    i64,
    *mut c_void,
) -> c_int;
pub type LinearF16Tiled = unsafe extern "C" fn(
    G,
    *const c_void,
    i64,
    i64,
    *const c_void,
    i64,
    *const f32,
    *mut c_void,
) -> c_int;

/// Default-off load control, parsed without changing process environment.
pub fn tiled_setting(value: Option<&str>) -> nextrocm_engine_core::Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(nextrocm_engine_core::Error(
            "NS_WHISPER_LINEAR_F16_TILED must be 0 or 1".into(),
        )),
    }
}

/// Initial dispatch is limited to the six validated encoder operator variants.
/// Exact constants also bound every byte span/cast well below the header cap.
pub fn tiled_shape(m: usize, n: usize, k: usize) -> bool {
    m == 1500 && matches!((n, k), (1280, 1280) | (5120, 1280) | (1280, 5120))
}

pub fn attention_setting(value: Option<&str>) -> nextrocm_engine_core::Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err(nextrocm_engine_core::Error(
            "NS_WHISPER_ATTENTION_TILED must be 0 or 1".into(),
        )),
    }
}

pub fn attention_shape(s: usize, h: usize, d: usize, stride: usize) -> bool {
    (s, h, d, stride) == (1500, 20, 64, 1280)
}

pub struct Api {
    pub transpose: unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int,
    pub linear_f16_tiled: Option<LinearF16Tiled>,
    pub attention_tiled: Option<AttentionTiled>,
}

// SAFETY: function pointers into the library, which stays loaded for the process.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

pub fn api(gpu: &nextrocm_engine_core::Gpu) -> nextrocm_engine_core::Result<&'static Api> {
    static API: nextrocm_engine_core::bindings::PerGpu<Api> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("whisper", gpu).map_err(|e| e.0)?;
        let p = lib.symbol("ns_audio_mm3_transpose");
        if p.is_null() {
            return Err("ns_audio_mm3_transpose is not in the kernel library (the audio library, rebuilt: ./build.sh kernels)".into());
        }
        let tiled = lib.symbol("ns_audio_whisper_linear_f16_tiled");
        let linear_f16_tiled = if tiled.is_null() { None } else {
            // SAFETY: private whisper.h declares this signature; library outlives Api.
            Some(unsafe { std::mem::transmute::<*mut c_void, LinearF16Tiled>(tiled) })
        };
        let attention = lib.symbol("ns_audio_whisper_attention_tiled");
        let attention_tiled = if attention.is_null() { None } else {
            // SAFETY: private whisper.h ABI; loaded library outlives Api.
            Some(unsafe { std::mem::transmute::<*mut c_void, AttentionTiled>(attention) })
        };
        // SAFETY: declared in mm3.h with this signature.
        Ok(Api { transpose: unsafe { std::mem::transmute::<*mut c_void, unsafe extern "C" fn(G, *const f32, i64, i64, *mut f32) -> c_int>(p) }, linear_f16_tiled, attention_tiled })
    })
    .as_ref()
    .map_err(|e| nextrocm_engine_core::Error(e.clone()))
}

#[cfg(test)]
mod tiled_tests {
    use super::*;
    #[test]
    fn attention_control_and_encoder_gate() {
        assert!(!attention_setting(None).unwrap());
        assert!(!attention_setting(Some("0")).unwrap());
        assert!(attention_setting(Some("1")).unwrap());
        for value in ["", "true", "2", " 1"] {
            assert!(attention_setting(Some(value)).is_err());
        }
        assert!(attention_shape(1500, 20, 64, 1280));
        for shape in [
            (1, 20, 64, 1280),
            (1499, 20, 64, 1280),
            (1500, 20, 32, 1280),
            (1500, 19, 64, 1280),
            (1500, 20, 64, 1281),
            (usize::MAX, 20, 64, 1280),
        ] {
            assert!(!attention_shape(shape.0, shape.1, shape.2, shape.3));
        }
    }
    #[test]
    fn load_control_is_explicit() {
        assert!(!tiled_setting(None).unwrap());
        assert!(!tiled_setting(Some("0")).unwrap());
        assert!(tiled_setting(Some("1")).unwrap());
        for value in ["", "true", "2", " 1"] {
            assert!(tiled_setting(Some(value)).is_err());
        }
    }
    #[test]
    fn only_profiled_encoder_shapes_select() {
        for (n, k) in [(1280, 1280), (5120, 1280), (1280, 5120)] {
            assert!(tiled_shape(1500, n, k));
            for m in [0, 1, 448, 1499, 1501, usize::MAX] {
                assert!(!tiled_shape(m, n, k));
            }
        }
        for (n, k) in [
            (0, 1280),
            (1280, 0),
            (5120, 5120),
            (1281, 1280),
            (usize::MAX, usize::MAX),
        ] {
            assert!(!tiled_shape(1500, n, k));
        }
    }
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
