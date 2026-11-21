//! Exact function signatures from kernels/audio/lowvram/bridge.cpp.
use nextrocm_audio::{Error, Result};
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

type Handle = *mut c_void;
pub type Progress = unsafe extern "C" fn(*mut c_void, *const c_char, i64, i64) -> c_int;
type Open = unsafe extern "C" fn(Handle, c_int, *const c_char, c_int, *mut Handle) -> c_int;
type Info = unsafe extern "C" fn(Handle) -> *const c_char;
type Transcribe = unsafe extern "C" fn(
    Handle,
    *const f32,
    i64,
    c_int,
    *const c_char,
    c_int,
    Progress,
    *mut c_void,
    *mut *const c_char,
) -> c_int;
type Close = unsafe extern "C" fn(Handle);

pub struct Api {
    pub open: Open,
    pub info: Info,
    pub transcribe: Transcribe,
    pub close: Close,
}

pub fn api() -> Result<&'static Api> {
    static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| {
        let lib = nextrocm_engine_core::api().map_err(|e| e.0)?;
        let symbol = |name: &str| {
            let p = lib.symbol(name);
            if p.is_null() {
                Err(format!(
                    "{name} missing from native kernel library; rebuild with the lowvram bridge"
                ))
            } else {
                Ok(p)
            }
        };
        // SAFETY: these exact signatures are defined by bridge.cpp. The engine-core
        // API retains the containing library for the process lifetime.
        unsafe {
            Ok(Api {
                open: std::mem::transmute::<Handle, Open>(symbol("ns_audio_lowvram_open")?),
                info: std::mem::transmute::<Handle, Info>(symbol("ns_audio_lowvram_info")?),
                transcribe: std::mem::transmute::<Handle, Transcribe>(symbol(
                    "ns_audio_lowvram_transcribe",
                )?),
                close: std::mem::transmute::<Handle, Close>(symbol("ns_audio_lowvram_close")?),
            })
        }
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

pub fn check(rc: c_int, context: &str) -> Result<()> {
    if rc == 0 {
        return Ok(());
    }
    let why = nextrocm_engine_core::api()?.error();
    Err(Error(format!("{context}: {why}")))
}
