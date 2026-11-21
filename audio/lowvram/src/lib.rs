//! Resident native Canary-Qwen (GGUF) and Whisper (GGML) AudioEngine adapters.
//! Integration and native lifetime requirements: docs/lowvram-adapter.md.
mod ffi;
mod language;

use nextrocm_audio::{
    At, Audio, AudioEngine, AudioKind, AudioRequest, Defaults, EngineOption, Error, Gpu,
    LoadOptions, ModelFiles, Recogniser, Result, Step, Timed, TranscribeRequest, Transcript,
};
use serde_json::Value;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Native Canary architecture (single GGUF model role).
pub const CANARY_ARCH: &str = "canary";
/// GGML Whisper is distinct from the existing safetensors `whisper` architecture.
pub const WHISPER_ARCH: &str = "whisper-ggml";
/// Both backends take one checkpoint file.
pub const ROLES: &[&str] = &["model"];
/// Whisper can quantize eligible weights at load; Canary uses its GGUF quantization.
pub const WHISPER_OPTIONS: &[EngineOption] = &[EngineOption {
    name: "q8",
    env: "NS_WH_GGML_Q8",
    value: "0|1",
    at: At::Load,
    help: "quantize GGML Whisper weights to Q8 at load (default off)",
}];

/// Registry descriptor for the native Canary engine.
pub fn canary_kind() -> AudioKind {
    AudioKind {
        archs: &[CANARY_ARCH],
        name: "Canary-Qwen (native GGUF speech recognition)",
        roles: ROLES,
        load: load_canary,
        options: &[],
    }
}
/// Registry descriptor for native GGML Whisper, leaving safetensors Whisper intact.
pub fn whisper_kind() -> AudioKind {
    AudioKind {
        archs: &[WHISPER_ARCH],
        name: "Whisper (native GGML speech recognition)",
        roles: ROLES,
        load: load_whisper,
        options: WHISPER_OPTIONS,
    }
}

/// Load Canary using the standard AudioKind loader contract.
/// # Errors
/// Missing files/GPU, invalid native metadata, or a native loading failure.
pub fn load_canary(
    files: &ModelFiles,
    gpus: &[Arc<Gpu>],
    options: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn AudioEngine>> {
    load(files, gpus, options, log, false)
}
/// Load GGML Whisper using the standard AudioKind loader contract.
/// # Errors
/// Missing files/GPU, invalid q8 setting/native metadata, or a native loading failure.
pub fn load_whisper(
    files: &ModelFiles,
    gpus: &[Arc<Gpu>],
    options: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn AudioEngine>> {
    load(files, gpus, options, log, true)
}

struct Resident {
    handle: NonNull<c_void>,
    api: &'static ffi::Api,
    // Kept here so even metadata parsing failures close before dropping the GPU.
    gpu: Arc<Gpu>,
}
// SAFETY: ownership may move between threads; every native entry selects the GPU.
// Access is serialized by Engine::resident. The native API must not retain callbacks.
unsafe impl Send for Resident {}
impl Drop for Resident {
    fn drop(&mut self) {
        // SAFETY: uniquely owned live handle, no active call, GPU and library still alive.
        unsafe { (self.api.close)(self.handle.as_ptr()) }
    }
}

struct Engine {
    resident: Mutex<Resident>,
    capabilities: Recogniser,
    whisper: bool,
    load_s: f64,
    report: String,
}

fn load(
    files: &ModelFiles,
    gpus: &[Arc<Gpu>],
    options: &LoadOptions,
    log: &mut dyn FnMut(String),
    whisper: bool,
) -> Result<Box<dyn AudioEngine>> {
    let start = Instant::now();
    let gpu = gpus
        .first()
        .ok_or_else(|| Error("no GPU for native speech recognition".into()))?;
    if gpu.raw().is_null() {
        return Err(Error(
            "native speech recognition requires a live GPU context".into(),
        ));
    }
    let path = files
        .get("model")
        .ok_or_else(|| Error("missing model file role".into()))?;
    if !path.is_file() {
        return Err(Error(format!("model is not a file: {}", path.display())));
    }
    let path_c = CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| Error("model path contains NUL".into()))?;
    let q8 = if whisper {
        match options.setting("NS_WH_GGML_Q8").as_deref() {
            None | Some("0" | "false") => false,
            Some("1" | "true") => true,
            Some(v) => {
                return Err(Error(format!(
                    "invalid NS_WH_GGML_Q8 {v:?}: expected 0|1 or false|true"
                )))
            }
        }
    } else {
        false
    };
    let api = ffi::api()?;
    let mut handle = std::ptr::null_mut();
    // SAFETY: GPU is retained, path and output slot live through this synchronous call.
    let rc = unsafe {
        (api.open)(
            gpu.raw(),
            i32::from(whisper),
            path_c.as_ptr(),
            i32::from(q8),
            &mut handle,
        )
    };
    ffi::check(rc, "loading native speech model")?;
    let resident = Resident {
        handle: NonNull::new(handle)
            .ok_or_else(|| Error("native open succeeded with a null handle".into()))?,
        api,
        gpu: Arc::clone(gpu),
    };
    // SAFETY: uniquely owned handle; info remains valid until close and is copied here.
    let info = unsafe { copy_json((api.info)(resident.handle.as_ptr())) }?;
    let capabilities = parse_info(&info)?;
    let report = format!(
        "{}: {} on {}",
        if whisper { WHISPER_ARCH } else { CANARY_ARCH },
        path.display(),
        resident.gpu.name
    );
    log(report.clone());
    Ok(Box::new(Engine {
        resident: Mutex::new(resident),
        capabilities,
        whisper,
        load_s: start.elapsed().as_secs_f64(),
        report,
    }))
}

/// # Safety
/// `p` is null or a valid NUL-terminated bridge-owned string; its owner must
/// remain locked and alive throughout the copy.
unsafe fn copy_json(p: *const c_char) -> Result<Value> {
    if p.is_null() {
        return Err(Error("native bridge returned null JSON".into()));
    }
    // SAFETY: caller guarantees a readable NUL-terminated string.
    let bytes = unsafe { CStr::from_ptr(p) }.to_bytes();
    serde_json::from_slice(bytes).map_err(|e| Error(format!("invalid native bridge JSON: {e}")))
}

fn parse_info(v: &Value) -> Result<Recogniser> {
    let languages = v
        .get("languages")
        .and_then(Value::as_array)
        .ok_or_else(|| Error("native metadata lacks languages array".into()))?
        .iter()
        .map(|v| {
            v.as_str()
                .filter(|s| language::LANGUAGES.iter().any(|(c, _)| c == s))
                .map(str::to_owned)
                .ok_or_else(|| Error("native metadata has an invalid language code".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    if languages.is_empty() {
        return Err(Error("native metadata has no supported languages".into()));
    }
    let translate = v
        .get("translate")
        .and_then(Value::as_bool)
        .ok_or_else(|| Error("native metadata lacks translate boolean".into()))?;
    Ok(Recogniser {
        languages,
        translate,
        words: false,
    })
}

fn validate(req: &TranscribeRequest, caps: &Recogniser) -> Result<String> {
    if req.samples.is_empty() {
        return Err(Error("the recording is empty".into()));
    }
    if req.rate == 0 || req.rate > i32::MAX as u32 {
        return Err(Error(
            "sample rate must be positive and fit the native ABI".into(),
        ));
    }
    if req.samples.iter().any(|v| !v.is_finite()) {
        return Err(Error("recording contains non-finite samples".into()));
    }
    if req.words {
        return Err(Error(
            "native speech bridge does not support word timestamps".into(),
        ));
    }
    if req.prompt.as_deref().is_some_and(|p| !p.trim().is_empty()) {
        return Err(Error(
            "native speech bridge does not support decoder prompts".into(),
        ));
    }
    if !req.extra.is_empty() {
        return Err(Error(format!(
            "native speech bridge has no request options: {}",
            req.extra.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    if req.translate && !caps.translate {
        return Err(Error("this checkpoint does not support translation".into()));
    }
    language::normalize(req.language.as_deref(), &caps.languages)
}

struct Callback<'a> {
    progress: &'a mut dyn FnMut(Step) -> Result<()>,
    start: Instant,
    error: Option<Error>,
}
impl Callback<'_> {
    fn emit(&mut self, phase: &'static str, at: u32, of: u32) -> c_int {
        if self.error.is_some() {
            return 1;
        }
        match catch_unwind(AssertUnwindSafe(|| {
            (self.progress)(Step {
                phase,
                at,
                of,
                seconds: self.start.elapsed().as_secs_f64(),
            })
        })) {
            Ok(Ok(())) => 0,
            Ok(Err(e)) => {
                self.error = Some(e);
                1
            }
            Err(payload) => {
                // A user panic payload can itself panic in Drop. Never unwind into C++.
                std::mem::forget(payload);
                self.error = Some(Error("progress callback panicked".into()));
                1
            }
        }
    }
}

// Called synchronously, serially, on the calling thread; the bridge never retains user.
unsafe extern "C" fn progress(
    user: *mut c_void,
    phase: *const c_char,
    done: i64,
    total: i64,
) -> c_int {
    if user.is_null() {
        return 1;
    }
    // SAFETY: transcribe passes a unique Callback valid for the entire native call.
    let state = unsafe { &mut *user.cast::<Callback<'_>>() };
    let phase = if phase.is_null() {
        "listen"
    } else {
        // SAFETY: bridge passes a NUL-terminated phase, valid during this callback.
        match unsafe { CStr::from_ptr(phase) }.to_bytes() {
            b"tokens" => "tokens",
            b"load" => "load",
            b"stt_chunk" => "listen",
            b"done" => "done",
            _ => "listen",
        }
    };
    state.emit(
        phase,
        done.clamp(0, i64::from(u32::MAX)) as u32,
        total.clamp(0, i64::from(u32::MAX)) as u32,
    )
}

fn transcript(
    v: &Value,
    duration: f64,
    req: &TranscribeRequest,
    caps: &Recogniser,
) -> Result<Transcript> {
    let text = v
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("native result lacks text".into()))?
        .to_owned();
    let raw_language = v
        .get("language")
        .and_then(Value::as_str)
        .ok_or_else(|| Error("native result lacks language".into()))?;
    // Whisper may return an empty detected language for silence. Keep that unknown
    // state; translation's output language is English regardless of source language.
    let language = if req.translate {
        "en".into()
    } else if raw_language.is_empty() {
        String::new()
    } else {
        let code = language::normalize(Some(raw_language), &caps.languages)?;
        if code == "auto" {
            return Err(Error(
                "native result language is not a detected code".into(),
            ));
        }
        code
    };
    let segments = v
        .get("segments")
        .and_then(Value::as_array)
        .ok_or_else(|| Error("native result lacks segments array".into()))?
        .iter()
        .map(|s| {
            let start = s
                .get("start")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error("invalid segment start".into()))?;
            let end = s
                .get("end")
                .and_then(Value::as_f64)
                .ok_or_else(|| Error("invalid segment end".into()))?;
            if !start.is_finite() || !end.is_finite() || start < 0.0 || end < start {
                return Err(Error("invalid native segment time range".into()));
            }
            let text = s
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| Error("invalid segment text".into()))?
                .to_owned();
            Ok(Timed {
                start,
                end,
                text,
                probability: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Transcript {
        text,
        language,
        segments,
        words: Vec::new(),
        duration,
    })
}

impl AudioEngine for Engine {
    fn arch(&self) -> &'static str {
        if self.whisper {
            WHISPER_ARCH
        } else {
            CANARY_ARCH
        }
    }
    fn defaults(&self) -> Defaults {
        Defaults {
            seconds: 0.0,
            max_seconds: 4.0 * 3600.0,
            steps: 0,
            cfg: 0.0,
            rate: 16000,
        }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn options(&self) -> &'static [EngineOption] {
        if self.whisper {
            WHISPER_OPTIONS
        } else {
            &[]
        }
    }
    fn load_seconds(&self) -> f64 {
        self.load_s
    }
    fn recogniser(&self) -> Option<Recogniser> {
        Some(self.capabilities.clone())
    }
    fn report(&self) -> Vec<String> {
        vec![self.report.clone()]
    }
    fn generate(&self, _: &AudioRequest, _: &mut dyn FnMut(Step) -> Result<()>) -> Result<Audio> {
        Err(Error(format!(
            "{} writes speech down; it makes no sound (audio.transcribe)",
            self.arch()
        )))
    }
    fn transcribe(
        &self,
        req: &TranscribeRequest,
        callback: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<Transcript> {
        let language = CString::new(validate(req, &self.capabilities)?)
            .map_err(|_| Error("language contains NUL".into()))?;
        let count = i64::try_from(req.samples.len())
            .map_err(|_| Error("recording too large for native ABI".into()))?;
        let mut state = Callback {
            progress: callback,
            start: Instant::now(),
            error: None,
        };
        if state.emit("listen", 0, 0) != 0 {
            return Err(state
                .error
                .take()
                .unwrap_or_else(|| Error("request cancelled".into())));
        }
        let resident = self
            .resident
            .lock()
            .map_err(|_| Error("native speech engine lock poisoned; reload model".into()))?;
        // Recheck cancellation after waiting for another request on this resident.
        if state.emit("listen", 0, 0) != 0 {
            return Err(state
                .error
                .take()
                .unwrap_or_else(|| Error("request cancelled".into())));
        }
        let mut result = std::ptr::null();
        // SAFETY: resident lock serializes all use; GPU, samples, language, callback,
        // and output slot live through this synchronous call. Native code must stop
        // using the callback/user before returning, including on cancellation.
        let rc = unsafe {
            (resident.api.transcribe)(
                resident.handle.as_ptr(),
                req.samples.as_ptr(),
                count,
                req.rate as c_int,
                language.as_ptr(),
                i32::from(req.translate),
                progress,
                (&mut state as *mut Callback<'_>).cast(),
                &mut result,
            )
        };
        if let Some(error) = state.error.take() {
            return Err(error);
        }
        ffi::check(rc, "native speech transcription")?;
        // SAFETY: successful call returned resident-owned JSON, copied while locked.
        let json = unsafe { copy_json(result) }?;
        drop(resident);
        let out = transcript(
            &json,
            req.samples.len() as f64 / f64::from(req.rate),
            req,
            &self.capabilities,
        )?;
        if state.emit("done", 1, 1) != 0 {
            return Err(state
                .error
                .take()
                .unwrap_or_else(|| Error("request cancelled".into())));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
