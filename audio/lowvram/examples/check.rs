use nextrocm_audio::{Error, Gpu, LoadOptions, ModelFiles, Result, TranscribeRequest};
use std::{path::PathBuf, time::Instant};
fn stats(g: &Gpu) -> Result<(u64, u64)> {
    let api = nextrocm_engine_core::api()?;
    let p = api.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err(Error("missing allocation stats".into()));
    }
    type F = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: symbol has this C ABI, library retained by engine-core, outputs and GPU live.
    let f: F = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    if unsafe { f(g.raw(), &mut current, &mut peak) } != 0 {
        return Err(Error(api.error()));
    }
    Ok((current, peak))
}
fn run() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err(Error("check canary|whisper MODEL WAV".into()));
    }
    nextrocm_engine_core::use_kind("audio");
    let g = Gpu::open(0)?;
    let k = match a[1].as_str() {
        "canary" => nextrocm_audio_lowvram::canary_kind(),
        "whisper" => nextrocm_audio_lowvram::whisper_kind(),
        _ => return Err(Error("unknown kind".into())),
    };
    let files: ModelFiles = [("model".into(), PathBuf::from(&a[2]))].into();
    let engine = (k.load)(&files, &[g.clone()], &LoadOptions::default(), &mut |s| {
        eprintln!("{s}")
    })?;
    let bytes = std::fs::read(&a[3]).map_err(|e| Error(e.to_string()))?;
    let (samples, rate) = nextrocm_audio::decode_audio(&bytes).map_err(Error)?;
    let req = TranscribeRequest {
        samples,
        rate,
        language: Some("english".into()),
        words: false,
        ..Default::default()
    };
    let t = Instant::now();
    let first = engine.transcribe(&req, &mut |_| Ok(()))?;
    if first.text.trim().is_empty() {
        return Err(Error("empty transcript".into()));
    }
    println!(
        "{}",
        serde_json::json!({"text":first.text,"language":first.language,"seconds":t.elapsed().as_secs_f64(),"memory":stats(&g)?})
    );
    let mut cancelled = false;
    let err = engine.transcribe(&req, &mut |s| {
        if s.phase == "tokens" {
            cancelled = true;
            Err(Error("fixture cancellation".into()))
        } else {
            Ok(())
        }
    });
    if !cancelled || err.is_ok() {
        return Err(Error("cancellation did not stop native decoding".into()));
    }
    let again = engine.transcribe(&req, &mut |_| Ok(()))?;
    if first.text != again.text {
        return Err(Error(format!("reused model differs: {:?}", again.text)));
    }
    drop(engine);
    let (current, peak) = stats(&g)?;
    if current != 0 || peak >= 8_000_000_000 {
        return Err(Error(format!("memory gate: current={current} peak={peak}")));
    }
    println!("PASS: cancellation, reusable resident and cleanup; tracked peak={peak}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
