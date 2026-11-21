//! Single-window intrusive event profile. Parent serializes all GPU execution.
//! NS_PROFILE=gpu NEXTROCM_LIB=/absolute/runtime.so profile_encoder
//! MODEL_DIR REFERENCE_DIR BASELINE_ENCODER_F32 NEW_OUTPUT_DIR FLAG GPU_INDEX
use nextrocm_audio::{Error, LoadOptions, Result};
use nextrocm_audio_whisper::{Whisper, check, files_in};
use nextrocm_engine_core::Gpu;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn need(ok: bool, s: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(err(s)) }
}
fn sha(p: &Path) -> Result<String> {
    let o = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(p)
        .output()
        .map_err(err)?;
    let s = String::from_utf8(o.stdout).map_err(err)?;
    let h = s.split_whitespace().next().unwrap_or("");
    need(
        o.status.success() && h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()),
        "hash failed",
    )?;
    Ok(h.into())
}
fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    need(
        matches!(a.len(), 7 | 8),
        "usage: profile_encoder MODEL REF BASE_ENCODER NEW_OUT LINEAR_FLAG GPU [ATTENTION_FLAG]",
    )?;
    need(
        std::env::var("NS_PROFILE").as_deref() == Ok("gpu"),
        "NS_PROFILE=gpu required before GPU open",
    )?;
    need(a[5] == "0" || a[5] == "1", "flag 0 or 1")?;
    let attention = a.get(7).map(String::as_str).unwrap_or("0");
    need(matches!(attention, "0" | "1"), "attention flag 0 or 1")?;
    let runtime = PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or_else(|| err("explicit NEXTROCM_LIB required"))?,
    );
    need(
        runtime.is_absolute() && runtime.is_file(),
        "absolute runtime file required",
    )?;
    let runtime_sha = sha(&runtime)?;
    let files = files_in(Path::new(&a[1]))?;
    let model_hashes: std::collections::BTreeMap<_, _> = files
        .iter()
        .map(|(r, p)| Ok((r.clone(), sha(p)?)))
        .collect::<Result<_>>()?;
    let mel_path = Path::new(&a[2]).join("mel.npy");
    let mel_sha = sha(&mel_path)?;
    let expected_path = Path::new(&a[3]);
    let expected = std::fs::read(expected_path).map_err(err)?;
    let expected_sha = sha(expected_path)?;
    let out = Path::new(&a[4]);
    std::fs::create_dir(out).map_err(err)?;
    let mut options = LoadOptions::default();
    options
        .settings
        .insert("NS_WHISPER_LINEAR_F16_TILED".into(), a[5].clone());
    options
        .settings
        .insert("NS_WHISPER_PROFILE".into(), "1".into());
    options
        .settings
        .insert("NS_WHISPER_ATTENTION_TILED".into(), attention.into());
    nextrocm_engine_core::use_kind("audio");
    let gpu = Gpu::open(a[6].parse().map_err(err)?)?;
    let engine = Whisper::load(&files, &gpu, &options, &mut |s| eprintln!("{s}"))?;
    let (shape, values) = check::read_npy(&mel_path)?;
    let mel: Vec<f32> = values.iter().map(|&x| x as f32).collect();
    need(
        shape == [engine.model.cfg.n_mels, 3000]
            && mel.len() == engine.model.cfg.n_mels * 3000
            && mel.iter().all(|x| x.is_finite()),
        "mel finite/geometry",
    )?;
    need(
        expected.len() == engine.model.cfg.audio_ctx * engine.model.cfg.d * 4,
        "baseline encoder geometry",
    )?;
    gpu.sync()?;
    let t = Instant::now();
    engine.model.encode(&mel)?;
    gpu.sync()?;
    let wall = t.elapsed().as_secs_f64();
    let x = engine.model.encoder_output()?;
    need(x.iter().all(|x| x.is_finite()), "nonfinite encoder")?;
    let bytes: Vec<u8> = x.iter().flat_map(|x| x.to_le_bytes()).collect();
    let output = out.join("encoder.f32");
    std::fs::write(&output, &bytes).map_err(err)?;
    need(
        bytes == expected,
        "profiled encoder changed native baseline output",
    )?;
    drop(engine);
    gpu.sync()?;
    need(
        sha(&runtime)? == runtime_sha
            && sha(&mel_path)? == mel_sha
            && sha(expected_path)? == expected_sha,
        "runtime/fixture changed",
    )?;
    for (role, p) in &files {
        need(sha(p)? == model_hashes[role], "weights changed")?;
    }
    let report = json!({"pass":true,"flag":a[5],"attention_flag":attention,"gpu":gpu.name,"runtime_sha256":runtime_sha,
        "executable_sha256":sha(&std::env::current_exe().map_err(err)?)?,"model_hashes":model_hashes,
        "mel_sha256":mel_sha,"baseline_encoder_sha256":expected_sha,"encoder_sha256":sha(&output)?,
        "profiled_encode_wall_seconds":wall,"scope":"Single diagnostic window with bounded event flushes; not unprofiled throughput. WHISPER_PROFILE stderr records attribution."});
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(err)?,
    )
    .map_err(err)?;
    println!("{}", report);
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1)
    }
}
