//! VAE-only full-bit chunking control; no language model, generated speech or quality claim.
//! NEXTROCM_LIB=... vae_encode_chunks MODEL_DIR WAV NEW_OUT CHUNK_FRAMES full|PREFIX_FRAMES
//! Parent supplies the exclusive GPU/host-memory watchdog. GPU 0, at most 120 s of input.
use nextrocm_audio::{Error, LoadOptions, Result};
use nextrocm_audio_voxcpm2::{budget, ops::Ops, pth, resample, vae::Vae};
use nextrocm_engine_core::{Gpu, silo_for, use_kind};
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn sha(path: &Path) -> Result<String> {
    let p = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(err)?;
    let text = String::from_utf8(p.stdout).map_err(err)?;
    let h = text.split_whitespace().next().unwrap_or("");
    if !p.status.success() || h.len() != 64 || !h.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(err(format!("hash failed: {}", path.display())));
    }
    Ok(h.into())
}
fn dump(path: &Path, values: &[f32]) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(err)?,
    );
    for value in values {
        f.write_all(&value.to_le_bytes()).map_err(err)?;
    }
    f.flush().map_err(err)
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(err("MODEL_DIR WAV NEW_OUT CHUNK_FRAMES full|PREFIX_FRAMES"));
    }
    let model = PathBuf::from(&args[0]);
    let wav_path = PathBuf::from(&args[1]);
    let out = PathBuf::from(&args[2]);
    let chunk: usize = args[3].parse().map_err(err)?;
    if chunk == 0 || chunk > 250 {
        return Err(err("chunk must be 1..250"));
    }
    let prefix = if args[4] == "full" {
        None
    } else {
        Some(args[4].parse::<usize>().map_err(err)?)
    };
    if prefix == Some(0) {
        return Err(err("prefix must be positive"));
    }
    let runtime = PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or_else(|| err("explicit NEXTROCM_LIB required"))?,
    );
    let config = model.join("config.json");
    let weights = model.join("audiovae.pth");
    let mut pins = BTreeMap::new();
    for p in [
        std::env::current_exe().map_err(err)?,
        runtime,
        config.clone(),
        weights.clone(),
        wav_path.clone(),
    ] {
        pins.insert(p.clone(), sha(&p)?);
    }
    if std::fs::metadata(&wav_path).map_err(err)?.len() > 64 * 1024 * 1024 {
        return Err(err("WAV exceeds 64 MiB"));
    }
    let (samples, rate) =
        nextrocm_audio::decode_audio(&std::fs::read(&wav_path).map_err(err)?).map_err(err)?;
    if rate == 0 || samples.len() > rate as usize * 120 {
        return Err(err("input exceeds 120 seconds"));
    }
    let mut wav = resample(&samples, rate, 16000);
    if wav.is_empty() || wav.iter().any(|v| !v.is_finite()) {
        return Err(err("invalid audio"));
    }
    std::fs::create_dir(&out).map_err(err)?;
    use_kind("audio");
    let gpu = Gpu::open(0).map_err(err)?;
    if let Some(s) = silo_for("voxcpm2", 0).map_err(err)? {
        pins.insert(s.path.clone(), sha(&s.path)?);
        if let Some((_, path)) = s.variant() {
            pins.insert(path.to_path_buf(), sha(path)?);
        }
    }
    let mut report = json!({"status":"INCOMPLETE","pins_before":pins,"chunk_frames":chunk,"prefix_frames":prefix,
        "scope":"same-runtime native VAE full-bit regression, not an independent numeric oracle or speech-quality proof"});
    let result = (|| -> Result<()> {
        if budget::stats(&gpu)?.0 != 0 {
            return Err(err("nonzero initial tracked allocations"));
        }
        let ops = Ops::new(&gpu, &LoadOptions::default())?;
        let conf = serde_json::from_slice(&std::fs::read(&config).map_err(err)?).map_err(err)?;
        let vae = Vae::load(&ops, &pth::load(&weights)?, &conf)?;
        if vae.hop_in == 0 || vae.latent == 0 {
            return Err(err("invalid VAE dimensions"));
        }
        // Match production reference padding to a full four-latent patch, on the right.
        let patch = budget::product(&[4, vae.hop_in])?;
        let original_samples = wav.len();
        let padded = budget::sum(&[wav.len(), (patch - wav.len() % patch) % patch])?;
        wav.resize(padded, 0.0);
        if let Some(frames) = prefix {
            let n = budget::product(&[frames, vae.hop_in])?;
            if n > wav.len() {
                return Err(err("requested prefix exceeds full input"));
            }
            wav.truncate(n);
        }
        report["original_16khz_samples"] = json!(original_samples);
        report["tested_samples"] = json!(wav.len());
        report["hop"] = json!(vae.hop_in);
        report["context_samples"] = json!(vae.encoder_context_samples()?);
        let start = Instant::now();
        let baseline = vae.encode_unchunked(&ops, &wav)?;
        report["control_seconds"] = json!(start.elapsed().as_secs_f64());
        let start = Instant::now();
        let candidate = vae.encode_chunked(&ops, &wav, chunk)?;
        report["chunked_seconds"] = json!(start.elapsed().as_secs_f64());
        let expected = budget::product(&[vae.latent, wav.len() / vae.hop_in])?;
        if baseline.len() != expected
            || candidate.len() != expected
            || baseline.iter().chain(&candidate).any(|x| !x.is_finite())
        {
            return Err(err("wrong latent extent or nonfinite output"));
        }
        dump(&out.join("unchunked.f32"), &baseline)?;
        dump(&out.join("chunked.f32"), &candidate)?;
        let mismatches = baseline
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        report["latent_values"] = json!(expected);
        report["bit_mismatches"] = json!(mismatches);
        report["control_sha256"] = json!(sha(&out.join("unchunked.f32"))?);
        report["chunked_sha256"] = json!(sha(&out.join("chunked.f32"))?);
        if mismatches != 0 {
            return Err(err(format!("{mismatches} unequal latent values")));
        }
        Ok(())
    })();
    let cleanup = gpu.sync().map_err(err).and_then(|_| budget::stats(&gpu));
    let final_result = result.and_then(|_| {
        let (current, peak) = cleanup.as_ref().map_err(err)?;
        if *current != 0 || *peak > 4_026_531_840 {
            return Err(err("tracked allocation/drop gate failed"));
        }
        for (p, h) in &pins {
            if sha(p)? != *h {
                return Err(err(format!("input changed: {}", p.display())));
            }
        }
        Ok(())
    });
    match &cleanup {
        Ok((current, peak)) => {
            report["tracked_current_bytes"] = json!(current);
            report["tracked_peak_bytes"] = json!(peak);
        }
        Err(e) => report["cleanup_error"] = json!(e.to_string()),
    }
    report["status"] = json!(if final_result.is_ok() { "PASS" } else { "FAIL" });
    if let Err(e) = &final_result {
        report["error"] = json!(e.to_string());
    }
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(err)?,
    )
    .map_err(err)?;
    final_result
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
