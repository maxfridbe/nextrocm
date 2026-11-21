//! Manual GPU validation, never run by cargo test.
//! variants_lifecycle DESIGN_DIR BASE_DIR REFERENCE_WAV OUTPUT_DIR
//! Reference transcript: "Hello from next rocm." (the staged CustomVoice fixture).
use nextrocm_audio::{Audio, AudioEngine, AudioRequest, Error, LoadOptions, Reference, Result};
use nextrocm_audio_qwen3tts::{files_in, kind};
use nextrocm_engine_core::{Gpu, api};
use std::{path::Path, sync::Arc};

const CAP: u64 = 8_000_000_000;
const SECONDS: f32 = 0.64; // Eight codec frames, enough to cancel after the first.
const CANCEL: &str = "variants_lifecycle requested cancellation";

fn ensure(ok: bool, message: impl Into<String>) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}

fn stats(gpu: &Gpu, expected: Option<u64>) -> Result<(u64, u64)> {
    gpu.sync()?;
    let api = api()?;
    let symbol = api.symbol("nr_allocation_stats");
    ensure(!symbol.is_null(), "allocation telemetry is required")?;
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: retained runtime exports this C ABI; gpu and output pointers remain live.
    let query: Stats = unsafe { std::mem::transmute(symbol) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: live GPU handle and writable u64 outputs, as required by the runtime ABI.
    ensure(
        unsafe { query(gpu.raw(), &mut current, &mut peak) } == 0,
        api.error(),
    )?;
    ensure(peak < CAP, format!("tracked device peak {peak} >= {CAP}"))?;
    if let Some(expected) = expected {
        ensure(
            current == expected,
            format!("allocation cleanup: {current}, expected {expected}"),
        )?;
    }
    Ok((current, peak))
}

fn validate_audio(audio: &Audio) -> Result<()> {
    ensure(
        audio.rate == 24000 && audio.channels == 1,
        "expected mono 24 kHz",
    )?;
    ensure(
        !audio.samples.is_empty() && audio.samples.len() <= 8 * 1920,
        "empty or oversized waveform",
    )?;
    ensure(
        audio.samples.iter().all(|v| v.is_finite()),
        "nonfinite waveform",
    )?;
    ensure(
        audio.samples.iter().any(|v| v.abs() > 1e-6),
        "silent waveform",
    )
}

fn generate(engine: &dyn AudioEngine, req: &AudioRequest) -> Result<Audio> {
    let (mut tokens, mut decoded) = (0, false);
    let audio = engine.generate(req, &mut |step| {
        ensure(
            step.seconds.is_finite() && step.seconds >= 0.0 && step.at <= step.of,
            "invalid progress",
        )?;
        if step.phase == "tokens" {
            ensure(
                step.at == tokens + 1 && step.of == 8,
                "invalid frame progress",
            )?;
            tokens = step.at;
        }
        if step.phase == "decode" && step.at == 1 {
            decoded = true;
        }
        Ok(())
    })?;
    ensure(
        tokens > 0 && decoded,
        "generation did not complete frame/decode progress",
    )?;
    validate_audio(&audio)?;
    Ok(audio)
}

fn cancel(engine: &dyn AudioEngine, req: &AudioRequest, partial: bool) -> Result<()> {
    let (mut calls, mut cancelled) = (0, false);
    let result = engine.generate(req, &mut |step| {
        ensure(!cancelled, "progress continued after cancellation")?;
        calls += 1;
        if !partial || (step.phase == "tokens" && step.at == 1) {
            if !partial {
                ensure(
                    step.phase == "prompt" && step.at == 0,
                    "unexpected first callback",
                )?;
            }
            cancelled = true;
            return Err(Error(CANCEL.into()));
        }
        Ok(())
    });
    ensure(
        cancelled && (partial || calls == 1),
        "cancellation point was not reached",
    )?;
    match result {
        Err(e) if e.0 == CANCEL => Ok(()),
        Err(e) => Err(Error(format!("cancellation error changed: {e}"))),
        Ok(_) => Err(Error("cancelled request returned audio".into())),
    }
}

fn invalid(
    engine: &dyn AudioEngine,
    req: &AudioRequest,
    gpu: &Gpu,
    resident: u64,
    label: &str,
) -> Result<()> {
    let before = stats(gpu, Some(resident))?;
    let mut calls = 0;
    let result = engine.generate(req, &mut |_| {
        calls += 1;
        Ok(())
    });
    ensure(
        result.is_err() && calls == 0,
        format!("{label}: expected rejection before progress"),
    )?;
    let after = stats(gpu, Some(resident))?;
    ensure(
        before.1 == after.1,
        format!("{label}: rejection increased device high-water mark"),
    )?;
    println!("PASS invalid {label}");
    Ok(())
}

fn run_variant(
    engine: &dyn AudioEngine,
    req: &AudioRequest,
    name: &str,
    gpu: &Gpu,
    out: &Path,
) -> Result<()> {
    let resident = stats(gpu, None)?.0;
    for seconds in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut bad = req.clone();
        bad.seconds = Some(seconds);
        invalid(engine, &bad, gpu, resident, "duration")?;
    }
    let mut bad = req.clone();
    if req.reference.is_none() {
        bad.instructions = Some("  ".into());
        invalid(engine, &bad, gpu, resident, "missing design instruction")?;
    } else {
        bad.reference = None;
        invalid(engine, &bad, gpu, resident, "missing reference")?;
        bad = req.clone();
        bad.instructions = Some("invalid for Base".into());
        invalid(engine, &bad, gpu, resident, "clone instructions")?;
        for (label, samples, rate) in [
            ("empty", vec![], 24000),
            ("zero rate", vec![0.0; 1920], 0),
            ("NaN", vec![f32::NAN; 1920], 24000),
            ("infinity", vec![f32::INFINITY; 1920], 24000),
            ("short", vec![0.0; 1919], 24000),
            ("resample budget", vec![0.0; 1920], 1),
        ] {
            bad = req.clone();
            bad.reference = Some(Reference {
                samples,
                rate,
                text: Some("test".into()),
            });
            invalid(engine, &bad, gpu, resident, label)?;
        }
    }
    let baseline = generate(engine, req)?;
    stats(gpu, Some(resident))?;
    for partial in [false, true] {
        cancel(engine, req, partial)?;
        stats(gpu, Some(resident))?;
        let reused = generate(engine, req)?;
        ensure(
            baseline.samples.len() == reused.samples.len()
                && baseline
                    .samples
                    .iter()
                    .zip(&reused.samples)
                    .all(|(a, b)| (a - b).abs() <= 1e-6),
            "greedy reuse waveform differs",
        )?;
        stats(gpu, Some(resident))?;
        println!("PASS {name} cancel_partial={partial} reuse");
    }
    let path = out.join(format!("{name}.wav"));
    baseline.write_wav(&path, &[]).map_err(Error)?;
    let bytes = std::fs::read(&path).map_err(|e| Error(e.to_string()))?;
    let decoded = nextrocm_audio::read_wav(&bytes).map_err(Error)?;
    validate_audio(&decoded)?;
    ensure(
        decoded.samples.len() == baseline.samples.len(),
        "WAV roundtrip length changed",
    )?;
    let (current, peak) = stats(gpu, Some(resident))?;
    println!(
        "PASS {name} samples={} current={current} peak={peak}",
        baseline.samples.len()
    );
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure(
        args.len() == 4,
        "usage: variants_lifecycle DESIGN_DIR BASE_DIR REFERENCE_WAV OUTPUT_DIR",
    )?;
    let out = Path::new(&args[3]);
    std::fs::create_dir_all(out).map_err(|e| Error(e.to_string()))?;
    let reference_path = Path::new(&args[2]);
    let size = std::fs::metadata(reference_path)
        .map_err(|e| Error(e.to_string()))?
        .len();
    ensure(
        size <= 4 * 1024 * 1024,
        "reference WAV exceeds runner's 4 MiB bound",
    )?;
    let bytes = std::fs::read(reference_path).map_err(|e| Error(e.to_string()))?;
    let audio = nextrocm_audio::read_wav(&bytes).map_err(Error)?;
    ensure(
        audio.channels == 1 && audio.rate == 24000 && audio.samples.len() <= 5 * 24000,
        "use the short mono 24 kHz CustomVoice fixture",
    )?;
    ensure(
        !audio.samples.is_empty() && audio.samples.iter().all(|x| x.is_finite()),
        "invalid reference samples",
    )?;
    let mut req = AudioRequest {
        prompt: "Hello. This is a short voice design test.".into(),
        instructions: Some("A warm, low male voice, speaking calmly and clearly.".into()),
        language: Some("English".into()),
        seconds: Some(SECONDS),
        seed: 0,
        ..Default::default()
    };
    req.extra.insert("greedy".into(), "1".into());
    let mut options = LoadOptions::default();
    options.settings.insert("NS_Q3T_INT8".into(), "0".into());
    nextrocm_engine_core::use_kind("audio");
    let gpu = Gpu::open(0)?;
    stats(&gpu, Some(0))?;
    for (index, design) in [(0, true), (1, false)] {
        let files = files_in(Path::new(&args[index]))?;
        let engine = (kind().load)(&files, &[Arc::clone(&gpu)], &options, &mut |s| {
            eprintln!("{s}")
        })?;
        let speech = engine
            .speech()
            .ok_or_else(|| Error("missing speech capabilities".into()))?;
        ensure(
            speech.design == design && speech.clone != design,
            "wrong checkpoint variant",
        )?;
        if design {
            run_variant(engine.as_ref(), &req, "design", &gpu, out)?;
        } else {
            req.prompt = "Hello. This is a short voice cloning test.".into();
            req.instructions = None;
            req.reference = Some(Reference {
                samples: audio.samples.clone(),
                rate: audio.rate,
                text: None,
            });
            run_variant(engine.as_ref(), &req, "base_xvector", &gpu, out)?;
            if let Some(reference) = &mut req.reference {
                reference.text = Some("Hello from next rocm.".into());
            }
            run_variant(engine.as_ref(), &req, "base_in_context", &gpu, out)?;
        }
        drop(engine);
        let (_, peak) = stats(&gpu, Some(0))?;
        println!("PASS model drop current=0 peak={peak}");
    }
    println!(
        "PASS variants lifecycle (tracked allocator cap; not a numerical reference comparison)"
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAIL variants lifecycle: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waveform_gate_rejects_invalid_and_unbounded_output() {
        let make = |samples| Audio {
            rate: 24000,
            channels: 1,
            samples,
        };
        for samples in [
            vec![],
            vec![0.0; 1920],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![0.1; 8 * 1920 + 1],
        ] {
            assert!(validate_audio(&make(samples)).is_err());
        }
        assert!(validate_audio(&make(vec![0.1; 8 * 1920])).is_ok());
        assert!(
            validate_audio(&Audio {
                rate: 16000,
                channels: 1,
                samples: vec![0.1]
            })
            .is_err()
        );
    }

    #[test]
    fn short_waveform_survives_original_wav_contract() {
        let audio = Audio {
            rate: 24000,
            channels: 1,
            samples: (0..1920).map(|n| (n as f32 * 0.1).sin() * 0.2).collect(),
        };
        let decoded = nextrocm_audio::read_wav(&audio.wav(&[])).expect("valid WAV roundtrip");
        validate_audio(&decoded).expect("finite nonzero output");
        assert_eq!(decoded.samples.len(), audio.samples.len());
    }
}
