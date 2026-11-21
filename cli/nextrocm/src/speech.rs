use nextrocm_audio::{AudioEngine, LoadOptions, TranscribeRequest};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_ops::kernels::{none, Dt, Nsd};
use serde_json::json;
use std::path::Path;

fn tensor(g: &std::sync::Arc<Gpu>, x: &[f32]) -> Result<DevBuf, String> {
    DevBuf::from_f32(g, x).map_err(|e| e.to_string())
}
pub fn ops_test() -> Result<(), String> {
    let g = Gpu::open(0).map_err(|e| e.to_string())?;
    let n = Nsd::new(&g).map_err(|e| e.to_string())?;
    let test = || -> nextrocm_engine_core::Result<()> {
        // Odd matrix sizes exercise both tile tails and reduction tails.
        let (m, k, c) = (19, 23, 17);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.) / 10.).collect();
        let b: Vec<f32> = (0..c * k).map(|i| ((i % 7) as f32 - 3.) / 7.).collect();
        let x = DevBuf::from_f32(&g, &a)?;
        let w = DevBuf::from_f32(&g, &b)?;
        let out = DevBuf::f32(&g, m * c)?;
        n.linear(
            x.ptr(),
            Dt::F32,
            m,
            k,
            w.ptr(),
            c,
            none(),
            out.ptr(),
            Dt::F32,
        )?;
        let got = out.to_f32()?;
        for r in 0..m {
            for j in 0..c {
                let want = (0..k).map(|z| a[r * k + z] * b[j * k + z]).sum::<f32>();
                if (got[r * c + j] - want).abs() > 1e-5 {
                    return Err(nextrocm_engine_core::Error(
                        "linear reference mismatch".into(),
                    ));
                }
            }
        }
        n.layer_norm(
            x.ptr(),
            Dt::F32,
            m,
            k,
            none(),
            none(),
            1e-5,
            x.ptr(),
            Dt::F32,
        )?;
        let got = x.to_f32()?;
        for r in 0..m {
            let mean = a[r * k..(r + 1) * k].iter().sum::<f32>() / k as f32;
            let var = a[r * k..(r + 1) * k]
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f32>()
                / k as f32;
            for j in 0..k {
                let want = (a[r * k + j] - mean) / (var + 1e-5).sqrt();
                if (got[r * k + j] - want).abs() > 1e-5 {
                    return Err(nextrocm_engine_core::Error(
                        "normalization reference mismatch".into(),
                    ));
                }
            }
        }
        // VibeVoice's two conditioning rows select different modulation/gate vectors.
        let width = 257;
        let values: Vec<f32> = (0..2 * width).map(|i| (i % 17) as f32 / 8. - 1.).collect();
        let scale: Vec<f32> = (0..2 * width)
            .map(|i| if i < width { 0.25 } else { -0.5 })
            .collect();
        let shift: Vec<f32> = (0..2 * width)
            .map(|i| if i < width { -0.2 } else { 0.3 })
            .collect();
        let input = DevBuf::from_f32(&g, &values)?;
        let sc = DevBuf::from_f32(&g, &scale)?;
        let sh = DevBuf::from_f32(&g, &shift)?;
        let rows = DevBuf::new(&g, 8)?;
        rows.write(0, &[1i32.to_le_bytes(), 0i32.to_le_bytes()].concat())?;
        let result = DevBuf::f32(&g, 2 * width)?;
        n.rms_norm_mod(
            input.ptr(),
            Dt::F32,
            2,
            width,
            none(),
            1e-6,
            rows.ptr(),
            sc.ptr(),
            sh.ptr(),
            result.ptr(),
            Dt::F32,
        )?;
        let got = result.to_f32()?;
        for r in 0..2 {
            let inv = (values[r * width..(r + 1) * width]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                / width as f32
                + 1e-6)
                .sqrt()
                .recip();
            for j in 0..width {
                let z = (1 - r) * width + j;
                let want = values[r * width + j] * inv * (1. + scale[z]) + shift[z];
                if (got[r * width + j] - want).abs() > 2e-5 {
                    return Err(nextrocm_engine_core::Error(
                        "modulated RMS reference mismatch".into(),
                    ));
                }
            }
        }
        n.gate_add(
            result.ptr(),
            Dt::F32,
            2,
            width,
            input.ptr(),
            Dt::F32,
            rows.ptr(),
            sc.ptr(),
        )?;
        let added = result.to_f32()?;
        for i in 0..2 * width {
            let want = got[i] + values[i] * scale[(1 - i / width) * width + i % width];
            if (added[i] - want).abs() > 2e-5 {
                return Err(nextrocm_engine_core::Error(
                    "row-gated add reference mismatch".into(),
                ));
            }
        }
        Ok(())
    };
    test().map_err(|e| e.to_string())?;
    // Uniform attention: zero queries/keys must average values, including causal row offsets.
    let q = tensor(&g, &[0.; 24])?;
    let k = tensor(&g, &[0.; 24])?;
    let v = tensor(&g, &(0..24).map(|i| i as f32).collect::<Vec<_>>())?;
    let out = tensor(&g, &[0.; 24])?;
    n.attention(
        q.ptr(),
        k.ptr(),
        v.ptr(),
        Dt::F32,
        3,
        2,
        4,
        8,
        out.ptr(),
        Dt::F32,
    )
    .map_err(|e| e.to_string())?;
    let got = out.to_f32().map_err(|e| e.to_string())?;
    for (i, x) in got.iter().enumerate() {
        if (*x - (8 + i % 8) as f32).abs() > 1e-5 {
            return Err("attention reference mismatch".into());
        }
    }
    n.attention_causal_rows(
        q.ptr(),
        k.ptr(),
        v.ptr(),
        Dt::F32,
        1,
        2,
        2,
        2,
        4,
        8,
        8,
        out.ptr(),
    )
    .map_err(|e| e.to_string())?;
    let got = out.to_f32().map_err(|e| e.to_string())?;
    for (i, x) in got[..16].iter().enumerate() {
        let want = (if i < 8 { 4 } else { 8 }) + (i % 8);
        if (*x - want as f32).abs() > 1e-5 {
            return Err("causal offset reference mismatch".into());
        }
    }
    println!("PASS: speech matrix, norm, modulated RMS, gated add, full and offset causal attention against CPU arithmetic");
    Ok(())
}
pub fn run(command: &str, model: &Path, input: &Path) -> Result<(), String> {
    let gpu = Gpu::open(0).map_err(|e| e.to_string())?;
    let (total, before) = gpu.memory().map_err(|e| e.to_string())?;
    if before.unwrap_or(0) < 3_000_000_000 {
        return Err("Whisper validation needs at least 3 GB free device memory".into());
    }
    let files = nextrocm_audio_whisper::files_in(model).map_err(|e| e.to_string())?;
    let mut log = |s| eprintln!("{s}");
    let engine =
        nextrocm_audio_whisper::Whisper::load(&files, &gpu, &LoadOptions::default(), &mut log)
            .map_err(|e| e.to_string())?;
    match command {
        "check" => {
            let error = nextrocm_audio_whisper::check::run(
                &engine,
                input,
                &["mel", "encoder", "logits"],
                &mut log,
            )
            .map_err(|e| e.to_string())?;
            if !error.is_finite() || error > 0.02 {
                return Err(format!(
                    "reference comparison failed: worst relative L2 {error}"
                ));
            }
        }
        "transcribe" => {
            let audio =
                nextrocm_audio::read_audio(&std::fs::read(input).map_err(|e| e.to_string())?)?;
            let req = TranscribeRequest {
                samples: audio.mono(),
                rate: audio.rate,
                language: Some("en".into()),
                words: true,
                extra: [("fallback".into(), "0".into())].into_iter().collect(),
                ..Default::default()
            };
            let start = std::time::Instant::now();
            let t = engine
                .transcribe(&req, &mut |step| {
                    eprintln!("{}", step.phase);
                    Ok(())
                })
                .map_err(|e| e.to_string())?;
            let segments: Vec<_> = t
                .segments
                .iter()
                .map(|s| json!({"start":s.start,"end":s.end,"text":s.text}))
                .collect();
            println!(
                "{}",
                json!({"text":t.text,"language":t.language,"duration":t.duration,"elapsed_s":start.elapsed().as_secs_f64(),"segments":segments})
            );
        }
        _ => return Err("expected transcribe or check".into()),
    }
    gpu.sync().map_err(|e| e.to_string())?;
    let (_, after) = gpu.memory().map_err(|e| e.to_string())?;
    eprintln!("device total {total}; free before {before:?}; free after {after:?}");
    let (current, peak) = allocation_stats(&gpu)?;
    eprintln!(
        "tensor bytes current={current}, peak={peak}; other device usage includes HIP and desktop"
    );
    if peak >= 7_500_000_000 {
        return Err("model exceeded tensor memory budget".into());
    }
    Ok(())
}

pub fn allocation_stats(gpu: &std::sync::Arc<Gpu>) -> Result<(u64, u64), String> {
    let api = nextrocm_engine_core::api().map_err(|e| e.to_string())?;
    let p = api.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err("allocation telemetry not available".into());
    }
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: exact signature declared by the shared HIP library; device and output locals are live.
    let stats: Stats = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: owning device and valid writable u64 locals.
    if unsafe { stats(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err(api.error());
    }
    Ok((current, peak))
}

pub fn tts(command: &str, dir: &Path, input: &str, out: Option<&Path>) -> Result<(), String> {
    let gpu = Gpu::open(0).map_err(|e| e.to_string())?;
    let files = nextrocm_audio_qwen3tts::files_in(dir).map_err(|e| e.to_string())?;
    let model =
        nextrocm_audio_qwen3tts::Qwen3Tts::load(&files, &gpu, &LoadOptions::default(), &mut |s| {
            eprintln!("{s}")
        })
        .map_err(|e| e.to_string())?;
    if command == "check" {
        let error = nextrocm_audio_qwen3tts::check::run(
            &model,
            Path::new(input),
            &["prompt", "codec"],
            &mut |s| eprintln!("{s}"),
        )
        .map_err(|e| e.to_string())?;
        if !error.is_finite() || error > 0.03 {
            return Err(format!(
                "TTS reference comparison failed: relative L2 {error}"
            ));
        }
    } else if command == "speak" {
        let request = nextrocm_audio::AudioRequest {
            prompt: input.into(),
            voice: Some("vivian".into()),
            language: Some("english".into()),
            seconds: Some(5.0),
            extra: [("greedy".into(), "1".into())].into_iter().collect(),
            ..Default::default()
        };
        let started = std::time::Instant::now();
        let audio = model
            .generate(&request, &mut |s| {
                if s.at % 10 == 0 {
                    eprintln!("{} {}/{}", s.phase, s.at, s.of);
                }
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        if audio.samples.is_empty() || audio.samples.iter().any(|v| !v.is_finite()) {
            return Err("invalid generated waveform".into());
        }
        audio.write_wav(out.ok_or("output WAV required")?, &[])?;
        println!(
            "{}",
            json!({"samples":audio.samples.len(),"rate":audio.rate,"elapsed_s":started.elapsed().as_secs_f64()})
        );
    } else {
        return Err("expected tts check|speak".into());
    }
    let (current, peak) = allocation_stats(&gpu)?;
    eprintln!(
        "TTS tensor bytes current={current}, peak={peak}; device memory {:?}",
        gpu.memory().map_err(|e| e.to_string())?
    );
    Ok(())
}
