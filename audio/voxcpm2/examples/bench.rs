//! Native diagnostic through the original AudioEngine API. Parent serializes GPU runs.
//! Stage metrics are wall time, not pure GPU time. No tensor taps or per-op profiling.
use nextrocm_audio::{AudioEngine, AudioRequest, Error, LoadOptions, Reference, Result};
use nextrocm_audio_voxcpm2::{VoxCpm2, budget, files_in};
use nextrocm_engine_core::Gpu;
use serde_json::json;
use std::{collections::BTreeMap, io::Write, path::Path, time::Instant};

const USAGE: &str = "bench --model DIR --output NEW_JSON [--wav-dir NEWDIR] [--gpu 0] [--int8 1] [--text TEXT] [--steps 10] [--cfg 2] [--seconds 12.8] [--seed 0] [--repeats 2] [--ref-audio WAV --ref-text TEXT] [--mode tts|clone|continue]";
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn sha(path: &Path) -> Result<String> {
    let output = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(err)?;
    let text = String::from_utf8(output.stdout).map_err(err)?;
    let digest = text.split_whitespace().next().unwrap_or("");
    if !output.status.success()
        || digest.len() != 64
        || !digest.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(err("sha256sum failed"));
    }
    Ok(digest.into())
}
fn run() -> Result<()> {
    let mut flags = BTreeMap::new();
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        if key == "--help" {
            println!("{USAGE}");
            return Ok(());
        }
        if !matches!(
            key.as_str(),
            "--model"
                | "--output"
                | "--gpu"
                | "--int8"
                | "--text"
                | "--steps"
                | "--cfg"
                | "--seconds"
                | "--seed"
                | "--repeats"
                | "--ref-audio"
                | "--ref-text"
                | "--mode"
                | "--wav-dir"
        ) {
            return Err(err(USAGE));
        }
        let value = args.next().ok_or_else(|| err(USAGE))?;
        if flags.insert(key, value).is_some() {
            return Err(err("duplicate flag"));
        }
    }
    let value = |key, default| flags.get(key).map(String::as_str).unwrap_or(default);
    let model = flags.get("--model").ok_or_else(|| err(USAGE))?;
    let output = flags.get("--output").ok_or_else(|| err(USAGE))?;
    let gpu_index = value("--gpu", "0").parse::<usize>().map_err(err)?;
    let int8 = value("--int8", "1");
    if !matches!(int8, "0" | "1") {
        return Err(err("int8 must be 0 or 1"));
    }
    let repeats = value("--repeats", "2").parse::<usize>().map_err(err)?;
    if !(1..=2).contains(&repeats) {
        return Err(err("repeats must be 1 or 2"));
    }
    let mode = value("--mode", "tts");
    if !matches!(mode, "tts" | "clone" | "continue") {
        return Err(err("unknown mode"));
    }
    let mut request = AudioRequest {
        prompt: value("--text", "Hello from next rocm.").into(),
        steps: Some(value("--steps", "10").parse().map_err(err)?),
        cfg: Some(value("--cfg", "2").parse().map_err(err)?),
        seconds: Some(value("--seconds", "12.8").parse().map_err(err)?),
        seed: value("--seed", "0").parse().map_err(err)?,
        ..Default::default()
    };
    if request.prompt.trim().is_empty()
        || request.steps.is_none_or(|v| !(1..=200).contains(&v))
        || request.cfg.is_none_or(|v| !v.is_finite())
        || request
            .seconds
            .is_none_or(|v| !v.is_finite() || v <= 0. || v > 320.)
    {
        return Err(err("invalid bounded request"));
    }
    let reference_hash = if mode != "tts" {
        let path = Path::new(
            flags
                .get("--ref-audio")
                .ok_or_else(|| err("voice modes require --ref-audio"))?,
        );
        let wav = nextrocm_audio::read_wav(&std::fs::read(path).map_err(err)?).map_err(err)?;
        let text = flags.get("--ref-text").cloned();
        if mode == "continue" && text.as_ref().is_none_or(|v| v.trim().is_empty()) {
            return Err(err("continuation requires transcript"));
        }
        request.reference = Some(Reference {
            samples: wav.mono(),
            rate: wav.rate,
            text,
        });
        Some(sha(path)?)
    } else {
        if flags.contains_key("--ref-audio") || flags.contains_key("--ref-text") {
            return Err(err("reference requires clone/continue mode"));
        }
        None
    };
    if mode == "continue" {
        request.extra.insert("continue".into(), "1".into());
    }
    let wav_dir = flags.get("--wav-dir").map(Path::new);
    if let Some(dir) = wav_dir {
        std::fs::create_dir(dir).map_err(err)?;
    }
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(err)?;
    let files = files_in(Path::new(model))?;
    let model_hashes = files
        .iter()
        .map(|(role, path)| Ok((role.clone(), sha(path)?)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut report = json!({"schema":"voxcpm2-bench-v1","model":model,"model_hashes":model_hashes,
        "executable_sha256":sha(&std::env::current_exe().map_err(err)?)?,"int8":int8,"mode":mode,
        "f32_rows_control":std::env::var("NEXTROCM_VCP_F32_ROWS").unwrap_or_else(|_|"0".into()),
        "rope_bf16_cache_control":std::env::var("NEXTROCM_VCP_ROPE_BF16_CACHE").unwrap_or_else(|_|"0".into()),
        "flow_f16_tiled_control":std::env::var("NS_VCP_FLOW_F16_TILED").unwrap_or_else(|_|"0".into()),
        "linear_f16_tiled_control":std::env::var("NS_VCP_LINEAR_F16_TILED").unwrap_or_else(|_|"0".into()),
        "f32_kv_control":std::env::var("NEXTROCM_VCP_F32_KV").unwrap_or_else(|_|"0".into()),
        "conv_up_tiled_control":std::env::var("NS_VCP_CONV_UP_TILED").unwrap_or_else(|_|"0".into()),
        "op_wall_profile":std::env::var("NEXTROCM_VCP_PROFILE").unwrap_or_else(|_|"0".into()),
        "wav_dir":wav_dir,
        "request":{"text":request.prompt,"steps":request.steps,"cfg":request.cfg,"seconds":request.seconds,"seed":request.seed,"reference_sha256":reference_hash,"reference_text":flags.get("--ref-text")},
        "notes":["generation wall includes progress callbacks and final synchronization; no tensor taps",
        "hashing outside load/generation timing; independent driver memory must be measured externally",
        "warm means consecutive requests on the same loaded engine; no admission claim"]});
    let result = (|| -> Result<()> {
        nextrocm_engine_core::use_kind("audio");
        let gpu = Gpu::open(gpu_index)?;
        report["gpu"] = json!({"index":gpu_index,"name":gpu.name});
        let (initial, _) = budget::stats(&gpu)?;
        let mut options = LoadOptions::default();
        options.settings.insert("NS_VCP_INT8".into(), int8.into());
        options
            .settings
            .insert("NS_VCP_CONTINUE".into(), "0".into());
        let start = Instant::now();
        let loaded = VoxCpm2::load(&files, &gpu, &options, &mut |s| eprintln!("{s}"));
        gpu.sync()?;
        report["load_seconds"] = json!(start.elapsed().as_secs_f64());
        let work = match loaded {
            Err(e) => Err(e),
            Ok(engine) => {
                let mut rows = Vec::new();
                let work = (|| -> Result<()> {
                    let (resident, _) = budget::stats(&gpu)?;
                    report["resident_bytes"] = json!(resident);
                    let mut baseline: Option<Vec<f32>> = None;
                    for index in 0..repeats {
                        let mut events = Vec::new();
                        let start = Instant::now();
                        let generated=engine.generate(&request,&mut |s|{
                            events.push(json!({"phase":s.phase,"at":s.at,"of":s.of,"engine_seconds":s.seconds,"wall_seconds":start.elapsed().as_secs_f64()}));
                            Ok(())
                        });
                        gpu.sync()?;
                        let seconds = start.elapsed().as_secs_f64();
                        let (live, peak) = budget::stats(&gpu)?;
                        let audio = generated?;
                        if audio.rate != 48000
                            || audio.channels != 1
                            || audio.samples.is_empty()
                            || audio.samples.len() % 7680 != 0
                            || audio
                                .samples
                                .iter()
                                .any(|v| !v.is_finite() || v.abs() > 1.00001)
                        {
                            return Err(err("invalid complete PCM output"));
                        }
                        let hash = audio
                            .samples
                            .iter()
                            .flat_map(|v| v.to_bits().to_le_bytes())
                            .fold(0xcbf29ce484222325u64, |h, b| {
                                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
                            });
                        let identical = baseline.as_ref().is_none_or(|b| {
                            b.len() == audio.samples.len()
                                && b.iter()
                                    .zip(&audio.samples)
                                    .all(|(x, y)| x.to_bits() == y.to_bits())
                        });
                        let duration = audio.samples.len() as f64 / 48000.;
                        let wav = if let Some(dir) = wav_dir {
                            let path = dir.join(format!("repeat-{}.wav", index + 1));
                            let bytes = audio.wav(&[("INAM", &request.prompt)]);
                            let mut file = std::fs::OpenOptions::new()
                                .write(true)
                                .create_new(true)
                                .open(&path)
                                .map_err(err)?;
                            file.write_all(&bytes).map_err(err)?;
                            drop(file);
                            Some(json!({"path":path,"sha256":sha(&path)?,
                                "bytes":bytes.len(),"encoding":"pcm_s16le",
                                "sample_rate":audio.rate,"channels":audio.channels}))
                        } else {
                            None
                        };
                        rows.push(json!({"repeat":index+1,"generation_seconds":seconds,"audio_seconds":duration,"real_time_factor":seconds/duration,"samples":audio.samples.len(),"pcm_fnv1a64":format!("{hash:016x}"),"same_seed_bit_identical":identical,"progress":events,"live_after":live,"allocator_peak":peak,"wav":wav}));
                        if !identical || live != resident || peak >= budget::CAP as u64 {
                            return Err(err("repeat/cleanup/memory gate failed"));
                        }
                        if baseline.is_none() {
                            baseline = Some(audio.samples);
                        }
                    }
                    Ok(())
                })();
                report["runs"] = json!(rows);
                drop(engine);
                work
            }
        };
        gpu.sync()?;
        let (live, peak) = budget::stats(&gpu)?;
        report["allocations_after_drop"] = json!({"initial":initial,"live":live,"peak":peak});
        work?;
        if live != initial || peak >= budget::CAP as u64 {
            return Err(err("engine cleanup/memory gate failed"));
        }
        Ok(())
    })();
    report["completed"] = json!(result.is_ok());
    report["error"] = json!(result.as_ref().err().map(|e| &e.0));
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    writeln!(out, "{text}").map_err(err)?;
    println!("{text}");
    result
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e.0);
        std::process::exit(1);
    }
}
