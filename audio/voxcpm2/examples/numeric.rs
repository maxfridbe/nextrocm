//! CPU structure inspection or parent-serialized native stage diagnostics.
//! numeric --fixture DIR --cpu-only [--output NEW_JSON]
//! numeric --fixture DIR --model MODEL_DIR --int8 0|1 [--gpu 0]
//!         [--stages prefill|prompt,prefill,patches,vae] [--output NEW_JSON]
//! Numerical metrics are NOT an admission pass; frozen acceptance-v1 controls,
//! independent driver memory and lifecycle validation remain required.
use nextrocm_audio::{Error, LoadOptions, Result};
use nextrocm_audio_voxcpm2::{VoxCpm2, budget, check, files_in};
use nextrocm_engine_core::Gpu;
use serde_json::json;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

const USAGE: &str = "numeric --fixture DIR (--cpu-only | --model MODEL_DIR --int8 0|1) [--gpu 0] [--stages prefill|prompt,prefill,patches,vae|replay|states|fsq] [--fsq-input CPU_STATE_DIR] [--dump NEW_DIR] [--output NEW_JSON]";
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}

// Validator-only integrity utility, outside timed inference. No Python or other
// inference runtime is used. Argument arrays preserve spaces in file paths.
fn sha256(path: &Path) -> Result<String> {
    let result = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(err)?;
    let text = String::from_utf8(result.stdout).map_err(err)?;
    let hash = text.split_whitespace().next().unwrap_or("");
    if !result.status.success() || hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(err(format!("sha256sum failed for {}", path.display())));
    }
    Ok(hash.to_owned())
}

fn hashes(dir: &Path) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(dir).map_err(err)? {
        let path = entry.map_err(err)?.path();
        if path.is_file() {
            out.insert(
                path.file_name()
                    .ok_or_else(|| err("invalid file name"))?
                    .to_string_lossy()
                    .into_owned(),
                sha256(&path)?,
            );
        }
    }
    Ok(out)
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut flags = BTreeMap::new();
    let mut cpu = false;
    while let Some(k) = args.next() {
        if k == "--help" {
            println!("{USAGE}");
            return Ok(());
        }
        if k == "--cpu-only" && !cpu {
            cpu = true;
            continue;
        }
        if !matches!(
            k.as_str(),
            "--fixture"
                | "--model"
                | "--int8"
                | "--gpu"
                | "--stages"
                | "--output"
                | "--dump"
                | "--fsq-input"
        ) {
            return Err(err(USAGE));
        }
        let v = args.next().ok_or_else(|| err(USAGE))?;
        if flags.insert(k, v).is_some() {
            return Err(err("duplicate flag"));
        }
    }
    let fixture = PathBuf::from(flags.get("--fixture").ok_or_else(|| err(USAGE))?);
    let stages: Vec<_> = flags
        .get("--stages")
        .map(String::as_str)
        .unwrap_or("prefill")
        .split(',')
        .collect();
    if stages.iter().any(|s| {
        !matches!(
            *s,
            "prompt" | "prefill" | "patches" | "vae" | "replay" | "states" | "fsq"
        )
    }) {
        return Err(err("unknown stage"));
    }
    let gpu_index: usize = flags
        .get("--gpu")
        .map(String::as_str)
        .unwrap_or("0")
        .parse()
        .map_err(err)?;
    if !cpu
        && (!flags.contains_key("--model")
            || !matches!(flags.get("--int8").map(String::as_str), Some("0" | "1")))
    {
        return Err(err(USAGE));
    }
    let fsq_input = flags.get("--fsq-input").map(PathBuf::from);
    if stages.contains(&"fsq") != fsq_input.is_some()
        || (fsq_input.is_some() && (cpu || stages != ["fsq"]))
    {
        return Err(err(
            "--fsq-input requires only --stages fsq and native execution",
        ));
    }
    let dump = flags.get("--dump").map(PathBuf::from);
    if (stages.contains(&"replay") || stages.contains(&"states") || stages.contains(&"fsq"))
        && dump.is_none()
    {
        return Err(err("replay requires --dump"));
    }
    if let Some(path) = &dump {
        std::fs::create_dir(path).map_err(err)?;
    }
    let fsq_hashes = fsq_input.as_deref().map(hashes).transpose()?;
    let mut output = flags
        .get("--output")
        .map(|p| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(p)
                .map_err(err)
        })
        .transpose()?;
    let mut report = json!({"schema":"voxcpm2-numeric-v1","fixture":fixture,"cpu_only":cpu,
        "model":flags.get("--model"),"int8":flags.get("--int8"),"stages":stages,
        "acceptance":"PENDING: evaluate retained native tensors with tooling/voxcpm2-accept.py",
        "prefill_tap_contract":"base_lm.forward/pre_fsq;residual_lm.forward",
        "f32_rows_control":std::env::var("NEXTROCM_VCP_F32_ROWS").unwrap_or_else(|_|"0".into()),
        "fsq_input_hashes":fsq_hashes,
        "rope_bf16_cache_control":std::env::var("NEXTROCM_VCP_ROPE_BF16_CACHE").unwrap_or_else(|_|"0".into()),
        "flow_f16_tiled_control":std::env::var("NS_VCP_FLOW_F16_TILED").unwrap_or_else(|_|"0".into()),
        "linear_f16_tiled_control":std::env::var("NS_VCP_LINEAR_F16_TILED").unwrap_or_else(|_|"0".into()),
        "f32_kv_control":std::env::var("NEXTROCM_VCP_F32_KV").unwrap_or_else(|_|"0".into()),
        "conv_up_tiled_control":std::env::var("NS_VCP_CONV_UP_TILED").unwrap_or_else(|_|"0".into()),
        "op_wall_profile":std::env::var("NEXTROCM_VCP_PROFILE").unwrap_or_else(|_|"0".into()),
        "hash_verification":"not performed; verify fixture manifests separately before acceptance",
        "notes":["stage times include synchronization, allocation and diagnostic host downloads",
            "prefill-only exits at prompt completion without running flow sampling",
            "allocator peak is context-wide and cumulative, not device-wide driver usage"]});
    let result = (|| -> Result<()> {
        let t0 = Instant::now();
        report["fixture_structure"] = check::inspect_fixture(&fixture)?;
        report["fixture_read_seconds"] = json!(t0.elapsed().as_secs_f64());
        if dump.is_some() {
            report["fixture_files"] = json!(hashes(&fixture)?);
        }
        if cpu {
            return Ok(());
        } // Must precede library discovery and GPU open.
        let model = PathBuf::from(flags.get("--model").ok_or_else(|| err(USAGE))?);
        let files = files_in(&model)?;
        if dump.is_some() {
            let model_hashes: BTreeMap<_, _> = files
                .iter()
                .map(|(role, path)| Ok((role.clone(), sha256(path)?)))
                .collect::<Result<_>>()?;
            report["model_hashes"] = json!(model_hashes);
            report["executable_sha256"] = json!(sha256(&std::env::current_exe().map_err(err)?)?);
        }
        let mut options = LoadOptions::default();
        options
            .settings
            .insert("NS_VCP_INT8".into(), flags["--int8"].clone());
        nextrocm_engine_core::use_kind("audio");
        let gpu = Gpu::open(gpu_index)?;
        report["gpu"] = json!({"index":gpu_index,"name":gpu.name});
        let (initial, _) = budget::stats(&gpu)?;
        let t0 = Instant::now();
        let loaded = VoxCpm2::load(&files, &gpu, &options, &mut |s| eprintln!("{s}"));
        report["load_seconds"] = json!(t0.elapsed().as_secs_f64());
        let work = match loaded {
            Err(e) => Err(e),
            Ok(engine) => {
                let mut rows = Vec::new();
                let work = (|| -> Result<()> {
                    let (resident, _) = budget::stats(&gpu)?;
                    report["resident_bytes"] = json!(resident);
                    for stage in &stages {
                        let stage_dump = dump.as_ref().map(|d| d.join(stage));
                        if let Some(path) = &stage_dump {
                            std::fs::create_dir(path).map_err(err)?;
                        }
                        let mut logs = Vec::new();
                        let t0 = Instant::now();
                        let mut replay = None;
                        let measured = if *stage == "fsq" {
                            let input = fsq_input
                                .as_deref()
                                .ok_or_else(|| err("missing FSQ input"))?;
                            let result = check::fsq_states(
                                &engine,
                                input,
                                stage_dump
                                    .as_deref()
                                    .ok_or_else(|| err("missing FSQ dump"))?,
                            )?;
                            if Some(hashes(input)?) != fsq_hashes {
                                return Err(err("FSQ input changed during execution"));
                            }
                            replay = Some(result);
                            Ok(0.)
                        } else if matches!(*stage, "replay" | "states") {
                            (if *stage == "states" {
                                check::replay_states
                            } else {
                                check::replay
                            })(
                                &engine,
                                &fixture,
                                stage_dump
                                    .as_deref()
                                    .ok_or_else(|| err("missing replay dump"))?,
                            )
                            .map(|v| {
                                replay = Some(v);
                                0.
                            })
                        } else {
                            check::run_with_dump(
                                &engine,
                                &fixture,
                                &[*stage],
                                &mut |s| logs.push(s),
                                stage_dump.as_deref(),
                            )
                        };
                        let elapsed = t0.elapsed().as_secs_f64();
                        gpu.sync()?;
                        let (live, peak) = budget::stats(&gpu)?;
                        let finite = measured.as_ref().is_ok_and(|v| v.is_finite());
                        rows.push(json!({"stage":stage,"wall_seconds":elapsed,"logs":logs,
                            "max_relative_l2":if matches!(*stage,"replay"|"states"|"fsq") {None} else {measured.as_ref().ok().filter(|v| v.is_finite())},
                            "replay":replay,"comparison_performed":!matches!(*stage,"replay"|"states"|"fsq"),
                            "finite_complete_comparison":finite && !matches!(*stage,"replay"|"states"|"fsq"),"live_bytes_after":live,"peak_bytes":peak,
                            "error":measured.as_ref().err().map(|e| &e.0)}));
                        measured?;
                        if !finite || live != resident || peak >= budget::CAP as u64 {
                            return Err(err(
                                "nonfinite/mismatched comparison or allocation cleanup/cap failure",
                            ));
                        }
                    }
                    Ok(())
                })();
                report["measurements"] = json!(rows);
                drop(engine);
                work
            }
        };
        gpu.sync()?;
        let (live, peak) = budget::stats(&gpu)?;
        report["allocations_after_drop"] =
            json!({"initial_bytes":initial,"current_bytes":live,"peak_bytes":peak});
        work?;
        if live != initial || peak >= budget::CAP as u64 {
            return Err(err("engine cleanup/cap failure"));
        }
        Ok(())
    })();
    report["completed"] = json!(result.is_ok());
    report["error"] = json!(result.as_ref().err().map(|e| &e.0));
    if let Some(root) = &dump {
        let mut outputs = BTreeMap::new();
        for stage in &stages {
            let dir = root.join(stage);
            if dir.is_dir() {
                outputs.insert(*stage, hashes(&dir)?);
            }
        }
        report["output_hashes"] = json!(outputs);
        std::fs::write(
            root.join("native.json"),
            serde_json::to_vec_pretty(&report).map_err(err)?,
        )
        .map_err(err)?;
    }
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    if let Some(out) = output.as_mut() {
        writeln!(out, "{text}").map_err(err)?;
    }
    println!("{text}");
    result
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e.0);
        std::process::exit(1);
    }
}
