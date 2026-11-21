//! Isolated first-block check. Parent pins and builds a diagnostic runtime that
//! includes both packed linear and DiT normalization operators; no service use.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image_qwenimage21::{block::Block, ffi::PackedKernel};
use serde_json::{json, Value};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn hash(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum").arg("--").arg(path).output()?;
    if !output.status.success() {
        return Err("sha256sum failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let sha = text.split_whitespace().next().ok_or("missing SHA256")?;
    if sha.len() != 64 || !sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid SHA256".into());
    }
    Ok(sha.into())
}

fn values(path: &Path, count: usize) -> Result<Vec<f32>> {
    let n = count.checked_mul(4).ok_or("fixture length overflow")?;
    if fs::metadata(path)?.len() != n as u64 {
        return Err("fixture byte count mismatch".into());
    }
    let raw = fs::read(path)?;
    let out: Vec<_> = raw
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
        .collect();
    if out.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite fixture".into());
    }
    Ok(out)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: block_numeric ABS_Q4_GGUF FIXTURE_DIR NEW_OUTPUT_DIR GPU_INDEX [0|1|paired]; absolute NEXTROCM_LIB required".into());
    }
    let selection = args.get(4).map(String::as_str);
    let kernels = if selection == Some("paired") {
        vec![PackedKernel::Baseline, PackedKernel::Register64]
    } else {
        vec![PackedKernel::from_setting(selection)?]
    };
    let model = fs::canonicalize(&args[0])?;
    let fixture = fs::canonicalize(&args[1])?;
    let output = PathBuf::from(&args[2]);
    let runtime = PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB missing")?);
    if !runtime.is_absolute() || output.exists() {
        return Err("runtime must be absolute; output must be new".into());
    }
    let executable = fs::canonicalize("/proc/self/exe")?;
    let manifest_path = fixture.join("manifest.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest["schema"] != 1
        || manifest["primary_oracle"] != true
        || manifest["layer"] != 0
        || manifest["relative_l2_max"].as_f64() != Some(0.001)
    {
        return Err("invalid primary block manifest".into());
    }
    let m = usize::try_from(manifest["rows"].as_u64().ok_or("rows")?)?;
    let text = usize::try_from(manifest["text_rows"].as_u64().ok_or("text rows")?)?;
    if !(text > 0 && text < m && m <= 4096) {
        return Err("invalid joint rows".into());
    }
    let mut identities = BTreeMap::new();
    for path in [&model, &runtime, &executable, &manifest_path] {
        identities.insert(path.clone(), hash(path)?);
    }
    if identities[&model] != "0f5a11ebbaf3110a16120311454eaac35db13798e0102339e4f2b8fd5be6ba82" {
        return Err("wrong Q4 checkpoint".into());
    }
    let mut arrays = BTreeMap::new();
    for (name, shape) in [
        ("input", vec![m, 4096]),
        ("modulation", vec![2, 16384]),
        ("rope", vec![m, 128]),
        ("expected", vec![m, 4096]),
    ] {
        let meta = &manifest["files"][name];
        let filename = format!("{name}.f32");
        if meta["file"] != filename || meta["shape"] != json!(shape) {
            return Err("fixture layout differs".into());
        }
        let path = fixture.join(filename);
        let sha = hash(&path)?;
        if meta["sha256"] != sha {
            return Err("fixture hash differs".into());
        }
        identities.insert(path.clone(), sha);
        arrays.insert(name, values(&path, shape.iter().product())?);
    }
    let gpu = Gpu::open(args[3].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    let pointer = api()?.symbol("nr_allocation_stats");
    if pointer.is_null() {
        return Err("allocation stats absent".into());
    }
    // SAFETY: matching private runtime statistics ABI; runtime stays loaded.
    let stats: Stats = unsafe { std::mem::transmute(pointer) };
    let stats_now = || -> Result<(u64, u64)> {
        let (mut live, mut peak) = (0, 0);
        nextrocm_image_qwenimage21::ffi::check(
            unsafe { stats(gpu.raw(), &mut live, &mut peak) },
            "allocation stats",
        )?;
        Ok((live, peak))
    };
    if stats_now()?.0 != 0 {
        return Err("expected empty initial allocator".into());
    }
    fs::create_dir(&output)?;
    fs::write(
        output.join("identities-before.json"),
        serde_json::to_vec_pretty(&identities)?,
    )?;
    let file = Gguf::open(&model)?;
    let mut baseline_bytes = None;
    let mut trials = Vec::new();
    let mut passed = true;
    for kernel in &kernels {
        let begin = Instant::now();
        let block = Block::load_with_kernel(&file, &gpu, 0, 256 * 1024 * 1024, *kernel)?;
        let load_seconds = begin.elapsed().as_secs_f64();
        println!("effective NS_QI_PACKED64={}", kernel.setting());
        let begin = Instant::now();
        let result = block.joint(
            &arrays["input"],
            text,
            &arrays["modulation"],
            &arrays["rope"],
        )?;
        let run_seconds = begin.elapsed().as_secs_f64();
        let got = result.to_f32()?;
        if got.len() != arrays["expected"].len() || got.iter().any(|x| !x.is_finite()) {
            return Err("invalid native output".into());
        }
        let (mut squared, mut reference, mut maximum) = (0f64, 0f64, 0f64);
        for (&x, &y) in got.iter().zip(&arrays["expected"]) {
            let delta = f64::from(x) - f64::from(y);
            squared += delta * delta;
            reference += f64::from(y) * f64::from(y);
            maximum = maximum.max(delta.abs());
        }
        let relative_l2 = (squared / reference.max(1e-30)).sqrt();
        let raw: Vec<u8> = got.iter().flat_map(|x| x.to_le_bytes()).collect();
        if let Some(baseline) = &baseline_bytes {
            if baseline != &raw {
                return Err("paired block outputs are not bitwise identical".into());
            }
        } else {
            baseline_bytes = Some(raw.clone());
        }
        fs::write(
            output.join(format!("flag{}-block0.f32", kernel.setting())),
            raw,
        )?;
        drop(result);
        drop(block);
        gpu.sync()?;
        let (live, peak) = stats_now()?;
        let trial_pass = relative_l2.is_finite()
            && relative_l2 <= 0.001
            && live == 0
            && peak < 1024 * 1024 * 1024;
        passed &= trial_pass;
        trials.push(
            json!({"setting":kernel.setting(),"pass":trial_pass,"load_seconds":load_seconds,
            "run_seconds":run_seconds,"relative_l2":relative_l2,"max_absolute":maximum,
            "tracked_live":live,"tracked_peak_cumulative":peak}),
        );
        if !trial_pass {
            break;
        }
    }
    for (path, before) in &identities {
        if hash(path)? != *before {
            return Err("identity changed during check".into());
        }
    }
    fs::write(
        output.join("identities-after.json"),
        serde_json::to_vec_pretty(&identities)?,
    )?;
    let report = json!({"status":if passed{"PASS"}else{"FAIL"},
        "scope":"one joint block, original F32 CPU oracle; opt-in packed64 comparison",
        "same_binary_runtime":true,"cross_flag_bitwise":if trials.len()==2 {Some(true)} else {None},
        "gpu":gpu.name,"rows":m,"text_rows":text,"trials":trials,
        "primary_gate":0.001,"fixture_manifest":manifest});
    fs::write(
        output.join("comparison.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    if !passed {
        return Err("isolated block numeric/memory gate failed".into());
    }
    Ok(())
}
