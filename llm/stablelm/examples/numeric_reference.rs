//! Parent-run GPU check against tooling/stablelm-numeric-reference.py artifacts.
//! Usage: numeric_reference MODEL REFERENCE_DIR [ABS_TOL=0.05] [REL_TOL=0.01]
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::LoadOptions;
use serde_json::{json, Value};
use std::{collections::BTreeMap, error::Error, path::Path};

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("numeric_reference MODEL REFERENCE_DIR [ABS_TOL] [REL_TOL]".into());
    }
    let abs: f32 = args.get(3).map_or(Ok(0.05), |s| s.parse())?;
    let rel: f32 = args.get(4).map_or(Ok(0.01), |s| s.parse())?;
    if !abs.is_finite() || !rel.is_finite() || abs < 0.0 || rel < 0.0 {
        return Err("tolerances must be finite and nonnegative".into());
    }
    let dir = Path::new(&args[2]);
    let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json"))?)?;
    let ids: Vec<u32> = serde_json::from_value(manifest["input_ids"].clone())?;
    if ids.is_empty() {
        return Err("empty reference prompt".into());
    }
    let mut references = BTreeMap::new();
    for row in manifest["taps"].as_array().ok_or("missing taps")? {
        if row["mode"] != "fp16kv" {
            continue;
        }
        let name = row["name"].as_str().ok_or("tap name")?.to_owned();
        let file = row["file"].as_str().ok_or("tap file")?;
        let raw = std::fs::read(dir.join(file))?;
        if raw.len() % 4 != 0 {
            return Err("unaligned reference data".into());
        }
        let values: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite reference".into());
        }
        references.insert(name, values);
    }
    if !references.contains_key("result_output") {
        return Err("missing output reference".into());
    }
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(0)?;
    let file = Gguf::open(Path::new(&args[1]))?;
    let engine = nextrocm_llm_stablelm::load(
        &file,
        &[gpu],
        &LoadOptions {
            draft: false,
            kv: (ids.len() + 1, 1),
            ..Default::default()
        },
        &mut |s| eprintln!("{s}"),
    )?;
    let mut session = engine.session(ids.len() + 1)?;
    let mut failures = 0usize;
    let mut checked = 0usize;
    let mut final_logits = Vec::new();
    for (position, id) in ids.iter().enumerate() {
        final_logits = engine.feed(&mut session, &[*id], &mut |name: &str, buf: &DevBuf| {
            let Some(reference) = references.get(name) else { return Ok(()); };
            if name == "result_output" && position + 1 != ids.len() { return Ok(()); }
            let actual = buf.to_f32()?;
            let start = if name == "result_output" { 0 } else { position * actual.len() };
            let expected = reference.get(start..start + actual.len()).ok_or_else(|| nextrocm_engine_core::Error(format!("bad reference shape: {name}")))?;
            let mut max_abs = 0.0f32;
            let mut squared = 0.0f64;
            let mut bad = 0usize;
            for (a, b) in actual.iter().zip(expected) {
                let delta = (a - b).abs();
                max_abs = max_abs.max(delta);
                squared += f64::from(delta).powi(2);
                if !a.is_finite() || delta > abs + rel * b.abs() { bad += 1; }
            }
            failures += bad;
            checked += 1;
            println!("{}", json!({"tap":name,"position":position,"max_abs":max_abs,"rms":(squared/actual.len() as f64).sqrt(),"outside_tolerance":bad}));
            Ok(())
        })?;
    }
    let greedy = final_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or("no logits")?
        .0;
    let expected = manifest["runs"]["fp16kv"]["expected_greedy_tokens"][0]
        .as_u64()
        .ok_or("missing greedy token")?;
    let expected_checks = (references.len() - 1) * ids.len() + 1;
    println!(
        "{}",
        json!({"checked":checked,"expected_checks":expected_checks,"greedy":greedy,"expected_greedy":expected,"outside_tolerance":failures,"abs_tol":abs,"rel_tol":rel})
    );
    if failures != 0 || greedy as u64 != expected || checked != expected_checks {
        return Err("numerical reference comparison failed".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
