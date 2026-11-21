//! GPU comparison against the independent Qwen3.6 CPU oracle, with explicit tap coverage.
//! Usage: qwen36_batch_numeric MODEL REFERENCE_DIR CHUNK [EXPERT_MIB=0]
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::LoadOptions;
use serde_json::{json, Value};
use std::{collections::BTreeMap, error::Error, path::Path};

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("qwen36_batch_numeric MODEL REFERENCE_DIR CHUNK [EXPERT_MIB]".into());
    }
    let chunk: usize = args[3].parse()?;
    if !matches!(chunk, 1 | 8 | 32 | 64) {
        return Err("expected chunk1/8/32/64".into());
    }
    let expert_mib: usize = args.get(4).map_or(Ok(0), |s| s.parse())?;
    let expert_bytes = expert_mib
        .checked_mul(1024 * 1024)
        .ok_or("cache size overflow")?;
    std::env::set_var("NS_Q35_CHUNK", chunk.to_string());
    let abs: f32 = 0.05;
    let rel: f32 = 0.01;
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
    let supported = |name: &str| {
        matches!(name, "inp_embd" | "result_norm" | "result_output")
            || [
                "attn_norm-",
                "Qcur-",
                "Kcur-",
                "Vcur-",
                "delta_state-",
                "ffn_inp-",
                "ffn_norm-",
                "router_logits-",
                "router_ids-",
                "router_weights-",
                "l_out-",
            ]
            .iter()
            .any(|p| name.starts_with(p))
    };
    for row in manifest["taps"].as_array().ok_or("missing taps")? {
        if row["mode"] != "fp16kv" {
            continue;
        }
        let name = row["name"].as_str().ok_or("tap name")?.to_owned();
        if !supported(&name) {
            continue;
        }
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
    let engine = (nextrocm_llm::kind_for(
        &[nextrocm_llm_qwen35::kind(), nextrocm_llm_qwen3moe::kind()],
        &file,
    )?
    .load)(
        &file,
        &[gpu],
        &LoadOptions {
            expert_bytes: (expert_bytes > 0).then_some(expert_bytes),
            draft: false,
            kv: (ids.len() + 1, 1),
            ..Default::default()
        },
        &mut |s| eprintln!("{s}"),
    )?;
    if engine.prefill_chunk() != chunk {
        return Err("batch mode was not activated".into());
    }
    let mut session = engine.session(ids.len() + 1)?;
    let mut failures = 0usize;
    let mut checked = 0usize;
    let mut visits = BTreeMap::<String, usize>::new();
    let dump = std::env::var_os("NS_Q35_NATIVE_TAPS").map(std::path::PathBuf::from);
    if let Some(dir) = &dump {
        std::fs::create_dir_all(dir)?;
    }
    let final_logits = engine.feed(&mut session, &ids, &mut |name: &str, buf: &DevBuf| {
        let Some(reference) = references.get(name) else { return Ok(()); };
        let visit = visits.entry(name.to_owned()).or_default();
        *visit += 1;
        let final_state = name.starts_with("delta_state-");
        if final_state && *visit != ids.len() { return Ok(()); }
        let final_head = name == "result_output" || name == "result_norm";
        let position = if final_state || final_head { ids.len()-1 } else { *visit-1 };
        let actual = buf.to_f32()?;
        let hash = actual.iter().flat_map(|v| v.to_bits().to_le_bytes()).fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3));
        if let Some(dir) = &dump {
            if position == 50 && matches!(name, "ffn_norm-22" | "router_logits-22" | "router_ids-22" | "router_weights-22") {
                let bytes: Vec<u8> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(dir.join(format!("{name}-{position}.f32")), bytes).map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            }
        }
        let start = if final_state || name == "result_output" { 0 } else { position.checked_mul(actual.len()).ok_or_else(|| nextrocm_engine_core::Error("reference offset overflow".into()))? };
        let expected = reference.get(start..start+actual.len()).ok_or_else(|| nextrocm_engine_core::Error(format!("bad reference shape: {name}")))?;
        if actual.is_empty() { return Err(nextrocm_engine_core::Error("empty native tap".into())); }
        let mut max_abs = 0.0f32;
        let mut squared = 0.0f64;
        let mut bad = 0usize;
        for (a,b) in actual.iter().zip(expected) {
            let d = (a-b).abs(); max_abs = max_abs.max(d); squared += f64::from(d).powi(2);
            let mismatch = if name.starts_with("router_ids-") { a != b } else { d > abs + rel*b.abs() };
            if !a.is_finite() || mismatch { bad+=1; }
        }
        failures += bad; checked += 1;
        println!("{}",json!({"tap":name,"position":position,"native_bits_fnv64":format!("{hash:016x}"),"max_abs":max_abs,"rms":(squared/actual.len() as f64).sqrt(),"outside_tolerance":bad}));
        Ok(())
    })?;
    let greedy = final_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .ok_or("no logits")?
        .0;
    let expected = manifest["runs"]["fp16kv"]["top10"][0]["id"]
        .as_u64()
        .ok_or("missing greedy token")?;
    let expected_checks: usize = references
        .keys()
        .map(|name| {
            if name == "result_output" || name == "result_norm" || name.starts_with("delta_state-")
            {
                1
            } else {
                ids.len()
            }
        })
        .sum();
    for name in references.keys() {
        let want = if name == "result_output" || name == "result_norm" {
            1
        } else {
            ids.len()
        };
        if visits.get(name).copied().unwrap_or(0) != want {
            return Err(format!("tap coverage: {name}").into());
        }
    }
    println!(
        "{}",
        json!({"chunk":chunk,"expert_bytes":expert_bytes,"checked":checked,"expected_checks":expected_checks,"greedy":greedy,"expected_greedy":expected,"outside_tolerance":failures,"abs_tol":abs,"rel_tol":rel})
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
