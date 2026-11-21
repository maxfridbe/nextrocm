//! Test-only single-point injections; frozen native baseline must reproduce all stages.
use nextrocm_engine_core::{Error, Gpu, Result};
use nextrocm_vision_qwen3vl::{Budget, TextEncoder};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn sha(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg(p).output().map_err(err)?;
    if !o.status.success() {
        return Err(err("sha256sum failed"));
    }
    let s = String::from_utf8(o.stdout).map_err(err)?;
    Ok(s.split_whitespace()
        .next()
        .ok_or_else(|| err("empty hash"))?
        .to_owned())
}
fn hashes(paths: &[PathBuf]) -> Result<BTreeMap<String, String>> {
    paths
        .iter()
        .map(|p| Ok((p.display().to_string(), sha(p)?)))
        .collect()
}
fn floats(b: &[u8]) -> Result<Vec<f32>> {
    if !b.len().is_multiple_of(4) {
        return Err(err("F32 byte geometry"));
    }
    let v = b
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
    if v.iter().any(|x| !x.is_finite()) {
        return Err(err("nonfinite F32"));
    }
    Ok(v)
}
fn metric(a: &[u8], b: &[u8]) -> Result<serde_json::Value> {
    if a.len() != 32 * 4096 * 4 || a.len() != b.len() {
        return Err(err("stage geometry"));
    }
    let (a, b) = (floats(a)?, floats(b)?);
    let (mut d, mut n, mut max) = (0f64, 0f64, 0f64);
    for (&a, &b) in a.iter().zip(&b) {
        let v = a as f64 - b as f64;
        d += v * v;
        n += (b as f64).powi(2);
        max = max.max(v.abs());
    }
    Ok(
        serde_json::json!({"relative_l2":d.sqrt()/n.sqrt().max(1e-30),"max_abs":max,"elements":a.len()}),
    )
}
fn run() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 8 {
        return Err(err(
            "usage: layer1_split_injection WEIGHTS TOKENIZER CONFIG FIXTURES NEW_OUT GPU PROMPT",
        ));
    }
    let fixture = Path::new(&a[4]);
    let out = Path::new(&a[5]);
    if out.exists() {
        return Err(err("output exists"));
    }
    let manifest = fixture.join("manifest.json");
    if sha(&manifest)? != "5c16b6fcb4daa421879c0b7bce10d9d8d96c21e27896e14a2d222605d9a2f9a6" {
        return Err(err("fixture manifest identity"));
    }
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).map_err(err)?).map_err(err)?;
    let runtime = PathBuf::from(std::env::var("NEXTROCM_LIB").map_err(err)?);
    if !runtime.is_absolute()
        || sha(&runtime)? != "123f2623224665dbd877f8c6fc9f59987c081031215f36b6cdc6cc86902dfb49"
    {
        return Err(err("frozen runtime required"));
    }
    let mut paths = vec![
        std::fs::read_link(format!("/proc/{}/exe", std::process::id())).map_err(err)?,
        runtime,
        manifest,
    ];
    for p in [&a[1], &a[2], &a[3]] {
        paths.push(std::fs::canonicalize(p).map_err(err)?);
    }
    if sha(&paths[3])? != "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f" {
        return Err(err("checkpoint identity"));
    }
    let entries = m["files"].as_object().ok_or_else(|| err("fixture files"))?;
    let mut data = BTreeMap::new();
    for (name, record) in entries {
        if name.contains('/') || name.contains('\\') || name == ".." {
            return Err(err("fixture filename"));
        }
        let p = fixture.join(name);
        if p.symlink_metadata().map_err(err)?.file_type().is_symlink() {
            return Err(err("fixture symlink"));
        }
        let bytes = std::fs::read(&p).map_err(err)?;
        if sha(&p)? != record["sha256"].as_str().ok_or_else(|| err("file hash"))?
            || Some(bytes.len() as u64) != record["bytes"].as_u64()
        {
            return Err(err("fixture changed"));
        }
        if name.ends_with(".f32") {
            floats(&bytes)?;
        }
        data.insert(name.clone(), bytes);
        paths.push(p);
    }
    let before = hashes(&paths)?;
    let gpu = Gpu::open(a[6].parse().map_err(err)?)?;
    let engine = TextEncoder::load_with_options(
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        gpu.clone(),
        nextrocm_vision_qwen3vl::LoadOptions {
            probability_first: true,
            ordered_headnorm: true,
            ordered_rms: true,
            score_division: true,
        },
    )?;
    let (ids, drop) = engine.tokenize(&a[7])?;
    if ids.len() != 32 || drop != 14 {
        return Err(err("fixed diagnostic geometry"));
    }
    std::fs::create_dir(out).map_err(err)?;
    let budget = Budget {
        allowance_bytes: 7_500_000_000,
        driver_margin_bytes: 1024 * 1024 * 1024,
    };
    let targets = [
        ("tree_only", 1, "qkv.rope", "attention"),
        ("normalize_only", 1, "qkv.rope", "attention"),
    ];
    let mut records = BTreeMap::new();
    let mut artifacts = BTreeMap::new();
    for case in ["baseline", "tree_only", "normalize_only"] {
        let dir = out.join(case);
        std::fs::create_dir(&dir).map_err(err)?;
        let mut state = Vec::new();
        let mut stage_metrics = Vec::new();
        let mut quant = serde_json::Value::Null;
        let mut checked = std::collections::BTreeSet::new();
        let mut matched = std::collections::BTreeSet::new();
        let mut injected = 0;
        for layer in 0..3 {
            let input = state.clone();
            let mut next = Vec::new();
            let mut capture = |name: &str,
                               buf: &nextrocm_engine_core::DevBuf,
                               bytes: usize|
             -> Result<()> {
                let mut actual = vec![0; bytes];
                buf.read(0, &mut actual)?;
                if !name.ends_with("quantized") && !name.ends_with("accum") {
                    floats(&actual)?;
                }
                for &(target, which, source, point) in &targets {
                    if which != layer || (case != "baseline" && case != target) {
                        continue;
                    }
                    if name == source {
                        if actual != data[&format!("{target}.input.f32")] {
                            return Err(err("same-input precondition differs"));
                        }
                        checked.insert(target);
                    }
                    if name == point {
                        if !checked.contains(target)
                            || actual != data[&format!("{target}.preimage.f32")]
                        {
                            return Err(err("target preimage differs"));
                        }
                        matched.insert(target);
                        if case == target {
                            let replacement = &data[&format!("{target}.replacement.f32")];
                            if replacement.len() != actual.len() {
                                return Err(err("replacement shape"));
                            }
                            let p = dir.join("target-preimage.f32");
                            std::fs::write(&p, &actual).map_err(err)?;
                            artifacts.insert(
                                p.strip_prefix(out).map_err(err)?.display().to_string(),
                                sha(&p)?,
                            );
                            buf.write(0, replacement)?;
                            gpu.sync()?;
                            let mut verify = vec![0; bytes];
                            buf.read(0, &mut verify)?;
                            if &verify != replacement {
                                return Err(err("injection readback"));
                            }
                            injected += 1;
                        }
                    }
                }
                if name == "embed" && actual != data["native-stage-00.f32"] {
                    return Err(err("embedding baseline"));
                }
                if name == "block_input" && actual != input {
                    return Err(err("chain input differs"));
                }
                if (layer == 0 && name == "layer0") || (layer > 0 && name == "block_output") {
                    let stage = layer + 1;
                    if actual.len() != 32 * 4096 * 4 {
                        return Err(err("stage shape"));
                    }
                    if case == "baseline" && actual != data[&format!("native-stage-{stage:02}.f32")]
                    {
                        return Err(err("baseline stage bits differ; interventions invalid"));
                    }
                    stage_metrics.push(metric(
                        &actual,
                        &data[&format!("cpu-stage-{stage:02}.f32")],
                    )?);
                    let p = dir.join(format!("stage-{stage:02}.f32"));
                    std::fs::write(&p, &actual).map_err(err)?;
                    artifacts.insert(
                        p.strip_prefix(out).map_err(err)?.display().to_string(),
                        sha(&p)?,
                    );
                    next = actual.clone();
                }
                if layer == 2 && name == "qkv.quantized" {
                    let reference = &data["cpu-block2-qkv.i8"];
                    if actual.len() != 32 * 4096 || actual.len() != reference.len() {
                        return Err(err("QKV A8 shape"));
                    }
                    let counts = (0..32)
                        .map(|r| {
                            actual[r * 4096..(r + 1) * 4096]
                                .iter()
                                .zip(&reference[r * 4096..(r + 1) * 4096])
                                .filter(|(a, b)| a != b)
                                .count()
                        })
                        .collect::<Vec<_>>();
                    let index = 23 * 4096 + 1039;
                    let observed = actual[index] as i8;
                    let expected = reference[index] as i8;
                    if case == "baseline"
                        && (counts.iter().sum::<usize>() != 1
                            || counts[23] != 1
                            || observed != 22
                            || expected != 23)
                    {
                        return Err(err("critical baseline flip not reproduced"));
                    }
                    quant = serde_json::json!({"mismatches":counts.iter().sum::<usize>(),"per_row":counts,"row23_col1039":observed,"original_row23_col1039":expected});
                    let p = dir.join("block2-qkv.i8");
                    std::fs::write(&p, &actual).map_err(err)?;
                    artifacts.insert(
                        p.strip_prefix(out).map_err(err)?.display().to_string(),
                        sha(&p)?,
                    );
                }
                Ok(())
            };
            match layer {
                0 => engine.trace_layer0(&a[7], &budget, &mut capture)?,
                1 => engine.trace_block1(&a[7], &budget, &floats(&input)?, &mut capture)?,
                _ => engine.trace_block2(&a[7], &budget, &floats(&input)?, &mut capture)?,
            }
            if next.len() != 32 * 4096 * 4 {
                return Err(err("missing stage output"));
            }
            state = next;
        }
        if case == "baseline" {
            if injected != 0 || matched.len() != 2 {
                return Err(err("baseline preimages incomplete"));
            }
        } else if injected != 1 || matched.len() != 1 {
            return Err(err("exactly one intervention required"));
        }
        if quant.is_null() || stage_metrics.len() != 3 {
            return Err(err("incomplete diagnostic chain"));
        }
        records.insert(case,serde_json::json!({"stage_metrics":stage_metrics,"block2_qkv":quant,"injected":injected,"preimages_verified":matched}));
        println!("CAPTURE {case} complete");
    }
    gpu.sync()?;
    if before != hashes(&paths)? {
        return Err(err("input identity changed"));
    }
    let report = serde_json::json!({"capture_complete":true,"model_gate_pass":false,"scope":"two independent layer1 softmax splits: GPU exp with ATen tree/division; GPU exp with GPU tree/reciprocal multiplication; fixed ascending PV and fresh downstream native operations; no model admission","controls":{"P":1,"head":1,"RMS":1,"division":1},"identities":before,"ids":ids,"drop":drop,"cases":records,"files":artifacts,"gpu":{"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc}});
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(err)?,
    )
    .map_err(err)?;
    println!("CAPTURE COMPLETE causal injection diagnostic; no model admission");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
}
