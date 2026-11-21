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
    let rel = |start: usize, end: usize| {
        let mut d = 0f64;
        let mut n = 0f64;
        for i in start..end {
            d += (a[i] as f64 - b[i] as f64).powi(2);
            n += (b[i] as f64).powi(2);
        }
        d.sqrt() / n.sqrt().max(1e-30)
    };
    Ok(
        serde_json::json!({"relative_l2":d.sqrt()/n.sqrt().max(1e-30),"max_abs":max,"elements":a.len(),"row_relative_l2":(0..32).map(|r|rel(r*4096,(r+1)*4096)).collect::<Vec<_>>(),"cropped_relative_l2":rel(14*4096,32*4096)}),
    )
}
fn run() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 8 {
        return Err(err(
            "usage: current_attention_injection WEIGHTS TOKENIZER CONFIG FIXTURES NEW_OUT GPU PROMPT",
        ));
    }
    let fixture = Path::new(&a[4]);
    let out = Path::new(&a[5]);
    if out.exists() {
        return Err(err("output exists"));
    }
    let manifest = fixture.join("manifest.json");
    if sha(&manifest)? != "58d73ef600ef8ecf6ae92990997e640dc521500280b8f78425e5080d5b26d808" {
        return Err(err("fixture manifest identity"));
    }
    let m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest).map_err(err)?).map_err(err)?;
    let runtime = PathBuf::from(std::env::var("NEXTROCM_LIB").map_err(err)?);
    if !runtime.is_absolute()
        || sha(&runtime)? != "793e31427b2e9107e024d61e48e704bb98615d03639c605c9be9c68723c0adfa"
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
    let engine = TextEncoder::load_with_tree32(
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
        true,
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
        ("layer0_attention", 0, "qkv.rope", "attention"),
        ("layer1_attention", 1, "qkv.rope", "attention"),
    ];
    let mut records = BTreeMap::new();
    let mut artifacts = BTreeMap::new();
    for case in ["baseline", "layer0_attention", "layer1_attention"] {
        let dir = out.join(case);
        std::fs::create_dir(&dir).map_err(err)?;
        let mut state = Vec::new();
        let mut stage_metrics = Vec::new();
        let mut quant = BTreeMap::new();
        let mut checked = std::collections::BTreeSet::new();
        let mut matched = std::collections::BTreeSet::new();
        let mut injected = 0;
        for layer in 0..4 {
            let input = state.clone();
            let mut next = Vec::new();
            let mut capture =
                |name: &str, buf: &nextrocm_engine_core::DevBuf, bytes: usize| -> Result<()> {
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
                        if case == "baseline"
                            && actual != data[&format!("native-stage-{stage:02}.f32")]
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
                    // Retain all integer codes and float quantization boundaries, not only tracked coordinates.
                    if name.ends_with(".quantized")
                        || name.ends_with(".rotated")
                        || name.ends_with(".row_scale")
                    {
                        let ext = if name.ends_with(".quantized") {
                            "i8"
                        } else {
                            "f32"
                        };
                        let p = dir.join(format!("block{layer}-{name}.{ext}"));
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
                2 => engine.trace_block2(&a[7], &budget, &floats(&input)?, &mut capture)?,
                _ => engine.trace_block3(&a[7], &budget, &floats(&input)?, &mut capture)?,
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
        for (block, prefix, columns, observed, expected) in [
            (2, "out", vec![3245], vec![-47i8], vec![-46i8]),
            (
                3,
                "qkv",
                vec![2340, 2431, 3508],
                vec![-29, -30, 1],
                vec![-30, -29, 0],
            ),
        ] {
            let key = format!("block{block}-{prefix}");
            let codes = std::fs::read(dir.join(format!("{key}.quantized.i8"))).map_err(err)?;
            let reference = &data[&format!("cpu-{key}.quantized.i8")];
            if codes.len() != 32 * 4096 || reference.len() != codes.len() {
                return Err(err("quant geometry"));
            }
            let counts = (0..32)
                .map(|r| {
                    codes[r * 4096..(r + 1) * 4096]
                        .iter()
                        .zip(&reference[r * 4096..(r + 1) * 4096])
                        .filter(|(x, y)| x != y)
                        .count()
                })
                .collect::<Vec<_>>();
            let rot = floats(&std::fs::read(dir.join(format!("{key}.rotated.f32"))).map_err(err)?)?;
            let scale =
                floats(&std::fs::read(dir.join(format!("{key}.row_scale.f32"))).map_err(err)?)?;
            let cr = floats(&data[&format!("cpu-{key}.rotated.f32")])?;
            let cs = floats(&data[&format!("cpu-{key}.row_scale.f32")])?;
            if rot.len() != 32 * 4096
                || cr.len() != rot.len()
                || scale.len() != 32
                || cs.len() != 32
                || scale[2] <= 0.0
                || cs[2] <= 0.0
            {
                return Err(err("quant float geometry"));
            }
            let mut tracked = Vec::new();
            for ((col, base), want) in columns.into_iter().zip(observed).zip(expected) {
                let i = 2 * 4096 + col;
                if reference[i] as i8 != want || (case == "baseline" && codes[i] as i8 != base) {
                    return Err(err("tracked baseline/reference code not reproduced"));
                }
                let q = rot[i] / scale[2];
                let cq = cr[i] / cs[2];
                if !q.is_finite() || !cq.is_finite() {
                    return Err(err("nonfinite quotient"));
                }
                let tie = (base as f32 + want as f32) * 0.5;
                tracked.push(serde_json::json!({"row":2,"column":col,"code":codes[i] as i8,"reference_code":want,"restored_reference":codes[i] as i8==want,"rotated":rot[i],"reference_rotated":cr[i],"scale":scale[2],"reference_scale":cs[2],"quotient_rn":q,"reference_quotient_rn":cq,"tie":tie,"tie_margin":q-tie,"reference_tie_margin":cq-tie}));
            }
            let baseline = std::fs::read(out.join("baseline").join(format!("{key}.quantized.i8")))
                .map_err(err)?;
            if baseline.len() != codes.len() {
                return Err(err("baseline quant geometry"));
            }
            let mut new_flips = Vec::new();
            let mut restored = Vec::new();
            let mut changed_wrong = Vec::new();
            for i in 0..codes.len() {
                let coord = serde_json::json!({"row":i/4096,"column":i%4096,"baseline":baseline[i] as i8,"observed":codes[i] as i8,"reference":reference[i] as i8});
                if baseline[i] == reference[i] && codes[i] != reference[i] {
                    new_flips.push(coord);
                } else if baseline[i] != reference[i] && codes[i] == reference[i] {
                    restored.push(coord);
                } else if baseline[i] != codes[i] && codes[i] != reference[i] {
                    changed_wrong.push(coord);
                }
            }
            quant.insert(key,serde_json::json!({"mismatches":counts.iter().sum::<usize>(),"per_row":counts,"tracked":tracked,"new_flips":new_flips,"restored":restored,"changed_but_wrong":changed_wrong}));
        }
        if quant.len() != 2 || stage_metrics.len() != 4 {
            return Err(err("incomplete diagnostic chain"));
        }
        records.insert(case,serde_json::json!({"stage_metrics":stage_metrics,"quantization":quant,"injected":injected,"preimages_verified":matched}));
        println!("CAPTURE {case} complete");
    }
    gpu.sync()?;
    if before != hashes(&paths)? {
        return Err(err("input identity changed"));
    }
    let report = serde_json::json!({"capture_complete":true,"model_gate_pass":false,"scope":"two independent current-tree layer0/layer1 original Torch attention substitutions; fresh native chain through block3; diagnostic only","controls":{"P":1,"head":1,"RMS":1,"division":1,"tree32":1},"identities":before,"ids":ids,"drop":drop,"cases":records,"files":artifacts,"gpu":{"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc}});
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
