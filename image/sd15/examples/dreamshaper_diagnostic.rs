//! Exact CPU-input UNet replay; no sampler, CLIP generation, or gate relaxation.
//! Build: cargo build -p nextrocm-image-sd15 --example dreamshaper_diagnostic
//! Inspect (default, no GPU): dreamshaper_diagnostic FIXTURE CHECKPOINT [--steps 0,8,9]
//! Execute ONLY in parent's serialized GPU window: append --run --output NEW_REPORT.json
//! --trajectory --capture-dir NEW_DIR records a shared-F64 Euler trajectory;
//! --recorded-timesteps independently fixes timesteps. --steps selects dumped
//! tensors only in trajectory mode; all steps still execute and compare.
//! Defaults to every recorded step. Full fixtures are on development-host under
//! /tmp/nextrocm-sd15-reference/ds8-512-development-host; the repo retains only a subset.
//! Block references exist ONLY for step 0 positive. Later replay localizes an
//! error to an exact-input UNet call, not to an individual internal operator.
//! Step-0 taps identify the first divergent block on a propagated native pass;
//! they do not inject CPU hidden states into each block or prove its root cause.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{Error, Result};
use nextrocm_image_sd15::check;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
};

const ATOL: f64 = 0.005;
const RTOL: f64 = 0.005;
const RMS: f64 = 0.001;
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).map_err(err)?;
    let mut h = Sha256::new();
    let mut buf = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(err)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
struct Oracle {
    dir: PathBuf,
    manifest: Value,
}
impl Oracle {
    fn entry(&self, name: &str) -> Result<&Value> {
        self.manifest["tensors"]
            .get(name)
            .ok_or_else(|| err(format!("missing tensor {name}")))
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        serde_json::from_value(self.entry(name)?["shape"].clone()).map_err(err)
    }
    fn read(&self, name: &str) -> Result<Vec<f32>> {
        let e = self.entry(name)?;
        let file = e["file"].as_str().ok_or_else(|| err("missing filename"))?;
        if file.is_empty()
            || Path::new(file)
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err(err("invalid relative fixture filename"));
        }
        if e["dtype"] != "F32" {
            return Err(err(format!("{name}: expected F32")));
        }
        let bytes = fs::read(self.dir.join(file)).map_err(|e| err(format!("{name}: {e}")))?;
        let shape = self.shape(name)?;
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
            .ok_or_else(|| err("invalid shape/product"))?;
        if count.checked_mul(4) != Some(bytes.len())
            || e["bytes"].as_u64() != Some(bytes.len() as u64)
            || e["sha256"].as_str() != Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
        {
            return Err(err(format!("{name}: shape/bytes/hash mismatch")));
        }
        let v: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(err(format!("{name}: nonfinite reference")));
        }
        Ok(v)
    }
    fn compare(&self, name: &str, shape: &[usize], actual: &[f32]) -> Result<Value> {
        if self.shape(name)? != shape {
            return Err(err(format!("{name}: shape mismatch")));
        }
        let mut row = metric(name, actual, &self.read(name)?)?;
        row["shape"] = json!(shape);
        if let Some(examples) = row["failure_examples"].as_array_mut() {
            for example in examples {
                let mut index = example["index"].as_u64().unwrap_or(0) as usize;
                let mut coordinates = vec![0; shape.len()];
                for (axis, &dim) in shape.iter().enumerate().rev() {
                    coordinates[axis] = index % dim;
                    index /= dim;
                }
                example["coordinates"] = json!(coordinates);
            }
        }
        Ok(row)
    }
}
fn metric(name: &str, actual: &[f32], expected: &[f32]) -> Result<Value> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err(err("comparison length mismatch"));
    }
    let (mut max, mut square, mut bad, mut nonfinite, mut worst) =
        (0f64, 0f64, 0usize, 0usize, 0usize);
    let mut examples = Vec::new();
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !b.is_finite() {
            return Err(err("nonfinite reference"));
        }
        if !a.is_finite() {
            nonfinite += 1;
            if examples.len() < 8 {
                examples.push(json!({"index":i,"actual":a.to_string(),"reference":b}));
            }
            continue;
        }
        let d = (a as f64 - b as f64).abs();
        square += d * d;
        if d > max {
            max = d;
            worst = i;
        }
        if d > ATOL + RTOL * (b as f64).abs() {
            bad += 1;
            if examples.len() < 8 {
                examples.push(json!({"index":i,"actual":a,"reference":b,"abs_error":d,"allowed_error":ATOL+RTOL*(b as f64).abs()}));
            }
        }
    }
    let rms = (square / actual.len() as f64).sqrt();
    Ok(
        json!({"name":name,"elements":actual.len(),"max_abs":max,"worst_index":worst,
        "worst_actual":actual[worst],"worst_reference":expected[worst],
        "rms":if nonfinite==0 {Some(rms)} else {None},"nonfinite":nonfinite,
        "elements_outside_abs_rel":bad,"failure_examples":examples,
        "pass":nonfinite==0 && bad==0 && rms<=RMS,"gate":"strict"}),
    )
}
fn blocks() -> BTreeSet<String> {
    let mut names = BTreeSet::from(["unet.time_embed".into(), "unet.out".into()]);
    for (group, n) in [
        ("input_blocks", 12),
        ("middle_block", 3),
        ("output_blocks", 12),
    ] {
        for i in 0..n {
            names.insert(format!("unet.{group}.{i}"));
        }
    }
    names
}
// Explicit operator-only mode; no checkpoint, graph, model load, or CPU inference backend.
fn gn_coverage(a: &[String]) -> Result<bool> {
    use nextrocm_engine_core::DevBuf;
    use nextrocm_image_sd15::kernels::{Kernels, Weight, WeightType};
    let dir = a
        .get(1)
        .ok_or_else(|| err("--gn-coverage FIXTURE [--run --output NEW.json]"))?;
    let mut execute = false;
    let mut output = None;
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--run" => execute = true,
            "--output" => {
                i += 1;
                output = Some(PathBuf::from(
                    a.get(i).ok_or_else(|| err("missing output"))?,
                ));
            }
            _ => return Err(err("unknown GN coverage argument")),
        }
        i += 1;
    }
    if execute && output.is_none() {
        return Err(err("--run requires new --output"));
    }
    let bytes = fs::read(Path::new(dir).join("manifest.json")).map_err(err)?;
    let oracle = Oracle {
        dir: PathBuf::from(dir),
        manifest: serde_json::from_slice(&bytes).map_err(err)?,
    };
    let cases = oracle.manifest["cases"]
        .as_array()
        .ok_or_else(|| err("missing cases"))?;
    if cases.is_empty() {
        return Err(err("empty cases"));
    }
    for case in cases {
        let name = case["name"]
            .as_str()
            .ok_or_else(|| err("missing case name"))?;
        let shape = oracle.shape(&format!("{name}.input"))?;
        let is_ln = case["kind"] == "layer_norm";
        if (is_ln && shape.len() != 2)
            || (!is_ln && (case["kind"] != "group_norm" || shape.len() != 4 || shape[1] % 32 != 0))
        {
            return Err(err("invalid operator shape/kind"));
        }
        for suffix in ["input", "output", "weight", "bias"] {
            oracle.read(&format!("{name}.{suffix}"))?;
        }
        if oracle.shape(&format!("{name}.output"))? != shape
            || oracle.shape(&format!("{name}.weight"))? != [shape[1]]
            || oracle.shape(&format!("{name}.bias"))? != [shape[1]]
        {
            return Err(err("invalid affine/output shape"));
        }
    }
    if !execute {
        println!(
            "{}",
            json!({"mode":"gn_coverage_preflight","cases":cases.len(),"pass":true})
        );
        return Ok(true);
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.ok_or_else(|| err("missing output"))?)
        .map_err(err)?;
    let gpu = Gpu::open(0)?;
    let baseline = check::allocation_stats(&gpu)?;
    let kernels = Kernels::new(gpu.clone())?;
    let mut rows = Vec::new();
    for case in cases {
        let name = case["name"].as_str().ok_or_else(|| err("missing name"))?;
        let shape = oracle.shape(&format!("{name}.input"))?;
        let values = oracle.read(&format!("{name}.input"))?;
        let mut row = {
            let x = DevBuf::from_f32(&gpu, &values)?;
            let w = DevBuf::from_f32(&gpu, &oracle.read(&format!("{name}.weight"))?)?;
            let b = DevBuf::from_f32(&gpu, &oracle.read(&format!("{name}.bias"))?)?;
            let y = DevBuf::f32(&gpu, values.len())?;
            let output_buffer = if case["in_place"] == true { &x } else { &y };
            let affine = case["affine"] != false;
            let weight = affine.then(|| Weight::new(&w, WeightType::F32));
            let bias = affine.then(|| Weight::new(&b, WeightType::F32));
            if case["kind"] == "layer_norm" {
                kernels.layer_norm(
                    &x,
                    weight,
                    bias,
                    output_buffer,
                    shape[0] as i64,
                    shape[1] as i64,
                    1e-5,
                )?;
            } else {
                kernels.group_norm(
                    &x,
                    weight,
                    bias,
                    output_buffer,
                    shape[0] as i64,
                    shape[1] as i64,
                    shape[2] as i64,
                    shape[3] as i64,
                    32,
                    1e-6,
                )?;
            }
            gpu.sync()?;
            let values = output_buffer.to_f32()?;
            let mut row = oracle.compare(&format!("{name}.output"), &shape, &values)?;
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            row["output_sha256"] = json!(format!("{:x}", Sha256::digest(&bytes)));
            row
        };
        let after = check::allocation_stats(&gpu)?;
        if let (Some((before, _)), Some((current, peak))) = (baseline, after) {
            if before != current || peak >= 8_000_000_000 {
                return Err(err("operator allocation cleanup/budget failure"));
            }
        }
        row["allocator_after"] = json!(after);
        println!("{row}");
        rows.push(row);
    }
    let pass = rows.iter().all(|r| r["pass"] == true);
    serde_json::to_writer_pretty(&mut file, &json!({"mode":"gn_operator_coverage","pass":pass,"metrics":rows,
        "manifest_sha256":format!("{:x}",Sha256::digest(&bytes)),"provenance":oracle.manifest["provenance"],
        "tolerances":{"atol":ATOL,"rtol":RTOL,"rms":RMS}})).map_err(err)?;
    file.flush().map_err(err)?;
    Ok(pass)
}

fn run() -> Result<bool> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.first().is_some_and(|v| v == "--gn-coverage") {
        return gn_coverage(&a);
    }
    if a.len() < 2 {
        return Err(err("usage: dreamshaper_diagnostic FIXTURE CHECKPOINT [--steps 0,1,...] [--gpu N] [--run --output NEW_REPORT.json]"));
    }
    let dir = PathBuf::from(&a[0]);
    let checkpoint = PathBuf::from(&a[1]);
    let mut selected = None;
    let mut gpu_index = 0;
    let mut execute = false;
    let mut output = None;
    let mut trajectory = false;
    let mut recorded_timesteps = false;
    let mut capture_dir: Option<PathBuf> = None;
    let mut i = 2;
    while i < a.len() {
        match a[i].as_str() {
            "--run" => execute = true,
            "--trajectory" => trajectory = true,
            "--recorded-timesteps" => recorded_timesteps = true,
            "--steps" | "--gpu" | "--output" | "--capture-dir" => {
                let flag = &a[i];
                i += 1;
                let value = a.get(i).ok_or_else(|| err("missing option value"))?;
                match flag.as_str() {
                    "--steps" => {
                        selected = Some(
                            value
                                .split(',')
                                .map(|v| v.parse::<usize>().map_err(err))
                                .collect::<Result<BTreeSet<_>>>()?,
                        )
                    }
                    "--capture-dir" => capture_dir = Some(PathBuf::from(value)),
                    "--gpu" => gpu_index = value.parse::<usize>().map_err(err)?,
                    _ => output = Some(PathBuf::from(value)),
                }
            }
            _ => return Err(err(format!("unknown option {}", a[i]))),
        }
        i += 1;
    }
    if execute && output.is_none() {
        return Err(err("--run requires --output (new file)"));
    }
    if trajectory != capture_dir.is_some() || (recorded_timesteps && !trajectory) {
        return Err(err(
            "--trajectory requires --capture-dir; --recorded-timesteps requires --trajectory",
        ));
    }
    if capture_dir.as_ref().is_some_and(|p| p.exists()) {
        return Err(err("capture directory must be new"));
    }
    let manifest_bytes = fs::read(dir.join("manifest.json")).map_err(err)?;
    let oracle = Oracle {
        dir,
        manifest: serde_json::from_slice(&manifest_bytes).map_err(err)?,
    };
    let count = oracle.manifest["request"]["steps"]
        .as_u64()
        .ok_or_else(|| err("missing steps"))?;
    if count == 0 || count > 200 {
        return Err(err("invalid step count"));
    }
    let steps = selected.unwrap_or_else(|| (0..count as usize).collect());
    if steps.is_empty() || steps.iter().any(|&s| s >= count as usize) {
        return Err(err("step outside oracle trajectory"));
    }
    let cfg = oracle.manifest["request"]["cfg"]
        .as_f64()
        .ok_or_else(|| err("missing CFG"))? as f32;
    if !cfg.is_finite() {
        return Err(err("invalid CFG"));
    }
    let checkpoint_hash = hash(&checkpoint)?;
    if oracle.manifest["provenance"]["checkpoint"]["sha256"].as_str() != Some(&checkpoint_hash) {
        return Err(err("checkpoint differs from oracle"));
    }
    for branch in ["positive", "negative"] {
        let name = format!("clip.{branch}.context");
        if oracle.shape(&name)? != [77, 768] {
            return Err(err("invalid context shape"));
        }
        oracle.read(&name)?;
    }
    // Preflight all required bytes before opening a GPU, including every block tap.
    let block_names = blocks();
    if steps.contains(&0) {
        for name in &block_names {
            oracle.read(name)?;
        }
    }
    for &step in &steps {
        let shape = oracle.shape(&format!("step.{step}.input"))?;
        if shape.len() != 4
            || shape[0] != 1
            || shape[1] != 4
            || shape[2] == 0
            || shape[3] == 0
            || shape[2] > 64
            || shape[3] > 64
            || shape[2] % 8 != 0
            || shape[3] % 8 != 0
        {
            return Err(err("invalid latent geometry"));
        }
        for suffix in [
            "input",
            "epsilon.positive",
            "epsilon.negative",
            "epsilon.cfg",
        ] {
            let name = format!("step.{step}.{suffix}");
            if oracle.shape(&name)? != shape {
                return Err(err(format!("{name}: incompatible shape")));
            }
            oracle.read(&name)?;
        }
        let name = format!("step.{step}.timestep");
        if oracle.shape(&name)? != [1] {
            return Err(err("invalid timestep shape"));
        }
        oracle.read(&name)?;
    }
    if trajectory {
        oracle.read("latent.initial")?;
        oracle.read("schedule.sigmas")?;
        for step in 0..count {
            for suffix in [
                "input",
                "timestep",
                "epsilon.positive",
                "epsilon.negative",
                "epsilon.cfg",
                "x0",
                "latent",
            ] {
                oracle.read(&format!("step.{step}.{suffix}"))?;
            }
        }
        for name in &block_names {
            oracle.read(name)?;
        }
    }
    let mut report = json!({"mode":"preflight","steps":steps,"fixture":oracle.dir,"checkpoint":checkpoint,
        "checkpoint_sha256":checkpoint_hash,"manifest_sha256":format!("{:x}",Sha256::digest(&manifest_bytes)),
        "tolerances":{"atol":ATOL,"rtol":RTOL,"rms":RMS},"semantics":oracle.manifest["semantics"],
        "scope":"Each UNet consumes oracle step input, timestep and context; no native trajectory or scheduler. Block references only step 0 positive; block errors are propagated, not isolated operator errors."});
    if trajectory {
        report["scope"] = json!("Shared-F64 Euler with frozen initial latent, sigmas and contexts. All emitted tensors compared; selected steps saved for CPU replay on identical native inputs. No gate changes.");
        report["diagnostic_timestep"] = json!(if recorded_timesteps {
            "recorded"
        } else {
            "native"
        });
        report["capture_dir"] = json!(capture_dir);
    }
    // Reserve a new report before expensive execution; never overwrite parent's evidence.
    let mut file = output
        .as_ref()
        .map(|p| {
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(p)
                .map_err(err)
        })
        .transpose()?;
    if let Some(f) = file.as_mut() {
        serde_json::to_writer_pretty(&mut *f, &report).map_err(err)?;
        f.flush().map_err(err)?;
    }
    if !execute {
        println!("{}", serde_json::to_string_pretty(&report).map_err(err)?);
        return Ok(true);
    }
    let gpu = Gpu::open(gpu_index)?;
    let baseline = check::allocation_stats(&gpu)?;
    let pos = oracle.read("clip.positive.context")?;
    let neg = oracle.read("clip.negative.context")?;
    let mut metrics = Vec::new();
    let mut memory = Vec::new();
    if trajectory {
        let destination = capture_dir
            .as_ref()
            .ok_or_else(|| err("missing capture directory"))?;
        fs::create_dir(destination).map_err(err)?;
        let initial = oracle.read("latent.initial")?;
        let sigmas = oracle.read("schedule.sigmas")?;
        let times = (0..count)
            .map(|step| oracle.read(&format!("step.{step}.timestep")).map(|t| t[0]))
            .collect::<Result<Vec<_>>>()?;
        let shape = oracle.shape("latent.initial")?;
        let mut captures = Vec::new();
        let mut tap = |name: &str, shape: &[usize], values: &[f32]| -> Result<()> {
            let row = oracle.compare(name, shape, values)?;
            println!("{row}");
            metrics.push(row);
            let selected = name
                .strip_prefix("step.")
                .and_then(|n| n.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|step| steps.contains(&step));
            if selected {
                let path = destination.join(format!("{name}.f32"));
                let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
                let mut file = fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .map_err(err)?;
                file.write_all(&bytes).map_err(err)?;
                captures.push(
                    json!({"name":name,"path":path,"shape":shape,"dtype":"F32","bytes":bytes.len(),
                    "sha256":format!("{:x}",Sha256::digest(&bytes))}),
                );
            }
            Ok(())
        };
        let out = check::sample_with_controls(
            gpu.clone(),
            &checkpoint,
            check::SampleInputs {
                initial: &initial,
                sigmas: &sigmas,
                positive: &pos,
                negative: &neg,
                height: shape[2],
                width: shape[3],
                cfg,
                sampler: nextrocm_image::Sampler::Euler,
            },
            Some(&mut tap),
            check::EulerArithmetic::SharedF64,
            if recorded_timesteps {
                Some(&times)
            } else {
                None
            },
        )?;
        let after = check::allocation_stats(&gpu)?;
        if let (Some((before, _)), Some((current, peak))) = (baseline, after) {
            if before != current || peak >= 8_000_000_000 {
                return Err(err("allocation cleanup/budget failure"));
            }
        }
        memory.push(
            json!({"weights":out.memory.weights,"activations_peak":out.memory.activations_peak,
            "allocator_before":baseline,"allocator_after":after}),
        );
        report["captures"] = json!(captures);
    } else {
        for &step in &steps {
            let input = oracle.read(&format!("step.{step}.input"))?;
            let shape = oracle.shape(&format!("step.{step}.input"))?;
            let t = oracle.read(&format!("step.{step}.timestep"))?[0];
            let mut branch_outputs = Vec::new();
            for (branch, ctx) in [("positive", &pos), ("negative", &neg)] {
                let mut seen = BTreeSet::new();
                let mut tap = |name: &str, shape: &[usize], values: &[f32]| -> Result<()> {
                    let row = oracle.compare(name, shape, values)?;
                    println!("{row}");
                    metrics.push(row);
                    seen.insert(name.to_owned());
                    Ok(())
                };
                let out = check::unet(
                    gpu.clone(),
                    &checkpoint,
                    &input,
                    shape[2],
                    shape[3],
                    t,
                    ctx,
                    if step == 0 && branch == "positive" {
                        Some(&mut tap)
                    } else {
                        None
                    },
                )?;
                if step == 0 && branch == "positive" && seen != block_names {
                    return Err(err("incomplete/unexpected step-0 block coverage"));
                }
                let row =
                    oracle.compare(&format!("step.{step}.epsilon.{branch}"), &shape, &out.value)?;
                println!("{row}");
                metrics.push(row);
                let after = check::allocation_stats(&gpu)?;
                if let (Some((before, _)), Some((current, peak))) = (baseline, after) {
                    if before != current || peak >= 8_000_000_000 {
                        return Err(err("allocation cleanup/budget failure"));
                    }
                }
                memory.push(json!({"step":step,"branch":branch,"weights":out.memory.weights,
                "activations_peak":out.memory.activations_peak,"allocator_before":baseline,"allocator_after":after}));
                branch_outputs.push(out.value);
            }
            let guided: Vec<_> = branch_outputs[0]
                .iter()
                .zip(&branch_outputs[1])
                .map(|(p, n)| n + cfg * (p - n))
                .collect();
            metrics.push(oracle.compare(&format!("step.{step}.epsilon.cfg"), &shape, &guided)?);
        }
    }
    if trajectory {
        let mut required = block_names.clone();
        for step in 0..count {
            for suffix in [
                "input",
                "timestep",
                "epsilon.positive",
                "epsilon.negative",
                "epsilon.cfg",
                "x0",
                "latent",
            ] {
                required.insert(format!("step.{step}.{suffix}"));
            }
        }
        let seen: BTreeSet<_> = metrics
            .iter()
            .filter_map(|m| m["name"].as_str().map(str::to_owned))
            .collect();
        if seen != required || metrics.len() != required.len() {
            return Err(err("incomplete or duplicate trajectory coverage"));
        }
    }
    let pass = metrics.iter().all(|m| m["pass"] == true);
    report["mode"] = json!(if trajectory {
        "native_trajectory_capture"
    } else {
        "exact_input_unet"
    });
    report["pass"] = json!(pass);
    report["first_failed_block"] = metrics
        .iter()
        .find(|m| m["name"].as_str().is_some_and(|n| n.starts_with("unet.")) && m["pass"] == false)
        .cloned()
        .unwrap_or(Value::Null);
    report["metrics"] = json!(metrics);
    report["memory"] = json!(memory);
    if let Some(mut f) = file {
        use std::io::{Seek, SeekFrom};
        f.set_len(0).map_err(err)?;
        f.seek(SeekFrom::Start(0)).map_err(err)?;
        serde_json::to_writer_pretty(&mut f, &report).map_err(err)?;
        f.flush().map_err(err)?;
    }
    Ok(pass)
}
fn main() {
    match run() {
        Ok(true) => (),
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2)
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_gate_keeps_rms_and_nonfinite_failures() {
        let r = metric("rms", &[1.002, 1.002], &[1., 1.]).unwrap();
        assert_eq!(r["elements_outside_abs_rel"], 0);
        assert_eq!(r["pass"], false);
        let r = metric("nan", &[f32::NAN, 1.], &[1., 1.]).unwrap();
        assert_eq!(r["nonfinite"], 1);
        assert_eq!(r["pass"], false);
        assert_eq!(metric("equal", &[0., 1.], &[0., 1.]).unwrap()["pass"], true);
    }
    #[test]
    fn oracle_block_inventory_is_complete() {
        assert_eq!(blocks().len(), 29);
    }
}

#[cfg(test)]
mod trajectory_metric_tests {
    use super::*;
    #[test]
    fn indexed_failures_keep_frozen_gates() {
        let row = metric("test", &[0.0, 0.016, 2.0], &[0.0, 0.001, 2.0]).unwrap();
        assert_eq!(row["pass"], false);
        assert_eq!(row["elements_outside_abs_rel"], 1);
        assert_eq!(row["worst_index"], 1);
        assert_eq!(row["failure_examples"][0]["index"], 1);
        assert!(metric("test", &[f32::NAN], &[0.0]).unwrap()["pass"] == false);
        assert!(metric("test", &[1.0], &[f32::NAN]).is_err());
        assert_eq!(metric("test", &[1.0], &[1.0]).unwrap()["pass"], true);
    }
}
