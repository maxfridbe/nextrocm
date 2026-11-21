//! Native graph/independent CPU fixture comparison. No subprocesses or inference backends.
//! Usage: sd15_numeric FIXTURE_DIR CHECKPOINT TOKENIZER_DIR [--vae PATH]
//!        [--stage all|clip|unet|vae|sample|primitives|inspect] [--sd1-euler shared-f64|oracle-f32] [--sd1-timestep native|recorded] [--gpu 0]
//!        [--atol .005 --rtol .005 --rms .001] [--output report.json] [--dump-latent native.f32]
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_image::{Error, Result, Sampler};
use nextrocm_image_sd15::{
    check,
    kernels::{Attention, Kernels, Weight, WeightType},
    schedule,
    tokenizer::Tokenizer,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};

struct Fixtures {
    dir: PathBuf,
    manifest: Value,
    seen: BTreeSet<String>,
    rows: Vec<Value>,
    atol: f64,
    rtol: f64,
    rms: f64,
    failed: bool,
    propagated_vae: bool,
}
fn normalized_error(error: f64, reference: f64) -> Option<f64> {
    if reference > 0.0 {
        Some(error / reference)
    } else if error == 0.0 {
        Some(0.0)
    } else {
        None
    }
}
fn uses_trajectory_gate(all_stage: bool, name: &str) -> bool {
    all_stage && name.starts_with("vae.") && name != "vae.decoder.conv_out" && name != "vae.rgb"
}
fn trajectory_pass(
    nonfinite: usize,
    relative_l2: Option<f64>,
    peak_normalized_max: Option<f64>,
) -> bool {
    nonfinite == 0
        && relative_l2.is_some_and(|v| v.is_finite() && v <= 0.001)
        && peak_normalized_max.is_some_and(|v| v.is_finite() && v <= 0.001)
}
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn hash(path: &Path) -> Result<String> {
    let mut f = File::open(path).map_err(err)?;
    let mut h = Sha256::new();
    let mut b = vec![0; 1 << 20];
    loop {
        let n = f.read(&mut b).map_err(err)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
impl Fixtures {
    fn entry(&self, name: &str) -> Result<&Value> {
        self.manifest["tensors"]
            .get(name)
            .ok_or_else(|| err(format!("missing fixture {name}")))
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        serde_json::from_value(self.entry(name)?["shape"].clone()).map_err(err)
    }
    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        let v = self.entry(name)?;
        let file = v["file"]
            .as_str()
            .ok_or_else(|| err("fixture filename missing"))?;
        if Path::new(file)
            .components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
        {
            return Err(err("invalid relative fixture path"));
        }
        let path = self.dir.join(file);
        let bytes = fs::read(&path).map_err(err)?;
        if bytes.len() as u64
            != v["bytes"]
                .as_u64()
                .ok_or_else(|| err("fixture byte count missing"))?
        {
            return Err(err(format!("{name}: byte count mismatch")));
        }
        if format!("{:x}", Sha256::digest(&bytes)) != v["sha256"].as_str().unwrap_or("") {
            return Err(err(format!("{name}: SHA256 mismatch")));
        }
        Ok(bytes)
    }
    fn floats(&self, name: &str) -> Result<Vec<f32>> {
        if self.entry(name)?["dtype"] != "F32" {
            return Err(err(format!("{name}: expected F32 fixture")));
        }
        let b = self.bytes(name)?;
        let shape = self.shape(name)?;
        let n = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| err("fixture shape overflow"))?;
        if n.checked_mul(4) != Some(b.len()) {
            return Err(err("fixture shape/bytes mismatch"));
        }
        let v: Vec<_> = b
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        if v.iter().any(|v| !v.is_finite()) {
            return Err(err(format!("{name}: nonfinite CPU reference")));
        }
        Ok(v)
    }
    fn ids(&self, name: &str) -> Result<[u32; 77]> {
        if self.entry(name)?["dtype"] != "I64" {
            return Err(err("token IDs must be I64"));
        }
        let b = self.bytes(name)?;
        if b.len() != 77 * 8 {
            return Err(err("expected 77 IDs"));
        }
        let mut ids = [0; 77];
        for (i, b) in b.chunks_exact(8).enumerate() {
            ids[i] = u32::try_from(i64::from_le_bytes(b.try_into().map_err(err)?)).map_err(err)?;
        }
        Ok(ids)
    }
    fn compare(&mut self, name: &str, shape: &[usize], actual: &[f32]) -> Result<()> {
        let expected = self.floats(name)?;
        let mut target = self.shape(name)?;
        // Oracle negative CLIP hooks retain batch axis; the native tower is batch-one token-major.
        if name.starts_with("negative.clip.") && target.len() == 3 && target[0] == 1 {
            target.remove(0);
        }
        if target != shape || actual.len() != expected.len() {
            return Err(err(format!(
                "{name}: native shape {shape:?}, expected {target:?}"
            )));
        }
        let (mut max, mut square, mut relative, mut bad, mut nonfinite) =
            (0f64, 0f64, 0f64, 0usize, 0usize);
        for (&a, &b) in actual.iter().zip(&expected) {
            if !a.is_finite() {
                nonfinite += 1;
                continue;
            }
            let d = (a as f64 - b as f64).abs();
            max = max.max(d);
            square += d * d;
            relative = relative.max(d / (b as f64).abs().max(1e-6));
            if d > self.atol + self.rtol * (b as f64).abs() {
                bad += 1;
            }
        }
        let rms = (square / actual.len().max(1) as f64).sqrt();
        let reference_rms = (expected.iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
            / expected.len().max(1) as f64)
            .sqrt();
        let reference_peak = expected
            .iter()
            .map(|&v| (v as f64).abs())
            .fold(0.0, f64::max);
        // A zero reference has no relative error scale: require exact agreement.
        let relative_l2 = normalized_error(rms, reference_rms);
        let peak_normalized_max = normalized_error(max, reference_peak);
        let strict_pass = nonfinite == 0 && bad == 0 && rms <= self.rms;
        let propagated = uses_trajectory_gate(self.propagated_vae, name);
        let pass = if propagated {
            trajectory_pass(nonfinite, relative_l2, peak_normalized_max)
        } else {
            strict_pass
        };
        self.failed |= !pass;
        self.seen.insert(name.into());
        let row = json!({"name":name,"shape":shape,"max_abs":max,"rms":rms,"max_relative_floor_1e-6":relative,"reference_rms":reference_rms,"reference_peak":reference_peak,"relative_l2":relative_l2,"peak_normalized_max":peak_normalized_max,"nonfinite":nonfinite,"elements_outside_abs_rel":bad,"strict_pass":strict_pass,"gate":if propagated {"propagated_vae_scale_aware"} else {"strict"},"pass":pass});
        println!("{row}");
        self.rows.push(row);
        Ok(())
    }
    fn required(&self, names: impl Iterator<Item = String>) -> Result<()> {
        let missing: Vec<_> = names.filter(|n| !self.seen.contains(n)).collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(err(format!("missing native tap coverage: {missing:?}")))
        }
    }
}
fn alloc_check(gpu: &Gpu, baseline: Option<(u64, u64)>, stage: &str) -> Result<Value> {
    let now = check::allocation_stats(gpu)?;
    if let (Some((before, _)), Some((after, peak))) = (baseline, now) {
        if before != after || peak >= 8_000_000_000 {
            return Err(err(format!(
                "{stage}: allocator before={before}, after={after}, peak={peak}"
            )));
        }
    }
    Ok(
        json!({"stage":stage,"allocator_before":baseline,"allocator_after":now,"drop_to_baseline":baseline.zip(now).map(|(a,b)|a.0==b.0)}),
    )
}
fn primitives(f: &mut Fixtures, gpu: &Arc<Gpu>) -> Result<()> {
    let k = Kernels::new(gpu.clone())?;
    {
        let x = DevBuf::from_f32(gpu, &f.floats("primitive.norm.input")?)?;
        let w = DevBuf::from_f32(gpu, &f.floats("primitive.norm.weight")?)?;
        let b = DevBuf::from_f32(gpu, &f.floats("primitive.norm.bias")?)?;
        let y = DevBuf::f32(gpu, 320 * 3 * 5)?;
        k.group_norm(
            &x,
            Some(Weight::new(&w, WeightType::F32)),
            Some(Weight::new(&b, WeightType::F32)),
            &y,
            1,
            320,
            3,
            5,
            32,
            1e-6,
        )?;
        gpu.sync()?;
        f.compare("primitive.norm.output", &[1, 320, 3, 5], &y.to_f32()?)?;
        k.layout(&x, &y, 1, 320, 3, 5, true)?;
        gpu.sync()?;
        f.compare("primitive.layout.nhwc", &[1, 3, 5, 320], &y.to_f32()?)?;
    }
    for dim in [40, 80, 160] {
        let p = format!("primitive.attention{dim}");
        let qs = f.shape(&format!("{p}.q"))?;
        let ks = f.shape(&format!("{p}.k"))?;
        // Independent fixture uses [batch,heads,tokens,dim].
        if qs.len() != 4 || ks.len() != 4 {
            return Err(err("attention fixture must be BHTD"));
        }
        let (b, n, m, h, d) = (
            qs[0] as i64,
            qs[2] as i64,
            ks[2] as i64,
            qs[1] as i64,
            qs[3] as i64,
        );
        let c = h * d;
        let q = DevBuf::from_f32(gpu, &f.floats(&format!("{p}.q"))?)?;
        let key = DevBuf::from_f32(gpu, &f.floats(&format!("{p}.k"))?)?;
        let v = DevBuf::from_f32(gpu, &f.floats(&format!("{p}.v"))?)?;
        let y = DevBuf::f32(gpu, (b * n * c) as usize)?;
        k.attention(
            &q,
            &key,
            &v,
            &y,
            Attention {
                batch: b,
                queries: n,
                keys: m,
                heads: h,
                dim: d,
                qb: n * c,
                qt: d,
                qh: n * d,
                kb: m * c,
                kt: d,
                kh: m * d,
                vb: m * c,
                vt: d,
                vh: m * d,
                ob: n * c,
                ot: d,
                oh: n * d,
                scale: 1.0 / (d as f32).sqrt(),
                causal: 0,
            },
        )?;
        gpu.sync()?;
        f.compare(&format!("{p}.output"), &qs, &y.to_f32()?)?;
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("SD1.5 numeric check FAILED: {}", e.0);
        std::process::exit(1);
    }
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        return Err(err("usage: sd15_numeric FIXTURE_DIR CHECKPOINT TOKENIZER_DIR [--vae PATH] [--stage all|clip|unet|vae|sample|primitives|inspect] [--sd1-euler shared-f64|oracle-f32] [--sd1-timestep native|recorded] [--gpu N] [--atol X --rtol X --rms X] [--output report.json] [--dump-latent native.f32]"));
    }
    let (dir, checkpoint, tokenizer_dir) = (
        PathBuf::from(&args[0]),
        PathBuf::from(&args[1]),
        PathBuf::from(&args[2]),
    );
    let mut dump_latent: Option<PathBuf> = None;
    let mut euler_arithmetic = check::EulerArithmetic::SharedF64;
    let mut use_recorded_timesteps = false;
    let (mut stage, mut vae, mut gpu_index, mut atol, mut rtol, mut rms, mut report) = (
        "all".to_owned(),
        None,
        0usize,
        0.005f64,
        0.005f64,
        0.001f64,
        None,
    );
    let mut i = 3;
    while i < args.len() {
        let value = args
            .get(i + 1)
            .ok_or_else(|| err("option requires a value"))?;
        match args[i].as_str() {
            "--stage" => stage = value.clone(),
            "--sd1-timestep" => {
                use_recorded_timesteps = match value.as_str() {
                    "native" => false,
                    "recorded" => true,
                    _ => return Err(err("--sd1-timestep requires native or recorded")),
                }
            }
            "--sd1-euler" => {
                euler_arithmetic = match value.as_str() {
                    "shared-f64" => check::EulerArithmetic::SharedF64,
                    "oracle-f32" => check::EulerArithmetic::OracleF32,
                    _ => return Err(err("--sd1-euler requires shared-f64 or oracle-f32")),
                }
            }
            "--vae" => vae = Some(PathBuf::from(value)),
            "--gpu" => gpu_index = value.parse().map_err(err)?,
            "--atol" => atol = value.parse().map_err(err)?,
            "--rtol" => rtol = value.parse().map_err(err)?,
            "--rms" => rms = value.parse().map_err(err)?,
            "--output" => report = Some(PathBuf::from(value)),
            "--dump-latent" => dump_latent = Some(PathBuf::from(value)),
            _ => return Err(err(format!("unknown option {}", args[i]))),
        }
        i += 2;
    }
    if ![atol, rtol, rms].iter().all(|v| v.is_finite() && *v >= 0.0)
        || !matches!(
            stage.as_str(),
            "all" | "clip" | "unet" | "vae" | "sample" | "primitives" | "inspect"
        )
    {
        return Err(err("invalid stage or tolerance"));
    }
    if euler_arithmetic == check::EulerArithmetic::OracleF32
        && !matches!(stage.as_str(), "sample" | "all")
    {
        return Err(err("--sd1-euler oracle-f32 requires --stage sample or all"));
    }
    if use_recorded_timesteps && !matches!(stage.as_str(), "sample" | "all") {
        return Err(err(
            "--sd1-timestep recorded requires --stage sample or all",
        ));
    }
    let manifest: Value =
        serde_json::from_slice(&fs::read(dir.join("manifest.json")).map_err(err)?).map_err(err)?;
    if manifest["schema"] != 1 || manifest["architecture"] != "sd15" || manifest["complete"] != true
    {
        return Err(err("incomplete or unsupported fixture manifest"));
    }
    let mut f = Fixtures {
        dir,
        manifest,
        seen: BTreeSet::new(),
        rows: Vec::new(),
        atol,
        rtol,
        rms,
        failed: false,
        propagated_vae: stage == "all",
    };
    let names: Vec<String> = f.manifest["tensors"]
        .as_object()
        .ok_or_else(|| err("missing tensor manifest"))?
        .keys()
        .cloned()
        .collect();
    // Verify all frozen artifacts even when selecting one stage.
    for name in &names {
        let _ = f.bytes(name)?;
    }
    if hash(&checkpoint)?
        != f.manifest["provenance"]["checkpoint"]["sha256"]
            .as_str()
            .unwrap_or("")
    {
        return Err(err("checkpoint SHA256 differs from oracle"));
    }
    if let Some(path) = &vae {
        if hash(path)?
            != f.manifest["provenance"]["vae"]["sha256"]
                .as_str()
                .unwrap_or("")
        {
            return Err(err("external VAE SHA256 differs from oracle"));
        }
    } else if !f.manifest["provenance"]["vae"].is_null() {
        return Err(err("oracle used external VAE; supply --vae"));
    }
    let tokenizer = Tokenizer::load(
        &tokenizer_dir.join("vocab.json"),
        &tokenizer_dir.join("merges.txt"),
    )?;
    for branch in ["positive", "negative"] {
        let key = if branch == "positive" {
            "prompt"
        } else {
            "negative"
        };
        let text = f.manifest["request"][key]
            .as_str()
            .ok_or_else(|| err("missing prompt"))?;
        if tokenizer.encode(text)? != f.ids(&format!("clip.{branch}.ids"))? {
            return Err(err(format!("{branch} tokenizer ID mismatch")));
        }
    }
    if stage == "inspect" {
        println!("verified {} artifacts, checkpoint/VAE hashes and both tokenizer sequences; no GPU opened",names.len());
        return Ok(());
    }
    if f.manifest["semantics"]["groupnorm_epsilon"].as_f64() != Some(1e-6)
        || !f.manifest["semantics"]["gelu"]
            .as_str()
            .unwrap_or("")
            .to_ascii_lowercase()
            .contains("tanh")
    {
        return Err(err("native target requires development-host GN1e-6 and tanh GEGLU; select authoritative -development-host fixtures"));
    }
    let size = f.manifest["request"]["size"]
        .as_u64()
        .ok_or_else(|| err("missing size"))? as usize;
    let h = size / 8;
    let steps = f.manifest["request"]["steps"]
        .as_u64()
        .ok_or_else(|| err("missing steps"))? as usize;
    if euler_arithmetic == check::EulerArithmetic::OracleF32
        && f.manifest["semantics"]["sampler"] != "Euler"
    {
        return Err(err(
            "oracle-f32 is diagnostic Euler arithmetic, not a DPM++ option",
        ));
    }
    // Hash-verified recorded scalars, loaded before opening a GPU.
    let recorded_timesteps = if use_recorded_timesteps {
        let mut values = Vec::with_capacity(steps);
        for step in 0..steps {
            let name = format!("step.{step}.timestep");
            if f.shape(&name)? != [1] {
                return Err(err("recorded timestep must have shape [1]"));
            }
            let value = f.floats(&name)?[0];
            if !(0.0..=999.0).contains(&value) {
                return Err(err("recorded timestep outside SD1 training range"));
            }
            values.push(value);
        }
        Some(values)
    } else {
        None
    };
    let gpu = Gpu::open(gpu_index)?;
    let baseline = check::allocation_stats(&gpu)?;
    let mut memory = Vec::new();
    if stage == "all" || stage == "primitives" {
        primitives(&mut f, &gpu)?;
        memory.push(alloc_check(&gpu, baseline, "primitives")?);
    }
    let mut pos = f.floats("clip.positive.context")?;
    let mut neg = f.floats("clip.negative.context")?;
    if stage == "all" || stage == "clip" {
        for branch in ["positive", "negative"] {
            let ids = f.ids(&format!("clip.{branch}.ids"))?;
            let mut tap = |name: &str, shape: &[usize], v: &[f32]| {
                let name = if branch == "negative" {
                    format!("negative.{name}")
                } else {
                    name.into()
                };
                f.compare(&name, shape, v)
            };
            let out = check::clip(gpu.clone(), &checkpoint, &ids, Some(&mut tap))?;
            f.compare(&format!("clip.{branch}.context"), &[77, 768], &out.value)?;
            memory.push(json!({"stage":format!("clip.{branch}"),"weights":out.memory.weights,"activation_peak":out.memory.activations_peak}));
            if branch == "positive" {
                pos = out.value
            } else {
                neg = out.value
            };
            memory.push(alloc_check(&gpu, baseline, &format!("clip.{branch}"))?);
        }
        f.required(
            names
                .iter()
                .filter(|n| n.starts_with("clip.") || n.starts_with("negative.clip."))
                .filter(|n| !n.ends_with(".ids"))
                .cloned(),
        )?;
    }
    if stage == "unet" {
        let input = f.floats("step.0.input")?;
        let t = f.floats("step.0.timestep")?[0];
        for (branch, ctx) in [("positive", &pos), ("negative", &neg)] {
            let mut tap = |name: &str, shape: &[usize], v: &[f32]| f.compare(name, shape, v);
            let out = check::unet(
                gpu.clone(),
                &checkpoint,
                &input,
                h,
                h,
                t,
                ctx,
                if branch == "positive" {
                    Some(&mut tap)
                } else {
                    None
                },
            )?;
            f.compare(
                &format!("step.0.epsilon.{branch}"),
                &[1, 4, h, h],
                &out.value,
            )?;
            memory.push(json!({"stage":branch,"weights":out.memory.weights,"activation_peak":out.memory.activations_peak}));
            memory.push(alloc_check(&gpu, baseline, branch)?);
        }
        f.required(names.iter().filter(|n| n.starts_with("unet.")).cloned())?;
    }
    let mut latent = f.floats(&format!("step.{}.latent", steps - 1))?;
    if stage == "all" || stage == "sample" {
        f.compare("schedule.train_sigmas", &[1000], &schedule::train_sigmas())?;
        f.compare(
            "schedule.sigmas",
            &[steps + 1],
            &schedule::karras(steps as u32)?,
        )?;
        let initial = f.floats("latent.initial")?;
        let sigmas = f.floats("schedule.sigmas")?;
        let cfg = f.manifest["request"]["cfg"]
            .as_f64()
            .ok_or_else(|| err("missing CFG"))? as f32;
        let sampler = match f.manifest["semantics"]["sampler"].as_str() {
            Some("Euler") => Sampler::Euler,
            Some("DPM++2M") => Sampler::DpmPp2m,
            _ => return Err(err("unsupported oracle sampler")),
        };
        let mut tap = |name: &str, shape: &[usize], v: &[f32]| f.compare(name, shape, v);
        let out = check::sample_with_controls(
            gpu.clone(),
            &checkpoint,
            check::SampleInputs {
                initial: &initial,
                sigmas: &sigmas,
                positive: &pos,
                negative: &neg,
                height: h,
                width: h,
                cfg,
                sampler,
            },
            Some(&mut tap),
            euler_arithmetic,
            recorded_timesteps.as_deref(),
        )?;
        latent = out.value;
        memory.push(json!({"stage":"sample","weights":out.memory.weights,"activation_peak":out.memory.activations_peak}));
        memory.push(alloc_check(&gpu, baseline, "sample")?);
        f.required(
            names
                .iter()
                .filter(|n| n.starts_with("unet.") || n.starts_with("step."))
                .cloned(),
        )?;
    }
    if let Some(path) = dump_latent {
        if !matches!(stage.as_str(), "all" | "sample") {
            return Err(err(
                "--dump-latent requires all or sample; do not label CPU inputs as native outputs",
            ));
        }
        let bytes: Vec<u8> = latent.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&path, &bytes).map_err(err)?;
        memory.push(json!({"native_final_latent":path,"shape":[1,4,h,h],"dtype":"F32_LE","bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    if stage == "all" || stage == "vae" {
        let mut tap = |name: &str, shape: &[usize], v: &[f32]| f.compare(name, shape, v);
        let out = check::vae(
            gpu.clone(),
            &checkpoint,
            vae.as_deref(),
            &latent,
            h,
            h,
            Some(&mut tap),
        )?;
        let reference = nextrocm_image_sd15::vae::rgb(&f.floats("vae.rgb")?, size, size)?;
        let mse = out
            .value
            .data
            .iter()
            .zip(&reference.data)
            .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
            .sum::<f64>()
            / reference.data.len() as f64;
        let max_byte_error = out
            .value
            .data
            .iter()
            .zip(&reference.data)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap_or(0);
        let psnr = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64.powi(2) / mse).log10()
        };
        let image_pass = max_byte_error <= 1 && psnr >= 60.0;
        f.failed |= !image_pass;
        memory.push(json!({"stage":"vae","weights":out.memory.weights,"activation_peak":out.memory.activations_peak,"pixel_mse":mse,"psnr_db":if mse==0.0{None}else{Some(psnr)},"pixels_exact":mse==0.0,"max_byte_error":max_byte_error,"image_pass":image_pass,"image_gate":{"max_byte_error":1,"min_psnr_db":60.0}}));
        memory.push(alloc_check(&gpu, baseline, "vae")?);
        f.required(
            names
                .iter()
                .filter(|n| n.starts_with("vae.decoder.") || n.as_str() == "vae.post_quant_conv")
                .cloned(),
        )?;
    }
    let result = json!({"pass":!f.failed,"stage":stage,"diagnostic_euler_arithmetic":if matches!(stage.as_str(),"all"|"sample") {Some(match euler_arithmetic {check::EulerArithmetic::SharedF64=>"shared-f64",check::EulerArithmetic::OracleF32=>"oracle-f32"})}else{None},"diagnostic_timestep":if matches!(stage.as_str(),"all"|"sample") {Some(if use_recorded_timesteps {"recorded"}else{"native"})}else{None},"fixture":f.dir,"semantics":f.manifest["semantics"],"tolerances":{"atol":atol,"rtol":rtol,"rms":rms,"status":"explicit numerical gate; not an admission claim","propagated_vae":{"enabled":stage=="all","relative_l2_max":0.001,"peak_normalized_max":0.001,"scope":"all-stage VAE intermediate taps only; final float conv_out and isolated stages remain strict","rationale":"RV128 isolated UNet and VAE passed strict CPU-input gates. Full-trajectory VAE reference RMS grows to 75.46 with relative L2 <=0.000217; absolute hidden RMS grows with signal amplitude. Keep strict_pass and raw failures visible. Thresholds are bounded propagation budgets, not exemptions for nonfinite values, wrong shapes, missing taps, or final image errors; must also validate 512 and DreamShaper."}},"coverage":f.seen,"metrics":f.rows,"memory":memory});
    if let Some(path) = report {
        fs::write(path, serde_json::to_vec_pretty(&result).map_err(err)?).map_err(err)?;
    }
    println!("{result}");
    if f.failed {
        return Err(err("one or more numerical tolerances failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trajectory_budget_is_narrow_and_rejects_spikes_and_nonfinite_values() {
        assert!(uses_trajectory_gate(true, "vae.decoder.up.0.block.2"));
        assert!(!uses_trajectory_gate(false, "vae.decoder.up.0.block.2"));
        assert!(!uses_trajectory_gate(true, "unet.out"));
        assert!(!uses_trajectory_gate(true, "vae.decoder.conv_out"));
        assert!(trajectory_pass(0, Some(0.000217), Some(0.00027)));
        assert!(!trajectory_pass(0, Some(0.00101), Some(0.00027)));
        assert!(!trajectory_pass(0, Some(0.000217), Some(0.00101)));
        assert!(!trajectory_pass(1, Some(0.0), Some(0.0)));
        assert!(!trajectory_pass(0, Some(f64::NAN), Some(0.0)));
        assert_eq!(normalized_error(0.0, 0.0), Some(0.0));
        assert_eq!(normalized_error(0.01, 0.0), None);
        assert!(!trajectory_pass(0, None, Some(0.0)));
    }
    #[test]
    fn oracle_reader_verifies_bytes_ids_and_metric_failures() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/sd15-cpu/rv51-128-development-host");
        if !dir.join("manifest.json").exists() {
            return;
        }
        let manifest =
            serde_json::from_slice(&fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        let mut f = Fixtures {
            dir,
            manifest,
            seen: BTreeSet::new(),
            rows: vec![],
            atol: 1e-5,
            rtol: 1e-5,
            rms: 1e-5,
            failed: false,
            propagated_vae: false,
        };
        let ids = f.ids("clip.positive.ids").unwrap();
        assert_eq!(ids[0], 49406);
        let x = f.floats("clip.final").unwrap();
        f.compare("clip.final", &[77, 768], &x).unwrap();
        assert!(!f.failed);
        assert!(f.compare("clip.final", &[1, 77, 768], &x).is_err());
        let mut bad = x.clone();
        bad[0] = f32::NAN;
        f.compare("clip.final", &[77, 768], &bad).unwrap();
        assert!(f.failed);
        assert!(f.required(["missing.tap".to_owned()].into_iter()).is_err());
    }
}
