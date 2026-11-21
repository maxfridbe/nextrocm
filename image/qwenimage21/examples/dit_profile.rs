//! Paired private velocity profiler. Test-only SHA subprocesses; inference
//! is Rust -> HIP. No service/model registration or image-quality admission.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image_qwenimage21::{
    denoiser::Denoiser,
    ffi::{self, PackedKernel},
    profiler, sched,
};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg("--").arg(p).output()?;
    if !o.status.success() {
        return Err("hash command failed".into());
    }
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("missing hash")?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn npy(p: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = fs::read(p)?;
    if b.len() < 12 || &b[..6] != b"\x93NUMPY" {
        return Err("bad NPY".into());
    }
    let (base, len) = match b[6] {
        1 => (10, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 => (12, u32::from_le_bytes(b[8..12].try_into()?) as usize),
        _ => return Err("NPY version".into()),
    };
    let h = std::str::from_utf8(b.get(base..base + len).ok_or("NPY header")?)?;
    if !h.contains("'<f4'") || !h.contains("'fortran_order': False") {
        return Err("NPY dtype/layout".into());
    }
    let shape = h
        .split("'shape':")
        .nth(1)
        .and_then(|x| x.split('(').nth(1))
        .and_then(|x| x.split(')').next())
        .ok_or("NPY shape")?
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or("NPY overflow")?;
    if count > 16_000_000 || b.len() != base + len + count * 4 {
        return Err("NPY length".into());
    }
    let v: Vec<_> = b[base + len..]
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite reference".into());
    }
    Ok((shape, v))
}
fn compare(out: &Path, name: &str, a: &[f32], r: &[f32]) -> Result<Value> {
    if a.len() != r.len() || a.iter().any(|x| !x.is_finite()) {
        return Err("native nonfinite/shape mismatch".into());
    }
    let (mut d, mut norm, mut max) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(r) {
        let e = f64::from(x) - f64::from(y);
        d += e * e;
        norm += f64::from(y).powi(2);
        max = max.max(e.abs());
    }
    let rel = d.sqrt() / norm.sqrt().max(1e-30);
    let passed = rel.is_finite() && rel <= 0.001;
    fs::write(
        out.join(format!("{name}.f32")),
        a.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let metric =
        json!({"pass":passed,"relative_l2":rel,"max_absolute":max,"elements":a.len(),"gate":0.001});
    fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&metric)?,
    )?;
    if !passed {
        return Err(format!("{name}: original F32 relative L2 {rel} exceeds .001").into());
    }
    Ok(metric)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err("usage: dit_profile GGUF CPU_REFERENCE NEW_OUTPUT GPU 0|1 FROZEN_NATIVE_V0; explicit absolute NEXTROCM_LIB".into());
    }
    let file = fs::canonicalize(&args[0])?;
    let reference = fs::canonicalize(&args[1])?;
    let output = PathBuf::from(&args[2]);
    let kernel = PackedKernel::from_setting(Some(&args[4]))?;
    let runtime = PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("runtime missing")?);
    if !runtime.is_absolute() || output.exists() {
        return Err("absolute runtime/new output required".into());
    }
    let mut ids = BTreeMap::new();
    for p in [file.clone(), runtime, fs::canonicalize("/proc/self/exe")?] {
        ids.insert(p.clone(), hash(&p)?);
    }
    if ids[&file] != "0f5a11ebbaf3110a16120311454eaac35db13798e0102339e4f2b8fd5be6ba82" {
        return Err("checkpoint differs".into());
    }
    let pp = reference.join("dit-original-f32-provenance.json");
    let ph = hash(&pp)?;
    if ph != "38abf0e797a095d62433c961e749b59483da0233aedc11eaa7b5f674e637b6fb" {
        return Err("primary CPU provenance differs".into());
    }
    ids.insert(pp.clone(), ph);
    let prov: Value = serde_json::from_slice(&fs::read(pp)?)?;
    if prov["status"] != "PASS"
        || prov["stage"] != "dit"
        || prov["primary_oracle"] != true
        || prov["precision"] != "original-f32"
    {
        return Err("not primary original F32 oracle".into());
    }
    let tepath = reference.join("te-original-f32-provenance.json");
    if hash(&tepath)? != "116d588780ba37becf4ce68e582102caae73a742ead7bb910b9f07c6192612d9" {
        return Err("primary TE provenance differs".into());
    }
    ids.insert(tepath.clone(), hash(&tepath)?);
    let te: Value = serde_json::from_slice(&fs::read(tepath)?)?;
    let mut arrays = BTreeMap::new();
    let mut names: Vec<String> = ["embeds", "noise", "sigmas", "timesteps", "block0", "v0"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    names.extend((0..8).map(|i| format!("latents_{i}")));
    for name in &names {
        let p = reference.join(format!("{name}.npy"));
        let sha = hash(&p)?;
        let source = if name == "embeds" { &te } else { &prov };
        if source["outputs"][format!("{name}.npy")]["sha256"] != sha {
            return Err(format!("fixture SHA differs {name}").into());
        }
        ids.insert(p.clone(), sha);
        arrays.insert(name.clone(), npy(&p)?);
    }
    let tokens = arrays["embeds"]
        .0
        .first()
        .copied()
        .ok_or("embedding shape")?;
    let n = 256usize;
    for (name, shape) in [
        ("embeds", vec![tokens, 4096]),
        ("noise", vec![n, 64]),
        ("sigmas", vec![9]),
        ("timesteps", vec![8]),
        ("block0", vec![tokens + n, 4096]),
        ("v0", vec![n, 64]),
    ] {
        if arrays[name].0 != shape {
            return Err(format!("wrong fixture shape {name}").into());
        }
    }
    for i in 0..8 {
        if arrays[&format!("latents_{i}")].0 != [n, 64] {
            return Err("wrong latent shape".into());
        }
    }
    if arrays["sigmas"].1 != sched::turbo_sigmas() {
        return Err("saved official eight sigmas differ".into());
    }
    let frozen = fs::canonicalize(&args[5])?;
    let frozen_hash = hash(&frozen)?;
    if frozen_hash != "71d79bd56449f0fc357f3c09e2c0ba55ea30570225c3ac63f58f6ac6ee91c5eb" {
        return Err("frozen full-trajectory native velocity differs".into());
    }
    ids.insert(frozen.clone(), frozen_hash);
    let frozen_bytes = fs::read(&frozen)?;
    if std::env::var("NS_PROFILE").as_deref() != Ok("gpu") {
        return Err("NS_PROFILE=gpu required before opening GPU".into());
    }
    let gpu = Gpu::open(args[3].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    let sym = api()?.symbol("nr_allocation_stats");
    if sym.is_null() {
        return Err("required stats ABI missing".into());
    }
    // SAFETY: exact ns statistics/private trajectory.h ABI, runtime stays loaded.
    let stats: Stats = unsafe { std::mem::transmute(sym) };
    let memory = || -> Result<(u64, u64)> {
        gpu.sync()?;
        let (mut l, mut p) = (0, 0);
        ffi::check(unsafe { stats(gpu.raw(), &mut l, &mut p) }, "stats")?;
        Ok((l, p))
    };
    if memory()?.0 != 0 {
        return Err("initial live allocation".into());
    }
    fs::create_dir(&output)?;
    fs::write(
        output.join("identities-before.json"),
        serde_json::to_vec_pretty(&ids)?,
    )?;
    let result = (|| -> Result<Value> {
        if std::env::var("NS_PROFILE").as_deref() != Ok("gpu") {
            return Err("NS_PROFILE=gpu required before opening GPU".into());
        }
        let gg = Gguf::open(&file)?;
        let start = Instant::now();
        let dit = Denoiser::load(&gg, &gpu, n, tokens, kernel, 0, 1024 * 1024 * 1024)?;
        let load_seconds = start.elapsed().as_secs_f64();
        let resident = memory()?;
        if resident.0 != dit.plan().resident {
            return Err("resident estimate differs".into());
        }
        let start = Instant::now();
        let prefix = dit.prefix(&arrays["embeds"].1, false)?;
        let prefix_seconds = start.elapsed().as_secs_f64();
        let lat = DevBuf::from_f32(&gpu, &arrays["noise"].1)?;
        let mut runs = Vec::new();
        let mut baseline = None;
        let mut profile = Value::Null;
        for (ordinal, enabled) in [false, true, false].into_iter().enumerate() {
            gpu.sync()?;
            if enabled {
                profiler::begin(&gpu)?;
            }
            let start = Instant::now();
            let velocity = dit.velocity(&prefix, &lat, (16, 16), arrays["timesteps"].1[0], None)?;
            gpu.sync()?;
            let velocity_seconds = start.elapsed().as_secs_f64();
            if enabled {
                profile = profiler::finish()?;
                let operations = profile["operations"]
                    .as_array()
                    .ok_or("profile operation list")?;
                for label in [
                    "block.norm1",
                    "block.norm2",
                    "block.qkv",
                    "block.prefix_copy",
                    "block.rms_q",
                    "block.rms_k",
                    "block.attention",
                    "block.out",
                    "block.gate1",
                    "block.gate2",
                    "block.gate_up",
                    "block.swiglu",
                    "block.down",
                ] {
                    let count: u64 = operations
                        .iter()
                        .filter(|r| r["operation"] == label)
                        .map(|r| r["count"].as_u64().unwrap_or(0))
                        .sum();
                    if count != 32 {
                        return Err(format!("profile coverage {label}: {count} expected32").into());
                    }
                }
                for label in [
                    "aux.timestep1",
                    "aux.timestep2",
                    "aux.modulation",
                    "aux.norm_scale",
                    "aux.img_in_f32",
                    "output.norm",
                    "output.to_float",
                    "aux.proj_out",
                ] {
                    let count: u64 = operations
                        .iter()
                        .filter(|r| r["operation"] == label)
                        .map(|r| r["count"].as_u64().unwrap_or(0))
                        .sum();
                    if count != 1 {
                        return Err(format!("profile coverage {label}: {count} expected1").into());
                    }
                }
                if profile["operation_count"] != 424 || profile["dropped_spans"] != 0 {
                    return Err("profile total/dropped coverage differs".into());
                }
            }
            let actual = velocity.to_f32()?;
            let bytes: Vec<_> = actual.iter().flat_map(|v| v.to_le_bytes()).collect();
            if bytes != frozen_bytes {
                return Err("velocity differs from frozen full-trajectory output".into());
            }
            if let Some(ref previous) = baseline {
                if &bytes != previous {
                    return Err("profile/repeat changed complete velocity bytes".into());
                }
            } else {
                baseline = Some(bytes);
            }
            let metric = compare(
                &output,
                &format!("velocity-{ordinal}"),
                &actual,
                &arrays["v0"].1,
            )?;
            drop(velocity);
            let mem = memory()?;
            if mem.1 > dit.plan().tracked {
                return Err("profile peak exceeds aggregate preflight".into());
            }
            runs.push(json!({"profile":enabled,"velocity_seconds":velocity_seconds,"metric":metric,"allocations":mem}));
        }
        fs::write(
            output.join("operators.json"),
            serde_json::to_vec_pretty(&profile)?,
        )?;
        let plan = json!({"resident":dit.plan().resident,"tracked":dit.plan().tracked,"with_margin":dit.plan().with_margin});
        drop((lat, prefix, dit));
        let dropped = memory()?;
        if dropped.0 != 0 {
            return Err("post-drop leak".into());
        }
        Ok(
            json!({"status":"PASS","scope":"same-binary/runtime profile off/on/off one velocity; no production throughput claim","full_output_bitwise":true,"capture_hooks":false,"packed64":kernel.setting(),"gpu":{"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc},"load_seconds":load_seconds,"prefix_seconds":prefix_seconds,"runs":runs,"plan":plan,"post_drop":dropped,"features":["operator-profile"]}),
        )
    })();
    if result.is_err() {
        profiler::abort();
    }
    for (p, h) in &ids {
        if hash(p)? != *h {
            return Err("identity changed during execution".into());
        }
    }
    fs::write(
        output.join("identities-after.json"),
        serde_json::to_vec_pretty(&ids)?,
    )?;
    let report = match &result {
        Ok(v) => v.clone(),
        Err(e) => {
            json!({"status":"FAIL","error":e.to_string(),"allocations_after_unwind":memory()?})
        }
    };
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{report}");
    result.map(|_| ())
}
