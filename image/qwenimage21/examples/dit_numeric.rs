//! Independent original F32 full DiT gate. Test-only SHA subprocesses; inference
//! is Rust -> HIP. No service/model registration or image-quality admission.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image_qwenimage21::{
    denoiser::Denoiser,
    ffi::{self, PackedKernel},
    sched,
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
    if args.len() != 5 {
        return Err("usage: dit_numeric GGUF CPU_REFERENCE NEW_OUTPUT GPU 0|1; explicit absolute NEXTROCM_LIB".into());
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
    let gpu = Gpu::open(args[3].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    type Axpy = unsafe extern "C" fn(*mut std::ffi::c_void, *mut f32, *const f32, i64, f32) -> i32;
    let sym = api()?.symbol("nr_allocation_stats");
    let ax = api()?.symbol("nrqi_axpy_f32");
    if sym.is_null() || ax.is_null() {
        return Err("required stats/Euler ABI missing".into());
    }
    // SAFETY: exact ns statistics/private trajectory.h ABI, runtime stays loaded.
    let stats: Stats = unsafe { std::mem::transmute(sym) };
    let axpy: Axpy = unsafe { std::mem::transmute(ax) };
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
        let gg = Gguf::open(&file)?;
        let start = Instant::now();
        let dit = Denoiser::load(&gg, &gpu, n, tokens, kernel, 0, 1024 * 1024 * 1024)?;
        let load_seconds = start.elapsed().as_secs_f64();
        let resident = memory()?;
        if resident.0 != dit.plan().resident {
            return Err("resident estimate differs".into());
        }
        let start = Instant::now();
        let prefix = dit.prefix(&arrays["embeds"].1, true)?;
        let prefix_seconds = start.elapsed().as_secs_f64();
        let lat = DevBuf::from_f32(&gpu, &arrays["noise"].1)?;
        let mut steps = Vec::new();
        let mut metrics = serde_json::Map::new();
        for i in 0..8 {
            let start = Instant::now();
            let mut hook = |v: &[f32]| -> nextrocm_engine_core::Result<()> {
                let m = compare(&output, "block0", v, &arrays["block0"].1)
                    .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
                metrics.insert("block0".into(), m);
                Ok(())
            };
            let v = dit.velocity(
                &prefix,
                &lat,
                (16, 16),
                arrays["timesteps"].1[i],
                if i == 0 { Some(&mut hook) } else { None },
            )?;
            let velocity_seconds = start.elapsed().as_secs_f64();
            if i == 0 {
                metrics.insert(
                    "v0".into(),
                    compare(&output, "v0", &v.to_f32()?, &arrays["v0"].1)?,
                );
            }
            let ds = arrays["sigmas"].1[i + 1] - arrays["sigmas"].1[i];
            // SAFETY: complete disjoint n*64 F32 latents/velocity on same GPU;
            // original Euler separate F32 multiply/add, GPU-resident update.
            ffi::check(
                unsafe { axpy(gpu.raw(), lat.fp(), v.fp(), (n * 64) as i64, ds) },
                "Euler update",
            )?;
            let name = format!("latents_{i}");
            metrics.insert(
                name.clone(),
                compare(&output, &name, &lat.to_f32()?, &arrays[&name].1)?,
            );
            drop(v);
            let mem = memory()?;
            if mem.1 > dit.plan().tracked {
                return Err("actual tracked peak exceeds aggregate preflight".into());
            }
            steps.push(json!({"step":i,"recorded_timestep":arrays["timesteps"].1[i],"velocity_seconds":velocity_seconds,"tracked_live":mem.0,"tracked_peak":mem.1}));
            println!("step {i} PASS velocity={velocity_seconds:.6}s");
        }
        let plan = json!({"resident":dit.plan().resident,"tracked":dit.plan().tracked,"with_margin":dit.plan().with_margin});
        drop((lat, prefix, dit));
        let dropped = memory()?;
        if dropped.0 != 0 {
            return Err("post-drop leak".into());
        }
        Ok(
            json!({"status":"PASS","scope":"CPU-embedding full32 DiT eight-step original F32 gates; excludes native TE/VAE/service","packed64":kernel.setting(),"gpu":{"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc},"load_seconds":load_seconds,"prefix_seconds":prefix_seconds,"steps":steps,"metrics":metrics,"plan":plan,"post_drop":dropped}),
        )
    })();
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
