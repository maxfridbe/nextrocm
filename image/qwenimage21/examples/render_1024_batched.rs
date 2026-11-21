//! Fixed1024 timing/render diagnostic; numerical DiT/VAE gates remain separate.
//! Recorded seed7 noise, not a native RNG claim. SHA subprocesses are test-only.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image_qwenimage21::{
    denoiser::Denoiser,
    ffi::{self, AttentionKernel, PackedKernel},
    sched,
    vae::Vae,
};
use nextrocm_vision_qwen3vl::{Budget, LoadOptions, Plan, TextEncoder};
use serde_json::{json, Value};
use std::io::Write;
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
const PROMPT: &str = "A red ceramic teapot on a wooden table.";
const ALLOWANCE: u64 = 7_500_000_000;
const MARGIN: u64 = 1024 * 1024 * 1024;
type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
type Axpy = unsafe extern "C" fn(*mut std::ffi::c_void, *mut f32, *const f32, i64, f32) -> i32;
fn pin(ids: &mut BTreeMap<PathBuf, String>, p: &Path, expected: Option<&str>) -> Result<()> {
    let h = hash(p)?;
    if expected.is_some_and(|e| e != h) {
        return Err(format!("identity differs: {}", p.display()).into());
    }
    ids.insert(p.to_owned(), h);
    Ok(())
}
fn json_file(p: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(p)?)?)
}
fn write_json(p: &Path, v: &Value) -> Result<()> {
    fs::write(p, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}
fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn allocations(gpu: &Gpu, stats: Stats) -> Result<(u64, u64)> {
    gpu.sync()?;
    let (mut live, mut peak) = (0, 0);
    // SAFETY: exact allocation-stats ABI, live context and writable local counters.
    ffi::check(
        unsafe { stats(gpu.raw(), &mut live, &mut peak) },
        "allocation statistics",
    )?;
    if peak >= ALLOWANCE || live > peak {
        return Err("cumulative device allocation allowance exceeded".into());
    }
    Ok((live, peak))
}
fn empty_snapshot(gpu: &Gpu, stats: Stats) -> Result<Value> {
    let a = allocations(gpu, stats)?;
    if a.0 != 0 {
        return Err("stage boundary has live device allocations".into());
    }
    Ok(json!({"tracked_live":a.0,"cumulative_tracked_peak":a.1,"driver_total_free":gpu.memory()?}))
}
fn progress(out: &Path, phase: &str, elapsed: f64, details: Value) -> Result<()> {
    write_json(
        &out.join("progress.json"),
        &json!({"status":"RUNNING","phase":phase,"diagnostic_seconds":elapsed,"details":details}),
    )?;
    println!("{phase}: {details}");
    std::io::stdout().flush()?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 7 {
        return Err("usage: render_1024_diagnostic ASSET_DIR OFFICIAL_CONFIG PRIMARY256_TE_REFERENCE INPUT1024_DIR INPUT_MANIFEST_SHA NEW_OUTPUT GPU; absolute NEXTROCM_LIB".into());
    }
    let assets = fs::canonicalize(&a[0])?;
    let config = fs::canonicalize(&a[1])?;
    let refs = fs::canonicalize(&a[2])?;
    let inputs = fs::canonicalize(&a[3])?;
    let out = PathBuf::from(&a[5]);
    let runtime = PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("runtime required")?);
    if !runtime.is_absolute()
        || out.exists()
        || a[4].len() != 64
        || !a[4].bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("runtime/new output/fixture pin invalid".into());
    }
    let te_file = assets.join("qwen3vl_8b_int8_convrot.safetensors");
    let dit_file = assets.join("qwen_image_2.1_turbo_Q4_K_M.gguf");
    let vae_file = assets.join("qwen_image_2.1_vae_bf16.safetensors");
    let mut ids = BTreeMap::new();
    for (p, h) in [
        (
            &te_file,
            "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f",
        ),
        (
            &dit_file,
            "0f5a11ebbaf3110a16120311454eaac35db13798e0102339e4f2b8fd5be6ba82",
        ),
        (
            &vae_file,
            "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9",
        ),
    ] {
        pin(&mut ids, p, Some(h))?;
    }
    pin(&mut ids, &runtime, None)?;
    pin(&mut ids, &fs::canonicalize("/proc/self/exe")?, None)?;
    let pp = refs.join("te-original-f32-provenance.json");
    pin(
        &mut ids,
        &pp,
        Some("116d588780ba37becf4ce68e582102caae73a742ead7bb910b9f07c6192612d9"),
    )?;
    let prov = json_file(&pp)?;
    if prov["status"] != "PASS"
        || prov["primary_oracle"] != true
        || prov["precision"] != "original-f32"
    {
        return Err("TE oracle contract".into());
    }
    let tokenizer = config.join("processor/tokenizer.json");
    let te_config = config.join("text_encoder/config.json");
    for (p, key) in [
        (&tokenizer, "processor/tokenizer.json"),
        (&te_config, "text_encoder/config.json"),
    ] {
        pin(
            &mut ids,
            p,
            Some(
                prov["configs"][key]["sha256"]
                    .as_str()
                    .ok_or("config pin")?,
            ),
        )?;
    }
    let mut te_arrays = BTreeMap::new();
    for (name, shape) in [("te_layer0", vec![32, 4096]), ("embeds", vec![18, 4096])] {
        let filename = format!("{name}.npy");
        let p = refs.join(&filename);
        pin(
            &mut ids,
            &p,
            Some(
                prov["outputs"][&filename]["sha256"]
                    .as_str()
                    .ok_or("TE array pin")?,
            ),
        )?;
        let (s, v) = npy(&p)?;
        if s != shape {
            return Err("TE shape".into());
        }
        te_arrays.insert(name, v);
    }
    let tp = refs.join("tokens.json");
    pin(
        &mut ids,
        &tp,
        Some(
            prov["outputs"]["tokens.json"]["sha256"]
                .as_str()
                .ok_or("token pin")?,
        ),
    )?;
    let tokens = json_file(&tp)?;
    let mp = inputs.join("manifest.json");
    pin(&mut ids, &mp, Some(&a[4]))?;
    let manifest = json_file(&mp)?;
    if manifest["status"] != "PASS"
        || manifest["size"] != 1024
        || manifest["seed"] != 7
        || manifest["steps"] != 8
        || manifest["original256_noise_bitwise"] != true
    {
        return Err("1024 input fixture contract".into());
    }
    let mut arrays = BTreeMap::new();
    for (name, shape) in [
        ("noise", vec![4096, 64]),
        ("sigmas", vec![9]),
        ("timesteps", vec![8]),
    ] {
        let filename = format!("{name}.npy");
        let p = inputs.join(&filename);
        if manifest["files"][name]["file"] != filename
            || manifest["files"][name]["shape"] != json!(shape)
        {
            return Err("1024 array manifest shape".into());
        }
        pin(
            &mut ids,
            &p,
            Some(
                manifest["files"][name]["sha256"]
                    .as_str()
                    .ok_or("1024 array pin")?,
            ),
        )?;
        let (s, v) = npy(&p)?;
        if s != shape {
            return Err("1024 array shape".into());
        }
        arrays.insert(name, v);
    }
    if arrays["sigmas"] != sched::turbo_sigmas() {
        return Err("official8 sigmas differ".into());
    }
    let vae_config = config.join("vae/config.json");
    pin(
        &mut ids,
        &vae_config,
        Some(
            prov["configs"]["vae/config.json"]["sha256"]
                .as_str()
                .ok_or("VAE config pin")?,
        ),
    )?;
    let vc = json_file(&vae_config)?;
    for name in ["config.json", "vae_config.json"] {
        let p = assets.join(name);
        if p.exists() {
            pin(&mut ids, &p, None)?;
        }
    }
    fs::create_dir(&out)?;
    write_json(&out.join("identities-before.json"), &json!(ids))?;
    let gpu = Gpu::open(a[6].parse()?)?;
    let s = api()?.symbol("nr_allocation_stats");
    let e = api()?.symbol("nrqi_axpy_f32");
    if s.is_null() || e.is_null() {
        return Err("stats/Euler ABI missing".into());
    }
    // SAFETY: exact reviewed private ABIs, live loaded runtime for the process.
    let stats: Stats = unsafe { std::mem::transmute(s) };
    let axpy: Axpy = unsafe { std::mem::transmute(e) };
    // Explicit1024 ABI check before any stage uploads. No oversized shared fallback.
    ffi::api()?.require_attention(AttentionKernel::Batched1024)?;
    ffi::api()?.require_kernel(PackedKernel::Register64)?;
    let initial = empty_snapshot(&gpu, stats)?;
    write_json(
        &out.join("gpu.json"),
        &json!({"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc}),
    )?;
    let all = Instant::now();
    let mut stages = serde_json::Map::new();
    let result = (|| -> Result<()> {
        let td = out.join("te");
        fs::create_dir(&td)?;
        let before = empty_snapshot(&gpu, stats)?;
        let t = Instant::now();
        let engine = TextEncoder::load_with_order32(
            &te_file,
            &tokenizer,
            &te_config,
            gpu.clone(),
            LoadOptions {
                probability_first: true,
                ordered_headnorm: true,
                ordered_rms: true,
                score_division: true,
            },
            true,
            true,
            true,
            true,
        )?;
        let load = t.elapsed().as_secs_f64();
        progress(
            &out,
            "te_loaded",
            all.elapsed().as_secs_f64(),
            json!({"load_seconds":load}),
        )?;
        let (tids, drop_prefix) = engine.tokenize(PROMPT)?;
        if tids.len() != 32
            || drop_prefix != 14
            || json!(tids) != tokens["ids"]
            || json!(drop_prefix) != tokens["drop"]
        {
            return Err("fixed prompt tokenization differs".into());
        }
        let budget = Budget {
            allowance_bytes: ALLOWANCE as usize,
            driver_margin_bytes: MARGIN as usize,
        };
        let plan = Plan::new(32, &budget)?;
        let mut count = 0;
        let mut first = Value::Null;
        let t = Instant::now();
        let encoded = engine.encode(PROMPT, &budget, &mut |stage, x| {
            let v = x.to_f32()?;
            let mem =
                allocations(&gpu, stats).map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            if stage != count
                || v.len() != 32 * 4096
                || v.iter().any(|x| !x.is_finite())
                || mem.1 > plan.tracked_bytes as u64
            {
                return Err(nextrocm_engine_core::Error(
                    "TE shape/finite/sequence/plan".into(),
                ));
            }
            fs::write(
                td.join(format!("stage-{stage:02}.f32")),
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            )
            .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            if stage == 1 {
                first = compare(&td, "layer0", &v, &te_arrays["te_layer0"])
                    .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            }
            count += 1;
            Ok(())
        })?;
        gpu.sync()?;
        let encode = t.elapsed().as_secs_f64();
        let embeddings = encoded.embeddings.to_f32()?;
        if count != 37
            || encoded.ids != tids
            || encoded.drop != 14
            || encoded.image_pad.len() != 18
            || encoded.image_pad.iter().any(|v| *v)
            || allocations(&gpu, stats)?.0 != 18 * 4096 * 4
        {
            return Err("TE metadata/residency".into());
        }
        let m = compare(&td, "embeds", &embeddings, &te_arrays["embeds"])?;
        drop(encoded);
        drop(engine);
        let report = json!({"status":"PASS","controls":[true,true,true,true,true,true,true,true],"effective_attention":"order32","layer0":first,"embeds":m,"load_seconds":load,"encode_seconds":encode,"before":before,"after_drop":empty_snapshot(&gpu,stats)?});
        write_json(&td.join("report.json"), &report)?;
        stages.insert("te".into(), report);
        progress(
            &out,
            "te_complete",
            all.elapsed().as_secs_f64(),
            json!({"encode_seconds":encode,"load_seconds":load}),
        )?;
        let dd = out.join("dit");
        fs::create_dir(&dd)?;
        let before = empty_snapshot(&gpu, stats)?;
        let t = Instant::now();
        let gg = Gguf::open(&dit_file)?;
        let dit = Denoiser::load_with_attention(
            &gg,
            &gpu,
            4096,
            18,
            PackedKernel::Register64,
            0,
            MARGIN,
            AttentionKernel::Batched1024,
        )?;
        gpu.sync()?;
        let load = t.elapsed().as_secs_f64();
        if allocations(&gpu, stats)?.0 != dit.plan().resident {
            return Err("DiT resident estimate differs".into());
        }
        progress(
            &out,
            "dit_loaded",
            all.elapsed().as_secs_f64(),
            json!({"load_seconds":load,"tracked_plan":dit.plan().tracked,"with_margin":dit.plan().with_margin}),
        )?;
        let t = Instant::now();
        let prefix = dit.prefix(&embeddings, true)?;
        gpu.sync()?;
        let prefix_seconds = t.elapsed().as_secs_f64();
        drop(embeddings);
        progress(
            &out,
            "prefix_ready",
            all.elapsed().as_secs_f64(),
            json!({"prefix_seconds":prefix_seconds}),
        )?;
        let lat = DevBuf::from_f32(&gpu, &arrays["noise"])?;
        let mut steps = Vec::new();
        for i in 0..8 {
            let t = Instant::now();
            let mut hook = |v: &[f32]| -> nextrocm_engine_core::Result<()> {
                if v.len() != 4114 * 4096 || v.iter().any(|x| !x.is_finite()) {
                    return Err(nextrocm_engine_core::Error(
                        "1024 first block shape/finite".into(),
                    ));
                }
                fs::write(
                    dd.join("block0.f32"),
                    v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
                )
                .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
                Ok(())
            };
            let velocity = dit.velocity(
                &prefix,
                &lat,
                (64, 64),
                arrays["timesteps"][i],
                if i == 0 { Some(&mut hook) } else { None },
            )?;
            gpu.sync()?;
            let seconds = t.elapsed().as_secs_f64();
            // Flush immediately after the synchronized full-resolution velocity.
            progress(
                &out,
                &format!("velocity_{i}_complete"),
                all.elapsed().as_secs_f64(),
                json!({"velocity_seconds":seconds,"step":i}),
            )?;
            let v = velocity.to_f32()?;
            if v.len() != 4096 * 64 || v.iter().any(|x| !x.is_finite()) {
                return Err("1024 velocity finite/shape".into());
            }
            fs::write(
                dd.join(format!("v{i}.f32")),
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            // SAFETY: disjoint complete same-context F32 tensors; original separate Euler arithmetic.
            ffi::check(
                unsafe {
                    axpy(
                        gpu.raw(),
                        lat.fp(),
                        velocity.fp(),
                        (4096 * 64) as i64,
                        arrays["sigmas"][i + 1] - arrays["sigmas"][i],
                    )
                },
                "Euler1024",
            )?;
            let l = lat.to_f32()?;
            if l.len() != 4096 * 64 || l.iter().any(|x| !x.is_finite()) {
                return Err("1024 latent finite/shape".into());
            }
            fs::write(
                dd.join(format!("latents_{i}.f32")),
                l.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            drop(velocity);
            let mem = allocations(&gpu, stats)?;
            if mem.1 > dit.plan().tracked {
                return Err("1024 peak exceeds aggregate plan".into());
            }
            steps.push(json!({"step":i,"velocity_seconds":seconds,"recorded_timestep":arrays["timesteps"][i],"tracked_live":mem.0,"cumulative_peak":mem.1}));
            write_json(&dd.join("steps.json"), &json!(steps))?;
        }
        let final_latents = lat.to_f32()?;
        let plan = json!({"attention_scratch":dit.plan().attention_scratch,"batch_queries":256,"resident":dit.plan().resident,"tracked":dit.plan().tracked,"with_margin":dit.plan().with_margin});
        drop(lat);
        drop(prefix);
        drop(dit);
        drop(gg);
        let report = json!({"status":"DIAGNOSTIC_COMPLETE","numeric_gate":"pending original1024 full reference; no PASS claim","shape":[4096,64],"attention":"Batched1024","key_rows":4114,"packed":"Register64","native_TE_input":true,"load_seconds":load,"prefix_seconds":prefix_seconds,"steps":steps,"plan":plan,"before":before,"after_drop":empty_snapshot(&gpu,stats)?});
        write_json(&dd.join("report.json"), &report)?;
        stages.insert("dit".into(), report);
        let vd = out.join("vae");
        fs::create_dir(&vd)?;
        let before = empty_snapshot(&gpu, stats)?;
        let t = Instant::now();
        let vae = Vae::load(&vae_file, &gpu, ALLOWANCE as usize)?;
        gpu.sync()?;
        let load = t.elapsed().as_secs_f64();
        for (name, actual) in [("latents_mean", &vae.mean), ("latents_std", &vae.std)] {
            let expected: Vec<f32> = vc[name]
                .as_array()
                .ok_or("original stats missing")?
                .iter()
                .map(|v| v.as_f64().map(|v| v as f32).ok_or("invalid stats"))
                .collect::<std::result::Result<_, _>>()?;
            if !bits_equal(actual, &expected) {
                return Err("VAE stats differ".into());
            }
        }
        let free_before = gpu.memory()?;
        progress(
            &out,
            "vae_loaded",
            all.elapsed().as_secs_f64(),
            json!({"load_seconds":load,"driver_total_free":free_before}),
        )?;
        let t = Instant::now();
        let (raw_image, w, h) = vae.decode_f32(&final_latents, 64, 64)?;
        gpu.sync()?;
        let seconds = t.elapsed().as_secs_f64();
        if (w, h) != (1024, 1024)
            || raw_image.len() != 1024 * 1024 * 4
            || raw_image.iter().any(|x| !x.is_finite())
        {
            return Err("VAE1024 shape/finite".into());
        }
        fs::write(
            vd.join("image_raw.f32"),
            raw_image
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let public: Vec<_> = raw_image.iter().map(|x| x.clamp(-1., 1.)).collect();
        fs::write(
            vd.join("image_public.f32"),
            public
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let rgba: Vec<u8> = public
            .iter()
            .map(|x| {
                let scaled = *x * 0.5f32;
                let shifted = scaled + 0.5f32;
                (shifted * 255.0f32).round().clamp(0., 255.) as u8
            })
            .collect();
        let png = vd.join("image.png");
        image::save_buffer_with_format(
            &png,
            &rgba,
            1024,
            1024,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
        if image::open(&png)?.to_rgba8().as_raw() != &rgba {
            return Err("PNG roundtrip".into());
        }
        fs::write(vd.join("image.rgba8"), &rgba)?;
        let resident = vae.resident_bytes;
        drop(vae);
        let report = json!({"status":"DIAGNOSTIC_COMPLETE","numeric_gate":"pending raw/public original1024 reference; unchanged .001 required later","native_DiT_latents":true,"shape":[1024,1024,4],"load_seconds":load,"decode_seconds":seconds,"resident":resident,"driver_before_decode":free_before,"before":before,"after_drop":empty_snapshot(&gpu,stats)?,"png_sha256":hash(&png)?,"preview":"original conversion from captured raw; decoded PNG bytes exact"});
        write_json(&vd.join("report.json"), &report)?;
        stages.insert("vae".into(), report);
        progress(
            &out,
            "render_complete",
            all.elapsed().as_secs_f64(),
            json!({"decode_seconds":seconds,"png":png}),
        )?;
        Ok(())
    })();
    let elapsed = all.elapsed().as_secs_f64();
    let mut after = BTreeMap::new();
    let mut changed = Vec::new();
    for (p, h) in &ids {
        match hash(p) {
            Ok(got) => {
                if &got != h {
                    changed.push(p.display().to_string());
                }
                after.insert(p.clone(), got);
            }
            Err(e) => changed.push(e.to_string()),
        }
    }
    write_json(&out.join("identities-after.json"), &json!(after))?;
    let cleanup = empty_snapshot(&gpu, stats);
    let complete = result.is_ok() && changed.is_empty() && cleanup.is_ok();
    let report = json!({"status":if complete{"DIAGNOSTIC_COMPLETE"}else{"INCOMPLETE"},"numeric_validation_pass":false,"scope":"native1024 fixedprompt/recordedseed7/official8 render and timing; full original1024 .001 references still required; no admission","prompt":PROMPT,"seed_fixture":7,"size":1024,"official_steps":8,"reference_gate_unchanged":0.001,"stages":stages,"initial":initial,"after_drop":cleanup.as_ref().ok(),"cleanup_error":cleanup.as_ref().err().map(ToString::to_string),"error":result.as_ref().err().map(ToString::to_string),"changed_identities":changed,"diagnostic_seconds":elapsed,"timing_scope":"includes first-block capture/stage readbacks/output; excludes initial hashing; not warm service latency","device_allowance":ALLOWANCE,"driver_measurement":"external sampler required; internal snapshots are total/free not process usage"});
    write_json(&out.join("report.json"), &report)?;
    println!("{report}");
    if !complete {
        return Err("1024 diagnostic incomplete; inspect retained artifacts".into());
    }
    Ok(())
}
