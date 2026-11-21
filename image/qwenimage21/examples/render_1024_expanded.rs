//! Fixed1024 timing/render diagnostic; numerical DiT/VAE gates remain separate.
//! Recorded seed7 noise, not a native RNG claim. SHA subprocesses are test-only.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image_qwenimage21::{
    denoiser::Denoiser,
    ffi::{self, AttentionKernel, PackedKernel},
    sched,
    vae::{Vae,ConvKernel},
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
    let baseline_dir=Path::new("/tmp/qi21-render1024-batched-result-v1");
    let baseline_pins:BTreeMap<String,String>=serde_json::from_str(BASELINE_OUTPUTS)?;
    pin(&mut ids,&baseline_dir.join("report.json"),Some("2488aab2f6c647ed7fa8dd7433297f9952b5f4e4509e17f2962f90a3a8852bca"))?;
    for (name,digest) in &baseline_pins {pin(&mut ids,&baseline_dir.join(name),Some(digest))?;}
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
    ffi::api()?.require_kernel(PackedKernel::ExpandedImage4096)?;
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
            PackedKernel::ExpandedImage4096,
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
        let plan = json!({"attention_scratch":dit.plan().attention_scratch,"packed_scratch":dit.plan().packed_scratch,"batch_queries":256,"resident":dit.plan().resident,"tracked":dit.plan().tracked,"with_margin":dit.plan().with_margin});
        drop(lat);
        drop(prefix);
        drop(dit);
        drop(gg);
        let report = json!({"status":"DIAGNOSTIC_COMPLETE","numeric_gate":"pending original1024 full reference; no PASS claim","shape":[4096,64],"attention":"Batched1024","key_rows":4114,"packed":"ExpandedImage4096","native_TE_input":true,"load_seconds":load,"prefix_seconds":prefix_seconds,"steps":steps,"plan":plan,"before":before,"after_drop":empty_snapshot(&gpu,stats)?});
        write_json(&dd.join("report.json"), &report)?;
        stages.insert("dit".into(), report);
        let vd = out.join("vae");
        fs::create_dir(&vd)?;
        let before = empty_snapshot(&gpu, stats)?;
        let t = Instant::now();
        let vae = Vae::load_with_conv(&vae_file, &gpu, ALLOWANCE as usize, ConvKernel::Register64Measured1024)?;
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
        let report = json!({"status":"DIAGNOSTIC_COMPLETE","numeric_gate":"pending raw/public original1024 reference; unchanged .001 required later","vae_conv":"Register64Measured1024","native_DiT_latents":true,"shape":[1024,1024,4],"load_seconds":load,"decode_seconds":seconds,"resident":resident,"driver_before_decode":free_before,"before":before,"after_drop":empty_snapshot(&gpu,stats)?,"png_sha256":hash(&png)?,"preview":"original conversion from captured raw; decoded PNG bytes exact"});
        write_json(&vd.join("report.json"), &report)?;
        stages.insert("vae".into(), report);
        progress(
            &out,
            "render_complete",
            all.elapsed().as_secs_f64(),
            json!({"decode_seconds":seconds,"png":png}),
        )?;
        let mut comparisons=Vec::new();
        for (name,digest) in &baseline_pins {
            let actual=out.join(name);let expected=baseline_dir.join(name);
            if fs::read(&actual)?!=fs::read(&expected)? {return Err(format!("optimization changed full baseline output {name}").into());}
            comparisons.push(json!({"file":name,"sha256":digest,"bytes":fs::metadata(&actual)?.len(),"bitwise_equal":true}));
        }
        write_json(&out.join("baseline-output-comparison.json"),&json!({"status":"PASS","scope":"optimization equivalence to frozen diagnostic; original numeric FAIL unchanged","files":comparisons}))?;
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
    let report = json!({"status":if complete{"DIAGNOSTIC_COMPLETE"}else{"INCOMPLETE"},"full_baseline_output_bitwise":complete,"numeric_validation_pass":false,"scope":"native1024 fixedprompt/recordedseed7/official8 render and timing; full original1024 .001 references still required; no admission","prompt":PROMPT,"seed_fixture":7,"size":1024,"official_steps":8,"reference_gate_unchanged":0.001,"stages":stages,"initial":initial,"after_drop":cleanup.as_ref().ok(),"cleanup_error":cleanup.as_ref().err().map(ToString::to_string),"error":result.as_ref().err().map(ToString::to_string),"changed_identities":changed,"diagnostic_seconds":elapsed,"timing_scope":"includes first-block capture/stage readbacks/output; excludes initial hashing; not warm service latency","device_allowance":ALLOWANCE,"driver_measurement":"external sampler required; internal snapshots are total/free not process usage"});
    write_json(&out.join("report.json"), &report)?;
    println!("{report}");
    if !complete {
        return Err("1024 diagnostic incomplete; inspect retained artifacts".into());
    }
    Ok(())
}

const BASELINE_OUTPUTS:&str=r#"{"te/stage-00.f32": "cb90a477eba3b118cf2edf9eda9fb3c8fdc185b07104bf2b616ce105214820b7", "te/stage-01.f32": "e49773565656c883e9977117c50798f2f538f93a664978c911136cf3b4e1728b", "te/stage-02.f32": "5898f5d6d8c666da336527e22e4d98f38971155f3679c1de427267b9ed9b665e", "te/stage-03.f32": "9f92d51dafc8291176fd8e514ec5123c5ad5c93343456a97146fe829e37f8774", "te/stage-04.f32": "7a162c11566d0248b4e93a11b53ef4b19525d54c73e248fb8fb0b4208eaf5509", "te/stage-05.f32": "8b5ad56b981ff422066067afe69f193dc259274941dbf37ed34f375e3ef70164", "te/stage-06.f32": "21cb1af09f04c2d590f8ae83944247cec35f148305953f2c39b49e94977de5fd", "te/stage-07.f32": "69ef73d6c61fc232bb603b2f4c6d0ed3443425ac9d42eb99063fc92e9fa7bf91", "te/stage-08.f32": "61a72a5379cf204dd0ac1729cc72b8e7186b034c75c467e7b8a2b3968a5c1da5", "te/stage-09.f32": "428ce64ed5a88670a8d7533565bc06778496ba4b54a9189730a72c5bd98464db", "te/stage-10.f32": "68b31c9103b0d278c9ce438885fbfca6388f9ac9b3c225c7b103024debe99922", "te/stage-11.f32": "88065a71c54f764f688649ad373ed720b8be6aab98e707ed770dbfc63d0b0e8d", "te/stage-12.f32": "2dd5908779eb122c7371bd345e81d9eac70eda1909c65aaf9320f53a4ba924cf", "te/stage-13.f32": "b2181145723b125291c4d61518f3ea7df19658f836d2edaa433e0a9918a959b9", "te/stage-14.f32": "8f999e06f567bcadd9b22f8762da1d7bdab28ab8a495523728db1b96ad3136d8", "te/stage-15.f32": "20489f0a94e28fbd4f22e52fad0bef032bf789b2b77173956221d1dc43221853", "te/stage-16.f32": "7cf73b17dda6f446607a8afc050c215e5608c65328afb2f4072b52f7054c037a", "te/stage-17.f32": "eae20f651be3e207c457039e0b44bca19c13e15e15ae3417c17bad423702a6f1", "te/stage-18.f32": "581991c5f3fd27af01ea649fdd942866838bb493b155f128ea9e6d04b0d9df01", "te/stage-19.f32": "1bbe9b31243029e98f32338ba7b2d620aacbe98954dea7740a0ed2d98f03afd3", "te/stage-20.f32": "1debfe431dbb69cb8984bd2ea2ab5c893bf555fa08eda8dfaf96efb03e9300b5", "te/stage-21.f32": "e675a21920fe0abca13e5c53a3db076b2dfbb5390a315665279193c5db8e5c1f", "te/stage-22.f32": "52c12f74de06733bbc9253bafff473a525b2ba6eb1ea8843d07f15859a2c99a9", "te/stage-23.f32": "2ec2869f863451700852a183723e600b7cb1d11a9e366650148867e4470a0943", "te/stage-24.f32": "7737c871db43e53c62455e635a76f83b03f9ff6c3eae6816dfe27a5f9dc068cc", "te/stage-25.f32": "3f774487e558c4f22e3e879c52ca33513c970d401b7bcc2b728ec5cef3009401", "te/stage-26.f32": "cc3ea09bc539273ed7dcd930ea759ae1bce410472181cd6c017ac2e60fe8ea89", "te/stage-27.f32": "64b4b93853c941a982de07e6f8339b271f8d022abd2361414ca7e980b2d3bc6b", "te/stage-28.f32": "6a00b58fcc7444d51858aae433b0504637b197ef3d82f93c3789ed787d81d369", "te/stage-29.f32": "e4e6dedadbb70bb0b3cc643586690adda28b399becc1d702a913f79f4563015e", "te/stage-30.f32": "24bf99ce535e6c1b12fc540768368dfc9efa890bd5c66ed531cafac70a16bc61", "te/stage-31.f32": "90a60be1a4ded714fcc6c976c3bb633ffcc2a39dca95dc33d5e0383d8069e1f3", "te/stage-32.f32": "2935dd6e4975dc0a194eeb834bd190c92767a87cf8928269b069edee1e54484e", "te/stage-33.f32": "82a2acc8877ebb1a6510701ca742883608c2fc37a507f1108c4ce04d6b7d5899", "te/stage-34.f32": "c1cec5327db4ff77f816711627b15fa70ae8d35bed0becb46550610d7769937b", "te/stage-35.f32": "5f6d4e9e58785b10915ab8a2722896e2ac1fd080ea048853237d3ccd411c11ee", "te/stage-36.f32": "c049ee828367cd81ed78a3e0a7a6a5c4a6599b5cba43acd516dc8d357e606b66", "te/embeds.f32": "1bc016991ab32ec19d4037047d538c4c07e03d3ed7a8a2dfc63371db2896d448", "dit/block0.f32": "f13e74309061e03065cf01c1821ff737509f6b537307dc9d745324693e7109bd", "dit/v0.f32": "f2dcfca32a1b350071d044dac1bb45841b87fe5dc48c3e0630c1283a6cf35e5c", "dit/v1.f32": "b0416a0eb400ffbb6982c6a44f0e88dc1c2e424221e630053017768b6ba8c01f", "dit/v2.f32": "f22f67bb4e6ac9a6aae49b02fb7e887c78fa0c26080759a56a03fa25ec082b59", "dit/v3.f32": "c9da92a5872ef40b1b52bfa1e44f9ed8826fc77694310da6f1f547c2a3c4369e", "dit/v4.f32": "236bbc61ba65e94752615599e7fe8bf4b27c306e8108463525fb414244275eaa", "dit/v5.f32": "86be1f2488595473fe7ad2770989916e77ba9c334b5e2a0cd4197649aadd6c71", "dit/v6.f32": "5f460e380b816e84034b91a3e8f5e73fe4c2c03f76efd020da962823bb99e8a9", "dit/v7.f32": "2671a51988852d6cf2d4a3c022f2215aad0c5de41a4915d573a4a12ca48d1b44", "dit/latents_0.f32": "447fc74794b9b421cc1f96d42db525177d5ceb371e5e651befd2cf442590168f", "dit/latents_1.f32": "60dc72d7c23e2dc9611e27d2d82064a94e580a8b30e176298f5b647d7bfe4fd3", "dit/latents_2.f32": "9cd147b7e6b423aca8a67a2c48e04ac750e7daa20b044a86bfa92151f7db02c5", "dit/latents_3.f32": "4dfe96d6c5c50461cc2a6daaf59118bdc88b1b4bfd697dc0c7dac4098fcbf259", "dit/latents_4.f32": "157f50139c6f7722078feb57f13a47900e2d1b877ab7a81325a392d68f4c427f", "dit/latents_5.f32": "6625a53b224570b6f03bfb1825bfb331d5aab4bc4768555b3e52adf701b0d0e4", "dit/latents_6.f32": "aafe044da5231ca19ca079fa81403d9e22db5925796f812f3f81f2b1717e557e", "dit/latents_7.f32": "47389c4ede78839fcd1de509860f6aafbe3acc374b76cbdfcab440fa3b3f5c15", "vae/image_raw.f32": "cde2c1033932c82c898e414b4a8c186f3eaf312133f402bc0815f2a6c9e243e1", "vae/image_public.f32": "232427256d95ffc8b14d3060ab9f9994dd2440a267d6e9a5a24a0210b79d925a", "vae/image.rgba8": "2824e0b10a30fbd13672b84cc377c8d5db0154cbd05866b65a240900a2d4925e", "vae/image.png": "64b927e9ec68b46b38bf8fe4b51f3fc2e01cce9e51f9e3651b5ee42fc671be1a"}"#;
