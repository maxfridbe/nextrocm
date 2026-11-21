//! Fixed256 prompt -> native TE -> eight-step DiT -> native VAE diagnostic.
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
fn raw(p: &Path, n: usize) -> Result<Vec<f32>> {
    let b = fs::read(p)?;
    if b.len() != n * 4 {
        return Err("raw fixture length".into());
    }
    let x: Vec<_> = b
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if x.iter().any(|x| !x.is_finite()) {
        return Err("raw fixture nonfinite".into());
    }
    Ok(x)
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
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if a.len() != 6 {
        return Err("usage: end_to_end_numeric ASSET_DIR OFFICIAL_CONFIG CPU_REFERENCE VAE_BOUNDARY_FIXTURE NEW_OUTPUT GPU; absolute NEXTROCM_LIB required".into());
    }
    let assets = fs::canonicalize(&a[0])?;
    let config = fs::canonicalize(&a[1])?;
    let refs = fs::canonicalize(&a[2])?;
    let vf = fs::canonicalize(&a[3])?;
    let out = PathBuf::from(&a[4]);
    let runtime =
        PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("absolute NEXTROCM_LIB required")?);
    if !runtime.is_absolute() || out.exists() {
        return Err("runtime must be absolute; output must be new".into());
    }
    let te_file = assets.join("qwen3vl_8b_int8_convrot.safetensors");
    let dit_file = assets.join("qwen_image_2.1_turbo_Q4_K_M.gguf");
    let vae_file = assets.join("qwen_image_2.1_vae_bf16.safetensors");
    let tokenizer = config.join("processor/tokenizer.json");
    let te_config = config.join("text_encoder/config.json");
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
    let mut provenances = BTreeMap::new();
    for (stage, h) in [
        (
            "te",
            "116d588780ba37becf4ce68e582102caae73a742ead7bb910b9f07c6192612d9",
        ),
        (
            "dit",
            "38abf0e797a095d62433c961e749b59483da0233aedc11eaa7b5f674e637b6fb",
        ),
        (
            "vae",
            "b379806eb756878965397154baf7c63a9c809e7cb6b266e5bb9a0667526b5614",
        ),
    ] {
        let p = refs.join(format!("{stage}-original-f32-provenance.json"));
        pin(&mut ids, &p, Some(h))?;
        let v = json_file(&p)?;
        if v["status"] != "PASS"
            || v["stage"] != stage
            || v["precision"] != "original-f32"
            || v["primary_oracle"] != true
        {
            return Err("primary provenance contract".into());
        }
        provenances.insert(stage, v);
    }
    for (path, key) in [
        (&tokenizer, "processor/tokenizer.json"),
        (&te_config, "text_encoder/config.json"),
    ] {
        pin(
            &mut ids,
            path,
            Some(
                provenances["te"]["configs"][key]["sha256"]
                    .as_str()
                    .ok_or("missing config pin")?,
            ),
        )?;
    }
    // Pin any adjacent VAE statistics files the unchanged loader may read. Compare loaded
    // statistics with original config below, so fallback/precedence cannot silently differ.
    for name in ["config.json", "vae_config.json"] {
        let p = assets.join(name);
        if p.exists() {
            pin(&mut ids, &p, None)?;
        }
    }
    let vc = config.join("vae/config.json");
    pin(
        &mut ids,
        &vc,
        Some(
            provenances["vae"]["configs"]["vae/config.json"]["sha256"]
                .as_str()
                .ok_or("missing VAE config pin")?,
        ),
    )?;
    let original_vae_config = json_file(&vc)?;
    let mut arrays = BTreeMap::new();
    let mut names: Vec<(String, &str, Vec<usize>)> = vec![
        ("te_layer0".into(), "te", vec![32, 4096]),
        ("embeds".into(), "te", vec![18, 4096]),
        ("noise".into(), "dit", vec![256, 64]),
        ("sigmas".into(), "dit", vec![9]),
        ("timesteps".into(), "dit", vec![8]),
        ("block0".into(), "dit", vec![274, 4096]),
        ("v0".into(), "dit", vec![256, 64]),
        ("image".into(), "vae", vec![256, 256, 4]),
    ];
    names.extend((0..8).map(|i| (format!("latents_{i}"), "dit", vec![256, 64])));
    for (name, stage, shape) in names {
        let filename = format!("{name}.npy");
        let p = refs.join(&filename);
        pin(
            &mut ids,
            &p,
            Some(
                provenances[stage]["outputs"][&filename]["sha256"]
                    .as_str()
                    .ok_or("missing array pin")?,
            ),
        )?;
        let (s, v) = npy(&p)?;
        if s != shape {
            return Err(format!("shape differs {name}").into());
        }
        arrays.insert(name, v);
    }
    if arrays["sigmas"] != sched::turbo_sigmas() {
        return Err("official eight-step schedule differs".into());
    }
    let tp = refs.join("tokens.json");
    pin(
        &mut ids,
        &tp,
        Some(
            provenances["te"]["outputs"]["tokens.json"]["sha256"]
                .as_str()
                .ok_or("missing token pin")?,
        ),
    )?;
    let tokens = json_file(&tp)?;
    let mp = vf.join("manifest.json");
    pin(
        &mut ids,
        &mp,
        Some("c314c66e73fcb66a050d5bc09327ed07b7f09fe55c81ace70d03b6eb05c877ab"),
    )?;
    let vm = json_file(&mp)?;
    if vm["schema"] != 2
        || vm["height"] != 256
        || vm["width"] != 256
        || vm["primary_oracle"] != true
        || vm["cpu_raw_clamp_bitwise_public"] != true
    {
        return Err("VAE boundary contract differs".into());
    }
    for (file, key) in [
        ("original-provenance.json", "original_provenance_sha256"),
        ("raw-trace-manifest.json", "raw_trace_manifest_sha256"),
    ] {
        pin(
            &mut ids,
            &vf.join(file),
            Some(vm[key].as_str().ok_or("VAE source pin")?),
        )?;
    }
    let mut vae_refs = BTreeMap::new();
    for (name, n, shape) in [
        ("latents", 256 * 64, vec![256, 64]),
        ("expected", 256 * 256 * 4, vec![256, 256, 4]),
        ("expected_raw", 256 * 256 * 4, vec![256, 256, 4]),
    ] {
        let filename = format!("{name}.f32");
        let p = vf.join(&filename);
        if vm["files"][name]["file"] != filename || vm["files"][name]["shape"] != json!(shape) {
            return Err("VAE fixture shape".into());
        }
        pin(
            &mut ids,
            &p,
            Some(
                vm["files"][name]["sha256"]
                    .as_str()
                    .ok_or("VAE array pin")?,
            ),
        )?;
        vae_refs.insert(name, raw(&p, n)?);
    }
    if !bits_equal(&vae_refs["latents"], &arrays["latents_7"])
        || !bits_equal(&vae_refs["expected"], &arrays["image"])
        || !vae_refs["expected_raw"]
            .iter()
            .zip(&vae_refs["expected"])
            .all(|(a, b)| a.clamp(-1., 1.).to_bits() == b.to_bits())
    {
        return Err("DiT/VAE original fixture chain differs".into());
    }
    fs::create_dir(&out)?;
    write_json(&out.join("identities-before.json"), &json!(ids))?;
    let gpu = Gpu::open(a[5].parse()?)?;
    let stat_sym = api()?.symbol("nr_allocation_stats");
    let ax_sym = api()?.symbol("nrqi_axpy_f32");
    if stat_sym.is_null() || ax_sym.is_null() {
        return Err("required allocation/Euler ABI absent".into());
    }
    // SAFETY: exact reviewed C ABIs, loaded runtime remains alive through the process.
    let stats: Stats = unsafe { std::mem::transmute(stat_sym) };
    let axpy: Axpy = unsafe { std::mem::transmute(ax_sym) };
    write_json(
        &out.join("gpu.json"),
        &json!({"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc}),
    )?;
    let mut stages = serde_json::Map::new();
    let initial = empty_snapshot(&gpu, stats)?;
    let start_all = Instant::now();
    let result = (|| -> Result<()> {
        let te_out = out.join("te");
        fs::create_dir(&te_out)?;
        let before = empty_snapshot(&gpu, stats)?;
        let start = Instant::now();
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
        let load_seconds = start.elapsed().as_secs_f64();
        if allocations(&gpu, stats)?.0 != 0 {
            return Err("TE metadata load allocated device weights".into());
        }
        let (token_ids, drop_prefix) = engine.tokenize(PROMPT)?;
        if token_ids.len() != 32
            || drop_prefix != 14
            || json!(token_ids) != tokens["ids"]
            || json!(drop_prefix) != tokens["drop"]
        {
            return Err("native tokenization differs".into());
        }
        let budget = Budget {
            allowance_bytes: ALLOWANCE as usize,
            driver_margin_bytes: MARGIN as usize,
        };
        let plan = Plan::new(32, &budget)?;
        let mut captured = 0usize;
        let mut layer0_metric = Value::Null;
        let mut stage_memory = Vec::new();
        let start = Instant::now();
        let encoded = engine.encode(PROMPT, &budget, &mut |stage, x| {
            let f = || -> Result<(Vec<f32>, (u64, u64))> {
                Ok((x.to_f32()?, allocations(&gpu, stats)?))
            };
            let (v, m) = f().map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            if stage != captured
                || v.len() != 32 * 4096
                || v.iter().any(|x| !x.is_finite())
                || m.1 > plan.tracked_bytes as u64
            {
                return Err(nextrocm_engine_core::Error(
                    "TE stage sequence/shape/finite/budget".into(),
                ));
            }
            fs::write(
                te_out.join(format!("stage-{stage:02}.f32")),
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            )
            .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            if stage == 1 {
                layer0_metric = compare(&te_out, "layer0", &v, &arrays["te_layer0"])
                    .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
            }
            stage_memory.push(json!({"stage":stage,"tracked_live":m.0,"cumulative_peak":m.1}));
            captured += 1;
            Ok(())
        })?;
        gpu.sync()?;
        let encode_seconds = start.elapsed().as_secs_f64();
        let embeddings = encoded.embeddings.to_f32()?;
        if captured != 37
            || encoded.ids != token_ids
            || encoded.drop != 14
            || encoded.image_pad.len() != 18
            || encoded.image_pad.iter().any(|v| *v)
            || allocations(&gpu, stats)?.0 != 18 * 4096 * 4
        {
            return Err("TE output metadata/residency differs".into());
        }
        let embed_metric = compare(&te_out, "embeds", &embeddings, &arrays["embeds"])?;
        drop(encoded);
        drop(engine);
        let te_report = json!({"status":"PASS","controls":[true,true,true,true,true,true,true,true],"effective_attention":"order32","effective_silu":"source","ids":token_ids,"drop":14,"shape":[18,4096],"captured_stages":captured,"load_seconds":load_seconds,"diagnostic_encode_seconds":encode_seconds,"layer0":layer0_metric,"embeds":embed_metric,"stage_memory":stage_memory,"planned_tracked":plan.tracked_bytes,"before":before,"after_drop":empty_snapshot(&gpu,stats)?});
        write_json(&te_out.join("report.json"), &te_report)?;
        stages.insert("te".into(), te_report);
        println!("TE PASS, native embeddings feed DiT");
        let dit_out = out.join("dit");
        fs::create_dir(&dit_out)?;
        let before = empty_snapshot(&gpu, stats)?;
        let start = Instant::now();
        let gg = Gguf::open(&dit_file)?;
        let dit = Denoiser::load_with_attention(
            &gg,
            &gpu,
            256,
            18,
            PackedKernel::Register64,
            0,
            MARGIN,
            AttentionKernel::Tiled,
        )?;
        gpu.sync()?;
        let load_seconds = start.elapsed().as_secs_f64();
        if allocations(&gpu, stats)?.0 != dit.plan().resident {
            return Err("DiT resident estimate differs".into());
        }
        let start = Instant::now();
        // The full joint first-block reference includes prefix outputs. Capture only
        // this diagnostic prefix once; all subsequent layers remain device resident.
        let prefix = dit.prefix(&embeddings, true)?;
        gpu.sync()?;
        let prefix_seconds = start.elapsed().as_secs_f64();
        drop(embeddings);
        let lat = DevBuf::from_f32(&gpu, &arrays["noise"])?;
        let mut steps = Vec::new();
        let mut metrics = serde_json::Map::new();
        for i in 0..8 {
            let start = Instant::now();
            let mut hook = |v: &[f32]| -> nextrocm_engine_core::Result<()> {
                let m = compare(&dit_out, "block0", v, &arrays["block0"])
                    .map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
                metrics.insert("block0".into(), m);
                Ok(())
            };
            let velocity = dit.velocity(
                &prefix,
                &lat,
                (16, 16),
                arrays["timesteps"][i],
                if i == 0 { Some(&mut hook) } else { None },
            )?;
            gpu.sync()?;
            let velocity_seconds = start.elapsed().as_secs_f64();
            let v = velocity.to_f32()?;
            if v.len() != 256 * 64 || v.iter().any(|v| !v.is_finite()) {
                return Err("DiT velocity shape/finite".into());
            }
            fs::write(
                dit_out.join(format!("v{i}.f32")),
                v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            if i == 0 {
                metrics.insert("v0".into(), compare(&dit_out, "v0", &v, &arrays["v0"])?);
            }
            // SAFETY: full disjoint same-context F32 buffers, fixed256*64 extent.
            ffi::check(
                unsafe {
                    axpy(
                        gpu.raw(),
                        lat.fp(),
                        velocity.fp(),
                        (256 * 64) as i64,
                        arrays["sigmas"][i + 1] - arrays["sigmas"][i],
                    )
                },
                "Euler",
            )?;
            let name = format!("latents_{i}");
            metrics.insert(
                name.clone(),
                compare(&dit_out, &name, &lat.to_f32()?, &arrays[&name])?,
            );
            drop(velocity);
            let mem = allocations(&gpu, stats)?;
            if mem.1 > dit.plan().tracked {
                return Err("DiT cumulative peak exceeds aggregate plan".into());
            }
            steps.push(json!({"step":i,"recorded_timestep":arrays["timesteps"][i],"velocity_seconds":velocity_seconds,"tracked_live":mem.0,"cumulative_peak":mem.1}));
            println!("DiT step {i} PASS {velocity_seconds:.6}s");
        }
        let final_latents = lat.to_f32()?;
        let plan = json!({"resident":dit.plan().resident,"tracked":dit.plan().tracked,"with_margin":dit.plan().with_margin});
        drop(lat);
        drop(prefix);
        drop(dit);
        drop(gg);
        let dit_report = json!({"status":"PASS","input":"native TE embeddings","noise":"recorded original seed7 fixture; no native RNG claim","sigmas":arrays["sigmas"],"packed":"Register64","attention_requested":"Tiled","attention_effective_geometry":[256,274,32,128,12288],"load_seconds":load_seconds,"prefix_seconds":prefix_seconds,"steps":steps,"metrics":metrics,"plan":plan,"before":before,"after_drop":empty_snapshot(&gpu,stats)?});
        write_json(&dit_out.join("report.json"), &dit_report)?;
        stages.insert("dit".into(), dit_report);
        let vae_out = out.join("vae");
        fs::create_dir(&vae_out)?;
        let before = empty_snapshot(&gpu, stats)?;
        let start = Instant::now();
        let vae = Vae::load(&vae_file, &gpu, ALLOWANCE as usize)?;
        gpu.sync()?;
        let load_seconds = start.elapsed().as_secs_f64();
        for (name, actual) in [("latents_mean", &vae.mean), ("latents_std", &vae.std)] {
            let expected: Vec<f32> = original_vae_config[name]
                .as_array()
                .ok_or("original latent stats missing")?
                .iter()
                .map(|v| v.as_f64().map(|f| f as f32).ok_or("latent stat not number"))
                .collect::<std::result::Result<_, _>>()?;
            if !bits_equal(actual, &expected) {
                return Err("VAE normalization statistics differ".into());
            }
        }
        let resident = vae.resident_bytes;
        let start = Instant::now();
        let (raw_image, w, h) = vae.decode_f32(&final_latents, 16, 16)?;
        gpu.sync()?;
        let decode_seconds = start.elapsed().as_secs_f64();
        if (w, h) != (256, 256)
            || raw_image.len() != 256 * 256 * 4
            || raw_image.iter().any(|v| !v.is_finite())
        {
            return Err("VAE output shape/finite".into());
        }
        let raw_metric = compare(&vae_out, "image_raw", &raw_image, &vae_refs["expected_raw"])?;
        let public: Vec<_> = raw_image.iter().map(|v| v.clamp(-1., 1.)).collect();
        let public_metric = compare(&vae_out, "image_public", &public, &arrays["image"])?;
        // Diagnostic preview of the captured raw image using the original separate
        // clamp, *0.5, +0.5, *255, positive round conversion (vae.hpp rgba).
        // This does not rerun decoding or substitute CPU inference.
        let rgba: Vec<u8> = public
            .iter()
            .map(|v| {
                let scaled = *v * 0.5f32;
                let shifted = scaled + 0.5f32;
                (shifted * 255.0f32).round().clamp(0., 255.) as u8
            })
            .collect();
        let png = vae_out.join("image.png");
        image::save_buffer_with_format(
            &png,
            &rgba,
            256,
            256,
            image::ColorType::Rgba8,
            image::ImageFormat::Png,
        )?;
        if image::open(&png)?.to_rgba8().as_raw() != &rgba {
            return Err("PNG roundtrip differs".into());
        }
        fs::write(vae_out.join("image.rgba8"), &rgba)?;
        let png_sha = hash(&png)?;
        drop(vae);
        let vae_report = json!({"status":"PASS","input":"native final DiT latents; statistics applied once by unchanged VAE","shape":[256,256,4],"raw":raw_metric,"public":public_metric,"load_seconds":load_seconds,"decode_seconds":decode_seconds,"resident_weight_bytes":resident,"png_sha256":png_sha,"preview":"original conversion of captured raw image; PNG roundtrip exact","before":before,"after_drop":empty_snapshot(&gpu,stats)?});
        write_json(&vae_out.join("report.json"), &vae_report)?;
        stages.insert("vae".into(), vae_report);
        println!("VAE raw/public PASS from native latents");
        Ok(())
    })();
    let diagnostic_seconds = start_all.elapsed().as_secs_f64();
    let mut after = BTreeMap::new();
    let mut identity_error = None;
    for (p, h) in &ids {
        match hash(p) {
            Ok(got) => {
                if got != *h {
                    identity_error = Some(format!("changed {}", p.display()));
                }
                after.insert(p.clone(), got);
            }
            Err(e) => {
                identity_error = Some(e.to_string());
            }
        }
    }
    write_json(&out.join("identities-after.json"), &json!(after))?;
    let cleanup = empty_snapshot(&gpu, stats);
    let pass = result.is_ok() && identity_error.is_none() && cleanup.is_ok();
    let report = json!({"status":if pass{"PASS"}else{"FAIL"},"scope":"fixed256 native prompt/TE/DiT/VAE chain with recorded seed7 noise; no native RNG/general prompt/service admission","prompt":PROMPT,"seed_fixture":7,"steps":8,"relative_l2_max":0.001,"native_te_feeds_dit":true,"native_latents_feed_vae":true,"device_allowance":ALLOWANCE,"driver_measurement":"external sampler required; internal driver_total_free are free-memory snapshots, not per-process usage","peak_scope":"cumulative across one context; stage unload requires live zero","initial":initial,"stages":stages,"diagnostic_wall_seconds":diagnostic_seconds,"timing_scope":"includes perstage validation/readback/file output; excludes initial asset hashing; not warm service latency","error":result.as_ref().err().map(ToString::to_string),"identity_error":identity_error,"after_drop":cleanup.as_ref().ok(),"cleanup_error":cleanup.as_ref().err().map(ToString::to_string)});
    write_json(&out.join("report.json"), &report)?;
    println!("{report}");
    if !pass {
        return Err("native chain numeric/identity/cleanup failure; see report".into());
    }
    Ok(())
}
