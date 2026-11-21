//! Same binary/runtime, immutable LoadOptions flag 0/1; no registration.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_image::{Error, ImageEngine, LoadOptions, Picture, VectorizeOutput, VectorizeRequest};
use nextrocm_image_starvector::{check, files_in, StarVector, FFN_SMALL_M_SETTING};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;

// Test-only hashing utility. No subprocess is introduced in production inference.
fn fingerprint(path: &Path) -> Result<serde_json::Value> {
    let result = Command::new("/usr/bin/sha256sum")
        .arg("--")
        .arg(path)
        .output()?;
    if !result.status.success() {
        return Err(format!("sha256sum failed for {}", path.display()).into());
    }
    let text = std::str::from_utf8(&result.stdout)?;
    let hash = text
        .split_whitespace()
        .next()
        .ok_or("missing SHA256 output")?;
    if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("malformed SHA256 output".into());
    }
    Ok(serde_json::json!({"sha256":hash,"bytes":std::fs::metadata(path)?.len()}))
}
fn identities(a: &[String]) -> Result<serde_json::Value> {
    let runtime =
        PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("set absolute NEXTROCM_LIB")?);
    if !runtime.is_absolute() {
        return Err("NEXTROCM_LIB must be absolute".into());
    }
    let mut files = BTreeMap::new();
    // /proc/<our PID>/exe hashes the actual running executable, even after rename.
    files.insert(
        "executable".to_string(),
        PathBuf::from(format!("/proc/{}/exe", std::process::id())),
    );
    files.insert("runtime".into(), runtime);
    files.insert("picture".into(), PathBuf::from(&a[1]));
    let pins: serde_json::Value = serde_json::from_str(include_str!("../weights.json"))?;
    for item in pins["files"]
        .as_array()
        .ok_or("invalid embedded weight manifest")?
    {
        let name = item["name"].as_str().ok_or("weight name")?;
        files.insert(format!("weight:{name}"), Path::new(&a[0]).join(name));
    }
    for name in [
        "pixels.npy",
        "enc.npy",
        "adapter.npy",
        "prompt.npy",
        "stop.npy",
        "logits0.npy",
        "tokens.npy",
        "logits.npy",
    ] {
        files.insert(format!("cpu48:{name}"), Path::new(&a[4]).join(name));
    }
    files.insert(
        "cpu_full:greedy.txt".into(),
        Path::new(&a[5]).join("greedy.txt"),
    );
    let mut records = BTreeMap::new();
    for (name, path) in files {
        let mut record = fingerprint(&path)?;
        record["path"] = serde_json::json!(path.canonicalize()?);
        records.insert(name, record);
    }
    for item in pins["files"].as_array().ok_or("invalid pins")? {
        let key = format!("weight:{}", item["name"].as_str().ok_or("weight name")?);
        if records[&key]["sha256"] != item["sha256"] || records[&key]["bytes"] != item["bytes"] {
            return Err(format!("pinned checkpoint mismatch: {key}").into());
        }
    }
    Ok(
        serde_json::json!({"files":records,"features":{"vision_f32":cfg!(feature="vision-f32"),
        "patch_f32":cfg!(feature="patch-f32"),"operator_profile":cfg!(feature="operator-profile"),
        "vision_taps":cfg!(feature="vision-taps")},"hash_utility":"/usr/bin/sha256sum (validation only)",
        "checkpoint_revision":pins["revision"]}),
    )
}

fn stats(f: Stats, gpu: &Gpu, phase: &str) -> Result<u64> {
    gpu.sync()?;
    let (mut live, mut peak) = (0, 0);
    // SAFETY: checked ns.h signature, live GPU and local output pointers.
    if unsafe { f(gpu.raw(), &mut live, &mut peak) } != 0 {
        return Err("allocation stats failed".into());
    }
    println!(
        "{}",
        serde_json::json!({"phase":phase,"live":live,"peak":peak})
    );
    if peak >= 8_000_000_000 {
        return Err("tracked device cap violated".into());
    }
    Ok(live)
}
fn same(a: &VectorizeOutput, b: &VectorizeOutput) -> bool {
    a.svg == b.svg
        && a.tokens == b.tokens
        && a.closed == b.closed
        && a.prompt_tokens == b.prompt_tokens
}
fn record(out: &Path, name: &str, v: &VectorizeOutput) -> Result<()> {
    std::fs::write(out.join(format!("{name}.svg")), &v.svg)?;
    std::fs::write(
        out.join(format!("{name}.json")),
        serde_json::to_vec_pretty(&serde_json::json!({
            "tokens":v.tokens,"closed":v.closed,"prompt_tokens":v.prompt_tokens,
            "prompt_seconds":v.prompt_seconds,"decode_seconds":v.seconds
        }))?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 6 {
        return Err("usage: starvector_ffn_compare MODEL PICTURE GPU_INDEX NEW_OUT CPU_48_FIXTURE CPU_FULL_FIXTURE".into());
    }
    if !cfg!(feature = "vision-f32")
        || cfg!(feature = "operator-profile")
        || cfg!(feature = "vision-taps")
    {
        return Err("build this comparison with vision-f32 only; no profile/taps features".into());
    }
    if std::env::var("NS_PROFILE").as_deref() == Ok("gpu")
        || std::env::var_os("NR_STARVECTOR_VISION_TAPS").is_some()
    {
        return Err("comparison requires unprofiled execution without tensor capture".into());
    }
    let out = Path::new(&a[3]);
    std::fs::create_dir(out)?;
    let before = identities(&a)?;
    std::fs::write(
        out.join("identities-before.json"),
        serde_json::to_vec_pretty(&before)?,
    )?;
    let files = files_in(Path::new(&a[0]))?;
    let picture = Picture::read_png(Path::new(&a[1])).map_err(Error)?;
    let expected = std::fs::read_to_string(Path::new(&a[5]).join("greedy.txt"))?;
    let gpu = Gpu::open(a[2].parse()?)?;
    let gpu_identity =
        serde_json::json!({"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc});
    std::fs::write(
        out.join("gpu.json"),
        serde_json::to_vec_pretty(&gpu_identity)?,
    )?;
    let symbol = api()?.symbol("nr_allocation_stats");
    if symbol.is_null() {
        return Err("missing allocation stats".into());
    }
    // SAFETY: exact ns.h signature; process retains runtime library.
    let f: Stats = unsafe { std::mem::transmute(symbol) };
    let mut baseline: Vec<VectorizeOutput> = Vec::new();
    let mut timings = Vec::new();
    for flag in [false, true] {
        let dir = out.join(if flag { "flag1" } else { "flag0" });
        std::fs::create_dir(&dir)?;
        if stats(f, &gpu, "before_load")? != 0 {
            return Err("initial tracked allocations".into());
        }
        let mut options = LoadOptions::default();
        options.settings.insert(
            FFN_SMALL_M_SETTING.into(),
            if flag { "1" } else { "0" }.into(),
        );
        let engine =
            StarVector::load_with_options(&files, &gpu, &options, &mut |s| println!("{s}"))?;
        let loaded = stats(f, &gpu, "loaded")?;
        // Changing the caller's options cannot change the already loaded engine.
        options.settings.insert(
            FFN_SMALL_M_SETTING.into(),
            if flag { "0" } else { "1" }.into(),
        );
        if engine
            .model
            .lock()
            .map_err(|_| "model lock poisoned")?
            .nsd
            .ffn_small_m_enabled()
            != flag
        {
            return Err("engine option was not captured at load".into());
        }
        let defaults = engine.settings(&VectorizeRequest {
            picture: picture.clone(),
            ..Default::default()
        })?;
        if defaults.beams != 2
            || !defaults.sample
            || defaults.top_p != 0.9
            || defaults.max_new != 4741
        {
            return Err("original generation defaults changed".into());
        }
        let worst = check::run(
            &engine,
            Path::new(&a[4]),
            Path::new(&a[1]),
            &["pixels", "encoder", "prefill", "logits"],
            &mut |s| println!("{s}"),
        )?;
        if !worst.is_finite() || worst > 0.001 {
            return Err(format!("unchanged numeric stage gate failed {worst}").into());
        }
        let full_req = VectorizeRequest {
            picture: picture.clone(),
            max_length: 5000,
            beams: Some(1),
            sample: Some(false),
            seed: 1234,
            ..Default::default()
        };
        let full = engine.vectorize(&full_req, &mut |_| Ok(()))?;
        record(&dir, "full-greedy", &full)?;
        if full.svg != expected
            || !full.closed
            || full.tokens != 165
            || !full.svg.trim_end().ends_with("</svg>")
        {
            return Err("full 165-token CPU SVG/closure mismatch".into());
        }
        timings.push(serde_json::json!({"flag":u8::from(flag),"load_seconds":engine.load_seconds(),
            "numeric_worst":worst,"full_prompt_seconds":full.prompt_seconds,"full_decode_seconds":full.seconds,
            "timing_scope":"one unprofiled full greedy run per flag after same numeric stage warmup; not live service timing"}));
        let mut outputs = vec![full];
        for beams in [1, 2, 8] {
            let req = VectorizeRequest {
                picture: picture.clone(),
                max_length: 267,
                beams: Some(beams),
                sample: Some(beams > 1),
                seed: 1234,
                ..Default::default()
            };
            let first = engine.vectorize(&req, &mut |_| Ok(()))?;
            record(&dir, &format!("short-m{beams}"), &first)?;
            let stable = stats(f, &gpu, &format!("short_m{beams}"))?;
            let repeat = engine.vectorize(&req, &mut |_| Ok(()))?;
            if !same(&first, &repeat) || stats(f, &gpu, "short_repeat")? != stable {
                return Err("short repeat/output/allocation mismatch".into());
            }
            outputs.push(first);
        }
        let stable = stats(f, &gpu, "eight_beam_stable")?;
        let cancel_req = VectorizeRequest {
            picture: picture.clone(),
            max_length: 267,
            beams: Some(8),
            sample: Some(true),
            seed: 1234,
            ..Default::default()
        };
        let mut callbacks = 0;
        let cancelled = engine.vectorize(&cancel_req, &mut |_| {
            callbacks += 1;
            if callbacks == 2 {
                Err(Error("FFN diagnostic cancellation".into()))
            } else {
                Ok(())
            }
        });
        if callbacks != 2
            || cancelled
                .as_ref()
                .err()
                .is_none_or(|e| !e.to_string().contains("FFN diagnostic cancellation"))
        {
            return Err("cancellation after a decoder step failed".into());
        }
        let recovered = engine.vectorize(&cancel_req, &mut |_| Ok(()))?;
        if !same(&outputs[3], &recovered) || stats(f, &gpu, "recovered")? != stable {
            return Err("cancel recovery drift".into());
        }
        let mut invalid = cancel_req;
        invalid.beams = Some(9);
        if engine.vectorize(&invalid, &mut |_| Ok(())).is_ok() {
            return Err("invalid beam count accepted".into());
        }
        if stats(f, &gpu, "after_invalid")? != stable {
            return Err("invalid call allocation drift".into());
        }
        drop(engine);
        if stats(f, &gpu, "dropped")? != 0 {
            return Err("engine drop leak".into());
        }
        options.settings.insert(
            FFN_SMALL_M_SETTING.into(),
            if flag { "1" } else { "0" }.into(),
        );
        let reload =
            StarVector::load_with_options(&files, &gpu, &options, &mut |s| println!("{s}"))?;
        if stats(f, &gpu, "reloaded")? != loaded {
            return Err("reload allocation drift".into());
        }
        let req = VectorizeRequest {
            picture: picture.clone(),
            max_length: 267,
            beams: Some(1),
            sample: Some(false),
            seed: 1234,
            ..Default::default()
        };
        let again = reload.vectorize(&req, &mut |_| Ok(()))?;
        if !same(&outputs[1], &again) {
            return Err("reload short output drift".into());
        }
        drop(reload);
        if stats(f, &gpu, "redropped")? != 0 {
            return Err("reload drop leak".into());
        }
        if !flag {
            baseline = outputs;
        } else if !baseline.iter().zip(&outputs).all(|(a, b)| same(a, b)) {
            return Err("flag0/flag1 output differs".into());
        }
    }
    let after = identities(&a)?;
    std::fs::write(
        out.join("identities-after.json"),
        serde_json::to_vec_pretty(&after)?,
    )?;
    if before != after {
        return Err("executable/runtime/weights/picture/fixtures changed during comparison".into());
    }
    std::fs::write(
        out.join("comparison.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"pass":true,
        "same_binary_runtime":true,"identities_unchanged":true,"provenance":before,"gpu":gpu_identity,
        "numeric_scope":"independent CPU relative-L2 <=0.001 per flag; no cross-flag tensor bitwise assertion",
        "output_scope":"exact SVG bytes, token counts, prompt counts and closure",
        "full_cpu_165_exact":true,"short_beams":[1,2,8],"short_repeat_exact":true,
        "cross_flag_outputs_exact":true,"immutable_options":true,"cancel_after_decoder_step":true,"drop_live_zero":true,"runs":timings}))?,
    )?;
    println!("PASS flag0/1 same-runtime stages/full165/short1,2,8/repeat/cancel/reload/drop; production unchanged");
    Ok(())
}
