//! Parent-run GPU controls. No production registration is needed.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{ModelFiles, OcrRequest, Picture};
use nextrocm_image_paddleocrvl::{operators, strict, PaddleOcrVl};
use std::{path::Path, time::Instant};
fn tracked(gpu: &std::sync::Arc<Gpu>) -> Result<(u64, u64), String> {
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> std::ffi::c_int;
    let api = nextrocm_engine_core::api().map_err(|e| e.0)?;
    let symbol = api.symbol("nr_allocation_stats");
    if symbol.is_null() {
        return Err("runtime lacks nr_allocation_stats".into());
    }
    // SAFETY: kernels/ns/ns.hip exports nr_allocation_stats with this exact signature.
    let stats = unsafe { std::mem::transmute::<*mut std::ffi::c_void, Stats>(symbol) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: live GPU handle and writable local u64 output slots.
    if unsafe { stats(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err("allocation stats failed".into());
    }
    Ok((current, peak))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let mode=a.get(1).ok_or("usage: paddleocr_native operators | validate WEIGHTS FIXTURE CONTRACT_SHA | read WEIGHTS IMAGE MAX_TOKENS")?;
    let g = Gpu::open(0)?;
    if mode == "operators" {
        operators::run(&g, &mut |s| println!("{s}"))?;
        return Ok(());
    }
    let before = tracked(&g)?;
    let dir = Path::new(a.get(2).ok_or("missing weights directory")?);
    let pins: serde_json::Value =
        serde_json::from_str(include_str!("../../../reference/paddleocr/weights.json"))?;
    let mut files = ModelFiles::new();
    for role in ["model", "mmproj", "tokenizer"] {
        let pin = &pins["files"][role];
        let name = Path::new(pin["path"].as_str().ok_or("weight path")?)
            .file_name()
            .ok_or("weight name")?;
        let path = dir.join(name);
        // Verify bytes before loading. Streaming hashing bounds host memory.
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        if file.metadata()?.len() != pin["size"].as_u64().ok_or("weight size")? {
            return Err(format!("{role}: size mismatch").into());
        }
        let mut hash = Sha256::new();
        let mut b = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut b)?;
            if n == 0 {
                break;
            }
            hash.update(&b[..n]);
        }
        if format!("{:x}", hash.finalize()) != pin["sha256"].as_str().ok_or("weight hash")? {
            return Err(format!("{role}: hash mismatch").into());
        }
        files.insert(role.into(), path);
    }
    let options = if let Some(at) = a.iter().position(|v| v == "--settings") {
        nextrocm_image::LoadOptions {
            settings: serde_json::from_slice(&std::fs::read(
                a.get(at + 1).ok_or("settings JSON file")?,
            )?)?,
            ..Default::default()
        }
    } else {
        nextrocm_image::LoadOptions::default()
    };
    let t = Instant::now();
    let m = PaddleOcrVl::load_with_options(&files, &g, &options, &mut |s| println!("{s}"))?;
    println!("load_seconds={}", t.elapsed().as_secs_f64());
    match mode.as_str() {
        "diagnose" => {
            let f = strict::Fixture::open(
                Path::new(a.get(3).ok_or("fixture directory")?),
                a.get(4).ok_or("contract SHA")?,
            )?;
            let report = nextrocm_image_paddleocrvl::diagnose::run(
                &m,
                &f,
                Path::new(a.get(5).ok_or("new output directory")?),
                &mut |s| println!("{s}"),
            )?;
            println!("{}", serde_json::to_string(&report)?);
        }
        "validate" => {
            let fixture = strict::Fixture::open(
                Path::new(a.get(3).ok_or("fixture directory")?),
                a.get(4).ok_or("contract SHA")?,
            )?;
            let report = strict::run(&m, &fixture, &mut |s| println!("{s}"))?;
            println!("{}", serde_json::to_string(&report)?);
        }
        "read" => {
            let picture = Picture::decode(&std::fs::read(a.get(3).ok_or("image")?)?)?;
            let max_tokens: u32 = a.get(4).ok_or("max_tokens")?.parse()?;
            let argument = |name: &str| a.iter().position(|v| v == name).and_then(|i| a.get(i + 1));
            let prompt = argument("--prompt").cloned().unwrap_or_default();
            let columns = match argument("--columns").map(String::as_str) {
                None => None,
                Some("0") => Some(false),
                Some("1") => Some(true),
                Some(_) => return Err("--columns requires 0 or 1".into()),
            };
            let pages: usize = argument("--pages").map_or(Ok(1), |s| s.parse())?;
            if !(1..=2).contains(&pages) {
                return Err("diagnostic --pages must be 1 or 2".into());
            }
            let mut prior = None;
            for run in 0..2 {
                println!("read_start run={run}");
                let t = Instant::now();
                let mut first_piece = true;
                let out = m.read(
                    &OcrRequest {
                        pages: vec![picture.clone(); pages],
                        prompt: prompt.clone(),
                        columns,
                        max_tokens,
                        seed: 0,
                        ..Default::default()
                    },
                    &mut |_| {
                        if first_piece {
                            println!("read_stream_started run={run}");
                            first_piece = false;
                        }
                    },
                )?;
                if prior.as_ref().is_some_and(|s| s != &out.text) {
                    return Err("repeat text differs".into());
                }
                prior = Some(out.text.clone());
                println!(
                    "{}",
                    serde_json::json!({"run":run,"text":out.text,"prompt":out.prompt,"pages":pages,"columns":columns,"tokens":out.tokens,"prompt_tokens":out.prompt_tokens,"wall_seconds":t.elapsed().as_secs_f64(),"prompt_seconds":out.prompt_seconds,"write_seconds":out.seconds})
                );
            }
        }
        _ => return Err("unknown mode".into()),
    }
    drop(m);
    g.sync()?;
    let after = tracked(&g)?;
    if after.0 != before.0 || after.1 >= 7_500_000_000 {
        return Err("allocation leak or tracked cap exceeded".into());
    }
    println!(
        "{}",
        serde_json::json!({"tracked_before":before,"tracked_after_drop":after,"device_fit_validated":false,"note":"Parent must sample DRM/RSS/host availability; tracked peak is not device peak"})
    );
    Ok(())
}
