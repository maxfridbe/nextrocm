// Native comparison with independently hashed ncnn/Torch CPU image fixtures.
// Usage: waifu2x_numeric MODELS_DIR FIXTURE_DIR REPORT.json
use nextrocm_engine_core::Gpu;
use nextrocm_image::{ImageEngine, ModelFiles, Picture, UpscaleRequest};
use serde_json::{json, Value};
use std::{error::Error, fs, path::Path, time::Instant};

fn stats(gpu: &Gpu) -> Result<(u64, u64), Box<dyn Error>> {
    type F = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    let api = nextrocm_engine_core::api()?;
    let p = api.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err("missing allocation stats".into());
    }
    // SAFETY: runtime's declared ABI, live device, and writable scalar outputs.
    let f: F = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    if unsafe { f(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err("allocation stats failed".into());
    }
    Ok((current, peak))
}
fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("waifu2x_numeric MODELS_DIR FIXTURE_DIR REPORT.json".into());
    }
    let root = Path::new(&args[1]);
    let dir = Path::new(&args[2]);
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("manifest.json"))?)?;
    let mut files = ModelFiles::new();
    for folder in fs::read_dir(root)? {
        let folder = folder?;
        if !folder.file_type()?.is_dir() {
            continue;
        }
        for entry in fs::read_dir(folder.path())? {
            let entry = entry?;
            if matches!(
                entry.path().extension().and_then(|e| e.to_str()),
                Some("param" | "bin")
            ) {
                files.insert(
                    format!(
                        "{}/{}",
                        folder.file_name().to_string_lossy(),
                        entry.file_name().to_string_lossy()
                    ),
                    entry.path(),
                );
            }
        }
    }
    nextrocm_engine_core::use_kind("image");
    let gpu = Gpu::open(0)?;
    let load_start = Instant::now();
    let engine =
        nextrocm_image_upscale::Upscaler::load(&files, std::slice::from_ref(&gpu), &mut |s| {
            eprintln!("{s}")
        })?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let mut rows = Vec::new();
    let mut failed = false;
    for case in manifest["cases"].as_array().ok_or("missing cases")? {
        for repetition in 0..2 {
            let input = fs::read(dir.join(case["input_file"].as_str().ok_or("input filename")?))?;
            let n = |k: &str| case[k].as_u64().ok_or_else(|| format!("missing {k}"));
            let req = UpscaleRequest {
                picture: Picture {
                    width: n("width")? as u32,
                    height: n("height")? as u32,
                    channels: n("channels")? as u8,
                    data: input,
                },
                variant: Some(case["variant"].as_str().ok_or("variant")?.into()),
                scale: Some(n("scale")? as u32),
                noise: Some(case["noise"].as_i64().ok_or("noise")? as i32),
                tile: Some(n("tile")? as u32),
            };
            let start = Instant::now();
            let (out, _) = engine.upscale(&req, &mut |_| Ok(()))?;
            let inference_seconds = start.elapsed().as_secs_f64();
            let shape: Vec<u32> = serde_json::from_value(case["output_shape"].clone())?;
            if shape != [out.height, out.width, u32::from(out.channels)] {
                return Err("output shape mismatch".into());
            }
            let mut comparisons = Vec::new();
            for reference in ["ncnn", "fp16"] {
                let want = fs::read(
                    dir.join(
                        case["outputs"][reference]["file"]
                            .as_str()
                            .ok_or("output filename")?,
                    ),
                )?;
                if want.len() != out.data.len() || want.is_empty() {
                    return Err("output length mismatch".into());
                }
                let mut max = 0u8;
                let mut sum = 0f64;
                for (&a, &b) in out.data.iter().zip(&want) {
                    let d = a.abs_diff(b);
                    max = max.max(d);
                    sum += f64::from(d).powi(2);
                }
                let mse = sum / want.len() as f64;
                let psnr = (mse > 0.0).then(|| 10.0 * (255.0f64.powi(2) / mse).log10());
                let pass = max <= if reference == "fp16" { 4 } else { 6 }
                    && psnr.is_none_or(|p| p >= 50.0);
                failed |= !pass;
                comparisons.push(
                    json!({"reference":reference,"max_byte_error":max,"psnr_db":psnr,"pass":pass}),
                );
            }
            let allocation = stats(&gpu)?;
            if allocation.1 >= 8_000_000_000 {
                return Err("memory cap exceeded".into());
            }
            let output_path = Path::new(&args[3]).with_extension(format!("{}.u8", rows.len()));
            fs::write(&output_path, &out.data)?;
            let row = json!({"case":case["name"],"repetition":repetition,"seconds":start.elapsed().as_secs_f64(),"inference_seconds":inference_seconds,"output_path":output_path,"comparisons":comparisons,"allocations":allocation});
            println!("{row}");
            rows.push(row);
        }
    }
    drop(engine);
    gpu.sync()?;
    let allocation = stats(&gpu)?;
    if allocation.0 != 0 {
        return Err("engine allocations leaked".into());
    }
    let report = json!({"pass":!failed,"load_seconds":load_seconds,"cases":rows,"allocation_after_drop":allocation,"limits":{"fp16_max_byte_error":4,"ncnn_max_byte_error":6,"minimum_psnr_db":50},"provenance":"fixture hashes must be verified before execution; expectations from independent ncnn CPU and Torch"});
    fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
    if failed {
        return Err("one or more image comparisons failed".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("waifu2x check failed: {e}");
        std::process::exit(1);
    }
}
