//! Hardware-only numeric/lifecycle checks. Compile here; parent serializes execution.
//! isnet_numeric MODEL.onnx FIXTURE_DIR REPORT.json
//! Predetermined gates are explained in reference/isnet/README.md; no tolerance flags.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{LoadOptions, ModelFiles, Picture};
use nextrocm_image_rembg::{
    IsNetEngine,
    isnet::{IsNet, SIDE},
    onnx, post_process_mask,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Read, path::Path, time::Instant};
type R<T> = Result<T, Box<dyn Error>>;
const CAP: u64 = 8_000_000_000;
const REL: f64 = 1e-3;
const ABS: f64 = 1e-3;
fn ensure(ok: bool, why: impl Into<String>) -> R<()> {
    if ok { Ok(()) } else { Err(why.into().into()) }
}
fn hash(path: &Path) -> R<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut b = [0u8; 65536];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        h.update(&b[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn floats(p: &Path) -> R<Vec<f32>> {
    let b = fs::read(p)?;
    ensure(b.len() % 4 == 0, "partial float")?;
    let v: Vec<_> = b
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().expect("four bytes")))
        .collect();
    ensure(v.iter().all(|x| x.is_finite()), "nonfinite fixture")?;
    Ok(v)
}
fn stats(gpu: &Gpu, clean: bool) -> R<Value> {
    gpu.sync()?;
    type F = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    let api = nextrocm_engine_core::api()?;
    let p = api.symbol("nr_allocation_stats");
    ensure(!p.is_null(), "nr_allocation_stats required")?;
    // SAFETY: runtime's declared ABI; live GPU handle and valid writable scalar outputs.
    let f: F = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: handle remains alive and both pointers reference writable u64 locals.
    ensure(
        unsafe { f(gpu.raw(), &mut current, &mut peak) } == 0,
        "allocation stats failed",
    )?;
    ensure(peak < CAP, format!("allocation peak {peak} exceeds cap"))?;
    ensure(!clean || current == 0, format!("allocation leak {current}"))?;
    Ok(json!({"current":current,"peak":peak}))
}
fn compare(a: &[f32], b: &[f32]) -> R<(f64, f64)> {
    ensure(a.len() == b.len() && !a.is_empty(), "float shape mismatch")?;
    ensure(
        a.iter().chain(b).all(|x| x.is_finite()),
        "nonfinite float output",
    )?;
    let (mut mx, mut sq, mut norm) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = f64::from(x) - f64::from(y);
        mx = mx.max(d.abs());
        sq += d * d;
        norm += f64::from(y).powi(2);
    }
    Ok((mx, (sq / norm.max(1e-30)).sqrt()))
}
fn byte_diff(a: &[u8], b: &[u8]) -> R<(u8, f64)> {
    ensure(a.len() == b.len() && !a.is_empty(), "byte shape mismatch")?;
    let mut max = 0;
    let mut sq = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        let d = x.abs_diff(y);
        max = max.max(d);
        sq += f64::from(d).powi(2);
    }
    Ok((max, (sq / a.len() as f64).sqrt()))
}
fn picture(dir: &Path, case: &Value) -> R<Picture> {
    let w = u32::try_from(case["width"].as_u64().ok_or("width")?)?;
    let h = u32::try_from(case["height"].as_u64().ok_or("height")?)?;
    let c = u8::try_from(case["channels"].as_u64().ok_or("channels")?)?;
    let data = fs::read(dir.join("picture.u8"))?;
    ensure(
        data.len() == w as usize * h as usize * c as usize,
        "input picture size",
    )?;
    Ok(Picture {
        width: w,
        height: h,
        channels: c,
        data,
    })
}
fn evaluate(model: &Path, root: &Path, report: &mut Value) -> R<()> {
    ensure(
        hash(model)? == "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        "checkpoint hash mismatch",
    )?;
    let manifest: Value = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
    let cases = manifest["cases"].as_array().ok_or("missing cases")?;
    ensure(cases.len() == 3, "expected exactly three CPU fixtures")?;
    for (case, (name, w, h, c)) in cases.iter().zip([
        ("pattern", 63, 47, 3),
        ("rgba", 17, 13, 4),
        ("black", 1, 1, 3),
    ]) {
        ensure(
            case["name"] == name
                && case["width"] == w
                && case["height"] == h
                && case["channels"] == c,
            "fixture coverage differs",
        )?;
        for (file, meta) in case["files"].as_object().ok_or("missing file inventory")? {
            ensure(
                !file.contains('/') && !file.contains(".."),
                "unsafe fixture filename",
            )?;
            let p = root.join(name).join(file);
            ensure(
                fs::metadata(&p)?.len() == meta["size"].as_u64().ok_or("size")?
                    && hash(&p)? == meta["sha256"].as_str().ok_or("hash")?,
                format!("fixture integrity {name}/{file}"),
            )?;
        }
        for file in [
            "picture.u8",
            "input.f32",
            "pred.f32",
            "mask.u8",
            "mask_pp.u8",
        ] {
            ensure(case["files"].get(file).is_some(), format!("missing {file}"))?;
        }
    }
    report["fixture_manifest_sha256"] = json!(hash(&root.join("manifest.json"))?);
    nextrocm_engine_core::use_kind("image");
    let gpu = Gpu::open(0)?;
    report["baseline"] = stats(&gpu, true)?;
    let result = (|| -> R<()> {
        let mut rows = Vec::new();
        let mut all = true;
        {
            let load_start = Instant::now();
            let m = onnx::Model::open(model)?;
            let net = IsNet::load(&m, &gpu)?;
            drop(m);
            gpu.sync()?;
            report["numeric_load_seconds"] = json!(load_start.elapsed().as_secs_f64());
            ensure(net.forward(&[]).is_err(), "invalid forward accepted")?;
            for case in cases {
                let name = case["name"].as_str().ok_or("case name")?;
                let dir = root.join(name);
                let pic = picture(&dir, case)?;
                let input = floats(&dir.join("input.f32"))?;
                ensure(input.len() == SIDE * SIDE * 3, "input tensor shape")?;
                let expected = floats(&dir.join("pred.f32"))?;
                ensure(expected.len() == SIDE * SIDE, "prediction tensor shape")?;
                let prep = compare(&IsNetEngine::input(&pic), &input)?;
                let start = Instant::now();
                let pred = net.forward(&input)?;
                let first_forward_seconds = start.elapsed().as_secs_f64();
                let numeric = compare(&pred, &expected)?;
                ensure(
                    pred.iter().all(|p| (0.0..=1.0).contains(p)),
                    "sigmoid out of range",
                )?;
                let refmask = fs::read(dir.join("mask.u8"))?;
                let refpp = fs::read(dir.join("mask_pp.u8"))?;
                let isolated_mask = IsNetEngine::mask_of(&expected, pic.width, pic.height);
                let isolated_pp = post_process_mask(&isolated_mask);
                let mask = IsNetEngine::mask_of(&pred, pic.width, pic.height);
                let mb = byte_diff(&mask.data, &refmask)?;
                let repeat_start = Instant::now();
                let repeated = net.forward(&input)?;
                let repeat_forward_seconds = repeat_start.elapsed().as_secs_f64();
                ensure(repeated == pred, "forward reuse changed output")?;
                let pass = prep.0 <= 1e-6
                    && numeric.0 <= ABS
                    && numeric.1 <= REL
                    && mb.0 <= 2
                    && mb.1 <= 1.0
                    && isolated_mask.data == refmask
                    && isolated_pp.data == refpp;
                all &= pass;
                rows.push(json!({"phase":"numeric","case":name,"pass":pass,"input_max_abs":prep.0,"prediction_max_abs":numeric.0,"prediction_relative_l2":numeric.1,"mask_max_byte":mb.0,"mask_rmse":mb.1,"isolated_mask_exact":isolated_mask.data==refmask,"isolated_post_exact":isolated_pp.data==refpp,"repeat_exact":true,"first_forward_seconds":first_forward_seconds,"repeat_forward_seconds":repeat_forward_seconds,"validation_seconds":start.elapsed().as_secs_f64(),"allocation":stats(&gpu,false)?}));
                report["cases"] = json!(rows);
            }
        }
        gpu.sync()?;
        report["numeric_drop"] = stats(&gpu, true)?;
        let mut files = ModelFiles::new();
        files.insert("isnet".into(), model.to_path_buf());
        for cycle in 0..2 {
            let load_start = Instant::now();
            let engine = (nextrocm_image_rembg::kind().load)(
                &files,
                std::slice::from_ref(&gpu),
                &LoadOptions::default(),
                &mut |s| eprintln!("{s}"),
            )?;
            gpu.sync()?;
            report[format!("engine_load_seconds_{cycle}")] =
                json!(load_start.elapsed().as_secs_f64());
            ensure(
                engine.arch() == "isnet" && engine.removes_background(),
                "wrong engine capabilities",
            )?;
            let invalid = Picture {
                width: 1,
                height: 1,
                channels: 3,
                data: vec![],
            };
            ensure(
                engine.remove_background(&invalid, false).is_err(),
                "invalid picture accepted",
            )?;
            for case in cases {
                let name = case["name"].as_str().ok_or("name")?;
                let dir = root.join(name);
                let pic = picture(&dir, case)?;
                for pp in [false, true] {
                    let request_start = Instant::now();
                    let (rgba, mask) = engine.remove_background(&pic, pp)?;
                    gpu.sync()?;
                    let request_seconds = request_start.elapsed().as_secs_f64();
                    ensure(
                        (rgba.width, rgba.height, rgba.channels) == (pic.width, pic.height, 4)
                            && rgba.data.len() == pic.width as usize * pic.height as usize * 4,
                        "RGBA output shape",
                    )?;
                    ensure(
                        (mask.width, mask.height, mask.channels) == (pic.width, pic.height, 1)
                            && mask.data.len() == pic.width as usize * pic.height as usize,
                        "mask output shape",
                    )?;
                    let rgb = pic.rgb();
                    ensure(
                        rgba.data
                            .chunks_exact(4)
                            .zip(rgb.data.chunks_exact(3))
                            .zip(&mask.data)
                            .all(|((r, p), a)| r[..3] == *p && r[3] == *a),
                        "RGB or alpha contract changed",
                    )?;
                    let want = fs::read(dir.join(if pp { "mask_pp.u8" } else { "mask.u8" }))?;
                    let d = byte_diff(&mask.data, &want)?;
                    let pass = if pp { d.0 == 0 } else { d.0 <= 2 && d.1 <= 1.0 };
                    all &= pass;
                    rows.push(json!({"phase":"engine","cycle":cycle,"case":name,"post_process":pp,"request_seconds":request_seconds,"max_byte":d.0,"rmse":d.1,"pass":pass,"allocation":stats(&gpu,false)?}));
                    report["cases"] = json!(rows);
                }
            }
            drop(engine);
            gpu.sync()?;
            report[format!("lifecycle_drop_{cycle}")] = stats(&gpu, true)?;
        }
        ensure(all, "one or more predetermined numeric gates failed")
    })();
    gpu.sync()?;
    report["final_allocation"] = stats(&gpu, true)?;
    result
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("isnet_numeric MODEL.onnx FIXTURE_DIR REPORT.json");
        std::process::exit(2)
    }
    let start = Instant::now();
    let mut report = json!({"pass":false,"cases":[],"timing_scope":{"numeric_load_seconds":"ONNX read, validation, weight upload and scratch allocation, synchronized","first_forward_seconds":"native forward including input upload, output download and synchronization; excludes preparation and comparisons","repeat_forward_seconds":"same resident native forward and input; excludes comparisons","validation_seconds":"two forwards plus reference comparisons and mask processing","request_seconds":"ImageEngine remove_background including CPU preparation, forward, mask and RGBA; no HTTP or image encoding","seconds":"whole validation process including fixture hashing, loads, forwards, comparisons and cleanup"},"gates":{"input_max_abs":1e-6,"prediction_max_abs":ABS,"prediction_relative_l2":REL,"mask_max_byte":2,"mask_rmse":1.0,"post_process_exact":true,"reuse_exact":true,"allocator_cap_strict":CAP,"allocator_after_drop":0},"cancellation":"not exposed by original remove_background API; invalid-request recovery and repeated reuse tested"});
    let path = Path::new(&args[3]);
    let result = (|| -> R<()> {
        ensure(
            !path.exists(),
            "report path already exists; choose a new report filename",
        )?;
        evaluate(Path::new(&args[1]), Path::new(&args[2]), &mut report)
    })();
    report["seconds"] = json!(start.elapsed().as_secs_f64());
    report["pass"] = json!(result.is_ok());
    if let Err(e) = &result {
        report["error"] = json!(e.to_string());
    }
    if path.exists() {
        eprintln!("refusing to overwrite report");
        std::process::exit(1)
    }
    let write = serde_json::to_vec_pretty(&report)
        .map_err(|e| e.to_string())
        .and_then(|b| fs::write(path, b).map_err(|e| e.to_string()));
    if let Err(e) = write {
        eprintln!("writing report: {e}");
        std::process::exit(1)
    }
    if let Err(e) = result {
        eprintln!("ISNet FAIL: {e}");
        std::process::exit(1)
    }
    println!("ISNet numeric/lifecycle PASS; {}", path.display());
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn numerical_comparison_rejects_missing_and_nonfinite_values() {
        assert!(compare(&[], &[]).is_err());
        assert!(compare(&[1.], &[1., 2.]).is_err());
        assert!(compare(&[f32::NAN], &[1.]).is_err());
        assert!(compare(&[1.], &[f32::INFINITY]).is_err());
        assert!(byte_diff(&[0], &[]).is_err());
    }
    #[test]
    fn numerical_comparison_covers_the_whole_tensor() {
        let (max, rel) = compare(&[1., 2., 4.], &[1., 2., 3.]).unwrap();
        assert_eq!(max, 1.);
        assert!((rel - (1.0f64 / 14.).sqrt()).abs() < 1e-12);
        assert_eq!(byte_diff(&[0, 255], &[0, 253]).unwrap(), (2, 2.0f64.sqrt()));
    }
}
