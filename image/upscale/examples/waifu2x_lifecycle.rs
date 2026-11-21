//! Compile only; parent owns the serialized GPU window.
//! waifu2x_lifecycle MODELS_DIR FIXTURE_DIR REPORT.json [X4_MODELS_DIR INPUT.png REFERENCE.png]
use nextrocm_engine_core::Gpu;
use nextrocm_image::{Error, ImageEngine, ModelFiles, Picture, Step, UpscaleRequest};
use nextrocm_image_upscale::Upscaler;
use serde_json::{json, Value};
use std::{collections::BTreeMap, error::Error as StdError, fs, path::Path, time::Instant};
type R<T> = Result<T, Box<dyn StdError>>;
const CANCEL: &str = "waifu2x-lifecycle-intentional-cancellation";
fn ensure(ok: bool, why: &str) -> R<()> {
    if ok {
        Ok(())
    } else {
        Err(why.into())
    }
}
fn stats(gpu: &Gpu, clean: bool) -> R<Value> {
    type F = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    let api = nextrocm_engine_core::api()?;
    let p = api.symbol("nr_allocation_stats");
    ensure(!p.is_null(), "allocation telemetry unavailable")?;
    // SAFETY: nr_allocation_stats has this ABI; runtime stays loaded for process lifetime.
    let f: F = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: GPU is live and outputs point to writable u64 locals.
    ensure(
        unsafe { f(gpu.raw(), &mut current, &mut peak) } == 0,
        "allocation telemetry failed",
    )?;
    ensure(peak < 8_000_000_000, "tracked peak exceeds strict 8 GB cap")?;
    ensure(
        !clean || current == 0,
        "device allocations remain after drop",
    )?;
    Ok(json!({"current":current,"peak":peak}))
}
fn files(root: &Path, nested: bool) -> R<ModelFiles> {
    let mut files = ModelFiles::new();
    let folders: Vec<_> = if nested {
        nextrocm_image_upscale::waifu2x::VARIANTS
            .iter()
            .map(|v| root.join(v.id))
            .collect()
    } else {
        vec![root.to_owned()]
    };
    for dir in folders {
        for entry in fs::read_dir(dir)? {
            let p = entry?.path();
            if matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("param" | "bin")
            ) {
                files.insert(p.strip_prefix(root)?.to_string_lossy().into_owned(), p);
            }
        }
    }
    Ok(files)
}
fn total(req: &UpscaleRequest) -> u32 {
    let (mut w, mut h) = (req.picture.width, req.picture.height);
    let tile = req.tile.unwrap_or(400);
    let scale = req.scale.unwrap_or(2);
    let passes = if scale == 1 { 1 } else { scale.ilog2() };
    let mut n = 0;
    for _ in 0..passes {
        n += w.div_ceil(tile) * h.div_ceil(tile);
        w *= 2;
        h *= 2;
    }
    n
}
#[derive(Default)]
struct Progress {
    at: u32,
    seconds: f64,
}
impl Progress {
    fn visit(&mut self, s: Step, expected: u32) -> nextrocm_image::Result<()> {
        if s.picture != 0
            || s.at != self.at + 1
            || s.of != expected
            || s.at > expected
            || !s.seconds.is_finite()
            || s.seconds < self.seconds
        {
            return Err(Error("invalid progress sequence".into()));
        }
        self.at = s.at;
        self.seconds = s.seconds;
        Ok(())
    }
}
fn full(engine: &Upscaler, req: &UpscaleRequest) -> R<Picture> {
    let expected = total(req);
    let mut p = Progress::default();
    let (out, info) = engine.upscale(req, &mut |s| p.visit(s, expected))?;
    ensure(
        p.at == expected && info.tiles == expected,
        "missing completed progress callbacks",
    )?;
    ensure(
        (out.width, out.height, out.channels)
            == (
                req.picture.width * req.scale.unwrap_or(2),
                req.picture.height * req.scale.unwrap_or(2),
                req.picture.channels,
            ),
        "output shape mismatch",
    )?;
    ensure(
        out.data.len() == out.width as usize * out.height as usize * out.channels as usize,
        "output storage mismatch",
    )?;
    Ok(out)
}
fn request(root: &Path, case: &Value) -> R<UpscaleRequest> {
    let n = |k: &str| -> R<u32> { Ok(u32::try_from(case[k].as_u64().ok_or("missing integer")?)?) };
    let picture = Picture {
        width: n("width")?,
        height: n("height")?,
        channels: u8::try_from(n("channels")?)?,
        data: fs::read(root.join(case["input_file"].as_str().ok_or("input filename")?))?,
    };
    ensure(
        picture.data.len()
            == picture.width as usize * picture.height as usize * picture.channels as usize,
        "input storage mismatch",
    )?;
    Ok(UpscaleRequest {
        picture,
        variant: Some(case["variant"].as_str().ok_or("variant")?.into()),
        scale: Some(n("scale")?),
        noise: Some(i32::try_from(case["noise"].as_i64().ok_or("noise")?)?),
        tile: Some(n("tile")?),
    })
}
fn run(args: &[String], report: &mut Value) -> R<()> {
    let manifest: Value =
        serde_json::from_slice(&fs::read(Path::new(&args[2]).join("manifest.json"))?)?;
    let cases = manifest["cases"].as_array().ok_or("missing cases")?;
    let selected = [
        ("cunet-rgba-tiles", 1),
        ("anime-repeat4", 1),
        ("photo-repeat8", 2),
    ];
    let mut requests = Vec::new();
    for (name, stop) in selected {
        let c = cases
            .iter()
            .find(|c| c["name"] == name)
            .ok_or("missing lifecycle fixture")?;
        let r = request(Path::new(&args[2]), c)?;
        ensure(total(&r) > stop, "cancellation must precede completion")?;
        requests.push((name, stop, r));
    }
    let weights = files(Path::new(&args[1]), true)?;
    nextrocm_engine_core::use_kind("image");
    let gpu = Gpu::open(0)?;
    report["baseline"] = stats(&gpu, true)?;
    let result = (|| -> R<()> {
        let mut rows = Vec::new();
        let mut reference = BTreeMap::new();
        for cycle in 0..2 {
            let engine = Upscaler::load(&weights, std::slice::from_ref(&gpu), &mut |s| {
                eprintln!("{s}")
            })?;
            for (name, stop, req) in &requests {
                let start = Instant::now();
                let warm = full(&engine, req)?;
                let mut p = Progress::default();
                let expected = total(req);
                let cancelled = engine.upscale(req, &mut |s| {
                    p.visit(s, expected)?;
                    if p.at == *stop {
                        Err(Error(CANCEL.into()))
                    } else {
                        Ok(())
                    }
                });
                ensure(
                    matches!(cancelled,Err(ref e) if e.0==CANCEL),
                    "cancellation sentinel not propagated",
                )?;
                ensure(p.at == *stop, "callback continued after cancellation")?;
                gpu.sync()?;
                let cancelled_stats = stats(&gpu, false)?;
                let reused = full(&engine, req)?;
                let again = full(&engine, req)?;
                ensure(
                    warm.data == reused.data && reused.data == again.data,
                    "reuse changed output after cancellation",
                )?;
                if let Some(previous) = reference.get(name) {
                    ensure(previous == &again.data, "reload changed output")?;
                } else {
                    reference.insert(*name, again.data);
                }
                rows.push(json!({"cycle":cycle,"case":name,"pass":true,"cancel_at":stop,"progress_total":expected,"cancel_sentinel_propagated":true,"reuse_exact":true,"reload_exact":cycle==1,"seconds":start.elapsed().as_secs_f64(),"allocation_after_cancel":cancelled_stats,"allocation_after_reuse":stats(&gpu,false)?}));
                report["cases"] = json!(rows);
            }
            drop(engine);
            gpu.sync()?;
            report[format!("drop_cycle_{cycle}")] = stats(&gpu, true)?;
        }
        if args.len() == 7 {
            let xfiles = files(Path::new(&args[4]), false)?;
            let input = Picture::decode(&fs::read(&args[5])?).map_err(Error)?;
            let want = Picture::decode(&fs::read(&args[6])?).map_err(Error)?;
            ensure(
                input.width <= 64 && input.height <= 64,
                "x4 regression must be a small fixture <=64x64",
            )?;
            let req = UpscaleRequest {
                picture: input,
                variant: Some("realesrgan-x4plus".into()),
                scale: Some(4),
                noise: None,
                tile: Some(32),
            };
            let engine = Upscaler::load(&xfiles, std::slice::from_ref(&gpu), &mut |s| {
                eprintln!("{s}")
            })?;
            let (out, _) = engine.upscale(&req, &mut |_| Ok(()))?;
            let (repeated, _) = engine.upscale(&req, &mut |_| Ok(()))?;
            ensure(out.data == repeated.data, "x4plus reuse differs")?;
            ensure(
                (out.width, out.height, out.channels) == (want.width, want.height, want.channels)
                    && out.data.len() == want.data.len(),
                "x4plus reference shape mismatch",
            )?;
            let mut max = 0;
            let mut squared = vec![0f64; out.channels as usize];
            for (i, (&a, &b)) in out.data.iter().zip(&want.data).enumerate() {
                let d = a.abs_diff(b);
                max = max.max(d);
                squared[i % out.channels as usize] += f64::from(d).powi(2);
            }
            let psnr: Vec<_> = squared
                .iter()
                .map(|sum| {
                    if *sum == 0.0 {
                        None
                    } else {
                        Some(
                            10.0 * (255f64.powi(2)
                                / (sum / (out.width as f64 * out.height as f64)))
                                .log10(),
                        )
                    }
                })
                .collect();
            let pass = max <= 16 && psnr.iter().all(|p| p.is_none_or(|v| v >= 40.0));
            report["x4plus"] = json!({"pass":pass,"max_byte":max,"per_channel_psnr":psnr,"reuse_exact":true,"input":args[5],"reference":args[6]});
            drop(engine);
            gpu.sync()?;
            report["x4plus_drop"] = stats(&gpu, true)?;
            ensure(pass, "x4plus original CLI numeric gate failed")?;
        } else {
            report["x4plus"] =
                json!({"skipped":"optional model/input/reference paths not provided"});
        }
        Ok(())
    })();
    gpu.sync()?;
    report["final_allocation"] = stats(&gpu, true)?;
    result
}
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    // Three required arguments plus three optional x4 regression arguments.
    if ![3, 6].contains(&args.len()) {
        eprintln!("waifu2x_lifecycle MODELS_DIR FIXTURE_DIR REPORT.json [X4_MODELS_DIR INPUT.png REFERENCE.png]");
        std::process::exit(2)
    }
    let report_path = Path::new(&args[2]);
    if report_path.exists() {
        eprintln!("report exists; use a new path");
        std::process::exit(2)
    }
    // Keep program-name indexing in the worker for parity with the original validation examples.
    let args: Vec<_> = std::iter::once("waifu2x_lifecycle".into())
        .chain(args)
        .collect();
    let mut report = json!({"pass":false,"cases":[],"cap_bytes_strict":8_000_000_000u64});
    let start = Instant::now();
    let result = run(&args, &mut report);
    report["pass"] = json!(result.is_ok());
    report["seconds"] = json!(start.elapsed().as_secs_f64());
    if let Err(e) = &result {
        report["error"] = json!(e.to_string());
    }
    let written = serde_json::to_vec_pretty(&report)
        .map_err(|e| e.to_string())
        .and_then(|b| fs::write(&args[3], b).map_err(|e| e.to_string()));
    if let Err(e) = written {
        eprintln!("report write failed: {e}");
        std::process::exit(1)
    }
    if let Err(e) = result {
        eprintln!("waifu2x lifecycle FAIL: {e}");
        std::process::exit(1)
    }
    println!("waifu2x lifecycle PASS; {}", args[3]);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn progress_rejects_skips_bad_totals_and_nonfinite_times() {
        let mut p = Progress::default();
        assert!(p
            .visit(
                Step {
                    picture: 0,
                    at: 2,
                    of: 4,
                    seconds: 0.
                },
                4
            )
            .is_err());
        assert!(p
            .visit(
                Step {
                    picture: 0,
                    at: 1,
                    of: 3,
                    seconds: 0.
                },
                4
            )
            .is_err());
        assert!(p
            .visit(
                Step {
                    picture: 0,
                    at: 1,
                    of: 4,
                    seconds: f64::NAN
                },
                4
            )
            .is_err());
        assert!(p
            .visit(
                Step {
                    picture: 0,
                    at: 1,
                    of: 4,
                    seconds: 1.
                },
                4
            )
            .is_ok());
        assert!(p
            .visit(
                Step {
                    picture: 0,
                    at: 2,
                    of: 4,
                    seconds: 0.5
                },
                4
            )
            .is_err());
    }
}
