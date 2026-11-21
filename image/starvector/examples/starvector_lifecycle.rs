//! Private admission harness; does not register a model or change settings.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_image::{Error, ImageEngine, Picture, VectorizeRequest};
use nextrocm_image_starvector::{files_in, StarVector};
use std::ffi::c_void;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;

fn stats(f: Stats, gpu: &Gpu, phase: &str) -> Result<u64> {
    gpu.sync()?;
    let (mut current, mut peak) = (0, 0);
    // SAFETY: live GPU, checked symbol signature from ns.h, valid output pointers.
    if unsafe { f(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err("allocation stats failed".into());
    }
    println!(
        "{}",
        serde_json::json!({"phase":phase,"tracked_current":current,"tracked_peak":peak})
    );
    if peak >= 8_000_000_000 {
        return Err("tracked peak violates device budget".into());
    }
    Ok(current)
}

fn request(picture: &Picture, new_tokens: u32) -> VectorizeRequest {
    VectorizeRequest {
        picture: picture.clone(),
        max_length: 259 + new_tokens,
        beams: Some(1),
        sample: Some(false),
        ..Default::default()
    }
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 5 || a.len() > 6 || !["lifecycle", "full-greedy"].contains(&a[4].as_str()) {
        return Err("usage: starvector_lifecycle MODEL_DIR PICTURE GPU_INDEX NEW_OUTPUT_DIR lifecycle|full-greedy [CPU_FULL_ORACLE_DIR]".into());
    }
    if a[4] == "full-greedy" && a.len() != 6 {
        return Err("full-greedy requires CPU_FULL_ORACLE_DIR".into());
    }
    if std::env::var_os("NR_STARVECTOR_VISION_TAPS").is_some() {
        return Err("unset NR_STARVECTOR_VISION_TAPS for lifecycle/full generation".into());
    }
    let out = Path::new(&a[3]);
    std::fs::create_dir(out)?;
    let picture = Picture::read_png(Path::new(&a[1])).map_err(Error)?;
    let gpu = Gpu::open(a[2].parse()?)?;
    let lib = api()?;
    let symbol = lib.symbol("nr_allocation_stats");
    if symbol.is_null() {
        return Err("missing allocation stats".into());
    }
    // SAFETY: exact private runtime signature; library remains loaded.
    let f: Stats = unsafe { std::mem::transmute(symbol) };
    if stats(f, &gpu, "before_load")? != 0 {
        return Err("nonempty initial context".into());
    }
    let files = files_in(Path::new(&a[0]))?;
    let engine = StarVector::load(&files, &gpu, &mut |s| println!("{s}"))?;
    let loaded = stats(f, &gpu, "loaded")?;
    let defaults = engine.settings(&VectorizeRequest {
        picture: picture.clone(),
        ..Default::default()
    })?;
    if defaults.beams != 2 || !defaults.sample || defaults.top_p != 0.9 || defaults.max_new != 4741
    {
        return Err("original default generation contract changed".into());
    }
    if a[4] == "lifecycle" {
        // Warm the bounded eight-beam cache before requiring stable allocations.
        let mut warm = request(&picture, 1);
        warm.beams = Some(8);
        engine.vectorize(&warm, &mut |_| Ok(()))?;
        let baseline = stats(f, &gpu, "eight_beam_warm")?;
        let short = request(&picture, 48);
        let first = engine.vectorize(&short, &mut |_| Ok(()))?;
        if stats(f, &gpu, "short_first")? != baseline {
            return Err("allocation drift after first request".into());
        }
        let mut callbacks = 0;
        let cancelled = engine.vectorize(&short, &mut |_| {
            callbacks += 1;
            Err(Error("diagnostic cancellation".into()))
        });
        if callbacks != 1
            || cancelled
                .as_ref()
                .err()
                .is_none_or(|e| !e.to_string().contains("diagnostic cancellation"))
        {
            return Err("callback cancellation was not propagated".into());
        }
        if stats(f, &gpu, "cancelled")? != baseline {
            return Err("allocation drift after cancellation".into());
        }
        let second = engine.vectorize(&short, &mut |_| Ok(()))?;
        if first.svg != second.svg || first.tokens != second.tokens || first.closed != second.closed
        {
            return Err("greedy repeat differs after cancellation".into());
        }
        std::fs::write(out.join("short-greedy.svg"), &first.svg)?;
        let sampled = VectorizeRequest {
            picture: picture.clone(),
            max_length: 275,
            seed: 1234,
            ..Default::default()
        };
        let s1 = engine.vectorize(&sampled, &mut |_| Ok(()))?;
        let s2 = engine.vectorize(&sampled, &mut |_| Ok(()))?;
        if s1.svg != s2.svg || s1.tokens != s2.tokens {
            return Err("same-seed sampled repeat differs".into());
        }
        let mut invalid = short;
        invalid.beams = Some(9);
        if engine.vectorize(&invalid, &mut |_| Ok(())).is_ok() {
            return Err("invalid beams accepted".into());
        }
        if stats(f, &gpu, "repeat_and_invalid")? != baseline {
            return Err("steady allocation drift".into());
        }
        drop(engine);
        if stats(f, &gpu, "dropped")? != 0 {
            return Err("tracked allocation leak after drop".into());
        }
        let reloaded = StarVector::load(&files, &gpu, &mut |s| println!("{s}"))?;
        if stats(f, &gpu, "reloaded")? != loaded {
            return Err("reload allocation drift".into());
        }
        drop(reloaded);
        if stats(f, &gpu, "redropped")? != 0 {
            return Err("tracked allocation leak after reload/drop".into());
        }
        println!("PASS lifecycle: 8-beam bound, greedy repeat, callback cancellation/recovery, sampled seed repeat, invalid request, reload/drop");
    } else {
        let req = request(&picture, 4741); // Original total max_length=5000, greedy override only.
        let mut callbacks = 0;
        let result = engine.vectorize(&req, &mut |_| {
            callbacks += 1;
            if callbacks % 32 == 0 {
                stats(f, &gpu, "full_progress").map_err(|e| Error(e.to_string()))?;
            }
            Ok(())
        });
        let result = match result {
            Ok(v) => v,
            Err(e) => {
                drop(engine);
                stats(f, &gpu, "full_error_drop")?;
                return Err(e.into());
            }
        };
        std::fs::write(out.join("native.svg"), &result.svg)?;
        let expected = std::fs::read_to_string(Path::new(&a[5]).join("greedy.txt"))?;
        let same = result.svg == expected;
        let closed = result.closed && result.svg.trim_end().ends_with("</svg>");
        std::fs::write(
            out.join("result.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"tokens":result.tokens,"closed":closed,"exact_cpu_text":same,"max_length":5000,"beams":1,"sample":false,"prompt_seconds":result.prompt_seconds,"decode_seconds":result.seconds}),
            )?,
        )?;
        stats(f, &gpu, "full_complete")?;
        drop(engine);
        if stats(f, &gpu, "full_dropped")? != 0 {
            return Err("full-generation allocation leak".into());
        }
        if !closed || !same {
            return Err(
                "full greedy closure/exact CPU text gate failed; artifacts retained".into(),
            );
        }
        println!(
            "PASS full greedy text and closure; independent XML/raster and service checks remain"
        );
    }
    Ok(())
}
