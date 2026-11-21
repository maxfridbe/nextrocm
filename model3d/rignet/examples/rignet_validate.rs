//! Native validator, intentionally requires --gpu-run. Compile here; run only in an exclusive GPU window.
//! Usage: rignet_validate --gpu-run ARCHIVE.zip MESH [REFERENCE_DIR [0.0001]]
use nextrocm_model3d::{Gpu, LoadOptions, Model3dEngine, ModelFiles, RigRequest};
use nextrocm_model3d_rignet::{check, RigNet};
use std::{path::Path, time::Instant};
// Private validator binding to the existing runtime's tracked-allocation counter.
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
fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if !matches!(a.len(), 3..=5) || a[0] != "--gpu-run" {
        return Err(
            "usage: rignet_validate --gpu-run ARCHIVE.zip MESH [REFERENCE_DIR [0.0001]]"
                .into(),
        );
    }
    let tolerance = check::FULL_STAGE_RELATIVE_TOLERANCE;
    if a.len() == 5 {
        let requested = a[4].parse::<f64>().map_err(|e| e.to_string())?;
        if requested != tolerance {
            return Err(format!(
                "full-stage tolerance is frozen at {tolerance}; see reference/tolerance.json"
            ));
        }
    }
    let data = std::fs::read(&a[2]).map_err(|e| e.to_string())?;
    let mesh = nextrocm_model3d::read_mesh(&data)?;
    let gpu = Gpu::open(0).map_err(|e| e.0)?;
    let kt = Instant::now();
    let kernel_checks = check::kernels(&gpu).map_err(|e| e.0)?;
    let kernel_check_s = kt.elapsed().as_secs_f64();
    let tracked_before = tracked(&gpu)?;
    let before = gpu.memory().map_err(|e| e.0)?;
    let mut files = ModelFiles::new();
    files.insert("models".into(), a[1].clone().into());
    let t = Instant::now();
    let model = RigNet::load(&files, &gpu, &LoadOptions::default(), &mut |s| {
        eprintln!("{s}")
    })
    .map_err(|e| e.0)?;
    gpu.sync().map_err(|e| e.0)?;
    let load_s = t.elapsed().as_secs_f64();
    let req = RigRequest {
        mesh,
        seed: 0,
        ..Default::default()
    };
    let mut durations = Vec::new();
    let mut output_counts = Vec::new();
    for _ in 0..2 {
        let t = Instant::now();
        let out = model
            .rig(&req, &mut |s| {
                eprintln!("{} {}/{} {:.6}s", s.phase, s.at, s.of, s.seconds);
                Ok(())
            })
            .map_err(|e| e.0)?;
        gpu.sync().map_err(|e| e.0)?;
        durations.push(t.elapsed().as_secs_f64());
        out.validate()?;
        let skin = out.skin.as_ref().ok_or("missing output skin")?;
        if skin
            .joints
            .iter()
            .flat_map(|j| j.head)
            .any(|v| !v.is_finite())
            || skin
                .weights
                .iter()
                .flatten()
                .any(|(_, v)| !v.is_finite() || *v < 0.)
        {
            return Err("invalid nonfinite rig output".into());
        }
        output_counts.push((out.positions.len(), skin.joints.len()));
    }
    // A cancelled request must release its work lock and permit the next request.
    let cancelled = model.rig(&req, &mut |_| {
        Err(nextrocm_model3d::Error("validator cancellation".into()))
    });
    if cancelled.is_ok() {
        return Err("cancellation was ignored".into());
    }
    let t = Instant::now();
    model
        .rig(&req, &mut |_| Ok(()))
        .map_err(|e| e.0)?
        .validate()?;
    gpu.sync().map_err(|e| e.0)?;
    let after_cancel_s = t.elapsed().as_secs_f64();
    let mut numeric = None;
    if a.len() >= 4 {
        let t = Instant::now();
        let error = check::run(
            &model,
            Path::new(&a[3]),
            &["joints", "root", "bones", "skin"],
            None,
            &mut |s| eprintln!("{s}"),
        )
        .map_err(|e| e.0)?;
        numeric = Some(
            serde_json::json!({"worst_relative_rms":error,"tolerance":tolerance,"seconds":t.elapsed().as_secs_f64()}),
        );
        if !error.is_finite() || error > tolerance {
            return Err(format!("numeric gate failed: {error} > {tolerance}"));
        }
    }
    let tracked_loaded = tracked(&gpu)?;
    let loaded_memory = gpu.memory().map_err(|e| e.0)?;
    drop(model);
    gpu.sync().map_err(|e| e.0)?;
    let tracked_after_drop = tracked(&gpu)?;
    if tracked_after_drop.0 != tracked_before.0 {
        return Err(format!(
            "allocation leak: before {tracked_before:?}, after {tracked_after_drop:?}"
        ));
    }
    if tracked_after_drop.1 >= 8_000_000_000 {
        return Err("tracked allocation peak exceeds strict cap".into());
    }
    let after_drop = gpu.memory().map_err(|e| e.0)?;
    println!(
        "{}",
        serde_json::json!({"kernel_checks":kernel_checks,"kernel_check_s":kernel_check_s,"load_s":load_s,"first_inference_s":durations[0],"warm_inference_s":durations[1],
        "after_cancel_s":after_cancel_s,"output_counts":output_counts,"numerical":numeric,
        "tracked_current_peak_before":tracked_before,"tracked_current_peak_loaded":tracked_loaded,"tracked_current_peak_after_drop":tracked_after_drop,
        "memory_total_free_before":before,"memory_total_free_loaded":loaded_memory,"memory_total_free_after_drop":after_drop,
        "device_peak_bytes":null,"device_fit_validated":false,"note":"Sample driver allocations and host RSS externally; snapshots are not peak memory."})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
