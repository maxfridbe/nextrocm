//! Measure partial text residency only; this is not model admission or inference.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_llm_inkling2::{config::MiniConfig, device::DeviceWeights, plan::ExecutionPlan, store::HostStore};
use std::{ffi::c_void, fs, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn available() -> Result<u64> {
    let text = fs::read_to_string("/proc/meminfo")?;
    let kb: u64 = text.lines().find_map(|line| line.strip_prefix("MemAvailable:"))
        .ok_or("missing MemAvailable")?.split_whitespace().next()
        .ok_or("missing memory value")?.parse()?;
    Ok(kb.checked_mul(1024).ok_or("memory overflow")?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: inspect_residency GGUF CONFIG CONTEXT GPU NEW_OUTPUT".into());
    }
    let out = Path::new(&args[5]);
    fs::create_dir(out)?;
    let model = fs::canonicalize(&args[1])?;
    let before = fs::metadata(&model)?;
    let identity = (before.len(), before.modified()?);
    let config_bytes = fs::read(&args[2])?;
    let config = MiniConfig::from_official_json(&config_bytes)?;
    let plan = ExecutionPlan::new(&config, args[3].parse()?)?;
    let host_before = available()?;
    let mut store = HostStore::open(&model, host_before)?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(args[4].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let symbol = api()?.symbol("nr_allocation_stats");
    if symbol.is_null() { return Err("allocation stats required".into()); }
    // SAFETY: the validation runtime exports this exact ns statistics ABI.
    let stats: Stats = unsafe { std::mem::transmute(symbol) };
    let measure = || -> Result<(u64, u64)> {
        let (mut live, mut peak) = (0, 0);
        // SAFETY: live GPU and writable u64 outputs.
        if unsafe { stats(gpu.raw(), &mut live, &mut peak) } != 0 {
            return Err("allocation statistics failed".into());
        }
        Ok((live, peak))
    };
    if measure()?.0 != 0 { return Err("expected empty device".into()); }
    let device_before = gpu.memory()?;
    let start = Instant::now();
    let weights = DeviceWeights::load(&mut store, &gpu, &plan, host_before, || Ok(()))?;
    gpu.sync()?;
    let load_seconds = start.elapsed().as_secs_f64();
    let (resident, peak) = measure()?;
    let expected = weights.estimate.resident_weights.checked_add(weights.estimate.kv)
        .and_then(|n| n.checked_add(weights.estimate.convolution_history))
        .and_then(|n| n.checked_add(weights.estimate.expert_pair)).ok_or("allocation overflow")?;
    if resident != expected || peak >= 8_000_000_000 { return Err("residency accounting mismatch/cap".into()); }
    let host_loaded = available()?;
    let device_loaded = gpu.memory()?;
    drop(weights);
    gpu.sync()?;
    let (after_drop, peak) = measure()?;
    if after_drop != 0 { return Err("device allocation leak".into()); }
    let after = fs::metadata(&model)?;
    if identity != (after.len(), after.modified()?) || config_bytes != fs::read(&args[2])? {
        return Err("input changed".into());
    }
    let report = serde_json::json!({
        "status":"PARTIAL_RESIDENCY_PASS", "model_admission":false,
        "scope":"Packed non-expert weights, F32 KV/history and one expert workspace; excludes activations, graph execution and vision",
        "model":model,"model_bytes":identity.0,"context":plan.context(),
        "load_seconds":load_seconds,"resident_bytes":resident,"tracked_peak":peak,
        "live_after_drop":after_drop,"host_available_before":host_before,
        "host_available_loaded":host_loaded,"device_before":device_before,"device_loaded":device_loaded
    });
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
