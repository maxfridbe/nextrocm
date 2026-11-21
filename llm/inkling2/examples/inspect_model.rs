//! No GPU or weight-payload reads; strict actual header plus official JSON inspection.
use nextrocm_gguf::Gguf;
use nextrocm_llm_inkling2::{
    config::MiniConfig, memory::Estimate, plan::ExecutionPlan, weights::Weights,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: inspect_model GGUF OFFICIAL_CONFIG CONTEXT PREFILL_ROWS DEVICE_FREE_BYTES NEW_REPORT".into());
    }
    let report = std::path::Path::new(&a[6]);
    if report.exists() {
        return Err("report must be new".into());
    }
    let _config = MiniConfig::from_official_json(&std::fs::read(&a[2])?)?;
    let gg = Gguf::open(std::path::Path::new(&a[1]))?;
    let w = Weights::read(&gg)?;
    let p = ExecutionPlan::new(&_config, a[3].parse()?)?;
    let mem = std::fs::read_to_string("/proc/meminfo")?;
    let available = mem
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|l| l.split_whitespace().next())
        .ok_or("missing MemAvailable")?
        .parse::<u64>()?
        .checked_mul(1024)
        .ok_or("MemAvailable overflow")?;
    let e = Estimate::text(&w, &p, a[4].parse()?, available, a[5].parse()?)?;
    let v = serde_json::json!({"status":"METADATA_AND_ESTIMATE_COMPLETE","numeric_validation_pass":false,"measured_fit":false,"weights_path":a[1],"config_path":a[2],"tensor_count":gg.tensors.len(),"packed_bytes":w.packed_bytes(),"physical_layers":24,"visits":p.visits().iter().map(|v|serde_json::json!({"ordinal":v.ordinal,"physical_layer":v.physical_layer,"hf_cache_slot":v.cache_slot,"root_gguf_cache_slot":v.ordinal,"core_pass":v.core_pass})).collect::<Vec<_>>(),"context":p.context(),"prefill_rows":a[4].parse::<usize>()?,"host_available_bytes":available,"device_free_bytes_supplied":a[5].parse::<u64>()?,"estimate":{"resident_weights":e.resident_weights,"kv_f32":e.kv,"convolution_history_f32":e.convolution_history,"one_expert_pair":e.expert_pair,"provisional_activation_reserve":e.activation_reserve,"tracked":e.tracked,"driver_margin":e.driver_margin,"device_total":e.device_total,"host_staging":e.host_staging,"host_reserve":e.host_reserve,"prospective_shared_physical":e.prospective_shared_physical},"limits":"No GPU allocation, model numerics, full-payload hash check, image projector residency, or measured admission. Device free is explicit caller snapshot; activation reserve provisional."});
    std::fs::write(report, serde_json::to_vec_pretty(&v)?)?;
    println!("{}", serde_json::to_string_pretty(&v)?);
    Ok(())
}
