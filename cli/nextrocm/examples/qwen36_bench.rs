//! Native, manual benchmark; does not run under cargo test.
//! qwen36_bench MODEL [--pp 16] [--tg 8] [--ctx 512] [--warm-runs 2] [--gpu 0]
//!                  [--tokens FILE.json | --prompt TEXT] [--expert-mib N] [--mirror-mib 0..2048]
//!                  [--save-profile PATH] [--profile]
//! Baseline: unset NS_PROFILE. Separate profiling run: NS_PROFILE=1 ... --profile.
//! Mirror omission defers to the loader environment; explicit zero disables it.
//! Route calibration: opt in with NS_Q35_RECORD_EXPERTS=1, use --tokens or --prompt with a
//! different prompt from measurement, then --save-profile after all PP/TG runs.
//! Saving does not enable route recording. Use --warm-runs 0 for one calibration
//! pass; later measurement loads the profile through NS_Q35_EXPERT_PROFILE.
//! PP is one Engine::feed call; TG is exactly N synchronized one-token feeds,
//! regardless of EOS. Tokenization, argmax and telemetry are outside feed timers.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::LoadOptions;
use nextrocm_tok::Tokenizer;
use serde_json::{json, Value};
use std::{
    error::Error,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const PROMPT: &str = "The capital of France is Paris. Explain how a computer stores numbers and performs arithmetic. ";
#[derive(Debug)]
struct Args {
    model: PathBuf,
    pp: usize,
    tg: usize,
    ctx: usize,
    warm: usize,
    gpu: usize,
    tokens: Option<PathBuf>,
    prompt: Option<String>,
    profile: bool,
    expert_mib: usize,
    mirror_mib: Option<usize>,
    save_profile: Option<PathBuf>,
}
fn args() -> Result<Args> {
    parse_args(
        std::env::args().skip(1),
        &std::env::var("NS_PROFILE").unwrap_or_default(),
    )
}
fn parse_args(mut it: impl Iterator<Item = String>, profile: &str) -> Result<Args> {
    let model=it.next().ok_or("usage: qwen36_bench MODEL [--pp N] [--tg N] [--ctx N] [--warm-runs N] [--gpu N] [--tokens FILE.json | --prompt TEXT] [--expert-mib N] [--mirror-mib 0..2048] [--save-profile PATH] [--profile]")?;
    let mut a = Args {
        model: model.into(),
        pp: 16,
        tg: 8,
        ctx: 512,
        warm: 2,
        gpu: 0,
        tokens: None,
        prompt: None,
        profile: false,
        expert_mib: 0,
        mirror_mib: None,
        save_profile: None,
    };
    while let Some(key) = it.next() {
        if key == "--profile" {
            a.profile = true;
            continue;
        }
        let v = it.next().ok_or("missing option value")?;
        match key.as_str() {
            "--pp" => a.pp = v.parse()?,
            "--tg" => a.tg = v.parse()?,
            "--ctx" => a.ctx = v.parse()?,
            "--warm-runs" => a.warm = v.parse()?,
            "--gpu" => a.gpu = v.parse()?,
            "--tokens" => a.tokens = Some(v.into()),
            "--prompt" => a.prompt = Some(v),
            "--expert-mib" => a.expert_mib = v.parse()?,
            "--mirror-mib" => a.mirror_mib = Some(v.parse()?),
            "--save-profile" => a.save_profile = Some(v.into()),
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if a.tokens.is_some() && a.prompt.is_some() {
        return Err("--tokens and --prompt are mutually exclusive".into());
    }
    if a.pp == 0
        || a.tg == 0
        || a.pp.checked_add(a.tg).is_none_or(|n| n > a.ctx)
        || a.ctx > 1_048_576
        || a.warm > 100
        || a.expert_mib > 7000
        || a.mirror_mib.is_some_and(|mib| mib > 2048)
    {
        return Err("require positive PP/TG, PP+TG <= ctx <= 1048576, warm-runs <=100, expert-mib <=7000, mirror-mib 0..2048".into());
    }
    if a.profile != matches!(profile, "1" | "gpu") {
        return Err("throughput run requires NS_PROFILE off; use NS_PROFILE=1|gpu with --profile for a separate profiling run".into());
    }
    Ok(a)
}
fn tap(_: &str, _: &DevBuf) -> nextrocm_engine_core::Result<()> {
    Ok(())
}
fn greedy(logits: &[f32]) -> Result<u32> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return Err("empty/nonfinite logits".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(u32::try_from(best)?) // equal logits select lowest token ID
}
fn allocator(g: &Gpu) -> Result<(u64, u64)> {
    let a = api()?;
    let p = a.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err("allocation telemetry missing".into());
    }
    type Query = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: retained runtime symbol has this ABI; GPU handle remains live.
    let query: Query = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: valid handle and writable outputs for allocation stats.
    if unsafe { query(g.raw(), &mut current, &mut peak) } != 0 {
        return Err(a.error().into());
    }
    Ok((current, peak))
}
fn proc_value(text: &str, key: &str, multiplier: u64) -> Result<u64> {
    let line = text
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .ok_or_else(|| format!("missing proc counter {key}"))?;
    Ok(line
        .split_whitespace()
        .next()
        .ok_or("missing proc value")?
        .parse::<u64>()?
        .checked_mul(multiplier)
        .ok_or("proc value overflow")?)
}
fn memory(g: &Gpu) -> Result<Value> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let mem = std::fs::read_to_string("/proc/meminfo")?;
    let io = std::fs::read_to_string("/proc/self/io")?;
    let stat = std::fs::read_to_string("/proc/self/stat")?;
    let tail = stat.rsplit_once(')').ok_or("invalid process stat")?.1;
    let fields: Vec<_> = tail.split_whitespace().collect();
    let minor = fields.get(7).ok_or("missing minflt")?.parse::<u64>()?;
    let major = fields.get(9).ok_or("missing majflt")?.parse::<u64>()?;
    let (current, peak) = allocator(g)?;
    let (total, free) = g.memory()?;
    Ok(
        json!({"tracked_current_bytes":current,"tracked_peak_bytes":peak,
        "rss_bytes":proc_value(&status,"VmRSS:",1024)?,"rss_high_water_bytes":proc_value(&status,"VmHWM:",1024)?,
        "host_total_bytes":proc_value(&mem,"MemTotal:",1024)?,"host_available_bytes":proc_value(&mem,"MemAvailable:",1024)?,
        "host_cached_bytes":proc_value(&mem,"Cached:",1024)?,"host_sreclaimable_bytes":proc_value(&mem,"SReclaimable:",1024)?,
        "read_bytes":proc_value(&io,"read_bytes:",1)?,"rchar":proc_value(&io,"rchar:",1)?,
        "minor_faults":minor,"major_faults":major,"fault_counter_source":"/proc/self/stat minflt/majflt (process only)",
        "device_total_bytes":total,"device_free_bytes":free}),
    )
}
fn delta(before: &Value, after: &Value) -> Value {
    let mut out = serde_json::Map::new();
    for key in ["read_bytes", "rchar", "minor_faults", "major_faults"] {
        out.insert(
            key.into(),
            json!(after[key]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(before[key].as_u64().unwrap_or(0))),
        );
    }
    Value::Object(out)
}
fn prompt(a: &Args, tok: &Tokenizer) -> Result<Vec<u32>> {
    if let Some(path) = &a.tokens {
        if std::fs::metadata(path)?.len() > 16 * 1024 * 1024 {
            return Err("token file exceeds16MiB".into());
        }
        let ids: Vec<u32> = serde_json::from_slice(&std::fs::read(path)?)?;
        if ids.len() != a.pp {
            return Err("token file count must equal --pp".into());
        }
        return Ok(ids);
    }
    let seed = tok.encode(a.prompt.as_deref().unwrap_or(PROMPT));
    if seed.is_empty() {
        return Err("prompt tokenization empty".into());
    }
    Ok(seed.into_iter().cycle().take(a.pp).collect())
}
fn run() -> Result<Value> {
    let a = args()?;
    let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let metadata = std::fs::metadata(&a.model)?;
    let file_t = Instant::now();
    let gguf = Gguf::open(&a.model)?;
    let tok = Tokenizer::from_gguf(&gguf)?;
    let ids = prompt(&a, &tok)?;
    let metadata_tokenizer_seconds = file_t.elapsed().as_secs_f64();
    nextrocm_engine_core::use_kind("llm");
    let gpu_t = Instant::now();
    let gpu = Gpu::open(a.gpu)?;
    gpu.sync()?;
    let gpu_open_seconds = gpu_t.elapsed().as_secs_f64();
    let before_load = memory(&gpu)?;
    let options = LoadOptions {
        expert_bytes: (a.expert_mib > 0).then_some(a.expert_mib * 1024 * 1024),
        mirror_bytes: a.mirror_mib.map(|mib| mib * 1024 * 1024),
        draft: false,
        kv: (a.ctx, 1),
    };
    let load_t = Instant::now();
    let engine = (nextrocm_llm_qwen35::kind().load)(
        &gguf,
        std::slice::from_ref(&gpu),
        &options,
        &mut |line| eprintln!("{line}"),
    )?;
    gpu.sync()?;
    let cold_load_seconds = load_t.elapsed().as_secs_f64();
    let after_load = memory(&gpu)?;
    let mut runs = Vec::new();
    let mut reference_ids = None;
    for index in 0..=a.warm {
        gpu.sync()?;
        let t = Instant::now();
        let before_session_bytes = allocator(&gpu)?.0;
        let mut session = engine.session(a.ctx)?;
        let session_bytes = allocator(&gpu)?
            .0
            .checked_sub(before_session_bytes)
            .ok_or("session allocation count regressed")?;
        gpu.sync()?;
        let session_seconds = t.elapsed().as_secs_f64();
        let before_pp = memory(&gpu)?;
        gpu.sync()?;
        let t = Instant::now();
        let mut logits = engine.feed(&mut session, &ids, &mut tap)?;
        gpu.sync()?;
        let pp_seconds = t.elapsed().as_secs_f64();
        let after_pp = memory(&gpu)?;
        let engine_report_after_pp = engine.report(a.pp);
        let first_token = greedy(&logits)?;
        let mut consumed = Vec::with_capacity(a.tg);
        let mut predicted = Vec::with_capacity(a.tg);
        let mut times = Vec::with_capacity(a.tg);
        let before_tg = memory(&gpu)?;
        let total_tg = Instant::now();
        for _ in 0..a.tg {
            let token = greedy(&logits)?;
            consumed.push(token);
            gpu.sync()?;
            let t = Instant::now();
            logits = engine.feed(&mut session, &[token], &mut tap)?;
            gpu.sync()?;
            times.push(t.elapsed().as_secs_f64());
            predicted.push(greedy(&logits)?);
        }
        let tg_wall_seconds = total_tg.elapsed().as_secs_f64();
        let tg_seconds: f64 = times.iter().sum();
        let after_tg = memory(&gpu)?;
        let equal = reference_ids.as_ref().is_none_or(|r| r == &consumed);
        if reference_ids.is_none() {
            reference_ids = Some(consumed.clone());
        }
        let report = engine.report(a.pp + a.tg);
        let t = Instant::now();
        let before_session_drop = allocator(&gpu)?.0;
        drop(session);
        gpu.sync()?;
        let session_drop_seconds = t.elapsed().as_secs_f64();
        let after_drop = memory(&gpu)?;
        if allocator(&gpu)?.0
            != before_session_drop
                .checked_sub(session_bytes)
                .ok_or("session allocation underflow")?
        {
            return Err("session drop did not free its exact allocations (expert cache may remain resident)".into());
        }
        runs.push(json!({"index":index,"class":if index==0{"cold_session"}else{"warm_session"},"session_create_seconds":session_seconds,
            "pp_tokens":a.pp,"pp_seconds":pp_seconds,"pp_tokens_per_second":a.pp as f64/pp_seconds,"pp_mean_seconds_per_token":pp_seconds/a.pp as f64,
            "first_token_from_pp":first_token,"tg_steps":a.tg,"tg_feed_seconds":tg_seconds,"tg_wall_seconds_including_argmax_and_sync":tg_wall_seconds,
            "tg_tokens_per_second":a.tg as f64/tg_seconds,"tg_seconds_per_token":times,"tg_input_token_ids":consumed,"tg_predicted_token_ids":predicted,
            "greedy_ids_match_cold_session":equal,"memory_before_pp":before_pp,"memory_after_pp":after_pp,"pp_io_fault_delta":delta(&before_pp,&after_pp),
            "memory_before_tg":before_tg,"memory_after_tg":after_tg,"tg_io_fault_delta":delta(&before_tg,&after_tg),
            "engine_report_after_pp":engine_report_after_pp,"engine_report_cumulative":report,"session_drop_seconds":session_drop_seconds,"memory_after_session_drop":after_drop}));
    }
    // All PP/TG timing and per-run memory snapshots have finished. Preserve the
    // original Engine contract: returned count is entries written, not routes.
    let saved_profile = if let Some(path) = &a.save_profile {
        let entries = engine.save_profile(path)?;
        Some(json!({"path":path,"entries_written":entries}))
    } else {
        None
    };
    let final_report = engine.report(a.pp + a.tg);
    let engine_load_seconds = engine.load_seconds();
    let engine_load_bytes = engine.load_bytes();
    let arch = engine.arch();
    let fingerprint = engine.cache_fingerprint();
    let prefill_chunk = engine.prefill_chunk();
    drop(engine);
    gpu.sync()?;
    let after_engine_drop = memory(&gpu)?;
    if allocator(&gpu)?.0
        != before_load["tracked_current_bytes"]
            .as_u64()
            .ok_or("missing initial allocation count")?
    {
        return Err("engine drop leaked allocations".into());
    }
    let mut env = serde_json::Map::new();
    for key in [
        "NS_PROFILE",
        "NS_Q35_PROFILE_DETAIL",
        "NS_Q35_PAGE_CACHE",
        "NS_Q35_PINNED_STAGE",
        "NS_Q35_CHUNK",
        "NS_Q35_SCALAR",
        "NS_Q35_HOST_MIB",
        "NS_Q35_EXPERT_PROFILE",
        "NS_Q35_PREFETCH_MIB",
        "NS_Q35_RECORD_EXPERTS",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_ARCH",
    ] {
        env.insert(key.into(), json!(std::env::var(key).ok()));
    }
    Ok(
        json!({"schema":"qwen36-native-bench-v1","started_unix_seconds":started,"mode":if a.profile{"profile_not_baseline"}else{"throughput"},
        "model_path":a.model.canonicalize()?.display().to_string(),"model_bytes":metadata.len(),"model_modified_unix_seconds":metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs(),
        "architecture":arch,"gpu_index":a.gpu,"gpu_name":gpu.name,"context":a.ctx,"load_options":{"expert_bytes":options.expert_bytes,"mirror_bytes":options.mirror_bytes,"draft":false,"kv_tokens":a.ctx,"kv_sessions":1},
        "environment":env,"prompt_source":if a.tokens.is_some(){"explicit_token_file"}else if a.prompt.is_some(){"explicit_text_cycle"}else{"fixed_seed_token_cycle"},"prompt_token_file":a.tokens,"prompt_seed_text":a.prompt.as_deref().unwrap_or(PROMPT),"prompt_token_ids":ids,"prompt_decoded":tok.decode(&ids),
        "chat_template_applied":false,"eos_early_stop":false,"tg_definition":"exactly TG one-token Engine::feed calls; first input is PP argmax; each call predicts another token; includes final feed even on EOS",
        "timing":"GPU synchronized before/after PP and each TG feed; no taps; PP per-token value is mean, TG array is measured; telemetry outside feed timers",
        "cache_scope":"cold means first engine load/session in this process, not dropped OS file cache; warm means new session on same loaded engine; OS cache excluded from RSS and reported separately",
        "memory_scope":"allocator peak is process runtime lifetime; RSS high-water lifetime; GPU/host free and cache are snapshots, not sampled peaks",
        "metadata_tokenizer_seconds":metadata_tokenizer_seconds,"gpu_open_seconds":gpu_open_seconds,"cold_load_seconds":cold_load_seconds,"engine_load_seconds":engine_load_seconds,"engine_load_bytes":engine_load_bytes,
        "cache_fingerprint":fingerprint,"prefill_chunk":prefill_chunk,"memory_before_load":before_load,"memory_after_load":after_load,"load_io_fault_delta":delta(&before_load,&after_load),
        "runs":runs,"saved_profile":saved_profile,"engine_report_before_drop":final_report,"memory_after_engine_drop":after_engine_drop}),
    )
}
fn main() {
    match run() {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("JSON value serialization")
        ),
        Err(e) => {
            eprintln!("{}", json!({"error":e.to_string()}));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mirror_omission_zero_and_bounds() {
        let parse = |values: &[&str]| parse_args(values.iter().map(|v| (*v).to_owned()), "");
        assert_eq!(parse(&["model"]).unwrap().mirror_mib, None);
        assert_eq!(
            parse(&["model", "--mirror-mib", "0"]).unwrap().mirror_mib,
            Some(0)
        );
        assert_eq!(
            parse(&["model", "--mirror-mib", "2048"])
                .unwrap()
                .mirror_mib,
            Some(2048)
        );
        for invalid in ["2049", "-1", "not-a-number"] {
            assert!(parse(&["model", "--mirror-mib", invalid]).is_err());
        }
        assert!(parse(&["model", "--mirror-mib"]).is_err());
        let calibration = parse(&[
            "model",
            "--tokens",
            "calibration.json",
            "--save-profile",
            "routes.json",
        ])
        .unwrap();
        assert_eq!(calibration.tokens, Some(PathBuf::from("calibration.json")));
        assert_eq!(calibration.save_profile, Some(PathBuf::from("routes.json")));
        assert!(!calibration.profile);
    }
    #[test]
    fn explicit_prompt_and_tokens_are_exclusive() {
        let parse = |values: &[&str]| parse_args(values.iter().map(|v| (*v).to_owned()), "");
        let a = parse(&[
            "model",
            "--prompt",
            "Weather and science.",
            "--pp",
            "64",
            "--tg",
            "128",
        ])
        .unwrap();
        assert_eq!(a.prompt.as_deref(), Some("Weather and science."));
        assert_eq!((a.pp, a.tg), (64, 128));
        assert!(a.tokens.is_none());
        assert!(parse(&["model", "--tokens", "ids.json", "--prompt", "text"]).is_err());
        assert!(parse(&["model", "--prompt", "text", "--tokens", "ids.json"]).is_err());
        assert!(parse(&["model", "--prompt"]).is_err());
    }
    #[test]
    fn greedy_tie_and_nonfinite() {
        assert_eq!(greedy(&[1.0, 2.0, 2.0]).unwrap(), 1);
        assert!(greedy(&[f32::NAN]).is_err());
        assert!(greedy(&[]).is_err());
    }
    #[test]
    fn proc_units() {
        assert_eq!(
            proc_value("VmRSS: 123 kB\n", "VmRSS:", 1024).unwrap(),
            125952
        );
        assert!(proc_value("", "missing", 1).is_err());
    }
}
