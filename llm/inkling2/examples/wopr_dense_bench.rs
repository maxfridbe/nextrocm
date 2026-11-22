#[path = "wopr_dense_support/mod.rs"]
mod staging_support;
// Separate phase-marked native benchmark; no model/kernel changes or intermediate taps.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm_inkling2::{config::MiniConfig, text_graph::TextGraph};
use nextrocm_tok::{inkling2_chat, Tokenizer};
use serde_json::{json, Value};
use std::{ffi::c_void, fs, io::Write, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const REPLY: &str = "7edc08355a9e4d504226aa4be56e4c6537a1ed6fb89d74115602f3dec9a2fc57";
#[repr(C)]
struct Timespec {
    sec: i64,
    nsec: i64,
}
unsafe extern "C" {
    fn clock_gettime(clock: i32, time: *mut Timespec) -> i32;
}
fn monotonic_ns() -> Result<u64> {
    let mut t = Timespec { sec: 0, nsec: 0 };
    // Linux CLOCK_MONOTONIC; both supported deployment and build targets are x86_64 Linux.
    if unsafe { clock_gettime(1, &mut t) } != 0
        || t.sec < 0
        || !(0..1_000_000_000).contains(&t.nsec)
    {
        return Err("CLOCK_MONOTONIC unavailable".into());
    }
    Ok((t.sec as u64)
        .checked_mul(1_000_000_000)
        .and_then(|n| n.checked_add(t.nsec as u64))
        .ok_or("clock overflow")?)
}
fn hash(p: &Path) -> Result<String> {
    let o = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(p)
        .output()?;
    if !o.status.success() {
        return Err("hash failed".into());
    }
    let text = String::from_utf8(o.stdout)?;
    let h = text.split_whitespace().next().ok_or("missing hash")?;
    if h.len() != 64 || !h.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn available() -> Result<u64> {
    let text = fs::read_to_string("/proc/meminfo")?;
    let kb = text
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|l| l.split_whitespace().next())
        .ok_or("MemAvailable missing")?
        .parse::<u64>()?;
    Ok(kb.checked_mul(1024).ok_or("memory overflow")?)
}
fn counters() -> Result<Value> {
    let mut io = serde_json::Map::new();
    for line in fs::read_to_string("/proc/self/io")?.lines() {
        let (k, v) = line.split_once(':').ok_or("bad proc IO")?;
        io.insert(k.into(), json!(v.trim().parse::<u64>()?));
    }
    let stat = fs::read_to_string("/proc/self/stat")?;
    let fields: Vec<_> = stat
        .rsplit_once(')')
        .ok_or("bad proc stat")?
        .1
        .split_whitespace()
        .collect();
    if fields.len() < 13 {
        return Err("short proc stat".into());
    }
    Ok(
        json!({"io":io,"cpu_user_ticks":fields[11].parse::<u64>()?,"cpu_system_ticks":fields[12].parse::<u64>()?}),
    )
}
fn mark(
    file: &mut fs::File,
    marks: &mut Vec<Value>,
    leg: usize,
    mode: &str,
    phase: &str,
    edge: &str,
) -> Result<()> {
    let ns = monotonic_ns()?;
    let value = json!({"leg":leg,"mode":mode,"phase":phase,"edge":edge,"clock_monotonic_ns":ns,"monotonic_seconds":ns as f64 / 1e9,"pid":std::process::id(),"counters":counters()?});
    writeln!(file, "{value}")?;
    file.flush()?;
    println!("{value}");
    std::io::stdout().flush()?;
    marks.push(value);
    Ok(())
}
fn greedy(logits: &[f32]) -> Result<u32> {
    if logits.len() != 200058 || logits.iter().any(|v| !v.is_finite()) {
        return Err("invalid full logits".into());
    }
    let mut best = 0;
    for i in 1..logits.len() {
        if logits[i] > logits[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("usage: wopr_dense_bench GGUF GPU NEW_OUT FROZEN_REPLY_REPORT; NEXTROCM_LIB=selected80d565 runtime".into());
    }
    let tg = 32usize;
    if std::env::var("NS_PROFILE").is_ok_and(|s| !s.is_empty() && s != "0") {
        return Err("kernel profiling must be off".into());
    }
    let embedding = nextrocm_llm_inkling2::device::EmbeddingMode::HostRow;
    let model = fs::canonicalize(&a[1])?;
    let runtime =
        fs::canonicalize(std::env::var_os("NEXTROCM_LIB").ok_or("explicit runtime required")?)?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let reply = fs::canonicalize(&a[4])?;
    let pins = [hash(&runtime)?, hash(&exe)?, hash(&reply)?];
    if pins[0] != "14f11de932424e7bf9fa22481382d632fdb26d20d97199a548f56822ef541ef2"
        || pins[2] != REPLY
    {
        return Err("runtime/reference pins differ".into());
    }
    let baseline: Value = serde_json::from_slice(&fs::read(&reply)?)?;
    let expected: Vec<u32> = serde_json::from_value(baseline["generated_ids"].clone())?;
    let ids: Vec<u32> = serde_json::from_value(baseline["input_ids"].clone())?;
    let header = Gguf::open(&model)?;
    let tok = Tokenizer::from_gguf(&header)?;
    let prompt = inkling2_chat(&[json!({"role":"user","content":"Say hello."})], None, &[])?;
    if ids.len() != 18
        || expected.len() != 63
        || expected.last() != Some(&200006)
        || tok.encode(&prompt) != ids
        || tok.stop != [200006]
    {
        return Err("frozen prompt/reply mismatch".into());
    }
    let meta = fs::metadata(&model)?;
    let identity = (meta.len(), meta.modified()?);
    if identity.0 != 16610543328 {
        return Err("wrong model bytes".into());
    }
    let config = MiniConfig::from_official_json(include_bytes!("../assets/mini-config.json"))?;
    let out = Path::new(&a[3]);
    fs::create_dir(out)?;
    let mut phases = fs::File::create(out.join("phases.jsonl"))?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(a[2].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let ptr = api()?.symbol("nr_allocation_stats");
    if ptr.is_null() {
        return Err("stats ABI missing".into());
    }
    // Exact runtime statistics ABI; runtime and GPU outlive all calls.
    let stats: Stats = unsafe { std::mem::transmute(ptr) };
    let stat = || -> Result<[u64; 2]> {
        let (mut live, mut peak) = (0, 0);
        if unsafe { stats(gpu.raw(), &mut live, &mut peak) } != 0 {
            return Err("stats failed".into());
        }
        Ok([live, peak])
    };
    if stat()?[0] != 0 {
        return Err("requires empty GPU context".into());
    }
    let mut legs = Vec::new();
    let mut all_marks = Vec::new();
    let result = (|| -> Result<()> {
        use nextrocm_llm_inkling2::expert_cache::ExpertMode;
        for (leg, (mode_name, dense_batch)) in [
            ("baseline4", false),
            ("dense4", true),
            ("dense4", true),
            ("baseline4", false),
        ]
        .into_iter()
        .enumerate()
        {
            let chunk_size = 4usize;
            let batch_experts = true;
            let expert_mode = ExpertMode::SegmentedLru {
                budget_bytes: 939524096,
            };
            if stat()?[0] != 0 {
                return Err("nonzero live allocation before next leg".into());
            }
            let mut marks = Vec::new();
            let mut expert_snapshots = serde_json::Map::new();
            let mut batch_snapshots = serde_json::Map::new();
            let mut dense_snapshots = serde_json::Map::new();
            let mut load_seconds = None;
            let mut pp_seconds = None;
            let mut tg_seconds = None;
            let mut pp_forward_seconds = Vec::new();
            let mut pp_forward_calls = 0usize;
            let mut tg_forward_seconds = Vec::new();
            let mut generated = Vec::new();
            let leg_result = (|| -> Result<()> {
                mark(&mut phases, &mut marks, leg, mode_name, "load", "begin")?;
                let start = Instant::now();
                let mut graph = TextGraph::load_with_dense_prefill_policies(
                    &model,
                    &config,
                    &gpu,
                    4096,
                    available()?,
                    expert_mode,
                    embedding,
                    chunk_size,
                    batch_experts,
                    dense_batch,
                    || Ok(()),
                )?;
                gpu.sync()?;
                if graph.expert_mode() != expert_mode {
                    return Err("effective expert mode differs".into());
                }
                dense_snapshots.insert(
                    "after_load".into(),
                    staging_support::dense_stats(
                        graph.dense_stats().family_calls,
                        graph.dense_stats().logical_vectors,
                    ),
                );
                batch_snapshots.insert(
                    "after_load".into(),
                    staging_support::batch_stats(graph.batch_stats()),
                );
                expert_snapshots.insert(
                    "after_load".into(),
                    staging_support::stats(graph.expert_stats()),
                );
                load_seconds = Some(start.elapsed().as_secs_f64());
                mark(&mut phases, &mut marks, leg, mode_name, "load", "end")?;
                mark(&mut phases, &mut marks, leg, mode_name, "pp", "begin")?;
                let start = Instant::now();
                let mut logits = Vec::new();
                for part in ids.chunks(chunk_size) {
                    let t = Instant::now();
                    let rows = if chunk_size == 1 {
                        vec![graph.token(part[0], || Ok(()), |_, _| Ok(()))?]
                    } else {
                        graph.chunk(part, || Ok(()), |_, _, _| Ok(()))?
                    };
                    pp_forward_seconds.push(t.elapsed().as_secs_f64());
                    if rows.len() != part.len() {
                        return Err("prefill row count differs".into());
                    }
                    for row in rows {
                        greedy(&row)?;
                        logits = row;
                        pp_forward_calls += 1;
                    }
                }
                gpu.sync()?;
                pp_seconds = Some(start.elapsed().as_secs_f64());
                mark(&mut phases, &mut marks, leg, mode_name, "pp", "end")?;
                staging_support::validate_batch(
                    graph.batch_stats(),
                    graph.expert_stats(),
                    18,
                    chunk_size,
                    batch_experts,
                    0,
                )?;
                staging_support::validate_dense(
                    graph.dense_stats().family_calls,
                    graph.dense_stats().logical_vectors,
                    18,
                    4,
                    dense_batch,
                )?;
                dense_snapshots.insert(
                    "after_pp".into(),
                    staging_support::dense_stats(
                        graph.dense_stats().family_calls,
                        graph.dense_stats().logical_vectors,
                    ),
                );
                batch_snapshots.insert(
                    "after_pp".into(),
                    staging_support::batch_stats(graph.batch_stats()),
                );
                expert_snapshots.insert(
                    "after_pp".into(),
                    staging_support::stats(graph.expert_stats()),
                );
                let mut next = greedy(&logits)?;
                if next != expected[0] {
                    return Err("PP greedy mismatch".into());
                }
                generated.push(next);
                mark(&mut phases, &mut marks, leg, mode_name, "tg", "begin")?;
                let start = Instant::now();
                for i in 0..tg {
                    if next == 200006 {
                        return Err("unexpected early EOS".into());
                    }
                    let t = Instant::now();
                    logits = graph.token(next, || Ok(()), |_, _| Ok(()))?;
                    tg_forward_seconds.push(t.elapsed().as_secs_f64());
                    next = greedy(&logits)?;
                    if next != expected[i + 1] {
                        return Err(format!("TG greedy mismatch at{}", i + 1).into());
                    }
                    generated.push(next);
                }
                gpu.sync()?;
                tg_seconds = Some(start.elapsed().as_secs_f64());
                mark(&mut phases, &mut marks, leg, mode_name, "tg", "end")?;
                staging_support::validate_batch(
                    graph.batch_stats(),
                    graph.expert_stats(),
                    18,
                    chunk_size,
                    batch_experts,
                    32,
                )?;
                staging_support::validate_dense(
                    graph.dense_stats().family_calls,
                    graph.dense_stats().logical_vectors,
                    18,
                    4,
                    dense_batch,
                )?;
                dense_snapshots.insert(
                    "after_tg".into(),
                    staging_support::dense_stats(
                        graph.dense_stats().family_calls,
                        graph.dense_stats().logical_vectors,
                    ),
                );
                batch_snapshots.insert(
                    "after_tg".into(),
                    staging_support::batch_stats(graph.batch_stats()),
                );
                expert_snapshots.insert(
                    "after_tg".into(),
                    staging_support::stats(graph.expert_stats()),
                );
                mark(&mut phases, &mut marks, leg, mode_name, "drop", "begin")?;
                if batch_snapshots["after_pp"] != batch_snapshots["after_tg"] {
                    return Err("serial TG changed batch diagnostics".into());
                }
                dense_snapshots.insert(
                    "before_drop".into(),
                    staging_support::dense_stats(
                        graph.dense_stats().family_calls,
                        graph.dense_stats().logical_vectors,
                    ),
                );
                batch_snapshots.insert(
                    "before_drop".into(),
                    staging_support::batch_stats(graph.batch_stats()),
                );
                expert_snapshots.insert(
                    "before_drop".into(),
                    staging_support::stats(graph.expert_stats()),
                );
                if dense_snapshots["after_pp"] != dense_snapshots["after_tg"] {
                    return Err("serial TG changed dense diagnostics".into());
                }
                drop(graph);
                gpu.sync()?;
                mark(&mut phases, &mut marks, leg, mode_name, "drop", "end")?;
                Ok(())
            })();

            let mut leg_errors = Vec::new();
            if let Err(e) = gpu.sync() {
                leg_errors.push(e.to_string());
            }
            let allocation = match stat() {
                Ok(v) => Some(v),
                Err(e) => {
                    leg_errors.push(e.to_string());
                    None
                }
            };
            if allocation.is_none_or(|v| v[0] != 0 || v[1] > 4_026_531_840) {
                leg_errors.push("leg drop/cumulative peak gate failed".into());
            }
            if let Some(first) = legs.first() {
                let first: &Value = first;
                if first["expert_snapshots"] != json!(expert_snapshots)
                    || first["batch_snapshots"] != json!(batch_snapshots)
                {
                    leg_errors.push("dense ABBA changed expert/cache snapshots".into());
                }
            }
            let leg_error = leg_result.err().map(|e| e.to_string());
            let pass = leg_error.is_none() && leg_errors.is_empty();
            all_marks.extend(marks.iter().cloned());
            legs.push(json!({"leg":leg,"mode":mode_name,"expert_mode":format!("{expert_mode:?}"),"status":if pass{"BENCHMARK_PASS"}else{"INCOMPLETE"},"error":leg_error,"post_errors":leg_errors,"context":4096,"embedding_mode":"HostRow","generated_ids":generated,"pp_forward_calls":pp_forward_calls,"chunk_size":chunk_size,"tg_forward_calls":tg_forward_seconds.len(),"load_seconds":load_seconds,"pp_seconds":pp_seconds,"tg_seconds":tg_seconds,"pp_chunk_seconds":pp_forward_seconds,"tg_forward_seconds":tg_forward_seconds,"marks":marks,"expert_snapshots":expert_snapshots,"batch_snapshots":batch_snapshots,"batch_experts":batch_experts,"dense_batch":dense_batch,"dense_snapshots":dense_snapshots,"allocation_after_drop":allocation,"allocation_peak_scope":"cumulative context high water across preceding legs"}));
            // Preserve completed/failed leg evidence even if a later process failure occurs.
            fs::write(
                out.join(format!("leg-{leg}.json")),
                serde_json::to_vec_pretty(legs.last().ok_or("missing leg")?)?,
            )?;
            if !pass {
                return Err(format!("leg{leg} incomplete").into());
            }
        }
        Ok(())
    })();
    let mut post_errors = Vec::new();
    if let Err(e) = gpu.sync() {
        post_errors.push(e.to_string());
    }
    let allocation = match stat() {
        Ok(v) => Some(v),
        Err(e) => {
            post_errors.push(e.to_string());
            None
        }
    };
    if allocation.is_none_or(|v| v[0] != 0 || v[1] > 4_026_531_840) {
        post_errors.push("drop/peak gate failed".into());
    }
    for (p, h) in [(&runtime, &pins[0]), (&exe, &pins[1]), (&reply, &pins[2])] {
        match hash(p) {
            Ok(now) if &now == h => (),
            _ => post_errors.push("post identity mismatch/unreadable".into()),
        }
    }
    match fs::metadata(&model).and_then(|m| Ok((m.len(), m.modified()?))) {
        Ok(now) if now == identity => (),
        _ => post_errors.push("model metadata changed/unreadable".into()),
    }
    let passed = result.is_ok() && post_errors.is_empty() && legs.len() == 4;
    let report = json!({"status":if passed{"PREFILL_BENCHMARK_PASS"}else{"INCOMPLETE"},"error":result.err().map(|e|e.to_string()),"post_errors":post_errors,"scope":"one process/runtime/context, four fresh graphs baseline4/dense4/dense4/baseline4; expert batching true throughout, only dense batching differs; each fixed18PP32actualTG,33greedyIDs; no new full-logit numerical gate or general performance claim","timing_scope":"all output rows retained; native timing loops including synchronization/full-logit readback and scans; fresh graph and staging buffers each leg; no capture/file/hash inside forward loops; load/drop outside PP/TG","clock":"CLOCK_MONOTONIC nanoseconds","model":model,"model_bytes":identity.0,"model_identity_scope":"size+mtime before/after; full asset SHA supplied externally","executable":exe,"runtime":runtime,"reference":reply,"pins":pins,"input_ids":ids,"expected_generated_ids":expected[..33],"context":4096,"legs":legs,"marks":all_marks,"allocation_after_drop":allocation,"allocation_peak_scope":"cumulative across all four legs in same GPU context","numeric_validation_pass":false});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !passed {
        return Err("utilization benchmark incomplete; see report".into());
    }
    Ok(())
}
