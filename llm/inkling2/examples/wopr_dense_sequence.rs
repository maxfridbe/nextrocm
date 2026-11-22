#[path = "wopr_dense_support/mod.rs"]
mod staging_support;
// Chronological explicit token-sequence diagnostic; no generation or service.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm_inkling2::{config::MiniConfig, text_graph::TextGraph};
use nextrocm_tok::Tokenizer;
use std::{
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg("--").arg(p).output()?;
    if !o.status.success() {
        return Err("hash command failed".into());
    }
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("hash missing")?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn available() -> Result<u64> {
    let s = fs::read_to_string("/proc/meminfo")?;
    let n = s
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|l| l.split_whitespace().next())
        .ok_or("MemAvailable missing")?
        .parse::<u64>()?;
    Ok(n.checked_mul(1024).ok_or("memory overflow")?)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 15 {
        return Err("usage: wopr_dense_sequence GGUF CONFIG GPU NEW_OUT tokens COMMA_IDS 0 MODE CACHE_BYTES EMBEDDING EXPECTED_RUNTIME_SHA CHUNK BATCH_BOOL DENSE_BOOL; NEXTROCM_LIB required".into());
    }
    let chunk_size: usize = a[12].parse()?;
    let batch_experts: bool = a[13].parse()?;
    let dense_batch: bool = a[14].parse()?;
    if dense_batch && !batch_experts {
        return Err("dense batching requires expert batching".into());
    }
    if ![1, 2, 4].contains(&chunk_size) {
        return Err("chunk must be1,2,4".into());
    }
    let mode = staging_support::parse(&a[8], &a[9])?;
    if mode == nextrocm_llm_inkling2::expert_cache::ExpertMode::Stream {
        return Err(
            "this grouped-callback diagnostic requires pair or cache staging, not Stream".into(),
        );
    }
    let embedding = match a[10].as_str() {
        "resident" => nextrocm_llm_inkling2::device::EmbeddingMode::Resident,
        "host" => nextrocm_llm_inkling2::device::EmbeddingMode::HostRow,
        _ => return Err("embedding must be resident or host".into()),
    };
    let max_new: usize = a[7].parse()?;
    if max_new != 0 {
        return Err("sequence diagnostic requires zero generated tokens".into());
    }
    let model = fs::canonicalize(&a[1])?;
    let cfg = fs::canonicalize(&a[2])?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let runtime = fs::canonicalize(PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB required")?,
    ))?;
    if !matches!(
        a[11].as_str(),
        "14f11de932424e7bf9fa22481382d632fdb26d20d97199a548f56822ef541ef2"
    ) {
        return Err("runtime SHA is not an approved comparison build".into());
    }
    let pins = [hash(&cfg)?, hash(&exe)?, hash(&runtime)?];
    if pins[2] != a[11] {
        return Err("runtime pin differs".into());
    }
    let meta = fs::metadata(&model)?;
    let identity = (meta.len(), meta.modified()?);
    let config = MiniConfig::from_official_json(&fs::read(&cfg)?)?;
    let header = Gguf::open(&model)?;
    let tokenizer = Tokenizer::from_gguf(&header)?;
    if tokenizer.eos != Some(200006) || tokenizer.stop != [200006] {
        return Err("Inkling exact EOS contract required".into());
    }
    if a[5] != "tokens" {
        return Err("sequence mode must be tokens".into());
    }
    let input: Vec<u32> = a[6]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    let prompt: Option<String> = None;
    if input.is_empty() || input.len() > 128 || input.iter().any(|&t| t >= 200058 || t == 200054) {
        return Err("input must contain1..128 text-only tokens".into());
    }
    let context = input
        .len()
        .checked_add(max_new)
        .ok_or("context overflow")?
        .max(chunk_size);
    let out = PathBuf::from(&a[4]);
    fs::create_dir(&out)?;
    fs::write(
        out.join("input.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"ids":input,"rendered_prompt":prompt,"max_new_tokens":max_new,"sampler":"greedy, ties lowerID","mode":a[5]}),
        )?,
    )?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(a[3].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let ptr = api()?.symbol("nr_allocation_stats");
    if ptr.is_null() {
        return Err("allocation stats missing".into());
    }
    // SAFETY: exact private ns stats ABI, permanent runtime and live GPU.
    let stats: Stats = unsafe { std::mem::transmute(ptr) };
    let stat = || -> Result<(u64, u64)> {
        let (mut live, mut peak) = (0, 0);
        if unsafe { stats(gpu.raw(), &mut live, &mut peak) } != 0 {
            return Err("stats failed".into());
        }
        Ok((live, peak))
    };
    if stat()?.0 != 0 {
        return Err("diagnostic requires empty device".into());
    }
    let mut files = serde_json::Map::new();
    let mut timings = Vec::new();
    let mut batch_statistics = serde_json::Value::Null;
    let mut dense_statistics = serde_json::Value::Null;
    let generated: Vec<u32> = Vec::new();
    let mut load_seconds = None;
    let start = Instant::now();
    let result = (|| -> Result<()> {
        let begin = Instant::now();
        let mut graph = TextGraph::load_with_dense_prefill_policies(
            &model,
            &config,
            &gpu,
            context,
            available()?,
            mode,
            embedding,
            chunk_size,
            batch_experts,
            dense_batch,
            || Ok(()),
        )?;
        load_seconds = Some(begin.elapsed().as_secs_f64());
        println!(
            "loaded {:.6}s, input_tokens={}",
            load_seconds.unwrap(),
            input.len()
        );
        for (chunk_index, ids) in input.chunks(chunk_size).enumerate() {
            let base = chunk_index * chunk_size;
            for row in 0..ids.len() {
                fs::create_dir(out.join(format!("token-{:03}", base + row)))?;
            }
            let mut visits = vec![0usize; ids.len()];
            let begin = Instant::now();
            let mut observe = |row: usize,
                               visit: &nextrocm_llm_inkling2::plan::Visit,
                               b: &nextrocm_engine_core::DevBuf|
             -> nextrocm_llm_inkling2::Result<()> {
                if row >= ids.len() || visit.ordinal != visits[row] || b.len != 2304 * 4 {
                    return Err("chunk visit order/index/shape changed".into());
                }
                let mut raw = vec![0u8; b.len];
                b.read(0, &mut raw).map_err(|e| e.0)?;
                if raw
                    .chunks_exact(4)
                    .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
                {
                    return Err("nonfinite visit".into());
                }
                let position = base + row;
                let name = format!(
                    "visit-{:02}-physical-{:02}.f32",
                    visit.ordinal, visit.physical_layer
                );
                let path = out.join(format!("token-{position:03}")).join(&name);
                fs::write(&path, raw).map_err(|e| e.to_string())?;
                files.insert(format!("token-{position:03}/{name}"),serde_json::json!({"sha256":hash(&path).map_err(|e|e.to_string())?,"elements":2304,"shape":[2304],"physical_layer":visit.physical_layer,"core_pass":visit.core_pass}));
                visits[row] += 1;
                Ok(())
            };
            // chunk1 deliberately exercises the unchanged serial entrypoint.
            let rows = if chunk_size == 1 {
                vec![graph.token(ids[0], || Ok(()), |v, b| observe(0, v, b))?]
            } else {
                graph.chunk(ids, || Ok(()), &mut observe)?
            };
            if rows.len() != ids.len() || visits.iter().any(|&n| n != 48) {
                return Err("chunk capture row/visit count".into());
            }
            for (row, logits) in rows.iter().enumerate() {
                if logits.len() != 200058 || logits.iter().any(|x| !x.is_finite()) {
                    return Err("logits shape/nonfinite".into());
                }
                let position = base + row;
                let path = out.join(format!("token-{position:03}/logits.f32"));
                fs::write(
                    &path,
                    logits
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                files.insert(
                    format!("token-{position:03}/logits.f32"),
                    serde_json::json!({"sha256":hash(&path)?,"elements":200058,"shape":[200058]}),
                );
            }
            let seconds = begin.elapsed().as_secs_f64();
            timings.push(seconds);
            println!(
                "chunk {chunk_index} start={base} rows={} visits_per_row=48 seconds={seconds:.6}",
                ids.len()
            );
        }
        staging_support::validate_batch(
            graph.batch_stats(),
            graph.expert_stats(),
            input.len(),
            chunk_size,
            batch_experts,
            0,
        )?;
        staging_support::validate_dense(
            graph.dense_stats().family_calls,
            graph.dense_stats().logical_vectors,
            input.len(),
            chunk_size,
            dense_batch,
        )?;
        dense_statistics = staging_support::dense_stats(
            graph.dense_stats().family_calls,
            graph.dense_stats().logical_vectors,
        );
        batch_statistics = staging_support::batch_stats(graph.batch_stats());
        drop(graph);
        gpu.sync()?;
        Ok(())
    })();
    // Preserve cleanup/identity errors in the report, rather than losing the
    // numerical capture on a post-execution GPU synchronization failure.
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
    let unchanged = match (|| -> Result<bool> {
        let after = fs::metadata(&model)?;
        Ok((after.len(), after.modified()?) == identity
            && [hash(&cfg)?, hash(&exe)?, hash(&runtime)?] == pins)
    })() {
        Ok(v) => v,
        Err(e) => {
            post_errors.push(e.to_string());
            false
        }
    };
    let complete = result.is_ok()
        && post_errors.is_empty()
        && allocation.is_some_and(|(live, peak)| live == 0 && peak <= 4_026_531_840)
        && unchanged;
    let report = serde_json::json!({"status":if complete{"DIAGNOSTIC_COMPLETE"}else{"INCOMPLETE"},"numeric_validation_pass":false,"embedding_mode":format!("{embedding:?}"),"expert_mode":format!("{mode:?}"),"model_admission":false,"scope":"Native48-visit per-token complete chunked explicit-token sequence, generation disabled; F32 same-quantized-weights baseline, no full-model original numerical gate yet","error":result.as_ref().err().map(ToString::to_string),"post_errors":post_errors,"model":model,"model_bytes":identity.0,"config":cfg,"config_sha256":pins[0],"executable":exe,"executable_sha256":pins[1],"runtime":runtime,"runtime_sha256":pins[2],"input_ids":input,"generated_ids":generated,"generated_text":tokenizer.decode(&generated),"context":context,"files":files,"load_seconds":load_seconds,"chunk_seconds":timings,"chunk_size":chunk_size,"batch_experts":batch_experts,"dense_batch":dense_batch,"dense_statistics":dense_statistics,"batch_statistics":batch_statistics,"seconds":start.elapsed().as_secs_f64(),"allocation_after_drop":allocation,"identities_unchanged":unchanged,"timing_scope":"includes48 synchronized per-visit readbacks and artifact hashing; not throughput benchmark"});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !complete {
        return Err("full text diagnostic incomplete; see report".into());
    }
    println!("DIAGNOSTIC_COMPLETE numerical validation pending");
    Ok(())
}
