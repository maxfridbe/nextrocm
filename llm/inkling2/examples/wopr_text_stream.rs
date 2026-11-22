//! Explicit 4GiB-device diagnostic: existing adapter with no device expert cache.
//! Buffered file reads may hit Linux page cache; counters distinguish logical
//! reads from storage I/O. No anonymous model mirror, math or default changes.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::{LoadOptions, Sampler};
use nextrocm_llm_inkling2::{adapter, output::TextOutput};
use nextrocm_tok::{inkling2_chat, Tokenizer};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap, ffi::c_void, fs, io::Write, path::Path, sync::Arc, time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MODEL: &str = "f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1";
const CAP: u64 = 4_026_531_840;
const HOST_RESERVE: u64 = 8 << 30;

fn hash(path: &Path) -> Result<String> {
    let output = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err("sha256sum failed".into());
    }
    let text = String::from_utf8(output.stdout)?;
    let value = text.split_whitespace().next().ok_or("missing SHA256")?;
    if value.len() != 64 || !value.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid SHA256".into());
    }
    Ok(value.into())
}
fn available() -> Result<u64> {
    let text = fs::read_to_string("/proc/meminfo")?;
    let kb = text
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|s| s.split_whitespace().next())
        .ok_or("MemAvailable absent")?
        .parse::<u64>()?;
    kb.checked_mul(1024)
        .ok_or_else(|| "MemAvailable overflow".into())
}
fn counters() -> Result<BTreeMap<String, u64>> {
    fs::read_to_string("/proc/self/io")?
        .lines()
        .map(|line| {
            let (name, value) = line.split_once(':').ok_or("bad proc I/O")?;
            Ok((name.into(), value.trim().parse()?))
        })
        .collect()
}
fn delta(before: &BTreeMap<String, u64>, after: &BTreeMap<String, u64>) -> Result<Value> {
    let mut values = BTreeMap::new();
    for key in ["rchar", "syscr", "read_bytes"] {
        let n = after
            .get(key)
            .ok_or("missing proc I/O counter")?
            .checked_sub(*before.get(key).ok_or("missing prior proc I/O counter")?)
            .ok_or("proc I/O counter decreased")?;
        values.insert(key, n);
    }
    Ok(json!(values))
}
fn stats(gpu: &Gpu) -> Result<(u64, u64)> {
    gpu.sync()?;
    let runtime = api()?;
    let ptr = runtime.symbol("nr_allocation_stats");
    if ptr.is_null() {
        return Err("allocation telemetry absent".into());
    }
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: exact published ABI, retained runtime and live GPU handle.
    let call: Stats = unsafe { std::mem::transmute(ptr) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: writable initialized output counters and live native context.
    if unsafe { call(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err(runtime.error().into());
    }
    if peak > CAP {
        return Err("tracked allocation exceeds 3.75GiB suite cap".into());
    }
    Ok((current, peak))
}
struct Greedy;
impl Sampler for Greedy {
    fn sample(&mut self, logits: &[f32]) -> u32 {
        self.try_sample(logits).unwrap_or(u32::MAX)
    }
    fn try_sample(&mut self, logits: &[f32]) -> nextrocm_llm::Result<u32> {
        nextrocm_llm::sampling::full_vocab::sample(
            logits,
            0.0,
            1.0,
            &mut nextrocm_llm::sampling::Rng(1),
        )
        .map_err(|e| nextrocm_llm::Error(e.into()))
    }
}
fn run() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        return Err("usage: wopr_text_stream GGUF GPU PROMPT MAX_NEW RUNTIME_SHA256 NEW_OUT; NEXTROCM_LIB required".into());
    }
    let max_new = a[4].parse::<usize>()?;
    if !(1..=64).contains(&max_new) {
        return Err("MAX_NEW must be1..64".into());
    }
    let out = Path::new(&a[6]);
    fs::create_dir(out)?;
    let model = Path::new(&a[1]);
    let runtime = std::path::PathBuf::from(std::env::var("NEXTROCM_LIB")?);
    if !runtime.is_absolute() || !model.is_absolute() {
        return Err("absolute model/runtime required".into());
    }
    let executable = std::env::current_exe()?;
    let mut report = json!({"status":"INCOMPLETE","expert_mode":"Stream","expert_bytes":0,"context":512,
        "prompt":a[3],"max_new":max_new,"numeric_pass":null,"scope":"Explicit streaming adapter; no model mirror. One short greedy run; no general quality claim.",
        "model_sha256":MODEL,"runtime_sha256_expected":a[5],"runtime_path":runtime,"model_path":model,
        "page_cache_note":"Full model hashing before load reads all file bytes and can warm reclaimable Linux page cache. Phase /proc/self/io read_bytes is observed storage I/O, not guaranteed residency."});
    let mut gpu_owner: Option<Arc<Gpu>> = None;
    let mut errors = Vec::new();
    let mut exe_before = None;
    let result = (|| -> Result<()> {
        if available()? < HOST_RESERVE {
            return Err("host reserve below8GiB".into());
        }
        let before_hash_io = counters()?;
        if hash(model)? != MODEL || hash(&runtime)? != a[5] {
            return Err("model/runtime pin mismatch".into());
        }
        exe_before = Some(hash(&executable)?);
        report["executable_sha256"] = json!(exe_before);
        report["io_before_hash"] = json!(before_hash_io);
        report["io_after_hash"] = json!(counters()?);
        // Original tokenizer/template and native text-only guards; no custom prompt prefix.
        let file = Gguf::open(model).map_err(|e| e.0)?;
        let tok = Tokenizer::from_gguf(&file)?;
        let rendered = inkling2_chat(&[json!({"role":"user","content":a[3]})], None, &[])?;
        let ids = tok.encode(&rendered);
        if ids.is_empty() || ids.len() + max_new > 512 || ids.contains(&200054) {
            return Err("prompt outside text-only context512 bound".into());
        }
        report["input_ids"] = json!(ids);
        let gpu = Gpu::open(a[2].parse()?)?;
        gpu_owner = Some(gpu.clone());
        if stats(&gpu)?.0 != 0 {
            return Err("initial tracked allocations nonzero".into());
        }
        let stop = || available().map_or(true, |n| n < HOST_RESERVE);
        let mut log = |s: String| eprintln!("{s}");
        let io_load = counters()?;
        let start = Instant::now();
        let engine = adapter::load(
            &file,
            &[gpu.clone()],
            &LoadOptions {
                expert_bytes: Some(0),
                kv: (512, 1),
                ..Default::default()
            },
            &mut log,
        )?;
        let load_seconds = start.elapsed().as_secs_f64();
        report["load"] = json!({"seconds":load_seconds,"io_delta":delta(&io_load,&counters()?)?});
        println!("load_seconds={load_seconds:.6}");
        std::io::stdout().flush()?;
        if stop() {
            return Err("host reserve below8GiB afterload".into());
        }
        let mut session = engine.session(512)?;
        let io_pp = counters()?;
        let start = Instant::now();
        let (fed, logits) =
            engine.feed_until(&mut session, &ids, &|_| stop(), &mut |_, _| Ok(()))?;
        if fed != ids.len() {
            return Err("prefill stopped before all tokens".into());
        }
        let pp_seconds = start.elapsed().as_secs_f64();
        report["prefill"] = json!({"tokens":ids.len(),"seconds":pp_seconds,"tokens_per_second":ids.len() as f64/pp_seconds,"io_delta":delta(&io_pp,&counters()?)?});
        println!("prefill_seconds={pp_seconds:.6}");
        std::io::stdout().flush()?;
        let mut decoder = engine.decoder(logits, false);
        decoder.set_context(&ids);
        let mut sampler = Greedy;
        let mut parser = TextOutput::new();
        let mut generated = Vec::new();
        let (mut content, mut reasoning) = (String::new(), String::new());
        let mut done = false;
        let io_tg = counters()?;
        let start = Instant::now();
        while generated.len() < max_new && !done {
            let tokens = engine.step_until(
                &mut session,
                &mut decoder,
                &mut sampler,
                &stop,
                &mut |_, _| Ok(()),
            )?;
            if tokens.len() != 1 {
                return Err("expected one non-draft token".into());
            }
            let id = tokens[0];
            generated.push(id);
            let part = parser.push_token(&tok, id)?;
            content.push_str(&part.content);
            reasoning.push_str(&part.reasoning);
            done = part.done;
        }
        let mut incomplete = false;
        if !done {
            let part = parser.finish_length()?;
            content.push_str(&part.content);
            reasoning.push_str(&part.reasoning);
            incomplete = part.incomplete;
        }
        let tg_seconds = start.elapsed().as_secs_f64();
        report["generation"] = json!({"tokens_including_eos":generated.len(),"ids":generated,
            "seconds":tg_seconds,"tokens_per_second_including_eos":generated.len() as f64/tg_seconds,
            "io_delta":delta(&io_tg,&counters()?)?,"content":content,"reasoning":reasoning,"eos":done,"incomplete":incomplete,
            "scope":"Sampling/parser plus native steps; first sample uses prefill logits, final emitted token need not be forwarded. No independent TTFT measurement."});
        fs::write(out.join("answer.txt"), &content)?;
        fs::write(out.join("reasoning.txt"), &reasoning)?;
        println!("generation_seconds={tg_seconds:.6}\nanswer={content}\nreasoning={reasoning}");
        drop(decoder);
        drop(session);
        drop(engine);
        report["host_available_after"] = json!(available()?);
        if available()? < HOST_RESERVE {
            return Err("host reserve below8GiB afterrun".into());
        }
        Ok(())
    })();
    if let Err(e) = result {
        errors.push(e.to_string());
    }
    if let Some(gpu) = gpu_owner.as_ref() {
        match stats(gpu) {
            Ok((live, peak)) => {
                report["after_drop"] = json!({"live":live,"peak":peak});
                if live != 0 {
                    errors.push("tracked allocation leak".into());
                }
            }
            Err(e) => errors.push(e.to_string()),
        }
    }
    for (name, path, expected) in [
        ("model", model, Some(MODEL.to_string())),
        ("runtime", runtime.as_path(), Some(a[5].clone())),
        ("executable", executable.as_path(), exe_before),
    ] {
        match hash(path) {
            Ok(value) => {
                report[format!("{name}_sha256_after")] = json!(value);
                if expected.as_ref() != Some(&value) {
                    errors.push(format!("{name} identity changed/unverified"));
                }
            }
            Err(e) => errors.push(e.to_string()),
        }
    }
    report["errors"] = json!(errors);
    report["status"] = json!(if errors.is_empty() {
        "DIAGNOSTIC_COMPLETE"
    } else {
        "INCOMPLETE"
    });
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; ").into())
    }
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL wopr_text_stream: {e}");
        std::process::exit(1);
    }
}
