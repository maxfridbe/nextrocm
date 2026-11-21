//! Full48-visit diagnostic capture and bounded greedy decoding, not a service.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm_inkling2::{config::MiniConfig, text_graph::TextGraph};
use nextrocm_tok::{inkling2_chat, Tokenizer};
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
    if a.len() != 8 {
        return Err("usage: inspect_text GGUF CONFIG GPU NEW_OUT token|prompt INPUT MAX_NEW_TOKENS; NEXTROCM_LIB required".into());
    }
    let max_new: usize = a[7].parse()?;
    if max_new > 8 {
        return Err("diagnostic allows0..8 new tokens".into());
    }
    let model = fs::canonicalize(&a[1])?;
    let cfg = fs::canonicalize(&a[2])?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let runtime = fs::canonicalize(PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB required")?,
    ))?;
    let pins = [hash(&cfg)?, hash(&exe)?, hash(&runtime)?];
    let meta = fs::metadata(&model)?;
    let identity = (meta.len(), meta.modified()?);
    let config = MiniConfig::from_official_json(&fs::read(&cfg)?)?;
    let header = Gguf::open(&model)?;
    let tokenizer = Tokenizer::from_gguf(&header)?;
    if tokenizer.eos != Some(200006) || tokenizer.stop != [200006] {
        return Err("Inkling exact EOS contract required".into());
    }
    let (input, prompt) = match a[5].as_str() {
        "token" => (vec![a[6].parse::<u32>()?], None),
        "prompt" => {
            let rendered = inkling2_chat(
                &[serde_json::json!({"role":"user","content":a[6]})],
                None,
                &[],
            )?;
            (tokenizer.encode(&rendered), Some(rendered))
        }
        _ => return Err("mode must be token or prompt".into()),
    };
    if input.is_empty() || input.len() > 128 || input.iter().any(|&t| t >= 200058 || t == 200054) {
        return Err("input must contain1..128 text-only tokens".into());
    }
    let context = input.len().checked_add(max_new).ok_or("context overflow")?;
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
    let mut generated = Vec::new();
    let mut load_seconds = None;
    let start = Instant::now();
    let result = (|| -> Result<()> {
        let begin = Instant::now();
        let mut graph = TextGraph::load(&model, &config, &gpu, context, available()?, || Ok(()))?;
        load_seconds = Some(begin.elapsed().as_secs_f64());
        println!(
            "loaded {:.6}s, input_tokens={}",
            load_seconds.unwrap(),
            input.len()
        );
        let mut evaluate = |id: u32, position: usize, graph: &mut TextGraph| -> Result<Vec<f32>> {
            let dir = out.join(format!("token-{position:03}"));
            fs::create_dir(&dir)?;
            let begin = Instant::now();
            let mut visits = 0usize;
            let logits=graph.token(id,||Ok(()),|visit,b|{
    if visit.ordinal!=visits||b.len!=2304*4{return Err("visit order/output shape changed".into())}
    let mut raw=vec![0u8;b.len];b.read(0,&mut raw).map_err(|e|e.0)?;
    if raw.chunks_exact(4).any(|x|!f32::from_le_bytes(x.try_into().unwrap()).is_finite()){return Err(format!("nonfinite visit{}",visit.ordinal))}
    let name=format!("visit-{:02}-physical-{:02}.f32",visit.ordinal,visit.physical_layer);let path=dir.join(&name);
    fs::write(&path,raw).map_err(|e|e.to_string())?;files.insert(format!("token-{position:03}/{name}"),serde_json::json!({"sha256":hash(&path).map_err(|e|e.to_string())?,"elements":2304,"shape":[2304],"physical_layer":visit.physical_layer,"core_pass":visit.core_pass}));visits+=1;Ok(())
   })?;
            if visits != 48 || logits.len() != 200058 || logits.iter().any(|x| !x.is_finite()) {
                return Err("full graph capture shape/nonfinite".into());
            }
            let path = dir.join("logits.f32");
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
            let seconds = begin.elapsed().as_secs_f64();
            timings.push(seconds);
            println!("token {position} id={id} visits=48 seconds={seconds:.6}");
            Ok(logits)
        };
        let mut logits = Vec::new();
        for (i, &id) in input.iter().enumerate() {
            logits = evaluate(id, i, &mut graph)?;
        }
        for i in 0..max_new {
            let mut best = 0usize;
            for j in 1..logits.len() {
                if logits[j] > logits[best] {
                    best = j;
                }
            }
            generated.push(best as u32);
            println!("greedy {i}: {best}");
            if tokenizer.stop.contains(&(best as u32)) || i + 1 == max_new {
                break;
            }
            logits = evaluate(best as u32, input.len() + i, &mut graph)?;
        }
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
        && allocation.is_some_and(|(live, peak)| live == 0 && peak < 7_500_000_000)
        && unchanged;
    let report = serde_json::json!({"status":if complete{"DIAGNOSTIC_COMPLETE"}else{"INCOMPLETE"},"numeric_validation_pass":false,"model_admission":false,"scope":"Native48-visit text graph, bounded greedy only; F32 same-quantized-weights baseline, no full-model original numerical gate yet","error":result.as_ref().err().map(ToString::to_string),"post_errors":post_errors,"model":model,"model_bytes":identity.0,"config":cfg,"config_sha256":pins[0],"executable":exe,"executable_sha256":pins[1],"runtime":runtime,"runtime_sha256":pins[2],"input_ids":input,"generated_ids":generated,"generated_text":tokenizer.decode(&generated),"context":context,"files":files,"load_seconds":load_seconds,"token_seconds":timings,"seconds":start.elapsed().as_secs_f64(),"allocation_after_drop":allocation,"identities_unchanged":unchanged,"timing_scope":"includes48 synchronized per-visit readbacks and artifact hashing; not throughput benchmark"});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !complete {
        return Err("full text diagnostic incomplete; see report".into());
    }
    println!("DIAGNOSTIC_COMPLETE numerical validation pending");
    Ok(())
}
