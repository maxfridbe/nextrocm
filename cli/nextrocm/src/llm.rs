//! LLM commands split out of nextcuda's cli/nextcuda/src/main.rs and the
//! native branch of service.rs. Uses the existing Engine/Server/control socket.
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::{ChatTemplate, Engine, EngineKind, LoadOptions};
use nextrocm_models::{registry, Config};
use nextrocm_serve::{
    cache::PromptCache,
    http::{self, Target},
    Server,
};
use nextrocm_tok::{Effort, Message, Tokenizer};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

const USAGE: &str = "nextrocm llm engines | info MODEL | tokenize MODEL TEXT | prepare MODEL
nextrocm llm generate MODEL --prompt TEXT | --prompt-file FILE | --ids FILE
    [--max N] [--temp T] [--top-p P] [--seed N] [--gpu N] [--ctx N] [--ignore-eos]
nextrocm llm serve MODEL_OR_ID [--host 127.0.0.1] [--port 8085] [--ctx N[,M...]]
    [--parallel N] [--name ID] [--socket PATH] [--prompt-cache-mib N] [--cache-dir DIR]
nextrocm llm start ID | --model FILE [serve options] | stop | logs [--no-follow]";

pub(crate) fn engines() -> Vec<EngineKind> {
    vec![
        nextrocm_llm_stablelm::kind(),
        nextrocm_llm_qwen35::kind(),
        nextrocm_llm_qwen3moe::kind(),
        nextrocm_llm_inkling2::kind(),
    ]
}
fn opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}
fn number<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> Result<T, String> {
    opt(args, key).map_or(Ok(default), |s| {
        s.parse()
            .map_err(|_| format!("{key}: invalid number {s:?}"))
    })
}
fn validate_options(args: &[String], values: &[&str], flags: &[&str]) -> Result<(), String> {
    let mut it = args.iter();
    while let Some(key) = it.next() {
        if flags.contains(&key.as_str()) {
            continue;
        }
        if !values.contains(&key.as_str()) {
            return Err(format!("unsupported option {key:?}"));
        }
        match it.next() {
            Some(v) if !v.starts_with("--") => {}
            _ => return Err(format!("{key} requires a value")),
        }
    }
    Ok(())
}

fn no_tap(_: &str, _: &DevBuf) -> nextrocm_engine_core::Result<()> {
    Ok(())
}
fn options(k: &EngineKind) -> Vec<nextrocm_engine_core::EngineOption> {
    nextrocm_llm::COMMON_OPTIONS
        .iter()
        .chain(k.options)
        .copied()
        .collect()
}
fn load_engine<'g>(
    f: &'g Gguf,
    gpus: &[Arc<Gpu>],
    o: LoadOptions,
) -> Result<Box<dyn Engine + 'g>, String> {
    let kinds = engines();
    let k = nextrocm_llm::kind_for(&kinds, f)?;
    for (var, value) in crate::opt_env(&options(k), nextrocm_engine_core::At::Load, k.name)? {
        std::env::set_var(var, value);
    }
    (k.load)(f, gpus, &o, &mut |line| eprintln!("[{line}]")).map_err(|e| e.0)
}
pub(crate) fn chat_of(f: &Gguf) -> Result<ChatTemplate, String> {
    if f.architecture() == nextrocm_llm_stablelm::ARCH {
        nextrocm_llm_stablelm::chat_for(f).map(ChatTemplate::Legacy)
    } else if matches!(f.architecture(), "qwen35" | "qwen35moe") {
        nextrocm_tok::qwen35_chat_for(f).map(ChatTemplate::Legacy)
    } else {
        Ok(nextrocm_llm::kind_for(&engines(), f)?.chat)
    }
}
fn context_limit(f: &Gguf) -> Result<usize, String> {
    let kinds = engines();
    let kind = nextrocm_llm::kind_for(&kinds, f)?;
    (kind.info)(f)?;
    f.arch_meta("context_length")
        .and_then(|v| v.as_u64())
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .map(|v| {
            if f.architecture() == "inkling2" {
                v.min(4096)
            } else {
                v
            }
        })
        .ok_or_else(|| "missing or invalid trained context length".to_owned())
}
fn validate_inkling_inputs(arch: &str, ids: &[u32], contexts: &[usize]) -> Result<(), String> {
    if arch == "inkling2"
        && (ids.iter().any(|id| *id >= 200058 || *id == 200054)
            || contexts.len() != 1
            || contexts[0] == 0
            || contexts[0] > 4096)
    {
        return Err("Inkling requires text token IDs, one session, and context1..4096".into());
    }
    Ok(())
}
struct CliSampler<'a> {
    full_vocab: bool,
    temp: f32,
    top_p: f32,
    rng: &'a mut nextrocm_llm::sampling::Rng,
}
impl nextrocm_llm::Sampler for CliSampler<'_> {
    fn sample(&mut self, row: &[f32]) -> u32 {
        // Legacy engines use the infallible method; Inkling uses try_sample.
        self.try_sample(row).unwrap_or(u32::MAX)
    }
    fn try_sample(&mut self, row: &[f32]) -> nextrocm_llm::Result<u32> {
        if self.full_vocab {
            nextrocm_llm::sampling::full_vocab::sample(row, self.temp, self.top_p, self.rng)
                .map_err(|e| nextrocm_llm::Error(e.into()))
        } else {
            Ok(nextrocm_llm::sampling::sample(
                row, self.temp, self.top_p, self.rng,
            ))
        }
    }
}
/// Registry's existing Describe callback; no GPU is opened.
pub(crate) fn describe(f: &Gguf) -> Result<String, String> {
    let kinds = engines();
    let kind = nextrocm_llm::kind_for(&kinds, f)?;
    (kind.info)(f)?;
    chat_of(f)?;
    Tokenizer::from_gguf(f)?;
    Ok(kind.name.to_string())
}

fn info(path: &Path) -> Result<(), String> {
    let f = Gguf::open(path).map_err(|e| e.0)?;
    println!(
        "file     : {} ({} files, {:.2} GiB of tensors, {} tensors)",
        path.display(),
        f.paths.len(),
        f.total_bytes() as f64 / (1u64 << 30) as f64,
        f.tensors.len()
    );
    let kinds = engines();
    println!("{}", (nextrocm_llm::kind_for(&kinds, &f)?.info)(&f)?);
    Ok(())
}
fn tokenize(path: &Path, text: &str) -> Result<(), String> {
    let f = Gguf::open(path).map_err(|e| e.0)?;
    let tok = Tokenizer::from_gguf(&f)?;
    let ids = tok.encode(text);
    println!("{} tokens: {ids:?}", ids.len());
    for id in &ids {
        println!("  {id:>7} {:?}", tok.decode(&[*id]));
    }
    println!(
        "round trip: {}",
        if tok.decode(&ids) == text {
            "identical"
        } else {
            "DIFFERENT (the tokenizer may normalize text)"
        }
    );
    Ok(())
}
fn gpu(args: &[String]) -> Result<Vec<Arc<Gpu>>, String> {
    let asked = opt(args, "--gpu").unwrap_or("0");
    // Current ROCm engines use one GPU; do not silently ignore other IDs.
    let id = if asked == "all" {
        let available = nextrocm_engine_core::gpus().map_err(|e| e.0)?;
        if available.len() != 1 {
            return Err("this ROCm backend is single-GPU; choose one explicit --gpu index".into());
        }
        available[0].0
    } else {
        asked
            .parse::<usize>()
            .map_err(|_| "--gpu requires one GPU index".to_string())?
    };
    Ok(vec![Gpu::open(id).map_err(|e| e.0)?])
}
fn prompt_ids(f: &Gguf, tok: &Tokenizer, args: &[String]) -> Result<Vec<u32>, String> {
    if let Some(path) = opt(args, "--ids") {
        return std::fs::read_to_string(path)
            .map_err(|e| e.to_string())?
            .split([',', ' ', '\n', '\r', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse()
                    .map_err(|_| format!("{path}: invalid token id {s:?}"))
            })
            .collect();
    }
    let prompt = match opt(args, "--prompt-file") {
        Some(path) => std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?,
        None => opt(args, "--prompt")
            .ok_or("--prompt TEXT, --prompt-file FILE or --ids FILE")?
            .to_owned(),
    };
    if let ChatTemplate::Checked(format) = chat_of(f)? {
        let mut request = serde_json::json!({"messages":[{"role":"user","content":prompt}]});
        if let Some(effort) = opt(args, "--effort") {
            request["reasoning_effort"] = serde_json::Value::String(effort.to_owned());
        }
        return Ok(tok.encode(&format(&request)?));
    }
    let effort =
        Effort::parse(opt(args, "--effort").unwrap_or("default")).ok_or("invalid --effort")?;
    let ChatTemplate::Legacy(format) = chat_of(f)? else {
        return Err("chat template changed during formatting".into());
    };
    Ok(tok.encode(&format(
        &[Message {
            role: "user",
            content: &prompt,
            ..Default::default()
        }],
        effort,
        &[],
    )))
}
fn generate(args: &[String]) -> Result<(), String> {
    validate_options(
        args.get(2..).unwrap_or(&[]),
        &[
            "--prompt",
            "--prompt-file",
            "--ids",
            "--max",
            "--temp",
            "--top-p",
            "--seed",
            "--gpu",
            "--ctx",
            "--effort",
        ],
        &["--ignore-eos", "--no-mtp"],
    )?;
    let model = args.get(1).ok_or(USAGE)?;
    let f = Gguf::open(Path::new(model)).map_err(|e| e.0)?;
    let trained_context = context_limit(&f)?;
    let tok = Tokenizer::from_gguf(&f)?;
    let ids = prompt_ids(&f, &tok, args)?;
    if ids.is_empty() {
        return Err("empty prompt".into());
    }
    let max = number(args, "--max", 256usize)?;
    let required = ids.len().checked_add(max).ok_or("context size overflow")?;
    let ctx = number(args, "--ctx", required)?;
    if ctx < required || ctx > trained_context {
        return Err(format!(
            "--ctx must fit prompt + --max ({required}) and trained context ({})",
            trained_context
        ));
    }
    let inkling = f.architecture() == "inkling2";
    if inkling && args.iter().any(|a| a == "--ignore-eos") {
        return Err("Inkling's structured output requires honoring EOS".into());
    }
    let temp = number(args, "--temp", if inkling { 1.0f32 } else { 0.0 })?;
    let top_p = number(args, "--top-p", if inkling { 1.0f32 } else { 0.95 })?;
    if !temp.is_finite()
        || temp < 0.0
        || !top_p.is_finite()
        || !(0.0..=1.0).contains(&top_p)
        || top_p == 0.0
    {
        return Err("invalid temperature or top-p".into());
    }
    let seed = number(args, "--seed", 1u64)?;
    let mut rng = nextrocm_llm::sampling::Rng(seed.max(1));
    validate_inkling_inputs(f.architecture(), &ids, &[ctx])?;
    let gs = gpu(args)?;
    let eng = load_engine(
        &f,
        &gs,
        LoadOptions {
            draft: false,
            kv: (ctx, 1),
            ..Default::default()
        },
    )?;
    let mut session = eng.session(ctx).map_err(|e| e.0)?;
    let start = Instant::now();
    let logits = eng.feed(&mut session, &ids, &mut no_tap).map_err(|e| e.0)?;
    let prefill = start.elapsed().as_secs_f64();
    let mut dec = eng.decoder(logits, false);
    dec.set_context(&ids);
    let mut draw = CliSampler {
        full_vocab: inkling,
        temp,
        top_p,
        rng: &mut rng,
    };
    let mut structured = inkling.then(nextrocm_llm_inkling2::output::TextOutput::new);
    let mut structured_done = false;
    let mut pending = Vec::new();
    let mut out = std::io::stdout().lock();
    let mut n = 0;
    let start = Instant::now();
    'generate: while n < max {
        let tokens = eng
            .step(&mut session, &mut dec, &mut draw, &mut no_tap)
            .map_err(|e| e.0)?;
        for id in tokens {
            if let Some(parser) = structured.as_mut() {
                let delta = parser.push_token(&tok, id)?;
                if !delta.reasoning.is_empty() {
                    eprint!("{}", delta.reasoning);
                }
                out.write_all(delta.content.as_bytes())
                    .map_err(|e| e.to_string())?;
                out.flush().map_err(|e| e.to_string())?;
                if delta.done {
                    structured_done = true;
                    break 'generate;
                }
                n += 1;
                continue;
            }
            if n >= max || (!args.iter().any(|a| a == "--ignore-eos") && tok.stop.contains(&id)) {
                break 'generate;
            }
            n += 1;
            pending.extend(tok.decode_bytes(&[id]));
            // Stream complete UTF-8, replacing invalid bytes while keeping an incomplete tail.
            loop {
                match std::str::from_utf8(&pending) {
                    Ok(s) => {
                        out.write_all(s.as_bytes()).map_err(|e| e.to_string())?;
                        pending.clear();
                        break;
                    }
                    Err(e) => {
                        let valid = e.valid_up_to();
                        out.write_all(&pending[..valid])
                            .map_err(|e| e.to_string())?;
                        pending.drain(..valid);
                        if let Some(bad) = e.error_len() {
                            out.write_all("�".as_bytes()).map_err(|e| e.to_string())?;
                            pending.drain(..bad);
                        } else {
                            break;
                        }
                    }
                }
            }
            out.flush().map_err(|e| e.to_string())?;
        }
    }
    if !structured_done {
        if let Some(parser) = structured.as_mut() {
            let delta = parser.finish_length()?;
            if !delta.reasoning.is_empty() {
                eprint!("{}", delta.reasoning);
            }
            out.write_all(delta.content.as_bytes())
                .map_err(|e| e.to_string())?;
        }
    }
    writeln!(out, "{}", String::from_utf8_lossy(&pending)).map_err(|e| e.to_string())?;
    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "[prompt {} tokens in {prefill:.2}s; {n} generated in {elapsed:.2}s ({:.2} tok/s)]",
        ids.len(),
        n as f64 / elapsed.max(1e-9)
    );
    for line in eng.report(n) {
        eprintln!("[{line}]");
    }
    Ok(())
}

/// The upstream in-process server, with bounded defaults for the selected engine.
fn serve_file(args: &[String]) -> Result<(), String> {
    validate_options(
        args.get(2..).unwrap_or(&[]),
        &[
            "--host",
            "--port",
            "--ctx",
            "--parallel",
            "--name",
            "--socket",
            "--gpu",
            "--effort",
            "--prompt-cache-mib",
            "--cache-dir",
            "--cache-disk-gib",
            "--cache-ttl-hours",
            "--keep-requests",
            "--max-tokens",
            "--cors",
        ],
        &["--no-mtp", "--here"],
    )?;
    let model = args.get(1).ok_or(USAGE)?;
    let f: &'static Gguf = Box::leak(Box::new(Gguf::open(Path::new(model)).map_err(|e| e.0)?));
    let trained_context = context_limit(f)?;
    let chat = chat_of(f)?;
    let tok = Tokenizer::from_gguf(f)?;
    let context = opt(args, "--ctx")
        .map(str::to_owned)
        .unwrap_or_else(|| trained_context.min(4096).to_string());
    let contexts: Vec<usize> = context
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|_| format!("invalid --ctx {context:?}"))
        })
        .collect::<Result<_, _>>()?;
    if contexts.is_empty() || contexts.iter().any(|c| *c < 256 || *c > trained_context) {
        return Err(format!(
            "--ctx must be 256..={} for each session",
            trained_context
        ));
    }
    let parallel = number(args, "--parallel", 1usize)?;
    if !(1..=8).contains(&parallel) {
        return Err("--parallel must be 1..=8".into());
    }
    let slots = if contexts.len() == 1 {
        vec![contexts[0]; parallel]
    } else {
        contexts
    };
    if slots.len() > 8 {
        return Err("at most 8 sessions".into());
    }
    validate_inkling_inputs(f.architecture(), &[], &slots)?;
    let reserve = slots
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or("KV reserve overflow")?;
    let cache_mib = number(args, "--prompt-cache-mib", 4096usize)?;
    let cache_bytes = cache_mib
        .checked_mul(1 << 20)
        .ok_or("prompt cache size overflow")?;
    let effort =
        Effort::parse(opt(args, "--effort").unwrap_or("default")).ok_or("invalid --effort")?;
    let keep = number(args, "--keep-requests", 100usize)?;
    let default_max = number(args, "--max-tokens", 0usize)?;
    let cors = opt(args, "--cors")
        .unwrap_or("")
        .split([',', ' '])
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    let gs = gpu(args)?;
    let eng = load_engine(
        f,
        &gs,
        LoadOptions {
            draft: false,
            kv: (reserve, slots.len()),
            ..Default::default()
        },
    )?;
    let mut cache = PromptCache::new(cache_bytes);
    if let Some(dir) = opt(args, "--cache-dir") {
        let gib = number(args, "--cache-disk-gib", 32f64)?;
        let hours = number(args, "--cache-ttl-hours", 24f64)?;
        if !gib.is_finite() || gib < 0.0 || !hours.is_finite() || hours < 0.0 {
            return Err("invalid disk cache budget or TTL".into());
        }
        let fp = format!(
            "{model} {} {}",
            std::fs::metadata(model).map_err(|e| e.to_string())?.len(),
            eng.cache_fingerprint()
        );
        cache = cache.with_disk(
            PathBuf::from(dir),
            (gib * (1u64 << 30) as f64) as usize,
            &fp,
            Duration::try_from_secs_f64(hours * 3600.0).map_err(|e| e.to_string())?,
        )?;
    }
    let name = opt(args, "--name").map(str::to_owned).unwrap_or_else(|| {
        Path::new(model)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned()
    });
    let server = Arc::new(Server::new(
        eng,
        tok,
        chat,
        name,
        slots,
        effort,
        cache,
        cors,
        keep,
        (default_max > 0).then_some(default_max),
    )?);
    let address = format!(
        "{}:{}",
        opt(args, "--host").unwrap_or("127.0.0.1"),
        opt(args, "--port").unwrap_or("8085")
    );
    server.run(&address, opt(args, "--socket").map(PathBuf::from))
}

/// Resolve the existing registry entry and its settings before native launch.
fn launch_args(cfg: &Config, raw: &[String]) -> Result<Vec<String>, String> {
    let mut flags = raw.to_vec();
    let mut env = Vec::new();
    if let Some(id) = raw
        .first()
        .filter(|s| !s.starts_with("--") && !Path::new(s).is_file())
    {
        let m = registry::find(cfg, id)?.ok_or_else(|| format!("no model {id}"))?;
        if m["enabled"] == false {
            return Err(format!("{id} is disabled"));
        }
        if registry::kind_of(&m) != "llm" {
            return Err(format!("{id} is not an LLM"));
        }
        crate::drives::not_moving(cfg, id)?;
        flags.remove(0);
        for (flag, key) in [
            ("--model", "file"),
            ("--ctx", "ctx"),
            ("--name", "id"),
            ("--gpu", "gpus"),
        ] {
            if opt(&flags, flag).is_none() {
                if let Some(v) = m[key].as_str().filter(|s| !s.is_empty()) {
                    flags.extend([flag.into(), v.into()]);
                }
            }
        }
        if let Some(settings) = m["env"].as_object() {
            env.extend(
                settings
                    .iter()
                    .filter_map(|(k, v)| v.as_str().map(|v| (k.clone(), v.to_owned()))),
            );
        }
    } else if raw.first().is_some_and(|s| Path::new(s).is_file()) {
        let path = flags.remove(0);
        flags.extend(["--model".into(), path]);
    }
    for (k, v) in env {
        std::env::set_var(k, v);
    }
    let model = opt(&flags, "--model")
        .map(str::to_owned)
        .or_else(|| cfg.get("NS_MODEL"))
        .ok_or("llm start ID or --model FILE")?;
    let f = Gguf::open(Path::new(&model)).map_err(|e| e.0)?;
    let trained_context = context_limit(&f)?;
    chat_of(&f)?;
    Tokenizer::from_gguf(&f)?;
    let mut out = vec!["serve".into(), model];
    let mut it = flags.iter();
    while let Some(v) = it.next() {
        if v == "--model" {
            it.next();
        } else {
            out.push(v.clone());
        }
    }
    let defaults = [
        ("--gpu", "NS_GPUS", "0".to_string()),
        ("--host", "NS_HOST", "127.0.0.1".into()),
        ("--port", "NS_PORT", "8085".into()),
        ("--ctx", "NS_CTX", trained_context.min(4096).to_string()),
        ("--parallel", "NS_PARALLEL", "1".into()),
        ("--effort", "NS_EFFORT", "default".into()),
        ("--prompt-cache-mib", "NS_PROMPT_CACHE_MIB", "4096".into()),
        ("--keep-requests", "NS_KEEP_REQUESTS", "100".into()),
    ];
    for (flag, key, default) in defaults {
        if opt(&out, flag).is_none() {
            out.extend([flag.into(), cfg.get(key).unwrap_or(default)]);
        }
    }
    for (flag, key) in [
        ("--max-tokens", "NS_MAX_TOKENS"),
        ("--cors", "NS_CORS"),
        ("--cache-dir", "NS_CACHE_DIR"),
        ("--cache-disk-gib", "NS_CACHE_DISK_GIB"),
        ("--cache-ttl-hours", "NS_CACHE_TTL_HOURS"),
    ] {
        if opt(&out, flag).is_none() {
            if let Some(v) = cfg.get(key) {
                out.extend([flag.into(), v]);
            }
        }
    }
    if opt(&out, "--socket").is_none() {
        out.extend([
            "--socket".into(),
            cfg.socket().to_string_lossy().into_owned(),
        ]);
    }
    Ok(out)
}
fn start(cfg: &Config, args: &[String]) -> Result<(), String> {
    if !crate::native::native(cfg) {
        return Err("this bounded ROCm LLM port uses the existing native process mode; unset NS_CONTAINER_ENGINE".into());
    }
    let prepared = launch_args(cfg, args)?;
    if crate::native::running(cfg, crate::container::SERVER) {
        return Err("LLM server already running".into());
    }
    let port = opt(&prepared, "--port").unwrap_or("8085");
    let target = Target::Unix(PathBuf::from(
        opt(&prepared, "--socket").ok_or("missing socket")?,
    ));
    let mut child = vec!["llm".into()];
    child.extend(prepared.clone());
    // The parent's --opt flags were stripped by main; forward their canonical form.
    for (name, value) in crate::engine_opts() {
        child.extend([format!("--opt-{name}"), value.clone()]);
    }
    crate::native::launch(
        cfg,
        crate::container::SERVER,
        port,
        &std::env::current_exe().map_err(|e| e.to_string())?,
        &child,
        false,
    )?;
    let start = Instant::now();
    loop {
        if http::call_for(&target, "GET", "/health", None, 4).is_ok() {
            println!("LLM server ready on port {port}");
            return Ok(());
        }
        if !crate::native::running(cfg, crate::container::SERVER) {
            return Err(format!(
                "LLM server ended while loading: {}",
                crate::native::last_line(cfg, crate::container::SERVER).unwrap_or_default()
            ));
        }
        if start.elapsed() > Duration::from_secs(900) {
            return Err("LLM server did not become ready in 900 seconds".into());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}
fn stop(cfg: &Config) -> Result<(), String> {
    if !crate::native::running(cfg, crate::container::SERVER) {
        println!("server: not running");
        return Ok(());
    }
    let _ = http::call(
        &Target::Unix(cfg.socket()),
        "POST",
        "/server/shutdown",
        None,
    );
    let start = Instant::now();
    while crate::native::running(cfg, crate::container::SERVER)
        && start.elapsed() < Duration::from_secs(300)
    {
        std::thread::sleep(Duration::from_millis(500));
    }
    crate::native::stop(cfg, crate::container::SERVER, 30);
    println!("server: stopped");
    Ok(())
}
/// Arguments start at the subcommand (`info`, `generate`, `serve`, ...).
pub(crate) fn cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    nextrocm_engine_core::use_kind("llm");
    match args.first().map(String::as_str) {
        Some("engines") => {
            crate::list_engines(
                engines()
                    .iter()
                    .map(|k| (k.archs.join(", "), k.name, options(k))),
            );
            Ok(())
        }
        Some("info") if args.len() == 2 => info(Path::new(&args[1])),
        Some("tokenize") if args.len() == 3 => tokenize(Path::new(&args[1]), &args[2]),
        Some("generate") => generate(args),
        Some("serve") if args.get(1).is_some_and(|s| Path::new(s).is_file()) => serve_file(args),
        Some("serve") => serve_file(&launch_args(cfg, &args[1..])?),
        Some("start") => start(cfg, &args[1..]),
        Some("stop") => stop(cfg),
        Some("logs") => {
            let mut c = std::process::Command::new("tail");
            c.args(["-n", "200"]);
            if !args.iter().any(|s| s == "--no-follow") {
                c.arg("-F");
            }
            let status = c
                .arg(crate::native::log_path(cfg, crate::container::SERVER))
                .status()
                .map_err(|e| e.to_string())?;
            if status.success() {
                Ok(())
            } else {
                Err(format!("logs: {status}"))
            }
        }
        Some("prepare") if args.len() == 2 => {
            let f = Gguf::open(Path::new(&args[1])).map_err(|e| e.0)?;
            describe(&f)?;
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inkling_sampling_error_is_fallible_before_commit() {
        use nextrocm_llm::Sampler;
        let mut rng = nextrocm_llm::sampling::Rng(7);
        let mut sampler = CliSampler {
            full_vocab: true,
            temp: 1.0,
            top_p: 1.0,
            rng: &mut rng,
        };
        assert!(sampler.try_sample(&[f32::NAN]).is_err());
        sampler.temp = 0.0;
        assert_eq!(sampler.try_sample(&[1.0, 1.0]).unwrap(), 0);
    }
    #[test]
    fn inkling_invalid_inputs_reject_without_gpu() {
        assert!(validate_inkling_inputs("inkling2", &[0, 200006], &[4096]).is_ok());
        for ids in [vec![200054], vec![200058], vec![u32::MAX]] {
            assert!(validate_inkling_inputs("inkling2", &ids, &[512]).is_err());
        }
        for ctx in [vec![], vec![0], vec![4097], vec![512, 512]] {
            assert!(validate_inkling_inputs("inkling2", &[0], &ctx).is_err());
        }
        assert!(validate_inkling_inputs("qwen35", &[200054], &[8192, 8192]).is_ok());
    }
    #[test]
    fn option_errors_are_not_silently_ignored() {
        let a = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(
            validate_options(&a(&["--ctx", "512", "--no-mtp"]), &["--ctx"], &["--no-mtp"]).is_ok()
        );
        assert!(validate_options(&a(&["--ctx"]), &["--ctx"], &[]).is_err());
        assert!(validate_options(&a(&["--ctx", "--gpu", "0"]), &["--ctx", "--gpu"], &[]).is_err());
        assert!(validate_options(&a(&["--score", "ignored"]), &["--ctx"], &[]).is_err());
    }
}
