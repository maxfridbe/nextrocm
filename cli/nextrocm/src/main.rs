mod audio;
mod container;
mod drives;
mod host;
mod image;
mod llm;
mod model3d;
mod native;
mod served;
mod service;
mod speech;

fn run() -> Result<(), String> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let _ = ENGINE_OPTS.set(nextrocm_engine_core::options::given(&raw));
    let mut args = nextrocm_engine_core::options::strip(&raw).into_iter();
    match args.next().as_deref() {
        Some("speech-selftest") => speech::ops_test()?,
        Some("tts") => {
            let command=args.next().ok_or("tts check|speak MODEL_DIR INPUT [OUTPUT_WAV]")?;
            let model=args.next().ok_or("missing model directory")?;
            let input=args.next().ok_or("missing text/reference path")?;
            let out=args.next();
            speech::tts(&command,std::path::Path::new(&model),&input,out.as_deref().map(std::path::Path::new))?;
        }
        Some("stt") => {
            let command=args.next().ok_or("stt transcribe|check MODEL_DIR INPUT")?;
            let model=args.next().ok_or("missing model directory")?;
            let input=args.next().ok_or("missing audio/reference path")?;
            speech::run(&command,std::path::Path::new(&model),std::path::Path::new(&input))?;
        }
        Some("switch") => host::switch(&args.collect::<Vec<_>>())?,
        Some("serve") => host::home(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("llm") => llm::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("model3d") => model3d::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("image") => image::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("audio") => audio::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>(), |_| speech::ops_test())?,
        Some("silo") => silo_cmd(&args.collect::<Vec<_>>())?,
        Some("gpus") => {
            let names = nextrocm_core::Gpu::devices()?;
            if names.is_empty() { return Err("no HIP devices found".into()); }
            for (i, name) in names.iter().enumerate() {
                let gpu = nextrocm_core::Gpu::open(i as i32)?;
                let (total, free) = gpu.memory()?;
                println!("{i}: {name}: total={total} free={free} bytes (HIP)");
            }
        }
        Some("selftest") => {
            let index = args.next().unwrap_or_else(|| "0".into()).parse::<i32>().map_err(|e| e.to_string())?;
            nextrocm_core::selftest(index)?;
            println!("PASS: HIP allocation, copies, fill, bounds and vector arithmetic");
        }
        Some("models") => nextrocm_models::registry::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>(), &llm::describe)?,
        Some("gen-api") => {
            let root=std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            for (name,text) in nextrocm_serve::home::api_files() {
                let path=root.join(name);
                if let Some(parent)=path.parent(){std::fs::create_dir_all(parent).map_err(|e|e.to_string())?;}
                std::fs::write(path,text).map_err(|e|e.to_string())?;
            }
        }
        Some("--version") => println!("nextrocm {}", env!("CARGO_PKG_VERSION")),
        None | Some("--help") => println!("nextrocm: Rust -> C ABI -> HIP\nUsage: nextrocm silo [ARCH] | gpus | selftest [GPU] | models | model3d --help | serve [--host HOST] [--port 8000] [--tls-port 8443] | switch --help | --version"),
        Some(other) => return Err(format!("unknown command: {other}")),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("nextrocm: {e}");
        std::process::exit(1);
    }
}

pub(crate) static ENGINE_OPTS: std::sync::OnceLock<nextrocm_engine_core::options::Given> =
    std::sync::OnceLock::new();

pub(crate) fn engine_opts() -> &'static nextrocm_engine_core::options::Given {
    ENGINE_OPTS.get_or_init(Default::default)
}

/// The variables the --opt-NAMEs set for an engine that takes `declared` (an error lists its options)
pub(crate) fn opt_env(
    declared: &[nextrocm_engine_core::EngineOption],
    at: nextrocm_engine_core::At,
    engine: &str,
) -> Result<Vec<(String, String)>, String> {
    nextrocm_engine_core::options::resolve(engine_opts(), declared, at, engine)
}

/// The engines and, under each, the options it takes (`--opt-NAME`)
fn list_engines(
    rows: impl Iterator<
        Item = (
            String,
            &'static str,
            Vec<nextrocm_engine_core::EngineOption>,
        ),
    >,
) {
    println!("{:<24} ENGINE", "ARCHITECTURE");
    for (a, n, opts) in rows {
        println!("{a:<24} {n}");
        if !opts.is_empty() {
            println!("{}", nextrocm_engine_core::options::usage(&opts));
        }
    }
    println!("\n--opt-NAME VALUE on any command (or NAME the variable it sets) reaches the engine; `options` in an API request too");
}

/// `nextrocm silo [ARCH]`: each GPU as a variant's match rule reads it (and its id in the gate's baselines: `card N ID
/// {json}` lines), and per silo (or ARCH's) the variant each GPU takes - `silo ARCH GPU N: NAME` / `default` lines, the
/// variant's settings under it (tooling/gate.sh reads them). No GPU is opened.
fn silo_cmd(args: &[String]) -> Result<(), String> {
    if args.len()>1 {return Err("usage: nextrocm silo [ARCH]".into());}
    let e = |x: nextrocm_engine_core::Error| x.0;
    let api = nextrocm_engine_core::api().map_err(e)?;
    let n = nextrocm_engine_core::gpus().map_err(e)?.len();
    let devs: Vec<nextrocm_engine_core::variant::DeviceInfo> = (0..n).map(|i| api.device_info(i)).collect::<Result<_, _>>()?;
    for d in &devs {
        println!("GPU {}: {}, {} CUs at {} MHz, {}, gfx {}, wave {}, integrated {}",d.index,d.name,d.units,d.mhz,
            d.vram_mb.map_or("memory not reported".into(),|m|format!("{m} MiB")),d.gfx.as_deref().unwrap_or("?"),
            d.wave_size.map_or("?".into(),|v|v.to_string()),d.integrated.map_or("?".into(),|v|v.to_string()));
    }
    for d in &devs {
        println!("card {} {} {}", d.index, d.card_id(), d.to_json());
    }
    // the silos: libnextrocm-<arch>.so (and its variants libnextrocm-<arch>@<name>.so) beside the kind's library
    let dir = api.silo_dir();
    let mut archs: Vec<String> = match args.first() {
        Some(a) => vec![a.clone()],
        None => {
            let mut v: Vec<String> = std::fs::read_dir(&dir).into_iter().flatten().flatten()
                .filter_map(|f| f.file_name().to_str()?.strip_prefix("libnextrocm-")?.strip_suffix(".so").map(|r| r.split('@').next().unwrap_or(r).to_string()))
                .collect();
            v.sort();
            v.dedup();
            v
        }
    };
    archs.retain(|a| !a.is_empty());
    if archs.is_empty() {
        println!("no silo in {}", dir.display());
    }
    let force = std::env::var("NS_SILO_VARIANT").ok().filter(|v| !v.trim().is_empty());
    for arch in archs {
        let vs = api.silo_variants(&api.silo_name(&arch))?;
        if vs.is_empty() {
            println!("silo {arch}: no variants (every GPU takes the default silo)");
        }
        for d in &devs {
            if let Some(w) = nextrocm_engine_core::variant::ambiguity(&vs, std::slice::from_ref(d), force.as_deref()) {
                println!("{w} (silo {arch})");
            }
            match nextrocm_engine_core::variant::choose(&vs, std::slice::from_ref(d), force.as_deref())? {
                Some(v) => {
                    println!("silo {arch} GPU {}: {} - {}", d.index, v.name, v.manifest.description);
                    if !v.manifest.settings.is_empty() {
                        let s: Vec<String> = v.manifest.settings.iter().map(|(k, x)| format!("{k}={x}")).collect();
                        println!("  settings (the engine's defaults where every GPU it loads on takes {}): {}", v.name, s.join(", "));
                    }
                }
                None => println!("silo {arch} GPU {}: default", d.index),
            }
        }
    }
    Ok(())
}
