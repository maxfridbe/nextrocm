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
        Some("serve") => host::home(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("llm") => llm::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("model3d") => model3d::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("image") => image::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>())?,
        Some("audio") => audio::cmd(&nextrocm_models::Config::load(), &args.collect::<Vec<_>>(), |_| speech::ops_test())?,
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
        None | Some("--help") => println!("nextrocm: Rust -> C ABI -> HIP\nUsage: nextrocm gpus | selftest [GPU] | models | model3d --help | serve [ADDRESS] | --version"),
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
