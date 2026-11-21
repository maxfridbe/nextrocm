//! Native, non-tap SD15 diagnostic. Main owns builds and exclusive GPU execution.
//! sd15_bench --checkpoint MODEL --vae VAE --vocab FILE --merges FILE
//!   [--width 512 --height 512 --steps 25 --seed 42 --repeats 1 --gpu 0]
//!   [--variant rv51 --label baseline --output NEW_FILE]
//!   [--cancel-after-first-step]
//! Repeats may be 1 or 2 and use the same engine and request. Cancellation mode
//! replaces full generation in each repeat; it does not change the sigma schedule.
//! CFG/sampler/scheduler remain production defaults. NEXTROCM_SD15_PROFILE=1
//! enables graph diagnostics; compare throughput only with that variable unset.
//! Hashing warms the file cache; these are not cold-storage measurements.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{Error, ImageRequest, LoadOptions, ModelFiles, Result, Sampler, Schedule};
use nextrocm_image_sd15::check;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

const CANCEL: &str = "sd15-bench-first-step-cancellation";
const USAGE: &str = "sd15_bench --checkpoint MODEL --vae VAE --vocab FILE --merges FILE [--variant rv51|dreamshaper8] [--width 512 --height 512 --steps 25 --seed 42 --repeats 1|2 --gpu 0] [--label baseline --output NEW_FILE] [--cancel-after-first-step]";
const OFFSET: u64 = 0xcbf29ce484222325;
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(err(message))
    }
}
struct Args {
    files: ModelFiles,
    variant: String,
    label: String,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    repeats: u32,
    gpu: usize,
    cancel: bool,
    output: Option<PathBuf>,
}
impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut flags = BTreeMap::new();
        let mut cancel = false;
        while let Some(key) = args.next() {
            if key == "--cancel-after-first-step" {
                ensure(!cancel, "duplicate cancellation flag")?;
                cancel = true;
                continue;
            }
            ensure(
                matches!(
                    key.as_str(),
                    "--checkpoint"
                        | "--vae"
                        | "--vocab"
                        | "--merges"
                        | "--variant"
                        | "--label"
                        | "--width"
                        | "--height"
                        | "--steps"
                        | "--seed"
                        | "--repeats"
                        | "--gpu"
                        | "--output"
                ),
                USAGE,
            )?;
            let value = args
                .next()
                .ok_or_else(|| err(format!("missing value for {key}")))?;
            ensure(flags.insert(key, value).is_none(), "duplicate argument")?;
        }
        let get =
            |key: &str, default: &str| flags.get(key).cloned().unwrap_or_else(|| default.into());
        let mut files = ModelFiles::new();
        for role in ["checkpoint", "vae", "vocab", "merges"] {
            let path = PathBuf::from(flags.get(&format!("--{role}")).ok_or_else(|| err(USAGE))?);
            ensure(path.is_file(), &format!("missing file {}", path.display()))?;
            files.insert(role.into(), path);
        }
        let out = Self {
            files,
            variant: get("--variant", "rv51"),
            label: get("--label", "unlabeled"),
            width: get("--width", "512").parse().map_err(err)?,
            height: get("--height", "512").parse().map_err(err)?,
            steps: get("--steps", "25").parse().map_err(err)?,
            seed: get("--seed", "42").parse().map_err(err)?,
            repeats: get("--repeats", "1").parse().map_err(err)?,
            gpu: get("--gpu", "0").parse().map_err(err)?,
            cancel,
            output: flags.get("--output").map(PathBuf::from),
        };
        ensure(
            matches!(out.variant.as_str(), "rv51" | "dreamshaper8"),
            "invalid variant",
        )?;
        ensure(
            [out.width, out.height]
                .iter()
                .all(|n| (64..=512).contains(n) && n % 64 == 0),
            "dimensions must be multiples of 64 in 64..=512",
        )?;
        ensure(
            out.steps > 0 && out.steps <= 200,
            "steps must be in 1..=200",
        )?;
        ensure(matches!(out.repeats, 1 | 2), "repeats must be 1 or 2")?;
        Ok(out)
    }
}
fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
    }
    hash
}
fn fingerprint(path: &Path) -> Result<Value> {
    let mut file = File::open(path).map_err(err)?;
    let mut buffer = vec![0u8; 1 << 20];
    let (mut hash, mut bytes) = (OFFSET, 0u64);
    loop {
        let n = file.read(&mut buffer).map_err(err)?;
        if n == 0 {
            break;
        }
        hash = hash_bytes(hash, &buffer[..n]);
        bytes += n as u64;
    }
    Ok(
        json!({"path":std::fs::canonicalize(path).map_err(err)?, "bytes":bytes,
        "fnv1a64":format!("{hash:016x}")}),
    )
}
fn allocations(gpu: &Gpu) -> Result<Value> {
    let (current, peak) =
        check::allocation_stats(gpu)?.ok_or_else(|| err("nr_allocation_stats unavailable"))?;
    Ok(json!({"current_bytes":current,"peak_bytes":peak,
        "clean":current == 0,"under_cap":peak < 8_000_000_000u64}))
}
fn native_libraries() -> Result<Vec<Value>> {
    // Inspect the actual mappings, rather than assuming NEXTROCM_LIB resolved to
    // a particular file. Linux native runner; no subprocess or GPU query here.
    let maps = std::fs::read_to_string("/proc/self/maps").map_err(err)?;
    let mut paths = BTreeSet::new();
    for line in maps.lines() {
        if let Some(at) = line.find('/') {
            let path = &line[at..];
            if path.contains("libnextrocm") && path.contains(".so") {
                paths.insert(path.to_owned());
            }
        }
    }
    paths.iter().map(|p| fingerprint(Path::new(p))).collect()
}
fn benchmark(args: &Args, report: &mut Value) -> Result<()> {
    let hash_start = Instant::now();
    let mut inputs = BTreeMap::new();
    for (role, path) in &args.files {
        inputs.insert(role, fingerprint(path)?);
    }
    report["inputs"] = json!(inputs);
    report["executable"] = fingerprint(&std::env::current_exe().map_err(err)?)?;
    report["input_hash_seconds"] = json!(hash_start.elapsed().as_secs_f64());
    nextrocm_engine_core::use_kind("image");
    let open_start = Instant::now();
    let gpu = Gpu::open(args.gpu)?;
    report["gpu_open_seconds"] = json!(open_start.elapsed().as_secs_f64());
    report["gpu"] = json!({"index":args.gpu,"name":gpu.name});
    let mut options = LoadOptions::default();
    options
        .settings
        .insert("NS_SD15_VARIANT".into(), args.variant.clone());
    let load_start = Instant::now();
    let loaded = (nextrocm_image_sd15::kind().load)(
        &args.files,
        std::slice::from_ref(&gpu),
        &options,
        &mut |s| eprintln!("{s}"),
    );
    report["load_seconds"] = json!(load_start.elapsed().as_secs_f64());
    let engine = match loaded {
        Ok(engine) => engine,
        Err(e) => {
            report["allocations_after_drop"] = allocations(&gpu)?;
            return Err(e);
        }
    };
    let work = (|| -> Result<()> {
        report["native_libraries"] = json!(native_libraries()?);
        let defaults = engine.defaults();
        ensure(
            defaults.sampler == Sampler::DpmPp2m && defaults.schedule == Schedule::Karras,
            "unexpected production sampler/scheduler defaults",
        )?;
        let request = ImageRequest {
            prompt: "a red wooden cabin beside a quiet lake, detailed landscape".into(),
            negative: Some("blurry, low quality".into()),
            width: Some(args.width),
            height: Some(args.height),
            steps: Some(args.steps),
            seed: args.seed,
            n: 1,
            ..Default::default()
        };
        report["request"] = json!({"width":args.width,"height":args.height,"steps":args.steps,
            "seed":args.seed,"n":1,"prompt":request.prompt,"negative":request.negative,
            "cfg":defaults.cfg,"sampler":"dpmpp_2m","schedule":"karras"});
        let mut previous_rgb = None;
        for repeat in 1..=args.repeats {
            // No taps, callback stderr, memory sampling thread, or per-callback
            // GPU queries. Store raw timestamps; serialize only after generation.
            let mut events = Vec::with_capacity(args.steps as usize * 3 + 4);
            let mut cancelled = false;
            let start = Instant::now();
            let result = engine.generate(&request, &mut |step| {
                let wall = start.elapsed().as_secs_f64();
                events.push((step, wall));
                if args.cancel && step.at == 1 {
                    cancelled = true;
                    return Err(err(CANCEL));
                }
                Ok(())
            });
            // Capture before hashing, report assembly or final allocation checks.
            let seconds = start.elapsed().as_secs_f64();
            let events: Vec<_> = events
                .iter()
                .enumerate()
                .map(|(i, (s, wall))| {
                    json!({"ordinal":i,"picture":s.picture,"at":s.at,"of":s.of,
                    "engine_seconds":s.seconds,"wall_seconds":wall})
                })
                .collect();
            let mut row = json!({"repeat":repeat,"generation_wall_seconds":seconds,
                "cancelled":cancelled,"progress":events,"pass":false});
            let outcome = (|| -> Result<()> {
                let stats = allocations(&gpu)?;
                row["allocations_after_generation"] = stats.clone();
                ensure(
                    stats["clean"] == true && stats["under_cap"] == true,
                    "allocation cleanup/cap check failed",
                )?;
                if args.cancel {
                    ensure(
                        cancelled && matches!(&result, Err(e) if e.0 == CANCEL),
                        "first-step cancellation did not return its marker",
                    )?;
                } else {
                    let pictures = result?;
                    ensure(pictures.len() == 1, "expected one RGB image")?;
                    let picture = &pictures[0];
                    ensure(
                        picture.width == args.width
                            && picture.height == args.height
                            && picture.channels == 3
                            && picture.data.len() == args.width as usize * args.height as usize * 3,
                        "invalid RGB image geometry",
                    )?;
                    row["rgb"] = json!({"bytes":picture.data.len(),
                        "fnv1a64":format!("{:016x}",hash_bytes(OFFSET, &picture.data)),
                        "min":picture.data.iter().min(),"max":picture.data.iter().max()});
                    if let Some(previous) = &previous_rgb {
                        let identical = previous == &picture.data;
                        row["identical_to_first"] = json!(identical);
                        ensure(identical, "repeat RGB differs from first generation")?;
                    } else {
                        previous_rgb = Some(picture.data.clone());
                    }
                }
                Ok(())
            })();
            row["pass"] = json!(outcome.is_ok());
            row["error"] = json!(outcome.as_ref().err().map(|e| &e.0));
            report["runs"]
                .as_array_mut()
                .ok_or_else(|| err("missing runs array"))?
                .push(row);
            outcome?;
        }
        Ok(())
    })();
    report["engine_report"] = json!(engine.report());
    drop(engine);
    let stats = allocations(&gpu)?;
    report["allocations_after_drop"] = stats.clone();
    work?;
    ensure(
        stats["clean"] == true && stats["under_cap"] == true,
        "final allocation cleanup/cap check failed",
    )
}
fn run() -> Result<()> {
    if std::env::args().skip(1).any(|s| s == "--help" || s == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let args = Args::parse(std::env::args().skip(1))?;
    // Refuse existing paths, including model aliases, before doing expensive work.
    let mut output = args
        .output
        .as_ref()
        .map(|p| {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(p)
                .map_err(err)
        })
        .transpose()?;
    let mut report = json!({"schema":"sd15-bench-v1","label":args.label,"variant":args.variant,
        "unix_seconds":SystemTime::now().duration_since(UNIX_EPOCH).map_err(err)?.as_secs(),
        "mode":if args.cancel {"cancel_after_first_step"} else {"full_generation"},
        "repeats":args.repeats,"runs":[],"pass":false,
        "environment":{"NEXTROCM_SD15_PROFILE":std::env::var("NEXTROCM_SD15_PROFILE").ok(),
            "NEXTROCM_LIB":std::env::var("NEXTROCM_LIB").ok()},
        "notes":["FNV-1a 64-bit hashes are diagnostic, not cryptographic integrity checks",
            "input hashing warms filesystem cache; load excludes hashing and GPU open",
            "generation includes component staging, callbacks, cleanup and enabled graph profiler emission; not pure GPU time",
            "allocation peak is cumulative for this GPU context; driver/device-wide memory is not sampled",
            "repeated step numbers are retained; at=steps alone does not imply VAE finished",
            "cancellation includes CLIP/staging and two CFG UNet passes before sampler update; excludes VAE",
            "same-engine repeats still restage component weights; no tensor taps"]});
    let result = benchmark(&args, &mut report);
    report["pass"] = json!(result.is_ok());
    report["error"] = json!(result.as_ref().err().map(|e| &e.0));
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    if let Some(file) = output.as_mut() {
        file.write_all(text.as_bytes()).map_err(err)?;
        file.write_all(b"\n").map_err(err)?;
    }
    println!("{text}");
    result
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e.0);
        std::process::exit(1);
    }
}
