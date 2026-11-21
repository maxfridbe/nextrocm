//! Hardware-only ImageEngine lifecycle validation; parent serializes GPU execution.
//! cargo run -p nextrocm --release --example sd15_lifecycle --
//!   --rv51 CHECKPOINT --rv51-vae MSE_VAE --dreamshaper8 CHECKPOINT
//!   --vocab vocab.json --merges merges.txt [--gpu 0] [--output report.json]
//! Optional: --variant both|rv51|dreamshaper8 (default both), --dreamshaper8-vae PATH,
//! --memory-cap-bytes N (default 8,000,000,000; may only lower the gate).
//! Runtime allocation peaks are exact. Device-wide memory is sampled every 20ms
//! and includes unrelated processes; it is not an exact process/driver peak.
//! No subprocesses, UI, or alternative inference engines.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{
    Error, ImageEngine, ImageRequest, LoadOptions, ModelFiles, Picture, Result, Sampler, Schedule,
    Step,
};
use nextrocm_image_sd15::check;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

const CAP: u64 = 8_000_000_000;
const CANCEL: &str = "sd15-lifecycle-intentional-cancellation";
const USAGE: &str = "sd15_lifecycle --rv51 CHECKPOINT --rv51-vae VAE --dreamshaper8 CHECKPOINT --vocab FILE --merges FILE [--variant both|rv51|dreamshaper8] [--dreamshaper8-vae VAE] [--gpu 0] [--output FILE] [--memory-cap-bytes N]";
fn err(message: impl std::fmt::Display) -> Error {
    Error(message.to_string())
}
fn ensure(ok: bool, message: impl std::fmt::Display) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(err(message))
    }
}
struct Args {
    models: Vec<(&'static str, PathBuf, Option<PathBuf>)>,
    vocab: PathBuf,
    merges: PathBuf,
    gpu: usize,
    cap: u64,
    output: Option<PathBuf>,
}
impl Args {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self> {
        let mut args = args;
        let mut flags = BTreeMap::new();
        while let Some(key) = args.next() {
            ensure(
                matches!(
                    key.as_str(),
                    "--rv51"
                        | "--rv51-vae"
                        | "--dreamshaper8"
                        | "--dreamshaper8-vae"
                        | "--vocab"
                        | "--merges"
                        | "--variant"
                        | "--gpu"
                        | "--output"
                        | "--memory-cap-bytes"
                ),
                format!("unknown flag {key}; {USAGE}"),
            )?;
            let value = args
                .next()
                .ok_or_else(|| err(format!("missing value for {key}")))?;
            ensure(
                flags.insert(key.clone(), value).is_none(),
                format!("duplicate flag {key}"),
            )?;
        }
        let required = |key: &str| {
            flags
                .get(key)
                .cloned()
                .ok_or_else(|| err(format!("missing {key}; {USAGE}")))
        };
        let variant = flags.get("--variant").map(String::as_str).unwrap_or("both");
        ensure(
            matches!(variant, "both" | "rv51" | "dreamshaper8"),
            "invalid --variant",
        )?;
        let mut models = Vec::new();
        if variant != "dreamshaper8" {
            models.push((
                "rv51",
                PathBuf::from(required("--rv51")?),
                Some(PathBuf::from(required("--rv51-vae")?)),
            ));
        }
        if variant != "rv51" {
            models.push((
                "dreamshaper8",
                PathBuf::from(required("--dreamshaper8")?),
                flags.get("--dreamshaper8-vae").map(PathBuf::from),
            ));
        }
        let cap = flags
            .get("--memory-cap-bytes")
            .map(|v| v.parse::<u64>().map_err(err))
            .transpose()?
            .unwrap_or(CAP);
        ensure(
            cap > 0 && cap <= CAP,
            "memory cap must be positive and <= 8,000,000,000 bytes",
        )?;
        Ok(Self {
            models,
            vocab: required("--vocab")?.into(),
            merges: required("--merges")?.into(),
            gpu: flags
                .get("--gpu")
                .map(|v| v.parse::<usize>().map_err(err))
                .transpose()?
                .unwrap_or(0),
            cap,
            output: flags.get("--output").map(PathBuf::from),
        })
    }
    fn validate_paths(&self) -> Result<()> {
        for path in std::iter::once(&self.vocab)
            .chain(std::iter::once(&self.merges))
            .chain(
                self.models
                    .iter()
                    .flat_map(|(_, checkpoint, vae)| std::iter::once(checkpoint).chain(vae.iter())),
            )
        {
            ensure(
                path.is_file(),
                format!("missing input file {}", path.display()),
            )?;
            // Prevent a report path from overwriting a model/tokenizer input.
            if let Some(output) = &self.output {
                let input = std::fs::canonicalize(path).map_err(err)?;
                if output.exists() {
                    ensure(
                        std::fs::canonicalize(output).map_err(err)? != input,
                        "report path aliases an input file",
                    )?;
                }
            }
        }
        Ok(())
    }
}
#[derive(Default)]
struct Samples {
    count: u64,
    initial: u64,
    peak: u64,
    last: u64,
    error: Option<String>,
}
struct Monitor {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Samples>>,
    worker: Option<JoinHandle<()>>,
}
fn device_used(gpu: &Gpu) -> Result<u64> {
    let (total, free) = gpu.memory()?;
    let free = free.ok_or_else(|| err("driver free-memory telemetry unavailable"))?;
    total
        .checked_sub(free)
        .ok_or_else(|| err("invalid driver memory counters"))
}
impl Monitor {
    fn start(gpu: Arc<Gpu>) -> Result<Self> {
        let initial = device_used(&gpu)?;
        let samples = Arc::new(Mutex::new(Samples {
            count: 1,
            initial,
            peak: initial,
            last: initial,
            error: None,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let (state, done) = (samples.clone(), stop.clone());
        let worker = thread::Builder::new()
            .name("sd15-memory-sampler".into())
            .spawn(move || {
                while !done.load(Ordering::Acquire) {
                    let reading = device_used(&gpu);
                    if let Ok(mut s) = state.lock() {
                        match reading {
                            Ok(used) => {
                                s.count += 1;
                                s.peak = s.peak.max(used);
                                s.last = used;
                            }
                            Err(e) => {
                                s.error = Some(e.0);
                                break;
                            }
                        }
                    } else {
                        break;
                    }
                    thread::sleep(Duration::from_millis(20));
                }
            })
            .map_err(err)?;
        Ok(Self {
            stop,
            samples,
            worker: Some(worker),
        })
    }
    fn gate(&self, cap: u64) -> Result<()> {
        let state = self
            .samples
            .lock()
            .map_err(|_| err("memory sampler poisoned"))?;
        if let Some(error) = &state.error {
            return Err(err(error));
        }
        ensure(
            state.peak < cap,
            format!(
                "sampled device-wide usage {} exceeds strict cap {cap}",
                state.peak
            ),
        )
    }
    fn finish(mut self) -> Result<Value> {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| err("memory sampler panicked"))?;
        }
        let s = self
            .samples
            .lock()
            .map_err(|_| err("memory sampler poisoned"))?;
        if let Some(error) = &s.error {
            return Err(err(error));
        }
        Ok(
            json!({"samples":s.count,"interval_ms":20,"initial_device_used":s.initial,
            "sampled_peak_device_used":s.peak,"last_sample_device_used":s.last,
            "scope":"device-wide, includes unrelated processes; sampled, not exact driver peak"}),
        )
    }
}
impl Drop for Monitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
fn allocations(gpu: &Gpu, cap: u64, clean: bool) -> Result<Value> {
    let (current, peak) = check::allocation_stats(gpu)?
        .ok_or_else(|| err("nr_allocation_stats required; cannot establish cleanup/cap"))?;
    ensure(
        peak < cap,
        format!("allocator peak {peak} exceeds strict cap {cap}"),
    )?;
    if clean {
        ensure(
            current == 0,
            format!("leaked {current} device allocation bytes"),
        )?;
    }
    Ok(json!({"current":current,"peak":peak}))
}
#[derive(Default)]
struct Progress {
    calls: u32,
    last: u32,
    zero_calls: u32,
    last_seconds: f64,
    cancel_at: Option<u32>,
    cancel_calls: u32,
    live_at_cancel: u64,
}
impl Progress {
    fn observe(&mut self, s: Step, steps: u32) -> Result<()> {
        ensure(
            s.picture == 0 && s.of == steps && s.at <= steps,
            "invalid progress picture/count",
        )?;
        ensure(
            s.at >= self.last && s.at <= self.last + 1,
            "nonmonotonic or skipped denoising progress",
        )?;
        ensure(
            s.seconds.is_finite() && s.seconds >= self.last_seconds,
            "nonmonotonic progress time",
        )?;
        ensure(
            self.calls > 0 || s.at == 0,
            "initial progress did not start at zero",
        )?;
        self.calls += 1;
        self.last = s.at;
        self.last_seconds = s.seconds;
        if s.at == 0 {
            self.zero_calls += 1;
        }
        Ok(())
    }
    fn json(&self) -> Value {
        json!({"callbacks":self.calls,"zero_callbacks":self.zero_calls,"last_step":self.last,
        "cancel_at":self.cancel_at,"cancel_callbacks":self.cancel_calls,"live_bytes_at_cancel":self.live_at_cancel})
    }
}
fn request() -> ImageRequest {
    // Leave width/height/steps/cfg/sampler/schedule unset: exercise engine defaults.
    ImageRequest {
        prompt: "a red wooden cabin beside a quiet lake, detailed landscape".into(),
        negative: Some("blurry, low quality".into()),
        seed: 42,
        n: 1,
        ..Default::default()
    }
}
fn small_request() -> ImageRequest {
    ImageRequest {
        width: Some(128),
        height: Some(128),
        steps: Some(2),
        ..request()
    }
}
fn cancel(
    engine: &dyn ImageEngine,
    gpu: &Gpu,
    monitor: &Monitor,
    cap: u64,
    at: u32,
    rows: &mut Vec<Value>,
) -> Result<()> {
    let request = request();
    let steps = engine.steps_for(&request);
    let mut p = Progress::default();
    let start = Instant::now();
    let result = engine.generate(&request, &mut |s| {
        p.observe(s, steps)?;
        monitor.gate(cap)?;
        let stats = allocations(gpu, cap, false)?;
        if s.at == at {
            p.cancel_at = Some(at);
            p.cancel_calls += 1;
            p.live_at_cancel = stats["current"]
                .as_u64()
                .ok_or_else(|| err("missing live bytes"))?;
            return Err(Error(CANCEL.into()));
        }
        Ok(())
    });
    let cleanup = allocations(gpu, cap, true)?;
    ensure(
        matches!(&result,Err(e) if e.0==CANCEL),
        format!(
            "cancellation marker was not returned: {}",
            result
                .err()
                .map(|e| e.0)
                .unwrap_or_else(|| "unexpected successful pictures".into())
        ),
    )?;
    ensure(
        p.cancel_calls == 1 && p.cancel_at == Some(at) && p.last == at,
        "cancellation callback was ignored or repeated",
    )?;
    if at == 0 {
        ensure(
            p.calls == 1 && p.live_at_cancel == 0,
            "immediate cancel performed device staging",
        )?;
    } else {
        ensure(
            at < steps && p.live_at_cancel > 0,
            "partial cancellation must follow a live denoising step before VAE",
        )?;
    }
    rows.push(
        json!({"case":if at==0{"immediate_cancel"}else{"denoising_cancel_before_vae"},"pass":true,
        "seconds":start.elapsed().as_secs_f64(),"progress":p.json(),"allocations_after":cleanup}),
    );
    Ok(())
}
fn generate(
    engine: &dyn ImageEngine,
    gpu: &Gpu,
    monitor: &Monitor,
    cap: u64,
    req: &ImageRequest,
    label: &str,
    rows: &mut Vec<Value>,
) -> Result<Picture> {
    let start = Instant::now();
    let steps = engine.steps_for(req);
    let mut progress = Progress::default();
    let result = engine.generate(req, &mut |s| {
        progress.observe(s, steps)?;
        monitor.gate(cap)?;
        allocations(gpu, cap, false)?;
        eprintln!("{label}: {}/{}", s.at, s.of);
        Ok(())
    });
    let cleanup = allocations(gpu, cap, true)?;
    let mut pictures = result?;
    ensure(
        progress.last == steps,
        "generation returned before final denoising progress",
    )?;
    ensure(
        pictures.len() == 1,
        "generation did not return exactly one picture",
    )?;
    let picture = pictures.remove(0);
    let defaults = engine.defaults();
    let (w, h) = (
        req.width.unwrap_or(defaults.width),
        req.height.unwrap_or(defaults.height),
    );
    ensure(
        picture.width == w
            && picture.height == h
            && picture.channels == 3
            && picture.data.len() == w as usize * h as usize * 3,
        "invalid RGB picture dimensions/length",
    )?;
    let min = picture
        .data
        .iter()
        .copied()
        .min()
        .ok_or_else(|| err("empty image"))?;
    let max = picture
        .data
        .iter()
        .copied()
        .max()
        .ok_or_else(|| err("empty image"))?;
    ensure(min < max, "constant/blank generated image")?;
    // Diagnostic checksum only; reuse below compares every byte, not the checksum.
    let checksum = picture.data.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    rows.push(json!({"case":label,"pass":true,"width":w,"height":h,"steps":steps,
        "seconds":start.elapsed().as_secs_f64(),"pixel_min":min,"pixel_max":max,"fnv1a64":format!("{checksum:016x}"),
        "progress":progress.json(),"allocations_after":cleanup}));
    Ok(picture)
}
fn lifecycle(
    engine: &dyn ImageEngine,
    gpu: &Gpu,
    monitor: &Monitor,
    args: &Args,
    variant: &str,
    rows: &mut Vec<Value>,
) -> Result<()> {
    let d = engine.defaults();
    ensure(
        engine.arch() == "sd15"
            && d.width == 512
            && d.height == 512
            && d.steps == 25
            && d.sampler == Sampler::DpmPp2m
            && d.schedule == Schedule::Karras,
        "unexpected SD1.5 defaults",
    )?;
    ensure(
        d.cfg == if variant == "rv51" { 5. } else { 7. },
        "wrong variant CFG default",
    )?;
    cancel(engine, gpu, monitor, args.cap, 0, rows)?;
    let baseline = generate(
        engine,
        gpu,
        monitor,
        args.cap,
        &small_request(),
        "reuse_after_immediate_128",
        rows,
    )?;
    cancel(engine, gpu, monitor, args.cap, 1, rows)?;
    let after_cancel = generate(
        engine,
        gpu,
        monitor,
        args.cap,
        &small_request(),
        "reuse_after_denoising_cancel_128",
        rows,
    )?;
    ensure(
        after_cancel == baseline,
        "partial cancellation changed deterministic reuse image",
    )?;
    let image = generate(
        engine,
        gpu,
        monitor,
        args.cap,
        &request(),
        "defaults_512_25",
        rows,
    )?;
    if let Some(report) = &args.output {
        let path = report.with_file_name(format!("sd15-{variant}-default-512.png"));
        image.write_png(&path, &[("model", variant.to_owned()), ("steps", "25".into())]).map_err(err)?;
        rows.push(json!({"case":"saved_default_image","path":path,"pass":true}));
    }
    drop(image);
    let after_default = generate(
        engine,
        gpu,
        monitor,
        args.cap,
        &small_request(),
        "reuse_after_defaults_128",
        rows,
    )?;
    ensure(
        after_default == baseline,
        "default request changed deterministic small reuse image",
    )?;
    rows.push(json!({"case":"reuse_byte_identity","pass":true,"compared_images":3,"bytes_each":baseline.data.len()}));
    Ok(())
}
fn host_high_water() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmHWM:")?
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()?
                .checked_mul(1024)
        })
}
fn run_variant(args: &Args, variant: &str, checkpoint: &PathBuf, vae: Option<&PathBuf>) -> Value {
    let start = Instant::now();
    let mut rows = Vec::new();
    let mut memory = Value::Null;
    let mut report = Vec::<String>::new();
    let result = (|| -> Result<()> {
        let gpu = Gpu::open(args.gpu)?;
        allocations(&gpu, args.cap, true)?;
        let monitor = Monitor::start(gpu.clone())?;
        let work = (|| -> Result<()> {
            let mut files = ModelFiles::new();
            files.insert("checkpoint".into(), checkpoint.clone());
            files.insert("vocab".into(), args.vocab.clone());
            files.insert("merges".into(), args.merges.clone());
            if let Some(vae) = vae {
                files.insert("vae".into(), vae.clone());
            }
            let mut options = LoadOptions::default();
            options
                .settings
                .insert("NS_SD15_VARIANT".into(), variant.into());
            let engine = (nextrocm_image_sd15::kind().load)(
                &files,
                std::slice::from_ref(&gpu),
                &options,
                &mut |s| eprintln!("{variant}: {s}"),
            )?;
            allocations(&gpu, args.cap, true)?;
            let generated = lifecycle(engine.as_ref(), &gpu, &monitor, args, variant, &mut rows);
            report = engine.report();
            drop(engine);
            let clean = allocations(&gpu, args.cap, true)?;
            rows.push(json!({"case":"engine_drop_cleanup","pass":true,"allocations":clean}));
            generated
        })();
        // Also inspect cleanup when load/generation returned an error.
        let cleanup = allocations(&gpu, args.cap, true);
        let device_gate = monitor.gate(args.cap);
        let sampled = monitor.finish();
        memory = sampled?;
        ensure(
            memory["sampled_peak_device_used"]
                .as_u64()
                .is_some_and(|peak| peak < args.cap),
            "final sampled device memory exceeds cap",
        )?;
        memory["allocator_after_drop"] = cleanup?;
        memory["device_used_after_drop"] = json!(device_used(&gpu)?);
        memory["process_host_high_water_bytes"] = json!(host_high_water());
        device_gate?;
        work
    })();
    json!({"variant":variant,"checkpoint":checkpoint,"external_vae":vae,"pass":result.is_ok(),
        "error":result.err().map(|e|e.0),"seconds":start.elapsed().as_secs_f64(),"cases":rows,
        "memory":memory,"engine_report":report})
}
fn run() -> Result<()> {
    if std::env::args().skip(1).any(|s| s == "--help" || s == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let args = Args::parse(std::env::args().skip(1))?;
    args.validate_paths()?;
    nextrocm_engine_core::use_kind("image");
    let reports: Vec<_> = args
        .models
        .iter()
        .map(|(variant, path, vae)| run_variant(&args, variant, path, vae.as_ref()))
        .collect();
    let passed = reports.iter().all(|r| r["pass"] == true);
    let report = json!({"schema":"sd15-lifecycle-v1","pass":passed,"strict_memory_cap_bytes":args.cap,
        "variants_tested":args.models.iter().map(|m|m.0).collect::<Vec<_>>(),"variants":reports,
        "coverage":"ImageEngine defaults512/25, immediate and step1 cancellation, three identical128/2 reuse images, cleanup",
        "memory_note":"exact native allocation peak; device-wide driver usage sampled20ms; host VmHWM reported separately"});
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    if let Some(path) = &args.output {
        std::fs::write(path, &text).map_err(err)?;
    }
    println!("{text}");
    ensure(passed, "SD1.5 lifecycle failed; see per-variant report")
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e.0);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_request_leaves_engine_defaults_unset() {
        let r = request();
        assert!(
            r.width.is_none()
                && r.height.is_none()
                && r.steps.is_none()
                && r.cfg.is_none()
                && r.sampler.is_none()
                && r.schedule.is_none()
        );
        assert_eq!(r.n, 1);
    }
    #[test]
    fn progress_allows_cfg_callbacks_but_rejects_skips() {
        let mut p = Progress::default();
        for at in [0, 0, 0, 0, 1, 1, 1, 2] {
            p.observe(
                Step {
                    picture: 0,
                    at,
                    of: 25,
                    seconds: 1.,
                },
                25,
            )
            .unwrap();
        }
        assert!(p
            .observe(
                Step {
                    picture: 0,
                    at: 4,
                    of: 25,
                    seconds: 2.
                },
                25
            )
            .is_err());
    }
    #[test]
    fn cap_cannot_be_relaxed() {
        let args="--variant dreamshaper8 --dreamshaper8 checkpoint --vocab vocab --merges merges --memory-cap-bytes 8000000001";
        assert!(Args::parse(args.split_whitespace().map(str::to_owned)).is_err());
    }
    #[test]
    fn both_variants_required_by_default() {
        let args = "--dreamshaper8 checkpoint --vocab vocab --merges merges";
        assert!(Args::parse(args.split_whitespace().map(str::to_owned)).is_err());
    }
}
