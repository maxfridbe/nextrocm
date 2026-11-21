//! Manual native lifecycle test. Parent serializes GPU execution; never run by cargo test.
//! lifecycle MODEL_DIR [--gpu 0] [--steps 4] [--output report.json]
//!   [--reference SPEECH.wav --reference-text "the transcript"]
//! Explicitly selects NS_VCP_INT8=1; the engine's original default remains half.
//! Generation duration is bounded to four patches (0.64s). Flow steps default to
//! the original10; --steps is an explicit test override. Uses the original
//! AudioEngine API for generation, and concrete tokens() only for prompt-width
//! assertions. No numeric-parity or intelligibility claim is made here.
use nextrocm_audio::{
    Audio, AudioEngine, AudioRequest, Error, LoadOptions, Reference, Result, Step,
};
use nextrocm_audio_voxcpm2::{VoxCpm2, budget, files_in};
use nextrocm_engine_core::Gpu;
use serde_json::{Value, json};
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
const CAP: u64 = 8_000_000_000;
const CANCEL: &str = "voxcpm2-lifecycle-intentional-cancel";
const SECONDS: f32 = 0.64;
const PATCHES: u32 = 4;
const SAMPLES_PER_PATCH: usize = 4 * 1920;
const USAGE: &str = "lifecycle MODEL_DIR [--gpu 0] [--steps 4] [--output FILE] [--reference WAV --reference-text TEXT]";
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn sha(path: &Path) -> Result<String> {
    let output = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(err)?;
    let text = String::from_utf8(output.stdout).map_err(err)?;
    let digest = text.split_whitespace().next().unwrap_or("");
    if !output.status.success()
        || digest.len() != 64
        || !digest.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err(err("sha256sum failed"));
    }
    Ok(digest.into())
}
fn ensure(ok: bool, message: impl std::fmt::Display) -> Result<()> {
    if ok { Ok(()) } else { Err(err(message)) }
}
struct Args {
    model: PathBuf,
    gpu: usize,
    steps: Option<u32>,
    output: Option<PathBuf>,
    reference: Option<PathBuf>,
    reference_text: Option<String>,
}
impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let model = args.next().ok_or_else(|| err(USAGE))?.into();
        let mut a = Self {
            model,
            gpu: 0,
            steps: None,
            output: None,
            reference: None,
            reference_text: None,
        };
        let mut seen = std::collections::BTreeSet::new();
        while let Some(flag) = args.next() {
            ensure(seen.insert(flag.clone()), format!("duplicate {flag}"))?;
            let v = args
                .next()
                .ok_or_else(|| err(format!("missing value for {flag}")))?;
            match flag.as_str() {
                "--gpu" => a.gpu = v.parse().map_err(err)?,
                "--steps" => a.steps = Some(v.parse().map_err(err)?),
                "--output" => a.output = Some(v.into()),
                "--reference" => a.reference = Some(v.into()),
                "--reference-text" => a.reference_text = Some(v),
                _ => return Err(err(format!("unknown {flag}; {USAGE}"))),
            }
        }
        ensure(
            a.steps.is_none_or(|n| (1..=200).contains(&n)),
            "steps must be1..=200",
        )?;
        ensure(
            a.reference.is_some() == a.reference_text.is_some(),
            "reference WAV and transcript must be supplied together",
        )?;
        ensure(
            a.reference_text
                .as_ref()
                .is_none_or(|s| !s.trim().is_empty()),
            "empty reference transcript",
        )?;
        Ok(a)
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
            .name("voxcpm2-memory-sampler".into())
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
fn stats(gpu: &Gpu, baseline: Option<u64>) -> Result<(u64, u64)> {
    gpu.sync()?;
    let (current, peak) = budget::stats(gpu)?;
    ensure(
        peak < CAP,
        format!("allocator peak {peak} exceeds strict cap {CAP}"),
    )?;
    if let Some(expected) = baseline {
        ensure(
            current == expected,
            format!("live allocation bytes {current}; expected {expected}"),
        )?;
    }
    Ok((current, peak))
}
#[derive(Default)]
struct Progress {
    calls: u32,
    prompt_started: bool,
    prompt_done: bool,
    patches: u32,
    decode_started: bool,
    decoded: bool,
    last_seconds: f64,
}
impl Progress {
    fn observe(&mut self, s: Step) -> Result<()> {
        ensure(
            s.seconds.is_finite() && s.seconds >= self.last_seconds,
            "invalid progress time",
        )?;
        self.last_seconds = s.seconds;
        self.calls += 1;
        match s.phase {
            "prompt" => {
                ensure(
                    s.of == 1 && self.patches == 0 && !self.decode_started,
                    "invalid prompt progress",
                )?;
                if s.at == 0 {
                    ensure(!self.prompt_started, "duplicate prompt start")?;
                    self.prompt_started = true;
                } else {
                    ensure(
                        s.at == 1 && self.prompt_started && !self.prompt_done,
                        "invalid prompt completion",
                    )?;
                    self.prompt_done = true;
                }
            }
            "tokens" => {
                ensure(
                    self.prompt_done
                        && !self.decode_started
                        && s.of == PATCHES
                        && s.at == self.patches + 1
                        && s.at <= s.of,
                    "invalid patch progress",
                )?;
                self.patches = s.at;
            }
            "decode" => {
                ensure(
                    self.patches == PATCHES && s.of == 1,
                    "decode before complete bounded generation",
                )?;
                if s.at == 0 {
                    ensure(!self.decode_started, "duplicate decode start")?;
                    self.decode_started = true;
                } else {
                    ensure(
                        s.at == 1 && self.decode_started && !self.decoded,
                        "invalid decode completion",
                    )?;
                    self.decoded = true;
                }
            }
            _ => return Err(err(format!("unexpected progress phase {}", s.phase))),
        }
        Ok(())
    }
    fn json(&self) -> Value {
        json!({"callbacks":self.calls,"prompt_started":self.prompt_started,"prefill_completed":self.prompt_done,"patches":self.patches,"decode_started":self.decode_started,"decoded":self.decoded})
    }
}
fn short(steps: Option<u32>) -> AudioRequest {
    AudioRequest {
        prompt: "Hello.".into(),
        seconds: Some(SECONDS),
        steps,
        seed: 42,
        ..Default::default()
    }
}
fn wide(steps: Option<u32>) -> AudioRequest {
    AudioRequest {
        prompt: "The quick brown fox jumps over the lazy dog and then walks toward a quiet river."
            .into(),
        ..short(steps)
    }
}
struct Harness<'a> {
    engine: &'a dyn AudioEngine,
    gpu: &'a Gpu,
    monitor: &'a Monitor,
    baseline: u64,
    rows: Vec<Value>,
}
impl Harness<'_> {
    fn generate(&mut self, req: &AudioRequest, label: &str) -> Result<Audio> {
        let start = Instant::now();
        let mut p = Progress::default();
        let result = self.engine.generate(req, &mut |s| {
            p.observe(s)?;
            self.monitor.gate(CAP)?;
            stats(self.gpu, None)?;
            eprintln!("{label}: {} {}/{}", s.phase, s.at, s.of);
            Ok(())
        });
        let (live, peak) = stats(self.gpu, Some(self.baseline))?;
        let audio = result?;
        ensure(p.decoded, "audio returned without completed decode")?;
        ensure(
            audio.rate == 48000 && audio.channels == 1,
            "expected mono48kHz output",
        )?;
        ensure(
            audio.samples.len() == p.patches as usize * SAMPLES_PER_PATCH,
            "wrong patch/sample mapping or untrimmed continuation context",
        )?;
        ensure(
            audio
                .samples
                .iter()
                .all(|v| v.is_finite() && v.abs() <= 1.00001),
            "nonfinite/out-of-range PCM",
        )?;
        let square = audio
            .samples
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>();
        let rms = (square / audio.samples.len() as f64).sqrt();
        ensure(rms > 1e-6, "silent output")?;
        let hash = audio
            .samples
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
        self.rows.push(json!({"case":label,"pass":true,"progress":p.json(),"samples":audio.samples.len(),"rate":audio.rate,"rms":rms,"fnv1a64":format!("{hash:016x}"),"seconds":start.elapsed().as_secs_f64(),"live_after":live,"allocator_peak":peak}));
        Ok(audio)
    }
    fn cancel(&mut self, req: &AudioRequest, partial: bool) -> Result<()> {
        let mut p = Progress::default();
        let mut hits = 0;
        let mut live_at_cancel = 0;
        let start = Instant::now();
        let result = self.engine.generate(req, &mut |s| {
            p.observe(s)?;
            self.monitor.gate(CAP)?;
            let (live, _) = stats(self.gpu, None)?;
            if hits > 0
                || (!partial && s.phase == "prompt" && s.at == 0)
                || (partial && s.phase == "tokens" && s.at == 1)
            {
                hits += 1;
                live_at_cancel = live;
                return Err(Error(CANCEL.into()));
            }
            Ok(())
        });
        let (live, peak) = stats(self.gpu, Some(self.baseline))?;
        ensure(
            matches!(&result,Err(e) if e.0==CANCEL),
            format!(
                "expected cancellation marker, got {}",
                result
                    .err()
                    .map(|e| e.0)
                    .unwrap_or_else(|| "successful audio".into())
            ),
        )?;
        ensure(
            hits == 1 && !p.decode_started && !p.decoded,
            "cancel ignored/repeated or VAE reached",
        )?;
        if partial {
            ensure(
                p.prompt_done && p.patches == 1 && live_at_cancel > self.baseline,
                "partial cancel did not execute prefill and one real flow patch",
            )?;
        } else {
            ensure(
                p.calls == 1 && !p.prompt_done && p.patches == 0 && live_at_cancel == self.baseline,
                "immediate cancel allocated request work",
            )?;
        }
        self.rows.push(json!({"case":if partial{"cancel_after_patch1_before_vae"}else{"cancel_before_prefill"},"pass":true,"progress":p.json(),"live_at_cancel":live_at_cancel,"live_after":live,"allocator_peak":peak,"seconds":start.elapsed().as_secs_f64()}));
        Ok(())
    }
    fn invalid_requests(&mut self, base: &AudioRequest) -> Result<()> {
        let mut empty = base.clone();
        empty.prompt.clear();
        let mut nan = base.clone();
        nan.cfg = Some(f32::NAN);
        let mut duration = base.clone();
        duration.seconds = Some(321.);
        let mut bad_reference = base.clone();
        bad_reference.reference = Some(Reference {
            samples: vec![0.; 16],
            rate: 0,
            text: None,
        });
        for request in [empty, nan, duration, bad_reference] {
            let mut calls = 0;
            let result = self.engine.generate(&request, &mut |_| {
                calls += 1;
                Ok(())
            });
            ensure(
                result.is_err() && calls == 0,
                "invalid request was not rejected before work",
            )?;
            stats(self.gpu, Some(self.baseline))?;
        }
        self.rows
            .push(json!({"case":"invalid_requests_before_work","pass":true,"requests":4}));
        Ok(())
    }
}
fn identical(a: &Audio, b: &Audio) -> Result<()> {
    ensure(
        a.rate == b.rate
            && a.channels == b.channels
            && a.samples.len() == b.samples.len()
            && a.samples
                .iter()
                .zip(&b.samples)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
        "seeded reuse PCM changed after cancellation/other request",
    )
}
fn exercise(
    engine: &VoxCpm2,
    gpu: &Gpu,
    monitor: &Monitor,
    args: &Args,
    rows: &mut Vec<Value>,
) -> Result<()> {
    let defaults = engine.defaults();
    ensure(
        defaults.rate == 48000
            && defaults.steps == 10
            && defaults.cfg == 2.
            && defaults.seconds == 60.
            && defaults.max_seconds == 320.,
        "original VoxCPM2 defaults changed",
    )?;
    let speech = engine
        .speech()
        .ok_or_else(|| err("missing speech contract"))?;
    ensure(
        speech.design
            && speech.clone
            && speech.direction
            && speech.clone_takes_text
            && !speech.clone_needs_text,
        "voice contract changed",
    )?;
    let (resident, _) = stats(gpu, None)?;
    ensure(resident > 0, "no resident model weights")?;
    let short = short(args.steps);
    let wide = wide(args.steps);
    let short_rows = engine.tokens(&short.prompt).len() + 1;
    let wide_rows = engine.tokens(&wide.prompt).len() + 1;
    ensure(
        short_rows <= 8 && wide_rows > 8,
        "prompts do not cover both original GEMV/wide-product precision paths",
    )?;
    rows.push(json!({"case":"original_contract_and_prefill_widths","pass":true,"short_prefill_rows":short_rows,"wide_prefill_rows":wide_rows,"flow_steps":args.steps.unwrap_or(defaults.steps),"cfg":defaults.cfg,"max_patches":PATCHES,"resident_bytes":resident}));
    let mut h = Harness {
        engine,
        gpu,
        monitor,
        baseline: resident,
        rows: Vec::new(),
    };
    let result = (|| -> Result<()> {
        h.invalid_requests(&short)?;
        h.cancel(&short, false)?;
        let baseline = h.generate(&short, "reuse_after_immediate_cancel_short")?;
        h.cancel(&wide, true)?;
        let reuse = h.generate(&short, "reuse_after_patch_cancel_short")?;
        identical(&baseline, &reuse)?;
        h.generate(&wide, "wide_prefill_complete")?;
        let design = AudioRequest {
            instructions: Some("A calm and warm voice".into()),
            ..short.clone()
        };
        h.generate(&design, "voice_design")?;
        let reference = if let Some(path) = &args.reference {
            let bytes = std::fs::read(path).map_err(err)?;
            let wav = nextrocm_audio::read_wav(&bytes).map_err(err)?;
            Reference {
                samples: wav.mono(),
                rate: wav.rate,
                text: args.reference_text.clone(),
            }
        } else {
            Reference {
                samples: baseline.samples.clone(),
                rate: baseline.rate,
                text: Some(short.prompt.clone()),
            }
        };
        ensure(
            reference.rate > 0
                && !reference.samples.is_empty()
                && reference.samples.iter().all(|x| x.is_finite()),
            "invalid lifecycle reference WAV",
        )?;
        let clone = AudioRequest {
            reference: Some(reference.clone()),
            ..short.clone()
        };
        h.generate(&clone, "voice_clone_timbre")?;
        let mut continuation = clone;
        continuation.extra.insert("continue".into(), "1".into());
        h.generate(&continuation, "voice_continuation_context_trim")?;
        let reused = h.generate(&short, "reuse_after_voice_modes_short")?;
        identical(&baseline, &reused)?;
        h.rows.push(json!({"case":"seeded_reuse_bit_identity","pass":true,"compared_outputs":3,"reference_source":args.reference.as_ref().map(|p|p.display().to_string()).unwrap_or_else(||"generated baseline, transcript Hello.".into())}));
        Ok(())
    })();
    rows.extend(h.rows);
    result
}
fn run() -> Result<()> {
    if std::env::args().skip(1).any(|v| v == "--help" || v == "-h") {
        println!("{USAGE}");
        return Ok(());
    }
    let args = Args::parse(std::env::args().skip(1))?;
    let files = files_in(&args.model)?;
    let mut output = args
        .output
        .as_ref()
        .map(|path| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(err)
        })
        .transpose()?;
    let model_hashes: std::collections::BTreeMap<_, _> = files
        .iter()
        .map(|(role, path)| Ok((role.clone(), sha(path)?)))
        .collect::<Result<_>>()?;
    let executable_sha256 = sha(&std::env::current_exe().map_err(err)?)?;
    let reference_sha256 = args.reference.as_deref().map(sha).transpose()?;
    nextrocm_engine_core::use_kind("audio");
    let gpu = Gpu::open(args.gpu)?;
    stats(&gpu, Some(0))?;
    let monitor = Monitor::start(gpu.clone())?;
    let mut rows = Vec::new();
    let start = Instant::now();
    let mut engine_report = Vec::new();
    let result = (|| -> Result<()> {
        let mut options = LoadOptions::default();
        options.settings.insert("NS_VCP_INT8".into(), "1".into());
        options
            .settings
            .insert("NS_VCP_CONTINUE".into(), "0".into());
        let engine = VoxCpm2::load(&files, &gpu, &options, &mut |s| eprintln!("{s}"))?;
        let result = exercise(&engine, &gpu, &monitor, &args, &mut rows);
        engine_report = engine.report();
        drop(engine);
        result
    })();
    let cleanup = stats(&gpu, Some(0));
    let sampled = monitor.finish();
    let sampled_ok = sampled.as_ref().is_ok_and(|v| {
        v["sampled_peak_device_used"]
            .as_u64()
            .is_some_and(|n| n < CAP)
    });
    let passed = result.is_ok() && cleanup.is_ok() && sampled_ok;
    let host_high_water = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?
                    .checked_mul(1024)
            })
        });
    let report = json!({"schema":"voxcpm2-lifecycle-v1","pass":passed,"model":args.model,"int8_explicit_setting":1,"engine_default_int8_unchanged":0,
  "seconds":start.elapsed().as_secs_f64(),"flow_steps":args.steps.unwrap_or(10),"patch_limit":PATCHES,"cases":rows,
  "error":result.err().map(|e|e.0),"cleanup_error":cleanup.as_ref().err().map(|e|&e.0),"allocator_after_drop":cleanup.ok(),
  "memory_error":sampled.as_ref().err().map(|e|&e.0),"driver_samples":sampled.as_ref().ok(),"strict_cap_bytes":CAP,"host_high_water_bytes":host_high_water,
  "engine_report":engine_report,"model_hashes":model_hashes,"executable_sha256":executable_sha256,
  "reference_sha256":reference_sha256,"scope":"lifecycle/finite audio/memory only; not numeric CPU oracle parity or intelligibility"});
    let text = serde_json::to_string_pretty(&report).map_err(err)?;
    if let Some(file) = output.as_mut() {
        writeln!(file, "{text}").map_err(err)?;
    }
    println!("{text}");
    ensure(passed, "VoxCPM2 lifecycle failed; see report")
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
    fn request_preserves_default_flow_and_bounds_duration() {
        let r = short(None);
        assert!(r.steps.is_none() && r.cfg.is_none());
        assert_eq!((r.seconds.unwrap() * 6.25).ceil() as u32, PATCHES);
    }
    #[test]
    fn progress_requires_prefill_and_real_patch_before_decode() {
        let mut p = Progress::default();
        for (phase, at, of) in [("prompt", 0, 1), ("prompt", 1, 1), ("tokens", 1, 4)] {
            p.observe(Step {
                phase,
                at,
                of,
                seconds: 0.,
            })
            .unwrap();
        }
        assert!(
            p.observe(Step {
                phase: "decode",
                at: 0,
                of: 1,
                seconds: 0.
            })
            .is_err()
        );
    }
    #[test]
    fn reference_requires_transcript() {
        assert!(
            Args::parse(
                ["model", "--reference", "speech.wav"]
                    .into_iter()
                    .map(str::to_owned)
            )
            .is_err()
        );
    }
    #[test]
    fn identity_compares_float_bits() {
        let a = Audio {
            rate: 48000,
            channels: 1,
            samples: vec![0.],
        };
        let mut b = a.clone();
        b.samples[0] = -0.;
        assert!(identical(&a, &b).is_err());
    }
}
