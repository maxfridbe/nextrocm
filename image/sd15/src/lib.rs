//! Native Rust SD1.5 graph, using original image contracts and sampler/PNG utilities.
//! One resident component at a time: CLIP -> UNet -> decoder. No external inference engine.
pub mod check;
mod clip;
mod graph;
pub mod kernels;
pub mod rng;
pub mod schedule;
pub mod tokenizer;
pub mod unet;
pub mod vae;
pub mod weights;

use graph::Graph;
use nextrocm_diffusion::samplers;
use nextrocm_engine_core::Gpu;
use nextrocm_image::{
    Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture,
    Result, Sampler, Schedule, Step,
};
use std::{
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
use weights::{Checkpoint, Stage};

pub const ARCH: &str = "sd15";
pub const ROLES: &[&str] = &["checkpoint", "vocab", "merges"];
/// Original image registry entry. Optional `vae` role overrides every baked VAE tensor.
pub fn kind() -> ImageKind {
    ImageKind {
        archs: &[ARCH],
        name: "Stable Diffusion 1.5 (Rust + HIP)",
        roles: ROLES,
        load,
        options: &[],
    }
}
fn load(
    files: &ModelFiles,
    gpus: &[Arc<Gpu>],
    o: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(Sd15::load(files, gpus, o, log)?))
}
fn role<'a>(files: &'a ModelFiles, name: &str) -> Result<&'a Path> {
    files
        .get(name)
        .map(|p| p.as_path())
        .ok_or_else(|| Error(format!("SD1.5 missing {name} role")))
}
/// Immutable source files and a mutex-owned request session; no weights survive a request.
pub struct Sd15 {
    gpu: Arc<Gpu>,
    checkpoint: Checkpoint,
    external: Option<Checkpoint>,
    tokenizer: tokenizer::Tokenizer,
    cfg: f32,
    loaded: f64,
    state: Mutex<check::Memory>,
}
impl Sd15 {
    pub fn load(
        files: &ModelFiles,
        gpus: &[Arc<Gpu>],
        options: &LoadOptions,
        log: &mut dyn FnMut(String),
    ) -> Result<Self> {
        let start = Instant::now();
        if !options.merge_loras.is_empty() {
            return Err(Error("SD1.5 merged LoRAs are not implemented".into()));
        }
        let gpu = gpus
            .first()
            .cloned()
            .ok_or_else(|| Error("SD1.5 requires a native HIP GPU".into()))?;
        if gpu.raw().is_null() {
            return Err(Error("SD1.5 requires a native HIP GPU".into()));
        }
        let path = role(files, "checkpoint")?;
        let variant = options.setting("NS_SD15_VARIANT").unwrap_or_else(|| {
            if path.file_name().is_some_and(|n| {
                n.to_string_lossy()
                    .to_ascii_lowercase()
                    .contains("realistic")
            }) {
                "rv51".into()
            } else {
                "dreamshaper8".into()
            }
        });
        let cfg = match variant.as_str() {
            "rv51" => {
                if !files.contains_key("vae") {
                    return Err(Error(
                        "Realistic Vision 5.1 requires its external MSE VAE role".into(),
                    ));
                }
                5.0
            }
            "dreamshaper8" => 7.0,
            _ => return Err(Error(format!("unsupported SD1.5 variant {variant}"))),
        };
        let checkpoint = Checkpoint::open(path, false)?;
        let external = files
            .get("vae")
            .map(|p| Checkpoint::open(p, true))
            .transpose()?;
        let tokenizer = tokenizer::Tokenizer::load(role(files, "vocab")?, role(files, "merges")?)?;
        // Resolve the complete private ABI before exposing the engine.
        kernels::Kernels::new(gpu.clone())?;
        log(format!("SD1.5 {variant}: checked original checkpoint; staged CLIP/UNet/VAE, GN epsilon 1e-6, no weights resident until generation"));
        Ok(Self {
            gpu,
            checkpoint,
            external,
            tokenizer,
            cfg,
            loaded: start.elapsed().as_secs_f64(),
            state: Mutex::new(check::Memory::default()),
        })
    }
    fn record(memory: &mut check::Memory, w: &weights::Weights, g: &Graph<'_>) {
        memory.weights = memory.weights.max(w.bytes);
        memory.activations_peak = memory.activations_peak.max(g.peak());
    }
}
/// Validate the implemented request surface before allocating or loading device weights.
pub fn validate_request(req: &ImageRequest) -> Result<()> {
    let (w, h) = (req.width.unwrap_or(512), req.height.unwrap_or(512));
    if w == 0 || h == 0 || w > 512 || h > 512 || w % 64 != 0 || h % 64 != 0 {
        return Err(Error(
            "SD1.5 sides must be multiples of 64 in 64..=512".into(),
        ));
    }
    if req.n != 1 {
        return Err(Error(
            "SD1.5 currently supports exactly one picture per request (n=1)".into(),
        ));
    }
    if req.edit.is_some()
        || req.rgba
        || req.shift.is_some()
        || !req.loras.is_empty()
        || !req.extra.is_empty()
    {
        return Err(Error(
            "SD1.5 does not support edit, RGBA, shift, LoRA or extra controls".into(),
        ));
    }
    if !matches!(
        req.sampler.unwrap_or(Sampler::DpmPp2m),
        Sampler::Euler | Sampler::DpmPp2m
    ) {
        return Err(Error("SD1.5 implements Euler and DPM++2M only".into()));
    }
    if req.schedule.unwrap_or(Schedule::Karras) != Schedule::Karras {
        return Err(Error("SD1.5 implements Karras schedule only".into()));
    }
    if !(1..=200).contains(&req.steps.unwrap_or(25))
        || req.cfg.is_some_and(|x| !x.is_finite() || x < 0.0)
    {
        return Err(Error(
            "SD1.5 requires 1..=200 steps and finite nonnegative CFG".into(),
        ));
    }
    Ok(())
}
impl ImageEngine for Sd15 {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults {
            width: 512,
            height: 512,
            steps: 25,
            cfg: self.cfg,
            sampler: Sampler::DpmPp2m,
            schedule: Schedule::Karras,
            shift: 1.0,
        }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler, Sampler::DpmPp2m]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Karras]
    }
    fn shift(&self) -> bool {
        false
    }
    fn takes_loras(&self) -> bool {
        false
    }
    fn grid(&self) -> u32 {
        64
    }
    fn sides(&self) -> (u32, u32) {
        (64, 512)
    }
    fn max_pixels(&self) -> Option<u32> {
        Some(512 * 512)
    }
    fn load_seconds(&self) -> f64 {
        self.loaded
    }
    fn report(&self) -> Vec<String> {
        let m = self.state.lock().unwrap_or_else(|p| p.into_inner());
        vec![format!("SD1.5 on {}: sequential CLIP/UNet/VAE; last request weight peak {} bytes, live activation peak {} bytes, kernel scratch 0 bytes; driver/host usage requires separate admission measurement",self.gpu.name,m.weights,m.activations_peak)]
    }
    fn generate(
        &self,
        req: &ImageRequest,
        progress: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<Vec<Picture>> {
        validate_request(req)?;
        let positive = self.tokenizer.encode(&req.prompt)?;
        let negative = self
            .tokenizer
            .encode(req.negative.as_deref().unwrap_or(""))?;
        let mut memory = self
            .state
            .lock()
            .map_err(|_| Error("SD1.5 session mutex poisoned".into()))?;
        *memory = check::Memory::default();
        // Explicit device plan <8e9, with a runtime margin, independently of eventual measured admission.
        let planned = (weights::WEIGHT_CAP + graph::ACTIVATION_CAP + (512 << 20)) as u64;
        let (total, free) = self.gpu.memory()?;
        if planned >= 8_000_000_000 || planned > free.unwrap_or(total) {
            return Err(Error(format!(
                "SD1.5 requires {planned} free bytes including runtime reserve"
            )));
        }
        let start = Instant::now();
        let steps = req.steps.unwrap_or(25);
        progress(Step {
            picture: 0,
            at: 0,
            of: steps,
            seconds: 0.0,
        })?;
        let (pos, neg) = {
            let w = self.checkpoint.stage(&self.gpu, Stage::Clip)?;
            let g = Graph::new(self.gpu.clone(), &w)?;
            let pos = clip::encode(&g, &self.checkpoint, &positive, &mut None)?.host()?;
            let neg = clip::encode(&g, &self.checkpoint, &negative, &mut None)?.host()?;
            Self::record(&mut memory, &w, &g);
            self.gpu.sync()?;
            (pos, neg)
        };
        progress(Step {
            picture: 0,
            at: 0,
            of: steps,
            seconds: start.elapsed().as_secs_f64(),
        })?;
        let (h, w) = (
            req.height.unwrap_or(512) as usize / 8,
            req.width.unwrap_or(512) as usize / 8,
        );
        let sigmas = schedule::karras(steps)?;
        let cfg = req.cfg.unwrap_or(self.cfg);
        let mut rng = rng::Philox::new(req.seed);
        let initial: Vec<_> = rng
            .randn(4 * h * w)
            .into_iter()
            .map(|n| n * sigmas[0])
            .collect();
        let latent = {
            let weights = self.checkpoint.stage(&self.gpu, Stage::Unet)?;
            let g = Graph::new(self.gpu.clone(), &weights)?;
            let mut at = 0u32;
            let mut model = |x: &[f32], sigma: f32| -> std::result::Result<Vec<f32>, String> {
                progress(Step {
                    picture: 0,
                    at,
                    of: steps,
                    seconds: start.elapsed().as_secs_f64(),
                })
                .map_err(|e| e.0)?;
                let t = schedule::timestep(sigma).map_err(|e| e.0)?;
                let scaled: Vec<_> = x.iter().map(|v| v / (1.0 + sigma * sigma).sqrt()).collect();
                let p = unet::forward(&g, &scaled, h, w, t, &pos, &mut None).map_err(|e| e.0)?;
                let eps = if cfg == 1.0 {
                    p
                } else {
                    progress(Step {
                        picture: 0,
                        at,
                        of: steps,
                        seconds: start.elapsed().as_secs_f64(),
                    })
                    .map_err(|e| e.0)?;
                    let n =
                        unet::forward(&g, &scaled, h, w, t, &neg, &mut None).map_err(|e| e.0)?;
                    p.iter().zip(n).map(|(p, n)| n + cfg * (p - n)).collect()
                };
                at += 1;
                let denoised: Vec<_> = x.iter().zip(eps).map(|(x, e)| x - sigma * e).collect();
                if denoised.iter().any(|v| !v.is_finite()) {
                    return Err("SD1.5 denoiser produced nonfinite values".into());
                }
                progress(Step {
                    picture: 0,
                    at,
                    of: steps,
                    seconds: start.elapsed().as_secs_f64(),
                })
                .map_err(|e| e.0)?;
                Ok(denoised)
            };
            let latent = samplers::sample(
                req.sampler.unwrap_or(Sampler::DpmPp2m),
                &mut model,
                initial,
                &sigmas,
                &mut |n| rng.randn(n),
                &mut |_| {},
            )
            .map_err(Error)?;
            Self::record(&mut memory, &weights, &g);
            self.gpu.sync()?;
            latent
        };
        progress(Step {
            picture: 0,
            at: steps,
            of: steps,
            seconds: start.elapsed().as_secs_f64(),
        })?;
        let picture = {
            let f = self.external.as_ref().unwrap_or(&self.checkpoint);
            let weights = f.stage(&self.gpu, Stage::Vae)?;
            let g = Graph::new(self.gpu.clone(), &weights)?;
            let picture = vae::decode(&g, &latent, h, w, &mut None)?;
            Self::record(&mut memory, &weights, &g);
            self.gpu.sync()?;
            picture
        };
        progress(Step {
            picture: 0,
            at: steps,
            of: steps,
            seconds: start.elapsed().as_secs_f64(),
        })?;
        Ok(vec![picture])
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_geometry_and_controls() {
        let mut r = ImageRequest {
            n: 1,
            width: Some(128),
            height: Some(128),
            ..Default::default()
        };
        assert!(validate_request(&r).is_ok());
        r.width = Some(512);
        assert!(validate_request(&r).is_ok());
        r.width = Some(0);
        assert!(validate_request(&r).is_err());
        r.width = Some(513);
        assert!(validate_request(&r).is_err());
        r.width = Some(512);
        r.shift = Some(1.0);
        assert!(validate_request(&r).is_err());
        r.shift = None;
        r.cfg = Some(f32::INFINITY);
        assert!(validate_request(&r).is_err());
    }
}
