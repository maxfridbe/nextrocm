//! Native service adapter: original image contract, staged residency, no reference fixtures.
//! Deployment is explicitly user-authorized despite the retained .001 numerical-gate failure.
use crate::{
    denoiser::Denoiser,
    ffi::{self, AttentionKernel, PackedKernel},
    plan::DitPlan,
    sched,
    vae::{ConvKernel, Vae},
};
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image::{
    Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture,
    Result, Sampler, Schedule, Step,
};
use nextrocm_vision_qwen3vl::{Budget, LoadOptions as TeOptions, TextEncoder};
use std::{
    ffi::c_void,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};
pub const ARCH: &str = "qwen-image-2.1";
pub const ROLES: &[&str] = &[
    "transformer",
    "text-encoder",
    "vae",
    "tokenizer",
    "text-config",
];
const ALLOWANCE: u64 = 7_500_000_000;
const MARGIN: u64 = 1 << 30;
const QUALITY:&str="User-authorized deployment: original1024 numerical gate FAIL (public image relative L2 0.004054345807295544 > 0.001); optimization outputs remain bitwise identical. Other prompts are not an accuracy validation claim.";
type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
type Axpy = unsafe extern "C" fn(*mut c_void, *mut f32, *const f32, i64, f32) -> i32;
pub fn kind() -> ImageKind {
    ImageKind {
        archs: &[ARCH],
        name: "Qwen-Image 2.1 Turbo (native Rust + HIP)",
        roles: ROLES,
        load,
        options: &[],
    }
}
fn load(
    files: &ModelFiles,
    gpus: &[Arc<Gpu>],
    options: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(Engine::load(files, gpus, options, log)?))
}
fn role(files: &ModelFiles, name: &str) -> Result<PathBuf> {
    let p = files
        .get(name)
        .ok_or_else(|| Error(format!("QwenImage21 missing {name} role")))?;
    if !p.is_file() {
        return Err(Error(format!("QwenImage21 {name} is not a file")));
    }
    Ok(p.clone())
}
/// Original nextsycl splitmix64 / Box-Muller noise. This is not Torch CPU randn.
pub fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut out = Vec::with_capacity(n + 1);
    while out.len() < n {
        let (u, v) = (next().max(1e-300), next());
        let r = (-2.0 * u.ln()).sqrt();
        out.push((r * (std::f64::consts::TAU * v).cos()) as f32);
        out.push((r * (std::f64::consts::TAU * v).sin()) as f32)
    }
    out.truncate(n);
    out
}
pub fn rgba_prompt(prompt: &str) -> String {
    let p = prompt.trim().trim_end_matches('.');
    if p.to_lowercase().starts_with("this is an rgba image") {
        prompt.to_owned()
    } else {
        format!("This is an RGBA image with transparency. {p}. The image has alpha channel and the background is transparent.")
    }
}
pub fn validate_request(r: &ImageRequest) -> Result<()> {
    if r.width.unwrap_or(1024) != 1024 || r.height.unwrap_or(1024) != 1024 {
        return Err(Error("QwenImage21 currently supports1024x1024".into()));
    }
    if r.steps.unwrap_or(8) != 8
        || r.cfg.unwrap_or(1.0) != 1.0
        || r.sampler.unwrap_or(Sampler::Euler) != Sampler::Euler
        || r.schedule.unwrap_or(Schedule::Shift) != Schedule::Shift
        || r.shift.is_some_and(|x| x != 0.0)
    {
        return Err(Error(
            "Turbo requires its official8-step Euler schedule, CFG1, no sigma shift".into(),
        ));
    }
    if r.edit.is_some()
        || !r.loras.is_empty()
        || r.negative.as_ref().is_some_and(|v| !v.trim().is_empty())
        || !r.extra.is_empty()
    {
        return Err(Error("QwenImage21 text generation currently supports no edits, LoRAs, negative guidance, or extra options".into()));
    }
    if r.prompt.trim().is_empty() || r.n == 0 || r.seed.checked_add(u64::from(r.n - 1)).is_none() {
        return Err(Error(
            "nonempty prompt, positive image count and nonoverflowing seeds required".into(),
        ));
    }
    Ok(())
}
struct Session {
    last: String,
}
pub struct Engine {
    gpu: Arc<Gpu>,
    transformer: PathBuf,
    text: PathBuf,
    vae: PathBuf,
    tokenizer: PathBuf,
    text_config: PathBuf,
    stats: Stats,
    axpy: Axpy,
    loaded: f64,
    session: Mutex<Session>,
}
impl Engine {
    fn text_encoder(&self) -> Result<TextEncoder> {
        TextEncoder::load_with_order32(
            &self.text,
            &self.tokenizer,
            &self.text_config,
            self.gpu.clone(),
            TeOptions {
                probability_first: true,
                ordered_headnorm: true,
                ordered_rms: true,
                score_division: true,
            },
            true,
            true,
            true,
            true,
        )
    }
    fn memory(&self) -> Result<(u64, u64)> {
        self.gpu.sync()?;
        let (mut live, mut peak) = (0, 0); /* SAFETY: frozen allocation ABI, valid output pointers and live GPU. */
        ffi::check(
            unsafe { (self.stats)(self.gpu.raw(), &mut live, &mut peak) },
            "QwenImage21 allocation stats",
        )?;
        if peak >= ALLOWANCE {
            return Err(Error(
                "QwenImage21 tracked allocation exceeds device allowance".into(),
            ));
        }
        Ok((live, peak))
    }
    fn empty(&self) -> Result<()> {
        if self.memory()?.0 != 0 {
            return Err(Error(
                "QwenImage21 stage did not release device allocations".into(),
            ));
        }
        Ok(())
    }
    pub fn load(
        files: &ModelFiles,
        gpus: &[Arc<Gpu>],
        o: &LoadOptions,
        log: &mut dyn FnMut(String),
    ) -> Result<Self> {
        let start = Instant::now();
        if !o.merge_loras.is_empty() {
            return Err(Error("QwenImage21 merged LoRAs unsupported".into()));
        }
        let gpu = gpus
            .first()
            .cloned()
            .ok_or_else(|| Error("QwenImage21 requires HIP GPU".into()))?;
        let lib = api()?;
        let stats = lib.symbol("nr_allocation_stats");
        let engine_lib = nextrocm_engine_core::bindings::Library::new("qwenimage21", &gpu)?;
        let axpy = engine_lib.symbol("nrqi_axpy_f32");
        if stats.is_null() || axpy.is_null() {
            return Err(Error("QwenImage21 stats/Euler ABI missing".into()));
        }
        for name in [
            "nrqi_f32_to_f16",
            "nrqi_f16_to_f32",
            "nrqi_vae_silu",
            "nrqi_vae_up2",
            "nrqi_vae_dupup_add",
            "nrqi_vae_pad_br",
            "nrqi_vae_avg_down_add",
            "nrqi_vae_rgba8",
            "nrqi_vae_softmax",
            "nrqi_vae_conv",
            "nrqi_vae_conv64",
            "nsd_linear",
            "nsd_rms_norm_mod",
            "nsd_gate_add",
        ] {
            if engine_lib.symbol(name).is_null() {
                return Err(Error(format!(
                    "QwenImage21 required VAE ABI missing: {name}"
                )));
            }
        }
        ffi::api(&gpu)?.require_kernel(PackedKernel::ExpandedImage4096)?;
        ffi::api(&gpu)?.require_attention(AttentionKernel::Batched1024)?;
        // SAFETY: exact private stats/Euler ABI; library retained by engine-core.
        let mut this = Self {
            gpu,
            transformer: role(files, "transformer")?,
            text: role(files, "text-encoder")?,
            vae: role(files, "vae")?,
            tokenizer: role(files, "tokenizer")?,
            text_config: role(files, "text-config")?,
            stats: unsafe { std::mem::transmute::<*mut c_void, Stats>(stats) },
            axpy: unsafe { std::mem::transmute::<*mut c_void, Axpy>(axpy) },
            loaded: 0.0,
            session: Mutex::new(Session {
                last: "idle; staged weights are loaded only during generation".into(),
            }),
        };
        let gg = Gguf::open(&this.transformer).map_err(|e| Error(e.0))?;
        DitPlan::validate(&gg)?;
        drop(gg);
        drop(this.text_encoder()?);
        this.empty()?;
        this.loaded = start.elapsed().as_secs_f64();
        log(QUALITY.into());
        log("Native Rust/HIP, staged TE/DiT/VAE; actual prompt and original native seed RNG;1024/official8/Euler/CFG1. No inference subprocess or fixture dependency.".into());
        Ok(this)
    }
    fn generate_inner(
        &self,
        r: &ImageRequest,
        progress: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<Vec<Picture>> {
        let start = Instant::now();
        let prompt = if r.rgba {
            rgba_prompt(&r.prompt)
        } else {
            r.prompt.clone()
        };
        let mut output = Vec::new();
        for picture in 0..r.n {
            let mut notify = |at| {
                progress(Step {
                    picture,
                    at,
                    of: 8,
                    seconds: start.elapsed().as_secs_f64(),
                })
            };
            notify(0)?;
            self.empty()?;
            let te = self.text_encoder()?;
            let (ids, drop_rows) = te.tokenize(&prompt)?;
            let tokens = ids
                .len()
                .checked_sub(drop_rows)
                .filter(|v| (1..=512).contains(v))
                .ok_or_else(|| {
                    Error(
                        "prompt must encode to1..512 image-conditioning tokens; no truncation"
                            .into(),
                    )
                })?;
            let encoded = te.encode(
                &prompt,
                &Budget {
                    allowance_bytes: ALLOWANCE as usize,
                    driver_margin_bytes: MARGIN as usize,
                },
                &mut |_, _| notify(0),
            )?;
            if encoded.image_pad.len() != tokens || encoded.image_pad.iter().any(|v| *v) {
                return Err(Error("text-only conditioning mask mismatch".into()));
            }
            let embeddings = encoded.embeddings.to_f32()?;
            if embeddings.len() != tokens * 4096 || embeddings.iter().any(|v| !v.is_finite()) {
                return Err(Error("TE output finite/shape".into()));
            }
            drop(encoded);
            drop(te);
            self.empty()?;
            notify(0)?;
            let gg = Gguf::open(&self.transformer).map_err(|e| Error(e.0))?;
            let dit = Denoiser::load_with_attention(
                &gg,
                &self.gpu,
                4096,
                tokens,
                PackedKernel::ExpandedImage4096,
                0,
                MARGIN,
                AttentionKernel::Batched1024,
            )?;
            let prefix = dit.prefix(&embeddings, false)?;
            drop(embeddings);
            let lat = DevBuf::from_f32(&self.gpu, &noise(r.seed + u64::from(picture), 4096 * 64))?;
            let sigmas = sched::turbo_sigmas();
            for i in 0..8 {
                notify(i as u32)?;
                let velocity = dit.velocity(&prefix, &lat, (64, 64), sigmas[i] * 1000.0, None)?;
                // SAFETY: same-context disjoint F32 tensors of exactly4096*64 values; frozen Euler ABI.
                ffi::check(
                    unsafe {
                        (self.axpy)(
                            self.gpu.raw(),
                            lat.fp(),
                            velocity.fp(),
                            (4096 * 64) as i64,
                            sigmas[i + 1] - sigmas[i],
                        )
                    },
                    "QwenImage21 Euler",
                )?;
                self.gpu.sync()?;
                drop(velocity);
                notify((i + 1) as u32)?;
            }
            let final_latents = lat.to_f32()?;
            if final_latents.iter().any(|v| !v.is_finite()) {
                return Err(Error("nonfinite final latents".into()));
            }
            drop((lat, prefix, dit, gg));
            self.empty()?;
            notify(8)?;
            let vae = Vae::load_with_conv(
                &self.vae,
                &self.gpu,
                ALLOWANCE as usize,
                ConvKernel::Register64Measured1024,
            )?;
            let (bytes, w, h) = vae.decode(&final_latents, 64, 64)?;
            drop(vae);
            self.empty()?;
            if (w, h) != (1024, 1024) || bytes.len() != 1024 * 1024 * 4 {
                return Err(Error("VAE image shape".into()));
            }
            notify(8)?;
            output.push(Picture {
                width: w as u32,
                height: h as u32,
                channels: if r.rgba { 4 } else { 3 },
                data: if r.rgba {
                    bytes
                } else {
                    bytes
                        .chunks_exact(4)
                        .flat_map(|c| [c[0], c[1], c[2]])
                        .collect()
                },
            });
        }
        Ok(output)
    }
}
impl ImageEngine for Engine {
    fn arch(&self) -> &'static str {
        ARCH
    }
    fn defaults(&self) -> Defaults {
        Defaults {
            width: 1024,
            height: 1024,
            steps: 8,
            cfg: 1.0,
            sampler: Sampler::Euler,
            schedule: Schedule::Shift,
            shift: 0.0,
        }
    }
    fn samplers(&self) -> Vec<Sampler> {
        vec![Sampler::Euler]
    }
    fn schedules(&self) -> Vec<Schedule> {
        vec![Schedule::Shift]
    }
    fn guidance(&self) -> bool {
        false
    }
    fn shift(&self) -> bool {
        false
    }
    fn takes_loras(&self) -> bool {
        false
    }
    fn steps_free(&self) -> bool {
        false
    }
    fn steps_for(&self, _: &ImageRequest) -> u32 {
        8
    }
    fn sides(&self) -> (u32, u32) {
        (1024, 1024)
    }
    fn max_pixels(&self) -> Option<u32> {
        Some(1024 * 1024)
    }
    fn transparent(&self) -> bool {
        true
    }
    fn load_seconds(&self) -> f64 {
        self.loaded
    }
    fn report(&self) -> Vec<String> {
        vec![
            QUALITY.into(),
            self.session
                .lock()
                .map(|s| s.last.clone())
                .unwrap_or_else(|_| "request mutex poisoned".into()),
        ]
    }
    fn generate(
        &self,
        r: &ImageRequest,
        progress: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<Vec<Picture>> {
        validate_request(r)?;
        let mut state = self
            .session
            .lock()
            .map_err(|_| Error("QwenImage21 request mutex poisoned".into()))?;
        let result = self.generate_inner(r, progress);
        let cleanup = self.empty();
        state.last = format!(
            "last request: {}; cleanup: {}",
            if result.is_ok() {
                "completed"
            } else {
                "failed/cancelled"
            },
            if cleanup.is_ok() {
                "zero device allocations"
            } else {
                "failed"
            }
        );
        match result {
            Err(e) => Err(e),
            Ok(v) => {
                cleanup?;
                Ok(v)
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_limits() {
        let mut r = ImageRequest {
            prompt: "A teapot".into(),
            n: 1,
            ..Default::default()
        };
        assert!(validate_request(&r).is_ok());
        r.width = Some(512);
        assert!(validate_request(&r).is_err());
        r.width = None;
        r.steps = Some(7);
        assert!(validate_request(&r).is_err());
        r.steps = None;
        r.cfg = Some(f32::NAN);
        assert!(validate_request(&r).is_err());
        r.cfg = None;
        r.seed = u64::MAX;
        r.n = 2;
        assert!(validate_request(&r).is_err());
    }
    #[test]
    fn native_seed_and_alpha() {
        assert_eq!(noise(7, 17), noise(7, 17));
        assert_ne!(noise(7, 17), noise(8, 17));
        assert!(noise(0, 100).iter().all(|x| x.is_finite()));
        let p = rgba_prompt("teapot.");
        assert_eq!(rgba_prompt(&p), p);
    }
}
