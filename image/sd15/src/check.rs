//! Public native stage checks for comparison with an independently executed CPU oracle.
//! These run only when explicitly called; tests do not create a GPU.
pub use crate::graph::Tap;
use crate::{
    graph::Graph,
    weights::{Checkpoint, Stage},
};
use nextrocm_engine_core::Gpu;
use nextrocm_image::{Picture, Result};
use std::{path::Path, sync::Arc};

/// Measured allocations owned by this Rust session (driver/runtime excluded).
#[derive(Clone, Copy, Debug, Default)]
pub struct Memory {
    pub weights: usize,
    pub activations_peak: usize,
    pub scratch: usize,
}
/// Stage output plus its allocation measurements.
pub struct Output<T> {
    pub value: T,
    pub memory: Memory,
}

/// Encode exactly 77 supplied CLIP IDs. Taps include embedding, twelve layers and final LN.
pub fn clip(
    gpu: Arc<Gpu>,
    checkpoint: &Path,
    ids: &[u32; 77],
    tap: Option<&mut Tap<'_>>,
) -> Result<Output<Vec<f32>>> {
    let f = Checkpoint::open(checkpoint, false)?;
    let w = f.stage(&gpu, Stage::Clip)?;
    let g = Graph::new(gpu, &w)?;
    let mut tap = tap;
    let x = crate::clip::encode(&g, &f, ids, &mut tap)?;
    Ok(Output {
        value: x.host()?,
        memory: Memory {
            weights: w.bytes,
            activations_peak: g.peak(),
            scratch: 0,
        },
    })
}
/// One epsilon evaluation, supplied scaled NCHW latent, fractional timestep and [77,768] context.
/// Latent sides are multiples of eight up to 64 (128-pixel fixtures use 16x16).
#[allow(clippy::too_many_arguments)]
pub fn unet(
    gpu: Arc<Gpu>,
    checkpoint: &Path,
    latent: &[f32],
    height: usize,
    width: usize,
    timestep: f32,
    context: &[f32],
    tap: Option<&mut Tap<'_>>,
) -> Result<Output<Vec<f32>>> {
    let f = Checkpoint::open(checkpoint, false)?;
    let w = f.stage(&gpu, Stage::Unet)?;
    let g = Graph::new(gpu, &w)?;
    let mut tap = tap;
    let value = crate::unet::forward(&g, latent, height, width, timestep, context, &mut tap)?;
    Ok(Output {
        value,
        memory: Memory {
            weights: w.bytes,
            activations_peak: g.peak(),
            scratch: 0,
        },
    })
}
/// Decode final diffusion-space NCHW latent. `external_vae` completely overrides baked VAE.
/// Float pre-clamp RGB is exposed as `vae.decoder.conv_out`; returned picture uses half-up rounding.
pub fn vae(
    gpu: Arc<Gpu>,
    checkpoint: &Path,
    external_vae: Option<&Path>,
    latent: &[f32],
    height: usize,
    width: usize,
    tap: Option<&mut Tap<'_>>,
) -> Result<Output<Picture>> {
    let f = Checkpoint::open(external_vae.unwrap_or(checkpoint), external_vae.is_some())?;
    let w = f.stage(&gpu, Stage::Vae)?;
    let g = Graph::new(gpu, &w)?;
    let mut tap = tap;
    let value = crate::vae::decode(&g, latent, height, width, &mut tap)?;
    Ok(Output {
        value,
        memory: Memory {
            weights: w.bytes,
            activations_peak: g.peak(),
            scratch: 0,
        },
    })
}

/// Native allocator current/peak bytes, when this runtime exports its accounting hook.
/// Synchronizes first so a zero-current check also covers queued work.
pub fn allocation_stats(gpu: &Gpu) -> Result<Option<(u64, u64)>> {
    gpu.sync()?;
    let api = nextrocm_engine_core::api()?;
    let symbol = api.symbol("nr_allocation_stats");
    if symbol.is_null() {
        return Ok(None);
    }
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: the optional symbol has this ABI in kernels/ns/ns.hip; Api retains the library.
    let f: Stats = unsafe { std::mem::transmute(symbol) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: gpu owns the live runtime handle and outputs point to initialized u64 storage.
    let code = unsafe { f(gpu.raw(), &mut current, &mut peak) };
    if code != 0 {
        return Err(nextrocm_image::Error(
            "native allocation statistics failed".into(),
        ));
    }
    Ok(Some((current, peak)))
}

/// Frozen independent inputs for validating the native denoiser and original shared samplers.
pub struct SampleInputs<'a> {
    pub initial: &'a [f32],
    pub sigmas: &'a [f32],
    pub positive: &'a [f32],
    pub negative: &'a [f32],
    pub height: usize,
    pub width: usize,
    pub cfg: f32,
    pub sampler: nextrocm_image::Sampler,
}
/// Run a whole schedule with UNet loaded once. Taps expose each scaled input, timestep,
/// epsilon branch, CFG, x0 and resulting latent; first positive pass also exposes all blocks.
/// Diagnostic arithmetic selection only; production generation keeps its shared sampler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EulerArithmetic {
    #[default]
    SharedF64,
    OracleF32,
}

pub fn sample(
    gpu: Arc<Gpu>,
    checkpoint: &Path,
    input: SampleInputs<'_>,
    tap: Option<&mut Tap<'_>>,
) -> Result<Output<Vec<f32>>> {
    sample_with_euler(gpu, checkpoint, input, tap, EulerArithmetic::SharedF64)
}

/// A/B entry for explicit validation runs. Only Euler accepts OracleF32.
/// This changes neither conditioning nor the denoiser, schedule or comparison gates.
pub fn sample_with_euler(
    gpu: Arc<Gpu>,
    checkpoint: &Path,
    input: SampleInputs<'_>,
    tap: Option<&mut Tap<'_>>,
    arithmetic: EulerArithmetic,
) -> Result<Output<Vec<f32>>> {
    sample_with_controls(gpu, checkpoint, input, tap, arithmetic, None)
}

/// Independent diagnostic controls. Recorded timesteps bypass only sigma-to-t
/// conversion; they never replace latent inputs or change sampling arithmetic.
/// Production generation and the default check retain native conversion.
pub fn sample_with_controls(
    gpu: Arc<Gpu>,
    checkpoint: &Path,
    input: SampleInputs<'_>,
    mut tap: Option<&mut Tap<'_>>,
    arithmetic: EulerArithmetic,
    recorded_timesteps: Option<&[f32]>,
) -> Result<Output<Vec<f32>>> {
    use nextrocm_image::{Error, Sampler};
    if (arithmetic == EulerArithmetic::OracleF32 && input.sampler != Sampler::Euler)
        || !matches!(input.sampler, Sampler::Euler | Sampler::DpmPp2m)
        || input.sigmas.len() < 2
        || input.sigmas.len() > 201
        || input.sigmas.last() != Some(&0.0)
        || input
            .sigmas
            .windows(2)
            .any(|p| !p[0].is_finite() || p[0] <= p[1])
        || !input.cfg.is_finite()
    {
        return Err(Error("invalid frozen sample inputs".into()));
    }
    if let Some(times) = recorded_timesteps {
        if times.len() != input.sigmas.len() - 1
            || times
                .iter()
                .any(|t| !t.is_finite() || *t < 0.0 || *t > 999.0)
        {
            return Err(Error("invalid recorded diagnostic timesteps".into()));
        }
    }
    crate::unet::validate_latent(input.height, input.width)?;
    if input.initial.len() != 4 * input.height * input.width {
        return Err(Error("initial latent shape mismatch".into()));
    }
    let f = Checkpoint::open(checkpoint, false)?;
    let w = f.stage(&gpu, Stage::Unet)?;
    let g = Graph::new(gpu, &w)?;
    let shape = [1, 4, input.height, input.width];
    let mut step = 0usize;
    let mut emit = |name: &str, shape: &[usize], v: &[f32]| -> Result<()> {
        if let Some(f) = tap.as_mut() {
            f(name, shape, v)?;
        }
        Ok(())
    };
    let mut model = |x: &[f32], sigma: f32| -> std::result::Result<Vec<f32>, String> {
        let mut run = || -> Result<Vec<f32>> {
            if step > 0 {
                emit(&format!("step.{}.latent", step - 1), &shape, x)?;
            }
            let scaled: Vec<_> = x.iter().map(|v| v / (1.0 + sigma * sigma).sqrt()).collect();
            let t = diagnostic_timestep(sigma, step, recorded_timesteps)?;
            emit(&format!("step.{step}.input"), &shape, &scaled)?;
            emit(&format!("step.{step}.timestep"), &[1], &[t])?;
            let p = if step == 0 {
                crate::unet::forward(
                    &g,
                    &scaled,
                    input.height,
                    input.width,
                    t,
                    input.positive,
                    &mut Some(&mut emit),
                )?
            } else {
                crate::unet::forward(
                    &g,
                    &scaled,
                    input.height,
                    input.width,
                    t,
                    input.positive,
                    &mut None,
                )?
            };
            let n = crate::unet::forward(
                &g,
                &scaled,
                input.height,
                input.width,
                t,
                input.negative,
                &mut None,
            )?;
            emit(&format!("step.{step}.epsilon.positive"), &shape, &p)?;
            emit(&format!("step.{step}.epsilon.negative"), &shape, &n)?;
            let cfg: Vec<_> = p
                .iter()
                .zip(n)
                .map(|(p, n)| n + input.cfg * (p - n))
                .collect();
            emit(&format!("step.{step}.epsilon.cfg"), &shape, &cfg)?;
            let out: Vec<_> = x.iter().zip(cfg).map(|(x, e)| x - sigma * e).collect();
            emit(&format!("step.{step}.x0"), &shape, &out)?;
            step += 1;
            Ok(out)
        };
        run().map_err(|e| e.0)
    };
    let value = match arithmetic {
        EulerArithmetic::SharedF64 => nextrocm_diffusion::samplers::sample(
            input.sampler,
            &mut model,
            input.initial.to_vec(),
            input.sigmas,
            &mut |n| vec![0.0; n],
            &mut |_| {},
        ),
        EulerArithmetic::OracleF32 => {
            oracle_euler_f32(&mut model, input.initial.to_vec(), input.sigmas)
        }
    }
    .map_err(Error)?;
    emit(
        &format!("step.{}.latent", input.sigmas.len() - 2),
        &shape,
        &value,
    )?;
    Ok(Output {
        value,
        memory: Memory {
            weights: w.bytes,
            activations_peak: g.peak(),
            scratch: 0,
        },
    })
}

fn diagnostic_timestep(sigma: f32, step: usize, recorded: Option<&[f32]>) -> Result<f32> {
    match recorded {
        Some(times) => times
            .get(step)
            .copied()
            .ok_or_else(|| nextrocm_image::Error("missing recorded diagnostic timestep".into())),
        None => crate::schedule::timestep(sigma),
    }
}

/// Private SD1 diagnostic recurrence matching retained oracle-script.py:
/// latent + ((latent - x0) / sigma) * (next_sigma - sigma), all F32.
/// Keep individual operations (no mul_add, algebraic epsilon substitution,
/// terminal x0 shortcut, or hidden F64 latent carried between model calls).
fn oracle_euler_f32(
    model: &mut nextrocm_diffusion::samplers::Model<'_>,
    mut latent: Vec<f32>,
    sigmas: &[f32],
) -> std::result::Result<Vec<f32>, String> {
    for pair in sigmas.windows(2) {
        let x0 = model(&latent, pair[0])?;
        if x0.len() != latent.len() {
            return Err("SD1 F32 Euler: denoiser length mismatch".into());
        }
        let dt = pair[1] - pair[0];
        for (x, denoised) in latent.iter_mut().zip(x0) {
            let difference = *x - denoised;
            let derivative = difference / pair[0];
            let increment = derivative * dt;
            *x += increment;
            if !x.is_finite() {
                return Err("SD1 F32 Euler: nonfinite latent".into());
            }
        }
    }
    Ok(latent)
}

#[cfg(test)]
mod euler_arithmetic_tests {
    use super::*;

    #[test]
    fn oracle_rounding_is_observable_at_the_next_model_call() {
        // Frozen per-operation IEEE-F32 oracle bits, not computed by this helper.
        // Shared F64 gives 2876.800048828125 for the first element instead.
        let mut calls = 0;
        let mut model = |x: &[f32], _: f32| {
            if calls == 1 {
                assert_eq!(
                    x.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    [0x4533cccc, 0x3db70ced, 0x420e6924]
                );
            }
            calls += 1;
            Ok(vec![0.1234567; x.len()])
        };
        oracle_euler_f32(&mut model, vec![10000.0, 0.005, 123.456], &[7.3, 2.1, 0.0]).unwrap();
        assert_eq!(calls, 2);
    }

    #[test]
    fn recorded_timestep_is_an_independent_exact_control() {
        let recorded = f32::from_bits(0x438acc49);
        assert_eq!(
            diagnostic_timestep(0.76952296, 0, Some(&[recorded]))
                .unwrap()
                .to_bits(),
            recorded.to_bits()
        );
        assert_eq!(
            diagnostic_timestep(0.76952296, 0, None).unwrap().to_bits(),
            crate::schedule::timestep(0.76952296).unwrap().to_bits()
        );
        assert!(diagnostic_timestep(1.0, 1, Some(&[recorded])).is_err());
    }

    #[test]
    fn bad_outputs_and_cancellation_are_errors() {
        assert!(oracle_euler_f32(&mut |_, _| Ok(vec![]), vec![1.], &[1., 0.]).is_err());
        assert!(oracle_euler_f32(&mut |_, _| Ok(vec![f32::NAN]), vec![1.], &[1., 0.]).is_err());
        let failure = oracle_euler_f32(&mut |_, _| Err("cancelled".into()), vec![1.], &[1., 0.]);
        assert_eq!(failure.unwrap_err(), "cancelled");
    }
}
