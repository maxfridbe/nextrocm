//! Admission checks around the reference graph, before reference-audio expansion.
use nextrocm_audio::{AudioRequest, Error, Result};
use nextrocm_engine_core::Gpu;

pub(crate) const LIMIT: u64 = 8_000_000_000;
pub(crate) const MARGIN: u64 = 256 * 1024 * 1024;

/// Header-derived upper bound, deliberately using half even when int8 is requested.
/// All final fused outputs are covered by summing their constituent tensors once.
#[derive(Debug, Default)]
pub(crate) struct LoadPlan {
    pub resident: u64,
    pub transient: u64,
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error("load estimate overflow".into()))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| Error("load estimate overflow".into()))
}

fn elements(t: &nextrocm_gguf::safetensors::StTensor) -> Result<u64> {
    let n = t.shape.iter().try_fold(1u64, |n, &d| mul(n, d))?;
    let width = match t.dtype.as_str() {
        "BF16" => 2,
        "F32" => 4,
        _ => {
            return Err(Error(format!(
                "{}: unsupported load dtype {}",
                t.name, t.dtype
            )));
        }
    };
    if n == 0 || mul(n, width)? != t.bytes {
        return Err(Error(format!("{}: invalid tensor extent", t.name)));
    }
    Ok(n)
}

fn plan_tensors(
    model: &[nextrocm_gguf::safetensors::StTensor],
    codec: &[nextrocm_gguf::safetensors::StTensor],
) -> Result<LoadPlan> {
    // Lm::load owns one 64-float rotary table for each of talker and predictor.
    let mut plan = LoadPlan {
        resident: 2 * 64 * 4,
        transient: 0,
    };
    for t in model {
        let n = elements(t)?;
        // Text rows and the ECAPA speaker graph remain on the host.
        if t.name == "talker.model.text_embedding.weight" || t.name.starts_with("speaker_encoder.")
        {
            continue;
        }
        if !t.name.starts_with("talker.") {
            return Err(Error(format!("{}: unknown model allocation role", t.name)));
        }
        let (resident, transient) = match t.shape.as_slice() {
            [rows, _] => {
                // Shards::mat: final half (or int8 + per-row f32 scales),
                // and 2N staging + 4N floats for the largest constituent only.
                (mul(n, 2)?.max(add(n, mul(*rows, 4)?)?), mul(n, 6)?)
            }
            [_] => (mul(n, 4)?, mul(n, 2)?), // dev_f32 + staging
            _ => return Err(Error(format!("{}: unsupported talker tensor rank", t.name))),
        };
        plan.resident = add(plan.resident, resident)?;
        plan.transient = plan.transient.max(transient);
    }
    // Codec final tensors are f32, including fused qkv/gu. Count the entire
    // codec, even host codebooks and the unused encoder in VoiceDesign: an upper
    // bound for both variants. stack() assembles on host, with no GPU scratch.
    for t in codec {
        let n = elements(t)?;
        plan.resident = add(plan.resident, mul(n, 4)?)?;
        if t.dtype == "BF16" {
            plan.transient = plan.transient.max(mul(n, 2)?);
        }
    }
    Ok(plan)
}

pub(crate) fn load_plan(model: &std::path::Path, codec: &std::path::Path) -> Result<LoadPlan> {
    use nextrocm_gguf::safetensors::SafeTensors;
    let read = |path: &std::path::Path| -> Result<SafeTensors> {
        let f = SafeTensors::open(path).map_err(|e| Error(e.0))?;
        let size = std::fs::metadata(path)
            .map_err(|e| Error(e.to_string()))?
            .len();
        for t in &f.tensors {
            if add(t.offset, t.bytes)? > size {
                return Err(Error(format!("{}: tensor exceeds checkpoint", t.name)));
            }
        }
        Ok(f)
    };
    plan_tensors(&read(model)?.tensors, &read(codec)?.tensors)
}

pub(crate) fn request(kind: &str, r: &AudioRequest) -> Result<()> {
    if !matches!(kind, "custom_voice" | "voice_design" | "base") {
        return Err(Error(format!("unsupported Qwen3-TTS mode {kind}")));
    }
    if r.seconds.is_some_and(|v| !v.is_finite() || v <= 0.0) {
        return Err(Error("speech duration must be finite and positive".into()));
    }
    if kind == "voice_design" && r.instructions.as_ref().is_none_or(|v| v.trim().is_empty()) {
        return Err(Error(
            "VoiceDesign requires a voice description in instructions".into(),
        ));
    }
    if kind == "base" {
        if r.instructions
            .as_ref()
            .is_some_and(|v| !v.trim().is_empty())
        {
            return Err(Error("instructions: not taken by a cloned voice".into()));
        }
        let reference = r
            .reference
            .as_ref()
            .ok_or_else(|| Error("Base requires a reference recording".into()))?;
        reference_samples(&reference.samples, reference.rate)?;
    }
    Ok(())
}

/// Conservative bounds for the original host speaker graph (mel frames, gathered
/// convolution taps and live feature maps). Reject before allocating resampled audio.
pub(crate) fn reference_samples(samples: &[f32], rate: u32) -> Result<usize> {
    if rate == 0 || samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return Err(Error(
            "reference audio needs a nonzero sample rate and finite nonempty samples".into(),
        ));
    }
    let n = (samples.len() as u128 * 24000 + u128::from(rate) / 2) / u128::from(rate);
    let n = usize::try_from(n).map_err(|_| Error("reference length overflow".into()))?;
    if n < 1920 {
        return Err(Error(
            "reference recording is shorter than one 12.5 Hz codec frame".into(),
        ));
    }
    let host = n
        .div_ceil(256)
        .checked_mul(128 * 1024)
        .and_then(|v| n.checked_mul(4).and_then(|audio| v.checked_add(audio)))
        .and_then(|v| v.checked_add(64 * 1024 * 1024))
        .ok_or_else(|| Error("reference working-set overflow".into()))?;
    if host >= 1024 * 1024 * 1024 {
        return Err(Error(
            "reference speaker working set exceeds the 1 GiB host budget".into(),
        ));
    }
    Ok(n)
}

/// Original codec encoder holds three 64-channel full-rate buffers, plus the
/// smaller transformer/quantizer intermediates. No encoder allocation for x-vector only.
pub(crate) fn encoder_work(samples: usize) -> Result<u64> {
    u64::try_from(samples)
        .ok()
        .and_then(|n| n.checked_add(8))
        .and_then(|n| n.checked_mul(3 * 64 * 4))
        .and_then(|n| n.checked_add(128 * 1024 * 1024))
        .ok_or_else(|| Error("reference encoder working-set overflow".into()))
}
pub(crate) fn available(gpu: &Gpu, resident: u64, work: u64) -> Result<()> {
    if resident
        .checked_add(work)
        .and_then(|v| v.checked_add(MARGIN))
        .is_none_or(|n| n >= LIMIT)
    {
        return Err(Error(
            "Qwen3-TTS resident/work estimate exceeds the 8 GB limit".into(),
        ));
    }
    let (total, free) = gpu.memory()?;
    if work
        .checked_add(MARGIN)
        .is_none_or(|n| n > free.unwrap_or(total))
    {
        return Err(Error(
            "insufficient free GPU memory for Qwen3-TTS working set".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextrocm_audio::Reference;
    #[test]
    fn modes_reject_missing_and_invalid_conditioning_before_work() {
        let mut r = AudioRequest::default();
        assert!(request("voice_design", &r).is_err());
        r.instructions = Some("A warm low voice".into());
        assert!(request("voice_design", &r).is_ok());
        assert!(request("base", &r).is_err());
        r.instructions = None;
        r.reference = Some(Reference {
            samples: vec![0.; 2400],
            rate: 24000,
            text: None,
        });
        assert!(request("base", &r).is_ok());
        r.seconds = Some(f32::NAN);
        assert!(request("base", &r).is_err());
        assert!(request("unknown", &r).is_err());
    }
    #[test]
    fn custom_voice_keeps_original_conditioning() {
        let r = AudioRequest {
            prompt: "Hello.".into(),
            voice: Some("vivian".into()),
            instructions: Some("Speak warmly.".into()),
            seconds: Some(3.),
            ..Default::default()
        };
        assert!(request("custom_voice", &r).is_ok());
        // The original CustomVoice path ignores reference audio; it does not
        // acquire Base's requirement for a reference or reject styling.
        let r = AudioRequest {
            reference: Some(Reference::default()),
            ..r
        };
        assert!(request("custom_voice", &r).is_ok());
    }
    #[test]
    fn reference_bounds_precede_resampling() {
        assert!(reference_samples(&[0.], 0).is_err());
        assert!(reference_samples(&[f32::NAN; 1920], 24000).is_err());
        assert!(reference_samples(&[0.; 1920], 1).is_err());
        assert_eq!(reference_samples(&[0.; 1600], 16000).unwrap(), 2400);
        assert_eq!(
            encoder_work(24000).unwrap(),
            3 * 64 * 4 * 24008 + 128 * 1024 * 1024
        );
        assert!(encoder_work(usize::MAX).is_err());
    }
}

#[cfg(test)]
mod load_tests {
    use super::*;
    use nextrocm_gguf::safetensors::StTensor;
    fn tensor(name: &str, shape: &[u64], dtype: &str) -> StTensor {
        StTensor {
            name: name.into(),
            shape: shape.into(),
            dtype: dtype.into(),
            offset: 0,
            bytes: shape.iter().product::<u64>() * if dtype == "BF16" { 2 } else { 4 },
        }
    }
    #[test]
    fn fused_outputs_sum_but_conversion_uses_largest_constituent() {
        let model = [
            tensor("talker.mlp.gate_proj.weight", &[6144, 2048], "BF16"),
            tensor("talker.mlp.up_proj.weight", &[6144, 2048], "BF16"),
            tensor("talker.norm.weight", &[2048], "BF16"),
            tensor(
                "talker.model.text_embedding.weight",
                &[151936, 2048],
                "BF16",
            ),
            tensor("speaker_encoder.fc.weight", &[2048, 3072, 1], "BF16"),
        ];
        let codec = [tensor("decoder.weight", &[4096, 4096], "F32")];
        let plan = plan_tensors(&model, &codec).expect("valid plan");
        assert_eq!(plan.transient, 75_497_472);
        assert_eq!(
            plan.resident,
            512 + 2 * 6144 * 2048 * 2 + 2048 * 4 + 4096 * 4096 * 4
        );
    }
    #[test]
    fn dtype_expansion_scales_and_bad_extents_are_accounted() {
        let model = [tensor("talker.small.weight", &[100, 1], "BF16")];
        let codec = [tensor("decoder.weight", &[1000], "BF16")];
        let plan = plan_tensors(&model, &codec).expect("valid plan");
        assert_eq!(plan.resident, 512 + 500 + 4000);
        assert_eq!(plan.transient, 2000);
        let mut bad = model[0].clone();
        bad.bytes -= 1;
        assert!(plan_tensors(&[bad.clone()], &[]).is_err());
        bad.shape = vec![u64::MAX, 2];
        assert!(plan_tensors(&[bad], &[]).is_err());
        assert!(plan_tensors(&[tensor("unexpected", &[1], "BF16")], &[]).is_err());
    }
}
