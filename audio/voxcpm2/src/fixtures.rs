//! Structural validation of the original nextcuda/nextsycl CPU oracle arrays.
//! No GPU access and no numerical acceptance thresholds. A manifest alone is not
//! a usable fixture: the complete arrays must be present and agree in shape.
use super::{f32s, read_npy};
use crate::{
    CONTEXT, Prompt, Spec,
    model::{FEAT, PATCH},
};
use nextrocm_audio::{Error, Result};
use std::path::Path;

pub(super) struct Fixture {
    pub prompt: Prompt,
    pub spec: Spec,
    pub steps: usize,
    pub cfg: f32,
    pub noise: Vec<Vec<f32>>,
    pub patches: Vec<Vec<f32>>,
    pub context: usize,
    pub lm: Vec<f32>,
    pub residual: Vec<f32>,
    pub speech: Vec<f32>,
}

fn shaped(dir: &Path, name: &str, expected: &[usize]) -> Result<Vec<f64>> {
    let (shape, values) = read_npy(&dir.join(name))?;
    if shape != expected {
        return Err(Error(format!(
            "{name}: shape {shape:?}, expected {expected:?}"
        )));
    }
    Ok(values)
}

fn vector(dir: &Path, name: &str) -> Result<Vec<f64>> {
    let (shape, values) = read_npy(&dir.join(name))?;
    if shape.len() != 1 || values.is_empty() {
        return Err(Error(format!(
            "{name}: expected a nonempty vector, got {shape:?}"
        )));
    }
    Ok(values)
}

fn masks(audio: &[f64], text: &[f64]) -> Result<Vec<bool>> {
    if audio.len() != text.len()
        || audio
            .iter()
            .zip(text)
            .any(|(&a, &t)| !matches!(a, 0.0 | 1.0) || !matches!(t, 0.0 | 1.0) || a + t != 1.0)
    {
        return Err(Error(
            "text/audio masks must be complementary binary vectors".into(),
        ));
    }
    Ok(audio.iter().map(|&a| a == 1.0).collect())
}

fn tokens(values: &[f64]) -> Result<Vec<u32>> {
    values
        .iter()
        .map(|&v| {
            if !v.is_finite() || v < 0.0 || v > u32::MAX as f64 || v.fract() != 0.0 {
                Err(Error("invalid reference token ID".into()))
            } else {
                Ok(v as u32)
            }
        })
        .collect()
}

fn patch_array(dir: &Path, name: &str, axes: [usize; 2]) -> Result<Vec<Vec<f32>>> {
    let (shape, values) = read_npy(&dir.join(name))?;
    if shape.len() != 3 || shape[0] == 0 || shape[1..] != axes {
        return Err(Error(format!(
            "{name}: expected [N,{},{}], got {shape:?}",
            axes[0], axes[1]
        )));
    }
    Ok(f32s(values)
        .chunks_exact(FEAT * PATCH)
        .map(<[f32]>::to_vec)
        .collect())
}

impl Fixture {
    pub(super) fn load(dir: &Path) -> Result<Self> {
        let r: serde_json::Value = serde_json::from_slice(
            &std::fs::read(dir.join("request.json")).map_err(|e| Error(e.to_string()))?,
        )
        .map_err(|e| Error(e.to_string()))?;
        let text = r["text"]
            .as_str()
            .filter(|s| !s.trim().is_empty())
            .ok_or_else(|| Error("fixture has no text".into()))?
            .to_string();
        let steps = r["steps"]
            .as_u64()
            .filter(|n| (1..=200).contains(n))
            .ok_or_else(|| Error("invalid fixture flow steps".into()))?
            as usize;
        let cfg = r["cfg"]
            .as_f64()
            .ok_or_else(|| Error("missing fixture CFG".into()))? as f32;
        if !cfg.is_finite() || cfg < 0.0 {
            return Err(Error("invalid fixture CFG".into()));
        }
        let toks = tokens(&vector(dir, "tokens.npy")?)?;
        let rows = toks.len();
        let audio = masks(
            &shaped(dir, "audio_mask.npy", &[rows])?,
            &shaped(dir, "text_mask.npy", &[rows])?,
        )?;
        let feats = f32s(shaped(dir, "audio_feat.npy", &[rows, PATCH, FEAT])?);
        let context = if audio.last() == Some(&true) {
            CONTEXT.min(audio.iter().filter(|a| **a).count())
        } else {
            0
        };
        let noise = patch_array(dir, "noise.npy", [FEAT, PATCH])?;
        let patches = patch_array(dir, "feats.npy", [PATCH, FEAT])?;
        if noise.len() > crate::MAX_PATCHES || patches.len() != noise.len() + context {
            return Err(Error(
                "decoder patch count must equal noise draws plus continuation context".into(),
            ));
        }
        let speech = f32s(shaped(
            dir,
            "speech.npy",
            &[crate::budget::product(&[noise.len(), PATCH, 1920])?],
        )?);
        let optional = |name: &str| -> Result<Option<Vec<f32>>> {
            // Absence is optional; malformed or unreadable present inputs are errors.
            match std::fs::metadata(dir.join(name)) {
                Ok(_) => Ok(Some(f32s(vector(dir, name)?))),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(Error(e.to_string())),
            }
        };
        let reference = optional("ref16k.npy")?;
        let prompt = optional("prompt16k.npy")?;
        match r["mode"].as_str() {
            Some("tts") if reference.is_none() && prompt.is_none() => {}
            Some("clone") if reference.is_some() && prompt.is_none() => {}
            Some("continue") if prompt.is_some() => {}
            _ => {
                return Err(Error(
                    "fixture mode and reference recordings disagree".into(),
                ));
            }
        }
        let prompt = prompt
            .map(|p| {
                let transcript = r["prompt_text"]
                    .as_str()
                    .filter(|s| !s.trim().is_empty())
                    .ok_or_else(|| Error("continuation fixture has no transcript".into()))?;
                Ok::<_, Error>((p, transcript.to_string()))
            })
            .transpose()?;
        Ok(Self {
            prompt: Prompt {
                tokens: toks,
                audio,
                patches: feats
                    .chunks_exact(PATCH * FEAT)
                    .map(<[f32]>::to_vec)
                    .collect(),
                target: 0,
            },
            spec: Spec {
                text,
                reference,
                prompt,
            },
            steps,
            cfg,
            noise,
            patches,
            context,
            lm: f32s(shaped(dir, "lm0.npy", &[2048])?),
            residual: f32s(shaped(dir, "res0.npy", &[2048])?),
            speech,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn token_casts_cannot_hide_invalid_fixture_ids() {
        for x in [-1., 0.5, u32::MAX as f64 + 1., f64::NAN] {
            assert!(tokens(&[x]).is_err());
        }
        assert_eq!(tokens(&[0., 42.]).unwrap(), [0, 42]);
    }
    #[test]
    fn masks_require_binary_values_and_complements() {
        assert_eq!(masks(&[0., 1.], &[1., 0.]).unwrap(), [false, true]);
        for (a, t) in [
            (vec![2.], vec![0.]),
            (vec![0.], vec![0.]),
            (vec![1.], vec![]),
        ] {
            assert!(masks(&a, &t).is_err());
        }
    }

    #[test]
    fn complete_fixture_shapes_and_missing_payloads() {
        struct Dir(std::path::PathBuf);
        impl Drop for Dir {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let dir = Dir(std::env::temp_dir().join(format!(
            "vcp-fixture-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )));
        std::fs::create_dir(&dir.0).unwrap();
        let array = |name: &str, shape: &[usize], value: f32| {
            let dims = shape.iter().map(|d| format!("{d},")).collect::<String>();
            let header =
                format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}\n");
            let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
            bytes.extend((header.len() as u16).to_le_bytes());
            bytes.extend(header.as_bytes());
            for _ in 0..shape.iter().product::<usize>() {
                bytes.extend(value.to_le_bytes());
            }
            std::fs::write(dir.0.join(name), bytes).unwrap();
        };
        std::fs::write(
            dir.0.join("request.json"),
            r#"{"mode":"continue","text":"Hello","prompt_text":"Before","steps":10,"cfg":2}"#,
        )
        .unwrap();
        array("tokens.npy", &[3], 0.);
        array("audio_mask.npy", &[3], 1.);
        array("text_mask.npy", &[3], 0.);
        array("audio_feat.npy", &[3, PATCH, FEAT], 0.);
        array("noise.npy", &[1, FEAT, PATCH], 0.);
        array("feats.npy", &[4, PATCH, FEAT], 0.);
        array("speech.npy", &[7680], 0.);
        array("lm0.npy", &[2048], 0.);
        array("res0.npy", &[2048], 0.);
        array("prompt16k.npy", &[160], 0.);
        let f = Fixture::load(&dir.0).unwrap();
        assert_eq!((f.context, f.patches.len(), f.noise.len()), (3, 4, 1));
        array("noise.npy", &[1, PATCH, FEAT], 0.); // same elements, wrong axes
        assert!(Fixture::load(&dir.0).is_err());
        array("noise.npy", &[1, FEAT, PATCH], 0.);
        array("speech.npy", &[7679], 0.); // no prefix-only waveform comparison
        assert!(Fixture::load(&dir.0).is_err());
        array("speech.npy", &[7680], 0.);
        std::fs::write(dir.0.join("prompt16k.npy"), b"truncated").unwrap();
        assert!(Fixture::load(&dir.0).is_err());
        array("prompt16k.npy", &[160], 0.);
        std::fs::remove_file(dir.0.join("noise.npy")).unwrap();
        assert!(Fixture::load(&dir.0).is_err());
    }
}
