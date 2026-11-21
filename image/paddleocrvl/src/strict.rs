//! Fail-closed independent fixture validation. Legacy `check` is diagnostic only.
//! Gates and SHA-256 identities are frozen in the CPU-produced contract before native runs.
use crate::{positions, PaddleOcrVl};
use nextrocm_engine_core::{DevBuf, Error, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn fail(s: impl Into<String>) -> Error {
    Error(s.into())
}
#[derive(Debug)]
pub struct Metrics {
    pub max_relative: f64,
    pub rms_relative: f64,
    pub cosine: f64,
}
pub fn metrics(a: &[f32], b: &[f32]) -> Result<Metrics> {
    if a.is_empty() || a.len() != b.len() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(fail("empty, mismatched or nonfinite tensors"));
    }
    let (mut max, mut scale, mut diff, mut aa, mut bb, mut ab) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        max = max.max((x - y).abs());
        scale = scale.max(y.abs());
        diff += (x - y).powi(2);
        aa += x * x;
        bb += y * y;
        ab += x * y;
    }
    Ok(Metrics {
        max_relative: max / scale.max(1e-30),
        rms_relative: (diff / bb.max(1e-60)).sqrt(),
        cosine: if aa == 0. && bb == 0. {
            1.
        } else {
            ab / (aa * bb).sqrt().max(1e-60)
        },
    })
}
fn gates(m: &Metrics, g: &Value) -> Result<()> {
    let get = |k: &str| {
        g[k].as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| fail(format!("missing/invalid gate {k}")))
    };
    let (max, rms, cos) = (
        get("max_relative")?,
        get("rms_relative")?,
        get("cosine_min")?,
    );
    if max < 0. || rms < 0. || !(0.0..=1.0).contains(&cos) {
        return Err(fail("invalid gate bounds"));
    }
    if m.max_relative > max || m.rms_relative > rms || m.cosine < cos {
        return Err(fail(format!("numeric gate failed {m:?}, limits {g}")));
    }
    Ok(())
}
pub struct Fixture {
    root: PathBuf,
    pub contract: Value,
}
impl Fixture {
    pub(crate) fn validate_precision(&self, decoder_f32: bool) -> Result<()> {
        let variant = self.contract["precision_variant"]
            .as_str()
            .unwrap_or("decoder-f16-control");
        match (decoder_f32, variant) {
            (false, "decoder-f16-control") => {
                if self.contract["source"]["control"]["mode"] != "native-precision" {
                    return Err(fail(
                        "F16 decoder requires its original precision-control fixture",
                    ));
                }
            }
            (true, "decoder-f32-original") => {
                if self.contract["precision_protocol_sha256"]
                    != sha256(include_bytes!("../reference/protocol-f32.json"))
                {
                    return Err(fail("F32 precision protocol SHA mismatch"));
                }
                if self.contract["source"]["original"]["mode"] != "original"
                    || self.contract["source"]["control"] != self.contract["source"]["original"]
                {
                    return Err(fail(
                        "F32 control must be the unchanged original CPU source",
                    ));
                }
                let stages = self.contract["gates"]
                    .as_object()
                    .ok_or_else(|| fail("missing F32 stages"))?;
                for name in stages.keys() {
                    let a = &self.contract["files"][format!("{name}.f32")];
                    let b = &self.contract["files"][format!("original-{name}.f32")];
                    if !a.is_object() || a != b {
                        return Err(fail(format!(
                            "{name}: F32 control is not byte-identical to original"
                        )));
                    }
                }
                let budget: Value = serde_json::from_slice(&self.bytes("precision-budget.json")?)
                    .map_err(|e| fail(e.to_string()))?;
                if budget["cpu_precision_allowance"] != 0
                    || budget["precision_variant"] != "decoder-f32-original"
                {
                    return Err(fail("F32 fixture must have zero CPU precision allowance"));
                }
            }
            _ => {
                return Err(fail(
                    "runtime decoder precision differs from frozen fixture",
                ))
            }
        }
        Ok(())
    }
    pub fn open(root: &Path, expected_sha: &str) -> Result<Self> {
        let bytes = std::fs::read(root.join("contract.json")).map_err(|e| fail(e.to_string()))?;
        if expected_sha.len() != 64 || sha256(&bytes) != expected_sha {
            return Err(fail("frozen contract hash mismatch"));
        }
        let contract: Value = serde_json::from_slice(&bytes).map_err(|e| fail(e.to_string()))?;
        if contract["version"] != 1 || !contract["source"].is_object() {
            return Err(fail("contract version/source missing"));
        }
        let s = Self {
            root: root.into(),
            contract,
        };
        let files = s.contract["files"]
            .as_object()
            .ok_or_else(|| fail("no fixture files"))?;
        for name in files.keys() {
            s.bytes(name)?;
        }
        Ok(s)
    }
    fn bytes(&self, name: &str) -> Result<Vec<u8>> {
        if name.contains('/') || name.contains('\\') || name == ".." {
            return Err(fail("invalid fixture filename"));
        }
        let hash = self.contract["files"][name]["sha256"]
            .as_str()
            .ok_or_else(|| fail(format!("unmanifested file {name}")))?;
        let b = std::fs::read(self.root.join(name)).map_err(|e| fail(format!("{name}: {e}")))?;
        if sha256(&b) != hash {
            return Err(fail(format!("{name}: hash mismatch")));
        }
        Ok(b)
    }
    pub(crate) fn nums(&self, name: &str) -> Result<Vec<u32>> {
        String::from_utf8(self.bytes(name)?)
            .map_err(|e| fail(e.to_string()))?
            .split_whitespace()
            .map(|v| {
                v.parse()
                    .map_err(|_| fail(format!("{name}: invalid integer")))
            })
            .collect()
    }
    pub(crate) fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let name = format!("{name}.f32");
        let b = self.bytes(&name)?;
        let dims = self.contract["files"][&name]["shape"]
            .as_array()
            .ok_or_else(|| fail(format!("{name}: shape missing")))?;
        let n = dims
            .iter()
            .try_fold(1usize, |n, v| {
                v.as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .filter(|v| *v > 0)
                    .and_then(|v| n.checked_mul(v))
            })
            .ok_or_else(|| fail("shape overflow/zero"))?;
        if dims.is_empty() || n.checked_mul(4) != Some(b.len()) {
            return Err(fail(format!("{name}: shape/byte length mismatch")));
        }
        let out: Vec<f32> = b
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(fail(format!("{name}: nonfinite")));
        }
        Ok(out)
    }
}
/// Validate all emitted layers against a precision-matched independent CPU control.
/// `gen.txt` supplies the reference tokens; it is never replaced by native predictions.
pub fn run(m: &PaddleOcrVl, f: &Fixture, log: &mut dyn FnMut(String)) -> Result<Value> {
    f.validate_precision(m.decoder.is_f32())?;
    let grid = f.nums("grid.txt")?;
    if grid.len() != 3
        || grid[0] != 1
        || grid[1] == 0
        || grid[2] == 0
        || grid[1] % 2 != 0
        || grid[2] % 2 != 0
    {
        return Err(fail("invalid image grid"));
    }
    let (gh, gw) = (grid[1] as usize, grid[2] as usize);
    let mut required: BTreeSet<String> = ["v_embd", "v_out", "projected", "logits_last"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    required.extend((0..m.vision.layer_count()).map(|n| format!("v_layer-{n}")));
    required.extend((0..m.decoder.g.layers).map(|n| format!("l_out-{n}")));
    let protocol: Value = serde_json::from_str(include_str!("../reference/protocol.json"))
        .map_err(|e| fail(e.to_string()))?;
    if f.contract["protocol_sha256"] != sha256(include_bytes!("../reference/protocol.json")) {
        return Err(fail("protocol SHA mismatch"));
    }
    let generated = f.nums("gen.txt")?;
    if generated != f.nums("original-gen.txt")? {
        return Err(fail(
            "CPU precision control differs from original greedy tokens",
        ));
    }
    if generated.is_empty() {
        return Err(fail("empty teacher-forced sequence"));
    }
    required.extend((0..generated.len() - 1).map(|n| format!("step_logits-{n}")));
    let declared: BTreeSet<String> = f.contract["gates"]
        .as_object()
        .ok_or_else(|| fail("no stage gates"))?
        .keys()
        .cloned()
        .collect();
    if declared != required {
        return Err(fail(format!(
            "stage coverage differs: required {required:?}, declared {declared:?}"
        )));
    }
    // Require the original all-F32/control budget report to be present and hash pinned.
    f.bytes("precision-budget.json")?;
    for name in &required {
        f.floats(name)?;
        f.floats(&format!("original-{name}"))?;
    }
    for name in &required {
        let kind = if name.contains("logits") {
            "logits"
        } else if name.starts_with("l_out-") {
            "decoder"
        } else {
            "vision"
        };
        if f.contract["gates"][name] != protocol[kind] {
            return Err(fail(format!("{name}: gates differ from frozen protocol")));
        }
    }
    if f.contract["preprocessing_max_abs"] != protocol["preprocessing_max_abs"] {
        return Err(fail("preprocessing gate differs"));
    }
    let prompt = f.contract["prompt"]
        .as_str()
        .ok_or_else(|| fail("missing prompt"))?;
    let want = f.nums("tokens.txt")?;
    let ids = m.prompt_tokens(prompt, gh * gw / 4);
    if ids != want {
        return Err(fail("prompt tokens differ"));
    }
    if ids
        .iter()
        .chain(&generated)
        .any(|&t| t as usize >= m.decoder.g.vocab)
    {
        return Err(fail("token outside vocabulary"));
    }
    let shape = |name: &str, want: &[usize]| -> Result<()> {
        let got: Vec<u64> = f.contract["files"][format!("{name}.f32")]["shape"]
            .as_array()
            .ok_or_else(|| fail("missing shape"))?
            .iter()
            .filter_map(|v| v.as_u64())
            .filter(|&v| v != 1)
            .collect();
        let want: Vec<u64> = want.iter().map(|&v| v as u64).filter(|&v| v != 1).collect();
        if got != want {
            return Err(fail(format!("{name}: logical shape {got:?} != {want:?}")));
        }
        Ok(())
    };
    shape("pixel_values", &[gh * gw, 588])?;
    for name in &required {
        let dims = if name.contains("logits") {
            vec![m.decoder.g.vocab]
        } else if name.starts_with("l_out-") {
            vec![ids.len(), m.decoder.g.n_embd]
        } else if name == "projected" {
            vec![gh * gw / 4, 1024]
        } else {
            vec![gh * gw, m.vision.c]
        };
        shape(name, &dims)?;
        shape(&format!("original-{name}"), &dims)?;
    }
    let pos = positions(&ids, gh / 2, gw / 2);
    let reference = f.nums("positions.txt")?;
    let flat: Vec<u32> = (0..3).flat_map(|a| pos.iter().map(move |p| p[a])).collect();
    if flat != reference {
        return Err(fail("multimodal positions differ"));
    }
    // Native preprocessing of the exact original encoded input is a separate control.
    let picture = nextrocm_image::Picture::decode(&f.bytes("input.png")?).map_err(fail)?;
    let cap = if prompt.starts_with("Spotting") {
        crate::SPOTTING_MAX_PIXELS
    } else {
        crate::prep::MAX_PIXELS
    };
    let prepared = crate::prep::see(&picture, cap);
    let reference_pixels = f.floats("pixel_values")?;
    if (prepared.gh, prepared.gw) != (gh, gw) || prepared.patches.len() != reference_pixels.len() {
        return Err(fail("preprocessing shape mismatch"));
    }
    let prep_limit = f.contract["preprocessing_max_abs"]
        .as_f64()
        .filter(|x| x.is_finite() && *x >= 0.)
        .ok_or_else(|| fail("invalid preprocessing gate"))?;
    if prepared
        .patches
        .iter()
        .zip(&reference_pixels)
        .any(|(a, b)| !a.is_finite() || (*a as f64 - *b as f64).abs() > prep_limit)
    {
        return Err(fail("preprocessing values differ"));
    }
    let mut rows = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let mut compare = |name: &str, a: &[f32]| -> Result<()> {
        if !seen.insert(name.to_string()) {
            return Err(fail(format!("duplicate stage {name}")));
        }
        let b = f.floats(name)?;
        let metric = metrics(a, &b)?;
        gates(&metric, &f.contract["gates"][name])?;
        let original = f.floats(&format!("original-{name}"))?;
        let cpu_gap = metrics(&b, &original)?;
        let to_original = metrics(a, &original)?;
        let scale = |x: &[f32]| {
            x.iter()
                .map(|v| (*v as f64).abs())
                .fold(0f64, f64::max)
                .max(1e-30)
        };
        let norm = |x: &[f32]| {
            x.iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-30)
        };
        let gate = &f.contract["gates"][name];
        let max_budget = cpu_gap.max_relative
            + gate["max_relative"].as_f64().unwrap() * scale(&b) / scale(&original);
        let rms_budget = cpu_gap.rms_relative
            + gate["rms_relative"].as_f64().unwrap() * norm(&b) / norm(&original);
        if to_original.max_relative > max_budget || to_original.rms_relative > rms_budget {
            return Err(fail(format!("{name}: original-F32 budget exceeded")));
        }
        log(format!(
            "{name}: {metric:?}; original max/rms budget {max_budget}/{rms_budget}"
        ));
        rows.insert(name.to_string(),json!({"values":a.len(),"max_relative":metric.max_relative,"rms_relative":metric.rms_relative,"cosine":metric.cosine}));
        Ok(())
    };
    let img = m.vision.see(
        &m.nsd,
        &f.floats("pixel_values")?,
        gh,
        gw,
        &mut |name, b: &DevBuf| compare(name, &b.to_f32()?),
    )?;
    let x = m.embed(&ids, &img)?;
    m.nsd.wait()?;
    let mut cache = m.decoder.cache(ids.len() + generated.len())?;
    let mut logits = m
        .decoder
        .pass(&m.nsd, &mut cache, &x, &pos, &mut |name, b| {
            compare(name, &b.to_f32()?)
        })?;
    compare("logits_last", &logits)?;
    let first = pos
        .last()
        .ok_or_else(|| fail("empty positions"))?
        .iter()
        .max()
        .copied()
        .unwrap()
        + 1;
    let stop = f.contract["stop"]
        .as_str()
        .ok_or_else(|| fail("missing stop reason"))?;
    if stop != "eos" && stop != "max_tokens" {
        return Err(fail("numeric fixture stop must be eos or max_tokens"));
    }
    for (i, &token) in generated.iter().enumerate() {
        let name = if i == 0 {
            "logits_last".to_string()
        } else {
            format!("step_logits-{}", i - 1)
        };
        for prefix in ["", "original-"] {
            let cpu = f.floats(&format!("{prefix}{name}"))?;
            let best = cpu
                .iter()
                .enumerate()
                .fold(0, |best, (j, &v)| if v > cpu[best] { j } else { best })
                as u32;
            if best != token {
                return Err(fail(format!(
                    "CPU {prefix}{name} argmax does not match gen.txt"
                )));
            }
        }
        let best = logits
            .iter()
            .enumerate()
            .fold(0, |best, (j, &v)| if v > logits[best] { j } else { best })
            as u32;
        if best != token {
            return Err(fail(format!("greedy token {i}: {best} != {token}")));
        }
        if token == m.eos && (i + 1 != generated.len() || stop != "eos") {
            return Err(fail("unexpected EOS"));
        }
        if i + 1 < generated.len() {
            logits = m
                .decoder
                .step(&m.nsd, &mut cache, token, first + i as u32)?;
            compare(&format!("step_logits-{i}"), &logits)?;
        }
    }
    if (generated.last() == Some(&m.eos)) != (stop == "eos") {
        return Err(fail("stop reason differs"));
    }
    drop(compare);
    if seen != required {
        return Err(fail("missing executed stages"));
    }
    Ok(
        json!({"passed":true,"precision_variant":if m.decoder.is_f32() { "decoder-f32-original" } else { "decoder-f16-control" },"stages":rows,"tokens":generated,"stop":stop}),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn precision_contract_rejects_cross_dtype_and_nonzero_allowance() {
        let dir = std::env::temp_dir().join(format!("pocr-precision-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let budget = br#"{"cpu_precision_allowance":0,"precision_variant":"decoder-f32-original"}"#;
        std::fs::write(dir.join("precision-budget.json"), budget).unwrap();
        let source = json!({"mode":"original"});
        let mut fixture = Fixture {
            root: dir.clone(),
            contract: json!({
                "precision_variant":"decoder-f32-original",
                "precision_protocol_sha256":sha256(include_bytes!("../reference/protocol-f32.json")),
                "source":{"original":source,"control":source},
                "gates":{"l_out-0":{}},
                "files":{"l_out-0.f32":{"sha256":"same","shape":[1]},
                    "original-l_out-0.f32":{"sha256":"same","shape":[1]},
                    "precision-budget.json":{"sha256":sha256(budget)}}
            }),
        };
        assert!(fixture.validate_precision(true).is_ok());
        assert!(fixture.validate_precision(false).is_err());
        fixture.contract["files"]["l_out-0.f32"]["sha256"] = json!("different");
        assert!(fixture.validate_precision(true).is_err());
        fixture.contract["files"]["l_out-0.f32"]["sha256"] = json!("same");
        let nonzero =
            br#"{"cpu_precision_allowance":0.001,"precision_variant":"decoder-f32-original"}"#;
        std::fs::write(dir.join("precision-budget.json"), nonzero).unwrap();
        fixture.contract["files"]["precision-budget.json"]["sha256"] = json!(sha256(nonzero));
        assert!(fixture.validate_precision(true).is_err());
        fixture.contract = json!({"source":{"control":{"mode":"native-precision"}}});
        assert!(fixture.validate_precision(false).is_ok());
        assert!(fixture.validate_precision(true).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn fixture_rejects_corruption_malformed_shape_and_missing_files() {
        let dir = std::env::temp_dir().join(format!("pocr-fixture-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let b = 1f32.to_le_bytes();
        std::fs::write(dir.join("x.f32"), b).unwrap();
        let value = json!({"version":1,"source":{"fixture":"unit"},"files":{"x.f32":{"sha256":sha256(&b),"shape":[2]}}});
        let bytes = serde_json::to_vec(&value).unwrap();
        std::fs::write(dir.join("contract.json"), &bytes).unwrap();
        let f = Fixture::open(&dir, &sha256(&bytes)).unwrap();
        assert!(f.floats("x").is_err());
        assert!(f.bytes("missing").is_err());
        assert!(f.bytes("../x").is_err());
        assert!(Fixture::open(&dir, &"0".repeat(64)).is_err());
        std::fs::write(dir.join("x.f32"), 2f32.to_le_bytes()).unwrap();
        assert!(Fixture::open(&dir, &sha256(&bytes)).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn strict_metrics_reject_truncation_and_nonfinite() {
        assert!(metrics(&[1.], &[1., 2.]).is_err());
        assert!(metrics(&[], &[]).is_err());
        assert!(metrics(&[f32::NAN], &[1.]).is_err());
        assert!(metrics(&[1.], &[f32::INFINITY]).is_err());
    }
    #[test]
    fn independent_metrics_and_gate_failure() {
        let m = metrics(&[3., 5.], &[3., 4.]).unwrap();
        assert_eq!(m.max_relative, 0.25);
        assert_eq!(m.rms_relative, 0.2);
        assert!(gates(
            &m,
            &json!({"max_relative":0.2,"rms_relative":0.2,"cosine_min":0.9})
        )
        .is_err());
        assert!(gates(&m, &json!({})).is_err());
        assert_eq!(metrics(&[0.], &[0.]).unwrap().cosine, 1.);
    }
}
