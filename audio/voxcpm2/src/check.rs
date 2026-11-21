//! Stage by stage against the reference (reference/voxcpm2/ref.py: voxcpm's own code in float32 on the same files),
//! each from the reference's own input.
//!
//! ```text
//!   prompt   request.json (+ ref16k / prompt16k.npy) -> tokens.npy, audio_mask.npy, audio_feat.npy (the VAE's encoder)
//!   prefill  the reference's prompt                   -> lm0.npy, res0.npy
//!   patches  noise.npy forced                         -> feats.npy (each patch from the reference's noise)
//!   vae      feats.npy                                -> speech.npy (48 kHz)
//! ```

use std::path::Path;
use std::time::Instant;

use nextrocm_audio::{Error, Result};

use crate::VoxCpm2;

#[path = "fixtures.rs"]
mod fixtures;

/// A little-endian .npy of float32, int32 or int64 (C order): (shape, values as f64)
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    parse_npy(&b).map_err(|e| Error(format!("{}: {}", path.display(), e.0)))
}

fn parse_npy(b: &[u8]) -> Result<(Vec<usize>, Vec<f64>)> {
    let bad = || Error("invalid/truncated C-order numeric NPY".into());
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(bad());
    }
    let (hl, at) = match b[6] {
        1 => (u16::from_le_bytes([b[8], b[9]]) as usize, 10usize),
        2 | 3 if b.len() >= 12 => (
            u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize,
            12usize,
        ),
        _ => return Err(bad()),
    };
    let end = at.checked_add(hl).ok_or_else(bad)?;
    let header = std::str::from_utf8(b.get(at..end).ok_or_else(bad)?).map_err(|_| bad())?;
    if !header.contains("'fortran_order': False") {
        return Err(bad());
    }
    let dims = header
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(bad)?;
    let shape = dims
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>().map_err(|_| bad()))
        .collect::<Result<Vec<_>>>()?;
    let width = if header.contains("'<f4'") || header.contains("'<i4'") {
        4
    } else if header.contains("'<i8'") {
        8
    } else {
        return Err(bad());
    };
    let count = crate::budget::product(&shape)?;
    let data = b.get(end..).ok_or_else(bad)?;
    if data.len() != crate::budget::product(&[count, width])? {
        return Err(bad());
    }
    let values = data
        .chunks_exact(width)
        .map(|c| {
            if header.contains("'<f4'") {
                f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64
            } else if width == 4 {
                i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64
            } else {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
            }
        })
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error("nonfinite reference tensor".into()));
    }
    Ok((shape, values))
}

fn f32s(v: Vec<f64>) -> Vec<f32> {
    v.into_iter().map(|x| x as f32).collect()
}

pub fn compare(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    if got.len() != want.len() || got.is_empty() || got.iter().chain(want).any(|x| !x.is_finite()) {
        return (f64::INFINITY, f64::INFINITY, 0.0);
    }
    let (mut e2, mut w2, mut g2, mut dot, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (*g as f64, *w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        mx = mx.max((g - w).abs());
    }
    (
        (e2 / w2.max(1e-30)).sqrt(),
        mx,
        dot / (w2.sqrt() * g2.sqrt()).max(1e-30),
    )
}

fn report(log: &mut dyn FnMut(String), worst: &mut f64, what: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        log(format!(
            "{what:<24} sizes differ: {} here, {} in the reference",
            got.len(),
            want.len()
        ));
        *worst = f64::INFINITY;
        return;
    }
    let (rel, mx, cos) = compare(got, want);
    log(format!(
        "{what:<24} rel {rel:.2e}  max {mx:.2e}  cos {cos:.6}"
    ));
    *worst = worst.max(rel);
}

/// CPU-only structure check. Does not verify manifest hashes or numerical parity.
pub fn inspect_fixture(dir: &Path) -> Result<serde_json::Value> {
    let f = fixtures::Fixture::load(dir)?;
    Ok(
        serde_json::json!({"structurally_valid":true,"prompt_rows":f.prompt.tokens.len(),
        "noise_draws":f.noise.len(),"decoder_patches":f.patches.len(),"context_patches":f.context,
        "speech_samples":f.speech.len(),"steps":f.steps,"cfg":f.cfg,
        "numerical_acceptance":"pending native comparison; structure alone is not parity"}),
    )
}

pub fn run(e: &VoxCpm2, dir: &Path, stages: &[&str], log: &mut dyn FnMut(String)) -> Result<f64> {
    run_with_dump(e, dir, stages, log, None)
}

/// Optional native tensor evidence, written only by explicit validators.
pub fn run_with_dump(
    e: &VoxCpm2,
    dir: &Path,
    stages: &[&str],
    log: &mut dyn FnMut(String),
    dump: Option<&Path>,
) -> Result<f64> {
    if stages.is_empty()
        || stages
            .iter()
            .any(|s| !matches!(*s, "prompt" | "prefill" | "patches" | "vae"))
    {
        return Err(Error(
            "select prompt/prefill/patches/vae reference stages".into(),
        ));
    }
    // Validate complete shapes/masks before launching any work. No zero-filled
    // missing noise, swallowed recording errors, or prefix-only comparisons.
    let mut f = fixtures::Fixture::load(dir)?;
    f.prompt.target = e.tokens(&f.spec.text).len();
    if f.prompt.tokens.iter().any(|&t| t as usize >= e.model.vocab) {
        return Err(Error("reference token exceeds model vocabulary".into()));
    }
    let mut worst = 0f64;
    if stages.contains(&"prompt") {
        let p = e.prompt(&f.spec)?;
        let same = p.tokens == f.prompt.tokens && p.audio == f.prompt.audio;
        log(format!(
            "prompt tokens/masks: {}",
            if same { "identical" } else { "DIFFERENT" }
        ));
        if !same {
            worst = f64::INFINITY;
        }
        // Include text rows too: the reference explicitly supplies zero features
        // there. Compare complete arrays, not only selected recording rows.
        report(
            log,
            &mut worst,
            "prompt features",
            &p.patches.concat(),
            &f.prompt.patches.concat(),
        );
    }
    if stages.contains(&"prefill") || stages.contains(&"patches") {
        let forced = stages.contains(&"patches");
        let mut keep = Vec::new();
        let t0 = Instant::now();
        const PREFILL_DONE: &str = "voxcpm2-check-prefill-complete";
        let result = e.generate_patches(
            &f.prompt,
            if forced { f.noise.len() } else { 1 },
            f.steps,
            f.cfg,
            &mut |i| f.noise[i].clone(),
            t0,
            &mut |step| {
                if !forced && step.phase == "prompt" && step.at == 1 {
                    return Err(Error(PREFILL_DONE.into()));
                }
                Ok(())
            },
            Some(&mut keep),
            None,
        );
        let generated = if forced {
            Some(result?)
        } else {
            // The hidden-state downloads precede this callback. Unwind the
            // session without sampling a patch or running its flow transformer.
            match result {
                Err(e) if e.0 == PREFILL_DONE => None,
                Err(e) => return Err(e),
                Ok(_) => return Err(Error("prefill completion callback missing".into())),
            }
        };
        log(format!(
            "prefill/forced-noise wall: {:.6} s (includes session and downloads)",
            t0.elapsed().as_secs_f64()
        ));
        if keep.len() != 2 {
            return Err(Error("missing prefill hidden states".into()));
        }
        report(log, &mut worst, "base LM (last row)", &keep[0], &f.lm);
        if let Some(dir) = dump {
            write_npy(&dir.join("lm0.npy"), &[keep[0].len()], &keep[0])?;
            write_npy(&dir.join("res0.npy"), &[keep[1].len()], &keep[1])?;
        }
        report(
            log,
            &mut worst,
            "residual LM (last row)",
            &keep[1],
            &f.residual,
        );
        if let Some((got, context)) = generated {
            if let Some(dir) = dump {
                write_npy(
                    &dir.join("free_patches.npy"),
                    &[got.len(), 4, 64],
                    &got.concat(),
                )?;
            }
            if context != f.context {
                return Err(Error("continuation context mismatch".into()));
            }
            let reference = f.patches.concat();
            report(
                log,
                &mut worst,
                "all patches (free history)",
                &got.concat(),
                &reference,
            );
            // Reference history excludes the leading decoder context patches.
            let teacher = &f.patches[context..];
            let t0 = Instant::now();
            let (got, context) = e.generate_patches(
                &f.prompt,
                f.noise.len(),
                f.steps,
                f.cfg,
                &mut |i| f.noise[i].clone(),
                t0,
                &mut |_| Ok(()),
                None,
                Some(teacher),
            )?;
            if let Some(dir) = dump {
                write_npy(
                    &dir.join("teacher_patches.npy"),
                    &[got.len(), 4, 64],
                    &got.concat(),
                )?;
            }
            if context != f.context {
                return Err(Error("teacher context mismatch".into()));
            }
            log(format!(
                "teacher-forced wall: {:.6} s",
                t0.elapsed().as_secs_f64()
            ));
            // Compare complete outputs first; per-patch diagnostics cannot hide
            // missing/extra patches, including the continuation context.
            report(
                log,
                &mut worst,
                "teacher-forced, all",
                &got.concat(),
                &reference,
            );
            if got.len() == f.patches.len() {
                for (i, (g, w)) in got.iter().zip(&f.patches).enumerate() {
                    report(log, &mut worst, &format!("teacher patch {i}"), g, w);
                }
            }
        }
    }
    if stages.contains(&"vae") {
        let t0 = Instant::now();
        let wav = e.decode(&f.patches, f.context)?;
        if let Some(dir) = dump {
            write_npy(&dir.join("speech.npy"), &[wav.len()], &wav)?;
        }
        log(format!(
            "vae: {} patches to {:.2} s in {:.6} s",
            f.patches.len(),
            wav.len() as f64 / 48000.0,
            t0.elapsed().as_secs_f64()
        ));
        report(log, &mut worst, "speech", &wav, &f.speech);
    }
    Ok(worst)
}

/// NPY v1 F32 C-order evidence. Never overwrite an earlier diagnostic.
pub fn write_npy(path: &Path, shape: &[usize], values: &[f32]) -> Result<()> {
    use std::io::Write;
    if crate::budget::product(shape)? != values.len() || values.iter().any(|v| !v.is_finite()) {
        return Err(Error("invalid native tensor dump".into()));
    }
    let dims = shape.iter().map(|d| format!("{d}, ")).collect::<String>();
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let length = u16::try_from(header.len()).map_err(|_| Error("NPY header too long".into()))?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| Error(e.to_string()))?;
    file.write_all(b"\x93NUMPY\x01\x00")
        .and_then(|_| file.write_all(&length.to_le_bytes()))
        .and_then(|_| file.write_all(header.as_bytes()))
        .and_then(|_| file.write_all(&crate::talker_bytes(values)))
        .map_err(|e| Error(e.to_string()))
}

/// Replay a fixed supplied history even past its predicted stop. Captures the
/// original class decision and i>2 eligibility without terminating the replay.
pub fn replay(e: &VoxCpm2, dir: &Path, dump: &Path) -> Result<serde_json::Value> {
    replay_impl(e, dir, dump, true)
}

/// Same history through the original LM/encoder/stop path, without flow or VAE.
/// Diagnostic hidden states only; never a sampled-output acceptance result.
pub fn replay_states(e: &VoxCpm2, dir: &Path, dump: &Path) -> Result<serde_json::Value> {
    replay_impl(e, dir, dump, false)
}

/// Run every supplied raw LM row through FSQ, even when its original prompt
/// position was text. This isolates the operator; it is not trajectory replay.
pub fn fsq_states(e: &VoxCpm2, input: &Path, dump: &Path) -> Result<serde_json::Value> {
    let (shape, values) = read_npy(&input.join("raw_lm_history.npy"))?;
    let h = e.model.base.hidden;
    if shape.len() != 2 || shape[0] == 0 || shape[0] > crate::MAX_PATCHES || shape[1] != h {
        return Err(Error("invalid FSQ input shape".into()));
    }
    let s = e.model.session(&e.ops, 1, 1)?;
    let mut traces = [Vec::new(), Vec::new(), Vec::new()];
    for row in values.chunks_exact(h) {
        let row: Vec<f32> = row.iter().map(|&v| v as f32).collect();
        let result = e.model.fsq_trace(&e.ops, &e.nsd, &s, &row)?;
        for (trace, values) in traces.iter_mut().zip(result) {
            trace.extend(values);
        }
    }
    for (name, values) in ["fsq_projected", "fsq_codes", "fsq_output"]
        .iter()
        .zip(&traces)
    {
        write_npy(
            &dump.join(format!("{name}.npy")),
            &[shape[0], values.len() / shape[0]],
            values,
        )?;
    }
    Ok(serde_json::json!({"rows":shape[0],"operator_rows":1,
        "scope":"diagnostic exact-input FSQ; every row quantized; no trajectory/admission"}))
}
fn replay_impl(e: &VoxCpm2, dir: &Path, dump: &Path, run_flow: bool) -> Result<serde_json::Value> {
    use nextrocm_engine_core::DevBuf;
    let f = fixtures::Fixture::load(dir)?;
    let (m, ops, nsd) = (&e.model, &e.ops, &e.nsd);
    let l = f.prompt.tokens.len();
    let h = m.base.hidden;
    let n = f.noise.len();
    let mut session = m.session(ops, l, n)?;
    if !run_flow {
        session.base.prefill_trace = Some(dump.to_path_buf());
    }
    let mut rows = vec![0f32; l * h];
    for (i, &t) in f.prompt.tokens.iter().enumerate() {
        if t as usize >= m.vocab {
            return Err(Error("reference token exceeds vocabulary".into()));
        }
        if !f.prompt.audio[i] {
            rows[i * h..(i + 1) * h].copy_from_slice(&e.weights.rows_f32(
                "base_lm.embed_tokens.weight",
                t as usize,
                t as usize + 1,
            )?);
        }
    }
    m.prefill(
        ops,
        nsd,
        &session,
        &rows,
        &f.prompt.audio,
        &f.prompt.patches,
    )?;
    let raw: DevBuf = session.base.out.view((l - 1) * h * 4, h * 4)?;
    write_npy(&dump.join("lm0.npy"), &[h], &raw.to_f32()?)?;
    write_npy(&dump.join("res0.npy"), &[h], &session.res_h.to_f32()?)?;
    let mut previous = f.prompt.patches[l - 1].clone();
    let mut output = f.patches[..f.context].to_vec();
    let (mut logits, mut decisions, mut effective) = (Vec::new(), Vec::new(), Vec::new());
    let (mut raw_states, mut lm_states, mut res_states, mut mu_states) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        m.mu(ops, nsd, &session)?;
        if run_flow {
            output.push(m.sample(ops, nsd, &session, &f.noise[i], &previous, f.steps, f.cfg)?);
        } else {
            let offset = if i == 0 { (l - 1) * h * 4 } else { 0 };
            raw_states.extend(session.base.out.view(offset, h * 4)?.to_f32()?);
            lm_states.extend(session.lm.to_f32()?);
            res_states.extend(session.res_h.to_f32()?);
            mu_states.extend(session.mu.to_f32()?);
        }
        let stop = m.stop_logits(ops, nsd, &session)?;
        logits.extend(stop);
        let decision = stop[1] > stop[0];
        decisions.push(if decision { 1. } else { 0. });
        effective.push(if i > 2 && decision { 1. } else { 0. });
        previous = f.patches[f.context + i].clone();
        if i + 1 < n {
            m.advance(ops, nsd, &session, &previous, l + i)?;
        }
    }
    ops.gpu.sync()?;
    if run_flow {
        write_npy(
            &dump.join("replay_patches.npy"),
            &[output.len(), 4, 64],
            &output.concat(),
        )?;
    } else {
        for (name, values) in [
            ("raw_lm_history.npy", &raw_states),
            ("lm_history.npy", &lm_states),
            ("res_history.npy", &res_states),
            ("mu_history.npy", &mu_states),
        ] {
            write_npy(&dump.join(name), &[n, values.len() / n], values)?;
        }
    }
    write_npy(&dump.join("stop_logits.npy"), &[n, 2], &logits)?;
    write_npy(&dump.join("stop_decisions.npy"), &[n], &decisions)?;
    write_npy(&dump.join("stop_effective.npy"), &[n], &effective)?;
    Ok(
        serde_json::json!({"patches":n,"context":f.context,"state_only":!run_flow,"stop_rule":"i > 2 and logit1 > logit0",
        "first_effective_stop":effective.iter().position(|&v| v==1.),"termination":"fixed_history_length",
        "numerical_acceptance":"pending matched controls and frozen gates"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn npy() -> Vec<u8> {
        let h = "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }\n";
        let mut b = b"\x93NUMPY\x01\x00".to_vec();
        b.extend((h.len() as u16).to_le_bytes());
        b.extend(h.as_bytes());
        b.extend(1f32.to_le_bytes());
        b.extend((-2f32).to_le_bytes());
        b
    }
    #[test]
    fn checks_npy_header_shape_and_truncation() {
        let b = npy();
        assert_eq!(parse_npy(&b).unwrap(), (vec![2], vec![1., -2.]));
        for n in 0..b.len() {
            assert!(parse_npy(&b[..n]).is_err());
        }
    }
    #[test]
    fn nonfinite_or_short_comparison_never_passes() {
        assert!(compare(&[f32::NAN], &[0.]).0.is_infinite());
        assert!(compare(&[1.], &[1., 2.]).0.is_infinite());
        assert_eq!(compare(&[1., -2.], &[1., -2.]).0, 0.);
    }
    #[test]
    fn report_rejects_matching_prefix_with_missing_patch() {
        let mut worst = 0.;
        report(&mut |_| {}, &mut worst, "teacher-forced", &[1.], &[1., 2.]);
        assert!(worst.is_infinite());
    }
}
