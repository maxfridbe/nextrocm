//! Parent-serialized diagnostic only. NEXTROCM_LIB must name the tested runtime.
//! paired_dump MODEL_DIR REFERENCE_DIR NEW_OUTPUT_DIR FLAG_0_OR_1 GPU_INDEX
//! See tests/compare_dumps.py; no inference runs in CPU unit tests.
use nextrocm_audio::{
    AudioEngine, Error, LoadOptions, Result, Timed, TranscribeRequest, Transcript,
};
use nextrocm_audio_whisper::{Whisper, align, files_in, model::Taps, text};
use nextrocm_engine_core::Gpu;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::c_void,
    path::{Path, PathBuf},
    time::Instant,
};
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(err(message)) }
}
fn sha(path: &Path) -> Result<String> {
    let o = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(err)?;
    let s = String::from_utf8(o.stdout).map_err(err)?;
    let h = s.split_whitespace().next().unwrap_or("");
    need(
        o.status.success() && h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()),
        "sha256sum failed",
    )?;
    Ok(h.into())
}
fn product(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &v| {
        n.checked_mul(v).ok_or_else(|| err("shape overflow"))
    })
}
// Strict independent reader: C-order little-endian f32/i64 only; reject trailing
// bytes, truncated headers, invalid shape tokens, and unsupported versions.
fn npy(path: &Path, dtype: &str) -> Result<(Vec<usize>, Vec<u8>)> {
    let b = std::fs::read(path).map_err(err)?;
    need(
        b.len() >= 10 && &b[..6] == b"\x93NUMPY",
        "bad npy magic/header",
    )?;
    let (at, hl) = match b[6] {
        1 => (10usize, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 => {
            need(b.len() >= 12, "truncated npy v2")?;
            (
                12,
                u32::from_le_bytes(b[8..12].try_into().map_err(err)?) as usize,
            )
        }
        _ => return Err(err("unsupported npy version")),
    };
    let end = at.checked_add(hl).ok_or_else(|| err("header overflow"))?;
    need(end <= b.len(), "truncated npy header")?;
    let h = std::str::from_utf8(&b[at..end]).map_err(err)?;
    need(
        h.contains(&format!("'descr': '{dtype}'")) && h.contains("'fortran_order': False"),
        "npy dtype/order mismatch",
    )?;
    let shape = h
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| err("missing npy shape"))?;
    let mut dims = Vec::new();
    for part in shape.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        dims.push(part.parse::<usize>().map_err(err)?);
    }
    need(
        !dims.is_empty() && dims.iter().all(|&d| d > 0),
        "empty npy shape",
    )?;
    let width = match dtype {
        "<f4" => 4,
        "<i8" => 8,
        _ => return Err(err("unsupported dtype")),
    };
    let bytes = product(&dims)?
        .checked_mul(width)
        .ok_or_else(|| err("npy length overflow"))?;
    need(b.len() - end == bytes, "npy exact byte length mismatch")?;
    Ok((dims, b[end..].to_vec()))
}
fn floats(path: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let (s, b) = npy(path, "<f4")?;
    let x: Vec<f32> = b
        .chunks_exact(4)
        .map(|a| f32::from_le_bytes(a.try_into().unwrap()))
        .collect();
    need(x.iter().all(|x| x.is_finite()), "nonfinite fixture")?;
    Ok((s, x))
}
fn tokens(path: &Path, vocab: usize, ctx: usize) -> Result<Vec<u32>> {
    let (s, b) = npy(path, "<i8")?;
    need(s.len() == 1 && s[0] <= ctx && s[0] <= 448, "token geometry")?;
    b.chunks_exact(8)
        .map(|a| {
            let t = i64::from_le_bytes(a.try_into().unwrap());
            need(t >= 0 && (t as usize) < vocab, "token outside vocabulary")?;
            Ok(t as u32)
        })
        .collect()
}
struct Dump {
    dir: PathBuf,
    tensors: BTreeMap<String, Value>,
}
impl Dump {
    fn tensor(&mut self, name: &str, shape: &[usize], x: &[f32]) -> Result<()> {
        need(
            product(shape)? == x.len() && x.iter().all(|v| v.is_finite()),
            "output geometry/nonfinite",
        )?;
        need(!self.tensors.contains_key(name), "duplicate tensor")?;
        let path = self.dir.join(format!("{name}.f32"));
        let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).map_err(err)?;
        self.tensors.insert(name.into(),json!({"shape":shape,"dtype":"<f4","file":format!("{name}.f32"),"bytes":bytes.len(),"sha256":sha(&path)?}));
        Ok(())
    }
}
fn metrics(got: &[f32], want: &[f32], width: usize) -> Result<Value> {
    need(
        !got.is_empty()
            && got.len() == want.len()
            && width > 0
            && got.len() % width == 0
            && got.iter().chain(want).all(|v| v.is_finite()),
        "reference metric geometry/finite",
    )?;
    let (mut e2, mut w2, mut g2, mut dot, mut max) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&g, &w) in got.iter().zip(want) {
        let (g, w) = (g as f64, w as f64);
        e2 += (g - w) * (g - w);
        w2 += w * w;
        g2 += g * g;
        dot += g * w;
        max = max.max((g - w).abs());
    }
    let argmax = |r: &[f32]| {
        r.iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &x)| {
                if x > best.1 { (i, x) } else { best }
            })
            .0
    };
    let a: Vec<usize> = got.chunks_exact(width).map(argmax).collect();
    let b: Vec<usize> = want.chunks_exact(width).map(argmax).collect();
    Ok(
        json!({"elements":got.len(),"max_abs":max,"rms":(e2/got.len() as f64).sqrt(),
        "relative_l2":if w2>0.{Some((e2/w2).sqrt())}else{None},"reference_zero_norm":w2==0.,
        "cosine":if w2>0.&&g2>0.{Some(dot/(w2*g2).sqrt())}else{None},
        "row_argmax_native":a,"row_argmax_reference":b,"scope":"Diagnostic CPU reference metrics, no invented tolerance or automatic accuracy admission"}),
    )
}
fn snapshot(gpu: &Gpu, expected: u64, phase: &str, records: &mut Vec<Value>) -> Result<()> {
    let (live, peak) = stats(gpu)?;
    records.push(json!({"phase":phase,"live_bytes":live,"peak_bytes":peak}));
    need(
        live == expected,
        "per-request/cancellation allocation growth",
    )
}
fn stats(gpu: &Gpu) -> Result<(u64, u64)> {
    gpu.sync()?;
    let a = nextrocm_engine_core::api()?;
    let p = a.symbol("nr_allocation_stats");
    need(!p.is_null(), "missing allocation stats")?;
    type F = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: exact native ABI, library and GPU stay live, valid output pointers.
    let f: F = unsafe { std::mem::transmute(p) };
    let (mut live, mut peak) = (0, 0);
    need(
        unsafe { f(gpu.raw(), &mut live, &mut peak) } == 0,
        "allocation stats failed",
    )?;
    need(peak < 8_000_000_000, "tracked device peak exceeds cap")?;
    Ok((live, peak))
}
fn timed(v: &Timed) -> Result<Value> {
    need(
        v.start.is_finite()
            && v.end.is_finite()
            && v.start <= v.end
            && v.probability.is_none_or(|p| p.is_finite()),
        "nonfinite/invalid transcript timing",
    )?;
    Ok(json!({"start":v.start,"end":v.end,"text":v.text,"probability":v.probability}))
}
fn transcript(t: &Transcript) -> Result<Value> {
    need(
        t.duration.is_finite()
            && t.duration > 0.
            && !t.text.trim().is_empty()
            && !t.segments.is_empty()
            && !t.words.is_empty(),
        "empty/nonfinite words transcription",
    )?;
    Ok(
        json!({"text":t.text,"language":t.language,"duration":t.duration,
        "segments":t.segments.iter().map(timed).collect::<Result<Vec<_>>>()?,
        "words":t.words.iter().map(timed).collect::<Result<Vec<_>>>()?}),
    )
}
fn words(w: &[align::Word]) -> Result<Value> {
    need(
        !w.is_empty()
            && w.iter().all(|v| {
                v.start.is_finite()
                    && v.end.is_finite()
                    && v.probability.is_finite()
                    && v.start <= v.end
            }),
        "invalid alignment words",
    )?;
    Ok(json!(w.iter().map(|v|json!({"word":v.word,"tokens":v.tokens,"start":v.start,"end":v.end,"probability":v.probability})).collect::<Vec<_>>()))
}
// Reconstruct the original CPU alignment preprocessing without changing the
// production path, retain its full cost/path, and compare its words to the
// production window_words call below.
fn dtw_cost(qk: &[Vec<f32>], rows: usize, m: usize, sot: usize) -> Result<Vec<f32>> {
    need(
        !qk.is_empty() && rows > sot + 1,
        "empty alignment heads/text",
    )?;
    let mut mean = vec![0f32; rows * m];
    for h in qk {
        need(
            h.len() == rows * m && h.iter().all(|x| x.is_finite()),
            "QK geometry",
        )?;
        let mut w = vec![0f32; rows * m];
        for r in 0..rows {
            let src = &h[r * m..(r + 1) * m];
            let mx = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let z: f64 = src.iter().map(|x| ((*x - mx) as f64).exp()).sum();
            for (o, x) in w[r * m..(r + 1) * m].iter_mut().zip(src) {
                *o = (((*x - mx) as f64).exp() / z) as f32;
            }
        }
        for c in 0..m {
            let (mut s, mut s2) = (0f64, 0f64);
            for r in 0..rows {
                let x = w[r * m + c] as f64;
                s += x;
                s2 += x * x;
            }
            let mu = s / rows as f64;
            let sd = (s2 / rows as f64 - mu * mu).max(0.).sqrt().max(1e-12);
            for r in 0..rows {
                w[r * m + c] = ((w[r * m + c] as f64 - mu) / sd) as f32;
            }
        }
        align::median_filter(&mut w, m, 7);
        for (a, b) in mean.iter_mut().zip(w) {
            *a += b / qk.len() as f32;
        }
    }
    Ok(mean[sot * m..(rows - 1) * m].iter().map(|x| -x).collect())
}
fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    need(
        matches!(args.len(), 6 | 7),
        "usage: paired_dump MODEL_DIR REFERENCE_DIR NEW_OUTPUT_DIR LINEAR_FLAG GPU_INDEX [ATTENTION_FLAG]",
    )?;
    let model = Path::new(&args[1]);
    let reference = Path::new(&args[2]);
    let out = PathBuf::from(&args[3]);
    need(args[4] == "0" || args[4] == "1", "flag must be 0 or 1")?;
    let attention = args.get(6).map(String::as_str).unwrap_or("0");
    need(
        matches!(
            (args[4].as_str(), attention),
            ("0", "0") | ("1", "0") | ("1", "1")
        ),
        "unsupported paired controls",
    )?;
    let runtime = PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or_else(|| err("explicit NEXTROCM_LIB required"))?,
    );
    need(
        runtime.is_absolute() && runtime.is_file(),
        "runtime must be absolute file",
    )?;
    std::fs::create_dir(&out).map_err(err)?;
    let mut d = Dump {
        dir: out.clone(),
        tensors: BTreeMap::new(),
    };
    let mut hashes = BTreeMap::new();
    let mut paths = BTreeMap::new();
    for (role, p) in files_in(model)? {
        paths.insert(format!("model/{role}"), p);
    }
    for name in [
        "audio16k.npy",
        "mel.npy",
        "tokens.npy",
        "align_tokens.npy",
        "align.json",
        "enc.npy",
        "logits.npy",
        "align_qk.npy",
        "result.json",
    ] {
        paths.insert(format!("reference/{name}"), reference.join(name));
    }
    paths.insert("runtime".into(), runtime);
    paths.insert("executable".into(), std::env::current_exe().map_err(err)?);
    for (k, p) in &paths {
        hashes.insert(k.clone(), sha(p)?);
    }
    let mut options = LoadOptions::default();
    options
        .settings
        .insert("NS_WHISPER_LINEAR_F16_TILED".into(), args[4].clone());
    options
        .settings
        .insert("NS_WHISPER_ATTENTION_TILED".into(), attention.into());
    options
        .settings
        .insert("NS_WHISPER_PROFILE".into(), "0".into());
    nextrocm_engine_core::use_kind("audio");
    let gpu = Gpu::open(args[5].parse().map_err(err)?)?;
    let initial = stats(&gpu)?.0;
    let t = Instant::now();
    let mut engine = Whisper::load(&files_in(model)?, &gpu, &options, &mut |s| eprintln!("{s}"))?;
    gpu.sync()?;
    let load = t.elapsed().as_secs_f64();
    let resident_bytes = stats(&gpu)?.0;
    let mut allocations = Vec::new();
    snapshot(&gpu, resident_bytes, "loaded", &mut allocations)?;
    need(
        engine.model.linear_f16_tiled == (args[4] == "1")
            && engine.model.attention_tiled == (attention == "1"),
        "effective flag mismatch",
    )?;
    let c = &engine.model.cfg;
    let (s, audio) = floats(&reference.join("audio16k.npy"))?;
    need(s.len() == 1, "audio geometry")?;
    let (s, mel) = floats(&reference.join("mel.npy"))?;
    need(s == [c.n_mels, 3000], "mel geometry")?;
    let toks = tokens(&reference.join("tokens.npy"), c.vocab, c.text_ctx)?;
    let fed = tokens(&reference.join("align_tokens.npy"), c.vocab, c.text_ctx)?;
    let spec: Value =
        serde_json::from_slice(&std::fs::read(reference.join("align.json")).map_err(err)?)
            .map_err(err)?;
    let lang = spec["language"]
        .as_str()
        .ok_or_else(|| err("alignment language missing"))?
        .to_owned();
    let sot = engine.vocab.sot_sequence(&lang, false);
    need(
        spec["sot_len"].as_u64() == Some(sot.len() as u64)
            && fed.starts_with(&sot)
            && fed.len() > sot.len() + 2
            && fed[sot.len()] == engine.vocab.no_timestamps
            && fed.last() == Some(&engine.vocab.eot),
        "alignment token grammar",
    )?;
    let text_tokens = &fed[sot.len() + 1..fed.len() - 1];
    need(
        text_tokens.iter().all(|&t| t < engine.vocab.eot),
        "alignment text tokens",
    )?;
    let frames = spec["num_frames"]
        .as_u64()
        .ok_or_else(|| err("missing num_frames"))? as usize;
    need((2..=3000).contains(&frames), "frame bounds")?;
    let heads = engine.vocab.alignment_heads.clone();
    let geometry = json!({"audio_ctx":c.audio_ctx,"d":c.d,"vocab":c.vocab,"n_mels":c.n_mels,"frames":frames,"sot_len":sot.len()});
    need(
        floats(&reference.join("enc.npy"))?.0 == [c.audio_ctx, c.d],
        "reference encoder geometry",
    )?;
    need(
        floats(&reference.join("logits.npy"))?.0 == [toks.len(), c.vocab],
        "reference logits geometry",
    )?;
    need(
        floats(&reference.join("align_qk.npy"))?.0 == [heads.len(), fed.len(), c.audio_ctx],
        "reference alignment geometry",
    )?;
    let req = TranscribeRequest {
        samples: audio,
        rate: 16000,
        language: Some(lang.clone()),
        words: true,
        extra: [
            ("fallback".into(), "0".into()),
            ("condition".into(), "0".into()),
            ("no-speech".into(), "0.6".into()),
        ]
        .into_iter()
        .collect(),
        ..Default::default()
    };
    let t = Instant::now();
    let first = transcript(&engine.transcribe(&req, &mut |_| Ok(()))?)?;
    gpu.sync()?;
    let first_s = t.elapsed().as_secs_f64();
    snapshot(&gpu, resident_bytes, "first", &mut allocations)?;
    let t = Instant::now();
    let warm = transcript(&engine.transcribe(&req, &mut |_| Ok(()))?)?;
    gpu.sync()?;
    let warm_s = t.elapsed().as_secs_f64();
    snapshot(&gpu, resident_bytes, "warm", &mut allocations)?;
    need(first == warm, "warm transcript differs")?;
    let c = &engine.model.cfg;
    let t = Instant::now();
    engine.model.encode(&mel)?;
    engine.model.nsd.wait()?;
    let encode_s = t.elapsed().as_secs_f64();
    d.tensor(
        "encoder",
        &[c.audio_ctx, c.d],
        &engine.model.encoder_output()?,
    )?;
    let mut cpu_metrics = BTreeMap::new();
    cpu_metrics.insert(
        "encoder".to_owned(),
        metrics(
            &engine.model.encoder_output()?,
            &floats(&reference.join("enc.npy"))?.1,
            c.d,
        )?,
    );
    let forced = engine.model.decode(&toks, 0, 0, None)?;
    d.tensor("forced_logits", &[toks.len(), c.vocab], &forced)?;
    let ref_logits = floats(&reference.join("logits.npy"))?.1;
    cpu_metrics.insert(
        "forced_logits".into(),
        metrics(&forced, &ref_logits, c.vocab)?,
    );
    let mut incremental = Vec::new();
    for (i, token) in toks.iter().enumerate() {
        incremental.extend(engine.model.decode(&[*token], i, 0, None)?);
    }
    d.tensor("incremental_logits", &[toks.len(), c.vocab], &incremental)?;
    // Forced vs incremental use distinct batch kernels; retain both. Each must
    // match its own native baseline exactly across flag0/1 (no CPU tolerance).
    cpu_metrics.insert(
        "incremental_logits".into(),
        metrics(&incremental, &ref_logits, c.vocab)?,
    );
    let greedy = engine.decode_window(&sot, 0, 0., 0, &mut |_| Ok(()))?;
    need(
        greedy.ended && greedy.avg_logprob.is_finite() && greedy.no_speech.is_finite(),
        "greedy did not end/finite",
    )?;
    let mut layers: Vec<usize> = engine.vocab.alignment_heads.iter().map(|h| h.0).collect();
    layers.sort();
    layers.dedup();
    need(!layers.is_empty(), "no alignment heads")?;
    let mut taps = Taps {
        layers: layers.clone(),
        q: Vec::new(),
    };
    let logits = engine.model.decode(&fed, 0, 0, Some(&mut taps))?;
    d.tensor("alignment_logits", &[fed.len(), c.vocab], &logits)?;
    need(taps.q.len() == layers.len(), "missing alignment taps")?;
    let m = (frames / 2).clamp(1, c.audio_ctx);
    let hd = c.d / c.dec_heads;
    let scale = 1. / (hd as f32).sqrt();
    let mut keys = BTreeMap::new();
    for (i, &layer) in layers.iter().enumerate() {
        d.tensor(
            &format!("query_layer_{layer}"),
            &[fed.len(), c.d],
            &taps.q[i],
        )?;
        let k = engine.model.cross_keys(layer)?;
        d.tensor(&format!("key_layer_{layer}"), &[c.audio_ctx, c.d], &k)?;
        keys.insert(layer, k);
    }
    let mut qk = Vec::new();
    let ref_qk = floats(&reference.join("align_qk.npy"))?.1;
    for &(layer, head) in &engine.vocab.alignment_heads {
        need(head < c.dec_heads, "bad head")?;
        let q = &taps.q[layers
            .iter()
            .position(|&l| l == layer)
            .ok_or_else(|| err("missing layer"))?];
        let k = &keys[&layer];
        let mut scores = vec![0f32; fed.len() * c.audio_ctx];
        for r in 0..fed.len() {
            for f in 0..c.audio_ctx {
                scores[r * c.audio_ctx + f] = q[r * c.d + head * hd..r * c.d + (head + 1) * hd]
                    .iter()
                    .zip(&k[f * c.d + head * hd..f * c.d + (head + 1) * hd])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    * scale;
            }
        }
        d.tensor(
            &format!("qk_layer_{layer}_head_{head}"),
            &[fed.len(), c.audio_ctx],
            &scores,
        )?;
        let offset = qk.len() * fed.len() * c.audio_ctx;
        cpu_metrics.insert(
            format!("qk_layer_{layer}_head_{head}"),
            metrics(&scores, &ref_qk[offset..offset + scores.len()], c.audio_ctx)?,
        );
        qk.push(
            scores
                .chunks_exact(c.audio_ctx)
                .flat_map(|r| r[..m].iter().copied())
                .collect(),
        );
    }
    let cost = dtw_cost(&qk, fed.len(), m, sot.len())?;
    let n = fed.len() - 1 - sot.len();
    d.tensor("dtw_cost", &[n, m], &cost)?;
    let (ti, fi) = align::dtw(&cost, n, m);
    need(
        !ti.is_empty()
            && ti.len() == fi.len()
            && ti.iter().all(|&i| i < n)
            && fi.iter().all(|&i| i < m),
        "DTW geometry",
    )?;
    let probs: Vec<f32> = text_tokens
        .iter()
        .enumerate()
        .map(|(i, &token)| {
            let row = &logits
                [(sot.len() + i) * c.vocab..(sot.len() + i) * c.vocab + engine.vocab.eot as usize];
            text::prob(row, token) as f32
        })
        .collect();
    d.tensor("alignment_probabilities", &[probs.len()], &probs)?;
    let reconstructed = align::find_alignment(
        &qk,
        fed.len(),
        m,
        frames,
        sot.len(),
        text_tokens,
        &probs,
        &engine.vocab,
        &lang,
        7,
    );
    let production = engine.window_words(text_tokens, &lang, false, frames)?;
    need(
        reconstructed == production,
        "reconstructed alignment differs from production",
    )?;
    let alignment_words = words(&production)?;
    snapshot(&gpu, resident_bytes, "stages", &mut allocations)?;
    let mut empty = req.clone();
    empty.samples.clear();
    need(
        engine.transcribe(&empty, &mut |_| Ok(())).is_err(),
        "empty input accepted",
    )?;
    snapshot(&gpu, resident_bytes, "invalid", &mut allocations)?;
    let mut cancelled = false;
    let cancel = engine.transcribe(&req, &mut |step| {
        if step.phase == "tokens" && step.at >= 1 {
            cancelled = true;
            Err(err("diagnostic cancellation"))
        } else {
            Ok(())
        }
    });
    need(
        cancelled && cancel.is_err_and(|e| e.0 == "diagnostic cancellation"),
        "cancellation not propagated",
    )?;
    snapshot(&gpu, resident_bytes, "cancel", &mut allocations)?;
    need(
        transcript(&engine.transcribe(&req, &mut |_| Ok(()))?)? == first,
        "recovery transcript differs",
    )?;
    snapshot(&gpu, resident_bytes, "recovery", &mut allocations)?;
    let resident = stats(&gpu)?;
    drop(engine);
    need(stats(&gpu)?.0 == initial, "drop leaked allocations")?;
    engine = Whisper::load(&files_in(model)?, &gpu, &options, &mut |s| eprintln!("{s}"))?;
    snapshot(&gpu, resident_bytes, "reloaded", &mut allocations)?;
    need(
        transcript(&engine.transcribe(&req, &mut |_| Ok(()))?)? == first,
        "reload transcript differs",
    )?;
    snapshot(&gpu, resident_bytes, "reload_request", &mut allocations)?;
    drop(engine);
    let after = stats(&gpu)?;
    need(after.0 == initial, "reload drop leaked allocations")?;
    for (k, p) in &paths {
        need(
            sha(p)? == hashes[k],
            "input/runtime/executable changed during run",
        )?;
    }
    let reference_transcription: Value =
        serde_json::from_slice(&std::fs::read(reference.join("result.json")).map_err(err)?)
            .map_err(err)?;
    let reference_text = reference_transcription["text"]
        .as_str()
        .ok_or_else(|| err("reference transcript missing text"))?;
    let cpu_wer =
        nextrocm_audio_whisper::check::wer(first["text"].as_str().unwrap_or(""), reference_text);
    let report = json!({"schema":2,"pass":true,"flag":args[4],"attention_flag":attention,"controls":{"linear_f16_tiled":args[4],"attention_tiled":attention,"encoder_profile":"0"},"hashes":hashes,"gpu":{"name":gpu.name,"index":args[5]},
        "tensors":d.tensors,"tokens":toks,"alignment_tokens":fed,"alignment_heads":heads,"geometry":geometry,
        "head_pairs":qk.len(),"dtw":{"tokens":ti,"frames":fi},"alignment_words":alignment_words,
        "greedy":{"tokens":greedy.tokens,"ended":greedy.ended,"avg_logprob":greedy.avg_logprob,"no_speech":greedy.no_speech,"temperature":greedy.temperature},
        "transcription":first,"reference_transcription":reference_transcription,"cpu_wer":cpu_wer,"cpu_metrics":cpu_metrics,"allocation_snapshots":allocations,"request":{"language":lang,"words":true,"fallback":0,"condition":0,"no_speech":0.6},
        "timings":{"load_s":load,"first_transcription_s":first_s,"warm_transcription_s":warm_s,"reference_encode_s":encode_s},
        "lifecycle":{"warm":true,"invalid_rejected":true,"cancel_tokens":true,"recovery":true,"reload":true,"initial_bytes":initial,"resident_bytes":resident.0,"tracked_peak_bytes":after.1,"after_drop_bytes":after.0},
        "scope":"Native paired equality; independent CPU reference artifacts hashed, not a relaxed reference admission. Tracked memory is not driver peak."});
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(err)?,
    )
    .map_err(err)?;
    println!("PASS dump {}", out.display());
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    fn fixture(bytes: &[u8], test: impl FnOnce(&Path)) {
        let path = std::env::temp_dir().join(format!(
            "whisper-paired-{}-{}.npy",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, bytes).unwrap();
        test(&path);
        std::fs::remove_file(path).unwrap();
    }
    fn array(values: &[f32]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}\n",
            values.len()
        );
        let mut data = b"\x93NUMPY\x01\x00".to_vec();
        data.extend_from_slice(&(header.len() as u16).to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        data.extend(values.iter().flat_map(|x| x.to_le_bytes()));
        data
    }
    #[test]
    fn strict_npy_complete_finite_geometry() {
        let bytes = array(&[0.0, -0.0, 1.25]);
        fixture(&bytes, |p| {
            let (s, x) = floats(p).unwrap();
            assert_eq!(s, [3]);
            assert_eq!(x[1].to_bits(), (-0.0f32).to_bits());
        });
        for count in 0..bytes.len() {
            fixture(&bytes[..count], |p| assert!(floats(p).is_err()));
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        fixture(&trailing, |p| assert!(floats(p).is_err()));
        fixture(&array(&[f32::NAN]), |p| assert!(floats(p).is_err()));
        fixture(&array(&[f32::INFINITY]), |p| assert!(floats(p).is_err()));
        fixture(&bytes, |p| assert!(npy(p, "<i8").is_err()));
    }
    #[test]
    fn tensor_shape_overflow_and_alignment_missing_are_errors() {
        assert!(product(&[usize::MAX, 2]).is_err());
        assert!(dtw_cost(&[], 4, 8, 2).is_err());
        assert!(dtw_cost(&[vec![0.; 1]], 4, 8, 2).is_err());
        assert!(dtw_cost(&[vec![f32::NAN; 32]], 4, 8, 2).is_err());
    }
}
