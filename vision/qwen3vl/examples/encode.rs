//! Parent-only stage gate, original CPU relL2<=1e-3. No model registration.
use nextrocm_engine_core::{Error, Gpu, Result};
use nextrocm_vision_qwen3vl::{Budget, Plan, TextEncoder};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
fn fail(s: impl Into<String>) -> Error {
    Error(s.into())
}
fn io<T>(r: std::io::Result<T>) -> Result<T> {
    r.map_err(|e| fail(e.to_string()))
}
fn sha(p: &Path) -> Result<String> {
    let out = io(Command::new("sha256sum").arg(p).output())?;
    if !out.status.success() {
        return Err(fail(format!("sha256sum failed {}", p.display())));
    }
    let s = String::from_utf8(out.stdout).map_err(|e| fail(e.to_string()))?;
    let h = s
        .split_whitespace()
        .next()
        .ok_or_else(|| fail("missing SHA"))?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(fail("invalid SHA"));
    }
    Ok(h.to_owned())
}
fn identities(paths: &[PathBuf]) -> Result<BTreeMap<String, String>> {
    paths
        .iter()
        .map(|p| Ok((p.display().to_string(), sha(p)?)))
        .collect()
}
fn stats(gpu: &Gpu) -> Result<(u64, u64)> {
    gpu.sync()?;
    let p = nextrocm_engine_core::api()?.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err(fail("missing allocation stats"));
    }
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: ns.hip exact ABI; livecontext and valid output pointers.
    let f: Stats = unsafe { std::mem::transmute(p) };
    let (mut live, mut peak) = (0, 0);
    if unsafe { f(gpu.raw(), &mut live, &mut peak) } != 0 || peak >= 8_000_000_000 {
        return Err(fail("allocation stats/cap"));
    }
    Ok((live, peak))
}
fn npy(path: &Path, rows: usize) -> Result<Vec<f32>> {
    let b = io(std::fs::read(path))?;
    if b.len() < 12 || &b[..6] != b"\x93NUMPY" {
        return Err(fail("bad NPY"));
    }
    let (base, len) = match b[6] {
        1 => (10, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 => (12, u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize),
        _ => return Err(fail("NPY version")),
    };
    let h = std::str::from_utf8(
        b.get(base..base + len)
            .ok_or_else(|| fail("NPY header bounds"))?,
    )
    .map_err(|e| fail(e.to_string()))?;
    if !h.contains("'<f4'") || !h.contains("'fortran_order': False") {
        return Err(fail("NPY dtype/layout"));
    }
    let dims = h
        .split("'shape':")
        .nth(1)
        .and_then(|v| v.split('(').nth(1))
        .and_then(|v| v.split(')').next())
        .ok_or_else(|| fail("NPY shape"))?;
    let shape = dims
        .split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| fail(e.to_string()))?;
    if shape != [rows, 4096] || b.len() != base + len + rows * 4096 * 4 {
        return Err(fail("NPY geometry/bytes"));
    }
    let values = b[base + len..]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect::<Vec<_>>();
    if values.iter().any(|v| !v.is_finite()) {
        return Err(fail("nonfinite CPU reference"));
    }
    Ok(values)
}
fn compare(actual: &[f32], reference: &[f32]) -> Result<serde_json::Value> {
    if actual.len() != reference.len() || actual.iter().any(|v| !v.is_finite()) {
        return Err(fail("native nonfinite/shape"));
    }
    let (mut diff, mut norm, mut max) = (0., 0., 0f64);
    for (&a, &r) in actual.iter().zip(reference) {
        let d = a as f64 - r as f64;
        diff += d * d;
        norm += (r as f64).powi(2);
        max = max.max(d.abs());
    }
    let rel = diff.sqrt() / norm.sqrt().max(1e-30);
    if rel > 1e-3 {
        return Err(fail(format!(
            "frozen full-tensor relL2 gate failed: {rel}, max_abs={max}"
        )));
    }
    Ok(serde_json::json!({"relative_l2":rel,"max_abs":max,"elements":actual.len(),"pass":true}))
}
fn run() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    if !(8..=12).contains(&a.len()) {
        return Err(fail(
            "usage: encode WEIGHTS TOKENIZER CONFIG CPU_REFERENCE NEW_OUT GPU PROMPT [ATTENTION_PROB0_OR1 [HEADNORM_ORDERED0_OR1 [RMS_ORDERED0_OR1 [SCORE_DIVISION0_OR1]]]]",
        ));
    }
    let probability_first = match a.get(8).map(String::as_str) {
        None | Some("0") => false,
        Some("1") => true,
        _ => return Err(fail("attention control must be0/1")),
    };
    let ordered_headnorm = match a.get(9).map(String::as_str) {
        None | Some("0") => false,
        Some("1") => true,
        _ => return Err(fail("ordered headnorm control must be0/1")),
    };
    let ordered_rms = match a.get(10).map(String::as_str) {
        None | Some("0") => false,
        Some("1") => true,
        _ => return Err(fail("ordered RMS control must be0/1")),
    };
    let score_division = match a.get(11).map(String::as_str) {
        None | Some("0") => false,
        Some("1") => true,
        _ => return Err(fail("score division control must be0/1")),
    };
    let reference = Path::new(&a[4]);
    let output = Path::new(&a[5]);
    if output.exists() {
        return Err(fail("output must be new"));
    }
    let lib = PathBuf::from(
        std::env::var("NEXTROCM_LIB")
            .map_err(|_| fail("explicit absolute NEXTROCM_LIB required"))?,
    );
    if !lib.is_absolute() {
        return Err(fail("absolute NEXTROCM_LIB required"));
    }
    let mut paths = vec![
        io(std::fs::read_link(format!(
            "/proc/{}/exe",
            std::process::id()
        )))?,
        lib,
    ];
    for p in [&a[1], &a[2], &a[3]] {
        paths.push(io(std::fs::canonicalize(p))?);
    }
    for f in [
        "tokens.json",
        "te_layer0.npy",
        "embeds.npy",
        "te-original-f32-provenance.json",
    ] {
        paths.push(io(std::fs::canonicalize(reference.join(f)))?);
    }
    let before = identities(&paths)?;
    let provenance_path = io(std::fs::canonicalize(
        reference.join("te-original-f32-provenance.json"),
    ))?;
    if before[&provenance_path.display().to_string()]
        != "116d588780ba37becf4ce68e582102caae73a742ead7bb910b9f07c6192612d9"
    {
        return Err(fail("pinned primary CPU provenance differs"));
    }
    let provenance: serde_json::Value =
        serde_json::from_slice(&io(std::fs::read(&provenance_path))?)
            .map_err(|e| fail(e.to_string()))?;
    if provenance["status"] != "PASS"
        || provenance["stage"] != "te"
        || provenance["precision"] != "original-f32"
        || provenance["primary_oracle"] != true
    {
        return Err(fail("CPU reference is not primary PASS"));
    }
    for (path, expected) in [
        (paths[2].clone(), &provenance["asset"]["sha256"]),
        (
            paths[3].clone(),
            &provenance["configs"]["processor/tokenizer.json"]["sha256"],
        ),
        (
            paths[4].clone(),
            &provenance["configs"]["text_encoder/config.json"]["sha256"],
        ),
    ] {
        if Some(before[&path.display().to_string()].as_str()) != expected.as_str() {
            return Err(fail("CPU/native weight/config identity mismatch"));
        }
    }
    for f in ["tokens.json", "te_layer0.npy", "embeds.npy"] {
        let path = io(std::fs::canonicalize(reference.join(f)))?;
        if Some(before[&path.display().to_string()].as_str())
            != provenance["outputs"][f]["sha256"].as_str()
        {
            return Err(fail("CPU output does not match pinned provenance"));
        }
    }
    let tokens: serde_json::Value =
        serde_json::from_slice(&io(std::fs::read(reference.join("tokens.json")))?)
            .map_err(|e| fail(e.to_string()))?;
    let gpu = Gpu::open(a[6].parse().map_err(|_| fail("GPU index"))?)?;
    let initial = stats(&gpu)?;
    if initial.0 != 0 {
        return Err(fail("initial allocations not zero"));
    }
    let start = Instant::now();
    let engine = TextEncoder::load_with_options(
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        gpu.clone(),
        nextrocm_vision_qwen3vl::LoadOptions {
            probability_first,
            ordered_headnorm,
            ordered_rms,
            score_division,
        },
    )?;
    let load_seconds = start.elapsed().as_secs_f64();
    if stats(&gpu)?.0 != 0 {
        return Err(fail("metadata load unexpectedly resident"));
    }
    let (ids, drop) = engine.tokenize(&a[7])?;
    if serde_json::json!(ids) != tokens["ids"] || serde_json::json!(drop) != tokens["drop"] {
        return Err(fail("original token IDs/drop differ"));
    }
    let layer0 = npy(&reference.join("te_layer0.npy"), ids.len())?;
    let embeds = npy(&reference.join("embeds.npy"), ids.len() - drop)?;
    io(std::fs::create_dir(output))?;
    let mut metrics = serde_json::Map::new();
    let budget = Budget {
        allowance_bytes: 7_500_000_000,
        driver_margin_bytes: 1024 * 1024 * 1024,
    };
    let plan = Plan::new(ids.len(), &budget)?;
    let mut memory = Vec::new();
    let start = Instant::now();
    let encoded = engine.encode(&a[7], &budget, &mut |stage, x| {
        let (live, peak) = stats(&gpu)?;
        if peak > plan.tracked_bytes as u64 {
            return Err(fail("actual tracked allocation exceeds preflight plan"));
        }
        memory.push(serde_json::json!({"stage":stage,"live_bytes":live,"peak_bytes":peak}));
        let v = x.to_f32()?;
        if v.iter().any(|v| !v.is_finite()) {
            return Err(fail(format!("nonfinite stage{stage}")));
        }
        io(std::fs::write(
            output.join(format!("stage-{stage:02}.f32")),
            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        ))?;
        if stage == 1 {
            metrics.insert("layer0".into(), compare(&v, &layer0)?);
        }
        println!("stage{stage} finite values={}", v.len());
        Ok(())
    })?;
    gpu.sync()?;
    let diagnostic_seconds = start.elapsed().as_secs_f64();
    let out = encoded.embeddings.to_f32()?;
    let returned = stats(&gpu)?;
    if returned.0 != ((ids.len() - drop) * 4096 * 4) as u64
        || returned.1 > plan.tracked_bytes as u64
    {
        return Err(fail("returned embedding residency/peak differs"));
    }
    metrics.insert("embeds".into(), compare(&out, &embeds)?);
    io(std::fs::write(
        output.join("embeds.f32"),
        out.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    ))?;
    let after = identities(&paths)?;
    if before != after {
        return Err(fail("input/runtime/executable identity changed"));
    }
    let mut report = serde_json::json!({"pass":true,"scope":"text-only original F32 CPU stage gates and tracked cleanup; no driver/lifecycle/performance admission",
  "attention_probability_first":probability_first,"headnorm_ordered":ordered_headnorm,"rms_ordered":ordered_rms,"score_division":score_division,
  "identities":before,"gpu":{"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc},
  "ids":encoded.ids,"drop":encoded.drop,"image_pad":encoded.image_pad,"metrics":metrics,
  "load_seconds":load_seconds,"diagnostic_encode_seconds":diagnostic_seconds,
  "planned_tracked_bytes":encoded.plan.tracked_bytes,"planned_with_margin_bytes":encoded.plan.with_margin_bytes,
  "initial_allocations":initial,"stage_allocations":memory,"returned_allocations":returned});
    drop_encoded(encoded);
    std::mem::drop(engine);
    gpu.sync()?;
    let dropped = stats(&gpu)?;
    if dropped.0 != initial.0 {
        return Err(fail("post-drop allocation leak"));
    }
    report["dropped_allocations"] = serde_json::json!(dropped);
    io(std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| fail(e.to_string()))?,
    ))?;
    println!("PASS original F32 TE layer0+cropped output; timing includes stage readback/dumps");
    Ok(())
}
fn drop_encoded(value: nextrocm_vision_qwen3vl::Encoded) {
    drop(value);
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
}
