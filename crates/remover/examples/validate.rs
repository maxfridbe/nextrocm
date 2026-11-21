//! Manual native validation only: validate WEIGHTS CPU_FIXTURE_DIR [MAX_REL_L2].
use nextrocm_diffusion::Picture;
use nextrocm_engine_core::{Error, Gpu, Result, api};
use nextrocm_remover::Remover;
use std::path::Path;

fn ensure(ok: bool, why: impl Into<String>) -> Result<()> {
    if ok { Ok(()) } else { Err(Error(why.into())) }
}
fn npy(path: &Path, count: usize) -> Result<Vec<f32>> {
    let size = std::fs::metadata(path)
        .map_err(|e| Error(e.to_string()))?
        .len();
    ensure(size <= count as u64 * 4 + 65536, "unexpected NPY file size")?;
    let b = std::fs::read(path).map_err(|e| Error(e.to_string()))?;
    ensure(
        b.len() >= 12 && &b[..6] == b"\x93NUMPY",
        "invalid NPY magic",
    )?;
    let (at, len) = match b[6] {
        1 => (10, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 | 3 => (12, u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize),
        _ => return Err(Error("unsupported NPY version".into())),
    };
    ensure(at + len <= b.len(), "truncated NPY header")?;
    let header = std::str::from_utf8(&b[at..at + len]).map_err(|e| Error(e.to_string()))?;
    ensure(
        header.contains("'<f4'") && header.contains("'fortran_order': False"),
        "expected contiguous little-endian f32",
    )?;
    ensure(
        b.len() - at - len == count * 4,
        "unexpected NPY element count",
    )?;
    Ok(b[at + len..]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
        .collect())
}
fn compare(name: &str, actual: &[f32], expected: &[f32], tolerance: f64) -> Result<()> {
    ensure(
        !actual.is_empty() && actual.len() == expected.len(),
        format!("{name}: length mismatch"),
    )?;
    let (mut sum, mut norm, mut max) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        ensure(
            a.is_finite() && b.is_finite(),
            format!("{name}: nonfinite value"),
        )?;
        let d = f64::from(a) - f64::from(b);
        sum += d * d;
        norm += f64::from(b).powi(2);
        max = max.max(d.abs());
    }
    let rel = (sum / norm.max(1e-30)).sqrt();
    println!("{name}: relative_l2={rel:.8e} max_abs={max:.8e}");
    ensure(
        rel <= tolerance,
        format!("{name}: relative error {rel} > {tolerance}"),
    )
}
fn stats(g: &Gpu) -> Result<(u64, u64)> {
    g.sync()?;
    let a = api()?;
    let p = a.symbol("nr_allocation_stats");
    ensure(!p.is_null(), "allocation telemetry missing")?;
    type Query = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: symbol is retained runtime's stats ABI with live GPU and output pointers.
    let query: Query = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: valid handle and writable outputs for the declared C ABI.
    ensure(
        unsafe { query(g.raw(), &mut current, &mut peak) } == 0,
        a.error(),
    )?;
    ensure(peak < 8_000_000_000, "device peak exceeds strict cap")?;
    Ok((current, peak))
}
fn checks(net: &Remover, gpu: &Gpu, dir: &Path, rel: f64) -> Result<()> {
    let resident = stats(gpu)?.0;
    let pic = Picture::read_png(&dir.join("picture.png")).map_err(Error)?;
    let input = npy(&dir.join("input.npy"), 1024 * 1024 * 3)?;
    compare("input", &Remover::input(&pic), &input, 1e-6)?;
    ensure(
        net.logits(&[0.0], false).is_err(),
        "bad input length accepted",
    )?;
    let mut bad = input.clone();
    bad[0] = f32::NAN;
    ensure(net.logits(&bad, false).is_err(), "NaN input accepted")?;
    drop(bad);
    ensure(stats(gpu)?.0 == resident, "invalid input leaked memory")?;
    let (logits, probe) = net.logits(&input, true)?;
    let probe = probe.ok_or_else(|| Error("missing probes".into()))?;
    compare(
        "x1",
        &probe.x1,
        &npy(&dir.join("x1.npy"), 256 * 256 * 384)?,
        rel,
    )?;
    compare(
        "x4",
        &probe.x4,
        &npy(&dir.join("x4.npy"), 32 * 32 * 5760)?,
        rel,
    )?;
    compare(
        "logits",
        &logits,
        &npy(&dir.join("logits.npy"), 1024 * 1024)?,
        rel,
    )?;
    ensure(
        stats(gpu)?.0 == resident,
        "forward leaked request allocations",
    )?;
    let mask = net.mask(&pic)?; // second request also validates reuse and native preprocessing
    let want = Picture::read_png(&dir.join("mask.png")).map_err(Error)?;
    let channels = want.channels as usize;
    ensure(
        mask.width == want.width && mask.height == want.height && mask.channels == 1,
        "mask geometry mismatch",
    )?;
    let max = mask
        .data
        .iter()
        .zip(want.data.chunks_exact(channels))
        .map(|(&a, b)| (i16::from(a) - i16::from(b[0])).abs())
        .max()
        .unwrap_or(0);
    println!("mask: max_byte_difference={max}");
    ensure(max <= 2, "mask byte difference exceeds 2")?;
    ensure(
        stats(gpu)?.0 == resident,
        "reuse leaked request allocations",
    )?;
    Ok(())
}
fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        (2..=3).contains(&args.len()),
        "usage: validate WEIGHTS CPU_FIXTURE_DIR [MAX_REL_L2]",
    )?;
    let tolerance = args
        .get(2)
        .map(|s| s.parse::<f64>())
        .transpose()
        .map_err(|e| Error(e.to_string()))?
        .unwrap_or(1e-3);
    ensure(
        tolerance.is_finite() && tolerance > 0.0,
        "invalid tolerance",
    )?;
    nextrocm_engine_core::use_kind("image");
    let gpu = Gpu::open(0)?;
    ensure(stats(&gpu)?.0 == 0, "nonempty initial allocator")?;
    let net = Remover::load(Path::new(&args[0]), &gpu)?;
    let result = checks(&net, &gpu, Path::new(&args[1]), tolerance);
    drop(net);
    let (current, peak) = stats(&gpu)?;
    println!("drop: current={current} peak={peak}");
    ensure(current == 0, "model drop leaked allocations")?;
    result?;
    println!("PASS BiRefNet numeric stages, invalid input, reuse and cleanup");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL BiRefNet: {e}");
        std::process::exit(1)
    }
}
