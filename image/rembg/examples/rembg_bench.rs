//! Native ImageEngine load/first/repeat timings and wrapper lifecycle, not an HTTP benchmark.
//! rembg_bench isnet|birefnet MODEL PICTURE.png NEW_REPORT.json
//! Run only in the parent's serialized GPU window. Independent parity uses the
//! existing isnet_numeric / nextrocm-remover validate runners, not this benchmark.
use nextrocm_engine_core::{Gpu, api};
use nextrocm_image::{LoadOptions, ModelFiles, Picture};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    io::{Read, Write},
    path::Path,
    time::{Instant, SystemTime, UNIX_EPOCH},
};
type R<T> = Result<T, Box<dyn Error>>;
fn ensure(ok: bool, message: &str) -> R<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
fn hash(path: &Path) -> R<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        digest.update(&buf[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
fn stats(gpu: &Gpu) -> R<(u64, u64)> {
    gpu.sync()?;
    let lib = api()?;
    let address = lib.symbol("nr_allocation_stats");
    ensure(!address.is_null(), "allocation telemetry required")?;
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: matching runtime ABI, library retained for process lifetime.
    let query: Stats = unsafe { std::mem::transmute(address) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: live GPU and valid writable scalar output pointers.
    ensure(
        unsafe { query(gpu.raw(), &mut current, &mut peak) } == 0,
        "allocation query failed",
    )?;
    ensure(peak < 8_000_000_000, "tracked peak exceeds strict 8 GB cap")?;
    Ok((current, peak))
}
fn validate_output(pic: &Picture, rgba: &Picture, mask: &Picture) -> R<()> {
    let n = pic.width as usize * pic.height as usize;
    ensure(
        (rgba.width, rgba.height, rgba.channels) == (pic.width, pic.height, 4)
            && rgba.data.len() == n * 4,
        "RGBA geometry/storage mismatch",
    )?;
    ensure(
        (mask.width, mask.height, mask.channels) == (pic.width, pic.height, 1)
            && mask.data.len() == n,
        "mask geometry/storage mismatch",
    )?;
    let rgb = pic.rgb();
    ensure(
        rgba.data
            .chunks_exact(4)
            .zip(rgb.data.chunks_exact(3))
            .zip(&mask.data)
            .all(|((r, p), a)| r[..3] == *p && r[3] == *a),
        "RGB/alpha contract mismatch",
    )
}
fn evaluate(arch: &str, model: &Path, image: &Path, report: &mut Value) -> R<()> {
    let (role, expected) = match arch {
        "isnet" => (
            "isnet",
            "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a",
        ),
        "birefnet" => (
            "remover",
            "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154",
        ),
        _ => return Err("arch must be isnet or birefnet".into()),
    };
    let digest = hash(model)?;
    ensure(
        digest == expected,
        "checkpoint hash differs from retained reference",
    )?;
    report["checkpoint"] = json!({"path":model,"sha256":digest});
    let pic = Picture::read_png(image)?;
    report["input"] = json!({"path":image,"sha256":hash(image)?,"width":pic.width,"height":pic.height,"channels":pic.channels});
    nextrocm_engine_core::use_kind("image");
    let gpu = Gpu::open(0)?;
    ensure(stats(&gpu)?.0 == 0, "initial tracked allocations not zero")?;
    report["gpu"] = json!(gpu.name);
    let result = (|| -> R<()> {
        let mut files = ModelFiles::new();
        files.insert(role.into(), model.to_path_buf());
        let start = Instant::now();
        let engine = (nextrocm_image_rembg::kind().load)(
            &files,
            std::slice::from_ref(&gpu),
            &LoadOptions::default(),
            &mut |s| eprintln!("{s}"),
        )?;
        gpu.sync()?;
        report["load_seconds"] = json!(start.elapsed().as_secs_f64());
        let resident = stats(&gpu)?.0;
        report["resident_tracked_bytes"] = json!(resident);
        ensure(
            engine.arch() == arch && engine.removes_background(),
            "wrong engine capabilities",
        )?;
        let bad = Picture {
            width: 1,
            height: 1,
            channels: 3,
            data: vec![],
        };
        ensure(
            engine.remove_background(&bad, false).is_err(),
            "invalid picture accepted",
        )?;
        ensure(
            stats(&gpu)?.0 == resident,
            "invalid request leaked allocations",
        )?;
        let mut rows = Vec::new();
        for post_process in [false, true] {
            let mut first: Option<(Picture, Picture)> = None;
            for repeat in 0..2 {
                gpu.sync()?;
                let start = Instant::now();
                let (rgba, mask) = engine.remove_background(&pic, post_process)?;
                gpu.sync()?;
                let seconds = start.elapsed().as_secs_f64();
                validate_output(&pic, &rgba, &mask)?;
                if let Some((a, b)) = &first {
                    ensure(
                        rgba.data == a.data && mask.data == b.data,
                        "repeat output differs",
                    )?;
                }
                let (current, peak) = stats(&gpu)?;
                ensure(
                    current == resident,
                    "request allocations did not return to resident baseline",
                )?;
                rows.push(
                    json!({"post_process":post_process,"repeat_index":repeat,"seconds":seconds,
                    "mask_sha256":format!("{:x}",Sha256::digest(&mask.data)),
                    "rgba_sha256":format!("{:x}",Sha256::digest(&rgba.data)),
                    "tracked_current_bytes":current,"tracked_peak_bytes":peak}),
                );
                report["runs"] = json!(rows);
                if repeat == 0 {
                    first = Some((rgba, mask));
                }
            }
        }
        Ok(()) // engine drops on both success and errors before final accounting
    })();
    let cleanup = stats(&gpu);
    if let Ok((current, peak)) = &cleanup {
        report["after_drop"] = json!({"current":current,"peak":peak});
    }
    result?;
    ensure(cleanup?.0 == 0, "model drop leaked allocations")
}
fn run() -> R<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure(
        args.len() == 4,
        "usage: rembg_bench isnet|birefnet MODEL PICTURE.png NEW_REPORT.json",
    )?;
    // Reserve a fresh artifact before any hardware work; failures remain reviewable.
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])?;
    let mut report = json!({"schema":"rembg-bench-v1","arch":args[0],"pass":false,
        "unix_started":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
        "scope":"native ImageEngine; one load, two requests without postprocess then two with postprocess; no HTTP, PNG encode or transport",
        "timing_boundary":"load includes synchronized weights/scratch; request includes CPU preparation, native forward, mask/RGBA and synchronization; excludes hashes/comparisons",
        "memory_scope":"exact runtime tracked allocations; excludes untracked driver allocations; sample DRM/RSS separately, do not add overlapping shared-memory figures",
        "parity":"repeat byte identity and wrapper contracts only; independent numerical parity is a separate required gate",
        "cancellation":"original remove_background API has no cancellation callback","runs":[]});
    let start = Instant::now();
    let result = evaluate(
        &args[0],
        Path::new(&args[1]),
        Path::new(&args[2]),
        &mut report,
    );
    report["validation_wall_seconds"] = json!(start.elapsed().as_secs_f64());
    report["pass"] = json!(result.is_ok());
    if let Err(error) = &result {
        report["error"] = json!(error.to_string());
    }
    out.write_all(&serde_json::to_vec_pretty(&report)?)?;
    out.write_all(b"\n")?;
    result
}
fn main() {
    if let Err(e) = run() {
        eprintln!("rembg benchmark FAIL: {e}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_contract_rejects_truncation_and_changed_alpha() {
        let pic = Picture {
            width: 1,
            height: 1,
            channels: 3,
            data: vec![1, 2, 3],
        };
        let mask = Picture {
            width: 1,
            height: 1,
            channels: 1,
            data: vec![42],
        };
        let mut rgba = Picture {
            width: 1,
            height: 1,
            channels: 4,
            data: vec![1, 2, 3, 42],
        };
        assert!(validate_output(&pic, &rgba, &mask).is_ok());
        rgba.data[3] = 41;
        assert!(validate_output(&pic, &rgba, &mask).is_err());
        rgba.data.pop();
        assert!(validate_output(&pic, &rgba, &mask).is_err());
    }
}
