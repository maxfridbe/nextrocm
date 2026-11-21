//! Parent-only native stage validator; never registers a model or changes UI.
use nextrocm_engine_core::Gpu;
use nextrocm_image_starvector::{check, files_in, StarVector};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: starvector_numeric MODEL_DIR FIXTURE_DIR PICTURE GPU_INDEX".into());
    }
    let gpu = Gpu::open(args[3].parse()?)?;
    let engine = StarVector::load(&files_in(Path::new(&args[0]))?, &gpu, &mut |s| {
        println!("{s}")
    })?;
    let worst = check::run(
        &engine,
        Path::new(&args[1]),
        Path::new(&args[2]),
        &["pixels", "encoder", "prefill", "logits"],
        &mut |s| println!("{s}"),
    )?;
    // Pinned upstream card reports ~4e-4 relative logits error. 1e-3 is a
    // predeclared initial stage gate, not a model admission or trajectory claim.
    if !worst.is_finite() || worst > 0.001 {
        return Err(format!("StarVector relative L2 gate failed: {worst}").into());
    }
    println!(
        "PASS covered exact-input stages; lifecycle/SVG/service/performance remain separate gates"
    );
    Ok(())
}
