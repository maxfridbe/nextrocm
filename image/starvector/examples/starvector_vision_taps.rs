use nextrocm_engine_core::Gpu;
use nextrocm_image_starvector::{check, files_in, StarVector};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4 || std::env::var_os("NR_STARVECTOR_VISION_TAPS").is_none() {
        return Err("set NR_STARVECTOR_VISION_TAPS to a new directory; usage: starvector_vision_taps MODEL_DIR FIXTURE_DIR PICTURE GPU_INDEX".into());
    }
    let gpu = Gpu::open(args[3].parse()?)?;
    let engine = StarVector::load(&files_in(Path::new(&args[0]))?, &gpu, &mut |s| {
        println!("{s}")
    })?;
    let worst = check::run(
        &engine,
        Path::new(&args[1]),
        Path::new(&args[2]),
        &["pixels", "encoder"],
        &mut |s| println!("{s}"),
    )?;
    if !worst.is_finite() || worst > 0.001 {
        return Err(
            format!("vision diagnostic relative L2 gate failed: {worst}; taps retained").into(),
        );
    }
    println!("PASS vision stages only");
    Ok(())
}
