//! CPU-only audit of all exact reference-card model files.
use nextrocm_image_upscale::{ncnn, waifu2x};
use std::{error::Error, path::PathBuf};
fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("waifu2x_inspect MODELS_DIR")?,
    );
    let mut count = 0;
    for variant in waifu2x::VARIANTS {
        for entry in std::fs::read_dir(root.join(variant.id))? {
            let path = entry?.path();
            if path.extension().is_none_or(|s| s != "param") {
                continue;
            }
            let graph = ncnn::parse_param(&std::fs::read_to_string(&path)?)?;
            let bin = std::fs::read(path.with_extension("bin"))?;
            let report = waifu2x::inspect(&graph, Some(&bin), 456, 456)?;
            println!(
                "{} layers={} output={}x{}x{} weight_bytes={} workspace_bound={}",
                path.display(),
                report.layers,
                report.output.w,
                report.output.h,
                report.output.c,
                report.weight_bytes,
                report.workspace_bound
            );
            count += 1;
        }
    }
    if count != 19 {
        return Err(format!("expected 19 graphs, got {count}").into());
    }
    println!("PASS: {count} complete graph/weight audits; no GPU opened");
    Ok(())
}
