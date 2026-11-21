//! CPU-only checkpoint validation and independent preprocessing/postprocessing fixture checks.
use nextrocm_image::{Error, Picture, Result};
use nextrocm_image_rembg::{isnet::IsNet, onnx, post_process_mask, IsNetEngine};
use std::path::PathBuf;
fn floats(path: &std::path::Path) -> Result<Vec<f32>> {
    let b = std::fs::read(path).map_err(|e| Error(e.to_string()))?;
    if b.len() % 4 != 0 {
        return Err(Error("partial f32".into()));
    }
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("four bytes")))
        .collect())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let model = PathBuf::from(
        args.get(1)
            .ok_or_else(|| Error("isnet_inspect MODEL [FIXTURE_ROOT]".into()))?,
    );
    let m = onnx::Model::open(&model)?;
    println!(
        "PASS: 119 convolution schema/weight checks; {} device weight bytes",
        IsNet::validate(&m)?
    );
    if let Some(dir) = args.get(2) {
        for name in ["pattern", "rgba", "black"] {
            let root = PathBuf::from(dir).join(name);
            let pic = Picture::decode(
                &std::fs::read(root.join("picture.png")).map_err(|e| Error(e.to_string()))?,
            )
            .map_err(Error)?;
            let expected = floats(&root.join("input.f32"))?;
            let actual = IsNetEngine::input(&pic);
            assert_eq!(actual.len(), expected.len());
            let max = actual
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if max > 1e-6 {
                return Err(Error(format!("{name} preprocessing error {max}")));
            }
            let pred = floats(&root.join("pred.f32"))?;
            let mask = IsNetEngine::mask_of(&pred, pic.width, pic.height);
            let want = std::fs::read(root.join("mask.u8")).map_err(|e| Error(e.to_string()))?;
            if mask.data != want {
                return Err(Error(format!("{name} mask differs")));
            }
            let pp = post_process_mask(&mask);
            let want = std::fs::read(root.join("mask_pp.u8")).map_err(|e| Error(e.to_string()))?;
            if pp.data != want {
                return Err(Error(format!("{name} post-process differs")));
            }
            println!("PASS: {name} input max_abs={max}, mask and post-process byte-exact");
        }
    }
    Ok(())
}
