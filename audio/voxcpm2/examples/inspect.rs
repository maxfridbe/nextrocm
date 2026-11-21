//! CPU-only parsing/admission inspection of the original files. Does not load HIP.
//! cargo run --release -p nextrocm-audio-voxcpm2 --example inspect -- MODEL_DIR
use nextrocm_audio::{Error, Result};
use nextrocm_audio_voxcpm2::{budget, files_in, ops::Shards, pth};
use std::path::PathBuf;
fn run() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| Error("inspect MODEL_DIR".into()))?,
    );
    let files = files_in(&dir)?;
    let shards = Shards::open(&[files["model"].clone()])?;
    let vae = pth::load(&files["vae"])?;
    let vae_bytes = vae.values().try_fold(0usize, |n, t| {
        budget::sum(&[n, budget::product(&[t.data.len(), 4])?])
    })?;
    let upper = shards.load_upper_bound(false)?;
    let int8_upper = shards.load_upper_bound(true)?;
    println!(
        "{}",
        serde_json::json!({"architecture":"voxcpm2","int8_default":false,
        "embedding":shards.shape("base_lm.embed_tokens.weight")?,"vae_tensors":vae.len(),"vae_f32_bytes":vae_bytes,
        "half_model_load_upper_bound":upper,"half_load_plus_vae_plus_reserve":budget::sum(&[upper,vae_bytes,budget::RESERVE])?,
        "explicit_int8_model_load_upper_bound":int8_upper,"explicit_int8_load_plus_vae_plus_reserve":budget::sum(&[int8_upper,vae_bytes,budget::RESERVE])?,
        "strict_cap":budget::CAP,"note":"load bound only; per-request session and VAE work admitted separately; no GPU measurement"})
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{}", e.0);
        std::process::exit(1);
    }
}
