//! Bounded private profile of actual model operators; never a throughput claim.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{Error, ImageEngine, Picture, VectorizeRequest};
use nextrocm_image_starvector::{files_in, profile, StarVector};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 6 {
        return Err("usage: starvector_profile MODEL_DIR PICTURE GPU_INDEX NEW_OUT BEAMS(1|2|8) TOKENS(1..8)".into());
    }
    if std::env::var("NS_PROFILE").as_deref() != Ok("gpu")
        || std::env::var_os("NR_STARVECTOR_VISION_TAPS").is_some()
    {
        return Err("requires NS_PROFILE=gpu and no vision tensor capture".into());
    }
    let beams: u32 = a[4].parse()?;
    let tokens: u32 = a[5].parse()?;
    if ![1, 2, 8].contains(&beams) || !(1..=8).contains(&tokens) {
        return Err("bounded profile geometry".into());
    }
    let out = Path::new(&a[3]);
    std::fs::create_dir(out)?;
    let gpu = Gpu::open(a[2].parse()?)?;
    let engine = StarVector::load(&files_in(Path::new(&a[0]))?, &gpu, &mut |s| println!("{s}"))?;
    let req = VectorizeRequest {
        picture: Picture::read_png(Path::new(&a[1])).map_err(Error)?,
        max_length: 259 + tokens,
        beams: Some(beams),
        sample: Some(beams > 1),
        seed: 1234,
        ..Default::default()
    };
    let baseline = engine.vectorize(&req, &mut |_| Ok(()))?;
    profile::begin()?;
    for _ in 0..8 {
        profile::measure(&gpu, "empty_event_pair".into(), || Ok(()))?;
    }
    let candidate = engine.vectorize(&req, &mut |_| Ok(()));
    let report = profile::finish(&out.join("operators.json"));
    let candidate = candidate?;
    report?;
    std::fs::write(out.join("baseline.svg"), &baseline.svg)?;
    std::fs::write(out.join("profiled.svg"), &candidate.svg)?;
    if baseline.svg != candidate.svg
        || baseline.tokens != candidate.tokens
        || baseline.closed != candidate.closed
    {
        return Err("profiling changed deterministic short output".into());
    }
    std::fs::write(
        out.join("replay.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"beams":beams,"token_bound":tokens,"tokens":baseline.tokens,"text_bytes_equal":true,"baseline_prompt_seconds":baseline.prompt_seconds,"baseline_decode_seconds":baseline.seconds,"profiled_prompt_seconds":candidate.prompt_seconds,"profiled_decode_seconds":candidate.seconds,"scope":"warm baseline versus synchronized diagnostics; not performance comparison"}),
        )?,
    )?;
    println!("PASS bounded profile; identical short output; inspect event categories before selecting a kernel");
    Ok(())
}
