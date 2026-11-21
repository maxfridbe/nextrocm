//! Diagnostic combined first-two-block capture; reproduce frozen failed full-run stage0/1/2.
use nextrocm_engine_core::{Error, Gpu, Result};
use nextrocm_vision_qwen3vl::{Budget, TextEncoder};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command,
};
fn err(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}
fn sha(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg(p).output().map_err(err)?;
    if !o.status.success() {
        return Err(err("sha256sum failed"));
    }
    let s = String::from_utf8(o.stdout).map_err(err)?;
    Ok(s.split_whitespace()
        .next()
        .ok_or_else(|| err("empty hash"))?
        .to_owned())
}
fn hashes(paths: &[PathBuf]) -> Result<BTreeMap<String, String>> {
    paths
        .iter()
        .map(|p| Ok((p.display().to_string(), sha(p)?)))
        .collect()
}
fn run() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 8 {
        return Err(err(
            "usage: trace_combined WEIGHTS TOKENIZER CONFIG FAILED_BASELINE NEW_OUT GPU PROMPT",
        ));
    }
    let out = Path::new(&a[5]);
    if out.exists() {
        return Err(err("output must be new"));
    }
    let runtime = PathBuf::from(std::env::var("NEXTROCM_LIB").map_err(err)?);
    if !runtime.is_absolute()
        || sha(&runtime)? != "123f2623224665dbd877f8c6fc9f59987c081031215f36b6cdc6cc86902dfb49"
    {
        return Err(err("exact failing runtime required for same-math capture"));
    }
    let base = Path::new(&a[4]);
    let mut inputs = vec![
        std::fs::read_link(format!("/proc/{}/exe", std::process::id())).map_err(err)?,
        runtime,
    ];
    for p in [&a[1], &a[2], &a[3]] {
        inputs.push(std::fs::canonicalize(p).map_err(err)?);
    }
    inputs.extend([
        base.join("stage-00.f32"),
        base.join("stage-01.f32"),
        base.join("stage-02.f32"),
    ]);
    for (path, expected) in [
        (
            &inputs[5],
            "cb90a477eba3b118cf2edf9eda9fb3c8fdc185b07104bf2b616ce105214820b7",
        ),
        (
            &inputs[6],
            "67d504701c0eec21fb0042a2dcf9fe4aa016f31bd3468fa72113e0fd85569cc0",
        ),
        (
            &inputs[7],
            "a00da03359855414d5f9726e8142c025d3a7d64608a2545b03254119d58abdb0",
        ),
    ] {
        if sha(path)? != expected {
            return Err(err("frozen strict stage identity"));
        }
    }
    let before = hashes(&inputs)?;
    if before[&inputs[2].display().to_string()]
        != "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"
    {
        return Err(err("original checkpoint identity"));
    }
    let gpu = Gpu::open(a[6].parse().map_err(err)?)?;
    let engine = TextEncoder::load_with_options(
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        gpu.clone(),
        nextrocm_vision_qwen3vl::LoadOptions {
            probability_first: true,
            ordered_headnorm: true,
            ordered_rms: true,
            score_division: true,
        },
    )?;
    let (ids, drop) = engine.tokenize(&a[7])?;
    std::fs::create_dir(out).map_err(err)?;
    let mut files = BTreeMap::new();
    let budget = Budget {
        allowance_bytes: 7_500_000_000,
        driver_margin_bytes: 1024 * 1024 * 1024,
    };
    for layer in 0..2 {
        let label = format!("layer{layer}");
        let sub = out.join(&label);
        std::fs::create_dir(&sub).map_err(err)?;
        let mut capture = |name: &str,
                           buffer: &nextrocm_engine_core::DevBuf,
                           bytes: usize|
         -> Result<()> {
            let mut data = vec![0; bytes];
            buffer.read(0, &mut data)?;
            let stage = match (layer, name) {
                (0, "embed") => Some(0),
                (0, "layer0") | (1, "block_input") => Some(1),
                (1, "block_output") => Some(2),
                _ => None,
            };
            if let Some(stage) = stage {
                if data != std::fs::read(base.join(format!("stage-{stage:02}.f32"))).map_err(err)? {
                    return Err(err(format!("trace changed frozen stage{stage} bits")));
                }
            }
            if !name.ends_with("quantized")
                && !name.ends_with("accum")
                && data
                    .chunks_exact(4)
                    .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            {
                return Err(err("nonfinite F32 diagnostic tap"));
            }
            let p = sub.join(format!("{name}.bin"));
            std::fs::write(&p, &data).map_err(err)?;
            files.insert(
                format!("{label}/{name}"),
                serde_json::json!({"bytes":bytes,"sha256":sha(&p)?}),
            );
            println!("CAPTURE {label}/{name} bytes={bytes}");
            Ok(())
        };
        if layer == 0 {
            engine.trace_layer0(&a[7], &budget, &mut capture)?;
        } else {
            let raw = std::fs::read(base.join("stage-01.f32")).map_err(err)?;
            if raw.len() != ids.len() * 4096 * 4 {
                return Err(err("stage1 input geometry"));
            }
            let values = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>();
            engine.trace_block1(&a[7], &budget, &values, &mut capture)?;
        }
    }
    if files.len() != 66 {
        return Err(err("expected66 boundary captures"));
    }
    gpu.sync()?;
    if before != hashes(&inputs)? {
        return Err(err("inputs changed"));
    }
    let report = serde_json::json!({"capture_complete":true,"model_gate_pass":false,"scope":"reproduces failed combined full-run stage0/1/2 exactly; diagnostic only",
  "attention_probability_first":true,"headnorm_ordered":true,"rms_ordered":true,"score_division":true,"identities":before,"ids":ids,"drop":drop,"files":files,"gpu":{"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc}});
    std::fs::write(
        out.join("trace.json"),
        serde_json::to_vec_pretty(&report).map_err(err)?,
    )
    .map_err(err)?;
    println!("CAPTURE COMPLETE; frozen failure reproduced, no numeric admission");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
}
