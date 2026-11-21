//! Diagnostic capture only; must reproduce the retained failing stage0/1 bits.
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
            "usage: trace_layer0 WEIGHTS TOKENIZER CONFIG FAILED_BASELINE NEW_OUT GPU PROMPT",
        ));
    }
    let out = Path::new(&a[5]);
    if out.exists() {
        return Err(err("output must be new"));
    }
    let runtime = PathBuf::from(std::env::var("NEXTROCM_LIB").map_err(err)?);
    if !runtime.is_absolute()
        || sha(&runtime)? != "9bcd81e6c8cbce4a12d07a61ffef8036513a56d6abef1491c6e35ef321536916"
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
    inputs.extend([base.join("stage-00.f32"), base.join("stage-01.f32")]);
    let before = hashes(&inputs)?;
    if before[&inputs[2].display().to_string()]
        != "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"
    {
        return Err(err("original checkpoint identity"));
    }
    let gpu = Gpu::open(a[6].parse().map_err(err)?)?;
    let engine = TextEncoder::load(
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        gpu.clone(),
    )?;
    let (ids, drop) = engine.tokenize(&a[7])?;
    std::fs::create_dir(out).map_err(err)?;
    let mut files = BTreeMap::new();
    engine.trace_layer0(
        &a[7],
        &Budget {
            allowance_bytes: 7_500_000_000,
            driver_margin_bytes: 1024 * 1024 * 1024,
        },
        &mut |name, buffer, bytes| {
            let mut data = vec![0; bytes];
            buffer.read(0, &mut data)?;
            if name == "embed" || name == "layer0" {
                let expected = std::fs::read(base.join(if name == "embed" {
                    "stage-00.f32"
                } else {
                    "stage-01.f32"
                }))
                .map_err(err)?;
                if data != expected {
                    return Err(err(format!("trace changed frozen {name} output bits")));
                }
            }
            let p = out.join(format!("{name}.bin"));
            std::fs::write(&p, &data).map_err(err)?;
            files.insert(
                name.to_owned(),
                serde_json::json!({"bytes":bytes,"sha256":sha(&p)?}),
            );
            println!("CAPTURE {name} bytes={bytes}");
            Ok(())
        },
    )?;
    gpu.sync()?;
    if before != hashes(&inputs)? {
        return Err(err("inputs changed"));
    }
    let report = serde_json::json!({"capture_complete":true,"model_gate_pass":false,"scope":"reproduces failed v1 stage0/1 exactly; diagnostic only",
  "identities":before,"ids":ids,"drop":drop,"files":files,"gpu":{"index":gpu.index,"name":gpu.name,"pci":gpu.pci,"cc":gpu.cc}});
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
