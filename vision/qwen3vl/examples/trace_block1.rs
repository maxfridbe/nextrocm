//! Diagnostic block1 replay: native input must reproduce frozen stage2 bits.
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
    if a.len() != 9 {
        return Err(err(
            "usage: trace_block1 WEIGHTS TOKENIZER CONFIG FAILED_PFIRST NEW_OUT GPU PROMPT CPU_STAGE1_F32",
        ));
    }
    let out = Path::new(&a[5]);
    if out.exists() {
        return Err(err("output must be new"));
    }
    let runtime = PathBuf::from(std::env::var("NEXTROCM_LIB").map_err(err)?);
    if !runtime.is_absolute()
        || sha(&runtime)? != "192110510c9d0041bdb0ffc8a09d27912e13d023220b1eaec9a71f1605203418"
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
        base.join("stage-01.f32"),
        base.join("stage-02.f32"),
        PathBuf::from(&a[8]),
    ]);
    for (path, expected) in [
        (
            &inputs[5],
            "1a2534d96fc8d292a9deef103b6f4493849c55e5e097f3f913027b7478545105",
        ),
        (
            &inputs[6],
            "b670860c85ceb79afe2659257a5f1bda8e6f19b0927be6b47dd9b8b0b670ceee",
        ),
        (
            &inputs[7],
            "e49773565656c883e9977117c50798f2f538f93a664978c911136cf3b4e1728b",
        ),
    ] {
        if sha(path)? != expected {
            return Err(err("pinned diagnostic tensor identity"));
        }
    }
    let before = hashes(&inputs)?;
    if before[&inputs[2].display().to_string()]
        != "8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f"
    {
        return Err(err("original checkpoint identity"));
    }
    let gpu = Gpu::open(a[6].parse().map_err(err)?)?;
    let engine = TextEncoder::load_with_attention(
        Path::new(&a[1]),
        Path::new(&a[2]),
        Path::new(&a[3]),
        gpu.clone(),
        true,
    )?;
    let (ids, drop) = engine.tokenize(&a[7])?;
    std::fs::create_dir(out).map_err(err)?;
    let mut files = BTreeMap::new();
    for (label, input_path) in [("native_input", &inputs[5]), ("cpu_input", &inputs[7])] {
        let raw = std::fs::read(input_path).map_err(err)?;
        if raw.len() != ids.len() * 4096 * 4 {
            return Err(err("input geometry"));
        }
        let values = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        let sub = out.join(label);
        std::fs::create_dir(&sub).map_err(err)?;
        engine.trace_block1(
            &a[7],
            &Budget {
                allowance_bytes: 7_500_000_000,
                driver_margin_bytes: 1024 * 1024 * 1024,
            },
            &values,
            &mut |name, buffer, bytes| {
                let mut data = vec![0; bytes];
                buffer.read(0, &mut data)?;
                if name == "block_input" && data != raw {
                    return Err(err("trace input bits changed"));
                }
                if name == "block_output"
                    && label == "native_input"
                    && data != std::fs::read(&inputs[6]).map_err(err)?
                {
                    return Err(err("trace changed frozen failed stage2 output bits"));
                }
                if !name.ends_with("quantized")
                    && !name.ends_with("accum")
                    && data
                        .chunks_exact(4)
                        .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
                {
                    return Err(err("nonfinite diagnostic float tap"));
                }
                let p = sub.join(format!("{name}.bin"));
                std::fs::write(&p, &data).map_err(err)?;
                files.insert(
                    format!("{label}/{name}"),
                    serde_json::json!({"bytes":bytes,"sha256":sha(&p)?}),
                );
                println!("CAPTURE {label}/{name} bytes={bytes}");
                Ok(())
            },
        )?;
    }
    gpu.sync()?;
    if before != hashes(&inputs)? {
        return Err(err("inputs changed"));
    }
    let report = serde_json::json!({"capture_complete":true,"model_gate_pass":false,"scope":"block1 native-input reproduces failed P-first stage2 exactly; exact CPU-input matched replay; diagnostic only","attention_probability_first":true,
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
