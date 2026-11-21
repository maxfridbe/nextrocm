//! First native text slice: actual packed token row -> F32 decode -> token RMS.
//! Captures intermediates; numeric acceptance belongs to the independent oracle.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::{GType, Gguf};
use nextrocm_llm_inkling2::{
    config::MiniConfig, primitives::Primitives, quant::Quant, weights::Weights,
};
use std::{ffi::c_void, fs, path::PathBuf, process::Command};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(path: &std::path::Path) -> Result<String> {
    let o = Command::new("sha256sum").arg("--").arg(path).output()?;
    if !o.status.success() {
        return Err("hash command failed".into());
    }
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("missing hash")?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("usage: inspect_embedding GGUF OFFICIAL_CONFIG TOKEN_ID GPU_INDEX NEW_OUTPUT; NEXTROCM_LIB required".into());
    }
    let model = fs::canonicalize(&a[1])?;
    let cfg = fs::canonicalize(&a[2])?;
    let token: usize = a[3].parse()?;
    if token >= 200058 {
        return Err("token outside unpadded vocabulary".into());
    }
    let runtime = fs::canonicalize(PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB required")?,
    ))?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let hashes = [hash(&cfg)?, hash(&runtime)?, hash(&exe)?];
    let _config = MiniConfig::from_official_json(&fs::read(&cfg)?)?;
    let file = Gguf::open(&model)?;
    let _weights = Weights::read(&file)?;
    let meta = fs::metadata(&model)?;
    let stamp = (meta.len(), meta.modified()?);
    let emb = file
        .tensor("token_embd.weight")
        .ok_or("missing embedding")?;
    let norm = file
        .tensor("token_embd_norm.weight")
        .ok_or("missing token norm")?;
    let row_bytes = 2304 / 32 * 34;
    let mut packed = vec![0u8; row_bytes];
    let mut gamma = vec![0u8; 2304 * 4];
    file.read_into(emb, (token * row_bytes) as u64, &mut packed)?;
    file.read_into(norm, 0, &mut gamma)?;
    if gamma
        .chunks_exact(4)
        .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
    {
        return Err("nonfinite norm weights".into());
    }
    let out = PathBuf::from(&a[5]);
    fs::create_dir(&out)?;
    fs::write(out.join("embedding.q8_0"), &packed)?;
    fs::write(out.join("gamma.f32"), &gamma)?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(a[4].parse()?)?;
    let q = Quant::load(&gpu)?;
    let p = Primitives::load(&gpu)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let raw = api()?.symbol("nr_allocation_stats");
    if raw.is_null() {
        return Err("allocation stats required".into());
    }
    // SAFETY: exact private ns stats ABI, permanent runtime and live GPU.
    let stats: Stats = unsafe { std::mem::transmute(raw) };
    let stat = || -> Result<(u64, u64)> {
        let (mut live, mut peak) = (0, 0);
        if unsafe { stats(gpu.raw(), &mut live, &mut peak) } != 0 {
            return Err("allocation stats failed".into());
        }
        Ok((live, peak))
    };
    if stat()?.0 != 0 {
        return Err("slice requires an empty device".into());
    }
    let (total, free) = gpu.memory()?;
    if free.is_none_or(|n| n < 1024 * 1024) {
        return Err("missing/insufficient free device memory".into());
    }
    let w = DevBuf::new(&gpu, packed.len())?;
    let g = DevBuf::new(&gpu, gamma.len())?;
    let decoded = DevBuf::new(&gpu, 2304 * 4)?;
    let normalized = DevBuf::new(&gpu, 2304 * 4)?;
    w.write(0, &packed)?;
    g.write(0, &gamma)?;
    q.decode(&w, GType::Q8_0, 1, 2304, &decoded)?;
    p.rms(&decoded, Some(&g), 1, 2304, &normalized)?;
    gpu.sync()?;
    for (name, b) in [("decoded.f32", &decoded), ("normalized.f32", &normalized)] {
        let mut v = vec![0u8; 2304 * 4];
        b.read(0, &mut v)?;
        if v.chunks_exact(4)
            .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
        {
            return Err("nonfinite native output".into());
        }
        fs::write(out.join(name), v)?;
    }
    drop((w, g, decoded, normalized));
    gpu.sync()?;
    let (live, peak) = stat()?;
    if live != 0 || peak >= 1024 * 1024 {
        return Err("slice allocation/drop bound failed".into());
    }
    let mut again = vec![0u8; packed.len()];
    file.read_into(emb, (token * row_bytes) as u64, &mut again)?;
    if again != packed {
        return Err("embedding row changed".into());
    }
    again.resize(gamma.len(), 0);
    file.read_into(norm, 0, &mut again)?;
    if again != gamma {
        return Err("norm changed".into());
    }
    let after = fs::metadata(&model)?;
    if (after.len(), after.modified()?) != stamp
        || [hash(&cfg)?, hash(&runtime)?, hash(&exe)?] != hashes
    {
        return Err("identity changed".into());
    }
    let mut outputs = serde_json::Map::new();
    for name in [
        "embedding.q8_0",
        "gamma.f32",
        "decoded.f32",
        "normalized.f32",
    ] {
        outputs.insert(name.into(),serde_json::json!({"sha256":hash(&out.join(name))?,"bytes":fs::metadata(out.join(name))?.len()}));
    }
    let report = serde_json::json!({"status":"CAPTURE_COMPLETE","numeric_validation_pass":false,"scope":"Actual token embedding decode and token RMS only; no transformer layer/generation or full-payload hash claim","model":model,"model_bytes":stamp.0,"config":cfg,"config_sha256":hashes[0],"runtime":runtime,"runtime_sha256":hashes[1],"executable":exe,"executable_sha256":hashes[2],"token_id":token,"width":2304,"outputs":outputs,"device_total":total,"device_free_before":free,"tracked_peak":peak,"live_after_drop":live});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
