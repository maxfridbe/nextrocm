//! Bounded physical-layer0 capture. Not full-model inference or numeric PASS.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_llm_inkling2::{config::MiniConfig, first_layer::FirstLayer};
use std::{
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg("--").arg(p).output()?;
    if !o.status.success() {
        return Err("hash command failed".into());
    }
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("hash missing")?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn available() -> Result<u64> {
    let s = fs::read_to_string("/proc/meminfo")?;
    let n = s
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|l| l.split_whitespace().next())
        .ok_or("MemAvailable missing")?
        .parse::<u64>()?;
    Ok(n.checked_mul(1024).ok_or("memory overflow")?)
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err(
            "usage: inspect_layer0 GGUF CONFIG COMMA_TOKEN_IDS GPU NEW_OUT; NEXTROCM_LIB required"
                .into(),
        );
    }
    let tokens: Vec<u32> = a[3]
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    if tokens.is_empty() || tokens.len() > 16 || tokens.iter().any(|&t| t >= 200058) {
        return Err("requires1..16 valid unpadded token IDs".into());
    }
    let model = fs::canonicalize(&a[1])?;
    let cfg = fs::canonicalize(&a[2])?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let runtime = fs::canonicalize(PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB required")?,
    ))?;
    let pins = [hash(&cfg)?, hash(&exe)?, hash(&runtime)?];
    let metadata = fs::metadata(&model)?;
    let identity = (metadata.len(), metadata.modified()?);
    let config = MiniConfig::from_official_json(&fs::read(&cfg)?)?;
    let out = PathBuf::from(&a[5]);
    fs::create_dir(&out)?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(a[4].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let ptr = api()?.symbol("nr_allocation_stats");
    if ptr.is_null() {
        return Err("allocation stats missing".into());
    }
    // SAFETY: exact private ns stats ABI, retained runtime and live GPU.
    let stats: Stats = unsafe { std::mem::transmute(ptr) };
    let stat = || -> Result<(u64, u64)> {
        let (mut live, mut peak) = (0, 0);
        if unsafe { stats(gpu.raw(), &mut live, &mut peak) } != 0 {
            return Err("stats failed".into());
        }
        Ok((live, peak))
    };
    if stat()?.0 != 0 {
        return Err("capture requires empty device".into());
    }
    let start = Instant::now();
    let mut files = serde_json::Map::new();
    let mut timings = Vec::new();
    let result = (|| -> Result<()> {
        let begin = Instant::now();
        let mut graph =
            FirstLayer::load(&model, &config, &gpu, tokens.len(), available()?, || Ok(()))?;
        println!("loaded {:.6}s", begin.elapsed().as_secs_f64());
        for (i, &token) in tokens.iter().enumerate() {
            let dir = out.join(format!("token-{i:03}"));
            fs::create_dir(&dir)?;
            let begin = Instant::now();
            let mut count = 0;
            graph.token(token,||Ok(()),|name,b|{
    let mut raw=vec![0u8;b.len];b.read(0,&mut raw).map_err(|e|e.0)?;
    if name!="router_ids" && raw.chunks_exact(4).any(|x|!f32::from_le_bytes(x.try_into().unwrap()).is_finite()){return Err(format!("nonfinite {name}"))}
    let path=dir.join(format!("{name}.bin"));if path.exists(){return Err("duplicate capture name".into())}
    fs::write(&path,&raw).map_err(|e|e.to_string())?;
    let sha=hash(&path).map_err(|e|e.to_string())?;
    files.insert(format!("token-{i:03}/{name}.bin"),serde_json::json!({"sha256":sha,"bytes":b.len,"dtype":if name=="router_ids"{"i32"}else{"f32"}}));count+=1;Ok(())
   })?;
            if count != 52 {
                return Err(format!("unexpected layer0 boundary count {count}, expected52").into());
            }
            let seconds = begin.elapsed().as_secs_f64();
            println!("token {i} id={token} seconds={seconds:.6} captures={count}");
            timings.push(seconds);
        }
        drop(graph);
        gpu.sync()?;
        Ok(())
    })();
    gpu.sync()?;
    let (live, peak) = stat()?;
    let after = fs::metadata(&model)?;
    let unchanged = (after.len(), after.modified()?) == identity
        && [hash(&cfg)?, hash(&exe)?, hash(&runtime)?] == pins;
    let complete = result.is_ok() && live == 0 && peak < 7_500_000_000 && unchanged;
    let report = serde_json::json!({"status":if complete{"CAPTURE_COMPLETE"}else{"INCOMPLETE"},"numeric_validation_pass":false,"scope":"Physical layer0 only, F32 primitive baseline, actual quantized weights; independent original CPU gates pending, no full-model admission","error":result.as_ref().err().map(ToString::to_string),"model":model,"model_bytes":identity.0,"config":cfg,"config_sha256":pins[0],"executable":exe,"executable_sha256":pins[1],"runtime":runtime,"runtime_sha256":pins[2],"tokens":tokens,"context":tokens.len(),"files":files,"token_seconds":timings,"seconds":start.elapsed().as_secs_f64(),"live_after_drop":live,"tracked_peak":peak,"identities_unchanged":unchanged});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !complete {
        return Err("layer0 capture incomplete; see report".into());
    }
    println!("CAPTURE_COMPLETE numeric gate pending");
    Ok(())
}
