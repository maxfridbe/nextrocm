//! Hardware-only lifecycle validation; not an application workflow.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::LoadOptions;
use nextrocm_tok::Tokenizer;
use std::{error::Error, path::Path};
fn tap(_: &str, _: &DevBuf) -> nextrocm_engine_core::Result<()> {
    Ok(())
}
fn equal(a: &[f32], b: &[f32]) -> Result<(), Box<dyn Error>> {
    if a.len() != b.len()
        || a.iter()
            .zip(b)
            .any(|(a, b)| !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-5)
    {
        return Err("logits mismatch".into());
    }
    Ok(())
}
fn stats(g: &Gpu) -> Result<(u64, u64), Box<dyn Error>> {
    let a = api()?;
    let p = a.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err("missing telemetry".into());
    }
    type F = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: runtime-owned C symbol of this signature, retained library and live outputs/GPU.
    let f: F = unsafe { std::mem::transmute(p) };
    let (mut c, mut p) = (0, 0);
    if unsafe { f(g.raw(), &mut c, &mut p) } != 0 {
        return Err(a.error().into());
    }
    Ok((c, p))
}
fn rss_kib() -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}
fn run() -> Result<(), Box<dyn Error>> {
    let file = std::env::args().nth(1).ok_or("lifecycle MODEL")?;
    nextrocm_engine_core::use_kind("llm");
    let g = Gpu::open(0)?;
    let f = Gguf::open(Path::new(&file))?;
    let tok = Tokenizer::from_gguf(&f)?;
    let kinds = [nextrocm_llm_qwen35::kind(), nextrocm_llm_qwen3moe::kind()];
    let kind = nextrocm_llm::kind_for(&kinds, &f)?;
    let options = LoadOptions {
        draft: false,
        kv: (4160, 2),
        ..Default::default()
    };
    let e = (kind.load)(&f, &[g.clone()], &options, &mut |s| eprintln!("{s}"))?;
    let mut a = e.session(4096)?;
    let mut b = e.session(64)?;
    let ids = tok.encode("The capital of France is");
    assert!(ids.len() > 2);
    let whole = e.feed(&mut a, &ids, &mut tap)?;
    let mut serial = Vec::new();
    for id in &ids {
        serial = e.feed(&mut b, &[*id], &mut tap)?
    }
    equal(&whole, &serial)?;
    e.rollback(&mut a, 0)?;
    equal(&whole, &e.feed(&mut a, &ids[ids.len() - 1..], &mut tap)?)?;
    let ck = e.save(&a)?;
    let next = tok.encode(" Paris")[0];
    let expected = e.feed(&mut a, &[next], &mut tap)?;
    e.restore(&mut a, &ck)?;
    equal(&expected, &e.feed(&mut a, &[next], &mut tap)?)?;
    e.rollback(&mut a, 0)?;
    equal(&expected, &e.feed(&mut a, &[next], &mut tap)?)?;
    let mut serialized = Vec::new();
    ck.write_to(&mut serialized)?;
    let roundtrip = e.read_checkpoint(&mut std::io::Cursor::new(&serialized))?;
    e.restore(&mut a, &roundtrip)?;
    equal(&expected, &e.feed(&mut a, &[next], &mut tap)?)?;
    drop(roundtrip);
    drop(serialized);
    e.restore(&mut a, &ck)?;
    e.copy_session(&mut b, &a)?;
    let rows = e.forward_batch(&mut [&mut a, &mut b], &[next, next], &mut tap)?;
    equal(&expected, &rows[0])?;
    equal(&expected, &rows[1])?;
    e.reset_session(&mut a)?;
    let (count, row) = e.feed_until(&mut a, &ids, &|_| true, &mut tap)?;
    assert_eq!(count, 0);
    assert!(row.is_empty());
    equal(&whole, &e.feed(&mut a, &ids, &mut tap)?)?;
    e.reset_session(&mut a)?;
    let chunk = e.prefill_chunk();
    let cancel_ids: Vec<_> = ids.iter().copied().cycle().take(chunk * 2 + 1).collect();
    let long_expected = e.feed(&mut a, &cancel_ids, &mut tap)?;
    e.reset_session(&mut a)?;
    let (count, partial_logits) = e.feed_until(&mut a, &cancel_ids, &|n| n >= chunk, &mut tap)?;
    assert_eq!(count, chunk);
    assert!(
        partial_logits.is_empty(),
        "cancelled prefill returned intermediate logits"
    );
    equal(
        &long_expected,
        &e.feed(&mut a, &cancel_ids[count..], &mut tap)?,
    )?;
    println!(
        "PASS: chunk/token logits, checkpoint serialization/restore, rollback, copied/batched sessions and partial cancellation/reuse; allocated={:?}",
        stats(&g)?
    );
    drop(ck);
    drop(a);
    drop(b);
    let before_drop_rss = rss_kib();
    drop(e);
    let (c, p) = stats(&g)?;
    if c != 0 || p >= 8_000_000_000 {
        return Err(format!("memory gate current={c} peak={p}").into());
    }
    println!("PASS: 4096-token session capacity; cleanup current={c}, peak={p}");
    println!("RSS KiB before engine drop={before_drop_rss:?}, after={:?} (diagnostic, not allocator accounting)", rss_kib());
    let host_mib = match std::env::var("NS_Q35_HOST_MIB") {
        Ok(value) => value.parse::<u64>()?,
        Err(std::env::VarError::NotPresent) => 0,
        Err(e) => return Err(e.into()),
    };
    if host_mib > 0 {
        // A leaked 2 GiB mirror reservation cannot reload under the 3 GiB
        // host cap. Reuse the exact kind/options, without allocating sessions.
        let before_reload_rss = rss_kib();
        let reloaded = (kind.load)(&f, &[g.clone()], &options, &mut |s| {
            eprintln!("reload: {s}")
        })?;
        let loaded_rss = rss_kib();
        drop(reloaded);
        let (c, p) = stats(&g)?;
        if c != 0 || p >= 8_000_000_000 {
            return Err(format!("reload memory gate current={c} peak={p}").into());
        }
        println!("PASS: host mirror {host_mib} MiB reload/drop; cleanup current={c}, peak={p}; RSS KiB before={before_reload_rss:?}, loaded={loaded_rss:?}, after={:?}", rss_kib());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
