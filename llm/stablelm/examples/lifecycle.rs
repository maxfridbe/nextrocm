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
fn run() -> Result<(), Box<dyn Error>> {
    let file = std::env::args().nth(1).ok_or("lifecycle MODEL")?;
    nextrocm_engine_core::use_kind("llm");
    let g = Gpu::open(0)?;
    let f = Gguf::open(Path::new(&file))?;
    let tok = Tokenizer::from_gguf(&f)?;
    let e = nextrocm_llm_stablelm::load(
        &f,
        &[g.clone()],
        &LoadOptions {
            draft: false,
            kv: (4160, 2),
            ..Default::default()
        },
        &mut |s| eprintln!("{s}"),
    )?;
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
    let ck = e.save(&a)?;
    let next = tok.encode(" Paris")[0];
    let expected = e.feed(&mut a, &[next], &mut tap)?;
    e.restore(&mut a, &ck)?;
    equal(&expected, &e.feed(&mut a, &[next], &mut tap)?)?;
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
    println!("PASS: chunk/token logits, checkpoint restore, copied/batched sessions and stopped prefill; allocated={:?}",stats(&g)?);
    drop(ck);
    drop(a);
    drop(b);
    drop(e);
    let (c, p) = stats(&g)?;
    if c != 0 || p >= 8_000_000_000 {
        return Err(format!("memory gate current={c} peak={p}").into());
    }
    println!("PASS: 4096-token session capacity; cleanup current={c}, peak={p}");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
