//! Synthetic complete-state chunk parity, not a real-text or HF reference.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_llm_inkling2::{
    config::MiniConfig, device::EmbeddingMode, expert_cache::ExpertMode, plan::Visit,
    text_graph::TextGraph,
};
use std::{cell::Cell, collections::BTreeMap, ffi::c_void, fs, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn ensure(v: bool, s: &str) -> Result<()> {
    if v {
        Ok(())
    } else {
        Err(s.into())
    }
}
fn hash(p: &Path) -> Result<String> {
    let o = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(p)
        .output()?;
    ensure(o.status.success(), "hash failed")?;
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("missing hash")?;
    ensure(
        h.len() == 64 && h.bytes().all(|b| b.is_ascii_hexdigit()),
        "bad hash",
    )?;
    Ok(h.into())
}
fn available() -> Result<u64> {
    let s = fs::read_to_string("/proc/meminfo")?;
    let n = s
        .lines()
        .find_map(|x| x.strip_prefix("MemAvailable:"))
        .and_then(|x| x.split_whitespace().next())
        .ok_or("memory missing")?
        .parse::<u64>()?;
    Ok(n.checked_mul(1024).ok_or("memory overflow")?)
}
type Captures = BTreeMap<(usize, usize), Vec<u8>>;
fn capture(g: &mut TextGraph, ids: &[u32], chunk: bool) -> Result<(Captures, Vec<Vec<f32>>)> {
    let mut files = Captures::new();
    let mut counts = vec![0usize; ids.len()];
    let mut observe = |row: usize, v: &Visit, b: &DevBuf| -> nextrocm_llm_inkling2::Result<()> {
        if row >= ids.len() || v.ordinal != counts[row] || b.len != 2304 * 4 {
            return Err("capture order/shape".into());
        }
        let mut raw = vec![0; b.len];
        b.read(0, &mut raw).map_err(|e| e.0)?;
        if raw
            .chunks_exact(4)
            .any(|x| !f32::from_le_bytes(x.try_into().unwrap()).is_finite())
        {
            return Err("nonfinite visit".into());
        }
        if files.insert((row, v.ordinal), raw).is_some() {
            return Err("duplicate visit".into());
        }
        counts[row] += 1;
        Ok(())
    };
    let logits = if chunk {
        g.chunk(ids, || Ok(()), &mut observe)?
    } else {
        let mut rows = Vec::new();
        for (row, &id) in ids.iter().enumerate() {
            rows.push(g.token(id, || Ok(()), |v, b| observe(row, v, b))?)
        }
        rows
    };
    ensure(
        counts.iter().all(|&n| n == 48) && logits.len() == ids.len(),
        "capture count",
    )?;
    ensure(
        logits
            .iter()
            .all(|r| r.len() == 200058 && r.iter().all(|x| x.is_finite())),
        "logits shape/finite",
    )?;
    Ok((files, logits))
}
fn exact(a: &(Captures, Vec<Vec<f32>>), b: &(Captures, Vec<Vec<f32>>)) -> Result<()> {
    ensure(a.0 == b.0, "visit byte mismatch")?;
    ensure(a.1.len() == b.1.len(), "logits rows")?;
    for (x, y) in a.1.iter().zip(&b.1) {
        ensure(
            x.len() == y.len() && x.iter().zip(y).all(|(x, y)| x.to_bits() == y.to_bits()),
            "logits bits mismatch",
        )?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure(
        args.len() == 4,
        "usage: wopr_expert_state GGUF GPU NEW_OUT; feature diagnostic-state required",
    )?;
    let model = fs::canonicalize(&args[1])?;
    let runtime = fs::canonicalize(std::env::var_os("NEXTROCM_LIB").ok_or("runtime required")?)?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let pins = [hash(&runtime)?, hash(&exe)?];
    ensure(
        pins[0] == "14f11de932424e7bf9fa22481382d632fdb26d20d97199a548f56822ef541ef2",
        "runtime differs",
    )?;
    let m = fs::metadata(&model)?;
    let identity = (m.len(), m.modified()?);
    ensure(identity.0 == 16610543328, "model size")?;
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(args[2].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let ptr = api()?.symbol("nr_allocation_stats");
    ensure(!ptr.is_null(), "stats missing")?;
    // SAFETY: exact pinned runtime ABI, GPU/runtime live through all calls.
    let stats: Stats = unsafe { std::mem::transmute(ptr) };
    let stat = || -> Result<[u64; 2]> {
        let (mut l, mut p) = (0, 0);
        ensure(
            unsafe { stats(gpu.raw(), &mut l, &mut p) } == 0,
            "stats failed",
        )?;
        Ok([l, p])
    };
    let mut checks = Vec::new();
    let result = (|| -> Result<()> {
        ensure(stat()?[0] == 0, "initial live")?;
        ensure(
            available()? >= (8u64 << 30) + (2u64 << 30),
            "host state-copy reserve",
        )?;
        let cfg = MiniConfig::from_official_json(include_bytes!("../assets/mini-config.json"))?;
        let mut g = TextGraph::load_with_expert_prefill_policies(
            &model,
            &cfg,
            &gpu,
            1030,
            available()?,
            ExpertMode::SegmentedLru {
                budget_bytes: 939524096,
            },
            EmbeddingMode::HostRow,
            4,
            true,
            || Ok(()),
        )?;
        let (pos, mut initial) = g.diagnostic_snapshot()?;
        ensure(
            pos == 0 && initial.len() == g.diagnostic_state_bytes(),
            "initial state layout",
        )?;
        for patterned in [false, true] {
            for (i, b) in initial.chunks_exact_mut(4).enumerate() {
                let x = if patterned {
                    ((i % 251) as i32 - 125) as f32 * 0.00001
                } else {
                    0.0
                };
                b.copy_from_slice(&x.to_le_bytes());
            }
            g.diagnostic_restore(1022, &initial)?;
            let serial = capture(&mut g, &[0, 1, 2, 3], false)?;
            let final_serial = g.diagnostic_snapshot()?;
            g.diagnostic_restore(1022, &initial)?;
            let batched = capture(&mut g, &[0, 1, 2, 3], true)?;
            exact(&serial, &batched)?;
            let final_chunk = g.diagnostic_snapshot()?;
            ensure(
                final_serial.0 == 1026 && final_chunk == final_serial,
                "full final KV/history/cursor differs",
            )?;
            checks.push(format!(
                "position1022_serial4_chunk4_full196outputs_and_state_pattern{patterned}"
            ));
        }
        // Observe completed mutation, then cancel at the next graph cancellation check.
        g.diagnostic_restore(1022, &initial)?;
        let mutated = Cell::new(false);
        let failed = g.chunk(
            &[0, 1, 2, 3],
            || {
                if mutated.get() {
                    Err("injected midvisit cancellation".into())
                } else {
                    Ok(())
                }
            },
            |_, _, _| {
                mutated.set(true);
                Ok(())
            },
        );
        ensure(
            mutated.get() && failed.is_err(),
            "midvisit cancellation not reached",
        )?;
        ensure(
            g.diagnostic_snapshot().is_err(),
            "poisoned snapshot accepted",
        )?;
        ensure(
            g.token(0, || Ok(()), |_, _| Ok(())).is_err(),
            "poisoned reuse accepted",
        )?;
        g.diagnostic_reset()?;
        let serial = capture(&mut g, &[0], false)?;
        let state = g.diagnostic_snapshot()?;
        g.diagnostic_reset()?;
        let chunk = capture(&mut g, &[0], true)?;
        exact(&serial, &chunk)?;
        ensure(
            g.diagnostic_snapshot()? == state,
            "reset recovery state mismatch",
        )?;
        checks.push("midvisit_cancel_poison_reject_snapshot_and_token_then_reset_fullbits".into());
        drop(g);
        gpu.sync()?;
        Ok(())
    })();
    let mut errors = Vec::new();
    if let Err(e) = gpu.sync() {
        errors.push(e.to_string());
    }
    let allocation = match stat() {
        Ok(v) => Some(v),
        Err(e) => {
            errors.push(e.to_string());
            None
        }
    };
    if !allocation.is_some_and(|v| v[0] == 0 && v[1] <= 4026531840) {
        errors.push("drop/cap".into());
    }
    for (p, h) in [(&runtime, &pins[0]), (&exe, &pins[1])] {
        if hash(p).ok().as_ref() != Some(h) {
            errors.push("identity changed/unreadable".into());
        }
    }
    if fs::metadata(&model)
        .and_then(|m| Ok((m.len(), m.modified()?)))
        .ok()
        != Some(identity)
    {
        errors.push("model identity changed".into());
    }
    let pass = result.is_ok() && errors.is_empty();
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":if pass{"SYNTHETIC_STATE_PASS"}else{"INCOMPLETE"},"checks":checks,"batch_experts":true,"error":result.err().map(|e|e.to_string()),"post_errors":errors,"allocation_after_drop":allocation,"runtime":runtime,"runtime_sha256":pins[0],"executable":exe,"executable_sha256":pins[1],"model":model,"model_bytes":identity.0,"scope":"synthetic complete zero/smallfinite KV+history atposition1022 context1030; serial4/chunk4 all196outputs and fullfinalstate byteparity, mutation/cancel poison/reset; NOTreal1022token text orHF reference; parentfullassetSHA prerequisite"}),
        )?,
    )?;
    ensure(pass, "state gate incomplete")
}
