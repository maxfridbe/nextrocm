//! Private shared-hook lifecycle gate; no network/service execution.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm::LoadOptions;
use nextrocm_llm_inkling2::adapter;
use std::{
    ffi::c_void,
    fs,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};
macro_rules! check {
    ($condition:expr) => {
        if !$condition {
            return Err(concat!("lifecycle gate failed: ", stringify!($condition)).into());
        }
    };
}
fn hash(p: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let o = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(p)
        .output()?;
    if !o.status.success() {
        return Err("hash failed".into());
    }
    Ok(String::from_utf8(o.stdout)?
        .split_whitespace()
        .next()
        .ok_or("hash missing")?
        .to_string())
}
fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == 200058
        && a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits())
}
fn rejects_stale(
    engine: &dyn nextrocm_llm::Engine,
    session: &mut nextrocm_llm::Session,
    decoder: &mut nextrocm_llm::Decoder,
) -> bool {
    let before = session.pos();
    let mut sampled = false;
    let result = engine.step(
        session,
        decoder,
        &mut |_: &[f32]| {
            sampled = true;
            0
        },
        &mut |_, _| Ok(()),
    );
    result.is_err() && !sampled && session.pos() == before
}
struct RejectSampler {
    fallback_called: bool,
}
impl nextrocm_llm::Sampler for RejectSampler {
    fn sample(&mut self, _: &[f32]) -> u32 {
        self.fallback_called = true;
        0
    }
    fn try_sample(&mut self, _: &[f32]) -> nextrocm_llm::Result<u32> {
        Err(nextrocm_llm::Error(
            "intentional sampler validation failure".into(),
        ))
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err(
            "usage: wopr_dense_adapter GGUF GPU NEW_OUT CHUNK2_OR4; NEXTROCM_LIB required".into(),
        );
    }
    let chunk: usize = a[4].parse()?;
    check!(matches!(chunk, 2 | 4));
    let out = Path::new(&a[3]);
    fs::create_dir(out)?;
    let runtime = fs::canonicalize(std::env::var_os("NEXTROCM_LIB").ok_or("runtime required")?)?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let pins = [hash(&runtime)?, hash(&exe)?];
    check!(pins[0] == "14f11de932424e7bf9fa22481382d632fdb26d20d97199a548f56822ef541ef2");
    let file = Gguf::open(Path::new(&a[1]))?;
    let meta = fs::metadata(&a[1])?;
    let identity = (meta.len(), meta.modified()?);
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(a[2].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let p = api()?.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err("allocation stats missing".into());
    }
    let stats: Stats = unsafe { std::mem::transmute(p) };
    let cancel = Arc::new(AtomicBool::new(false));
    let mut checks = Vec::new();
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let (mut initial_live, mut initial_peak) = (0, 0);
        let initial_rc = unsafe { stats(gpu.raw(), &mut initial_live, &mut initial_peak) };
        check!(initial_rc == 0 && initial_live == 0);
        checks.push("initial_tracked_live_zero");
        let mut dense_enabled_log = false;
        let engine = adapter::load_with_prefill_policies(
            &file,
            &[gpu.clone()],
            &LoadOptions {
                kv: (8, 1),
                ..Default::default()
            },
            &mut |s| {
                dense_enabled_log |=
                    s.contains("expert batchtrue, dense batchtrue (feed_until serial)");
                println!("{s}");
            },
            cancel.clone(),
            nextrocm_llm_inkling2::expert_cache::ExpertMode::SegmentedLru {
                budget_bytes: 939524096,
            },
            nextrocm_llm_inkling2::device::EmbeddingMode::HostRow,
            chunk,
        )?;
        check!(dense_enabled_log);
        let mut s = engine.session(8)?;
        check!(engine.session(8).is_err());
        checks.push("second_live_session_rejected");
        let mut noop = |_: &str, _: &nextrocm_engine_core::DevBuf| Ok(());
        let first = engine.forward(&mut s, &[0], &mut noop)?;
        let ck = engine.save(&s)?;
        let second = engine.forward(&mut s, &[1], &mut noop)?;
        let mut stale_restore = engine.decoder(second.clone(), false);
        engine.restore(&mut s, &ck)?;
        check!(rejects_stale(engine.as_ref(), &mut s, &mut stale_restore));
        checks.push("stale_decoder_after_restore_rejected_before_sampling");
        let restored = engine.forward(&mut s, &[1], &mut noop)?;
        check!(same(&second, &restored));
        checks.push("restore_continuation_full_logits_bitwise");
        let mut serialized = Vec::new();
        ck.write_to(&mut serialized)?;
        let ck2 = engine.read_checkpoint(&mut serialized.as_slice())?;
        engine.restore(&mut s, &ck2)?;
        check!(same(&second, &engine.forward(&mut s, &[1], &mut noop)?));
        checks.push("serialized_complete_state_continuation_bitwise");
        let mut stale_reset = engine.decoder(second.clone(), false);
        engine.reset_session(&mut s)?;
        check!(rejects_stale(engine.as_ref(), &mut s, &mut stale_reset));
        checks.push("stale_decoder_after_reset_rejected_before_sampling");
        check!(same(&first, &engine.forward(&mut s, &[0], &mut noop)?));
        checks.push("reset_token0_bitwise");
        engine.reset_session(&mut s)?;
        let failed = engine.forward(&mut s, &[0], &mut |_, _| {
            cancel.store(true, Ordering::Relaxed);
            Ok(())
        });
        check!(failed.is_err());
        cancel.store(false, Ordering::Relaxed);
        check!(engine.forward(&mut s, &[0], &mut noop).is_err());
        check!(engine.save(&s).is_err());
        checks.push("midtoken_cancel_poison_rejects_reuse_and_save");
        engine.reset_session(&mut s)?;
        check!(same(&first, &engine.forward(&mut s, &[0], &mut noop)?));
        checks.push("reset_after_cancel_bitwise");
        let mut stale_new = engine.decoder(first.clone(), false);
        drop(s);
        let mut fresh = engine.session(8)?;
        check!(rejects_stale(engine.as_ref(), &mut fresh, &mut stale_new));
        checks.push("stale_decoder_after_new_lease_rejected_before_sampling");
        check!(same(&first, &engine.forward(&mut fresh, &[0], &mut noop)?));
        checks.push("new_request_zero_state_bitwise");
        let mut greedy = |x: &[f32]| {
            x.iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                .unwrap()
                .0 as u32
        };
        let expected = greedy(&first);
        let mut dec = engine.decoder(first.clone(), false);
        cancel.store(true, Ordering::Relaxed);
        let mut sampled = false;
        check!(engine
            .step(
                &mut fresh,
                &mut dec,
                &mut |_: &[f32]| {
                    sampled = true;
                    0
                },
                &mut noop
            )
            .is_err());
        check!(!sampled);
        cancel.store(false, Ordering::Relaxed);
        checks.push("cancelled_pending_logits_never_samples");
        let generated = engine.step(&mut fresh, &mut dec, &mut greedy, &mut noop)?;
        check!(generated == [expected]);
        check!(dec.pending() == Some(expected));
        checks.push("existing_sampler_and_pending_contract");
        check!(engine.max_sessions() == 1 && engine.default_top_p() == 1.0);
        check!(matches!(
            engine.output_format(),
            nextrocm_llm::OutputFormat::Inkling2
        ));
        check!(matches!(
            engine.sampling_policy(),
            nextrocm_llm::SamplingPolicy::FullVocabulary
        ));
        checks.push("explicit_inkling_policy_and_single_session_hooks");
        let mut stopped_decoder = engine.decoder(first.clone(), false);
        let mut sampled_by_stopped = false;
        let prior_position = fresh.pos();
        check!(engine
            .step_until(
                &mut fresh,
                &mut stopped_decoder,
                &mut |_: &[f32]| {
                    sampled_by_stopped = true;
                    0
                },
                &|| true,
                &mut noop
            )
            .is_err());
        check!(!sampled_by_stopped && fresh.pos() == prior_position);
        checks.push("stop_callback_before_ready_logits_never_samples");
        let mut reject = RejectSampler {
            fallback_called: false,
        };
        check!(engine
            .step_until(
                &mut fresh,
                &mut stopped_decoder,
                &mut reject,
                &|| false,
                &mut noop
            )
            .is_err());
        check!(!reject.fallback_called && fresh.pos() == prior_position);
        check!(
            engine.step_until(
                &mut fresh,
                &mut stopped_decoder,
                &mut greedy,
                &|| false,
                &mut noop
            )? == [expected]
        );
        checks.push("fallible_sampler_error_no_fallback_no_commit_retry_valid");
        let stop_at_visit = AtomicBool::new(false);
        let mut pending = engine.decoder_after(1, false);
        let mut sampled_after_cancel = false;
        check!(engine
            .step_until(
                &mut fresh,
                &mut pending,
                &mut |_: &[f32]| {
                    sampled_after_cancel = true;
                    0
                },
                &|| stop_at_visit.load(Ordering::Relaxed),
                &mut |_, _| {
                    stop_at_visit.store(true, Ordering::Relaxed);
                    Ok(())
                }
            )
            .is_err());
        check!(!sampled_after_cancel);
        stop_at_visit.store(false, Ordering::Relaxed);
        check!(engine.forward(&mut fresh, &[1], &mut noop).is_err());
        engine.reset_session(&mut fresh)?;
        check!(same(&first, &engine.forward(&mut fresh, &[0], &mut noop)?));
        checks.push("stop_callback_midvisit_poison_then_reset_fullbits");
        // Compare complete adapter checkpoints, not just visible cursor/logits.
        fn state_bytes(
            engine: &dyn nextrocm_llm::Engine,
            s: &nextrocm_llm::Session,
        ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
            let checkpoint = engine.save(s)?;
            let mut bytes = Vec::new();
            checkpoint.write_to(&mut bytes)?;
            Ok(bytes)
        }
        check!(engine.prefill_chunk() == 1);
        let ids: Vec<u32> = (0..(chunk + 1) as u32).collect();
        for prefix in [false, true] {
            engine.reset_session(&mut fresh)?;
            if prefix {
                engine.forward(&mut fresh, &[7], &mut noop)?;
            }
            let start = fresh.pos();
            let mut serial_last = Vec::new();
            for &id in &ids {
                serial_last = engine.forward(&mut fresh, &[id], &mut noop)?;
            }
            let serial_state = state_bytes(engine.as_ref(), &fresh)?;
            engine.reset_session(&mut fresh)?;
            if prefix {
                engine.forward(&mut fresh, &[7], &mut noop)?;
            }
            let actual = engine.feed(&mut fresh, &ids, &mut noop)?;
            check!(fresh.pos() == start + ids.len());
            check!(same(&actual, &serial_last));
            check!(state_bytes(engine.as_ref(), &fresh)? == serial_state);
            checks.push(if prefix {
                "chunk_feed_nonempty_prefix_tail_full_logits_state_position"
            } else {
                "chunk_feed_zero_prefix_tail_full_logits_state_position"
            });
        }
        engine.reset_session(&mut fresh)?;
        let (consumed, stopped) = engine.feed_until(&mut fresh, &ids, &|n| n == 1, &mut noop)?;
        check!(consumed == 1 && fresh.pos() == 1 && same(&stopped, &first));
        let stopped_state = state_bytes(engine.as_ref(), &fresh)?;
        let mut original_first_state = Vec::new();
        ck.write_to(&mut original_first_state)?;
        check!(stopped_state == original_first_state);
        checks.push("feed_until_stop1_exact_one_token_full_state_serial_contract");
        engine.reset_session(&mut fresh)?;
        let failed = engine.feed(&mut fresh, &ids, &mut |_, _| {
            cancel.store(true, Ordering::Relaxed);
            Ok(())
        });
        check!(failed.is_err() && fresh.pos() == 0);
        cancel.store(false, Ordering::Relaxed);
        check!(engine.save(&fresh).is_err());
        check!(engine.feed(&mut fresh, &ids, &mut noop).is_err());
        engine.reset_session(&mut fresh)?;
        check!(same(&first, &engine.feed(&mut fresh, &[0], &mut noop)?));
        check!(state_bytes(engine.as_ref(), &fresh)? == original_first_state);
        checks.push("chunk_feed_cancel_atomic_position_poison_then_reset_fullbits");
        for (name, row) in [
            ("token0-logits.f32", &first),
            ("token1-logits.f32", &second),
        ] {
            let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
            fs::write(out.join(name), bytes)?;
        }
        drop(fresh);
        drop(engine);
        Ok(())
    })();
    let mut post = Vec::new();
    if let Err(e) = gpu.sync() {
        post.push(e.to_string());
    }
    let (mut live, mut peak) = (0, 0);
    let rc = unsafe { stats(gpu.raw(), &mut live, &mut peak) };
    if rc != 0 || live != 0 || peak > 4_026_531_840 {
        post.push("allocation/drop check failed".into());
    }
    for (path, expected) in [(&runtime, &pins[0]), (&exe, &pins[1])] {
        match hash(path) {
            Ok(actual) if &actual == expected => {}
            Ok(_) => post.push("identity changed".into()),
            Err(e) => post.push(format!("identity verification failed: {e}")),
        }
    }
    match fs::metadata(&a[1]).and_then(|m| Ok((m.len(), m.modified()?))) {
        Ok(actual) if actual == identity => {}
        Ok(_) => post.push("asset metadata changed".into()),
        Err(e) => post.push(format!("asset metadata verification failed: {e}")),
    }
    let pass = result.is_ok() && post.is_empty();
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":if pass{"LIFECYCLE_PASS"}else{"INCOMPLETE"},"chunk":chunk,"dense_batch":true,"dense_enabled_log_required":true,"scope":"context8 original17 lifecycle gates plus plain-feed chunk2/4 prefix/tail full state and final logits, stop1 serial contract, chunk cancellation/reset; no service prefill batching or long-context admission","checks":checks,"error":result.as_ref().err().map(ToString::to_string),"post_errors":post,"runtime":runtime,"runtime_sha256":pins[0],"executable":exe,"executable_sha256":pins[1],"asset_bytes":identity.0,"allocation_after_drop":[live,peak]}),
        )?,
    )?;
    if !pass {
        return Err("adapter lifecycle failed; see report".into());
    }
    println!("LIFECYCLE_PASS");
    Ok(())
}
