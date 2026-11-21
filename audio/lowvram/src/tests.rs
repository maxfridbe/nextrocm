use super::*;
use serde_json::json;

fn caps() -> Recogniser {
    Recogniser {
        languages: vec!["en".into(), "de".into(), "zh".into()],
        translate: true,
        words: false,
    }
}
fn request() -> TranscribeRequest {
    TranscribeRequest {
        samples: vec![0.0; 1600],
        rate: 16000,
        ..Default::default()
    }
}
#[test]
fn language_names_aliases_and_checkpoint_limits() {
    assert_eq!(
        language::normalize(Some(" German "), &caps().languages).unwrap(),
        "de"
    );
    assert_eq!(
        language::normalize(Some("Mandarin"), &caps().languages).unwrap(),
        "zh"
    );
    assert_eq!(
        language::normalize(Some("unknown"), &caps().languages).unwrap(),
        "auto"
    );
    assert!(language::normalize(Some("french"), &caps().languages).is_err());
    assert!(language::normalize(Some("en\0"), &caps().languages).is_err());
}
#[test]
fn unsupported_requests_fail_before_ffi() {
    let caps = caps();
    let mut req = request();
    assert!(validate(&req, &caps).is_ok());
    req.words = true;
    assert!(validate(&req, &caps)
        .unwrap_err()
        .0
        .contains("word timestamps"));
    req.words = false;
    req.prompt = Some("names".into());
    assert!(validate(&req, &caps).unwrap_err().0.contains("prompts"));
    req.prompt = None;
    req.extra.insert("q8".into(), "1".into());
    assert!(validate(&req, &caps)
        .unwrap_err()
        .0
        .contains("request options"));
    req.extra.clear();
    req.rate = 0;
    assert!(validate(&req, &caps).is_err());
    req.rate = u32::MAX;
    assert!(validate(&req, &caps).is_err());
    req.rate = 16000;
    req.samples[0] = f32::NAN;
    assert!(validate(&req, &caps).is_err());
    req.samples.clear();
    assert!(validate(&req, &caps).is_err());
    let req = TranscribeRequest {
        translate: true,
        ..request()
    };
    assert!(validate(
        &req,
        &Recogniser {
            translate: false,
            ..caps
        }
    )
    .is_err());
}
#[test]
fn metadata_and_results_are_checked() {
    assert!(parse_info(&json!({"languages": ["en"], "translate": false})).is_ok());
    for v in [
        json!({}),
        json!({"languages":[],"translate":true}),
        json!({"languages":["klingon"],"translate":false}),
        json!({"languages":["en"]}),
    ] {
        assert!(parse_info(&v).is_err());
    }
    let mut v = json!({"text":"hello 世界", "language":"de", "segments":[{"start":0.0,"end":0.1,"text":"hello 世界"}]});
    let out = transcript(&v, 0.1, &request(), &caps()).unwrap();
    assert_eq!(out.text, "hello 世界");
    assert_eq!(out.duration, 0.1);
    assert!(out.words.is_empty());
    assert_eq!(out.segments[0].probability, None);
    let req = TranscribeRequest {
        translate: true,
        ..request()
    };
    assert_eq!(transcript(&v, 0.1, &req, &caps()).unwrap().language, "en");
    v["segments"][0]["end"] = json!(-1);
    assert!(transcript(&v, 0.1, &request(), &caps()).is_err());
    assert!(transcript(&json!({"text":"x"}), 0.1, &request(), &caps()).is_err());
}
#[test]
fn callback_cancellation_is_preserved_and_latched() {
    let mut calls = 0;
    let mut cb = |_: Step| {
        calls += 1;
        Err(Error("user cancelled 42".into()))
    };
    let mut state = Callback {
        progress: &mut cb,
        start: Instant::now(),
        error: None,
    };
    assert_eq!(state.emit("tokens", 0, 1), 1);
    assert_eq!(state.emit("tokens", 1, 1), 1);
    assert_eq!(state.error.as_ref().unwrap().0, "user cancelled 42");
    drop(state);
    assert_eq!(calls, 1);
}
#[test]
fn callback_panic_cannot_cross_c_boundary() {
    let mut cb = |_: Step| -> Result<()> { panic!("user callback panic") };
    let mut state = Callback {
        progress: &mut cb,
        start: Instant::now(),
        error: None,
    };
    // SAFETY: live callback state and NUL-terminated static phase through this call.
    let rc = unsafe {
        progress(
            (&mut state as *mut Callback<'_>).cast(),
            c"tokens".as_ptr(),
            0,
            1,
        )
    };
    assert_eq!(rc, 1);
    assert!(state.error.unwrap().0.contains("panicked"));
}
#[test]
fn callback_maps_phases_and_saturates_counters() {
    let mut seen = Vec::new();
    let mut cb = |s: Step| {
        seen.push((s.phase, s.at, s.of));
        Ok(())
    };
    let mut state = Callback {
        progress: &mut cb,
        start: Instant::now(),
        error: None,
    };
    // SAFETY: live callback state and static phase through each synchronous call.
    unsafe {
        assert_eq!(
            progress(
                (&mut state as *mut Callback<'_>).cast(),
                c"stt_chunk".as_ptr(),
                -1,
                i64::MAX
            ),
            0
        );
        assert_eq!(
            progress(
                (&mut state as *mut Callback<'_>).cast(),
                std::ptr::null(),
                1,
                2
            ),
            0
        );
    }
    drop(state);
    assert_eq!(seen, vec![("listen", 0, u32::MAX), ("listen", 1, 2)]);
}
#[test]
fn loader_exports_match_contract_and_keep_whisper_distinct() {
    let _: nextrocm_audio::LoadFn = load_canary;
    let _: nextrocm_audio::LoadFn = load_whisper;
    assert_eq!(canary_kind().roles, &["model"]);
    assert_eq!(whisper_kind().archs, &["whisper-ggml"]);
}
