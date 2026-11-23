# Upstream architecture and functionality migration

Source snapshots fetched on 2026-11-23: nextcuda `9bee2b4d97e0df5a536f6a9325aa98953c0956e9`, nextsycl `d6ed19267284c7a52b49ee9b60fbad99cdad6de7`. Implementation branch: `port/upstream-architecture`, based on `e426a78`. This migration concerns runtime packaging, card selection, services, and UI; it does not add model engines.

## Architecture

Keep one default silo per engine architecture. A card variant is an optional library containing only changed operations, resolved before the default silo and shared engine library. Match real HIP properties (gfx target, wave width, integrated/discrete, memory, compute units and clock); CUDA compute capability is not an AMD capability. Missing optional properties remain unknown. Keep explicit `NEXTROCM_LIB` and legacy runtime-only installations compatible.

Variant manifests must reject unknown fields and unsafe settings, detect overlapping matches, and respect request/explicit/card/variant/default precedence. Memory admission uses reported free/total memory, not marketing names or fixed card sizes. Device speed ordering is an exposed heuristic, not a cross-generation performance guarantee.

Build manifests enumerate translation units and floating-point flags explicitly. Compiler profiles distinguish older wave64 and newer wave32 targets. A declared target is not evidence that a particular compiler/card combination passed validation. Regression tooling must distinguish unsupported, not run, missing baseline, numerical failure, and performance regression.

## Implementation and verification plan

| Area | Port | Verification | Status |
| --- | --- | --- | --- |
| Device/runtime | Additive HIP properties, per-kind libraries, default silo plus variant resolution | Parser/settings tests, optional ABI compatibility, loader tests | Implemented and CPU-tested |
| Packaging | Explicit TU/flags manifest, 18 architecture libraries and two target profiles | Both HIP builds, exact export inventories, RX 550 loading and core checks | Built for gfx803/gfx1035; execution tested on gfx803 |
| CLI/catalog | Silo inspection, variant settings and card configuration | CLI and option precedence tests | Implemented and CPU-tested |
| Service allocation | Answer-room session selection, per-card memory admission and one-of GPU selection | Pure selection tests and service fixtures | Implemented and CPU-tested |
| Lifecycle | Idle-only per-card unload, active-request protection, copy ages and frontdoor idle cleanup | Busy/idle/pinned/copy fixtures | Implemented and CPU-tested |
| UI | Per-card unload, relative ages, drives and model disk management | TypeScript and mocked UI tests | Implemented and CPU-tested |
| Job reuse | Request/build/model-file fingerprint for reentrant jobs | Changed-input and unchanged-bookkeeping tests | Implemented and CPU-tested |
| Streaming | Remove outer stream flags from inner speech generation requests | HTTP streamed-response test | Implemented and CPU-tested |
| Transport | HTTPS frontdoor and TCP/TLS WebSocket forwarding | Certificate and local transport tests | Implemented and CPU-tested |
| API contracts | Regenerate clients/docs after backend changes | Generated-file parity and UI checks | Regenerated; parity passed |

Favicons already exist in the target applications and need no replacement. Engine-dependent upstream endpoints are not advertised unless their corresponding engine exists in this port. No synthetic performance baselines will be committed. Existing deployed services remain untouched while this branch is built and tested.

## Architecture migration boundary

The loader, settings, device ABI and packaging framework now cover 18 default architecture libraries. Existing engine-private bindings select immutable per-device tables; shared device/context and utility operations remain shared, with a compatible fallback. The [kernel migration inventory](../tooling/ARCHITECTURE-MIGRATION.md) records every closure and the explicit target-specific Inkling source selection. No new inference arithmetic or tuned hardware variant was introduced.

Both target profiles were compiled in the existing WOPR HIP 5.7 / Clang 17 environment: each compiled 49 translation units and linked 23 libraries, with exact exports checked for all 18 architecture libraries. On the RX 550, all 18 libraries loaded successfully; actual gfx803/wave64/4 GiB properties, a bounded allocation/lifetime test, and the core copy/bounds/arithmetic test passed. gfx1035 has compilation/export coverage only in this study. Existing model receipts are not treated as numerical admission of every newly linked library, and no performance baseline was fabricated.

## Using the new infrastructure

- `nextrocm silo [ARCH]` reports device properties and silo/variant selection.
- `kernels/build.sh --target gfx803 --out /absolute/new/output` prints the isolated build plan; `--execute` compiles with a compatible HIP compiler. `gfx1035` is a separate profile.
- `tooling/gate.sh plan BASE HEAD` identifies impacted engines. Missing recorded baseline coverage is unvalidated, not a pass. See [runner and comparison contracts](../tooling/README.md).
- `nextrocm serve --tls-port 8443` serves HTTPS beside HTTP; `--tls-port 0` disables HTTPS. Explicit certificate/key paths are load-only and never overwritten.
- `nextrocm switch` exposes the existing chat switcher, including GPU reservation checks. Configure the same `NS_GPU_LOCK` path for services and external runners.
- Cards can declare `silo.variant` and generic `controls`/`annotations` metadata. No model-specific quality claims were imported.

The service/UI ports include per-copy idle unload and ages, drive management, one-of GPU placement, per-card admission, external GPU reservations, answer-room session selection, job fingerprints and safe cancellation selectors, speech streaming, and declared markup preservation. Existing loaded workers can answer while a reservation blocks new loads.

## Initial infrastructure verification (before the engine/UI completion below)

- Service suite: **185 passed**, no failures. The existing live-box sample test remains ignored; no live service was contacted.
- CLI unit tests: **18 passed**. Architecture integration test passed, including ten offline build/gate/mock-HIP checks.
- Core, sys, catalog and silo-binding suites: **5 + 5 + 9 + 6 passed**. The sys subprocess-only helper is marked ignored in the normal list and is invoked by its loader test for five isolated loading cases.
- GPU watchdog: **18 CPU/mock-process tests passed**, including timeout cleanup and checking pinned inputs under the queue lock before launch. No GPU job ran.
- All four existing frontends compile strictly (**82 modules**); mocked clock, unload, RPC and drive-confirmation tests passed. Generated API parity passed.
- Shell syntax and `git diff --check` passed. Existing unrelated unused-code warnings remain.

Those initial checks established CPU behavior only. Subsequent HIP compilation and isolated hardware validation are recorded below. Deployed services, GPU power settings, model weights and inference arithmetic were not changed.

## Testing UI completion and hardware checks

The image page now supports the capabilities of existing registered engines:
upload-and-upscale, background removal with an optional mask, and SVG conversion,
alongside its original generation workflow. Results can be previewed/downloaded
and saved image history is retained. Capability reporting drives the controls;
model names are not used to guess supported operations. The image worker also
preserves compatible LoRA options across loading and request-specific reloads.
See [testing UI instructions](testing-uis.md).

The updated native executable was tested with the new image kind library and
`NS_SILO_REQUIRED=1`. Real-ESRGAN passed the original RGB, tiled RGBA and singleton
CPU-reference gates, two native runs each, with zero tracked allocations after
engine drop. The original thresholds remained 40 dB per channel and maximum
error 16/255. Live compiled UI actions uploaded RGB and RGBA fixtures, downloaded
the results and verified history. Both downloaded outputs differed from their
CPU references by at most 1/255; alpha was exact. The actual HTTP-served 27-module
UI was byte-identical to the compiled modules used by the action test. The test
server exited normally, with its process group empty and sampled DRM allocation
peaking at 218,464,256 bytes. These are application-action and module checks,
not an automated visual-browser test or a production deployment.

Final CPU verification: 603 workspace library tests passed (three existing
ignored tests), 20 CLI tests passed, architecture integration passed, and the
standalone PaddleOCR/Qwen3VL suites passed 17/3 tests. Five subprocess loader
cases also ran inside the sys loader test. The original sampler fixture, missing
from the checkout, was restored byte-for-byte from both pinned upstream sources;
no expected values or tolerances were regenerated. Image interaction tests cover
upscale, SVG, cutout/mask, errors, retry, downloads/history and generation parity.

Remaining hardware coverage is explicit: other engines' newly linked libraries
have build/export/loading coverage but have not all rerun their full model
numerical suites, and no gfx1035 device execution occurred in this study.
Background-removal and SVG UI controls have mocked contract coverage here;
Real-ESRGAN additionally has live GPU/UI-action and independent pixel coverage.
The initial properties-only packaging probe correctly failed the strict memory
watchdog for having no measurable DRM allocation; its replacement added a real
64 KiB allocation/lifetime check rather than weakening the watchdog.

The audio page now exposes transcription for recognizers, including file upload,
advertised language/word/translation controls, timed results and text/JSON
downloads. Music and speech-generation controls remain available for their
capabilities. Two additional mock API tests and the audio interaction suite
passed. With the required lowvram silo, Whisper tiny.en transcribed the retained
11-second JFK WAV through the actual compiled UI actions twice, matching the
retained expected transcript exactly (apart from surrounding whitespace). All
39 served audio modules matched the tested build. This is a native transcript
regression and application-action check, not a general ASR accuracy study or
visual-browser test. Both isolated servers exited normally and their process
groups drained. Final device observation was automatic power policy, zero GPU
busy and no KFD processes.
