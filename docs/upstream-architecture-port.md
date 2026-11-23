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
| Packaging | Explicit TU/flags manifest, isolated target profiles and variant deltas | Offline plan tests, source closure checks; hardware builds separately reported | CPU-tested; HIP build pending |
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

The loader, settings, device ABI and packaging framework are implemented. StableLM is the first concrete default silo, using its existing operations and per-GPU binding; no arithmetic changed. Other engine families continue to use their shared kind library or compatibility runtime. Their precise extraction boundaries are recorded in [the kernel migration inventory](../tooling/ARCHITECTURE-MIGRATION.md). This is not a claim that every engine has been extracted or validated on every HIP target.

The local development environment has no HIP compiler or GPU test queue. New kind/silo binaries, target-specific exports and numerical hardware gates therefore remain unvalidated. Existing model receipts do not validate a newly linked binary. No tuned variant or fabricated baseline has been added.

## Using the new infrastructure

- `nextrocm silo [ARCH]` reports device properties and silo/variant selection.
- `kernels/build.sh --target gfx803 --out /absolute/new/output` prints the isolated build plan; `--execute` compiles with a compatible HIP compiler. `gfx1035` is a separate profile.
- `tooling/gate.sh plan BASE HEAD` identifies impacted engines. Missing recorded baseline coverage is unvalidated, not a pass. See [runner and comparison contracts](../tooling/README.md).
- `nextrocm serve --tls-port 8443` serves HTTPS beside HTTP; `--tls-port 0` disables HTTPS. Explicit certificate/key paths are load-only and never overwritten.
- `nextrocm switch` exposes the existing chat switcher, including GPU reservation checks. Configure the same `NS_GPU_LOCK` path for services and external runners.
- Cards can declare `silo.variant` and generic `controls`/`annotations` metadata. No model-specific quality claims were imported.

The service/UI ports include per-copy idle unload and ages, drive management, one-of GPU placement, per-card admission, external GPU reservations, answer-room session selection, job fingerprints and safe cancellation selectors, speech streaming, and declared markup preservation. Existing loaded workers can answer while a reservation blocks new loads.

## Verification recorded on this branch

- Service suite: **185 passed**, no failures. The existing live-box sample test remains ignored; no live service was contacted.
- CLI unit tests: **18 passed**. Architecture integration test passed, including ten offline build/gate/mock-HIP checks.
- Core, sys, catalog and silo-binding suites: **5 + 5 + 9 + 6 passed**. The sys subprocess-only helper is marked ignored in the normal list and is invoked by its loader test for five isolated loading cases.
- GPU watchdog: **18 CPU/mock-process tests passed**, including timeout cleanup and checking pinned inputs under the queue lock before launch. No GPU job ran.
- All four existing frontends compile strictly (**82 modules**); mocked clock, unload, RPC and drive-confirmation tests passed. Generated API parity passed.
- Shell syntax and `git diff --check` passed. Existing unrelated unused-code warnings remain.

Hardware/compiler combinations not executed remain unvalidated. The new library layout is ready for compatible HIP compilation and subsequent device-specific regression gates; successful CPU tests do not establish GPU numerical or performance admission. No live service, GPU power setting, model weights, or inference arithmetic was changed.
