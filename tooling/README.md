# Architecture build and regression infrastructure

`bash kernels/build.sh --target gfx803 --out /absolute/new/output` prints a
compile plan. Add `--execute` to compile; it does not run a GPU job or install
anything. `gfx1035` is also supported as a build profile. Use separate output
directories and compatible HIP toolchains. A successful compilation does not
establish numerical support for every engine on that target.

`kernels/build-manifest.json` lists translation units and arithmetic flags
explicitly, retaining the existing public runtime recipe. It produces kind
libraries and the compatibility `libnextrocm-runtime.so`. No model algorithms
or tuned variants are introduced. The tracked legacy recipe is
[`docs/public-release/build-runtime.sh`](../docs/public-release/build-runtime.sh),
not a `tests/public-release` path. Its repository-relative path and SHA-256
are recorded in the manifest's `provenance`; no private absolute path is needed.
Build profiles select HIP code objects and wave-size compile
definitions; they are not model-architecture silos or card-name variants.

Validation so far covers CPU tests and source/flag review. HIP is unavailable
in the current local test container, so the new kind/silo build, post-link
export checks and GPU numerical/hardware validation are **pending**. The
`gfx803` and `gfx1035` profile names are build configurations, not support or
performance guarantees. The legacy recipe's earlier receipts do not validate
the new library layout.

Optional `silos` entries declare an architecture, its exported operations and
translation units. Variants declare a name, a `variant.yml` path, a nonempty
subset of base exports and their own translation units/flags. A variant links
only its delta, with `-Bsymbolic`; its manifest is copied beside the library.
The Rust resolver is responsible for device matching, ambiguity rejection and
engine-option validation. The first concrete base silo is StableLM: the same
existing translation unit and flags, exporting exactly its seven `nslm_*`
operations. Export maps isolate its ABI, and a post-link `nm` check requires
actual exports to equal the declaration. The compatibility runtime still
contains the original operations. Host binding and GPU regression admission
remain separate gates. No tuned variants or hardware baselines are invented.
No library is named after a card or CUDA compute capability.

`bash tooling/gate.sh plan BASE HEAD` produces an affected-engine plan.
`bash tooling/gate.sh compare PLAN.json RESULTS.json` compares saved runner
receipts against declared per-device baselines. Missing coverage, an empty
baseline catalogue, or unmapped changes are **UNVALIDATED**, with a nonzero
exit status. The initial catalogue intentionally contains no invented results.
Variant changes conservatively require every recorded device for their engine.

The comparator requires exact outputs on every repeat, independent numerical
acceptance, successful exit/process cleanup, memory and idle gates, device
provenance, and candidate binary/runtime identities. Performance uses the
declared best of repetitions with a 3% regression limit. No nondeterministic
output exemption is enabled. Do not interpret a plan or an empty catalogue as a
merge gate pass.

`bash tooling/gate.sh run JOB.json --out /new/run --lock /shared/queue.lock
--amd-device /sys/class/drm/cardN/device` executes one explicitly authorized job
using the retained WOPR watchdog in `tooling/gpu`. A job declares `argv` (absolute
native ELF, followed by arguments), `executable_sha256`, `runtime` (absolute),
`inputs_sha256` (absolute path-to-hash mapping including runtime and model),
`timeout_seconds` (1–1800), `device_cap_bytes`, and optionally
`host_reserve_bytes` (at least 8 GiB). No shell command is interpreted. The runtime
is explicitly selected and all inputs are checked before and after execution.
The cap must not exceed the selected AMD device's reported sysfs VRAM total;
missing telemetry rejects the job. Owned DRM clients must match its PCI address.
The runner requires the shared exclusive lock, empty KFD compute inventory and
idle samples, monitors process-group DRM VRAM+GTT and host reserve, and drains
the owned group with TERM/KILL on timeout, signal or failure. Same-UID visibility
is not system-wide graphics exclusivity; the operator must own the GPU queue.
It never unloads a service or changes power settings. A watchdog PASS is only
execution/memory/cleanup evidence: architecture-specific numerical comparisons,
output hashes and timing extraction must still feed `compare` separately.

Run CPU-only checks with `python3 -B tooling/test_architecture.py` or the CLI
`architecture` integration test. Tests use synthetic receipts explicitly;
they neither record hardware baselines nor launch model inference.
