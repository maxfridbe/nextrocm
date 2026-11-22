# Inkling prompt processing on the RX 550

## Selected dense configuration

The explicit chunk-4 dense-plus-expert path has completed numerical, state, adapter, normal-CLI, and final cleanup validation in this test silo. Default chunk size remains 1; this is not a production deployment or an HTTP throughput claim.

A subsequent four-leg baseline/candidate/candidate/baseline study compared **expert batching at chunk 4** with **expert plus dense batching at chunk 4**. All eight sampled timed phases reported PCIe **2.5 GT/s ×8**, with equal expert-cache counters and the same executable/runtime. Each leg used a fresh graph and the fixed 18-prompt-token / 32-generation-step workload.

| Configuration | Median prompt time | Prompt tokens/s | Median generation time | Median combined time |
|---|---:|---:|---:|---:|
| Expert batching, chunk 4 | 17.773 s | 1.013 | 36.650 s | 54.423 s |
| Expert plus dense batching, chunk 4 | 16.955 s | 1.062 | 36.620 s | 53.575 s |

Dense batching reduced prompt time by **4.60%**, increased prompt throughput by **4.82%**, and reduced combined time by **1.56%**. Generation was essentially unchanged. This is a separate study from the chunk-1/2/4 comparison below: do not add its percentages to the earlier gains or treat their controls as one continuous experiment. Mycroft's historical 1.736 prompt tokens/s remains unbeaten. These observations cover this fixed workload, not general prompt throughput.

The selected extension batches seven large Q8 projections: Q, K, V, relative-position R, attention gate, attention output O, and the final output head. Smaller projections remain scalar. It uses the same `14f11de932424e7bf9fa22481382d632fdb26d20d97199a548f56822ef541ef2` runtime. Per-row normalization, chronological convolution/KV/attention updates, and expert rank accumulation remain unchanged. The final head computes all 201,024 padded rows before softcapping and returning the original 200,058-token vocabulary.

Two reusable device buffers add **3,290,112 bytes for chunk 4**, bringing the context-4096 host-embedding/896 MiB segmented-cache plan to **3,980,735,248 required free device bytes**, including the unchanged 1 GiB admission margin. All scratch is accounted for before weight allocation. The ABBA study reached **3.220 GB sampled DRM allocation**; actual admission and the external 3.75 GiB cap remain required.

The dense candidate passed both 98-array reference checks, all 3,920 arrays and 63 greedy decisions in the 80-token regression, and the synthetic full-state/ring/cancellation probe. Both chunk-2 and chunk-4 integrated adapters passed all 21 lifecycle checks with dense batching active, including 400,116 saved logits matching the validated scalar path and zero remaining tracked allocations. The built executable path is `$suite/bin/nextrocm-inkling-dense-v1`, SHA-256 `1511f0fa7c83ceff458b8af4a70396f2d517c2b9dd5d27f561898bc6e20ad933` (local binary rehash verified). The normal CLI request returned `Hello!` and passed independent audit. Final closure verified the unchanged full model hash, all eight integrated source hashes, exact executable/runtime identities, automatic driver mode, 0% GPU busy, and no remaining KFD compute clients. Default chunk size remains 1, and `feed_until` stays serial: no HTTP speedup is established.

Retained study receipts: `inkling-prefill-dense-v1/dense-abba-ampere-summary-v1.json`, its independent audit, `dense-state-avicenna-audit.json`, `normal-cli-independent-aristotle-audit.json`, and `inkling-dense-final-close-v1.json`.

The selected CLI reported **14.80 s for 16 prompt tokens** and **52.60 s for 46 generated nonterminal tokens**, approximately **0.87 tokens/s**, with dense batching active. Generated tokens include reasoning/control tokens; this request still produced reasoning despite `--effort none`. The CLI run reached **3.219 GB sampled DRM allocation**. Whole-run link observations mixed 8 and 2.5 GT/s without prompt/generation phase boundaries. This 16/46 smoke request is distinct from the matched 18/32 study. Do not infer a speedup by comparing it with the historical CLI timings; the matched ABBA comparison establishes the 4.60% prompt-time reduction.

Validated current command:

```sh
suite=/var/home/maxfridbe/nextrocm-gpu-suite-v1
model=/var/home/maxfridbe/inkling2-reference-v1/assets/Inkling-2-Mini-UD-Q4_K_M.gguf
NEXTROCM_LIB="$suite/inkling-expert-batch-build-v1/libnextrocm-inkling-expert-batch.so" \
  "$suite/bin/nextrocm-inkling-dense-v1" llm generate "$model" \
  --prompt 'Say hello.' --max 64 --ctx 4096 --gpu 0 \
  --temp 0 --top-p 1 --seed 1 --effort none \
  --inkling-host-embedding --expert-gib .875 --inkling-cache-policy segmented \
  --inkling-prefill 4
```

## Earlier expert-batching study

The six-leg comparison used chunk sizes 1, 2, and 4, each twice, with fresh graphs. All sampled timed phases reported PCIe **2.5 GT/s ×8**, in automatic driver mode. The table reports the audited per-mode aggregate times from this study.

| Prompt chunk size | Prompt processing, 18 tokens | Prompt tokens/s | Generation, 32 steps | Combined time |
|---|---:|---:|---:|---:|
| 1, control | 20.405 s | 0.882 | 36.535 s | 56.940 s |
| 2 | 19.120 s | 0.941 | 36.654 s | 55.773 s |
| 4 | 17.798 s | 1.011 | 36.644 s | 54.442 s |

Chunk 4 improved prompt throughput by **14.65%** and reduced combined time by **4.39%**. Generation stayed serial and was approximately 0.30% slower in this small sample. This measures grouping and batched matrix operations together; it does not isolate a kernel-only speedup or establish results across prompts.

Mycroft's historical prompt-processing rate of **1.736 tokens/s** remains unbeaten. The earlier RX 550 observation of approximately 1.464 prompt tokens/s used an 8 GT/s link; comparing that number directly with this slow-link study would confound software and PCIe state.

## What changes

For each model visit, up to four prompt rows reach routing in chronological order. Rows selecting the same expert share one weight-pair access and a batched matrix operation. Each row retains its original arithmetic order, expert scales, and rank-ordered accumulation. Short-convolution history and attention-cache updates remain chronological. Every row's logits are computed, including rows whose logits the normal prompt-feed API does not return.

The explicit CLI opt-in is `--inkling-prefill 4`; supported values are 1, 2, and 4, with default 1. It composes with `--inkling-host-embedding --expert-gib .875 --inkling-cache-policy segmented --ctx 4096`. Use the matching integrated CLI below; older validated binaries do not establish support for this flag.

Only the ordinary prompt-feed path uses this batching. The cancellation-aware `feed_until` path remains serial. These measurements therefore establish no HTTP prompt-processing speedup. Decode behavior and default selection are unchanged.

For the earlier expert-only path at context 4096, the reviewed chunk-4 expert-batched plan requires **3,977,445,136 free device bytes**, including the unchanged 1 GiB admission margin. The extra expert scratch is 423,936 bytes. Actual allocation admission and the external 3.75 GiB sampled-allocation cap remain required; an estimate alone does not prove fit.

## Historical expert-only CLI request

The following command returned `Hello!` on stdout with expert batching enabled. The independent CLI audit passed: the watchdog recorded exit 0 and an empty process group. Final closure verified the unchanged model hash, exact executable/runtime hashes, automatic driver mode, 0% GPU busy, and no remaining KFD compute clients.

```sh
suite=/var/home/maxfridbe/nextrocm-gpu-suite-v1
model=/var/home/maxfridbe/inkling2-reference-v1/assets/Inkling-2-Mini-UD-Q4_K_M.gguf
NEXTROCM_LIB="$suite/inkling-expert-batch-build-v1/libnextrocm-inkling-expert-batch.so" \
  "$suite/bin/nextrocm-inkling-prefill-v1" llm generate "$model" \
  --prompt 'Say hello.' --max 64 --ctx 4096 --gpu 0 \
  --temp 0 --top-p 1 --seed 1 --effort none \
  --inkling-host-embedding --expert-gib .875 --inkling-cache-policy segmented \
  --inkling-prefill 4
```

CLI SHA-256: `95dc2c8980ed97060a94f0a562b1456b2a2f9e13f8dc0baea983af0ae1243c79`.

The CLI reported **15.56 s for 16 prompt tokens** and **52.59 s for 46 generated nonterminal tokens**, approximately **0.87 tokens/s**. Generated tokens include reasoning/control tokens; `--effort none` requests no reasoning but this run still emitted reasoning. This 16/46 request differs from the controlled 18/32 study above. Its timings are not a matched speed comparison with the older CLI or another PCIe state. Sampled DRM allocation peaked at **3.215 GB**; this is a sampled bound, not continuous measurement.

## Runtime and validation

The tested expert-batch runtime SHA-256 is:

```text
14f11de932424e7bf9fa22481382d632fdb26d20d97199a548f56822ef541ef2
```

Build from the frozen expert-batch source closure using its compile-only recipe:

```sh
# Set these to the extracted frozen closure and a NEW disk-backed output path.
export SOURCE_ROOT=/absolute/path/to/expert-batch-source
export OUT_DIR=/var/home/developer/inkling-expert-batch-build
bash "$SOURCE_ROOT/tests/wopr-llm/build-inkling-expert-batch.sh"
```

The output is `libnextrocm-inkling-expert-batch.so`, selected through `NEXTROCM_LIB` by the matching candidate harness. The recipe targets gfx803 and retains the strict arithmetic flags. A different compiler need not reproduce identical runtime bytes.

Completed checks include the original batched operator references, two-token 98-array comparisons for chunk 2 and the chunk-4 path with a two-row final chunk, and synthetic full-state parity across the 1024-position ring boundary. The state probe compares all 196 outputs and complete final KV/history/cursor state for four tokens starting at position 1022, using zero and finite patterned initial state. It also checks cancellation after a completed visit, poisoned-session rejection, and reset recovery. This is not a real 1022-token prompt or a HIP-failure injection test.

Both chunk sizes also passed the 80-token regression: all 3,920 arrays were byte-identical to the validated serial baseline, with the unchanged numerical gate and all 63 greedy decisions preserved. This is a native regression against the original teacher sequence, not an independent HF 80-token oracle.

Each integrated adapter run passed 21 checks covering leases, checkpoint/restore, stale decoders, cancellation, reset recovery, full-chunk and partial-chunk prompt feeds, and the serial `feed_until` contract. The two saved logit arrays (400,116 values) matched the validated scalar path byte-for-byte. These adapter checks used context 8; checkpoint equality is asserted inside the reviewed executable, and full checkpoints are not exported for offline comparison. Tracked allocations returned to zero.

The synthetic state run returned tracked allocations to zero and reached a sampled DRM allocation peak of **3.016 GB**. That measurement belongs to the context-1030 state probe, not a context-4096 memory claim. Full-array comparisons and state checks are separate from the timing study's token and counter checks.

Private retained receipts: `inkling-prefill-expert-v1/expert-sixleg-ampere-summary-v1.json`, its linked independent audit, the chunk-2/chunk-4 comparison audits, and `expert-state-avicenna-audit.json`. These receipts are local validation evidence, not bundled public fixtures. The final normal-request audit is `normal-cli-independent-aristotle-audit.json`; closure is `inkling-expert-final-close-v1.json`. This validates an explicit test-silo option, not a production deployment or default change.
