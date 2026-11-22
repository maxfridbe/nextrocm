# Inkling prompt processing on the RX 550

**The explicit chunk-4 path has completed validation in the WOPR test silo.** Expert batching reduced prompt-processing time by **12.78%** on the measured 18-prompt-token / 32-generation-step workload. The longer 80-token comparisons and integrated adapter lifecycle checks have passed for chunks 2 and 4. The normal CLI independent audit and final cleanup/identity closure also passed. Default chunk size remains 1. Earlier measurements and commands remain documented in [INKLING-PERFORMANCE.md](INKLING-PERFORMANCE.md).

## Measured result

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

At context 4096, the reviewed chunk-4 expert-batched plan requires **3,977,445,136 free device bytes**, including the unchanged 1 GiB admission margin. The extra expert scratch is 423,936 bytes. Actual allocation admission and the external 3.75 GiB sampled-allocation cap remain required; an estimate alone does not prove fit.

## Completed CLI request

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
