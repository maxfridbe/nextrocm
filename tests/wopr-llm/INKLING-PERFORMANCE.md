# Inkling Mini performance on the RX 550

The selected combination now measures **1.455 tokens/s** (repeat range **1.453–1.458**) on the fixed 18-prompt-token / 32-generation-step workload. It remains below Mycroft's measured **1.844 tokens/s** (about 79% of that rate). Both combined runs and their control had PCIe **8 GT/s ×8** throughout sampled timed phases. These are bounded workload results, not a general throughput guarantee. Final closure passed: the model hash was unchanged, driver mode was `auto`, GPU busy was 0%, and no KFD compute clients remained.

Earlier automatic-mode measurements improved from **0.468 to 0.702 tokens/s**, but those ran with a slower PCIe link. Do not attribute the entire difference between 0.702 and the new result to software changes.
WOPR has an RX 550 (gfx803, wave64) with 4 GiB GPU memory and approximately 64 GB host RAM. The verified quantized model is 16.61 GB. The selected configuration:

- Uses type-specialized logical-32 GPU kernels.
- Keeps the large input embedding in the file-backed host store and uploads only the selected 2,448-byte packed token row. The output projection remains on the GPU.
- Uses an explicit 896 MiB device expert-cache budget (`--expert-gib .875`) with segmented eviction and variable-size storage. The allocation is still bounded by 145 largest-pair slots worth of bytes; smaller pairs can share that capacity more efficiently.
- Computes independent depth-mixing dot products in parallel, preserving each dot product’s arithmetic order.
- Uses context capacity 4096 without weakening the existing memory admission margins.

The operating system serves model reads from reclaimable file page cache. Observed zero storage `read_bytes` during measured phases supports cache-served reads; this is not a pinned full-model mirror or a guarantee that all weights remain resident. The checked context-4096 plan requires 3,972,302,320 free device bytes including the existing 1 GiB margin. Actual admission still checks available memory.

## Completed normal CLI run

The following selected v2 command completed with `Hello!` on stdout. CLI SHA-256: `1453803e84abcc27ef8978694aee583cb21ebe4c4cbeffea5b9388dd60a8725f`. Independent normal-request audit passed, including the local executable rehash and drained process group. The older working binary in [INKLING-WAVE64.md](INKLING-WAVE64.md) is a separate baseline.

```sh
suite=/var/home/maxfridbe/nextrocm-gpu-suite-v1
model=/var/home/maxfridbe/inkling2-reference-v1/assets/Inkling-2-Mini-UD-Q4_K_M.gguf
NEXTROCM_LIB="$suite/inkling-depth-parallel-build-v1/libnextrocm-inkling-depth-parallel.so" \
  "$suite/bin/nextrocm-inkling-optimized-v2" llm generate "$model" \
  --prompt 'Say hello.' --max 64 --ctx 4096 --gpu 0 \
  --temp 0 --top-p 1 --seed 1 --effort none \
  --inkling-host-embedding --expert-gib .875 --inkling-cache-policy segmented
```

This normal CLI run measured **11.00 s for 16 prompt tokens** and **31.50 s for 46 generated tokens**, approximately **1.46 tokens/s**. Generated tokens include reasoning/control tokens and exclude EOS. This workload differs from the matched 18/32 benchmark. The previous v1 CLI measured 12.30 s / 35.70 s on the 16/46 workload; that historical observation is not a matched comparison with controlled PCIe state.

Required runtime SHA-256: `80d5652da06cf55785de048f559335c71cca2a60e550682904545f193936df8d`. Model SHA-256: `f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1`. `--effort none` requests no reasoning; actual runs still emitted reasoning. This is a test-silo configuration, not a production deployment or registry change.

For a runtime rebuild, use the staged frozen source closure, not the separate host/CLI overlay:

```sh
SOURCE_ROOT="$suite/inkling-depth-parallel-source-v1" \
OUT_DIR="$suite/inkling-depth-parallel-rebuild-v1" \
  bash "$suite/inkling-depth-parallel-source-v1/tests/wopr-llm/build-inkling-depth-parallel.sh"
```

All 20 build-source files match archive `ad582de61acdaa6b6c28fd0fa0039ca5de7461202dab1867d1e66c0be715d487`. A different compiler need not reproduce identical binary bytes.

## What the second round changed

| Attempt | Measured generation result | Decision |
|---|---|---|
| Parallel depth mixing, fixed cache | 24.872 → 24.199 s; **2.71% less time** | Included |
| Segmented cache, original depth mixing | 24.853 → 22.596 s; **9.08% less time** | Included |
| Both selected changes | **22.028 and 21.952 s**, versus 22.596 s for the segmented-cache-only control | Measured directly; do not add the separate percentage gains |
| Fused expert down-projection and accumulation | 24.846 → 24.847 s | Rejected: no meaningful gain |
| Overlapping uploads, without an expert cache | 28.463 → 27.755 s | Small gain against its uncached control, still slower than the selected cached path |
| Variable-size cache with ordinary LRU | Exploratory 45.615 s at the slower link state | No meaningful gain established |

The individual depth, segmented-cache, fusion and overlap comparisons used four fresh graphs in baseline/candidate/candidate/baseline order, with all sampled timed phases at 8 GT/s ×8 in automatic mode. The combined test used candidate/control/candidate runs with the same sampled link state. These small samples establish observations for this workload, not statistical confidence across prompts. Combined runs reached a sampled DRM allocation peak of **3.215 GB**, and tracked allocations returned to zero.

The selected combination passed the original two-token 98-array comparison, the longer 80-token comparison, and all 17 lifecycle checks. Timing runs separately checked the same 33 greedy IDs; they do not replace the full-array checks. The Mycroft speed target remains unmet.

## What may still limit speed

During active work, the actual PCIe link reported **2.5 GT/s ×8**, despite idle samples showing 8 GT/s. An independently audited upload benchmark measured **1.42–1.43 GB/s** from pinned host buffers and **1.32–1.38 GB/s** from pageable buffers for transfers of 4.48–51.76 MB. All 20 benchmark cases passed full-byte/guard checks, with zero remaining device allocations. All 80 busy-positive link samples reported 2.5 GT/s ×8.

This supports a transfer bottleneck hypothesis. It does not prove that all unaccounted model time is PCIe transfer: upload synchronization can also wait for earlier GPU work, and event intervals can include host submission gaps. See [the benchmark scope](h2d/README.md).

## Temporary high-mode comparison: faster uploads, inconsistent model speed

The user opened the reviewed, bounded `high` driver window. The same upload executable/runtime passed all 20 correctness cases in both modes. For large pinned transfers, throughput rose from **1.42–1.43 to 5.61–5.68 GB/s**, approximately **3.94–3.97×**. Pageable transfers rose from **1.32–1.38 to 4.24–4.89 GB/s**. All 22 busy-positive high-mode upload samples showed 8 GT/s ×8; the auto-mode upload samples showed 2.5 GT/s ×8. These are idle-queue transfer measurements, not model throughput.

Matched model runs within that high-mode window have differed substantially:

| Run | Generation time | Generation rate | Current interpretation |
|---|---:|---:|---|
| High 1 | 45.294 s | 0.706 tokens/s | Slow result; model observations included 2.5 GT/s despite `high` |
| High 2 | 24.731 s | 1.294 tokens/s | Faster result; link-phase association is separate evidence, not proof of a reliable setting |
| High 3 | 45.355678 s | 0.705535 tokens/s | Audited result; observed model link again at 2.5 GT/s despite `high` |

The generated IDs and work counters matched across all three runs, with memory checks passing. The third run also measured 24.922278 s for prompt processing; its final model hash was retained. Do **not** average these heterogeneous runs or describe `high` as a reliable fix. The setting has not reliably held the faster PCIe state during model execution, and it can also affect core/memory clocks. The faster model result remains below Mycroft's **1.844 tokens/s** target; that goal has not been met.

No permanent setting was changed. **Restoration to `auto` was verified** in `inkling-high-window-close-v1.json`: GPU busy 0%, no KFD compute clients, and an idle link of 8 GT/s ×8. The final model rehash after the third run remained `f90c…`. The bounded experiment is complete. One faster run and two slower runs do not establish a reliable high-mode fix; WOPR has not reached the Mycroft target.

Private retained evidence: `reference/wopr-gpu-suite-v1/inkling-performance-v1/`, including `inkling-h2d-auto-v1-independent-audit.json`. Raw hardware results and private host evidence are not part of a public-source portability claim.

Second-round private receipts: `inkling-optimization-round2/{depth-abba-independent-audit-v1,cache-abba-independent-audit,down-abba-independent-audit-v1,overlap-abba-aristotle-audit,combined-benchmark-independent-audit-v2,combined-sequence2-independent-audit,combined-sequence80-independent-audit,combined-lifecycle-independent-audit,selected-depth-source-closure-audit}.json`.

Final receipts: `segmented-selected-cli-aristotle-audit.json` and `inkling-round2-close-v1.json`. The normal CLI does not expose an allocator-live counter; zero tracked allocations are established by the separate lifecycle and benchmark checks, not inferred from CLI exit.
