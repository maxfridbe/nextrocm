# Inkling 2 Mini on Wopr (RX 550, 4 GiB)

This is an explicit test configuration for AMD gfx803 / wave64 with HIP 5.7 in `vibe-deb`. It does not install a runtime, change a model card or registry, or deploy a service. Use the dedicated runtime below; the previously published wave32-only Inkling runtime is not a substitute.

## Validated scope

- All 66 quantized operator cases passed the original full-output CPU wave-order and F64 gates, including tail rows and invalid-call checks.
- All 114 primitive, 19 state, and 54 mixing cases passed their unchanged CPU-reference gates. Tracked allocations returned to zero.
- The two-token chronological model run passed all 98 comparisons against the frozen full-sequence HF F32 reference, including both greedy token choices.
- The 80-token regression passed all 3,920 saved-array comparisons at the unchanged 0.001 relative-L2 gate and all 63 greedy decisions. Worst relative L2 was 5.49e-6; sampled DRM allocation was 2.194 GB, with zero tracked allocations remaining after cleanup. This is regression against the validated native baseline, not a new 80-token HF oracle.
- A streaming-adapter greeting run completed with `Hello!` and EOS: load 1.75 s, 18 prompt tokens in 39.17 s, and 63 generated tokens including reasoning/control/EOS in 135.03 s (0.467 tokens/s). This is one default-effort greeting, not content-only throughput. Tracked peak was 2.033 GB; the completed watchdog measured a 2.36420 GB DRM peak and verified an empty process group. Cleanup returned to zero. All 63 token IDs, EOS, final content, and reasoning matched the baseline.
- Lifecycle validation passed all 17 checks, including 400,116 finite output values matching the validated outputs byte-for-byte. Sampled DRM allocation peaked at 2.16247 GB and tracked allocations returned to zero.
- The normal CLI command below completed successfully: `Hello!` on stdout, with reasoning still on stderr despite `--effort none`. Prefill processed 16 tokens in 34.19 s; generation produced 46 tokens in 98.38 s (0.47 tokens/s). Combined prefill and generation took **2 min 12.57 s**, excluding load. Sampled DRM peaked at **2.36420 GB**; the process exited successfully and its process group was empty. The option requests no reasoning; it does not guarantee its omission. No sustained throughput, long-context quality, or production-deployment claim follows from these checks.

## Build the test runtime

Run inside `vibe-deb`. The build uses a new output directory and never replaces an installed runtime. Use a fresh directory if the example output already exists.

```bash
export SOURCE_ROOT=/var/home/maxfridbe/nextrocm-gpu-suite-v1/source
export OUT_DIR=/var/home/maxfridbe/nextrocm-gpu-suite-v1/inkling-logical32-build-local
bash "$SOURCE_ROOT/tests/wopr-llm/build-inkling-logical32.sh"
```

The script compiles gfx803 objects and emits the shared library, compiler identity, source hashes, LLVM IR, and assembly. It replaces the quantization translation unit with the logical-32 dispatch; do not link both quant runtime implementations together. Wave64 uses two independent 32-lane reductions per hardware wave. Wave32 retains the existing guarded implementation.

The tested runtime SHA-256 is `c28295d07a45d9cbb6d985095fe3673e632f9174522de762a1c88a9dbcb60fb3`. A rebuild has its own identity and needs validation; a matching source tree alone is not a numerical result.

## Explicit CLI configuration

The suite CLI was built with `--expert-gib` support and its two targeted CPU tests passed. Its SHA-256 is `d6d5036ba0f59d8e4612c32696db34f243a01d6028edd3bc6d02816eb65a301f`. The command below completed successfully. Suite tests use a watchdog with an 8 GiB host-available reserve and a 3.75 GiB sampled DRM allocation cap; these bounds remain part of the reported test conditions.

```bash
suite=/var/home/maxfridbe/nextrocm-gpu-suite-v1
model=/var/home/maxfridbe/inkling2-reference-v1/assets/Inkling-2-Mini-UD-Q4_K_M.gguf
NEXTROCM_LIB="$suite/inkling-logical32-build-v1/libnextrocm-inkling-logical32.so" \
NS_LIB_FROM_DISK=1 \
"$suite/bin/nextrocm-inkling-wave64-v1" llm generate "$model" \
  --prompt 'Say hello.' --max 64 --ctx 512 --gpu 0 \
  --temp 0 --top-p 1 --seed 1 --expert-gib 0 --effort none
```

The verified GGUF is 16,610,543,328 bytes, SHA-256 `f90c5a4df44d335ac9f9eea2a15c31d6c5a377b1b3cd9ff7fc2cf123f3cdf7e1`. Verify the staged file before testing; do not replace it with another quantization under the same filename. Explicit `NEXTROCM_LIB` selects this private runtime instead of relying on the CLI's embedded library.

## What uses RAM and GPU memory

`--expert-gib 0` selects streaming, not the default 1 GiB device expert cache. Selected expert pairs are read from the disk-backed GGUF and staged to the GPU. The operating system can keep frequently accessed file pages in host RAM; this is reclaimable page cache, **not a pinned full-model mirror**. This command does not create a new 16.6 GB host copy or lock the whole model in RAM. Keep the asset on disk-backed storage, not `/tmp` tmpfs.

For context 512 with chronological one-token prefill, the source planner estimates 2,049,570,912 bytes of tracked device allocations and requires 3,123,312,736 bytes free after including its unchanged 1 GiB driver margin. These are planning figures, not measured request peaks. Streaming also retains the existing 16 MiB host staging buffer. Actual free memory and the external watchdog still decide whether a run is admissible.

The default 1 GiB cache is deliberately not used here: its context-512 free-memory requirement is about 4.19 GB, and context 4096 requires about 4.59 GB. Do not weaken those guards to fit a 4 GiB card. Host RAM availability cannot replace required GPU allocations. Report sampled DRM VRAM+GTT and host memory separately; neither a reservation nor a planner estimate is an observed peak.

The greeting report recorded zero `/proc/self/io` `read_bytes` during load, prefill, and generation, while still issuing file reads. This supports page-cache-served I/O for that run; it does not prove that all weights stay resident. Full-model hashing before load can warm the cache. The default-effort adapter run and the normal CLI command above use different formatted inputs and token counts; their timings are separate observations.
