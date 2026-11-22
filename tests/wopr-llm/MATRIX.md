# WOPR LLM test queue

Source: `cbc7e320c423f9e20dc3c3b1bd3ea291008c8d25`. Inventory hashes and model-card asset identities are in `inventory.json`. No GPU test has been executed by this silo. Parent owns remote execution and serial GPU scheduling.

Confirmed hardware (parent receipt): RX550, gfx803, wave64, 4 GiB; HIP 5.7.31921 / clang17 in vibe-deb. Build on `/var/home`, not `/tmp` (tmpfs). Never use HSA_OVERRIDE_GFX_VERSION. Native test runtime is a new isolated artifact, never the deployed DSO. The public full-runtime recipe intentionally accepts only gfx1035; `build-runtime.sh` here is a compile-only, LLM-only gfx803 recipe preserving each included TU's existing flags.

## Mandatory execution envelope

Before each GPU command: verify model bytes/SHA against the card, verify executable/runtime/source identities, record actual free GPU memory and host MemAvailable, and use the parent's direct-child DRM/RSS sampler and serialized maintenance wrapper. A timeout must drain the owned GPU process before the next run. Retain stdout/stderr, exit status, allocation cleanup, measured memory and original comparison gates. The examples' general 8 GB check does NOT establish fit on this 4 GiB card. Missing assets/reference data are BLOCKED, unsupported wave dispatch is UNSUPPORTED, comparison failures are FAIL; none count as PASS.

Do not stage all models at once: assets range from 1.7 GB to 21.2 GB and the host has only about 43 GB disk free. Locate and hash existing read-only WOPR assets first. Sparse tokenizer GGUF files are not model assets and must never be used for inference.

## Queue and commands

Commands below run inside vibe-deb from the staged source. Set CARGO_TARGET_DIR to a new `/var/home/.../target`; set NEXTROCM_LIB to the new absolute runtime. Variables MODEL/REF/FIXTURE/CONFIG denote validated absolute inputs, not guessed paths. Timeout values are upper bounds for an initial test, not predicted runtimes.

| Family / gate | Command | Prerequisite and interpretation |
|---|---|---|
| All four host libraries | `timeout 900 cargo test --locked --offline -p nextrocm-llm-stablelm -p nextrocm-llm-qwen3moe -p nextrocm-llm-qwen35 -p nextrocm-llm-inkling2 --lib` | CPU only; preserve complete output. |
| All examples compile | `timeout 1200 cargo check --locked --offline -p nextrocm-llm-stablelm -p nextrocm-llm-qwen3moe -p nextrocm-llm-inkling2 -p nextrocm --examples` | Compile only, not hardware proof; dependency availability may block. |
| StableLM quant decoder | `c++ -std=c++17 -O2 kernels/llm/stablelm/quants_test.cpp -o "$HOST_OUT/stablelm-quants"` then `timeout 30 "$HOST_OUT/stablelm-quants"` | CPU packing/decoding only. |
| Qwen35 quant decoder | `c++ -std=c++17 -O2 kernels/llm/qwen35/quants_test.cpp -o "$HOST_OUT/qwen35-quants"` then `timeout 30 "$HOST_OUT/qwen35-quants"` | CPU only. |
| Inkling quant decoder | `c++ -std=c++17 -O2 -ffp-contract=off kernels/llm/inkling2/quant_numeric.cpp -o "$HOST_OUT/inkling-quants"` then `timeout 60 "$HOST_OUT/inkling-quants" --self-test` | Includes retained upstream oracle header; CPU only. |
| StableLM tokenizer | `timeout 120 "$STABLE_BIN/tokenizer_parity" "$FIXTURE"` | Independent tokenizer fixture directory. |
| Qwen3moe tokenizer | `timeout 120 "$MOE_BIN/tokenizer_parity" "$MODEL" "$FIXTURE"` | Model plus source-tokenized JSON. |
| StableLM three model variants | `timeout 600 "$STABLE_BIN/numeric_reference" "$MODEL" "$REF"` | Separate Zephyr, Code, Code-Instruct assets and matching original taps; defaults abs .05 / rel .01 unchanged. Short context first. |
| StableLM lifecycle | `timeout 600 "$STABLE_BIN/lifecycle" "$MODEL"` | Original load reserve4160 tokens once, sessions4096+64; roughly1.363 GB KV +1.709 GB model plus scratch/margin. Actual free-memory guard decides fit. No duplicate 4160 reservation assumption. |
| Qwen3moe numeric | `timeout 900 "$MOE_BIN/numeric_reference" "$MODEL" "$REF"` | 18.56 GB asset streamed experts, original taps/gates unchanged. |
| Qwen3moe lifecycle | `timeout 900 "$MOE_BIN/lifecycle" "$MODEL"` | Original4096+64 sessions; ~98304 KV bytes/token, dense resident ~1 GB; validate actual host/device before run. |
| Qwen35 / Qwen36 numeric | `timeout 900 "$CLI_BIN/qwen36_numeric" "$MODEL" "$REF"` | Run dense4B and MoE35B separately, corresponding original taps; defaults abs .05 / rel .01 unchanged. |
| Qwen35 / Qwen36 lifecycle | `timeout 900 "$CLI_BIN/qwen36_lifecycle" "$MODEL"` | Original4160 reserve/4096+64 sessions; dense model fit is tight and must be checked. |
| Qwen36 MoE scalar variants | `NS_PROFILE=gpu timeout 900 "$CLI_BIN/qwen36_scalar_check" "$MODEL" --gpu 0 --reps 1` | Requires qwen35moe asset; dense4B rejected. NS_PROFILE=gpu is mandatory. Shuffle falls back to original shared256 tree on wave64. |
| Qwen36 MoE batch variants | `timeout 900 "$CLI_BIN/qwen36_batch_check" "$MODEL" --gpu 0 --reps 1` | Requires qwen35moe asset; dense4B rejected. Only four individual matrices, not full model. Do not infer from scalar PASS. |
| Qwen35 batch original taps | `timeout 900 "$CLI_BIN/qwen36_batch_numeric" "$MODEL" "$REF" "$CHUNK" 0` | CHUNK1,8,32,64 queued individually only after bounds/preflight; original .05/.01 gates; no expert cache for this control. |
| Inkling metadata | `timeout 120 "$INK_BIN/inspect_model" "$MODEL" "$CONFIG" 4096 1 "$ACTUAL_FREE_BYTES" "$NEW_REPORT"` | CPU strict734 tensors / 48 visits / checked estimates; not fit proof. |
| Inkling embedding | `timeout 120 "$INK_BIN/inspect_embedding" "$MODEL" "$CONFIG" 0 0 "$NEW_OUT"` | Decode + RMS do not use wave32 matvec. Requires independent original Q8/RMS full2304-value comparison; capture alone is not numeric PASS. |
| Inkling full graph/lifecycle | NOT QUEUED on published dispatch | quant_wave32 requires hardware wave32 and explicitly rejects gfx803. Source scalar quant.hip could form a separate reviewed fallback candidate, never silently replace dispatch. |

Build Rust executables separately by package/example into distinct output paths: StableLM and Qwen3moe both name `numeric_reference` and `lifecycle`, so a shared target examples directory can overwrite them. Pin or copy each binary immediately after its package build before building the next package. cargo check does not produce runnable binaries.

## Inkling diagnostic limits

`inspect_sequence` accepts `GGUF CONFIG GPU NEW_OUT tokens 0,1 0` and uses explicit Stream TextGraph loading; it can be reused only after a separately approved wave64-compatible runtime gate. `inspect_layer0`, `inspect_adapter`, `inspect_service_hooks`, and `inspect_reply` provide the prior progression but require original captured/oracle inputs and runtime-compatible bindings. `utilization_bench` hardpins gfx1035 runtime994f and wave profile examples bind prior runtime receipts; unchanged execution with a rebuilt gfx803 runtime must reject. Do not edit those pins to manufacture parity. Current adapter default1GiB cache context4096 estimates4.594 GB including driver margin, over4GiB; explicit Stream is a separately reported policy, not automatic admission fallback.

Historical `vibe_lowvram` WOPR Vulkan/CPU-offload operation is not evidence that these native HIP ports pass. gfx803 compilation, operator correctness, full numeric gates, lifecycle and measured fit must each be established independently.

## Existing independent StableLM reference location

The original private tooling note `tooling/stablelm-numeric-reference.md` records WOPR `/tmp/nextrocm-stablelm-numeric/output`: 42 files plus manifest, the corrected 20-ID Zephyr prompt, greedy510. The older23-ID whitespace variant is superseded. Current local copy is absent; parent must verify remote availability, manifest hashes, shapes and the actual74b2613b model identity before queueing. This is an existing HF-source/CPU-dequantized oracle, not candidate-produced data. Code and Code-Instruct have retained result summaries (181/581 comparisons), but those summaries are not substitute full reference arrays; their actual tap directories remain unlocated.

While references are unavailable, the bounded original CLI smoke is `timeout 180 "$CLI" llm generate "$MODEL" --prompt 'Say hello briefly.' --max 16 --ctx 512 --gpu 0 --temp 0`, followed by the original lifecycle example if memory preflight admits it. Output coherence and lifecycle PASS do not establish numerical parity.
