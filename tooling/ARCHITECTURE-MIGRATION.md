# Kernel-library migration status

This is infrastructure migration, not model porting or tuning. Model architecture
names and HIP target profiles are different dimensions. `gfx803`/`gfx1035`
select code objects and wave-dependent build definitions; an engine architecture
selects its own library. No per-card silo filename or invented tuned variant is
introduced.

| Existing source family | Current packaging / next boundary |
| --- | --- |
| StableLM | Concrete default silo: unchanged `stablelm.hip`, seven explicitly exported `nslm_*` operations, per-GPU host binding with shared fallback. GPU regression admission remains outstanding. |
| Qwen3 MoE | One existing `qwen3moe.hip` translation unit; next migration needs an explicit export inventory and engine-owned binding before splitting it from the compatibility runtime. |
| Qwen3.5 | `qwen35.hip`, `batch.hip`, `scalar_fast.hip` form a closure; keep all three and their flags together. Audit host bindings and actual exported operations before a silo declaration. |
| Inkling2 | Quant dispatcher plus primitives/state/mix form a closure. Multiple historical diagnostic wrappers export the same public names: select one validated dispatcher explicitly, never glob all `.hip` files. Target capability dispatch and existing runtime receipts must remain explicit. |
| Background removal | `image/rembg/rb.hip` depends on the shared vision operations; `image/rembg/src/ffi.rs` currently binds the shared API. Split private removal operations only, retaining the shared vision dependency. |
| Upscaling / SD1.5 / StarVector / RigNet | Existing individual translation units are candidate boundaries, with their shared vision/speech utility dependencies retained. StarVector's current compile define is preserved, not inferred from a card name. Each host binding/export inventory is a separate migration. |
| MiniMaxMusic3 / Qwen3TTS / VoxCPM2 / Whisper | Individual family translation units plus shared speech utilities. Do not duplicate shared `ns_*` device/context ownership into a silo. |
| Low-VRAM audio | `ops.hip` plus bridge/canary/whisper/qwen/GGUF/BPE/WAV sources are one linked closure. Its wave-size compile definition differs by HIP target profile. Separate library work needs an explicit ABI boundary, not moving only `ops.hip`. |
| Qwen Image / Qwen3VL encoder | Numerous separately compiled strict arithmetic wrappers and attention/linear/VAE objects. Preserve per-TU flags and avoid duplicate standalone test-wrapper definitions. This closure requires a dedicated symbol/dependency inventory before architecture extraction. |

The new kind libraries and compatibility runtime currently retain the exact
48 translation units and ordered per-TU flags of the frozen public recipe.
The build manifest is deliberately explicit. Building kind libraries does not
mean every model has migrated to a silo, nor that every engine works on both
HIP target profiles. In particular, unsupported physical-wave dispatch guards
are not removed by packaging.

Next validation steps:

1. CPU: source/manifest/variant-rule tests, mock HIP properties, fake-DSO
   override/fallback tests, gate negative controls and bounded watchdog cleanup.
2. Compile: build each supported target with its compatible HIP toolchain;
   inspect symbols and strict arithmetic lowering where changed packaging
   could affect link resolution. Compare all published exports.
3. GPU, owned serialized queue only: core allocation/copy smoke, then the
   unchanged engine's numerical/reference and lifecycle gates against its
   original baseline. Record actual binary, device, driver/compiler and memory
   identities. Do not mark absent baselines as passing.
4. Admit a hardware variant only with a nonambiguous capability match, a
   nonempty operation delta, validated engine settings and a recorded matching
   hardware baseline. No memory-plan sizes belong in variant settings.
