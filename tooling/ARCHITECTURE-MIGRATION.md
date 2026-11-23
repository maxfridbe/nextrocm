# Kernel-library migration status

This is infrastructure migration, not model porting or tuning. HIP target
profiles (`gfx803`/`gfx1035`) and model architecture libraries are separate
choices. No per-card library filename or invented tuned variant is introduced.

The manifest now declares 18 default architecture libraries. Identical source
and ordered compiler flags reuse one object across kind libraries, architecture
libraries and the compatibility runtime. Each architecture library has an
explicit export map, binds its own internal definitions with `-Bsymbolic`, and
must pass an exact post-link dynamic-symbol check. Shared device/context and
vision/speech utility operations remain in the runtime; no silo creates a
second device owner. Host bindings select a per-GPU silo and retain shared
fallback for missing operations.

| Architecture | Default source closure |
| --- | --- |
| stablelm | One StableLM TU, seven operations |
| qwen35 | Main, batch and scalar-fast TUs |
| qwen3moe | One Qwen3 MoE TU |
| inkling2 | Quant dispatcher, primitives, state and mix; explicit target selection below |
| rembg | Background-removal TU; shared vision utilities remain shared |
| upscale | Upscaling TU |
| sd15 | SD1.5 TU |
| starvector | StarVector TU with its existing compile define |
| rignet | RigNet TU |
| minimaxmusic3 | MiniMaxMusic3 TU |
| qwen3tts | Qwen3TTS TU; shared decoder utilities remain available |
| voxcpm2 | VoxCPM2 TU |
| whisper | Whisper private tiled operations; other shared operations retain fallback |
| lowvram | Ops, selftest, bridge, Canary, Whisper, Qwen, GGUF, BPE and WAV closure |
| qwen3vl | Eight explicitly selected encoder TUs with individual strict flags |
| qwenimage21 | Nine selected packed-linear, attention, trajectory and VAE TUs |
| paddleocrvl | Existing PaddleOCR TU, newly declared conservative compile recipe |
| vibevoice | Existing MiniMaxMusic3 and Qwen3TTS TUs, restricted to eleven VibeVoice operations |

The original 48-TU source and ordered-flag inventory remains recorded unchanged
in `units`. PaddleOCR is explicitly separate in `additional_units`: C++17,
`-O2`, PIC and the shared header include path, without fast-math. Its new
packaging is compile-only until numerical gates run; it is not represented as
part of the old recipe's validation.

For gfx803, `source_overrides` selects the previously validated Inkling
`quant_batch32.hip` and `mix_depth_parallel.hip` closure, retaining the original
strict flags. Its additional batch, typed/logical and baseline-depth exports
are declared only in that profile. This preserves the selected WOPR operation
paths rather than reverting to an unsupported physical-wave32 dispatcher.
For gfx1035, the original public dispatcher and mix closure remain selected;
this profile does **not** promise the newer optional batch ABI. Other unsupported
wave guards remain intact. Profiles are build choices, not blanket hardware
support or performance guarantees.

CPU closure checks cover all non-shared TUs, exact source/flag reuse, duplicate
architectures/TUs, target-specific selection and exports. Both targets have now
built successfully with HIP 5.7.31921 / Debian Clang 17.0.6: 49 compilations,
23 links and 18 exact export inventories each. Independent receipt inspection
matched all 236 source hashes before/after and all compiler/linker commands.
Compiler warnings about unsupported floating-point exceptions and existing
ignored low-VRAM error-path synchronization results remain recorded; no
floating-point exception support is claimed.

On gfx803, all 18 libraries passed dynamic loading, reported device properties,
a 64 KiB allocation/free with zero remaining tracked allocation, and the strict
watchdog. Core allocation/copy/bounds/arithmetic checks also passed. The first
properties-only probe lacked positive DRM telemetry; its incomplete watchdog
receipt is preserved separately from the successful allocation probe.

The gfx1035 build has **not been executed on matching hardware**. Required-silo
ESRGAN passed three native fixtures (RGB, RGBA edge and tiny input), plus two
live UI RGB/RGBA requests against the retained CPU reference. Maximum byte error
was 1; the RGBA alpha channel matched exactly and native allocations returned to
zero. Two live audio UI requests reproduced the retained 11-second JFK
transcript exactly, with plaintext/JSON downloads matching and 39 served audio
modules verified against built sources. These are compiled UI action/HTTP tests,
not browser visual-rendering checks or general recognition accuracy evidence.
All corresponding watchdogs completed cleanup; final device busy was zero and
no KFD users remained. Live server exit is not an independent allocator-drop
measurement. This is bounded packaging/core and fixed image/audio evidence,
not numerical admission for all architecture libraries. Existing model receipts
do not automatically validate newly linked binaries, and no performance
baseline is fabricated from these checks.

Validation proceeds with a compatible HIP compiler, exact export inspection,
then core allocation/copy smoke and unchanged engine numerical/reference and
lifecycle gates on the serialized GPU queue. Binary/device/driver/compiler and
memory identities must accompany results. Missing baselines remain unvalidated.
Hardware variants require a nonambiguous capability match, nonempty operation
delta and recorded matching-device baseline; no variants are fabricated here.
