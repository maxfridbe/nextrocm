# Inkling wave64 scalar fallback: explicit candidate

This candidate changes only hardware dispatch. The original wave32-only TU,
scalar implementation, decoders, and other model kernels remain unchanged.
`quant_portable_dispatch.hip` replaces the quant TU in a separate runtime:

- Actual hardware wave32: delegate to the existing wave32 implementation.
- Actual hardware wave64: delegate to the existing scalar ascending-F32-FMA implementation.
- Null context, device-query failure, or any other wave size: return an error.

Both implementations retain their own shape, span, alignment, alias, launch,
and postscale checks. The ABI and decode entry point are unchanged. Scalar and
wave reductions are not presumed bitwise equal. No speed or model-support
claim follows from this source change alone.

## Parent-owned build and validation

Use a new disk-backed directory and the isolated five-TU recipe:

```sh
SOURCE_ROOT=/absolute/source OUT_DIR=/var/home/USER/new-inkling-build \
  bash tests/wopr-llm/build-inkling-portable.sh
```

The recipe selects gfx803 without a target override, preserves the prior
per-TU floating-point flags, records source and object identities, and emits
quant IR/ISA for review. It does not launch GPU work or install a runtime.

First run the unchanged scalar quant validator under the parent's serialized
watchdog. Its 40 original CPU fixtures cover all five types, complete decoded
rows, scalar FMA bits, F64 bounds, postscale, unaligned packed rows, nonunit
float strides, padding, repeats, 38 invalid/no-write cases, and tracked drop.
The pinned fixture manifest is
`a4df19ab389879dc3c53f5fc20d38d1ca049098004080a9f93c6565347b32f0e`.
The unchanged validator SHA is
`7906439c9d2ebad33ef25487e64afddb12bcc1546108b12fe1222e355db4ea60`.

```sh
python3 validate_quant.py --runtime /absolute/new-runtime.so \
  --runtime-sha ACTUAL_DSO_SHA --fixtures /absolute/original-fixtures \
  --manifest-sha a4df19ab389879dc3c53f5fc20d38d1ca049098004080a9f93c6565347b32f0e \
  --out /absolute/new-operator-result
```

This operator gate does not replace the original full HF sequence comparisons
(all 98 arrays at the unchanged .001 gate plus expected greedy IDs), nor the
actual 4GiB memory/admission checks. Host work separately uses explicit Stream
mode and a bounded context. Do not silently enable the 1GiB device cache.

## Separate logical32 candidate

`quant_logical32_dispatch.hip` lets two logical 32-lane rows share one
hardware wave64 using width-32 shuffles. This separate candidate keeps
invalid-row lanes active through the reduction, then suppresses their stores.
It preserves the old wave32 implementation on physical wave32 hardware.
Use `build-inkling-logical32.sh` for a separate runtime; do not mix quant TUs.
This requires actual gfx803
shuffle lowering review and full original wave32-order CPU fixtures,
including odd-row tails, before model validation. Removing the existing
wave32 hardware guard alone is not an accepted portability change.

`validate_inkling_logical32.py` preserves the original 66-case operator gates.
Only the baseline/candidate ABI bindings and identification text differ from
the old validator. The baseline binds the scalar alias; the candidate binds
the public model entry point. The 26 actual matrices and 40 tail cases retain
full F64 checks for every output, exact original wave-order CPU bits, separate
scalar bits, full small-case decoder bytes, 19 invalid/no-write cases, five
alternating timing pairs, and tracked drop. The fixture manifest is
`3ec116114190c5384c62309b89060e68f0a9df1727e4b6f1aa914b936b94afe5`.
The original fixture producer and CPU references are unchanged. Saved output
files are expected bytes written only after full GPU-byte equality checks;
they are not independent raw captures. Neither candidate has numerical or
model admission merely from source review or successful compilation.
