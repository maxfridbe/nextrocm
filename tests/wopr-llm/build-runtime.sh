#!/usr/bin/env bash
# Compile-only isolated LLM runtime. No GPU launch or installation.
set -euo pipefail
: "${SOURCE_ROOT:?absolute published source root}"
: "${OUT_DIR:?new absolute output directory on /var/home, not tmpfs}"
[[ "$SOURCE_ROOT" == /* && "$OUT_DIR" == /var/home/* && ! -e "$OUT_DIR" ]]
cd "$SOURCE_ROOT"
hipcc=${HIPCC:-/usr/bin/hipcc}
mkdir -p "$OUT_DIR/objects"
common=(-std=c++17 -O2 -fPIC --offload-arch=gfx803 -Ikernels/ns)
"$hipcc" --version > "$OUT_DIR/compiler.txt"
for name in quant_wave32_dispatch primitives state mix; do
  flags=(-ffp-contract=off)
  case "$name" in
    quant_wave32_dispatch|mix) flags+=(-fno-unsafe-math-optimizations -fno-approx-func);;
    primitives) flags+=(-fno-fast-math);;
  esac
  "$hipcc" "${common[@]}" "${flags[@]}" -c "kernels/llm/inkling2/$name.hip" -o "$OUT_DIR/objects/$name.o"
done
# Wave32 dispatch is deliberately preserved: Inkling matvec must reject wave64.
# No scalar replacement or target spoof is hidden in this recipe.
"$hipcc" "${common[@]}" -shared -x hip \
  kernels/ns/ns.hip kernels/selftest.hip \
  kernels/llm/stablelm/stablelm.hip \
  kernels/llm/qwen3moe/qwen3moe.hip \
  kernels/llm/qwen35/qwen35.hip kernels/llm/qwen35/batch.hip \
  kernels/llm/qwen35/scalar_fast.hip -x none \
  "$OUT_DIR/objects/quant_wave32_dispatch.o" "$OUT_DIR/objects/primitives.o" \
  "$OUT_DIR/objects/state.o" "$OUT_DIR/objects/mix.o" \
  -o "$OUT_DIR/libnextrocm-runtime.so"
sha256sum "$OUT_DIR/libnextrocm-runtime.so" > "$OUT_DIR/runtime.sha256"
