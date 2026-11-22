#!/usr/bin/env bash
# Compile only. No GPU launch, installation, or modification of old runtimes.
set -euo pipefail
: "${SOURCE_ROOT:?absolute source root}"
: "${OUT_DIR:?new disk-backed output directory}"
[[ "$SOURCE_ROOT" == /* && "$OUT_DIR" == /var/home/* && ! -e "$OUT_DIR" ]]
cd "$SOURCE_ROOT"
hipcc=${HIPCC:-/usr/bin/hipcc}
mkdir -p "$OUT_DIR/objects"
common=(-std=c++17 -O2 -fPIC --offload-arch=gfx803 -Ikernels/ns)
"$hipcc" --version > "$OUT_DIR/compiler.txt"
sha256sum kernels/ns/* kernels/llm/inkling2/* > "$OUT_DIR/source-before.sha256"
set -x
"$hipcc" "${common[@]}" -c kernels/ns/ns.hip -o "$OUT_DIR/objects/ns.o"
for name in quant_typed32_dispatch primitives state mix_depth_parallel; do
  flags=(-ffp-contract=off)
  case "$name" in
    quant_typed32_dispatch|mix_depth_parallel) flags+=(-fno-unsafe-math-optimizations -fno-approx-func);;
    primitives) flags+=(-fno-fast-math);;
  esac
  "$hipcc" "${common[@]}" "${flags[@]}" -c "kernels/llm/inkling2/$name.hip" -o "$OUT_DIR/objects/$name.o"
done
"$hipcc" "${common[@]}" -shared "$OUT_DIR/objects/ns.o" \
  "$OUT_DIR/objects/quant_typed32_dispatch.o" "$OUT_DIR/objects/primitives.o" \
  "$OUT_DIR/objects/state.o" "$OUT_DIR/objects/mix_depth_parallel.o" \
  -o "$OUT_DIR/libnextrocm-inkling-depth-parallel.so"
"$hipcc" "${common[@]}" -ffp-contract=off -fno-unsafe-math-optimizations -fno-approx-func \
  --cuda-device-only -S -emit-llvm kernels/llm/inkling2/mix_depth_parallel.hip -o "$OUT_DIR/depth.ll"
"$hipcc" "${common[@]}" -ffp-contract=off -fno-unsafe-math-optimizations -fno-approx-func \
  --cuda-device-only -S kernels/llm/inkling2/mix_depth_parallel.hip -o "$OUT_DIR/depth.s"
set +x
sha256sum kernels/ns/* kernels/llm/inkling2/* > "$OUT_DIR/source-after.sha256"
cmp "$OUT_DIR/source-before.sha256" "$OUT_DIR/source-after.sha256"
sha256sum "$OUT_DIR/objects/"*.o "$OUT_DIR/libnextrocm-inkling-depth-parallel.so" \
  "$OUT_DIR/depth.ll" "$OUT_DIR/depth.s" > "$OUT_DIR/build.sha256"
