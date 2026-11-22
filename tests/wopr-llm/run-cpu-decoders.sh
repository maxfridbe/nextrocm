#!/usr/bin/env bash
# CPU-only original decoder/geometry tests; never loads HIP or model weights.
set -euo pipefail
: "${SOURCE_ROOT:?absolute published source root}"
: "${OUT_DIR:?new absolute directory on /var/home}"
[[ "$SOURCE_ROOT" == /* && "$OUT_DIR" == /var/home/* && ! -e "$OUT_DIR" ]]
cd "$SOURCE_ROOT"
cxx=${CXX:-/usr/bin/clang++}
[[ "$cxx" == /* && -x "$cxx" ]]
mkdir -p "$OUT_DIR"
"$cxx" --version > "$OUT_DIR/compiler.txt"
sources=(kernels/llm/stablelm/quants_test.cpp kernels/llm/stablelm/quants.hpp
 kernels/llm/qwen35/quants_test.cpp kernels/llm/qwen35/quants.hpp
 kernels/llm/inkling2/quant_numeric.cpp kernels/llm/inkling2/quant.hpp
 reference/inkling2-mini/nvfp4-decoder-v1/quant_upstream_oracle.hpp
 reference/inkling2-mini/nvfp4-decoder-v1/upstream_oracle.hpp
 kernels/llm/inkling2/state_guards.cpp kernels/llm/inkling2/state.hpp)
sha256sum "${sources[@]}" > "$OUT_DIR/source-before.sha256"
printf 'case\tcompile_exit\trun_exit\n' > "$OUT_DIR/results.tsv"
failed=0
for name in stablelm qwen35 inkling-quant inkling-state; do
  args=()
  case "$name" in
    stablelm) src=kernels/llm/stablelm/quants_test.cpp;;
    qwen35) src=kernels/llm/qwen35/quants_test.cpp;;
    inkling-quant) src=kernels/llm/inkling2/quant_numeric.cpp; args=(--self-test);;
    inkling-state) src=kernels/llm/inkling2/state_guards.cpp;;
  esac
  cmd=("$cxx" -std=c++17 -O2 -ffp-contract=off -fno-fast-math "$src" -o "$OUT_DIR/$name")
  printf '%q ' "${cmd[@]}" > "$OUT_DIR/$name.command"
  printf '\n' >> "$OUT_DIR/$name.command"
  compile=0
  /usr/bin/timeout --kill-after=5 120 "${cmd[@]}" > "$OUT_DIR/$name.compile.log" 2>&1 || compile=$?
  run=NOT_RUN
  if [[ "$compile" == 0 ]]; then
    sha256sum "$OUT_DIR/$name" > "$OUT_DIR/$name.exe.sha256"
    run=0
    /usr/bin/timeout --kill-after=5 60 "$OUT_DIR/$name" "${args[@]}" > "$OUT_DIR/$name.run.log" 2>&1 || run=$?
  fi
  printf '%s\t%s\t%s\n' "$name" "$compile" "$run" >> "$OUT_DIR/results.tsv"
  if [[ "$compile" != 0 || "$run" != 0 ]]; then failed=1; fi
done
sha256sum "${sources[@]}" > "$OUT_DIR/source-after.sha256"
cmp "$OUT_DIR/source-before.sha256" "$OUT_DIR/source-after.sha256" || failed=1
printf 'cpu_only=true\ngpu_executed=false\naggregate_exit=%s\n' "$failed" > "$OUT_DIR/scope.txt"
exit "$failed"
