#!/usr/bin/env bash
# Web frontends: strict TypeScript/TSX compiled offline with vendored tooling.
# The home, image, audio and model3d apps share vendor/ and static/ assets.
# Node is a build dependency only; the Rust services serve dist/wfe/.
# kind's server serves as they are (/ui/..., the page at /).
#   ./build.sh [app...]    type-check + compile (default: every app)
#   ./build.sh --check     also a page's smoke test against a running server (NS_WFE=http://host:port: an image or audio one)
# Only the four frontends present in this port are built by default.
set -euo pipefail
cd "$(dirname "$0")"
NODE="${NODE:-$(command -v node || echo "$HOME/.local/bin/node")}"
[ -x "$NODE" ] || { echo "need node on PATH or at \$NODE (only the build needs it)"; exit 1; }
TSC="vendor/typescript/tsc.js"
OUT=../dist/wfe
check=0; apps=()
for a in "$@"; do [ "$a" = "--check" ] && check=1 || apps+=("$a"); done
[ ${#apps[@]} -eq 0 ] && apps=(home image audio model3d)
mkdir -p "$OUT/vendor" "$OUT/static"
rm -rf build
for app in "${apps[@]}"; do
  echo "==> $app: tsc (strict)"
  "$NODE" "$TSC" -p "$app/tsconfig.json"
  rm -rf "$OUT/$app" && mkdir -p "$OUT/$app"
  cp -r "build/$app/src" "$OUT/$app/"
  cp "$app/index.html" "$app/style.css" "$OUT/$app/"
done
cp -r vendor/snabbdom "$OUT/vendor/"
cp -r vendor/pdfjs "$OUT/vendor/"
cp static/* "$OUT/static/"
rm -rf build
if [ "$check" = 1 ]; then
  echo "==> smoke test (every served module parses, the page renders headlessly)"
  "$NODE" check.mjs
fi
echo "==> done: $(find "$OUT" -name '*.js' | wc -l) modules, $(du -sh "$OUT" | cut -f1) in dist/wfe"
