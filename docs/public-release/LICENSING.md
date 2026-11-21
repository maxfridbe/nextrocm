# Licensing inventory

This inventory reports existing declarations; it does not relicense code or create a new copyright grant.

- The root Cargo workspace declares `MIT` as its default package license. A root copyright/license grant text was not present in the development snapshot. This export preserves package declarations without inventing a root copyright holder or adding a blanket grant. Consult the applicable package and component terms; absence of a root grant is not an MIT-only license for the entire tree.
- `cli/nextrocm` declares `GPL-3.0-only` because it links the RigNet engine. `model3d/rignet` also declares `GPL-3.0-only`; its license text and the kernel-side RigNet license are retained. A combined executable is not MIT-only.
- TypeScript, Snabbdom and PDF.js retain their vendor license files. PDF.js font and CMap notices are separate and retained.
- Source-derived numerical helpers retain the SLEEF/Boost and PyTorch notices alongside the corresponding source. Test-only GGML oracle code retains its upstream notice.
- Model checkpoint licenses are separate. No checkpoint, downloaded weight slice or generated reference tensor is included in the public candidate.

The build depends on other Rust packages through Cargo.lock. Their own licenses and notices still apply; this source inventory is not a complete binary-distribution notice bundle. Do not remove copyright headers or replace third-party notices with a blanket workspace license.

## Recovered notices and public font replacement

- The public export replaces the development snapshot's unprovenanced `nf.woff2` with the official Nerd Fonts v3.4.0 `SymbolsNerdFontMono-Regular.ttf`. Its exact same-directory MIT notice is retained in `notices/Nerd-Fonts-SymbolsOnly-LICENSE.txt`. `notices/font-provenance.json` records the source and hashes. All 32 icon codepoints used by the UI are present; this does not assert identical glyph outlines. The original font is not distributed in the public export.
- All five vendored rpcgen Rust/manifests match the authoritative Git blobs at revision `4ce40a4`. Its exact upstream MIT license, including the original copyright attribution, is retained in `notices/rpcgen-LICENSE.txt`; `notices/rpcgen-provenance.json` records the source matches. This notice applies to rpcgen, not the entire nextrocm repository.
- The retained GGML MIT notice in `notices/llama.cpp-LICENSE.txt` accompanies the source-derived quantization work. A source notice does not replace a complete dependency-notice review for binary distribution.

## Qwen Image materials

The pinned Qwen Image 2.1 Turbo model identifies its license as `qwen-research`. Its upstream agreement limits the model materials to research/evaluation use without a separate commercial license. The model card therefore records `commercial: false`. The exact pinned agreement and attribution are retained in `notices/Qwen-Image-2.1-Turbo-LICENSE.txt` and `notices/Qwen-Image-2.1-Turbo-NOTICE.txt` for copied upstream model materials/configurations.

This is a component-specific notice, not a blanket license assignment to the independently authored engine or repository. The scope and compatibility of any derived materials still need review before a redistribution-ready release is claimed. Default workspace MIT declarations do not override upstream terms.
