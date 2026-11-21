# Public source preparation

Use a clean source export, never a blanket first commit of the development workspace. Private reference runs, weight slices, generated images, host inventories, service registries, credentials and binaries are deliberately not part of the public source candidate. Originals remain local.

`stage.py --out /absolute/new-directory --audit /absolute/audit-directory` accepts repeatable `--private-name NAME` options for local host labels and produces an allowlisted candidate and separate SHA-256 file inventory, exclusions, path redactions, credential-pattern findings and license inventory. An empty pattern scan is not a guarantee that arbitrary future files are safe. Review the exact export before publishing; the script does not publish, push or initialize a remote.

## Complete native runtime

`build-runtime.sh` preserves the existing per-translation-unit compiler flags and adds the tested Qwen Image and Inkling families. It is restricted to the validated `gfx1035` target. Run it from the source tree as shown in the root README. This recipe builds a new output directory and does not modify a running service.

Build the web front ends, then build the CLI with `NS_LIB_DIST` pointing to that new runtime directory. Confirm the generated `libs_embedded.rs` names that directory. A distribution check must confirm that the executable embeds the intended runtime bytes and that a default-path worker loads the same library. Neither `NEXTROCM_LIB` nor `NS_LIB_FROM_DISK` should be set during that check. Export-symbol preservation is useful but does not replace execution tests.

A CPU-only source check can run without a built HIP library:

```sh
cargo check --locked --workspace --all-targets
```

It verifies Rust source integration, not inference. GPU tests need matching models, hardware and independently pinned reference fixtures. Those large private development captures are intentionally absent from the public candidate; test tools requiring them are not self-contained benchmarks.

## Publication boundary

Do not copy the private `.git` directory, service configuration, home directories, `.env` files, model checkpoints, raw execution logs or reference/evidence folders into the release. Keep the candidate's file inventory and scan results outside the release tree. Retain third-party notices, including notices embedded in vendored source. The licensing inventory distinguishes existing package declarations, recovered component notices and model-specific restrictions; it does not add a blanket repository grant.

The exporter keeps tiny compile-time JSON and synthetic test fixtures, small CPU oracle headers and their notices. It excludes large numerical captures. Some standalone GPU validators therefore still need separately supplied fixtures; their source headers are included so missing private data is not confused with missing build source. The `tiny-state.pth` fixture is a small synthetic serialization test, not a model checkpoint.
