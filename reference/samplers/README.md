# Sampler regression fixture

`expected.txt` is copied byte-for-byte from nextsycl commit
`d6ed19267284c7a52b49ee9b60fbad99cdad6de7`,
`reference/samplers/expected.txt`. The same bytes are present in nextcuda
commit `9bee2b4d97e0df5a536f6a9325aa98953c0956e9`.

SHA-256: `f09aa4e62667a20934726cac1e9644c9afa5ce19ce7d266e8dea7c535bf626df`.

The upstream `reference/samplers/comfy_ref.py` describes generation using
ComfyUI's own sampler/scheduler code, a small analytic denoiser, and fixed
noise. This port restores the missing fixture required by the existing
`crates/diffusion` tests. It does not regenerate expected values from the
implementation being tested or change comparison tolerances. These are CPU
sampler tests, not GPU model validation.
