# Qwen3.5 4B

Use the existing chat page or OpenAI-compatible chat API with model `qwen3.5-4b-q4km`. This checkpoint's embedded template disables thinking by default. The existing reasoning effort settings can request thinking explicitly; `off` or `none` keeps it disabled.

The native text engine uses the actual mixed Q4_K, Q5_K, Q6_K and Q8_0 weights without expanding the matrices. This port does not load the separate vision projector. The model uses a 4096-token context and a 256 MiB prompt cache. Enable/disable, loading, pinning and unloading use the original controls.
