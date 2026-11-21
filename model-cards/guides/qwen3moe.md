# Qwen3-Coder 30B A3B

Use the existing chat page or OpenAI-compatible API with model `qwen3-coder-30b-a3b-q4`. The original model controls enable, load, pin and unload it.

The native Rust/HIP engine keeps packed non-expert weights on the GPU and streams selected experts from the 18.56 GB checkpoint through a bounded buffer. It does not retain the whole checkpoint in host RAM. Disk access makes generation slower than full GPU residency. The context is 4096 tokens with a 256 MiB prompt cache.

The actual checkpoint template supports XML tool calls and has no thinking prefix. This is a text model. Tokenization preserves the official NFC normalization.
