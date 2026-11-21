# Qwen3.6 35B A3B Heretic

Use the existing chat page or OpenAI-compatible chat API with model `qwen3.6-35b-a3b-heretic-q4`. The engine applies the checkpoint's chat template. It thinks by default; use the existing reasoning effort setting `off` (or `none`) to request an answer without a thinking stream.

This port supports text conversations. The separate vision projector is not ported. The Q4_K_M file occupies 21.17 GB on disk; the native backend keeps packed non-expert weights resident and streams only selected experts in bounded tiles. It does not hold the full file in GPU or host memory. Selected weights are read into a reusable 16 MiB pinned host buffer; Linux may retain file pages in reclaimable system RAM. The VRAM limit remains below 8 GB.

Prompt processing groups up to 64 rows by layer and expert; decoding uses the validated specialized native scalar kernel. The card also selects bounded read-ahead for routed expert ranges and a 2 GiB pinned-RAM mirror of 1,126 profile-ranked experts. This improves measured PP/TG while loaded, but adds roughly 3–5 seconds of preload time; short cold requests can finish sooner with the mirror disabled. The profile is model/path-bound and kept at `/home/developer/.config/nextrocm/profiles/qwen36-experts.json` on development-host. Use the original engine options `host-mib=0` to disable the mirror or `prefetch-mib=0` to disable read-ahead. RAM admission preserves system headroom. Optional device expert caching remains disabled after its initial throughput trials.

The catalog uses a 4096-token context and a 256 MiB prompt cache. Model selection, enable/disable, loading, pinning and unloading use the original controls. Source and measured validation are recorded in `docs/qwen36-memory.md` and `docs/validation.md`.
