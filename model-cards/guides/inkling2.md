# Inkling 2 Mini (text only)

Use the existing chat interface or `/v1/chat/completions` with model `inkling2-mini-ud-q4km`. Write ordinary text prompts; the model-specific chat template is applied automatically.

This entry supports a 4096-token context and one active session. It can return a separate reasoning stream before the final answer; allow enough output tokens for both. A short output limit may end during reasoning before an answer appears.

Images, audio, tool calls and token log probabilities are not supported by this entry. Do not include literal model control tokens in messages. Vision support in the upstream model does not imply vision support in this native text-only port.

The 16.61 GB quantized weight file is read through bounded host staging; it is not all resident on the GPU. The deployed policy uses a bounded 1 GiB packed-expert cache with pinned host staging; this card reserves 5 GiB of device memory. Host RAM and available memory still matter, particularly on a shared-memory GPU. Insufficient memory causes loading to fail rather than silently changing the cache policy.

For native API callers, `LoadOptions.expert_bytes = Some(0)` explicitly selects streamed experts; positive budgets up to 1 GiB must hold at least one complete expert pair. The current CLI does not expose this option, so there is no documented CLI fallback flag.

The cache upgrade passed default embedded-runtime private API validation and transactional live deployment. Streaming and nonstreaming text/reasoning outputs matched on the fixed test prompt, with reported sampled device allocation below 3.83 GB. The existing model settings were preserved apart from the 5 GiB reservation, and the worker unloaded with maintenance off after validation. These bounded checks do not establish general answer quality or throughput.

The model uses the Apache 2.0 license; see the pinned upstream license link in its model card.
