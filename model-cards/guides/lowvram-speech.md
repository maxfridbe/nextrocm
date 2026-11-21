# Native speech recognition

Use the existing audio transcription service or `nextrocm audio transcribe FILE --model ID --no-words`. These engines return text and timed segments; word timestamps and decoder prompts are unsupported. API requests must set `words` to false. Canary and English-only Whisper checkpoints accept English and do not translate. Multilingual Whisper accepts language codes/names or automatic detection and can translate into English. Whisper GGML load option `q8` optionally quantizes eligible weights; default is off.

Canary weights are CC BY 4.0: attribute NVIDIA Canary-Qwen 2.5B and the lipik GGUF conversion, retain license notices, and identify modifications. Whisper weights are MIT.
