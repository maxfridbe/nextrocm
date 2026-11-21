// Entry points for the two pipelines.
#pragma once

#include <string>
#include <vector>

struct SttOptions {
    std::string model, input;
    int max_new_tokens = 512;
    double chunk_seconds = 30.0;
    std::string language = "auto";  // Whisper only: ISO code or "auto"
    bool translate = false;         // Whisper only: translate to English
    bool q8 = false;                // Whisper only: 8-bit weights
};
struct SttSegment {
    double start, end;  // seconds
    std::string text;
};
struct SttResult {
    std::string text, language;
    std::vector<SttSegment> segments;  // one per transcribed piece (<= chunk_seconds each)
};
// Canary-Qwen (GGUF).
SttResult run_stt(const SttOptions &o);
// Whisper (whisper.cpp ggml-*.bin).
bool is_whisper_model(const std::string &path);
SttResult run_whisper(const SttOptions &o);

struct TtsOptions {
    std::string model, vocab, merges, script_path, out;
    std::vector<std::string> voices;
    std::string cache_dir;
    std::string stream_path;  // if set, raw s16le samples are appended here as each frame is made
    float cfg_scale = 1.3f;
    int steps = 10;
    unsigned long long seed = 0;  // 0 = random
    int max_ctx = 4096;
    bool q8 = false;  // 8-bit tokenizer / diffusion-head weights (faster, slightly lossy)
};
// Writes a 24 kHz WAV; returns the number of seconds generated.
double run_tts(const TtsOptions &o);

// Matmul micro-benchmark.
void run_bench();
// Tokenizer round-trip test (16-bit vs 8-bit weights).
void run_recon(const std::string &model, const std::string &voice, const std::string &out);
