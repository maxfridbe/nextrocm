// Qwen2 / Qwen3 style decoder-only transformer with a multi-sequence f16 KV cache.
#pragma once

#include <string>
#include <vector>

#include "gguf.h"
#include "ops.h"

struct LmConfig {
    int n_layer = 0, hidden = 0, n_head = 0, n_kv = 0, head_dim = 0, ffn = 0;
    float eps = 1e-6f, rope_theta = 1e6f;
    bool qk_norm = false;  // Qwen3: per-head RMSNorm on q and k
};

// printf-style tensor name patterns, %d = layer index
struct LmNames {
    std::string attn_norm, ffn_norm, q, k, v, o, q_bias, k_bias, v_bias, q_norm, k_norm, gate, up, down, out_norm;
};

struct LmLayer {
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    Weight q, k, v, o, gate, up, down;
};

class Lm {
public:
    LmConfig c;
    std::vector<LmLayer> layers;
    float *out_norm = nullptr;
    KvCache kv;

    void load(const Gguf &g, const LmConfig &cfg, const LmNames &n);
    void init_cache(int n_seq, int max_ctx);
    size_t weight_bytes() const;
    void free();

    // x: [M][hidden] input embeddings (overwritten). Row m is token row_pos[m] of sequence row_seq[m];
    // rows must be ordered so that a row's earlier positions are either cached or earlier in the batch.
    // out: [M][hidden] final-normed hidden states.
    void forward(Arena &ar, float *x, int M, const std::vector<int> &row_seq, const std::vector<int> &row_pos,
                 float *out);
};
