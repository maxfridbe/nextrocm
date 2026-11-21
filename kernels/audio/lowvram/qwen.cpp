#include "qwen.h"

#include <algorithm>
#include <cstdio>

#include "common.h"

namespace {

std::string fmt(const std::string &pattern, int i) {
    char buf[256];
    snprintf(buf, sizeof buf, pattern.c_str(), i);
    return buf;
}

}  // namespace

void Lm::load(const Gguf &g, const LmConfig &cfg, const LmNames &n) {
    c = cfg;
    layers.resize(c.n_layer);
    for (int i = 0; i < c.n_layer; i++) {
        LmLayer &L = layers[i];
        L.attn_norm = upload_f32(g.tensor(fmt(n.attn_norm, i)));
        L.ffn_norm = upload_f32(g.tensor(fmt(n.ffn_norm, i)));
        auto opt = [&](const std::string &pat) { return pat.empty() ? nullptr : g.find(fmt(pat, i)); };
        L.q = upload_linear(g.tensor(fmt(n.q, i)), opt(n.q_bias));
        L.k = upload_linear(g.tensor(fmt(n.k, i)), opt(n.k_bias));
        L.v = upload_linear(g.tensor(fmt(n.v, i)), opt(n.v_bias));
        L.o = upload_linear(g.tensor(fmt(n.o, i)));
        if (c.qk_norm) {
            L.q_norm = upload_f32(g.tensor(fmt(n.q_norm, i)));
            L.k_norm = upload_f32(g.tensor(fmt(n.k_norm, i)));
        }
        L.gate = upload_linear(g.tensor(fmt(n.gate, i)));
        L.up = upload_linear(g.tensor(fmt(n.up, i)));
        L.down = upload_linear(g.tensor(fmt(n.down, i)));
        if (L.q.N != c.n_head * c.head_dim || L.k.N != c.n_kv * c.head_dim || L.q.K != c.hidden)
            fail("LM layer %d: unexpected attention shapes", i);
    }
    out_norm = upload_f32(g.tensor(n.out_norm));
}

void Lm::init_cache(int n_seq, int max_ctx) { kv.init(n_seq, c.n_layer, max_ctx, c.n_kv * c.head_dim); }

size_t Lm::weight_bytes() const {
    size_t b = 0;
    for (auto &L : layers) b += L.q.bytes() + L.k.bytes() + L.v.bytes() + L.o.bytes() + L.gate.bytes() + L.up.bytes() + L.down.bytes();
    return b;
}

void Lm::free() {
    for (auto &L : layers) {
        for (Weight *w : {&L.q, &L.k, &L.v, &L.o, &L.gate, &L.up, &L.down}) w->free();
        for (float *p : {L.attn_norm, L.ffn_norm, L.q_norm, L.k_norm}) dfree(p);
    }
    layers.clear();
    dfree(out_norm);
    out_norm = nullptr;
    kv.free();
}

void Lm::forward(Arena &ar, float *x, int M, const std::vector<int> &row_seq, const std::vector<int> &row_pos,
                 float *out) {
    size_t mark = ar.mark();
    int H = c.hidden, D = c.head_dim;
    int qd = c.n_head * D, kd = c.n_kv * D;
    int max_keys = 0;
    for (int m = 0; m < M; m++) {
        if (row_pos[m] >= kv.max_ctx) fail("context length %d exceeds the KV cache (%d)", row_pos[m] + 1, kv.max_ctx);
        max_keys = std::max(max_keys, row_pos[m] + 1);
    }
    int *seq_d = ar.i(M), *pos_d = ar.i(M);
    upload(seq_d, row_seq.data(), M * sizeof(int));
    upload(pos_d, row_pos.data(), M * sizeof(int));

    float *h = ar.f((size_t)M * H);
    float *q = ar.f((size_t)M * qd);
    float *k = ar.f((size_t)M * kd);
    float *v = ar.f((size_t)M * kd);
    float *a = ar.f((size_t)M * qd);
    float *g = ar.f((size_t)M * c.ffn);
    float *u = ar.f((size_t)M * c.ffn);

    for (int l = 0; l < c.n_layer; l++) {
        LmLayer &L = layers[l];
        rmsnorm(x, M, H, L.attn_norm, c.eps, h);
        matmul(h, M, L.q, q);
        matmul(h, M, L.k, k);
        matmul(h, M, L.v, v);
        if (c.qk_norm) {
            rmsnorm(q, M * c.n_head, D, L.q_norm, c.eps, q);
            rmsnorm(k, M * c.n_kv, D, L.k_norm, c.eps, k);
        }
        rope_neox(q, M, c.n_head, D, pos_d, c.rope_theta);
        rope_neox(k, M, c.n_kv, D, pos_d, c.rope_theta);
        kv_store(kv, l, k, v, M, seq_d, pos_d);
        attn_cached(q, kv, l, M, c.n_head, c.n_kv, D, seq_d, pos_d, max_keys, a);
        matmul(a, M, L.o, x, true);
        rmsnorm(x, M, H, L.ffn_norm, c.eps, h);
        matmul(h, M, L.gate, g);
        matmul(h, M, L.up, u);
        swiglu(g, u, g, (size_t)M * c.ffn);
        matmul(g, M, L.down, x, true);
    }
    rmsnorm(x, M, H, out_norm, c.eps, out);
    ar.reset(mark);
}
