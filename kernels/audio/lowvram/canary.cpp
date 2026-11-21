// Canary-Qwen 2.5B speech-to-text: NeMo mel frontend -> FastConformer encoder (32 blocks)
// -> Linear 1024->2048 -> audio rows spliced into a Qwen3-1.7B chat prompt -> greedy decode.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <string>
#include <vector>

#include "bpe.h"
#include "common.h"
#include "gguf.h"
#include "models.h"
#include "bridge.h"
#include "ops.h"
#include "qwen.h"
#include "wav.h"

namespace {

constexpr float kLnEps = 1e-5f;
constexpr float kBnEps = 1e-5f;

// ---------------- mel frontend (host) ----------------

struct Mel {
    int n_fft = 512, win = 400, hop = 160, n_mels = 128;
    float preemph = 0.97f;
    bool reflect = true;
    std::vector<double> window;  // n_fft, centered
    std::vector<float> fb;       // [n_mels][n_fft/2+1]

    void load(const Gguf &g) {
        n_fft = (int)g.get_int("stt.frontend.n_fft", 512);
        win = (int)g.get_int("stt.frontend.win_length", 400);
        hop = (int)g.get_int("stt.frontend.hop_length", 160);
        n_mels = (int)g.get_int("stt.frontend.num_mels", 128);
        preemph = (float)g.get_float("stt.frontend.pre_emphasis", 0.97);
        reflect = g.get_str("stt.frontend.pad_mode", "reflect") != "constant";
        std::vector<float> w = to_f32(g.tensor("frontend.window"));
        window.assign(n_fft, 0.0);
        int off = (n_fft - win) / 2;
        for (int i = 0; i < win && i < (int)w.size(); i++) window[off + i] = w[i];
        fb = to_f32(g.tensor("frontend.mel_filterbank"));
        if ((int)fb.size() != n_mels * (n_fft / 2 + 1)) fail("mel filterbank has unexpected size");
    }

    static void fft(std::vector<std::complex<double>> &a) {
        size_t n = a.size();
        for (size_t i = 1, j = 0; i < n; i++) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (size_t len = 2; len <= n; len <<= 1) {
            double ang = -2 * M_PI / len;
            std::complex<double> wl(cos(ang), sin(ang));
            for (size_t i = 0; i < n; i += len) {
                std::complex<double> w(1);
                for (size_t k = 0; k < len / 2; k++) {
                    auto u = a[i + k], v = a[i + k + len / 2] * w;
                    a[i + k] = u + v;
                    a[i + k + len / 2] = u - v;
                    w *= wl;
                }
            }
        }
    }

    // Returns normalized log-mel as [T][n_mels].
    std::vector<float> compute(const std::vector<float> &pcm, int &T) const {
        size_t n = pcm.size();
        int pad = n_fft / 2;
        if (n < (size_t)pad + 2) fail("audio too short");
        std::vector<double> e(n);
        e[0] = pcm[0];
        for (size_t i = 1; i < n; i++) e[i] = (double)pcm[i] - (double)preemph * pcm[i - 1];
        std::vector<double> p(n + 2 * pad, 0.0);
        for (size_t i = 0; i < n; i++) p[pad + i] = e[i];
        if (reflect) {
            for (int i = 0; i < pad; i++) p[i] = e[pad - i];
            for (int i = 0; i < pad; i++) p[pad + n + i] = e[n - 2 - i];
        }
        T = (int)(n / hop) + 1;
        int nf = n_fft / 2 + 1;
        std::vector<float> mel((size_t)T * n_mels);
        std::vector<std::complex<double>> buf(n_fft);
        std::vector<double> pw(nf);
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < n_fft; k++) buf[k] = p[(size_t)t * hop + k] * window[k];
            fft(buf);
            for (int k = 0; k < nf; k++) pw[k] = std::norm(buf[k]);
            for (int m = 0; m < n_mels; m++) {
                double s = 0;
                const float *row = &fb[(size_t)m * nf];
                for (int k = 0; k < nf; k++) s += row[k] * pw[k];
                mel[(size_t)t * n_mels + m] = (float)std::log(s + 5.9604644775390625e-08);
            }
        }
        // per-feature normalization; constant padding masks the trailing frame
        int n_norm = reflect ? T : T - 1;
        for (int m = 0; m < n_mels; m++) {
            double s = 0;
            for (int t = 0; t < n_norm; t++) s += mel[(size_t)t * n_mels + m];
            double mean = s / n_norm, v = 0;
            for (int t = 0; t < n_norm; t++) {
                double d = mel[(size_t)t * n_mels + m] - mean;
                v += d * d;
            }
            double inv = 1.0 / (std::sqrt(v / (n_norm - 1)) + 1e-5);
            for (int t = 0; t < n_norm; t++)
                mel[(size_t)t * n_mels + m] = (float)((mel[(size_t)t * n_mels + m] - mean) * inv);
            if (!reflect) mel[(size_t)(T - 1) * n_mels + m] = 0.f;
        }
        return mel;
    }
};

// ---------------- FastConformer encoder ----------------

struct ConfBlock {
    float *ln_ff1_w, *ln_ff1_b, *ln_att_w, *ln_att_b, *ln_conv_w, *ln_conv_b, *ln_ff2_w, *ln_ff2_b, *ln_out_w, *ln_out_b;
    Weight ff1_1, ff1_2, ff2_1, ff2_2, q, k, v, o, pos, pw1, pw2;
    float *pos_u, *pos_v, *dw_w, *dw_b, *bn_scale, *bn_bias;
};

struct Encoder {
    int d = 1024, heads = 8, ffn = 4096, kernel = 9, sub_ch = 256;
    float *c0_w, *c0_b, *c2_w, *c2_b, *c5_w, *c5_b;
    Weight c3, c6, pre_out, proj;
    std::vector<ConfBlock> blocks;

    void load(const Gguf &g) {
        int n_layers = (int)g.get_int("stt.canary_qwen.encoder.n_layers", 32);
        d = (int)g.get_int("stt.canary_qwen.encoder.d_model", 1024);
        heads = (int)g.get_int("stt.canary_qwen.encoder.n_heads", 8);
        ffn = (int)g.get_int("stt.canary_qwen.encoder.d_ff", 4096);
        kernel = (int)g.get_int("stt.canary_qwen.encoder.conv_kernel", 9);
        sub_ch = (int)g.get_int("stt.canary_qwen.encoder.subsampling_channels", 256);
        auto T = [&](const std::string &n) -> const GgufTensor & { return g.tensor(n); };
        auto F = [&](const std::string &n) { return upload_f32(T(n)); };
        auto L = [&](const std::string &n, bool bias = true) {
            return upload_linear(T(n + ".weight"), bias ? &T(n + ".bias") : nullptr);
        };
        c0_w = F("enc.pre_encode.conv.0.weight");
        c0_b = F("enc.pre_encode.conv.0.bias");
        c2_w = F("enc.pre_encode.conv.2.weight");
        c2_b = F("enc.pre_encode.conv.2.bias");
        c3 = L("enc.pre_encode.conv.3");
        c5_w = F("enc.pre_encode.conv.5.weight");
        c5_b = F("enc.pre_encode.conv.5.bias");
        c6 = L("enc.pre_encode.conv.6");
        pre_out = L("enc.pre_encode.out");
        proj = L("enc.proj");
        blocks.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            std::string p = "enc.blocks." + std::to_string(i) + ".";
            ConfBlock &b = blocks[i];
            b.ln_ff1_w = F(p + "norm_ff1.weight");
            b.ln_ff1_b = F(p + "norm_ff1.bias");
            b.ln_att_w = F(p + "norm_attn.weight");
            b.ln_att_b = F(p + "norm_attn.bias");
            b.ln_conv_w = F(p + "norm_conv.weight");
            b.ln_conv_b = F(p + "norm_conv.bias");
            b.ln_ff2_w = F(p + "norm_ff2.weight");
            b.ln_ff2_b = F(p + "norm_ff2.bias");
            b.ln_out_w = F(p + "norm_out.weight");
            b.ln_out_b = F(p + "norm_out.bias");
            b.ff1_1 = L(p + "ff1.linear1");
            b.ff1_2 = L(p + "ff1.linear2");
            b.ff2_1 = L(p + "ff2.linear1");
            b.ff2_2 = L(p + "ff2.linear2");
            b.q = L(p + "attn.linear_q");
            b.k = L(p + "attn.linear_k");
            b.v = L(p + "attn.linear_v");
            b.o = L(p + "attn.linear_out");
            b.pos = L(p + "attn.linear_pos", false);
            b.pos_u = F(p + "attn.pos_bias_u");
            b.pos_v = F(p + "attn.pos_bias_v");
            b.pw1 = L(p + "conv.pointwise1");
            b.pw2 = L(p + "conv.pointwise2");
            b.dw_w = F(p + "conv.depthwise.weight");
            b.dw_b = F(p + "conv.depthwise.bias");
            // fold eval-mode BatchNorm into a per-channel affine
            std::vector<float> w = to_f32(T(p + "conv.bn.weight")), bb = to_f32(T(p + "conv.bn.bias"));
            std::vector<float> mu = to_f32(T(p + "conv.bn.running_mean")), var = to_f32(T(p + "conv.bn.running_var"));
            std::vector<float> sc(d), bi(d);
            for (int c = 0; c < d; c++) {
                sc[c] = w[c] / std::sqrt(var[c] + kBnEps);
                bi[c] = bb[c] - mu[c] * sc[c];
            }
            b.bn_scale = dalloc_f(d);
            b.bn_bias = dalloc_f(d);
            upload(b.bn_scale, sc.data(), d * 4);
            upload(b.bn_bias, bi.data(), d * 4);
        }
    }

    // mel [T_mel][n_mels] on device -> audio embeddings [T][proj.N]; returns T.
    int forward(Arena &ar, const float *mel, int T_mel, int n_mels, float *&out) {
        auto down = [](int n) { return (n - 1) / 2 + 1; };
        int H1 = down(T_mel), W1 = down(n_mels);
        int H2 = down(H1), W2 = down(W1);
        int H3 = down(H2), W3 = down(W2);
        int C = sub_ch;
        float *a = ar.f((size_t)H1 * W1 * C);
        conv2d_s2_c1(mel, T_mel, n_mels, c0_w, c0_b, C, a);
        relu(a, (size_t)H1 * W1 * C);
        float *b = ar.f((size_t)H2 * W2 * C);
        dwconv2d_s2(a, H1, W1, C, c2_w, c2_b, b);
        float *c = ar.f((size_t)H2 * W2 * C);
        matmul(b, H2 * W2, c3, c);
        relu(c, (size_t)H2 * W2 * C);
        float *e = ar.f((size_t)H3 * W3 * C);
        dwconv2d_s2(c, H2, W2, C, c5_w, c5_b, e);
        float *f = ar.f((size_t)H3 * W3 * C);
        matmul(e, H3 * W3, c6, f);
        relu(f, (size_t)H3 * W3 * C);
        float *flat = ar.f((size_t)H3 * W3 * C);
        transpose12(f, H3, W3, C, flat);  // [T][F][C] -> [T][C][F]
        int T = H3;
        float *x = ar.f((size_t)T * d);
        matmul(flat, T, pre_out, x);

        // relative positional encodings for positions T-1 .. -(T-1)
        int P = 2 * T - 1;
        std::vector<float> pe((size_t)P * d);
        for (int k = 0; k < P; k++) {
            double pos = T - 1 - k;
            for (int i = 0; i < d / 2; i++) {
                double div = std::exp(-(2.0 * i) * std::log(10000.0) / d);
                pe[(size_t)k * d + 2 * i] = (float)std::sin(pos * div);
                pe[(size_t)k * d + 2 * i + 1] = (float)std::cos(pos * div);
            }
        }
        float *pe_d = ar.f((size_t)P * d);
        upload(pe_d, pe.data(), pe.size() * 4);

        float *h = ar.f((size_t)T * d), *t1 = ar.f((size_t)T * ffn), *t2 = ar.f((size_t)T * d);
        float *q = ar.f((size_t)T * d), *k = ar.f((size_t)T * d), *v = ar.f((size_t)T * d);
        float *p = ar.f((size_t)P * d), *att = ar.f((size_t)T * d), *g2 = ar.f((size_t)T * 2 * d);
        int D = d / heads;
        for (size_t i = 0; i < blocks.size(); i++) {
            ConfBlock &B = blocks[i];
            // macaron FF 1
            layernorm(x, T, d, B.ln_ff1_w, B.ln_ff1_b, kLnEps, h);
            matmul(h, T, B.ff1_1, t1);
            silu(t1, t1, (size_t)T * ffn);
            matmul(t1, T, B.ff1_2, t2);
            add_scaled(x, t2, 0.5f, (size_t)T * d);
            // rel-pos self attention
            layernorm(x, T, d, B.ln_att_w, B.ln_att_b, kLnEps, h);
            matmul(h, T, B.q, q);
            matmul(h, T, B.k, k);
            matmul(h, T, B.v, v);
            matmul(pe_d, P, B.pos, p);
            attn_relpos(q, k, v, p, B.pos_u, B.pos_v, T, heads, D, att);
            matmul(att, T, B.o, x, true);
            // conv module
            layernorm(x, T, d, B.ln_conv_w, B.ln_conv_b, kLnEps, h);
            matmul(h, T, B.pw1, g2);
            glu(g2, T, d, t2);
            dwconv1d(t2, T, d, B.dw_w, B.dw_b, kernel, (kernel - 1) / 2, T, h);
            affine_cols(h, B.bn_scale, B.bn_bias, T, d);
            silu(h, h, (size_t)T * d);
            matmul(h, T, B.pw2, x, true);
            // macaron FF 2
            layernorm(x, T, d, B.ln_ff2_w, B.ln_ff2_b, kLnEps, h);
            matmul(h, T, B.ff2_1, t1);
            silu(t1, t1, (size_t)T * ffn);
            matmul(t1, T, B.ff2_2, t2);
            add_scaled(x, t2, 0.5f, (size_t)T * d);
            layernorm(x, T, d, B.ln_out_w, B.ln_out_b, kLnEps, x);
        }
        out = ar.f((size_t)T * proj.N);
        matmul(x, T, proj, out);
        return T;
    }
};

}  // namespace

class CanaryBackend final : public SttBackend {
    Gguf g; Bpe tok; Mel mel; Encoder enc; LmConfig lc; Lm lm; Weight embd; Arena ar;
    int sr=16000, im_start=0,im_end=0,eot=0,eos=0; std::vector<int> prefix,suffix;
public:
    explicit CanaryBackend(const SttOptions& o):g(o.model) {
        double t0=now_s();
        if(g.get_str("general.architecture","")!="canary_qwen")fail("not a canary_qwen model");
        sr=(int)g.get_int("stt.frontend.sample_rate",16000);
    tok.load_gguf(g);
    mel.load(g);

    progress("load", 0, 1);
    enc.load(g);
    lc.n_layer = (int)g.get_int("stt.canary_qwen.decoder.n_layers", 28);
    lc.hidden = (int)g.get_int("stt.canary_qwen.decoder.hidden_size", 2048);
    lc.n_head = (int)g.get_int("stt.canary_qwen.decoder.n_heads", 16);
    lc.n_kv = (int)g.get_int("stt.canary_qwen.decoder.n_kv_heads", 8);
    lc.head_dim = (int)g.get_int("stt.canary_qwen.decoder.head_dim", 128);
    lc.ffn = (int)g.get_int("stt.canary_qwen.decoder.intermediate_size", 6144);
    lc.eps = (float)g.get_float("stt.canary_qwen.decoder.rms_norm_eps", 1e-6);
    lc.rope_theta = (float)g.get_float("stt.canary_qwen.decoder.rope_theta", 1e6);
    lc.qk_norm = true;
    LmNames ln;
    ln.attn_norm = "dec.blocks.%d.norm_attn.weight";
    ln.ffn_norm = "dec.blocks.%d.norm_ffn.weight";
    ln.q = "dec.blocks.%d.attn.q.weight";
    ln.k = "dec.blocks.%d.attn.k.weight";
    ln.v = "dec.blocks.%d.attn.v.weight";
    ln.o = "dec.blocks.%d.attn.o.weight";
    ln.q_norm = "dec.blocks.%d.attn.q_norm.weight";
    ln.k_norm = "dec.blocks.%d.attn.k_norm.weight";
    ln.gate = "dec.blocks.%d.ffn.gate.weight";
    ln.up = "dec.blocks.%d.ffn.up.weight";
    ln.down = "dec.blocks.%d.ffn.down.weight";
    ln.out_norm = "dec.output_norm.weight";
    lm.load(g, lc, ln);
    embd = upload_linear(g.tensor("dec.token_embd.weight"));

    int audio_tokens_max = (int)(o.chunk_seconds * sr / mel.hop / 8) + 2;
    int max_ctx = 64 + audio_tokens_max + o.max_new_tokens;
    lm.init_cache(1, max_ctx);
    ar.init((size_t)(o.chunk_seconds * 6e6) + (256u << 20));
    progress("load", 1, 1);
    logf("INFO loaded in %.1fs, %.0f MiB VRAM free", now_s() - t0, vram_free() / 1048576.0);

    im_start = tok.id("<|im_start|>"); im_end = tok.id("<|im_end|>"); eot = tok.id("<|endoftext|>");
    eos = (int)g.get_int("tokenizer.ggml.eos_token_id", im_end);
    prefix = {im_start}; suffix = {im_end};
    for (int t : tok.encode("user\nTranscribe the following: ")) prefix.push_back(t);
    for (int t : tok.encode("\n")) suffix.push_back(t);
    suffix.push_back(im_start);
    for (int t : tok.encode("assistant\n")) suffix.push_back(t);

    }
    std::string info() const override { return "{\"languages\":[\"en\"],\"translate\":false}"; }
    SttResult transcribe(Audio audio,const SttOptions& o) override {
        double t0=now_s(); ar.reset(0);
        if(audio.sample_rate!=sr)audio.samples=resample(audio.samples,audio.sample_rate,sr);
        if(audio.samples.size()<(size_t)sr/10)fail("audio is shorter than 0.1 s");
    auto pieces = split_audio(audio.samples, (size_t)(o.chunk_seconds * sr), sr);
    SttResult res;
    res.language = "en";
    std::vector<float> logits(embd.N);
    int H = lc.hidden;
    for (size_t pi = 0; pi < pieces.size(); pi++) {
        progress("stt_chunk", (long)pi, (long)pieces.size());
        std::vector<float> pcm(audio.samples.begin() + pieces[pi].first, audio.samples.begin() + pieces[pi].second);
        if (pcm.size() < (size_t)sr / 10) continue;
        size_t mark = ar.mark();
        int T_mel;
        std::vector<float> m = mel.compute(pcm, T_mel);
        float *mel_d = ar.f(m.size());
        upload(mel_d, m.data(), m.size() * 4);
        float *aud = nullptr;
        int T = enc.forward(ar, mel_d, T_mel, mel.n_mels, aud);

        // prompt = prefix + audio rows + suffix
        int P = (int)prefix.size() + T + (int)suffix.size();
        float *x = ar.f((size_t)P * H);
        int *ids_d = ar.i(prefix.size() + suffix.size());
        upload(ids_d, prefix.data(), prefix.size() * 4);
        upload(ids_d + prefix.size(), suffix.data(), suffix.size() * 4);
        embed_rows(embd, ids_d, (int)prefix.size(), x);
        dcopy(x + prefix.size() * H, aud, (size_t)T * H * 4);
        embed_rows(embd, ids_d + prefix.size(), (int)suffix.size(), x + (size_t)(prefix.size() + T) * H);
        std::vector<int> seq(P, 0), pos(P);
        for (int i = 0; i < P; i++) pos[i] = i;
        float *hid = ar.f((size_t)P * H);
        lm.forward(ar, x, P, seq, pos, hid);
        float *lg = ar.f(embd.N);
        float *one = ar.f(H);
        int *id1 = ar.i(1);
        std::vector<int> out_ids;
        const float *last = hid + (size_t)(P - 1) * H;
        for (int step = 0; step < o.max_new_tokens; step++) {
            progress("tokens",step,o.max_new_tokens);
            matmul(last, 1, embd, lg);
            download(logits.data(), lg, logits.size() * 4);
            int best = (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (best == eos || best == im_end || best == eot) break;
            out_ids.push_back(best);
            if (step + 1 == o.max_new_tokens) break;
            upload(id1, &best, 4);
            embed_rows(embd, id1, 1, one);
            std::vector<int> s1 = {0}, p1 = {P + step};
            lm.forward(ar, one, 1, s1, p1, one);
            last = one;
        }
        std::string text = tok.decode(out_ids);
        size_t a = text.find_first_not_of(" \n\t"), b = text.find_last_not_of(" \n\t");
        text = a == std::string::npos ? "" : text.substr(a, b - a + 1);
        if (!text.empty()) {
            if (!res.text.empty()) res.text += ' ';
            res.text += text;
            res.segments.push_back({pieces[pi].first / (double)sr, pieces[pi].second / (double)sr, text});
        }
        fprintf(stderr, "PARTIAL %s\n", text.c_str());
        ar.reset(mark);
    }
    progress("stt_chunk", (long)pieces.size(), (long)pieces.size());
    logf("INFO transcribed %.1fs of audio in %.1fs", audio.samples.size() / (double)sr, now_s() - t0);
    return res;
    }
};
std::unique_ptr<SttBackend> make_canary(const SttOptions& o){return std::make_unique<CanaryBackend>(o);}
