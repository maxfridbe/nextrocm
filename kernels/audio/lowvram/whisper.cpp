// OpenAI Whisper speech-to-text (tiny .. large-v3-turbo) from whisper.cpp's ggml-*.bin files:
// log-mel frontend (DFT on the GPU) -> conv stem + transformer encoder over a 30 s window
// -> transformer decoder with self and cross attention -> greedy decode.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "gguf.h"
#include "models.h"
#include "bridge.h"
#include "ops.h"
#include "wav.h"

namespace {

constexpr float kLnEps = 1e-5f;
constexpr int kSampleRate = 16000, kNfft = 400, kHop = 160, kChunkFrames = 3000;

const char *const kLanguages[] = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id",
    "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg",
    "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br",
    "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so",
    "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt",
    "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue"};

// Symbols Whisper is told never to emit (whisper.cpp's suppress_nst list).
const char *const kNonSpeech[] = {
    "\"", "#", "(", ")", "*", "+", "/", ":", ";", "<", "=", ">", "@", "[", "\\", "]", "^", "_", "`", "{", "|",
    "}", "~", "「", "」", "『", "』", "<<", ">>", "<<<", ">>>", "--", "---", "-(", "-[", "('", "(\"", "((", "))",
    "(((", ")))", "[[", "]]", "{{", "}}", "♪♪", "♪♪♪", "♩", "♪", "♫", "♬", "♭", "♮", "♯"};

// ---------------- ggml-*.bin (whisper.cpp's legacy container) ----------------

struct Hparams {
    int n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer;
    int n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype;
};

class GgmlBin {
public:
    Hparams hp{};
    int filt_mels = 0, filt_bins = 0;
    std::vector<float> filters;      // [n_mels][n_fft/2+1]
    std::vector<std::string> vocab;  // raw bytes per token id

    explicit GgmlBin(const std::string &path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) fail("cannot open %s", path.c_str());
        struct stat st;
        fstat(fd_, &st);
        size_ = st.st_size;
        void *m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (m == MAP_FAILED) fail("cannot mmap %s", path.c_str());
        map_ = static_cast<const uint8_t *>(m);
        p_ = map_;
        if (get<uint32_t>() != 0x67676d6c) fail("%s is not a whisper.cpp ggml model", path.c_str());
        int32_t *h = &hp.n_vocab;
        for (int i = 0; i < 11; i++) h[i] = get<int32_t>();
        filt_mels = get<int32_t>();
        filt_bins = get<int32_t>();
        filters.resize((size_t)filt_mels * filt_bins);
        take(filters.data(), filters.size() * 4);
        int nv = get<int32_t>();
        vocab.resize(nv);
        for (int i = 0; i < nv; i++) {
            uint32_t len = get<uint32_t>();
            vocab[i].resize(len);
            take(vocab[i].data(), len);
        }
        while (p_ < map_ + size_) {
            int32_t nd = get<int32_t>(), len = get<int32_t>(), type = get<int32_t>();
            GgufTensor t;
            t.type = (uint32_t)type;
            for (int d = 0; d < nd; d++) t.ne.push_back(get<int32_t>());
            t.name.resize(len);
            take(t.name.data(), len);
            if (type != GGML_F32 && type != GGML_F16 && type != GGML_Q8_0)
                fail("%s: tensor %s has ggml type %d; use an f16 or q8_0 model", path.c_str(), t.name.c_str(), type);
            t.data = p_;
            if (p_ + t.nbytes() > map_ + size_) fail("%s: tensor %s is truncated", path.c_str(), t.name.c_str());
            p_ += t.nbytes();
            tensors_[t.name] = t;
        }
    }
    ~GgmlBin() {
        if (map_) munmap(const_cast<uint8_t *>(map_), size_);
        if (fd_ >= 0) close(fd_);
    }
    GgmlBin(const GgmlBin &) = delete;
    GgmlBin &operator=(const GgmlBin &) = delete;

    const GgufTensor &tensor(const std::string &n) const {
        auto it = tensors_.find(n);
        if (it == tensors_.end()) fail("model is missing tensor %s", n.c_str());
        return it->second;
    }
    size_t weight_bytes() const {
        size_t s = 0;
        for (auto &kv : tensors_) s += kv.second.nbytes();
        return s;
    }

private:
    int fd_ = -1;
    size_t size_ = 0;
    const uint8_t *map_ = nullptr, *p_ = nullptr;
    std::map<std::string, GgufTensor> tensors_;

    template <class T>
    T get() {
        T v;
        take(&v, sizeof v);
        return v;
    }
    void take(void *dst, size_t n) {
        if (p_ + n > map_ + size_) fail("whisper model file is truncated");
        memcpy(dst, p_, n);
        p_ += n;
    }
};

// ---------------- special tokens (as whisper.cpp numbers them) ----------------

struct Tokens {
    int eot = 50256, sot = 50257, translate = 50357, transcribe = 50358, nosp = 50361, notimestamps = 50362;
    int n_langs = 0;
    bool multilingual = false;

    explicit Tokens(int n_vocab) {
        multilingual = n_vocab >= 51865;
        n_langs = n_vocab - 51765 - (multilingual ? 1 : 0);
        if (multilingual) {
            eot++;
            sot++;
            int dt = n_langs - 98;
            translate += dt;
            transcribe += dt;
            nosp += dt;
            notimestamps += dt;
        }
    }
    int lang_token(int lang_id) const { return sot + 1 + lang_id; }
    int ts_begin() const { return notimestamps + 1; }  // <|0.00|>, then one token per 20 ms
};

int language_id(const std::string &code) {
    for (int i = 0; i < (int)(sizeof kLanguages / sizeof *kLanguages); i++)
        if (code == kLanguages[i]) return i;
    return -1;
}

// ---------------- weights ----------------

struct EncLayer {
    float *attn_ln_w, *attn_ln_b, *mlp_ln_w, *mlp_ln_b;
    Weight q, k, v, o, fc1, fc2;
};

struct DecLayer {
    float *attn_ln_w, *attn_ln_b, *cross_ln_w, *cross_ln_b, *mlp_ln_w, *mlp_ln_b;
    Weight q, k, v, o, cq, ck, cv, co, fc1, fc2;
};

struct Model {
    Hparams hp{};
    Weight conv1, conv2;
    float *enc_pos, *ln_post_w, *ln_post_b;
    std::vector<EncLayer> enc;
    Weight tok_emb;
    float *dec_pos, *dec_ln_w, *dec_ln_b;
    std::vector<DecLayer> dec;
    Weight dft, mel;  // frontend as matmuls: [2*201][400] cos|sin rows, [n_mels][201] filterbank

    void load(const GgmlBin &f, bool q8) {
        hp = f.hp;
        auto T = [&](const std::string &n) -> const GgufTensor & { return f.tensor(n); };
        auto F = [&](const std::string &n) { return upload_f32(T(n)); };
        auto L = [&](const std::string &n, bool bias = true) {
            return upload_linear(T(n + ".weight"), bias ? &T(n + ".bias") : nullptr, q8);
        };
        conv1 = upload_linear(T("encoder.conv1.weight"), &T("encoder.conv1.bias"));
        conv2 = upload_linear(T("encoder.conv2.weight"), &T("encoder.conv2.bias"), q8);
        enc_pos = F("encoder.positional_embedding");
        ln_post_w = F("encoder.ln_post.weight");
        ln_post_b = F("encoder.ln_post.bias");
        enc.resize(hp.n_audio_layer);
        for (int i = 0; i < hp.n_audio_layer; i++) {
            std::string p = "encoder.blocks." + std::to_string(i) + ".";
            EncLayer &l = enc[i];
            l.attn_ln_w = F(p + "attn_ln.weight");
            l.attn_ln_b = F(p + "attn_ln.bias");
            l.mlp_ln_w = F(p + "mlp_ln.weight");
            l.mlp_ln_b = F(p + "mlp_ln.bias");
            l.q = L(p + "attn.query");
            l.k = L(p + "attn.key", false);
            l.v = L(p + "attn.value");
            l.o = L(p + "attn.out");
            l.fc1 = L(p + "mlp.0");
            l.fc2 = L(p + "mlp.2");
        }
        tok_emb = upload_linear(T("decoder.token_embedding.weight"), nullptr, q8);
        dec_pos = F("decoder.positional_embedding");
        dec_ln_w = F("decoder.ln.weight");
        dec_ln_b = F("decoder.ln.bias");
        dec.resize(hp.n_text_layer);
        for (int i = 0; i < hp.n_text_layer; i++) {
            std::string p = "decoder.blocks." + std::to_string(i) + ".";
            DecLayer &l = dec[i];
            l.attn_ln_w = F(p + "attn_ln.weight");
            l.attn_ln_b = F(p + "attn_ln.bias");
            l.cross_ln_w = F(p + "cross_attn_ln.weight");
            l.cross_ln_b = F(p + "cross_attn_ln.bias");
            l.mlp_ln_w = F(p + "mlp_ln.weight");
            l.mlp_ln_b = F(p + "mlp_ln.bias");
            l.q = L(p + "attn.query");
            l.k = L(p + "attn.key", false);
            l.v = L(p + "attn.value");
            l.o = L(p + "attn.out");
            l.cq = L(p + "cross_attn.query");
            l.ck = L(p + "cross_attn.key", false);
            l.cv = L(p + "cross_attn.value");
            l.co = L(p + "cross_attn.out");
            l.fc1 = L(p + "mlp.0");
            l.fc2 = L(p + "mlp.2");
        }

        int nb = kNfft / 2 + 1;
        if (f.filt_bins != nb || f.filt_mels != hp.n_mels) fail("unexpected mel filterbank %dx%d", f.filt_mels, f.filt_bins);
        std::vector<float> d((size_t)2 * nb * kNfft);
        for (int k = 0; k < nb; k++)
            for (int n = 0; n < kNfft; n++) {
                double a = 2 * M_PI * (double)k * n / kNfft;
                d[(size_t)k * kNfft + n] = (float)std::cos(a);
                d[(size_t)(nb + k) * kNfft + n] = (float)-std::sin(a);
            }
        dft = make_f32(d, 2 * nb, kNfft);
        mel = make_f32(f.filters, hp.n_mels, nb);
    }

    static Weight make_f32(const std::vector<float> &h, int N, int K) {
        Weight W;
        W.t = WT::F32;
        W.N = N;
        W.K = K;
        W.w = dmalloc(h.size() * 4);
        upload(W.w, h.data(), h.size() * 4);
        return W;
    }
};

// A view of a device f32 matrix [N][K] as a matmul weight (not owned).
Weight view(const float *p, int N, int K) {
    Weight W;
    W.t = WT::F32;
    W.N = N;
    W.K = K;
    W.w = const_cast<float *>(p);
    return W;
}

// ---------------- frontend ----------------

// pcm (<= 30 s at 16 kHz) -> normalized log-mel [3000][n_mels] on the device.
void log_mel(Arena &ar, const Model &m, const std::vector<float> &pcm, float *out) {
    int pad = kNfft / 2;
    std::vector<float> padded(pad + (size_t)kChunkFrames * kHop + kNfft, 0.f);
    size_t n = std::min(pcm.size(), (size_t)kChunkFrames * kHop);
    std::copy(pcm.begin(), pcm.begin() + n, padded.begin() + pad);
    for (int i = 0; i < pad && i + 1 < (int)n; i++) padded[pad - 1 - i] = pcm[i + 1];  // reflect
    std::vector<float> frames((size_t)kChunkFrames * kNfft);
    for (int t = 0; t < kChunkFrames; t++)
        for (int k = 0; k < kNfft; k++)
            frames[(size_t)t * kNfft + k] =
                padded[(size_t)t * kHop + k] * (float)(0.5 * (1.0 - std::cos(2 * M_PI * k / kNfft)));
    size_t mark = ar.mark();
    int nb = kNfft / 2 + 1, nm = m.hp.n_mels;
    float *fr = ar.f(frames.size()), *spec = ar.f((size_t)kChunkFrames * 2 * nb), *pw = ar.f((size_t)kChunkFrames * nb);
    upload(fr, frames.data(), frames.size() * 4);
    matmul(fr, kChunkFrames, m.dft, spec);
    power_spec(spec, kChunkFrames, nb, pw);
    matmul(pw, kChunkFrames, m.mel, out);
    std::vector<float> mel((size_t)kChunkFrames * nm);
    download(mel.data(), out, mel.size() * 4);
    ar.reset(mark);
    float mx = -1e30f;
    for (float &v : mel) {
        v = std::log10(std::max(v, 1e-10f));
        mx = std::max(mx, v);
    }
    for (float &v : mel) v = (std::max(v, mx - 8.f) + 4.f) / 4.f;
    upload(out, mel.data(), mel.size() * 4);
}

// ---------------- encoder ----------------

// mel [3000][n_mels] -> encoder states [1500][n_audio_state] (written to out).
void encode(Arena &ar, const Model &m, const float *mel, float *out) {
    const Hparams &hp = m.hp;
    int C = hp.n_audio_state, H = hp.n_audio_head, D = C / H, T = hp.n_audio_ctx, nm = hp.n_mels;
    size_t mark = ar.mark();
    // conv stem: k=3 pad=1 (stride 1, then stride 2), GELU after each
    float *xp = ar.f((size_t)(kChunkFrames + 2) * nm);
    dzero(xp, (size_t)(kChunkFrames + 2) * nm * 4);
    dcopy(xp + nm, mel, (size_t)kChunkFrames * nm * 4);
    float *X1 = ar.f((size_t)kChunkFrames * nm * 3);
    im2col1d(xp, nm, 3, 1, kChunkFrames, X1);
    float *hp1 = ar.f((size_t)(kChunkFrames + 2) * C);
    dzero(hp1, (size_t)C * 4);
    dzero(hp1 + (size_t)(kChunkFrames + 1) * C, (size_t)C * 4);
    matmul(X1, kChunkFrames, m.conv1, hp1 + C);
    gelu(hp1 + C, hp1 + C, (size_t)kChunkFrames * C);
    float *X2 = ar.f((size_t)T * C * 3);
    im2col1d(hp1, C, 3, 2, T, X2);
    matmul(X2, T, m.conv2, out);
    gelu(out, out, (size_t)T * C);
    ar.reset(mark);
    add(out, m.enc_pos, (size_t)T * C, out);

    float *x = out;
    float *h = ar.f((size_t)T * C), *q = ar.f((size_t)T * C), *k = ar.f((size_t)T * C), *v = ar.f((size_t)T * C);
    float *qh = ar.f((size_t)T * C), *kh = ar.f((size_t)T * C), *vt = ar.f((size_t)T * C), *oh = ar.f((size_t)T * C);
    float *S = ar.f((size_t)T * T), *f1 = ar.f((size_t)T * m.enc[0].fc1.N);
    float scale = 1.f / std::sqrt((float)D);
    for (const EncLayer &l : m.enc) {
        layernorm(x, T, C, l.attn_ln_w, l.attn_ln_b, kLnEps, h);
        matmul(h, T, l.q, q);
        matmul(h, T, l.k, k);
        matmul(h, T, l.v, v);
        permute3(q, T, H, D, 1, 0, 2, qh);  // [H][T][D]
        permute3(k, T, H, D, 1, 0, 2, kh);  // [H][T][D]
        permute3(v, T, H, D, 1, 2, 0, vt);  // [H][D][T]
        for (int hh = 0; hh < H; hh++) {
            size_t off = (size_t)hh * T * D;
            matmul(qh + off, T, view(kh + off, T, D), S);
            softmax_rows(S, T, T, scale);
            matmul(S, T, view(vt + off, D, T), oh + off);
        }
        permute3(oh, H, T, D, 1, 0, 2, h);  // back to [T][H][D]
        matmul(h, T, l.o, x, true);
        layernorm(x, T, C, l.mlp_ln_w, l.mlp_ln_b, kLnEps, h);
        matmul(h, T, l.fc1, f1);
        gelu(f1, f1, (size_t)T * l.fc1.N);
        matmul(f1, T, l.fc2, x, true);
    }
    layernorm(x, T, C, m.ln_post_w, m.ln_post_b, kLnEps, x);
    ar.reset(mark);
}

// ---------------- decoder ----------------

struct Decoder {
    const Model &m;
    KvCache self, cross;
    int *zeros_d = nullptr, *pos_d = nullptr, *cross_pos_d = nullptr;

    explicit Decoder(const Model &model) : m(model) {
        const Hparams &hp = m.hp;
        self.init(1, hp.n_text_layer, hp.n_text_ctx, hp.n_text_state);
        cross.init(1, hp.n_text_layer, hp.n_audio_ctx, hp.n_text_state);
        int n = std::max(hp.n_text_ctx, hp.n_audio_ctx);
        std::vector<int> z(n, 0), p(n), last(n, hp.n_audio_ctx - 1);
        for (int i = 0; i < n; i++) p[i] = i;
        zeros_d = (int *)dmalloc(n * 4);
        pos_d = (int *)dmalloc(n * 4);
        cross_pos_d = (int *)dmalloc(n * 4);
        upload(zeros_d, z.data(), n * 4);
        upload(pos_d, p.data(), n * 4);
        upload(cross_pos_d, last.data(), n * 4);
    }

    // Cross-attention keys/values from the encoder output, once per window.
    void set_audio(Arena &ar, const float *enc) {
        int T = m.hp.n_audio_ctx, C = m.hp.n_text_state;
        size_t mark = ar.mark();
        float *k = ar.f((size_t)T * C), *v = ar.f((size_t)T * C);
        for (int i = 0; i < (int)m.dec.size(); i++) {
            matmul(enc, T, m.dec[i].ck, k);
            matmul(enc, T, m.dec[i].cv, v);
            kv_store(cross, i, k, v, T, zeros_d, pos_d);
        }
        ar.reset(mark);
    }

    // Run M tokens at positions [p0, p0+M); writes final-norm hidden states [M][C] to out.
    void forward(Arena &ar, const int *ids_d, int M, int p0, float *out) {
        const Hparams &hp = m.hp;
        int C = hp.n_text_state, H = hp.n_text_head, D = C / H;
        size_t mark = ar.mark();
        float *x = ar.f((size_t)M * C), *h = ar.f((size_t)M * C), *q = ar.f((size_t)M * C), *k = ar.f((size_t)M * C);
        float *v = ar.f((size_t)M * C), *a = ar.f((size_t)M * C), *f1 = ar.f((size_t)M * m.dec[0].fc1.N);
        embed_rows(m.tok_emb, ids_d, M, x);
        add(x, m.dec_pos + (size_t)p0 * C, (size_t)M * C, x);
        for (int i = 0; i < (int)m.dec.size(); i++) {
            const DecLayer &l = m.dec[i];
            layernorm(x, M, C, l.attn_ln_w, l.attn_ln_b, kLnEps, h);
            matmul(h, M, l.q, q);
            matmul(h, M, l.k, k);
            matmul(h, M, l.v, v);
            kv_store(self, i, k, v, M, zeros_d, pos_d + p0);
            attn_cached(q, self, i, M, H, H, D, zeros_d, pos_d + p0, p0 + M, a);
            matmul(a, M, l.o, x, true);
            layernorm(x, M, C, l.cross_ln_w, l.cross_ln_b, kLnEps, h);
            matmul(h, M, l.cq, q);
            attn_cached(q, cross, i, M, H, H, D, zeros_d, cross_pos_d, hp.n_audio_ctx, a);
            matmul(a, M, l.co, x, true);
            layernorm(x, M, C, l.mlp_ln_w, l.mlp_ln_b, kLnEps, h);
            matmul(h, M, l.fc1, f1);
            gelu(f1, f1, (size_t)M * l.fc1.N);
            matmul(f1, M, l.fc2, x, true);
        }
        layernorm(x, M, C, m.dec_ln_w, m.dec_ln_b, kLnEps, out);
        ar.reset(mark);
    }
};

// Drops a trailing loop ("a b c a b c a b c ...") down to a single copy; returns true if one was found.
bool trim_repeats(std::vector<int> &ids) {
    int n = (int)ids.size();
    for (int p = 1; p <= 24 && p * 4 <= n; p++) {
        int reps = 1;
        while ((reps + 1) * p <= n &&
               std::equal(ids.end() - p, ids.end(), ids.end() - (reps + 1) * p))
            reps++;
        if (reps >= (p == 1 ? 8 : 4)) {
            ids.resize(n - (reps - 1) * p);
            return true;
        }
    }
    return false;
}

}  // namespace

bool is_whisper_model(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t magic = 0;
    size_t got = fread(&magic, 4, 1, f);
    fclose(f);
    return got == 1 && magic == 0x67676d6c;
}

class WhisperBackend final : public SttBackend {
    GgmlBin file; Hparams hp; Tokens tk; Model m; std::unique_ptr<Decoder> dec; Arena ar;
    std::vector<char> banned; int space=-1;
public:
    explicit WhisperBackend(const SttOptions& o):file(o.model),hp(file.hp),tk(hp.n_vocab){
        double t0=now_s(); progress("load",0,1);
    if (hp.n_audio_ctx * 2 != kChunkFrames) fail("unexpected audio context %d", hp.n_audio_ctx);
    // 16-bit large-v3 is 3.1 GB: fall back to 8-bit weights when the card can't hold it plus working memory.
    // (A fresh context reports ~4080 MiB free on the 4 GB RX 550, but only ~3.7 GB of it can be allocated.)
    size_t need = file.weight_bytes() + (size_t)hp.n_text_layer * hp.n_audio_ctx * hp.n_text_state * 4 + (1100u << 20);
    bool q8 = o.q8 || need > vram_free();
    if (q8 && !o.q8) logf("INFO model needs %.0f MiB, %.0f MiB free: using 8-bit weights", need / 1048576.0,
                          vram_free() / 1048576.0);
    m.load(file, q8);
    dec=std::make_unique<Decoder>(m);
    ar.init((size_t)hp.n_audio_ctx * hp.n_audio_state * 4 * 12 + (size_t)hp.n_audio_ctx * hp.n_audio_ctx * 4 +
            (size_t)kChunkFrames * hp.n_audio_state * 4 * 2 + (64u << 20));
    progress("load", 1, 1);
    logf("INFO loaded %s in %.1fs (%d+%d layers, %s), %.0f MiB VRAM free", o.model.c_str(), now_s() - t0,
         hp.n_audio_layer, hp.n_text_layer, tk.multilingual ? "multilingual" : "English-only",
         vram_free() / 1048576.0);

    // tokens never sampled: everything special except end-of-text, plus non-speech symbols
    banned.assign(hp.n_vocab,0);
    for (int i = tk.eot + 1; i < hp.n_vocab; i++) banned[i] = 1;
    {
        std::map<std::string, int> by_text;
        for (int i = 0; i < (int)file.vocab.size() && i < tk.eot; i++) by_text.emplace(file.vocab[i], i);
        for (const char *s : kNonSpeech)
            for (std::string t : {std::string(s), " " + std::string(s)}) {
                auto it = by_text.find(t);
                if (it != by_text.end()) banned[it->second] = 1;
            }
        auto dash = by_text.find(" -");
        if (dash != by_text.end()) banned[dash->second] = 1;
        auto quote = by_text.find(" '");
        if (quote != by_text.end()) banned[quote->second] = 1;
    }
    space=-1;
    for (int i = 0; i < (int)file.vocab.size() && i < tk.eot; i++)
        if (file.vocab[i] == " ") space = i;

    }
    std::string info() const override {
        std::string out="{\"languages\":[";
        if(!tk.multilingual)out+="\"en\"";
        else for(int i=0;i<tk.n_langs;i++){if(i)out+=",";out+="\""+std::string(kLanguages[i])+"\"";}
        return out+"],\"translate\":"+(tk.multilingual?"true":"false")+"}";
    }
    SttResult transcribe(Audio audio,const SttOptions& o) override {
        double t0=now_s();ar.reset(0);
        if(audio.sample_rate!=kSampleRate)audio.samples=resample(audio.samples,audio.sample_rate,kSampleRate);
        if(audio.samples.size()<(size_t)kSampleRate/10)fail("audio is shorter than 0.1 s");
    int lang = -1;
    if (tk.multilingual && !o.language.empty() && o.language != "auto") {
        lang = language_id(o.language);
        if (lang < 0 || lang >= tk.n_langs) fail("unknown language code '%s'", o.language.c_str());
    }

    int C = hp.n_text_state, V = hp.n_vocab;
    float *enc = ar.f((size_t)hp.n_audio_ctx * hp.n_audio_state);
    float *mel = ar.f((size_t)kChunkFrames * hp.n_mels);
    float *hid = ar.f((size_t)8 * C), *lg = ar.f((size_t)2 * V);
    int *ids_d = ar.i(hp.n_text_ctx);
    std::vector<float> logits(V);
    auto pieces = split_audio(audio.samples, (size_t)std::min(o.chunk_seconds, 30.0) * kSampleRate, kSampleRate);
    SttResult res;
    int max_new = std::min(o.max_new_tokens, hp.n_text_ctx / 2);
    for (size_t pi = 0; pi < pieces.size(); pi++) {
        progress("stt_chunk", (long)pi, (long)pieces.size());
        std::vector<float> pcm(audio.samples.begin() + pieces[pi].first, audio.samples.begin() + pieces[pi].second);
        if (pcm.size() < (size_t)kSampleRate / 10) continue;
        // Whisper may stop early (end of text after a closing timestamp); like the reference
        // decoders, continue from its last timestamp until the piece is covered
        std::string text;
        size_t seek = 0;
        for (int pass = 0; pass < 16; pass++) {
        std::vector<float> win(pcm.begin() + seek, pcm.end());
        log_mel(ar, m, win, mel);
        encode(ar, m, mel, enc);
        dec->set_audio(ar, enc);

        if (tk.multilingual && lang < 0) {
            int sot = tk.sot;
            upload(ids_d, &sot, 4);
            dec->forward(ar, ids_d, 1, 0, hid);
            matmul(hid, 1, m.tok_emb, lg);
            download(logits.data(), lg, V * 4);
            int best = 0;
            for (int i = 1; i < tk.n_langs; i++)
                if (logits[tk.lang_token(i)] > logits[tk.lang_token(best)]) best = i;
            lang = best;
            logf("INFO detected language: %s", kLanguages[lang]);
        }
        std::vector<int> prompt = {tk.sot};
        if (tk.multilingual) {
            prompt.push_back(tk.lang_token(lang));
            prompt.push_back(o.translate ? tk.translate : tk.transcribe);
        }
        // timestamp mode: Whisper marks phrase boundaries, which keeps it anchored in time
        // (large-v3-turbo skips whole sentences without it) and gives per-phrase segments
        int P = (int)prompt.size();
        upload(ids_d, prompt.data(), P * 4);
        dec->forward(ar, ids_d, P, 0, hid);

        // no-speech probability comes from the logits at the start-of-transcript position
        matmul(hid, 1, m.tok_emb, lg);
        download(logits.data(), lg, V * 4);
        float mx = *std::max_element(logits.begin(), logits.end());
        double z = 0;
        for (float v : logits) z += std::exp((double)v - mx);
        double p_nosp = std::exp((double)logits[tk.nosp] - mx) / z;

        std::vector<int> out;
        double sum_logp = 0;
        const int ts0 = tk.ts_begin();
        const float *last = hid + (size_t)(P - 1) * C;
        for (int step = 0; step < max_new && P + step < hp.n_text_ctx; step++) {
            progress("tokens",step,max_new);
            matmul(last, 1, m.tok_emb, lg);
            download(logits.data(), lg, V * 4);
            float lmx = *std::max_element(logits.begin(), logits.end());
            double lz = 0;
            for (float v : logits) lz += std::exp((double)v - lmx);
            // timestamp rules (as OpenAI's decoder applies them)
            bool last_ts = !out.empty() && out.back() >= ts0;
            bool prev_ts = out.size() < 2 || out[out.size() - 2] >= ts0;
            bool text_ok = step > 0 && !(last_ts && !prev_ts);  // a lone timestamp must be closed first
            bool ts_ok = !(last_ts && prev_ts);                   // ...and pairs are not followed by a third
            int ts_min = ts0, ts_max = V - 1;
            for (int i = (int)out.size() - 1; i >= 0; i--)
                if (out[i] >= ts0) {  // timestamps never go backwards
                    ts_min = (last_ts && !prev_ts) ? out[i] : out[i] + 1;
                    break;
                }
            if (step == 0) ts_max = ts0 + 50;  // the first phrase starts within 1 s
            if (text_ok && ts_ok) {  // when timestamps together beat every word, take a timestamp
                float tmx = -INFINITY, wmx = -INFINITY;
                for (int i = ts_min; i <= ts_max; i++) tmx = std::max(tmx, logits[i]);
                for (int i = 0; i < tk.eot; i++)
                    if (!banned[i]) wmx = std::max(wmx, logits[i]);
                double tz = 0;
                for (int i = ts_min; i <= ts_max; i++) tz += std::exp((double)logits[i] - tmx);
                if (tmx + std::log(tz) > wmx) text_ok = false;
            }
            float best_v = -INFINITY;
            int best = tk.eot;
            auto consider = [&](int i) {
                if (logits[i] > best_v) {
                    best_v = logits[i];
                    best = i;
                }
            };
            if (text_ok)
                for (int i = 0; i < tk.eot; i++)
                    if (!banned[i] && !(step == 0 && i == space)) consider(i);
            if (ts_ok)
                for (int i = ts_min; i <= ts_max; i++) consider(i);
            if (step > 0) consider(tk.eot);
            sum_logp += best_v - lmx - std::log(lz);
            if (best == tk.eot) break;
            out.push_back(best);
            if (trim_repeats(out)) break;
            upload(ids_d + P + step, &best, 4);
            dec->forward(ar, ids_d + P + step, 1, P + step, hid);
            last = hid;
        }
        int n_text = 0;
        for (int id : out) n_text += id < tk.eot;
        double avg_logp = sum_logp / std::max(1, n_text + 1);
        bool silent = p_nosp > 0.6 && avg_logp < -1.0;
        if (silent) logf("INFO chunk %zu looks like silence (no-speech %.2f, avg logprob %.2f): skipped", pi, p_nosp, avg_logp);
        // phrases between timestamps; an unclosed last phrase runs to the end of the piece
        double base = (pieces[pi].first + seek) / (double)kSampleRate, piece_end = pieces[pi].second / (double)kSampleRate;
        auto trim = [](const std::string &t) {
            size_t a = t.find_first_not_of(" \n\t"), b = t.find_last_not_of(" \n\t");
            return a == std::string::npos ? std::string() : t.substr(a, b - a + 1);
        };
        std::string phrase;
        double start = base;
        auto close = [&](double end) {
            std::string t = trim(phrase);
            phrase.clear();
            if (t.empty() || silent) return;
            res.segments.push_back({start, std::min(std::max(end, start), piece_end), t});
            if (!text.empty()) text += ' ';
            text += t;
        };
        for (int id : out) {
            if (id >= ts0) {
                double t = base + (id - ts0) * 0.02;
                if (!trim(phrase).empty()) close(t);
                start = t;
            } else if (id < (int)file.vocab.size() && id < tk.eot) {
                phrase += file.vocab[id];
            }
        }
        close(piece_end);
        int last_off = 0;  // samples from the window start to the last timestamp
        for (int id : out)
            if (id >= ts0) last_off = (id - ts0) * kSampleRate / 50;
        bool ended_closed = !out.empty() && out.back() >= ts0;
        if (silent || !ended_closed || last_off <= 0 || seek + last_off + kSampleRate > pcm.size()) break;
        seek += last_off;
        }
        if (!text.empty()) {
            if (!res.text.empty()) res.text += ' ';
            res.text += text;
        }
        fprintf(stderr, "PARTIAL %s\n", text.c_str());
    }
    progress("stt_chunk", (long)pieces.size(), (long)pieces.size());
    res.language = tk.multilingual ? (lang >= 0 ? kLanguages[lang] : "") : "en";
    logf("INFO transcribed %.1fs of audio in %.1fs", audio.samples.size() / (double)kSampleRate, now_s() - t0);
    return res;
    }
};
std::unique_ptr<SttBackend> make_whisper(const SttOptions& o){return std::make_unique<WhisperBackend>(o);}
