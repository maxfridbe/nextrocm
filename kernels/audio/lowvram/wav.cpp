#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>

#include "common.h"

Audio read_wav(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fail("cannot open %s", path.c_str());
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 12 || memcmp(d.data(), "RIFF", 4) || memcmp(d.data() + 8, "WAVE", 4)) fail("%s: not a WAV file", path.c_str());
    size_t p = 12;
    int fmt = 0, channels = 0, rate = 0, bits = 0;
    const uint8_t *data = nullptr;
    size_t data_len = 0;
    while (p + 8 <= d.size()) {
        uint32_t len;
        memcpy(&len, d.data() + p + 4, 4);
        const uint8_t *body = d.data() + p + 8;
        size_t avail = d.size() - (p + 8);
        if (!memcmp(d.data() + p, "fmt ", 4) && len >= 16) {
            uint16_t f16;
            memcpy(&f16, body, 2);
            fmt = f16;
            memcpy(&f16, body + 2, 2);
            channels = f16;
            uint32_t r;
            memcpy(&r, body + 4, 4);
            rate = (int)r;
            memcpy(&f16, body + 14, 2);
            bits = f16;
            if (fmt == 0xfffe && len >= 26) {  // WAVE_FORMAT_EXTENSIBLE: sub-format GUID starts at 24
                memcpy(&f16, body + 24, 2);
                fmt = f16;
            }
        } else if (!memcmp(d.data() + p, "data", 4)) {
            data = body;
            data_len = len > avail ? avail : len;
        }
        p += 8 + len + (len & 1);
    }
    if (!data || !channels || !rate) fail("%s: missing fmt or data chunk", path.c_str());
    int bps = bits / 8;
    size_t frames = data_len / (bps * channels);
    Audio a;
    a.sample_rate = rate;
    a.samples.resize(frames);
    for (size_t i = 0; i < frames; i++) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
            const uint8_t *s = data + (i * channels + c) * bps;
            double v;
            if (fmt == 3 && bits == 32) {
                float x;
                memcpy(&x, s, 4);
                v = x;
            } else if (fmt == 1 && bits == 16) {
                int16_t x;
                memcpy(&x, s, 2);
                v = x / 32768.0;
            } else if (fmt == 1 && bits == 24) {
                int32_t x = (s[0] << 8) | (s[1] << 16) | (s[2] << 24);
                v = (x >> 8) / 8388608.0;
            } else if (fmt == 1 && bits == 32) {
                int32_t x;
                memcpy(&x, s, 4);
                v = x / 2147483648.0;
            } else {
                fail("%s: unsupported WAV encoding (format %d, %d bits)", path.c_str(), fmt, bits);
            }
            acc += v;
        }
        a.samples[i] = (float)(acc / channels);
    }
    return a;
}

static int16_t to_s16(float v) {
    return (int16_t)std::lrint(std::fmax(-1.f, std::fmin(1.f, v)) * 32767.f);
}

void append_pcm16(FILE *f, const float *x, size_t n) {
    std::vector<int16_t> s(n);
    for (size_t i = 0; i < n; i++) s[i] = to_s16(x[i]);
    fwrite(s.data(), 2, n, f);
    fflush(f);
}

void write_wav(const std::string &path, const std::vector<float> &x, int rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) fail("cannot write %s", path.c_str());
    uint32_t data_len = (uint32_t)(x.size() * 2);
    auto u32 = [&](uint32_t v) { f.write((const char *)&v, 4); };
    auto u16 = [&](uint16_t v) { f.write((const char *)&v, 2); };
    f.write("RIFF", 4);
    u32(36 + data_len);
    f.write("WAVEfmt ", 8);
    u32(16);
    u16(1);
    u16(1);
    u32(rate);
    u32(rate * 2);
    u16(2);
    u16(16);
    f.write("data", 4);
    u32(data_len);
    for (float v : x) {
        int16_t s = to_s16(v);
        f.write((const char *)&s, 2);
    }
}

std::vector<float> resample(const std::vector<float> &x, int from, int to) {
    if (from == to || x.empty()) return x;
    size_t n = (size_t)((double)x.size() * to / from);
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) {
        double t = (double)i * from / to;
        size_t a = (size_t)t;
        double fr = t - a;
        float va = x[std::min(a, x.size() - 1)], vb = x[std::min(a + 1, x.size() - 1)];
        y[i] = (float)(va + (vb - va) * fr);
    }
    return y;
}

// Split long audio at quiet points so each piece is at most max_len samples.
std::vector<std::pair<size_t, size_t>> split_audio(const std::vector<float> &x, size_t max_len, int sr) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t start = 0, win = sr / 10;
    while (x.size() - start > max_len) {
        size_t lo = start + max_len * 2 / 3, hi = start + max_len;
        size_t best = hi;
        double best_e = 1e30;
        for (size_t s = lo; s + win <= hi; s += win / 2) {
            double e = 0;
            for (size_t i = s; i < s + win; i++) e += (double)x[i] * x[i];
            if (e < best_e) {
                best_e = e;
                best = s + win / 2;
            }
        }
        out.push_back({start, best});
        start = best;
    }
    out.push_back({start, x.size()});
    return out;
}
