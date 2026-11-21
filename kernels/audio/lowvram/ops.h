// GPU tensors and HIP kernels. Activations are always f32, row-major [rows][cols].
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gguf.h"

// ---- device memory ----

void *dmalloc(size_t bytes);
void dfree(void *p);
float *dalloc_f(size_t n);
void upload(void *dst, const void *src, size_t bytes);
void download(void *dst, const void *src, size_t bytes);
void dcopy(void *dst, const void *src, size_t bytes);
void dzero(void *dst, size_t bytes);
void dsync();
size_t vram_free();

// Bump allocator for per-step scratch buffers.
struct Arena {
    char *base = nullptr;
    size_t cap = 0, used = 0;
    void init(size_t bytes);
    void release();
    float *f(size_t n);
    int *i(size_t n);
    size_t mark() const { return used; }
    void reset(size_t m) { used = m; }
};

// ---- weights ----

enum class WT { F32, F16, BF16, Q8 };

// Linear weight W[N][K] (output-major), optional bias[N]. Q8 keeps int8 values
// [N][K] plus one f16 scale per 32 values; F16/BF16 keep 16-bit values.
struct Weight {
    WT t = WT::F32;
    int N = 0, K = 0;
    void *w = nullptr;
    void *scales = nullptr;
    float *bias = nullptr;
    size_t bytes() const;
    void free();
};

// Upload a GGUF tensor as a linear layer: ne[0..n-2] flatten into K, ne[n-1] is N.
// q8: re-quantize F32/F16/BF16 weights to Q8_0-style blocks (halves memory traffic vs 16-bit).
Weight upload_linear(const GgufTensor &t, const GgufTensor *bias = nullptr, bool q8 = false);
// ConvTranspose1d weight stored [in][out][k] uploaded as W[(out*k)][in] (bias is applied by convtr_ola).
Weight upload_convtr(const GgufTensor &t, bool q8 = false);
// Any tensor converted to f32 on the device.
float *upload_f32(const GgufTensor &t);
// Any tensor converted to f32 on the host.
std::vector<float> to_f32(const GgufTensor &t);

// ---- kernels ----

// y[M][N] = x[M][K] . W^T + bias  (accumulate: y += ...)
void matmul(const float *x, int M, const Weight &W, float *y, bool accumulate = false);
// Gather rows of a weight matrix (embedding lookup) into f32 rows.
void embed_rows(const Weight &W, const int *ids_dev, int n, float *out);

void rmsnorm(const float *x, int M, int C, const float *w, float eps, float *y);
void layernorm(const float *x, int M, int C, const float *w, const float *b, float eps, float *y);

void add(const float *a, const float *b, size_t n, float *y);               // y = a + b
void add_scaled(float *y, const float *x, float s, size_t n);               // y += s * x
void add_bias_rows(float *y, const float *b, int M, int C);                 // y[m][c] += b[c]
void residual_gamma(float *res, const float *x, const float *g, int M, int C);  // res += x * g[c]
void scale_inplace(float *x, float s, size_t n);
void silu(const float *x, float *y, size_t n);
void gelu(const float *x, float *y, size_t n);
void relu(float *x, size_t n);
void swiglu(const float *gate, const float *up, float *y, size_t n);        // y = silu(gate) * up
void glu(const float *x, int M, int C, float *y);                           // y = x[:, :C] * sigmoid(x[:, C:])
void affine_cols(float *x, const float *s, const float *b, int M, int C);   // x = x * s[c] + b[c]
// y = x * (1 + scale) + shift, where shift/scale rows have stride ld.
void modulate(const float *x, const float *shift, const float *scale, int ld, int M, int C, float *y);
// x += gate * y with gate rows of stride ld.
void gated_add(float *x, const float *gate, int ld, const float *y, int M, int C);

// NeoX-style rotary embedding on x[M][H][D] using positions pos[M] (device).
void rope_neox(float *x, int M, int H, int D, const int *pos, float theta);

// KV cache: half [n_seq][n_layer][max_ctx][HD]
struct KvCache {
    uint16_t *k = nullptr, *v = nullptr;
    int n_seq = 0, n_layer = 0, max_ctx = 0, HD = 0;
    void init(int n_seq, int n_layer, int max_ctx, int HD);
    void free();
};
void kv_store(KvCache &kv, int layer, const float *k, const float *v, int M, const int *row_seq, const int *row_pos);
// Causal attention against the cache. q[M][Hq][D] -> out[M][Hq][D]. Row m sees cache positions [0, row_pos[m]].
void attn_cached(const float *q, const KvCache &kv, int layer, int M, int Hq, int Hkv, int D, const int *row_seq,
                 const int *row_pos, int max_keys, float *out);
// FastConformer relative-position attention (Transformer-XL), no cache.
// q,k,v [T][H][D]; p [2T-1][H][D]; u,vb [H][D].
void attn_relpos(const float *q, const float *k, const float *v, const float *p, const float *u, const float *vb,
                 int T, int H, int D, float *out);

// Depthwise conv over time on x[T_in][C]: out[t][c] = b[c] + sum_j w[c][j] * x[t + j - pad_left][c]
// (zero outside [0, T_in)), for t in [0, T_out).
void dwconv1d(const float *x, int T_in, int C, const float *w, const float *b, int k, int pad_left, int T_out,
              float *out);
// im2col: X[t][c*k + j] = x[t*stride + j][c], for t in [0, T_out).
void im2col1d(const float *x, int C, int k, int stride, int T_out, float *X);
// Overlap-add for ConvTranspose1d with kernel 2*stride: Y[T][Cout*K] -> out[T*stride][Cout].
// prev (nullable) is the Y row of the frame before Y[0].
void convtr_ola(const float *Y, const float *prev, int T, int Cout, int stride, const float *b, float *out);

// Pre-encode 2D convs, channels-last layout [H][W][C], kernel 3x3, stride 2, pad 1.
void conv2d_s2_c1(const float *x, int H, int W, const float *w, const float *b, int Cout, float *out);
void dwconv2d_s2(const float *x, int H, int W, int C, const float *w, const float *b, float *out);
// [A][B][C] -> [A][C][B]
void transpose12(const float *x, int A, int B, int C, float *y);
// y = x[d0][d1][d2] with axes reordered: output axis i is input axis p_i (e.g. 1,0,2 swaps the first two).
void permute3(const float *x, int d0, int d1, int d2, int p0, int p1, int p2, float *y);
// Row-wise softmax(x * scale) in place on x[M][N].
void softmax_rows(float *x, int M, int N, float scale);
// spec[T][2*nb] = (re | im) -> p[T][nb] = re^2 + im^2
void power_spec(const float *spec, int T, int nb, float *p);
