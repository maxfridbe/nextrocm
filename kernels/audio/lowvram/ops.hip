// HIP kernels for the vibe-audio engine (written for gfx803 / RX 550, wave64).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common.h"
#include "ops.h"
#include "bridge.h"

namespace {

#ifndef NR_LOWVRAM_WAVE_SIZE
#error "build must specify the target wavefront size"
#endif
constexpr int kWave = NR_LOWVRAM_WAVE_SIZE;
static_assert(kWave == 32 || kWave == 64);

// Native clang vectors: HIP's float4/uint4 structs get split into dword loads on gfx803.
typedef float vf4 __attribute__((ext_vector_type(4)));
typedef uint32_t vu4 __attribute__((ext_vector_type(4)));
typedef uint32_t vu2 __attribute__((ext_vector_type(2)));

// ---- host-side half / bfloat conversions ----

float h2f_host(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) {
                man <<= 1;
                exp--;
            }
            man &= 0x3ff;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

float bf2f_host(uint16_t b) {
    uint32_t bits = (uint32_t)b << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ---- device helpers ----

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half(f)); }
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t)b << 16); }

template <WT T>
__device__ __forceinline__ float loadw(const void *w, const void *sc, int K, int n, int k) {
    size_t idx = (size_t)n * K + k;
    if constexpr (T == WT::F32) {
        return ((const float *)w)[idx];
    } else if constexpr (T == WT::F16) {
        return h2f(((const uint16_t *)w)[idx]);
    } else if constexpr (T == WT::BF16) {
        return bf2f(((const uint16_t *)w)[idx]);
    } else {
        return (float)((const int8_t *)w)[idx] * h2f(((const uint16_t *)sc)[idx >> 5]);
    }
}

// Four consecutive weights starting at k (k % 4 == 0, K % 4 == 0).
template <WT T>
__device__ __forceinline__ void load4(const void *w, const void *sc, int K, int n, int k, float *o) {
    size_t idx = (size_t)n * K + k;
    if constexpr (T == WT::F32) {
        vf4 a = *(const vf4 *)((const float *)w + idx);
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
    } else if constexpr (T == WT::F16 || T == WT::BF16) {
        vu2 v = *(const vu2 *)((const uint16_t *)w + idx);
        uint32_t u[2] = {v.x, v.y};
#pragma unroll
        for (int i = 0; i < 2; i++) {
            if constexpr (T == WT::F16) {
                o[2 * i] = h2f(u[i] & 0xffff);
                o[2 * i + 1] = h2f(u[i] >> 16);
            } else {
                o[2 * i] = bf2f(u[i] & 0xffff);
                o[2 * i + 1] = bf2f(u[i] >> 16);
            }
        }
    } else {
        uint32_t v = *(const uint32_t *)((const int8_t *)w + idx);
        float s = h2f(((const uint16_t *)sc)[idx >> 5]);
#pragma unroll
        for (int j = 0; j < 4; j++) o[j] = (float)(int8_t)((v >> (8 * j)) & 0xff) * s;
    }
}

// Eight consecutive weights starting at k (k % 8 == 0, K % 8 == 0).
template <WT T>
__device__ __forceinline__ void load8(const void *w, const void *sc, int K, int n, int k, float *o) {
    size_t idx = (size_t)n * K + k;
    if constexpr (T == WT::F32) {
        const vf4 *p = (const vf4 *)((const float *)w + idx);
        vf4 a = p[0], b = p[1];
        o[0] = a.x; o[1] = a.y; o[2] = a.z; o[3] = a.w;
        o[4] = b.x; o[5] = b.y; o[6] = b.z; o[7] = b.w;
    } else if constexpr (T == WT::F16 || T == WT::BF16) {
        vu4 v = *(const vu4 *)((const uint16_t *)w + idx);
        uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint16_t lo = u[i] & 0xffff, hi = u[i] >> 16;
            if constexpr (T == WT::F16) {
                o[2 * i] = h2f(lo);
                o[2 * i + 1] = h2f(hi);
            } else {
                o[2 * i] = bf2f(lo);
                o[2 * i + 1] = bf2f(hi);
            }
        }
    } else {
        vu2 v = *(const vu2 *)((const int8_t *)w + idx);
        float s = h2f(((const uint16_t *)sc)[idx >> 5]);
        uint32_t u[2] = {v.x, v.y};
#pragma unroll
        for (int i = 0; i < 2; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) o[4 * i + j] = (float)(int8_t)((u[i] >> (8 * j)) & 0xff) * s;
        }
    }
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
#pragma unroll
    for (int off = kWave / 2; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor(v, off, kWave));
    return v;
}

// Block-wide reduction for blockDim.x a multiple of kWave (<= 1024). red needs 32 floats.
__device__ float block_sum(float v, float *red) {
    v = wave_sum(v);
    int lane = threadIdx.x % kWave, wid = threadIdx.x / kWave, nw = blockDim.x / kWave;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = 0.f;
    for (int i = 0; i < nw; i++) t += red[i];
    return t;
}

__device__ float block_max(float v, float *red) {
    v = wave_max(v);
    int lane = threadIdx.x % kWave, wid = threadIdx.x / kWave, nw = blockDim.x / kWave;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < nw; i++) t = fmaxf(t, red[i]);
    return t;
}

// ---- matmul ----

// A wave is split into groups of LPR lanes; each group computes NR consecutive outputs for MR input
// rows. Activations are loaded once per k-slice and reused across the NR weight rows. LPR is picked
// per K so every lane loops a few times and the final shuffle reduction stays short.
template <WT T, int MR, int NR, int LPR, int V>
__global__ __launch_bounds__(256) void gemv_kernel(const float *__restrict__ x, int K, const void *__restrict__ w,
                                                   const void *__restrict__ sc, const float *__restrict__ bias,
                                                   float *__restrict__ y, int N, int accumulate) {
    constexpr int kGroups = kWave / LPR;
    int lane = threadIdx.x % kWave;
    int l = lane % LPR;
    int wave = blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave;
    int n0 = (wave * kGroups + lane / LPR) * NR;
    if (n0 >= N) return;  // whole groups leave together; shuffles stay inside a group
    float acc[MR][NR];
#pragma unroll
    for (int m = 0; m < MR; m++)
#pragma unroll
        for (int r = 0; r < NR; r++) acc[m][r] = 0.f;
    for (int k = l * V; k < K; k += LPR * V) {
        float xv[MR][V];
#pragma unroll
        for (int m = 0; m < MR; m++) {
            const vf4 *xp = (const vf4 *)(x + (size_t)m * K + k);
#pragma unroll
            for (int q = 0; q < V / 4; q++) {
                vf4 a = xp[q];
                xv[m][4 * q] = a.x; xv[m][4 * q + 1] = a.y; xv[m][4 * q + 2] = a.z; xv[m][4 * q + 3] = a.w;
            }
        }
        if constexpr (V == 16 && T == WT::Q8) {
#pragma unroll
            for (int r = 0; r < NR; r++) {
                if (n0 + r >= N) break;
                size_t idx = (size_t)(n0 + r) * K + k;
                vu4 q = *(const vu4 *)((const int8_t *)w + idx);
                float scale = h2f(((const uint16_t *)sc)[idx >> 5]);
                uint32_t u[4] = {q.x, q.y, q.z, q.w};
                float wv[16];
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) wv[4 * i + j] = (float)(int8_t)((u[i] >> (8 * j)) & 0xff);
#pragma unroll
                for (int m = 0; m < MR; m++) {
                    float d = 0.f;
#pragma unroll
                    for (int j = 0; j < 16; j++) d += xv[m][j] * wv[j];
                    acc[m][r] += scale * d;
                }
            }
        } else {
#pragma unroll
            for (int r = 0; r < NR; r++) {
                if (n0 + r >= N) break;
                float wv[V];
                load8<T>(w, sc, K, n0 + r, k, wv);
#pragma unroll
                for (int m = 0; m < MR; m++) {
                    float d = 0.f;
#pragma unroll
                    for (int j = 0; j < V; j++) d += xv[m][j] * wv[j];
                    acc[m][r] += d;
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MR; m++) {
#pragma unroll
        for (int r = 0; r < NR; r++) {
            float v = acc[m][r];
#pragma unroll
            for (int off = LPR / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
            if (l == 0 && n0 + r < N) {
                if (bias) v += bias[n0 + r];
                float *dst = y + (size_t)m * N + n0 + r;
                *dst = accumulate ? *dst + v : v;
            }
        }
    }
}

// 64x64 output tiles, 16-deep K slices, 4x4 outputs per thread. Handles any M, N, K.
template <WT T>
__global__ __launch_bounds__(256) void gemm_kernel(const float *__restrict__ x, int M, int K,
                                                   const void *__restrict__ w, const void *__restrict__ sc,
                                                   const float *__restrict__ bias, float *__restrict__ y, int N,
                                                   int accumulate) {
    __shared__ vf4 As4[16][17];  // [k][m/4], padded
    __shared__ vf4 Bs4[16][17];  // [k][n/4]
    float *As = (float *)As4;
    float *Bs = (float *)Bs4;
    const int stride = 17 * 4;
    int tid = threadIdx.x;
    int tx = tid % 16, ty = tid / 16;
    int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64;
    float c[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) c[i][j] = 0.f;

    int lr = tid / 4, lk = (tid % 4) * 4;
    bool vec_ok = (K % 4) == 0;
    for (int k0 = 0; k0 < K; k0 += 16) {
        {
            int gm = m0 + lr, gk = k0 + lk;
            if (gm < M && vec_ok && gk + 3 < K) {
                vf4 a = *(const vf4 *)(x + (size_t)gm * K + gk);
                As[(lk + 0) * stride + lr] = a.x;
                As[(lk + 1) * stride + lr] = a.y;
                As[(lk + 2) * stride + lr] = a.z;
                As[(lk + 3) * stride + lr] = a.w;
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    As[(lk + j) * stride + lr] = (gm < M && gk + j < K) ? x[(size_t)gm * K + gk + j] : 0.f;
            }
            int gn = n0 + lr;
            if (gn < N && vec_ok && gk + 3 < K) {
                float b4[4];
                load4<T>(w, sc, K, gn, gk, b4);
#pragma unroll
                for (int j = 0; j < 4; j++) Bs[(lk + j) * stride + lr] = b4[j];
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++)
                    Bs[(lk + j) * stride + lr] = (gn < N && gk + j < K) ? loadw<T>(w, sc, K, gn, gk + j) : 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            vf4 a = As4[kk][ty];
            vf4 b = Bs4[kk][tx];
            float av[4] = {a.x, a.y, a.z, a.w};
            float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) c[i][j] += av[i] * bv[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int gm = m0 + ty * 4 + i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int gn = n0 + tx * 4 + j;
            if (gn >= N) continue;
            float v = c[i][j] + (bias ? bias[gn] : 0.f);
            float *dst = y + (size_t)gm * N + gn;
            *dst = accumulate ? *dst + v : v;
        }
    }
}

template <WT T, int LPR, int V, int NR = 4>
void gemv_launch(const float *x, int M, const Weight &W, float *y, bool acc) {
    // 16-wide Q8 slices hold twice the activations in registers: at most 4 rows per pass
    constexpr int MRMAX = V == 16 ? 4 : 8;
    int N = W.N, K = W.K;
    int rows_per_block = (256 / LPR) * NR;
    dim3 grid((N + rows_per_block - 1) / rows_per_block);
    for (int m0 = 0; m0 < M; m0 += MRMAX) {
        int mr = std::min(MRMAX, M - m0);
        const float *xs = x + (size_t)m0 * K;
        float *ys = y + (size_t)m0 * N;
        switch (mr) {
#define GEMV_CASE(R)                                                                                          \
    case R:                                                                                                   \
        if constexpr (R <= MRMAX)                                                                             \
            hipLaunchKernelGGL((gemv_kernel<T, R, NR, LPR, V>), grid, dim3(256), 0, 0, xs, K, W.w, W.scales, \
                               W.bias, ys, N, acc);                                                           \
        break;
            GEMV_CASE(1) GEMV_CASE(2) GEMV_CASE(3) GEMV_CASE(4) GEMV_CASE(5) GEMV_CASE(6) GEMV_CASE(7) GEMV_CASE(8)
#undef GEMV_CASE
        }
    }
}

template <WT T>
void matmul_t(const float *x, int M, const Weight &W, float *y, bool acc) {
    int N = W.N, K = W.K;
    if (M <= 24 && K % 8 == 0) {
        // Lanes per row: enough that each lane loops a few times; 4 outputs per lane group.
        // (Measured on the RX 550: other splits won single shapes but lost end to end.)
#define PICK_NR(LPR, V) gemv_launch<T, (LPR < kWave ? LPR : kWave), V, 4>(x, M, W, y, acc)
        bool done = false;
        if constexpr (T == WT::Q8) {
            if (M == 1 && K % 16 == 0) {  // 16-byte int8 slices: faster for single rows only
                if (K >= 64 * 48) PICK_NR(64, 16);
                else if (K >= 32 * 48) PICK_NR(32, 16);
                else PICK_NR(16, 16);
                done = true;
            }
        }
        if (!done) {
            if (K >= 64 * 32) PICK_NR(64, 8);
            else if (K >= 32 * 32) PICK_NR(32, 8);
            else if (K >= 16 * 32) PICK_NR(16, 8);
            else PICK_NR(8, 8);
        }
#undef PICK_NR
    } else {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        hipLaunchKernelGGL((gemm_kernel<T>), grid, dim3(256), 0, 0, x, M, K, W.w, W.scales, W.bias, y, N, acc);
    }
    HIP_CHECK(hipGetLastError());
}

template <WT T>
__global__ void embed_kernel(const void *w, const void *sc, int K, const int *ids, float *out) {
    int r = blockIdx.x;
    int id = ids[r];
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[(size_t)r * K + k] = loadw<T>(w, sc, K, id, k);
}

// ---- norms / elementwise ----

__global__ void rmsnorm_kernel(const float *x, int C, const float *w, float eps, float *y) {
    __shared__ float red[32];
    const float *xr = x + (size_t)blockIdx.x * C;
    float *yr = y + (size_t)blockIdx.x * C;
    float ss = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) ss += xr[c] * xr[c];
    ss = block_sum(ss, red);
    float inv = rsqrtf(ss / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = xr[c] * inv * (w ? w[c] : 1.f);
}

__global__ void layernorm_kernel(const float *x, int C, const float *w, const float *b, float eps, float *y) {
    __shared__ float red[32];
    const float *xr = x + (size_t)blockIdx.x * C;
    float *yr = y + (size_t)blockIdx.x * C;
    float s = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) s += xr[c];
    float mean = block_sum(s, red) / C;
    float v = 0.f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = xr[c] - mean;
        v += d * d;
    }
    float inv = rsqrtf(block_sum(v, red) / C + eps);
    for (int c = threadIdx.x; c < C; c += blockDim.x) yr[c] = (xr[c] - mean) * inv * w[c] + (b ? b[c] : 0.f);
}

#define ELEMWISE(name, params, body)                                               \
    __global__ void name##_kernel params {                                         \
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;                  \
        if (i >= n) return;                                                        \
        body;                                                                      \
    }

ELEMWISE(add, (const float *a, const float *b, size_t n, float *y), y[i] = a[i] + b[i])
ELEMWISE(add_scaled, (float *y, const float *x, float s, size_t n), y[i] += s * x[i])
ELEMWISE(scale, (float *x, float s, size_t n), x[i] *= s)
ELEMWISE(silu, (const float *x, float *y, size_t n), { float v = x[i]; y[i] = v / (1.f + expf(-v)); })
ELEMWISE(gelu, (const float *x, float *y, size_t n),
         { float v = x[i]; y[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f)); })
ELEMWISE(relu, (float *x, size_t n), x[i] = fmaxf(x[i], 0.f))
ELEMWISE(swiglu, (const float *g, const float *u, float *y, size_t n),
         { float v = g[i]; y[i] = v / (1.f + expf(-v)) * u[i]; })

__global__ void add_bias_rows_kernel(float *y, const float *b, int C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % C];
}

__global__ void residual_gamma_kernel(float *res, const float *x, const float *g, int C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] += x[i] * g[i % C];
}

__global__ void affine_cols_kernel(float *x, const float *s, const float *b, int C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * s[i % C] + b[i % C];
}

__global__ void glu_kernel(const float *x, int C, size_t n, float *y) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t m = i / C, c = i % C;
    float a = x[m * 2 * C + c], g = x[m * 2 * C + C + c];
    y[i] = a / (1.f + expf(-g));
}

__global__ void modulate_kernel(const float *x, const float *shift, const float *scale, int ld, int C, size_t n,
                                float *y) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t m = i / C, c = i % C;
    y[i] = x[i] * (1.f + scale[m * ld + c]) + shift[m * ld + c];
}

__global__ void gated_add_kernel(float *x, const float *gate, int ld, const float *y, int C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t m = i / C, c = i % C;
    x[i] += gate[m * ld + c] * y[i];
}

__global__ void rope_neox_kernel(float *x, int H, int D, const int *pos, float theta) {
    int row = blockIdx.x;  // m * H + h
    int m = row / H;
    int half = D / 2;
    float p = (float)pos[m];
    float *xr = x + (size_t)row * D;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float inv = 1.f / powf(theta, (float)(2 * i) / (float)D);
        float s, c;
        sincosf(p * inv, &s, &c);
        float a = xr[i], b = xr[i + half];
        xr[i] = a * c - b * s;
        xr[i + half] = b * c + a * s;
    }
}

// ---- attention ----

__global__ void kv_store_kernel(uint16_t *kc, uint16_t *vc, int layer, int n_layer, int max_ctx, int HD,
                                const float *k, const float *v, const int *row_seq, const int *row_pos) {
    int m = blockIdx.x;
    size_t base = ((size_t)(row_seq[m] * n_layer + layer) * max_ctx + row_pos[m]) * HD;
    for (int j = threadIdx.x; j < HD; j += blockDim.x) {
        kc[base + j] = f2h(k[(size_t)m * HD + j]);
        vc[base + j] = f2h(v[(size_t)m * HD + j]);
    }
}

__global__ __launch_bounds__(128) void attn_cached_kernel(const float *q, const uint16_t *kc, const uint16_t *vc,
                                                          int layer, int n_layer, int max_ctx, int Hq, int Hkv,
                                                          int D, const int *row_seq, const int *row_pos,
                                                          float scale, float *out) {
    extern __shared__ float sm[];
    __shared__ float red[32];
    int h = blockIdx.x, m = blockIdx.y;
    int hk = h / (Hq / Hkv);
    int HD = Hkv * D;
    int nk = row_pos[m] + 1;
    size_t base = ((size_t)(row_seq[m] * n_layer + layer) * max_ctx) * HD + (size_t)hk * D;
    float *qs = sm;
    float *sc = sm + D;
    const float *qr = q + ((size_t)m * Hq + h) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) qs[d] = qr[d];
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        const vu4 *kr = (const vu4 *)(kc + base + (size_t)j * HD);
        float s = 0.f;
        for (int d8 = 0; d8 < D / 8; d8++) {
            vu4 v = kr[d8];
            uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s += qs[d8 * 8 + 2 * i] * h2f(u[i] & 0xffff) + qs[d8 * 8 + 2 * i + 1] * h2f(u[i] >> 16);
            }
        }
        s *= scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    float inv = 1.f / sum;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        const uint16_t *vr = vc + base + d;
        for (int j = 0; j < nk; j++) acc += sc[j] * h2f(vr[(size_t)j * HD]);
        out[((size_t)m * Hq + h) * D + d] = acc * inv;
    }
}

__global__ __launch_bounds__(128) void attn_relpos_kernel(const float *q, const float *k, const float *v,
                                                          const float *p, const float *u, const float *vb, int T,
                                                          int H, int D, float scale, float *out) {
    extern __shared__ float sm[];
    __shared__ float red[32];
    int h = blockIdx.x, i = blockIdx.y;
    int HD = H * D;
    float *qu = sm;
    float *qv = sm + D;
    float *sc = sm + 2 * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float qq = q[(size_t)i * HD + h * D + d];
        qu[d] = qq + u[h * D + d];
        qv[d] = qq + vb[h * D + d];
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        const vf4 *kr = (const vf4 *)(k + (size_t)j * HD + h * D);
        const vf4 *pr = (const vf4 *)(p + (size_t)(T - 1 - i + j) * HD + h * D);
        float s = 0.f;
        for (int d4 = 0; d4 < D / 4; d4++) {
            vf4 a = kr[d4], b = pr[d4];
            s += qu[4 * d4] * a.x + qu[4 * d4 + 1] * a.y + qu[4 * d4 + 2] * a.z + qu[4 * d4 + 3] * a.w;
            s += qv[4 * d4] * b.x + qv[4 * d4 + 1] * b.y + qv[4 * d4 + 2] * b.z + qv[4 * d4 + 3] * b.w;
        }
        s *= scale;
        sc[j] = s;
        mx = fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float e = expf(sc[j] - mx);
        sc[j] = e;
        sum += e;
    }
    sum = block_sum(sum, red);
    float inv = 1.f / sum;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < T; j++) acc += sc[j] * v[(size_t)j * HD + h * D + d];
        out[(size_t)i * HD + h * D + d] = acc * inv;
    }
}

// ---- convolutions ----

__global__ void dwconv1d_kernel(const float *x, int T_in, int C, const float *w, const float *b, int k, int pad_left,
                                int T_out, float *out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T_out * C) return;
    int t = idx / C, c = idx % C;
    float acc = b ? b[c] : 0.f;
    for (int j = 0; j < k; j++) {
        int ti = t + j - pad_left;
        if (ti >= 0 && ti < T_in) acc += w[c * k + j] * x[(size_t)ti * C + c];
    }
    out[idx] = acc;
}

__global__ void im2col1d_kernel(const float *x, int C, int k, int stride, int T_out, float *X) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = (size_t)C * k;
    if (idx >= (size_t)T_out * row) return;
    int t = idx / row;
    int r = idx % row;
    int c = r / k, j = r % k;
    X[idx] = x[(size_t)(t * stride + j) * C + c];
}

__global__ void convtr_ola_kernel(const float *Y, const float *prev, int T, int Cout, int s, const float *b,
                                  float *out) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)T * s * Cout) return;
    int t = idx / Cout, o = idx % Cout;
    int i = t / s, r = t % s;
    int K = 2 * s;
    size_t CK = (size_t)Cout * K;
    float v = (b ? b[o] : 0.f) + Y[i * CK + o * K + r];
    if (i > 0) v += Y[(i - 1) * CK + o * K + r + s];
    else if (prev) v += prev[o * K + r + s];
    out[idx] = v;
}

__global__ void conv2d_s2_c1_kernel(const float *x, int H, int W, const float *w, const float *b, int Cout,
                                    float *out) {
    int H2 = (H - 1) / 2 + 1, W2 = (W - 1) / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H2 * W2 * Cout) return;
    int o = idx % Cout;
    int px = (idx / Cout) % W2, py = idx / Cout / W2;
    float acc = b[o];
    for (int ky = 0; ky < 3; ky++) {
        int iy = 2 * py - 1 + ky;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 3; kx++) {
            int ix = 2 * px - 1 + kx;
            if (ix < 0 || ix >= W) continue;
            acc += w[o * 9 + ky * 3 + kx] * x[(size_t)iy * W + ix];
        }
    }
    out[idx] = acc;
}

__global__ void dwconv2d_s2_kernel(const float *x, int H, int W, int C, const float *w, const float *b, float *out) {
    int H2 = (H - 1) / 2 + 1, W2 = (W - 1) / 2 + 1;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)H2 * W2 * C) return;
    int c = idx % C;
    int px = (idx / C) % W2, py = idx / C / W2;
    float acc = b[c];
    for (int ky = 0; ky < 3; ky++) {
        int iy = 2 * py - 1 + ky;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 3; kx++) {
            int ix = 2 * px - 1 + kx;
            if (ix < 0 || ix >= W) continue;
            acc += w[c * 9 + ky * 3 + kx] * x[((size_t)iy * W + ix) * C + c];
        }
    }
    out[idx] = acc;
}

__global__ void transpose12_kernel(const float *x, int A, int B, int C, float *y) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)A * B * C) return;
    int c = idx % C;
    int b = (idx / C) % B;
    int a = idx / C / B;
    y[((size_t)a * C + c) * B + b] = x[idx];
}

// y = x permuted: x is [d0][d1][d2]; output axis i is input axis p_i.
__global__ void permute3_kernel(const float *x, int d0, int d1, int d2, int p0, int p1, int p2, float *y) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = (size_t)d0 * d1 * d2;
    if (idx >= n) return;
    int dims[3] = {d0, d1, d2};
    int o2 = idx % dims[p2];
    int o1 = (idx / dims[p2]) % dims[p1];
    int o0 = idx / dims[p2] / dims[p1];
    int in[3];
    in[p0] = o0;
    in[p1] = o1;
    in[p2] = o2;
    y[idx] = x[((size_t)in[0] * d1 + in[1]) * d2 + in[2]];
}

__global__ void softmax_rows_kernel(float *x, int N, float scale) {
    __shared__ float red[32];
    float *r = x + (size_t)blockIdx.x * N;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, r[j] * scale);
    mx = block_max(mx, red);
    float s = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float e = expf(r[j] * scale - mx);
        r[j] = e;
        s += e;
    }
    float inv = 1.f / block_sum(s, red);
    for (int j = threadIdx.x; j < N; j += blockDim.x) r[j] *= inv;
}

__global__ void power_spec_kernel(const float *spec, int nb, size_t n, float *p) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t t = i / nb, k = i % nb;
    float re = spec[t * 2 * nb + k], im = spec[t * 2 * nb + nb + k];
    p[i] = re * re + im * im;
}

inline unsigned blocks(size_t n, int t = 256) { return (unsigned)((n + t - 1) / t); }

}  // namespace

// ---- memory ----

void *dmalloc(size_t bytes) {return nr_lowvram_allocate(bytes);}
void dfree(void *p) {nr_lowvram_free(p);}
float *dalloc_f(size_t n) { return (float *)dmalloc(n * sizeof(float)); }
void upload(void *dst, const void *src, size_t bytes) { HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice)); }
void download(void *dst, const void *src, size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}
void dcopy(void *dst, const void *src, size_t bytes) {
    HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, 0));
}
void dzero(void *dst, size_t bytes) { HIP_CHECK(hipMemsetAsync(dst, 0, bytes, 0)); }
void dsync() { HIP_CHECK(hipDeviceSynchronize()); }
size_t vram_free() {
    size_t f = 0, t = 0;
    HIP_CHECK(hipMemGetInfo(&f, &t));
    return f;
}

void Arena::init(size_t bytes) {
    base = (char *)dmalloc(bytes);
    cap = bytes;
    used = 0;
}
void Arena::release() {
    dfree(base);
    base = nullptr;
    cap = used = 0;
}
float *Arena::f(size_t n) {
    size_t bytes = (n * sizeof(float) + 255) & ~size_t(255);
    if (used + bytes > cap) fail("scratch arena exhausted (%zu + %zu > %zu)", used, bytes, cap);
    float *p = (float *)(base + used);
    used += bytes;
    return p;
}
int *Arena::i(size_t n) { return (int *)f(n); }

// ---- weights ----

size_t Weight::bytes() const {
    size_t n = (size_t)N * K;
    switch (t) {
        case WT::F32: return n * 4;
        case WT::F16:
        case WT::BF16: return n * 2;
        case WT::Q8: return n + n / 32 * 2;
    }
    return 0;
}

void Weight::free() {
    dfree(w);
    dfree(scales);
    dfree(bias);
    w = scales = nullptr;
    bias = nullptr;
}

std::vector<float> to_f32(const GgufTensor &t) {
    size_t n = t.nelements();
    std::vector<float> out(n);
    switch (t.type) {
        case GGML_F32: memcpy(out.data(), t.data, n * 4); break;
        case GGML_F16:
            for (size_t i = 0; i < n; i++) out[i] = h2f_host(((const uint16_t *)t.data)[i]);
            break;
        case GGML_BF16:
            for (size_t i = 0; i < n; i++) out[i] = bf2f_host(((const uint16_t *)t.data)[i]);
            break;
        case GGML_Q8_0:
            for (size_t b = 0; b < n / 32; b++) {
                const uint8_t *blk = t.data + b * 34;
                uint16_t d;
                memcpy(&d, blk, 2);
                float s = h2f_host(d);
                for (int j = 0; j < 32; j++) out[b * 32 + j] = (float)(int8_t)blk[2 + j] * s;
            }
            break;
        default: fail("tensor %s: unsupported type %u", t.name.c_str(), t.type);
    }
    return out;
}

float *upload_f32(const GgufTensor &t) {
    std::vector<float> h = to_f32(t);
    float *d = dalloc_f(h.size());
    upload(d, h.data(), h.size() * 4);
    return d;
}

namespace {

uint16_t f2h_host(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        uint32_t shift = 14 - exp;
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1) h++;
        return (uint16_t)(sign | h);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (man >> 13);
    if (man & 0x1000) h++;
    return (uint16_t)h;
}

// Quantize f32 rows (row length K % 32 == 0) into int8 values + one f16 scale per 32.
void quantize_q8(Weight &W, const std::vector<float> &src) {
    size_t n = src.size(), nb = n / 32;
    std::vector<int8_t> q(n);
    std::vector<uint16_t> s(nb);
    for (size_t b = 0; b < nb; b++) {
        float amax = 0;
        for (int j = 0; j < 32; j++) amax = std::max(amax, std::fabs(src[b * 32 + j]));
        float d = amax / 127.f;
        uint16_t dh = f2h_host(d);
        float dq = h2f_host(dh);
        float id = dq > 0 ? 1.f / dq : 0.f;
        for (int j = 0; j < 32; j++) q[b * 32 + j] = (int8_t)std::max(-127.f, std::min(127.f, std::nearbyint(src[b * 32 + j] * id)));
        s[b] = dh;
    }
    W.t = WT::Q8;
    W.w = dmalloc(n);
    upload(W.w, q.data(), n);
    W.scales = dmalloc(nb * 2);
    upload(W.scales, s.data(), nb * 2);
}

}  // namespace

Weight upload_linear(const GgufTensor &t, const GgufTensor *bias, bool q8) {
    Weight W;
    if (t.ne.size() < 2) fail("tensor %s is not a matrix", t.name.c_str());
    int64_t K = 1;
    for (size_t i = 0; i + 1 < t.ne.size(); i++) K *= t.ne[i];
    W.K = (int)K;
    W.N = (int)t.ne.back();
    size_t n = (size_t)W.N * W.K;
    if (q8 && t.type != GGML_Q8_0 && W.K % 32 == 0) {
        quantize_q8(W, to_f32(t));
        if (bias) W.bias = upload_f32(*bias);
        return W;
    }
    switch (t.type) {
        case GGML_F32:
            W.t = WT::F32;
            W.w = dmalloc(n * 4);
            upload(W.w, t.data, n * 4);
            break;
        case GGML_F16:
        case GGML_BF16:
            W.t = t.type == GGML_F16 ? WT::F16 : WT::BF16;
            W.w = dmalloc(n * 2);
            upload(W.w, t.data, n * 2);
            break;
        case GGML_Q8_0: {
            W.t = WT::Q8;
            if (W.K % 32) fail("tensor %s: Q8_0 row length %d not a multiple of 32", t.name.c_str(), W.K);
            size_t nb = n / 32;
            std::vector<int8_t> q(n);
            std::vector<uint16_t> s(nb);
            for (size_t b = 0; b < nb; b++) {
                const uint8_t *blk = t.data + b * 34;
                memcpy(&s[b], blk, 2);
                memcpy(&q[b * 32], blk + 2, 32);
            }
            W.w = dmalloc(n);
            upload(W.w, q.data(), n);
            W.scales = dmalloc(nb * 2);
            upload(W.scales, s.data(), nb * 2);
            break;
        }
        default: fail("tensor %s: unsupported type %u", t.name.c_str(), t.type);
    }
    if (bias) W.bias = upload_f32(*bias);
    return W;
}

Weight upload_convtr(const GgufTensor &t, bool q8) {
    // ne = (k, out, in) -> memory [in][out][k]; we want [(out*k)][in]
    if (t.ne.size() != 3) fail("convtr %s: expected 3 dims", t.name.c_str());
    int k = (int)t.ne[0], out = (int)t.ne[1], in = (int)t.ne[2];
    Weight W;
    W.N = out * k;
    W.K = in;
    size_t n = (size_t)W.N * W.K;
    if (q8 && in % 32 == 0) {
        std::vector<float> src = to_f32(t);
        std::vector<float> dst(n);
        for (int i = 0; i < in; i++)
            for (int o = 0; o < out; o++)
                for (int j = 0; j < k; j++) dst[((size_t)o * k + j) * in + i] = src[((size_t)i * out + o) * k + j];
        quantize_q8(W, dst);
    } else if (t.type == GGML_BF16 || t.type == GGML_F16) {
        W.t = t.type == GGML_BF16 ? WT::BF16 : WT::F16;
        const uint16_t *src = (const uint16_t *)t.data;
        std::vector<uint16_t> dst(n);
        for (int i = 0; i < in; i++)
            for (int o = 0; o < out; o++)
                for (int j = 0; j < k; j++) dst[((size_t)o * k + j) * in + i] = src[((size_t)i * out + o) * k + j];
        W.w = dmalloc(n * 2);
        upload(W.w, dst.data(), n * 2);
    } else {
        W.t = WT::F32;
        std::vector<float> src = to_f32(t);
        std::vector<float> dst(n);
        for (int i = 0; i < in; i++)
            for (int o = 0; o < out; o++)
                for (int j = 0; j < k; j++) dst[((size_t)o * k + j) * in + i] = src[((size_t)i * out + o) * k + j];
        W.w = dmalloc(n * 4);
        upload(W.w, dst.data(), n * 4);
    }
    return W;
}

// ---- launch wrappers ----

void matmul(const float *x, int M, const Weight &W, float *y, bool accumulate) {
    if (M <= 0) return;
    switch (W.t) {
        case WT::F32: matmul_t<WT::F32>(x, M, W, y, accumulate); break;
        case WT::F16: matmul_t<WT::F16>(x, M, W, y, accumulate); break;
        case WT::BF16: matmul_t<WT::BF16>(x, M, W, y, accumulate); break;
        case WT::Q8: matmul_t<WT::Q8>(x, M, W, y, accumulate); break;
    }
}

void embed_rows(const Weight &W, const int *ids, int n, float *out) {
    if (n <= 0) return;
    switch (W.t) {
        case WT::F32: hipLaunchKernelGGL(embed_kernel<WT::F32>, dim3(n), dim3(256), 0, 0, W.w, W.scales, W.K, ids, out); break;
        case WT::F16: hipLaunchKernelGGL(embed_kernel<WT::F16>, dim3(n), dim3(256), 0, 0, W.w, W.scales, W.K, ids, out); break;
        case WT::BF16: hipLaunchKernelGGL(embed_kernel<WT::BF16>, dim3(n), dim3(256), 0, 0, W.w, W.scales, W.K, ids, out); break;
        case WT::Q8: hipLaunchKernelGGL(embed_kernel<WT::Q8>, dim3(n), dim3(256), 0, 0, W.w, W.scales, W.K, ids, out); break;
    }
    HIP_CHECK(hipGetLastError());
}

void rmsnorm(const float *x, int M, int C, const float *w, float eps, float *y) {
    if (M <= 0) return;
    hipLaunchKernelGGL(rmsnorm_kernel, dim3(M), dim3(C >= 1024 ? 256 : 128), 0, 0, x, C, w, eps, y);
}
void layernorm(const float *x, int M, int C, const float *w, const float *b, float eps, float *y) {
    if (M <= 0) return;
    hipLaunchKernelGGL(layernorm_kernel, dim3(M), dim3(256), 0, 0, x, C, w, b, eps, y);
}
void add(const float *a, const float *b, size_t n, float *y) {
    if (n) hipLaunchKernelGGL(add_kernel, dim3(blocks(n)), dim3(256), 0, 0, a, b, n, y);
}
void add_scaled(float *y, const float *x, float s, size_t n) {
    if (n) hipLaunchKernelGGL(add_scaled_kernel, dim3(blocks(n)), dim3(256), 0, 0, y, x, s, n);
}
void add_bias_rows(float *y, const float *b, int M, int C) {
    size_t n = (size_t)M * C;
    if (n) hipLaunchKernelGGL(add_bias_rows_kernel, dim3(blocks(n)), dim3(256), 0, 0, y, b, C, n);
}
void residual_gamma(float *res, const float *x, const float *g, int M, int C) {
    size_t n = (size_t)M * C;
    if (n) hipLaunchKernelGGL(residual_gamma_kernel, dim3(blocks(n)), dim3(256), 0, 0, res, x, g, C, n);
}
void scale_inplace(float *x, float s, size_t n) {
    if (n) hipLaunchKernelGGL(scale_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, s, n);
}
void silu(const float *x, float *y, size_t n) {
    if (n) hipLaunchKernelGGL(silu_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, y, n);
}
void gelu(const float *x, float *y, size_t n) {
    if (n) hipLaunchKernelGGL(gelu_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, y, n);
}
void relu(float *x, size_t n) {
    if (n) hipLaunchKernelGGL(relu_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, n);
}
void swiglu(const float *g, const float *u, float *y, size_t n) {
    if (n) hipLaunchKernelGGL(swiglu_kernel, dim3(blocks(n)), dim3(256), 0, 0, g, u, y, n);
}
void glu(const float *x, int M, int C, float *y) {
    size_t n = (size_t)M * C;
    if (n) hipLaunchKernelGGL(glu_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, C, n, y);
}
void affine_cols(float *x, const float *s, const float *b, int M, int C) {
    size_t n = (size_t)M * C;
    if (n) hipLaunchKernelGGL(affine_cols_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, s, b, C, n);
}
void modulate(const float *x, const float *shift, const float *scale, int ld, int M, int C, float *y) {
    size_t n = (size_t)M * C;
    if (n) hipLaunchKernelGGL(modulate_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, shift, scale, ld, C, n, y);
}
void gated_add(float *x, const float *gate, int ld, const float *y, int M, int C) {
    size_t n = (size_t)M * C;
    if (n) hipLaunchKernelGGL(gated_add_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, gate, ld, y, C, n);
}
void rope_neox(float *x, int M, int H, int D, const int *pos, float theta) {
    if (M) hipLaunchKernelGGL(rope_neox_kernel, dim3(M * H), dim3(64), 0, 0, x, H, D, pos, theta);
}

void KvCache::init(int ns, int nl, int mc, int hd) {
    n_seq = ns;
    n_layer = nl;
    max_ctx = mc;
    HD = hd;
    size_t n = (size_t)ns * nl * mc * hd;
    k = (uint16_t *)dmalloc(n * 2);
    v = (uint16_t *)dmalloc(n * 2);
}
void KvCache::free() {
    dfree(k);
    dfree(v);
    k = v = nullptr;
}

void kv_store(KvCache &kv, int layer, const float *k, const float *v, int M, const int *row_seq, const int *row_pos) {
    if (M) hipLaunchKernelGGL(kv_store_kernel, dim3(M), dim3(256), 0, 0, kv.k, kv.v, layer, kv.n_layer, kv.max_ctx,
                              kv.HD, k, v, row_seq, row_pos);
}

void attn_cached(const float *q, const KvCache &kv, int layer, int M, int Hq, int Hkv, int D, const int *row_seq,
                 const int *row_pos, int max_keys, float *out) {
    if (!M) return;
    size_t shmem = (size_t)(D + max_keys) * sizeof(float);
    if (shmem > 60 * 1024) fail("attention context %d too long", max_keys);
    hipLaunchKernelGGL(attn_cached_kernel, dim3(Hq, M), dim3(128), shmem, 0, q, kv.k, kv.v, layer, kv.n_layer,
                       kv.max_ctx, Hq, Hkv, D, row_seq, row_pos, 1.f / sqrtf((float)D), out);
    HIP_CHECK(hipGetLastError());
}

void attn_relpos(const float *q, const float *k, const float *v, const float *p, const float *u, const float *vb,
                 int T, int H, int D, float *out) {
    size_t shmem = (size_t)(2 * D + T) * sizeof(float);
    if (shmem > 60 * 1024) fail("encoder sequence %d too long", T);
    hipLaunchKernelGGL(attn_relpos_kernel, dim3(H, T), dim3(128), shmem, 0, q, k, v, p, u, vb, T, H, D,
                       1.f / sqrtf((float)D), out);
    HIP_CHECK(hipGetLastError());
}

void dwconv1d(const float *x, int T_in, int C, const float *w, const float *b, int k, int pad_left, int T_out,
              float *out) {
    size_t n = (size_t)T_out * C;
    if (n) hipLaunchKernelGGL(dwconv1d_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, T_in, C, w, b, k, pad_left,
                              T_out, out);
}
void im2col1d(const float *x, int C, int k, int stride, int T_out, float *X) {
    size_t n = (size_t)T_out * C * k;
    if (n) hipLaunchKernelGGL(im2col1d_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, C, k, stride, T_out, X);
}
void convtr_ola(const float *Y, const float *prev, int T, int Cout, int stride, const float *b, float *out) {
    size_t n = (size_t)T * stride * Cout;
    if (n) hipLaunchKernelGGL(convtr_ola_kernel, dim3(blocks(n)), dim3(256), 0, 0, Y, prev, T, Cout, stride, b, out);
}
void conv2d_s2_c1(const float *x, int H, int W, const float *w, const float *b, int Cout, float *out) {
    size_t n = (size_t)((H - 1) / 2 + 1) * ((W - 1) / 2 + 1) * Cout;
    hipLaunchKernelGGL(conv2d_s2_c1_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, H, W, w, b, Cout, out);
}
void dwconv2d_s2(const float *x, int H, int W, int C, const float *w, const float *b, float *out) {
    size_t n = (size_t)((H - 1) / 2 + 1) * ((W - 1) / 2 + 1) * C;
    hipLaunchKernelGGL(dwconv2d_s2_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, H, W, C, w, b, out);
}
void transpose12(const float *x, int A, int B, int C, float *y) {
    size_t n = (size_t)A * B * C;
    if (n) hipLaunchKernelGGL(transpose12_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, A, B, C, y);
}
void permute3(const float *x, int d0, int d1, int d2, int p0, int p1, int p2, float *y) {
    size_t n = (size_t)d0 * d1 * d2;
    if (n) hipLaunchKernelGGL(permute3_kernel, dim3(blocks(n)), dim3(256), 0, 0, x, d0, d1, d2, p0, p1, p2, y);
}
void softmax_rows(float *x, int M, int N, float scale) {
    if (M) hipLaunchKernelGGL(softmax_rows_kernel, dim3(M), dim3(256), 0, 0, x, N, scale);
}
void power_spec(const float *spec, int T, int nb, float *p) {
    size_t n = (size_t)T * nb;
    if (n) hipLaunchKernelGGL(power_spec_kernel, dim3(blocks(n)), dim3(256), 0, 0, spec, nb, n, p);
}
