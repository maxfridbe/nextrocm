// Private precision diagnostic, copied from minimaxmusic3/mm3.hip cached
// attention/store. The only arithmetic-path changes are F32 cache loads/stores
// instead of half conversions. Same split count, scale, reduction and merging.
// Include once after ns_internal.hpp. No shared MM3 kernel changes.
#pragma once
#include <cmath>
#include <cstdint>
namespace vcp_kv_f32 {
constexpr int WARP = 32;
__device__ inline float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
constexpr int64_t KB = 256;  // keys a split (decode)
constexpr int NW_A = 4;

int64_t splits(int64_t S, int64_t kv) { return S == 1 && kv > KB ? (kv + KB - 1) / KB : 1; }

template <int DL, int G>
__global__ void attn_k(const float* qp, int64_t qs, const float* kc, const float* vc, int64_t S, int64_t Hq, int64_t Hkv, int64_t T,
                       int64_t p0, int64_t ns, float* out, float* part) {
    constexpr int D = WARP * DL, WG = WARP * NW_A;
    extern __shared__ float slm[];
    const int lane = threadIdx.x & 31, sgi = threadIdx.x >> 5;
    const float scale = rsqrtf((float) D);
    int64_t gi = blockIdx.x;
    const int64_t sp = gi % ns;
    gi /= ns;
    const int64_t hk = gi % Hkv, r = gi / Hkv, b = r / S, s = r % S;
    const int64_t kv_end = p0 + s + 1;
    const int64_t j0 = sp * (ns > 1 ? KB : kv_end), j1 = min(kv_end, ns > 1 ? (sp + 1) * KB : kv_end);
    float qv[G][DL], acc[G][DL], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; ++g) {
        const float* qr = qp + r * qs + (hk * G + g) * D + lane * DL;
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            qv[g][i] = qr[i] * scale;
            acc[g][i] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }
    const float* kb = kc + (b * Hkv + hk) * T * D + lane * DL;
    const float* vb = vc + (b * Hkv + hk) * T * D + lane * DL;
    for (int64_t j = j0 + sgi; j < j1; j += NW_A) {
        float kv[DL], vv[DL];
#pragma unroll
        for (int i = 0; i < DL; ++i) {
            kv[i] = kb[j * D + i];
            vv[i] = vb[j * D + i];
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
            float d = 0.0f;
#pragma unroll
            for (int i = 0; i < DL; ++i) d += qv[g][i] * kv[i];
            const float sc = warp_sum(d);
            const float mn = fmaxf(m[g], sc);
            const float c = expf(m[g] - mn), e = expf(sc - mn);
#pragma unroll
            for (int i = 0; i < DL; ++i) acc[g][i] = acc[g][i] * c + e * vv[i];
            l[g] = l[g] * c + e;
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float* o = &slm[(sgi * G + g) * (D + 2)];
#pragma unroll
        for (int i = 0; i < DL; ++i) o[lane * DL + i] = acc[g][i];
        if (lane == 0) {
            o[D] = m[g];
            o[D + 1] = l[g];
        }
    }
    __syncthreads();
    for (int idx = threadIdx.x; idx < G * D; idx += WG) {
        const int g = idx / D, d = idx % D;
        float mx = -INFINITY;
        for (int k = 0; k < NW_A; ++k) mx = fmaxf(mx, slm[(k * G + g) * (D + 2) + D]);
        float L = 0.0f, A = 0.0f;
        if (mx != -INFINITY) {
            for (int k = 0; k < NW_A; ++k) {
                const float* o = &slm[(k * G + g) * (D + 2)];
                const float e = expf(o[D] - mx);
                L += o[D + 1] * e;
                A += o[d] * e;
            }
        }
        const int64_t hq = hk * G + g;
        if (ns == 1) {
            out[r * Hq * D + hq * D + d] = L > 0.0f ? A / L : 0.0f;
        } else {
            float* p = part + ((r * Hq + hq) * ns + sp) * (D + 2);
            p[d] = A;
            if (d == 0) {
                p[D] = mx;
                p[D + 1] = L;
            }
        }
    }
}

__global__ void attn_merge_k(const float* part, int64_t ns, int64_t D, float* out) {
    const int64_t rh = blockIdx.x, d = threadIdx.x;
    const float* p = part + rh * ns * (D + 2);
    float mx = -INFINITY;
    for (int64_t k = 0; k < ns; ++k) mx = fmaxf(mx, p[k * (D + 2) + D]);
    float L = 0.0f, A = 0.0f;
    if (mx != -INFINITY) {
        for (int64_t k = 0; k < ns; ++k) {
            const float e = expf(p[k * (D + 2) + D] - mx);
            L += p[k * (D + 2) + D + 1] * e;
            A += p[k * (D + 2) + d] * e;
        }
    }
    out[rh * D + d] = L > 0.0f ? A / L : 0.0f;
}

template <int DL, int G>
int attn(ns_gpu* g, const float* q, int64_t qs, const float* k, const float* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t T,
         int64_t p0, float* out, float* part) {
    constexpr int D = WARP * DL;
    const int64_t ns = splits(S, p0 + S);
    const size_t shm = (size_t) NW_A * G * (D + 2) * sizeof(float);
    attn_k<DL, G><<<(unsigned) (B * S * Hkv * ns), WARP * NW_A, shm, g->q>>>(q, qs, k, v, S, Hq, Hkv, T, p0, ns, out, part);
    NS_LAUNCH_CHECK();
    if (ns > 1) {
        attn_merge_k<<<(unsigned) (B * S * Hq), D, 0, g->q>>>(part, ns, D, out);
        NS_LAUNCH_CHECK();
    }
    return 0;
}

__global__ void kv_store_k(const float* k, const float* v, int64_t ks, int64_t S, int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* ko,
                           float* vo, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t d = i % D, h = (i / D) % Hkv, r = i / (D * Hkv), b = r / S, s = r % S;
    const int64_t at = ((b * Hkv + h) * T + p0 + s) * D + d;
    ko[at] = k[r * ks + h * D + d];
    vo[at] = v[r * ks + h * D + d];
}

} // namespace vcp_kv_f32
