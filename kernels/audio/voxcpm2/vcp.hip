// vcp.cu: the VoxCPM2 engine's own kernels (vcp.h). Everything here is small - a local sequence is 5 or 11 rows, the
// VAE's channels a few hundred thousand samples - so these are plain element-per-thread kernels, as nextsycl's
// vcp.cpp (the same names, arguments and arithmetic).
#include "ns.h"
#include "ns_internal.hpp"
#include "vcp.h"
#include "conv_up_tiled.hpp"
#include "kv_f32.hpp"
#include "linear_f16_tiled.hpp"

#include <cmath>
#include <initializer_list>
#include <cstdint>

namespace {

// Bound every cache/activation span before address and launch arithmetic. This
// is only geometry validation; Rust admits the full model/session allocation.
bool kv_span(std::initializer_list<int64_t> dims) {
    int64_t n=1;
    for (int64_t d:dims) {
        if (d<=0 || n>1874999999LL/d) return false;
        n*=d;
    }
    return true;
}

// All activation indexing fits signed byte spans and a portable 1-D launch.
bool shape(std::initializer_list<int64_t> dims) {
    int64_t count=1;
    for (int64_t d:dims) { if(d<=0 || count>INT64_MAX/4/d)return false;count*=d; }
    return (count+63)/64<=2147483647;
}
unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

#define ID() const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x

template<bool BF16_CACHE=false>
__global__ void rope_k(float* x, int64_t xs, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t half = D / 2;
    const int64_t r = id / (H * half), h = (id / half) % H, i = id % half;
    float* p = x + r * xs + h * D;
    const float a = (float) (p0 + r % S) * inv_freq[i];
    float c = cosf(a), s = sinf(a);
    if constexpr (BF16_CACHE) {
        // Match original loader's F32 -> BF16 -> F32 cached trig buffers.
        // Finite sin/cos values only, ties to even (including signed zero).
        unsigned cb=__float_as_uint(c), sb=__float_as_uint(s);
        c=__uint_as_float((cb+0x7fffu+((cb>>16)&1u))&0xffff0000u);
        s=__uint_as_float((sb+0x7fffu+((sb>>16)&1u))&0xffff0000u);
    }
    const float x1 = p[i], x2 = p[i + half];
    p[i] = x1 * c - x2 * s;
    p[i + half] = x2 * c + x1 * s;
}

__global__ void local_attn_k(const float* q, const float* k, const float* v, int64_t S, int64_t Hq, int64_t G, int64_t D, int64_t qs, int64_t kvs,
                             float scale, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t r = id / Hq, h = id % Hq, b = r / S, kh = h / G;
    const float* qi = q + r * qs + h * D;
    float sc[64];
    float mx = -INFINITY;
    for (int64_t j = 0; j < S && j < 64; ++j) {
        const float* kj = k + (b * S + j) * kvs + kh * D;
        float s = 0.0f;
        for (int64_t d = 0; d < D; ++d) s += qi[d] * kj[d];
        sc[j] = s * scale;
        mx = fmaxf(mx, sc[j]);
    }
    float acc[128];
    for (int64_t d = 0; d < D; ++d) acc[d] = 0.0f;
    float sum = 0.0f;
    for (int64_t j = 0; j < S && j < 64; ++j) {
        const float p = expf(sc[j] - mx);
        sum += p;
        const float* vj = v + (b * S + j) * kvs + kh * D;
        for (int64_t d = 0; d < D; ++d) acc[d] += p * vj[d];
    }
    float* o = out + r * (Hq * D) + h * D;
    for (int64_t d = 0; d < D; ++d) o[d] = acc[d] / sum;
}

__global__ void snake_k(const float* x, int64_t L, const float* alpha, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const float a = alpha[id / L], v = x[id], s = sinf(a * v);
    out[id] = v + s * s / (a + 1e-9f);
}

__global__ void dwconv_k(const float* x, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t c = id / L, t = id % L;
    float a = bias ? bias[c] : 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        const int64_t s = t - (K - 1 - k) * dil;
        if (s >= 0) a += w[c * K + k] * x[c * L + s];
    }
    out[id] = a;
}

__global__ void affine_k(float* x, int64_t L, const float* scale, const float* bias, int64_t n) {
    ID();
    if (id >= n) return;
    const int64_t c = id / L;
    x[id] = x[id] * scale[c] + bias[c];
}

__global__ void fsq_k(float* x, int64_t n, float s) {
    ID();
    if (id < n) x[id] = rintf(tanhf(x[id]) * s) / s;
}

}  // namespace

extern "C" {

int ns_audio_vcp_linear_f16_tiled(ns_gpu* g, const void* x, int64_t M,
    int64_t K, const void* w, int64_t N, const float* bias, float* out) {
    if (!g || !vcp_linear_f16::arguments(x,M,K,w,N,bias,out))
        return ns_fail("vcp linear F16: invalid/unsupported arguments");
    NS_CU(hipSetDevice(g->dev));
    NS_CU(vcp_linear_f16::launch(x,M,K,w,N,bias,out,g->q));
    return 0;
}

int ns_audio_vcp_rope_bf16_cache(ns_gpu* g, float* x, int64_t xs, int64_t rows,
    int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0) {
    if (!g || !x || !inv_freq || !shape({rows,xs}) || !shape({H,D}) || S<=0 || D%2 || xs<H*D || p0<0 || p0>INT64_MAX-S)
        return ns_fail("vcp BF16 rotary cache: invalid arguments");
    NS_CU(hipSetDevice(g->dev));
    const int64_t n=rows*H*(D/2);
    rope_k<true><<<blocks(n),256,0,g->q>>>(x,xs,S,H,D,inv_freq,p0,n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_kv_store_f32(ns_gpu* g, const float* k, const float* v,
    int64_t ks, int64_t B, int64_t S, int64_t Hkv, int64_t D,
    int64_t T, int64_t p0, void* kc, void* vc) {
    if (!g || !k || !v || !kc || !vc || D!=128 || p0<0 || T<=0 ||
        S<=0 || S>T || p0>T-S || !kv_span({B,Hkv,T,D}) ||
        !kv_span({B,S,ks}) || ks<Hkv*D)
        return ns_fail("vcp F32 KV: invalid store geometry/pointers");
    NS_CU(hipSetDevice(g->dev));
    const int64_t n=B*S*Hkv*D;
    vcp_kv_f32::kv_store_k<<<blocks(n),256,0,g->q>>>(
        k,v,ks,S,Hkv,D,T,p0,(float*)kc,(float*)vc,n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_attn_f32(ns_gpu* g, const float* q, int64_t qs,
    const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
    int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part) {
    if (!g || !q || !kc || !vc || !out || D!=128 || p0<0 || T<=0 ||
        S<=0 || S>T || p0>T-S || !kv_span({B,Hkv,T,D}) ||
        !kv_span({B,S,Hq,D}) || !kv_span({B,S,qs}) || qs<Hq*D ||
        Hq%Hkv!=0 || !(Hq/Hkv==2 || Hq/Hkv==4 || Hq/Hkv==8))
        return ns_fail("vcp F32 KV: invalid attention geometry/pointers");
    const int64_t ns=vcp_kv_f32::splits(S,p0+S);
    if (ns>1 && (!part || !kv_span({B,S,Hq,ns,D+2})))
        return ns_fail("vcp F32 KV: missing/oversized attention scratch");
    NS_CU(hipSetDevice(g->dev));
    const float* k=(const float*)kc;
    const float* v=(const float*)vc;
    if (Hq/Hkv==2) return vcp_kv_f32::attn<4,2>(g,q,qs,k,v,B,S,Hq,Hkv,T,p0,out,part);
    if (Hq/Hkv==4) return vcp_kv_f32::attn<4,4>(g,q,qs,k,v,B,S,Hq,Hkv,T,p0,out,part);
    return vcp_kv_f32::attn<4,8>(g,q,qs,k,v,B,S,Hq,Hkv,T,p0,out,part);
}

int ns_audio_vcp_conv_up_tiled(ns_gpu* g, const float* x, int64_t B,
    int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
    const float* bias, int64_t stride, float* out) {
    if (!g) return ns_fail("vcp conv_up: null GPU");
    if (!vcp_conv_up::arguments(x,w,bias,out,B,Ci,L,Co,K,stride))
        return ns_fail("vcp conv_up: invalid/unsupported arguments");
    NS_CU(hipSetDevice(g->dev));
    NS_CU(vcp_conv_up::launch(x,B,Ci,L,w,Co,K,bias,stride,out,g->q));
    return 0;
}

int ns_audio_vcp_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0) {
    if (!g || !x || !inv_freq || !shape({rows,xs}) || !shape({H,D}) || S<=0 || D%2 || xs<H*D || p0<0 || p0>INT64_MAX-S)
        return ns_fail("ns_audio_vcp_rope: invalid pointer, dimension or stride");
    NS_CU(hipSetDevice(g->dev));
    const int64_t n = rows * H * (D / 2);
    rope_k<false><<<blocks(n), 256, 0, g->q>>>(x, xs, S, H, D, inv_freq, p0, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_local_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                            int64_t qs, int64_t kvs, float* out) {
    if (!g || !q || !k || !v || !out || !shape({B,S,qs}) || !shape({B,S,kvs})
        || !shape({Hq,D}) || !shape({Hkv,D}) || S>64 || D>128 || Hq%Hkv || qs<Hq*D || kvs<Hkv*D)
        return ns_fail("ns_audio_vcp_local_attn: S<=64, D<=128, valid GQA heads/strides required");
    NS_CU(hipSetDevice(g->dev));
    const int64_t n = B * S * Hq;
    local_attn_k<<<blocks(n, 64), 64, 0, g->q>>>(q, k, v, S, Hq, Hq / Hkv, D, qs, kvs, 1.0f / sqrtf((float) D), out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, float* out) {
    if (!g || !x || !alpha || !out || !shape({C,L})) return ns_fail("ns_audio_vcp_snake: invalid shape or pointer");
    NS_CU(hipSetDevice(g->dev));
    snake_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, alpha, out, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out) {
    if (!g || !x || !w || !out || !shape({C,L}) || !shape({C,K}) || !shape({K,dil})) return ns_fail("ns_audio_vcp_dwconv: invalid shape/dilation or pointer");
    NS_CU(hipSetDevice(g->dev));
    dwconv_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, w, bias, K, dil, out, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_affine(ns_gpu* g, float* x, int64_t C, int64_t L, const float* scale, const float* bias) {
    if (!g || !x || !scale || !bias || !shape({C,L})) return ns_fail("ns_audio_vcp_affine: invalid shape or pointer");
    NS_CU(hipSetDevice(g->dev));
    affine_k<<<blocks(C * L), 256, 0, g->q>>>(x, L, scale, bias, C * L);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_audio_vcp_fsq(ns_gpu* g, float* x, int64_t n, float s) {
    if (!g || !x || !shape({n}) || !std::isfinite(s) || s<=0) return ns_fail("ns_audio_vcp_fsq: invalid count, scale or pointer");
    NS_CU(hipSetDevice(g->dev));
    fsq_k<<<blocks(n), 256, 0, g->q>>>(x, n, s);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"
