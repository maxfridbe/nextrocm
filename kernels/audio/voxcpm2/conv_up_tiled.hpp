// Private opt-in VoxCPM2 decoder transpose-convolution candidate.
// Contract: contiguous F32 x[B,Ci,L], w[Ci,Co,K], optional bias[Co],
// out[B,Co,L*S], K=2*S, dilation=1, left padding=0, right trim=K-S.
// This preserves mm3.hip speech_conv_k: ci then ascending k, skipped taps
// (NOT zero-valued FMAs), explicit F32 fmaf, and bias added AFTER reduction.
// No scratch, weight cache, shared ABI changes, or automatic dispatch.
// Include in ONE HIP TU. Parent integration wrapper should validate g, call
// hipSetDevice(g->dev), launch(..., g->q), and translate hipError_t via ns_hip.
// Compile with ordinary -O2, never -ffast-math / -ffinite-math-only.
//
// Integration handoff (NOT enabled by this header): add this private C ABI to
// vcp.h/vcp.hip, and bind exactly this signature in the audio crate:
// int ns_audio_vcp_conv_up_tiled(ns_gpu* g, const float* x, int64_t B,
//     int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
//     const float* bias, int64_t stride, float* out);
// Wrapper body:
//   if (!g) return ns_fail("vcp conv_up: null GPU");
//   if (!vcp_conv_up::arguments(x,w,bias,out,B,Ci,L,Co,K,stride))
//       return ns_fail("vcp conv_up: invalid/unsupported arguments");
//   NS_CU(hipSetDevice(g->dev));
//   NS_CU(vcp_conv_up::launch(x,B,Ci,L,w,Co,K,bias,stride,out,g->q));
//   return 0;
// Proposed experiment gate: NS_VCP_CONV_UP_TILED=1 (unset stays baseline).
// Select only K=2*stride, stride in {2,5,6,8}, and supported() geometry; preserve
// the existing conv1d_lr call for all other shapes or when the gate is absent.
// A selected candidate's errors propagate, not an after-launch fallback.
// Keep weights F32: this path must never receive the GEMV half-weight cache.
// CPU tests, actual baseline GPU parity and measured A/B must pass before
// opt-in trajectory validation; no default switch is authorized here.
#pragma once
#if defined(__FAST_MATH__)
#error "VoxCPM2 conv_up requires strict F32 FMA and final bias rounding"
#endif
#include <cstdint>
#include <climits>
#include <initializer_list>

namespace vcp_conv_up {
using I = int64_t;
constexpr I time_tile = 64, channel_tile = 16;
// Bound each addressable span AND their aggregate below the tracked device
// ceiling. The caller still owns allocation lengths and total model admission.
inline bool span(std::initializer_list<I> dims, I& n) {
    n = 1;
    for (I d : dims) {
        if (d <= 0 || n > 1874999999LL / d) return false;
        n *= d;
    }
    return true;
}
inline bool supported(I B, I Ci, I L, I Co, I K, I S) {
    if (!(S == 2 || S == 5 || S == 6 || S == 8) || K != 2*S) return false;
    I nx, nw, ny;
    if (!span({B,Ci,L},nx) || !span({Ci,Co,K},nw) ||
        !span({B,Co,L,S},ny)) return false;
    if (nx + nw + ny + Co >= 1875000000LL) return false;
    // Flatten batch/channel/phase into grid.y, honoring HIP's portable bound.
    return (L+time_tile-1)/time_tile <= INT_MAX &&
           B*((Co+channel_tile-1)/channel_tile)*S <= 65535;
}
inline bool arguments(const float* x, const float* w, const float* bias,
                      float* out, I B, I Ci, I L, I Co, I K, I S) {
    if (!supported(B,Ci,L,Co,K,S) || !x || !w || !out) return false;
    auto aligned=[](const void* p) { return reinterpret_cast<uintptr_t>(p)%alignof(float)==0; };
    if (!aligned(x)||!aligned(w)||!aligned(out)||(bias&&!aligned(bias))) return false;
    // Reject address wrap and output overlap with any source (partial too).
    const uintptr_t o=reinterpret_cast<uintptr_t>(out);
    const uintptr_t bytes=uintptr_t(B*Co*L*S)*sizeof(float);
    if (o>UINTPTR_MAX-bytes) return false;
    auto disjoint=[&](const float* p, I n) {
        const uintptr_t a=reinterpret_cast<uintptr_t>(p), size=uintptr_t(n)*sizeof(float);
        return a<=UINTPTR_MAX-size && (o+bytes<=a || a+size<=o);
    };
    return disjoint(x,B*Ci*L) && disjoint(w,Ci*Co*K) && (!bias||disjoint(bias,Co));
}
} // namespace vcp_conv_up

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <cmath>
namespace vcp_conv_up {
// Each logical wave computes 4 channels x 64 input positions. Two temporal
// values per lane reuse weights; four channels reuse each input. No shuffle or
// wave-width assumption: independent threads also work on wave64 hardware.
template<int S>
__global__ void tiled(const float* x, const float* w, const float* bias,
                      float* out, I Ci, I L, I Co) {
    const I phase=blockIdx.y%S, group=blockIdx.y/S;
    const I groups=(Co+channel_tile-1)/channel_tile;
    const I b=group/groups, c0=(group%groups)*channel_tile+(threadIdx.x/32)*4;
    const I t0=I(blockIdx.x)*time_tile+threadIdx.x%32;
    float acc[4][2] = {};
    for (I ci=0; ci<Ci; ++ci) {
        #pragma unroll
        for (int tap=0; tap<2; ++tap) {
            // At pos=t*S+phase only k=phase and phase+S contribute, in
            // that order. For t=0 the second tap MUST NOT perform an FMA.
            float xv[2];
            #pragma unroll
            for (int j=0;j<2;++j) {
                const I t=t0+j*32;
                xv[j]=(t<L && t>=tap)?x[(b*Ci+ci)*L+t-tap]:0.f;
            }
            #pragma unroll
            for (int c=0;c<4;++c) {
                if (c0+c<Co) {
                    const float weight=w[(ci*Co+c0+c)*(2*S)+phase+tap*S];
                    #pragma unroll
                    for (int j=0;j<2;++j) {
                        const I t=t0+j*32;
                        if (t<L && t>=tap) acc[c][j]=fmaf(xv[j],weight,acc[c][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for(int c=0;c<4;++c) if(c0+c<Co) {
        const float bv=bias?bias[c0+c]:0.f;
        #pragma unroll
        for(int j=0;j<2;++j) {
            const I t=t0+j*32;
            if(t<L) out[((b*Co+c0+c)*L+t)*S+phase]=acc[c][j]+bv;
        }
    }
}
// Does not synchronize. Invalid/unsupported calls launch nothing and return
// hipErrorInvalidValue. On a launch failure propagate the error; do not silently
// fall back after a possibly enqueued operation.
inline hipError_t launch(const float* x, I B, I Ci, I L, const float* w,
                         I Co, I K, const float* bias, I S, float* out,
                         hipStream_t stream) {
    if(!arguments(x,w,bias,out,B,Ci,L,Co,K,S)) return hipErrorInvalidValue;
    const dim3 grid(unsigned((L+63)/64),unsigned(B*((Co+15)/16)*S));
    #define VCP_UP_CASE(s) case s: tiled<s><<<grid,128,0,stream>>>(x,w,bias,out,Ci,L,Co); break
    switch(S) { VCP_UP_CASE(2); VCP_UP_CASE(5); VCP_UP_CASE(6); VCP_UP_CASE(8); }
    #undef VCP_UP_CASE
    return hipGetLastError();
}
} // namespace vcp_conv_up
#endif
