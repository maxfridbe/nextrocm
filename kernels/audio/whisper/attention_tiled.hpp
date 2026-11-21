// Private noncausal Whisper encoder F16 Q/K/V -> F16 output, contiguous
// [S,H,64]. No production dispatch/default change. Each block handles four
// queries of one head, cooperatively loading a 64-key x 64-dimension tile.
// Preserve speech.hip att_kernel exactly: ascending dimension fmaf dot,
// rsqrtf(64) scaling, thread j=t+256*n max/exp sums and 256-thread reduction
// trees, ascending-key fmaf PV, divide only after PV, one final half rounding.
// No online softmax, split dot, reassociation, padded PV FMAs, or global scratch.
// Proposed ABI: ns_audio_whisper_attention_tiled(ns_gpu*, const void* q,
// const void* k, const void* v, int64_t S, int64_t H, int64_t D,
// int64_t stride, void* out). Wrapper checks GPU/arguments, sets device, calls
// launch on g->q and propagates errors. Opt-in NS_WHISPER_ATTENTION_TILED=1;
// initial model dispatch ONLY actual encoder S1500 H20 D64 stride1280.
#pragma once
#ifdef __FAST_MATH__
#error "Whisper attention requires baseline F32 arithmetic and expf"
#endif
#include <cstdint>
#include <initializer_list>
namespace whisper_attention {
using I=int64_t;
constexpr I queries=4, key_tile=64, dimension=64;
inline bool supported(I S,I H,I D,I stride) {
    return S>=1&&S<=2048&&H>=1&&H<=32&&D==64&&stride==H*64;
}
inline I shared_bytes(I S) {return (4*S+64*65+4*64+4*256)*I(sizeof(float));}
inline bool arguments(const void* q,const void* k,const void* v,I S,I H,I D,I stride,void* out) {
    if(!supported(S,H,D,stride)||!q||!k||!v||!out)return false;
    const uintptr_t bytes=uintptr_t(S*H*64)*2,c=reinterpret_cast<uintptr_t>(out);
    if(c%2||c>UINTPTR_MAX-bytes)return false;
    for(const void* p:{q,k,v}) {
        const uintptr_t a=reinterpret_cast<uintptr_t>(p);
        if(a%2||a>UINTPTR_MAX-bytes||!(a+bytes<=c||c+bytes<=a))return false;
    }
    return true;
}
}
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
namespace whisper_attention {
__global__ __launch_bounds__(256) void tiled(const __half* q,const __half* k,const __half* v,I S,I H,I D,__half* out) {
    extern __shared__ float score[]; // [4,S], each query's baseline score row
    __shared__ float keys[64][65],query[4][64],tmp[4][256];
    const int t=threadIdx.x,qi=t/64,ki=t%64;
    const I row0=I(blockIdx.x)*4,h=blockIdx.y,stride=H*64;
    query[qi][ki]=(row0+qi<S)?__half2float(q[(row0+qi)*stride+h*64+ki]):0.f;
    __syncthreads();
    for(I base=0;base<S;base+=64) {
        // Adjacent lanes load adjacent half dimensions, then transpose in LDS.
        for(int z=t;z<4096;z+=256) {
            const int key=z/64,a=z%64;
            keys[a][key]=(base+key<S)?__half2float(k[(base+key)*stride+h*64+a]):0.f;
        }
        __syncthreads();
        if(base+ki<S) {
            float z=0;
            #pragma unroll
            for(int a=0;a<64;++a)z=fmaf(query[qi][a],keys[a][ki],z);
            z*=rsqrtf(float(D));score[qi*S+base+ki]=z;
        }
        __syncthreads();
    }
    // Keep each query's original 256-thread lane partition and tree exactly.
    float best[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for(I j=t;j<S;j+=256) {
        #pragma unroll
        for(int r=0;r<4;++r)best[r]=fmaxf(best[r],score[r*S+j]);
    }
    #pragma unroll
    for(int r=0;r<4;++r)tmp[r][t]=best[r];
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(t<step) {
            #pragma unroll
            for(int r=0;r<4;++r)tmp[r][t]=fmaxf(tmp[r][t],tmp[r][t+step]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<4;++r)best[r]=tmp[r][0];
    __syncthreads();
    float sum[4]={};
    for(I j=t;j<S;j+=256) {
        #pragma unroll
        for(int r=0;r<4;++r) {float z=expf(score[r*S+j]-best[r]);score[r*S+j]=z;sum[r]+=z;}
    }
    #pragma unroll
    for(int r=0;r<4;++r)tmp[r][t]=sum[r];
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(t<step) {
            #pragma unroll
            for(int r=0;r<4;++r)tmp[r][t]+=tmp[r][t+step];
        }
        __syncthreads();
    }
    // Four queries occupy all 256 threads. Feature lanes read contiguous V;
    // query groups reuse the same V cache lines, scores broadcast within waves.
    if(row0+qi<S) {
        float z=0;
        for(I j=0;j<S;++j)z=fmaf(score[qi*S+j],__half2float(v[j*stride+h*64+ki]),z);
        out[(row0+qi)*stride+h*64+ki]=__float2half_rn(z/tmp[qi][0]);
    }
}
inline hipError_t launch(const void* q,const void* k,const void* v,I S,I H,I D,I stride,void* out,hipStream_t stream) {
    if(!arguments(q,k,v,S,H,D,stride,out))return hipErrorInvalidValue;
    tiled<<<dim3(unsigned((S+3)/4),unsigned(H)),256,size_t(4*S)*sizeof(float),stream>>>(
        static_cast<const __half*>(q),static_cast<const __half*>(k),static_cast<const __half*>(v),S,H,D,static_cast<__half*>(out));
    return hipGetLastError();
}
}
#endif
