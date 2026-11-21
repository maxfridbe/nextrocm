// Private image DiT attention candidate; no TE/causal/shared dispatch changes.
// Separate 1024 candidate: half LDS retains the exact input bits; each
// multiplication still consumes the same float value as the frozen baseline.
// F16 Q/K/V -> F16 contiguous output, two queries per head/block.
// Preserve speech.hip ascending-QK FMA, runtime rsqrtf(D), 256-lane max/sum
// trees, expf scores, ascending-PV FMA, divide last and half RNE. No P-first
// half rounding, online softmax, split-K, reassociation or global scratch.
#pragma once
#ifdef __FAST_MATH__
#error "Qwen image attention requires baseline arithmetic"
#endif
#include <cstdint>
namespace nrqi_attention_1024 {
using I=int64_t;
inline bool supported(I sq,I sk,I h,I d,I qs,I ks) {
    return sq>=1&&sq<=4096&&sk>=1&&sk<=4608&&h>=1&&h<=32&&d==128
        &&qs>=h*d&&qs<=12288&&ks>=h*d&&ks<=12288;
}
inline I shared_bytes(I sk) {return (2*sk+2*128+2*256)*I(sizeof(float))+128*65*I(sizeof(uint16_t));}
inline bool span(const void* p,uintptr_t bytes,uintptr_t& first,uintptr_t& end) {
    first=reinterpret_cast<uintptr_t>(p);
    if(!p||first%2||first>UINTPTR_MAX-bytes)return false;
    end=first+bytes;return true;
}
inline bool arguments(const void* q,const void* k,const void* v,I sq,I sk,I h,I d,I qs,I ks,void* out) {
    if(!supported(sq,sk,h,d,qs,ks))return false;
    uintptr_t o,oe,a,ae;
    if(!span(out,uintptr_t(sq*h*d)*2,o,oe))return false;
    const void* inputs[]={q,k,v};const I rows[]={sq,sk,sk};const I strides[]={qs,ks,ks};
    for(int i=0;i<3;++i) {
        if(!span(inputs[i],uintptr_t((rows[i]-1)*strides[i]+h*d)*2,a,ae)||!(ae<=o||oe<=a))return false;
    }
    return true;
}
}
#if defined(__HIPCC__) && !defined(NRQI_ATTENTION_1024_GUARDS_ONLY)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include "../../ns/ns_internal.hpp"
namespace nrqi_attention_1024 {
__global__ __launch_bounds__(256) void tiled(const __half* q,const __half* k,const __half* v,I sq,I sk,I h,I d,I qs,I ks,__half* out) {
    extern __shared__ float score[];
    __shared__ __half keys[128][65];
    __shared__ float query[2][128],tmp[2][256];
    const int t=threadIdx.x,qi=t/128,ki=t%128;
    const I row0=I(blockIdx.x)*2,head=blockIdx.y;
    for(int z=t;z<256;z+=256) {
        const int r=z/128,a=z%128;
        query[r][a]=(row0+r<sq)?__half2float(q[(row0+r)*qs+head*128+a]):0.f;
    }
    __syncthreads();
    for(I base=0;base<sk;base+=64) {
        for(int z=t;z<8192;z+=256) {
            const int key=z/128,a=z%128;
            keys[a][key]=(base+key<sk)?k[(base+key)*ks+head*128+a]:__float2half_rn(0.f);
        }
        __syncthreads();
        if(ki<64&&base+ki<sk) {
            float z=0;
            #pragma unroll
            for(int a=0;a<128;++a)z=fmaf(query[qi][a],__half2float(keys[a][ki]),z);
            z*=rsqrtf(float(d));score[qi*sk+base+ki]=z;
        }
        __syncthreads();
    }
    float best[2]={-INFINITY,-INFINITY};
    for(I j=t;j<sk;j+=256) {
        #pragma unroll
        for(int r=0;r<2;++r)best[r]=fmaxf(best[r],score[r*sk+j]);
    }
    #pragma unroll
    for(int r=0;r<2;++r)tmp[r][t]=best[r];
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(t<step) {
            #pragma unroll
            for(int r=0;r<2;++r)tmp[r][t]=fmaxf(tmp[r][t],tmp[r][t+step]);
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<2;++r)best[r]=tmp[r][0];
    __syncthreads();
    float sum[2]={};
    for(I j=t;j<sk;j+=256) {
        #pragma unroll
        for(int r=0;r<2;++r) {float z=expf(score[r*sk+j]-best[r]);score[r*sk+j]=z;sum[r]+=z;}
    }
    #pragma unroll
    for(int r=0;r<2;++r)tmp[r][t]=sum[r];
    __syncthreads();
    for(int step=128;step;step/=2) {
        if(t<step) {
            #pragma unroll
            for(int r=0;r<2;++r)tmp[r][t]+=tmp[r][t+step];
        }
        __syncthreads();
    }
    if(row0+qi<sq) {
        float a=0;
        for(I j=0;j<sk;++j) {
            a=fmaf(score[qi*sk+j],__half2float(v[j*ks+head*128+ki]),a);
        }
        out[((row0+qi)*h+head)*128+ki]=__float2half_rn(a/tmp[qi][0]);
    }
}
inline int launch(void* ctx,const void* q,const void* k,const void* v,I sq,I sk,I h,I d,I qs,I ks,void* out) {
    if(!ctx||!arguments(q,k,v,sq,sk,h,d,qs,ks,out))return ns_fail("Qwen image 1024 attention: invalid shape/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
    tiled<<<dim3(unsigned((sq+1)/2),unsigned(h)),256,size_t(2*sk)*sizeof(float),g->q>>>(
        static_cast<const __half*>(q),static_cast<const __half*>(k),static_cast<const __half*>(v),sq,sk,h,d,qs,ks,static_cast<__half*>(out));
    NS_LAUNCH_CHECK();return 0;
}
}
#endif
