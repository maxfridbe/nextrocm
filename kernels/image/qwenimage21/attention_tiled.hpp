// Private image DiT attention candidate; no TE/causal/shared dispatch changes.
// F16 Q/K/V -> F16 contiguous output, four queries per head/block.
// Preserve speech.hip ascending-QK FMA, runtime rsqrtf(D), 256-lane max/sum
// trees, expf scores, ascending-PV FMA, divide last and half RNE. No P-first
// half rounding, online softmax, split-K, reassociation or global scratch.
#pragma once
#ifdef __FAST_MATH__
#error "Qwen image attention requires baseline arithmetic"
#endif
#include <cstdint>
namespace nrqi_attention {
using I=int64_t;
inline bool supported(I sq,I sk,I h,I d,I qs,I ks) {
    return sq>=1&&sq<=512&&sk>=1&&sk<=512&&h>=1&&h<=32&&d==128
        &&qs>=h*d&&qs<=12288&&ks>=h*d&&ks<=12288;
}
inline I shared_bytes(I sk) {return (4*sk+128*65+4*128+4*256)*I(sizeof(float));}
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
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include "../../ns/ns_internal.hpp"
namespace nrqi_attention {
__global__ __launch_bounds__(256) void tiled(const __half* q,const __half* k,const __half* v,I sq,I sk,I h,I d,I qs,I ks,__half* out) {
    extern __shared__ float score[];
    __shared__ float keys[128][65],query[4][128],tmp[4][256];
    const int t=threadIdx.x,qi=t/64,ki=t%64;
    const I row0=I(blockIdx.x)*4,head=blockIdx.y;
    for(int z=t;z<512;z+=256) {
        const int r=z/128,a=z%128;
        query[r][a]=(row0+r<sq)?__half2float(q[(row0+r)*qs+head*128+a]):0.f;
    }
    __syncthreads();
    for(I base=0;base<sk;base+=64) {
        for(int z=t;z<8192;z+=256) {
            const int key=z/128,a=z%128;
            keys[a][key]=(base+key<sk)?__half2float(k[(base+key)*ks+head*128+a]):0.f;
        }
        __syncthreads();
        if(base+ki<sk) {
            float z=0;
            #pragma unroll
            for(int a=0;a<128;++a)z=fmaf(query[qi][a],keys[a][ki],z);
            z*=rsqrtf(float(d));score[qi*sk+base+ki]=z;
        }
        __syncthreads();
    }
    float best[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
    for(I j=t;j<sk;j+=256) {
        #pragma unroll
        for(int r=0;r<4;++r)best[r]=fmaxf(best[r],score[r*sk+j]);
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
    for(I j=t;j<sk;j+=256) {
        #pragma unroll
        for(int r=0;r<4;++r) {float z=expf(score[r*sk+j]-best[r]);score[r*sk+j]=z;sum[r]+=z;}
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
    if(row0+qi<sq) {
        float a=0,b=0;
        for(I j=0;j<sk;++j) {
            a=fmaf(score[qi*sk+j],__half2float(v[j*ks+head*128+ki]),a);
            b=fmaf(score[qi*sk+j],__half2float(v[j*ks+head*128+ki+64]),b);
        }
        out[((row0+qi)*h+head)*128+ki]=__float2half_rn(a/tmp[qi][0]);
        out[((row0+qi)*h+head)*128+ki+64]=__float2half_rn(b/tmp[qi][0]);
    }
}
inline int launch(void* ctx,const void* q,const void* k,const void* v,I sq,I sk,I h,I d,I qs,I ks,void* out) {
    if(!ctx||!arguments(q,k,v,sq,sk,h,d,qs,ks,out))return ns_fail("Qwen image attention: invalid shape/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
    tiled<<<dim3(unsigned((sq+3)/4),unsigned(h)),256,size_t(4*sk)*sizeof(float),g->q>>>(
        static_cast<const __half*>(q),static_cast<const __half*>(k),static_cast<const __half*>(v),sq,sk,h,d,qs,ks,static_cast<__half*>(out));
    NS_LAUNCH_CHECK();return 0;
}
}
#endif
