// Private Qwen Image packed Q4_K/Q6_K linear prototype; no production registration.
// Source contract: nextsycl 98124afb... kernels/diffusion/nsd.cpp nsd_linear_q:
// packed -> F16 weights, F16 activations, F32 dot, optional final F16 rounding.
// Each CTA decodes only a 16x32 weight tile. No global expanded weights/scratch.
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>

namespace nrqi_packed {
using I = int64_t;
struct Span { uintptr_t lo=0, hi=0; };
inline bool span(const void* p, uint64_t count, unsigned bytes, unsigned alignment, Span& s) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(p);
    if (!p || !count || a%alignment || count>UINTPTR_MAX/bytes) return false;
    const uintptr_t len=uintptr_t(count)*bytes;
    if (len>UINTPTR_MAX-a) return false;
    s={a,a+len}; return true;
}
inline bool overlap(Span a,Span b) {return a.lo<b.hi && b.lo<a.hi;}
inline bool valid(const void* ctx,const void* x,I m,I k,const void* w,int type,I n,
                  void* y,int odt,I stride) {
    // Bounds are a private prototype envelope, not a model admission/fit claim.
    if (!ctx || m<1 || m>65536 || k<256 || k>16384 || k%256 || n<1 || n>16384 ||
        (type!=12 && type!=14) || (odt!=0 && odt!=1) || stride<n || stride>65536) return false;
    Span xs,ws,ys;
    return span(x,uint64_t(m)*k,2,2,xs) &&
        span(w,uint64_t(n)*(k/256)*(type==12?144:210),1,2,ws) &&
        span(y,uint64_t(m-1)*stride+n,odt==0?4:2,odt==0?4:2,ys) &&
        !overlap(ys,xs) && !overlap(ys,ws);
}
}

#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
namespace nrqi_packed {
__device__ inline float half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}
template<int Type> __device__ inline float weight(const uint8_t* row,I i) {
    const uint8_t* b=row+(i/256)*(Type==12?144:210);
    const int z=int(i%256);
    float value;
    if constexpr (Type==12) {
        const int g=z/32,l=z%32;const uint8_t* sc=b+4;
        const int s=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
        const int mn=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
        const int q=(b[16+(g/2)*32+l]>>(4*(g%2)))&15;
        // Explicit operations match independent GGML row decode, no reassociation.
        const float ds=__fmul_rn(half_at(b),float(s));
        const float dm=__fmul_rn(half_at(b+2),float(mn));
        value=__fsub_rn(__fmul_rn(ds,float(q)),dm);
    } else {
        const int h=z/128,g=(z%128)/32,l=z%32;
        const int lo=(b[h*64+(g%2)*32+l]>>(4*(g/2)))&15;
        const int hi=(b[128+h*32+l]>>(2*g))&3;
        const int scale=int(reinterpret_cast<const int8_t*>(b+192)[h*8+g*2+l/16]);
        value=__fmul_rn(__fmul_rn(half_at(b+208),float(scale)),float((lo|(hi<<4))-32));
    }
    return __half2float(__float2half_rn(value));
}
template<int Type,int Odt> __global__ void tiled(const __half* x,I m,I k,
                                                const uint8_t* w,I n,void* y,I stride) {
    constexpr int B=16,D=32;
    const int t=int(threadIdx.x),r=t/B,c=t%B;
    const I row=I(blockIdx.y)*B+r,col=I(blockIdx.x)*B+c;
    const I rb=(k/256)*(Type==12?144:210);
    __shared__ float xx[B][D];
    __shared__ float ww[D][B+1];
    float acc=0;
    for(I base=0;base<k;base+=D) {
        for(int a=t;a<B*D;a+=256) {
            const int rr=a/D,j=a%D;
            const I xr=I(blockIdx.y)*B+rr,wc=I(blockIdx.x)*B+rr;
            xx[rr][j]=xr<m?__half2float(x[xr*k+base+j]):0.0f;
            ww[j][rr]=wc<n?weight<Type>(w+wc*rb,base+j):0.0f;
        }
        __syncthreads();
        #pragma unroll
        for(int j=0;j<D;++j) acc=fmaf(xx[r][j],ww[j][c],acc);
        __syncthreads();
    }
    if(row<m && col<n) {
        if constexpr(Odt==0) static_cast<float*>(y)[row*stride+col]=acc;
        else static_cast<__half*>(y)[row*stride+col]=__float2half_rn(acc);
    }
}
inline int launch(void* ctx,const void* x,I m,I k,const void* w,int type,I n,void* y,int odt,I stride) {
    if(stride==0) stride=n;
    if(!valid(ctx,x,m,k,w,type,n,y,odt,stride)) return ns_fail("Qwen packed linear: invalid geometry/type/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);
    NS_CU(hipSetDevice(g->dev));
    const dim3 grid((n+15)/16,(m+15)/16),block(256);
#define NRQI_LAUNCH(T,O) hipLaunchKernelGGL((tiled<T,O>),grid,block,0,g->q,static_cast<const __half*>(x),m,k,static_cast<const uint8_t*>(w),n,y,stride)
    if(type==12) {if(odt==0) {NRQI_LAUNCH(12,0);} else {NRQI_LAUNCH(12,1);}}
    else {if(odt==0) {NRQI_LAUNCH(14,0);} else {NRQI_LAUNCH(14,1);}}
#undef NRQI_LAUNCH
    NS_LAUNCH_CHECK(); return 0;
}
}
#endif
