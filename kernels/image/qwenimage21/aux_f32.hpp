// Private Qwen Image auxiliary Q4_K linear: X/W/Y F32, no half rounding.
// Separate from validated block F16 path; no production integration.
#pragma once
#include "packed_linear.hpp"
namespace nrqi_aux_f32 {
using I=int64_t;
inline bool valid(const void* ctx,const void* x,I m,I k,const void* w,int type,I n,void* y,I stride) {
    if(!ctx||m<1||m>65536||k<256||k>4096||k%256||n<1||n>16384||type!=12||stride<n||stride>65536)return false;
    nrqi_packed::Span xs,ws,ys;
    return nrqi_packed::span(x,uint64_t(m)*k,4,4,xs)&&
        nrqi_packed::span(w,uint64_t(n)*(k/256)*144,1,2,ws)&&
        nrqi_packed::span(y,uint64_t(m-1)*stride+n,4,4,ys)&&
        !nrqi_packed::overlap(xs,ys)&&!nrqi_packed::overlap(ws,ys);
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
namespace nrqi_aux_f32 {
using I = int64_t;
__device__ inline float half_at(const uint8_t* p) {
    return __half2float(*reinterpret_cast<const __half*>(p));
}
__device__ inline float weight(const uint8_t* row,I i) {
    const uint8_t* b=row+(i/256)*144;
    const int z=int(i%256);
    float value;
    const int g=z/32,l=z%32;const uint8_t* sc=b+4;
    const int s=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
    const int mn=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
    const int q=(b[16+(g/2)*32+l]>>(4*(g%2)))&15;
    // Explicit operations match independent GGML row decode, no reassociation.
    const float ds=__fmul_rn(half_at(b),float(s));
    const float dm=__fmul_rn(half_at(b+2),float(mn));
    value=__fsub_rn(__fmul_rn(ds,float(q)),dm);
    return value; // auxiliary source semantics retain decoded F32 weights
}
__global__ void tiled(const float* x,I m,I k,
                                                const uint8_t* w,I n,void* y,I stride) {
    constexpr int B=16,D=32;
    const int t=int(threadIdx.x),r=t/B,c=t%B;
    const I row=I(blockIdx.y)*B+r,col=I(blockIdx.x)*B+c;
    const I rb=(k/256)*144;
    __shared__ float xx[B][D];
    __shared__ float ww[D][B+1];
    float acc=0;
    for(I base=0;base<k;base+=D) {
        for(int a=t;a<B*D;a+=256) {
            const int rr=a/D,j=a%D;
            const I xr=I(blockIdx.y)*B+rr,wc=I(blockIdx.x)*B+rr;
            xx[rr][j]=xr<m?x[xr*k+base+j]:0.0f;
            ww[j][rr]=wc<n?weight(w+wc*rb,base+j):0.0f;
        }
        __syncthreads();
        #pragma unroll
        for(int j=0;j<D;++j) acc=fmaf(xx[r][j],ww[j][c],acc);
        __syncthreads();
    }
    if(row<m && col<n) static_cast<float*>(y)[row*stride+col]=acc;
}

inline int launch(void* ctx,const float* x,I m,I k,const void* w,int type,I n,float* y,I stride) {
    if(stride==0)stride=n;
    if(!valid(ctx,x,m,k,w,type,n,y,stride))return ns_fail("Qwen auxiliary F32: invalid geometry/type/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
    hipLaunchKernelGGL(tiled,dim3((n+15)/16,(m+15)/16),dim3(256),0,g->q,x,m,k,static_cast<const uint8_t*>(w),n,y,stride);
    NS_LAUNCH_CHECK();return 0;
}
}
#endif
