// Private opt-in 64x64 register-tiled block-half candidate.
// Reuses the validated packed decoder and guards without changing the baseline.
#pragma once
#include "packed_linear.hpp"
#ifdef __HIPCC__
namespace nrqi_packed64 {
using I=int64_t;
template<int Type,int Odt>
__global__ void tiled(const __half* x,I m,I k,const uint8_t* w,I n,void* y,I stride) {
    constexpr int B=64,D=32;
    const int t=int(threadIdx.x),r=t/16,c=t%16;
    const I row0=I(blockIdx.y)*B,col0=I(blockIdx.x)*B;
    const I rb=(k/256)*(Type==12?144:210);
    __shared__ float xx[B][D];
    __shared__ float ww[D][B+1];
    float sum[4][4]={};
    for(I base=0;base<k;base+=D) {
        // Each of2048 activation/weight tile elements has exactly one owner.
        for(int a=t;a<B*D;a+=256) {
            const int p=a/D,j=a%D;
            xx[p][j]=row0+p<m?__half2float(x[(row0+p)*k+base+j]):0.0f;
            ww[j][p]=col0+p<n?nrqi_packed::weight<Type>(w+(col0+p)*rb,base+j):0.0f;
        }
        __syncthreads();
        #pragma unroll
        for(int j=0;j<D;++j) {
            float xv[4],wv[4];
            #pragma unroll
            for(int a=0;a<4;++a) {xv[a]=xx[r+16*a][j];wv[a]=ww[j][c+16*a];}
            #pragma unroll
            for(int a=0;a<4;++a) {
                #pragma unroll
                for(int b=0;b<4;++b) sum[a][b]=fmaf(xv[a],wv[b],sum[a][b]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int a=0;a<4;++a) {
        #pragma unroll
        for(int b=0;b<4;++b) {
            const I row=row0+r+16*a,col=col0+c+16*b;
            if(row<m && col<n) {
                if constexpr(Odt==0) static_cast<float*>(y)[row*stride+col]=sum[a][b];
                else static_cast<__half*>(y)[row*stride+col]=__float2half_rn(sum[a][b]);
            }
        }
    }
}
inline int launch(void* ctx,const void* x,I m,I k,const void* w,int type,I n,void* y,int odt,I stride) {
    if(stride==0)stride=n;
    if(!nrqi_packed::valid(ctx,x,m,k,w,type,n,y,odt,stride))return ns_fail("Qwen packed64: invalid geometry/type/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
    const dim3 grid((n+63)/64,(m+63)/64),block(256);
#define NRQI64_LAUNCH(T,O) hipLaunchKernelGGL((tiled<T,O>),grid,block,0,g->q,static_cast<const __half*>(x),m,k,static_cast<const uint8_t*>(w),n,y,stride)
    if(type==12) {if(odt==0){NRQI64_LAUNCH(12,0);}else{NRQI64_LAUNCH(12,1);}}
    else {if(odt==0){NRQI64_LAUNCH(14,0);}else{NRQI64_LAUNCH(14,1);}}
#undef NRQI64_LAUNCH
    NS_LAUNCH_CHECK();return 0;
}
}
#endif
