// Private per-matrix expansion experiment. No persistent weight cache.
// Frozen packed decoder -> F16 scratch -> ordered M128xN64 FMA.
#pragma once
#include "packed128.hpp"
namespace nrqi_expanded {
using I=int64_t;
inline I workspace(I k,I n){
 if(k<256||k>16384||k%256||n<1||n>16384||k*n>50331648)return 0;
 return k*n*2;
}
inline bool valid(const void*ctx,const void*x,I m,I k,const void*w,int type,I n,void*y,int odt,I stride,void*scratch,I bytes){
 if(stride==0)stride=n;
 const I need=workspace(k,n);
 if(!need||bytes<need||bytes>100663296||!nrqi_packed::valid(ctx,x,m,k,w,type,n,y,odt,stride))return false;
 nrqi_packed::Span ss,xs,ws,ys;
 return nrqi_packed::span(scratch,uint64_t(bytes),1,2,ss)&&
 nrqi_packed::span(x,uint64_t(m)*k,2,2,xs)&&
 nrqi_packed::span(w,uint64_t(n)*(k/256)*(type==12?144:210),1,2,ws)&&
 nrqi_packed::span(y,uint64_t(m-1)*stride+n,odt?2:4,odt?2:4,ys)&&
 !nrqi_packed::overlap(ss,xs)&&!nrqi_packed::overlap(ss,ws)&&!nrqi_packed::overlap(ss,ys);
}
}
#ifdef __HIPCC__
namespace nrqi_expanded {
template<int Type> __device__ inline __half decoded_half(const uint8_t* row,I i) {
    const uint8_t* b=row+(i/256)*(Type==12?144:210);
    const int z=int(i%256);
    float value;
    if constexpr (Type==12) {
        const int g=z/32,l=z%32;const uint8_t* sc=b+4;
        const int s=g<4?(sc[g]&63):((sc[g+4]&15)|((sc[g-4]>>6)<<4));
        const int mn=g<4?(sc[g+4]&63):((sc[g+4]>>4)|((sc[g]>>6)<<4));
        const int q=(b[16+(g/2)*32+l]>>(4*(g%2)))&15;
        // Explicit operations match independent GGML row decode, no reassociation.
        const float ds=__fmul_rn(nrqi_packed::half_at(b),float(s));
        const float dm=__fmul_rn(nrqi_packed::half_at(b+2),float(mn));
        value=__fsub_rn(__fmul_rn(ds,float(q)),dm);
    } else {
        const int h=z/128,g=(z%128)/32,l=z%32;
        const int lo=(b[h*64+(g%2)*32+l]>>(4*(g/2)))&15;
        const int hi=(b[128+h*32+l]>>(2*g))&3;
        const int scale=int(reinterpret_cast<const int8_t*>(b+192)[h*8+g*2+l/16]);
        value=__fmul_rn(__fmul_rn(nrqi_packed::half_at(b+208),float(scale)),float((lo|(hi<<4))-32));
    }
    // Register identity boundary forbids backend mul+half -> mixed FMA(+0).
    // Preserve the source-required F32 rounded product before F16 rounding.
    asm volatile("" : "+v"(value));
    return __float2half_rn(value);
}
template<int Type> __global__ void expand(const uint8_t*w,I k,I n,__half*dst){
 const I i=I(blockIdx.x)*256+threadIdx.x;
 if(i<k*n){const I row=i/k,j=i%k,rb=(k/256)*(Type==12?144:210);
 dst[i]=decoded_half<Type>(w+row*rb,j);}
}
template<int Odt>
__global__ __launch_bounds__(256) void tiled(const __half*x,I m,I k,const __half*w,I n,void*y,I stride){
 const int t=threadIdx.x,r=t/16,c=t%16;
 const I row0=I(blockIdx.y)*128,col0=I(blockIdx.x)*64;
 __shared__ float xx[128][32],ww[32][65];
 float acc[8][4]={};
 for(I base=0;base<k;base+=32){
  for(int z=t;z<4096;z+=256){int row=z/32,j=z%32;xx[row][j]=row0+row<m?__half2float(x[(row0+row)*k+base+j]):0.f;}
  for(int z=t;z<2048;z+=256){int col=z/32,j=z%32;ww[j][col]=col0+col<n?__half2float(w[(col0+col)*k+base+j]):0.f;}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<32;++j){
   float wv[4];
   #pragma unroll
   for(int b=0;b<4;++b)wv[b]=ww[j][c+16*b];
   #pragma unroll
   for(int a=0;a<8;++a){const float xv=xx[r+16*a][j];
    #pragma unroll
    for(int b=0;b<4;++b)acc[a][b]=fmaf(xv,wv[b],acc[a][b]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int a=0;a<8;++a){
  #pragma unroll
  for(int b=0;b<4;++b){I row=row0+r+16*a,col=col0+c+16*b;
   if(row<m&&col<n){if constexpr(Odt==0)static_cast<float*>(y)[row*stride+col]=acc[a][b];else static_cast<__half*>(y)[row*stride+col]=__float2half_rn(acc[a][b]);}
  }
 }
}
inline int launch(void*ctx,const void*x,I m,I k,const void*w,int type,I n,void*y,int odt,I stride,void*scratch,I bytes){
 if(stride==0)stride=n;
 if(!valid(ctx,x,m,k,w,type,n,y,odt,stride,scratch,bytes))return ns_fail("Qwen expanded: invalid shape/span/alias/workspace");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 const dim3 eg(unsigned((k*n+255)/256)),grid(unsigned((n+63)/64),unsigned((m+127)/128)),block(256);
 if(type==12){hipLaunchKernelGGL((expand<12>),eg,block,0,g->q,static_cast<const uint8_t*>(w),k,n,static_cast<__half*>(scratch));}
 else{hipLaunchKernelGGL((expand<14>),eg,block,0,g->q,static_cast<const uint8_t*>(w),k,n,static_cast<__half*>(scratch));}
 NS_LAUNCH_CHECK();
 if(odt==0){hipLaunchKernelGGL((tiled<0>),grid,block,0,g->q,static_cast<const __half*>(x),m,k,static_cast<const __half*>(scratch),n,y,stride);}
 else{hipLaunchKernelGGL((tiled<1>),grid,block,0,g->q,static_cast<const __half*>(x),m,k,static_cast<const __half*>(scratch),n,y,stride);}
 NS_LAUNCH_CHECK();return 0;
}
}
#endif
