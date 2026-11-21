// Private M128 x N64 register tile; unchanged packed->half decoder/FMA order.
// Twice the activation rows reuse each decoded weight tile versus Register64.
// No full-weight expansion, persistent scratch, host dispatch or default change.
#pragma once
#include "packed64.hpp"
#ifdef __HIPCC__
namespace nrqi_packed128 {
using I=int64_t;
template<int Type,int Odt>
__global__ __launch_bounds__(256) void tiled(const __half*x,I m,I k,const uint8_t*w,I n,void*y,I stride){
 const int t=threadIdx.x,r=t/16,c=t%16;
 const I row0=I(blockIdx.y)*128,col0=I(blockIdx.x)*64;
 const I rb=(k/256)*(Type==12?144:210);
 __shared__ float xx[128][32],ww[32][65];
 float acc[8][4]={};
 for(I base=0;base<k;base+=32){
  for(int z=t;z<4096;z+=256){int row=z/32,j=z%32;xx[row][j]=row0+row<m?__half2float(x[(row0+row)*k+base+j]):0.f;}
  for(int z=t;z<2048;z+=256){int col=z/32,j=z%32;ww[j][col]=col0+col<n?nrqi_packed::weight<Type>(w+(col0+col)*rb,base+j):0.f;}
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
inline int launch(void*ctx,const void*x,I m,I k,const void*w,int type,I n,void*y,int odt,I stride){
 if(stride==0)stride=n;
 if(!nrqi_packed::valid(ctx,x,m,k,w,type,n,y,odt,stride))return ns_fail("Qwen packed128: invalid geometry/type/span/alias");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 const dim3 grid(unsigned((n+63)/64),unsigned((m+127)/128)),block(256);
#define QI128(T,O) hipLaunchKernelGGL((tiled<T,O>),grid,block,0,g->q,static_cast<const __half*>(x),m,k,static_cast<const uint8_t*>(w),n,y,stride)
 if(type==12){if(odt==0){QI128(12,0);}else{QI128(12,1);}}else{if(odt==0){QI128(14,0);}else{QI128(14,1);}}
#undef QI128
 NS_LAUNCH_CHECK();return 0;
}
}
#endif
