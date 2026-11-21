// Private diagnostic candidate: 64 pixels x64 channels, ascending padded16 FMA.
#pragma once
#include "vae.hpp"
#ifdef __HIPCC__
namespace nrqi_vae64 {
using I=int64_t;
__global__ __launch_bounds__(256) void conv(const __half*x,I h,I w,I ci,const __half*weight,I co,int k,int stride,const float*bias,__half*out,I oh,I ow){
 const int t=threadIdx.x,r=t/16,c=t%16;
 const I p0=I(blockIdx.x)*64,c0=I(blockIdx.y)*64,reduction=ci*k*k;
 const int pad=stride==1?k/2:0;
 __shared__ float xx[64][16],ww[16][65];
 float acc[4][4]={};
 for(I base=0;base<reduction;base+=16){
  for(int z=t;z<1024;z+=256){
   const I p=p0+z/16,j=base+z%16,ch=j/(k*k);
   const I iy=(p/ow)*stride-pad+(j/k)%k,ix=(p%ow)*stride-pad+j%k;
   xx[z/16][z%16]=(p<oh*ow&&j<reduction&&iy>=0&&iy<h&&ix>=0&&ix<w)?__half2float(x[(iy*w+ix)*ci+ch]):0.f;
   const I oc=c0+z/16;
   ww[z%16][z/16]=(oc<co&&j<reduction)?__half2float(weight[oc*reduction+j]):0.f;
  }
  __syncthreads();
  // Include all16 iterations in the final tile, exactly as the frozen baseline.
  #pragma unroll
  for(int j=0;j<16;++j){
   float v[4];
   #pragma unroll
   for(int b=0;b<4;++b)v[b]=ww[j][c+16*b];
   #pragma unroll
   for(int a=0;a<4;++a){const float q=xx[r+16*a][j];
    #pragma unroll
    for(int b=0;b<4;++b)acc[a][b]=fmaf(q,v[b],acc[a][b]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int a=0;a<4;++a){
  #pragma unroll
  for(int b=0;b<4;++b){const I p=p0+r+16*a,oc=c0+c+16*b;
   if(p<oh*ow&&oc<co)out[p*co+oc]=__float2half_rn(__fadd_rn(acc[a][b],bias?bias[oc]:0.f));
  }
 }
}
inline int launch(void*ctx,const void*x,I h,I w,I ci,const void*wt,I co,int k,int stride,const float*bias,void*out){
 if(!nrqi_vae::conv_valid(ctx,x,h,w,ci,wt,co,k,stride,bias,out))return ns_fail("VAE conv64: invalid geometry/span");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 const I oh=stride==1?h:(h-k)/2+1,ow=stride==1?w:(w-k)/2+1;
 conv<<<dim3(unsigned((oh*ow+63)/64),unsigned((co+63)/64)),256,0,g->q>>>(static_cast<const __half*>(x),h,w,ci,static_cast<const __half*>(wt),co,k,stride,bias,static_cast<__half*>(out),oh,ow);
 NS_LAUNCH_CHECK();return 0;
}
}
#endif
