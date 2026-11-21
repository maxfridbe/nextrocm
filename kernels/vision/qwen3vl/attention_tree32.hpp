// Private exact32-token denominator-order candidate. Baseline headers stay frozen.
#pragma once
#include <cstddef>
#include <cstdint>
namespace qv_tree32 {
inline bool valid(void*ctx,const float*qkv,int64_t rows,float*out){
 if(!ctx||rows!=32)return false;
 const auto span=[](const void*p,size_t n,uintptr_t&e){const auto b=reinterpret_cast<uintptr_t>(p);if(!p||b%4||n>SIZE_MAX/4||n*4>UINTPTR_MAX-b)return false;e=b+n*4;return true;};
 uintptr_t ae,be;if(!span(qkv,size_t(rows)*6144,ae)||!span(out,size_t(rows)*4096,be))return false;
 return ae<=reinterpret_cast<uintptr_t>(out)||be<=reinterpret_cast<uintptr_t>(qkv);
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "Probability-first TE attention requires F32 arithmetic"
#endif
namespace qv_tree32 {
__global__ void kernel(const float*qkv,int64_t d,float divisor,float*out){
 extern __shared__ float score[];
 __shared__ float tmp[256];
 const unsigned t=threadIdx.x;const int64_t r=blockIdx.x,h=blockIdx.y,kh=h/4,end=r+1;
 float best=-INFINITY;
 for(int64_t j=t;j<end;j+=256){
  float z=0;
  for(int64_t a=0;a<d;++a)z=fmaf(qkv[r*6144+h*d+a],qkv[j*6144+4096+kh*d+a],z);
  z=__fdiv_rn(z,divisor);score[j]=z;best=fmaxf(best,z);
 }
 tmp[t]=best;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s)tmp[t]=fmaxf(tmp[t],tmp[t+s]);__syncthreads();}
 best=tmp[0];__syncthreads();
 // Full sequence32, not causal prefix: masked exp slots contribute zero.
 for(int64_t j=t;j<end;j+=256){float z=expf(score[j]-best);score[j]=z;}
 if(t<32 && int64_t(t)>=end)score[t]=0.f;
 __syncthreads();
 if(t<8){
  float sum=score[t];
  sum+=score[t+8];sum+=score[t+16];sum+=score[t+24];
  tmp[t]=sum;
 }
 __syncthreads();
 // Equivalent lane0 result of source AVX2 shuffle4/2/1 additions.
 if(t==0){
  float a=tmp[0]+tmp[4],b=tmp[2]+tmp[6];
  float c=tmp[1]+tmp[5],d1=tmp[3]+tmp[7];
  a=a+b;c=c+d1;tmp[0]=a+c;
 }
 __syncthreads();
 const float den=tmp[0];
 for(int64_t j=t;j<end;j+=256)score[j]=__fdiv_rn(score[j],den);
 __syncthreads();
 for(int64_t a=t;a<d;a+=256){float z=0;for(int64_t j=0;j<end;++j)z=fmaf(score[j],qkv[j*6144+5120+kh*d+a],z);out[(r*32+h)*d+a]=z;}
}
inline int launch(void*ctx,const float*qkv,int64_t rows,float*out){
 if(!valid(ctx,qkv,rows,out))return ns_fail("nrqv_attention_tree32_f32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 kernel<<<dim3(unsigned(rows),32),256,size_t(rows)*4,g->q>>>(qkv,128,0x1.6a09e6p+3f,out);
 NS_LAUNCH_CHECK();return 0;
}
}
#endif
