// Private F32 probability-first causal GQA, fixed Q32/KV8/D128.
// Preserve baseline ascending QK FMA, runtime scale and 256lane softmax tree.
// The sole intended arithmetic change is F32 normalization before ascending PV.
#pragma once
#include "int8_convrot.hpp"
namespace qv_prob {
inline bool valid(void*ctx,const float*qkv,int64_t rows,float*out){
 if(!ctx||rows<1||rows>4096)return false;
 qv_int8::Span a,b;
 return qv_int8::span(qkv,size_t(rows)*6144,4,a)&&qv_int8::span(out,size_t(rows)*4096,4,b)&&qv_int8::separate(a,b);
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "Probability-first TE attention requires F32 arithmetic"
#endif
namespace qv_prob {
__global__ void kernel(const float*qkv,int64_t d,float*out){
 extern __shared__ float score[];
 __shared__ float tmp[256];
 const unsigned t=threadIdx.x;const int64_t r=blockIdx.x,h=blockIdx.y,kh=h/4,end=r+1;
 float best=-INFINITY;
 for(int64_t j=t;j<end;j+=256){
  float z=0;
  for(int64_t a=0;a<d;++a)z=fmaf(qkv[r*6144+h*d+a],qkv[j*6144+4096+kh*d+a],z);
  z*=rsqrtf(float(d));score[j]=z;best=fmaxf(best,z);
 }
 tmp[t]=best;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s)tmp[t]=fmaxf(tmp[t],tmp[t+s]);__syncthreads();}
 best=tmp[0];__syncthreads();float sum=0;
 for(int64_t j=t;j<end;j+=256){float z=expf(score[j]-best);score[j]=z;sum+=z;}
 tmp[t]=sum;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}
 const float den=tmp[0];
 for(int64_t j=t;j<end;j+=256)score[j]=__fdiv_rn(score[j],den);
 __syncthreads();
 for(int64_t a=t;a<d;a+=256){float z=0;for(int64_t j=0;j<end;++j)z=fmaf(score[j],qkv[j*6144+5120+kh*d+a],z);out[(r*32+h)*d+a]=z;}
}
inline int launch(void*ctx,const float*qkv,int64_t rows,float*out){
 if(!valid(ctx,qkv,rows,out))return ns_fail("nrqv_attention_prob_f32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 kernel<<<dim3(unsigned(rows),32),256,size_t(rows)*4,g->q>>>(qkv,128,out);
 NS_LAUNCH_CHECK();return 0;
}
}
#endif
