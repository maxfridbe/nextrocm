// Exact32 private attention composition; uses unchanged GPU-validated source softmax.
// CPU-proved S32 QK/PV order; unchanged rounded score division/source softmax.
#pragma once
#define qv_softmax_source32 qv_order32_softmax
#include "softmax_source32.hpp"
#undef qv_softmax_source32
namespace qv_attention_order32 {
constexpr size_t RECORD=163;
inline bool valid(void*ctx,const float*qkv,int64_t rows,float*out,float*records,uint32_t*status){
 if(!ctx||rows!=32)return false;
 const auto span=[](const void*p,size_t n,uintptr_t&b,uintptr_t&e){b=reinterpret_cast<uintptr_t>(p);if(!p||b%4||n>SIZE_MAX/4||n*4>UINTPTR_MAX-b)return false;e=b+n*4;return true;};
 uintptr_t b[4],e[4];unsigned count=3;
 if(!span(qkv,32*6144,b[0],e[0])||!span(out,32*4096,b[1],e[1])||!span(status,1,b[2],e[2]))return false;
 if(records){if(!span(records,32*32*RECORD,b[3],e[3]))return false;++count;}
 for(unsigned i=0;i<count;++i)for(unsigned j=i+1;j<count;++j)if(!(e[i]<=b[j]||e[j]<=b[i]))return false;
 return true;
}
}
#ifdef __HIPCC__
namespace qv_attention_order32 {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
__global__ void kernel(const float*qkv,int64_t d,float divisor,float*out,float*records,uint32_t*status){
 const unsigned t=threadIdx.x;const size_t r=blockIdx.x,h=blockIdx.y,kh=h/4;
 __shared__ float logits[32],ex[32],prob[32],maximum,den,inverse;
 __shared__ unsigned bad;
 if(t==0)bad=0;__syncthreads();
 float raw=0.f,scaled=-INFINITY;
 float even=0.f,odd=0.f;
 // Compute every QK score, including masked keys, for full-original proof.
 for(int64_t a=0;a<d;++a){
  const float q=qkv[r*6144+h*d+a],k=qkv[t*6144+4096+kh*d+a],v=qkv[t*6144+5120+kh*d+a];
  if(!isfinite(q)||!isfinite(k)||!isfinite(v))atomicOr(&bad,1u);
  if(r<30)raw=fmaf(q,k,raw);
  else {const float product=q*k;if(a&1)odd=odd+product;else even=even+product;}
 }
 if(r>=30)raw=even+odd;
 raw=raw*1.f;
 if(!isfinite(raw))atomicOr(&bad,2u);
 if(t<=r){scaled=raw/divisor;if(!isfinite(scaled))atomicOr(&bad,2u);}

 logits[t]=scaled;__syncthreads();
 if(bad){if(t==0)atomicOr(status,bad);return;}
 if(t==0)maximum=qv_order32_softmax::reduce(logits,true);__syncthreads();
 const float shifted=scaled-maximum;
 const float exponential=qv_order32_softmax::exp_nonpositive(shifted);ex[t]=exponential;
 __syncthreads();
 if(t==0){den=qv_order32_softmax::reduce(ex,false);inverse=1.f/den;}__syncthreads();
 const float p=exponential*inverse;prob[t]=p;
 if(!isfinite(maximum)||!isfinite(den)||den<=0||!isfinite(inverse)||inverse<=0||(!isfinite(shifted)&&shifted!=-INFINITY)||!isfinite(exponential)||exponential<0||!isfinite(p)||p<0)atomicOr(&bad,2u);
 __syncthreads();
 if(bad){if(t==0)atomicOr(status,bad);return;}
 float values[4];
 for(unsigned f=0;f<4;++f){
  const size_t a=t+32*f;float z=0.f,e=0.f,o=0.f;
  for(size_t j=0;j<32;++j){const float v=qkv[j*6144+5120+kh*d+a];
   if(r<30)z=fmaf(prob[j],v,z);
   else {const float product=prob[j]*v;if(j&1)o=o+product;else e=e+product;}
  }
  if(r>=30)z=e+o;
  z=z*1.f;values[f]=z;if(!isfinite(z))atomicOr(&bad,2u);
 }
 __syncthreads();
 if(bad){if(t==0)atomicOr(status,bad);return;}
 for(unsigned f=0;f<4;++f)out[(r*32+h)*d+t+32*f]=values[f];
 if(records){float*rec=records+(r*32+h)*RECORD;rec[t]=raw;rec[32+t]=scaled;rec[65+t]=shifted;rec[97+t]=exponential;rec[131+t]=p;if(t==0){rec[64]=maximum;rec[129]=den;rec[130]=inverse;}}
}
inline int launch(void*ctx,const float*qkv,int64_t rows,float*out,float*records,uint32_t*status){
 if(!valid(ctx,qkv,rows,out,records,status))return ns_fail("nrqv_attention_order32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));NS_CU(hipMemsetAsync(status,0,4,g->q));
 kernel<<<dim3(32,32),32,0,g->q>>>(qkv,128,0x1.6a09e6p+3f,out,records,status);NS_LAUNCH_CHECK();return 0;
}
}
#endif
