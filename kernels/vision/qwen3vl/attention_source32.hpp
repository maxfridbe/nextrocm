// Exact32 private attention composition; uses unchanged GPU-validated source softmax.
// No shared kernel changes. No change to ascending QK/PV FMA or rounded score division.
#pragma once
#include "softmax_source32.hpp"
namespace qv_attention_source32 {
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
namespace qv_attention_source32 {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
__global__ void kernel(const float*qkv,int64_t d,float divisor,float*out,float*records,uint32_t*status){
 const unsigned t=threadIdx.x;const size_t r=blockIdx.x,h=blockIdx.y,kh=h/4;
 __shared__ float logits[32],ex[32],prob[32],maximum,den,inverse;
 __shared__ unsigned bad;
 if(t==0)bad=0;__syncthreads();
 float raw=0.f,scaled=-INFINITY;
 if(t<=r){
  for(int64_t a=0;a<d;++a){const float q=qkv[r*6144+h*d+a],k=qkv[t*6144+4096+kh*d+a],v=qkv[t*6144+5120+kh*d+a];if(!isfinite(q)||!isfinite(k)||!isfinite(v))atomicOr(&bad,1u);raw=fmaf(q,k,raw);}
  scaled=raw/divisor;if(!isfinite(raw)||!isfinite(scaled))atomicOr(&bad,2u);
 }
 logits[t]=scaled;__syncthreads();
 if(bad){if(t==0)atomicOr(status,bad);return;}
 if(t==0)maximum=qv_softmax_source32::reduce(logits,true);__syncthreads();
 const float shifted=scaled-maximum;
 const float exponential=qv_softmax_source32::exp_nonpositive(shifted);ex[t]=exponential;
 __syncthreads();
 if(t==0){den=qv_softmax_source32::reduce(ex,false);inverse=1.f/den;}__syncthreads();
 const float p=exponential*inverse;prob[t]=p;
 if(!isfinite(maximum)||!isfinite(den)||den<=0||!isfinite(inverse)||inverse<=0||(!isfinite(shifted)&&shifted!=-INFINITY)||!isfinite(exponential)||exponential<0||!isfinite(p)||p<0)atomicOr(&bad,2u);
 __syncthreads();
 if(bad){if(t==0)atomicOr(status,bad);return;}
 float values[4];
 for(unsigned f=0;f<4;++f){const size_t a=t+32*f;float z=0;for(size_t j=0;j<=r;++j)z=fmaf(prob[j],qkv[j*6144+5120+kh*d+a],z);values[f]=z;if(!isfinite(z))atomicOr(&bad,2u);}
 __syncthreads();
 if(bad){if(t==0)atomicOr(status,bad);return;}
 for(unsigned f=0;f<4;++f)out[(r*32+h)*d+t+32*f]=values[f];
 if(records){float*rec=records+(r*32+h)*RECORD;rec[t]=raw;rec[32+t]=scaled;rec[65+t]=shifted;rec[97+t]=exponential;rec[131+t]=p;if(t==0){rec[64]=maximum;rec[129]=den;rec[130]=inverse;}}
}
inline int launch(void*ctx,const float*qkv,int64_t rows,float*out,float*records,uint32_t*status){
 if(!valid(ctx,qkv,rows,out,records,status))return ns_fail("nrqv_attention_source32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));NS_CU(hipMemsetAsync(status,0,4,g->q));
 kernel<<<dim3(32,32),32,0,g->q>>>(qkv,128,0x1.6a09e6p+3f,out,records,status);NS_LAUNCH_CHECK();return 0;
}
}
#endif
