// Private standalone S32 softmax; not wired into encoder or shared operators.
// Exponential derived from SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors 2010 - 2024.
// Boost Software License 1.0: see softmax_source32-LICENSE.txt (full notice).
// ATen reduction semantics: see softmax_source32-Torch-LICENSE.txt.
#pragma once
#include <cstddef>
#include <cstdint>
namespace qv_softmax_source32 {
constexpr size_t RECORD=99;
// row record: maximum0, shifted1..32, exp33..64, sum65, inverse66, P67..98.
inline bool valid(void*ctx,const float*x,int64_t rows,int64_t width,float*out,uint32_t*status){
 if(!ctx||rows<1||rows>4096||width!=32)return false;
 const auto span=[](const void*p,size_t n,uintptr_t&begin,uintptr_t&end){begin=reinterpret_cast<uintptr_t>(p);if(!p||begin%4||n>SIZE_MAX/4||n*4>UINTPTR_MAX-begin)return false;end=begin+n*4;return true;};
 uintptr_t b[3],e[3];
 if(!span(x,size_t(rows)*32,b[0],e[0])||!span(out,size_t(rows)*RECORD,b[1],e[1])||!span(status,1,b[2],e[2]))return false;
 for(unsigned i=0;i<3;++i)for(unsigned j=i+1;j<3;++j)if(!(e[i]<=b[j]||e[j]<=b[i]))return false;
 return true;
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "source softmax requires strict FP"
#endif
namespace qv_softmax_source32 {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
__device__ inline float fb(uint32_t b){return __uint_as_float(b);}
__device__ inline float exp_nonpositive(float d){
 if(d < -104.f)return 0.f;
 const float z=d*fb(0x3fb8aa3bu);const int q=__float2int_rn(z);
 float s=fmaf(float(q),-fb(0x3f317200u),d);s=fmaf(float(q),-fb(0x35bfbe8eu),s);
 float u=fb(0x39502bdau);
 u=fmaf(u,s,fb(0x3ab696c9u));u=fmaf(u,s,fb(0x3c0888a6u));u=fmaf(u,s,fb(0x3d2aaa7au));u=fmaf(u,s,fb(0x3e2aaaabu));u=fmaf(u,s,fb(0x3f000000u));
 const float square=s*s;u=1.f+fmaf(square,u,s);
 const int a=q>=0?q/2:-((-q+1)/2),b=q-a;
 const float scaled=u*fb(uint32_t(a+127)<<23);return scaled*fb(uint32_t(b+127)<<23);
}
__device__ inline float max_second(float a,float b){return a>b?a:b;}
__device__ inline float reduce(const float*x,bool maximum){
 float a[8];for(unsigned i=0;i<8;++i)a[i]=x[i];
 for(unsigned j=8;j<32;j+=8)for(unsigned i=0;i<8;++i)a[i]=maximum?max_second(a[i],x[j+i]):a[i]+x[j+i];
 for(unsigned step=4;step;step>>=1){float old[8];for(unsigned i=0;i<8;++i)old[i]=a[i];for(unsigned i=0;i<8;++i)a[i]=maximum?max_second(old[i],old[i^step]):old[i]+old[i^step];}
 return a[0];
}
__global__ void kernel(const float*input,float*records,uint32_t*status){
 const unsigned t=threadIdx.x;const size_t r=blockIdx.x;
 __shared__ float x[32],ex[32],maximum,den,inverse;
 __shared__ unsigned bad,finite;
 if(t==0){bad=0;finite=0;}__syncthreads();
 const float v=input[r*32+t];x[t]=v;
 if(isnan(v)||v==INFINITY)atomicOr(&bad,1u);
 if(isfinite(v))atomicOr(&finite,1u);
 __syncthreads();
 if(bad||!finite){if(t==0)atomicOr(status,1u);return;}
 if(t==0)maximum=reduce(x,true);__syncthreads();
 const float shifted=v-maximum;const float exponential=exp_nonpositive(shifted);ex[t]=exponential;
 __syncthreads();
 if(t==0){den=reduce(ex,false);inverse=1.f/den;}__syncthreads();
 const float p=exponential*inverse;
 if(!isfinite(maximum)||!isfinite(den)||den<=0||!isfinite(inverse)||inverse<=0||(!isfinite(shifted)&&shifted!=-INFINITY)||!isfinite(exponential)||exponential<0||!isfinite(p)||p<0)atomicOr(&bad,2u);
 __syncthreads();
 if(bad){if(t==0)atomicOr(status,2u);return;}
 float*out=records+r*RECORD;
 if(t==0){out[0]=maximum;out[65]=den;out[66]=inverse;}
 out[1+t]=shifted;out[33+t]=exponential;out[67+t]=p;
}
inline int launch(void*ctx,const float*x,int64_t rows,int64_t width,float*out,uint32_t*status){
 if(!valid(ctx,x,rows,width,out,status))return ns_fail("nrqv_softmax_source32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 NS_CU(hipMemsetAsync(status,0,sizeof(uint32_t),g->q));
 kernel<<<dim3(unsigned(rows)),32,0,g->q>>>(x,out,status);NS_LAUNCH_CHECK();return 0;
}
}
#endif
