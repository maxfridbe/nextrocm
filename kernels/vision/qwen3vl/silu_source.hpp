// Private source SiLU operator, standalone validation only; no shared dispatch.
// SLEEF5a1d179, Copyright Naoki Shibata and contributors 2010-2024.
// Full notices: softmax_source32-LICENSE.txt and softmax_source32-Torch-LICENSE.txt.
#pragma once
#include <cstddef>
#include <cstdint>
namespace qv_silu_source {
// Fused input [rows,2*cols] gate/up; output [rows,cols]. Optional capture
// [rows*cols,4]: exp(-gate), rounded denominator, rounded SiLU, rounded product.
// IEEE nonfinite outputs are intentional; NaN category only, no payload promise.
inline bool valid(void*ctx,const float*x,int64_t rows,int64_t cols,float*out,float*records){
 if(!ctx||rows<1||rows>4096||cols<1||cols>12288||uint64_t(rows)*uint64_t(cols)>4194304)return false;
 size_t n=size_t(rows)*size_t(cols);uintptr_t b[3],e[3];
 const auto span=[](const void*p,size_t n,uintptr_t&b,uintptr_t&e){b=reinterpret_cast<uintptr_t>(p);if(!p||b%4||n>SIZE_MAX/4||n*4>UINTPTR_MAX-b)return false;e=b+n*4;return true;};
 if(!span(x,n*2,b[0],e[0])||!span(out,n,b[1],e[1]))return false;
 const unsigned count=records?3:2;if(records&&!span(records,n*4,b[2],e[2]))return false;
 for(unsigned i=0;i<count;++i)for(unsigned j=i+1;j<count;++j)if(!(e[i]<=b[j]||e[j]<=b[i]))return false;
 return true;
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "source SiLU requires strict FP"
#endif
namespace qv_silu_source {
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
__device__ inline float fb(uint32_t b){return __uint_as_float(b);}
__device__ inline float exp_two_sided(float d){
 if(isnan(d))return fb(0x7fc00000u);
 if(d>100.f)return INFINITY;
 if(d< -104.f)return 0.f;
 const float z=d*fb(0x3fb8aa3bu);const int q=__float2int_rn(z);
 float s=fmaf(float(q),-fb(0x3f317200u),d);s=fmaf(float(q),-fb(0x35bfbe8eu),s);
 float u=fb(0x39502bdau);
 u=fmaf(u,s,fb(0x3ab696c9u));u=fmaf(u,s,fb(0x3c0888a6u));u=fmaf(u,s,fb(0x3d2aaa7au));u=fmaf(u,s,fb(0x3e2aaaabu));u=fmaf(u,s,fb(0x3f000000u));
 const float square=s*s;u=1.f+fmaf(square,u,s);
 const int a=q>=0?q/2:-((-q+1)/2),b=q-a;
 const float scaled=u*fb(uint32_t(a+127)<<23);return scaled*fb(uint32_t(b+127)<<23);
}
__global__ void kernel(const float*x,size_t n,size_t cols,float*out,float*records){
 const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
 const size_t row=i/cols,c=i%cols;const float gate=x[row*2*cols+c],up=x[row*2*cols+cols+c];
 const float ex=exp_two_sided(-gate),den=1.f+ex,si=gate/den,product=si*up;
 out[i]=product;if(records){records[i*4]=ex;records[i*4+1]=den;records[i*4+2]=si;records[i*4+3]=product;}
}
inline int launch(void*ctx,const float*x,int64_t rows,int64_t cols,float*out,float*records){
 if(!valid(ctx,x,rows,cols,out,records))return ns_fail("nrqv_silu_source: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));const size_t n=size_t(rows)*size_t(cols);
 kernel<<<dim3(unsigned((n+255)/256)),256,0,g->q>>>(x,n,size_t(cols),out,records);NS_LAUNCH_CHECK();return 0;
}
}
#endif
