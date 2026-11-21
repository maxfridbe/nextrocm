// Private casts. Separate from the frozen four DiT math operators.
#pragma once
#include "convert.h"
#include <cstdint>
namespace nrqi_cast {
inline bool span(const void*p,uint64_t n,unsigned size,uintptr_t&end){
 const uintptr_t start=reinterpret_cast<uintptr_t>(p);
 if(!p||start%size||n>UINTPTR_MAX/size||n*size>UINTPTR_MAX-start)return false;
 end=start+n*size;return true;
}
inline bool valid(void*g,const void*x,int64_t count,void*out,unsigned from,unsigned to){
 if(!g||count<1||count>INT32_MAX)return false;
 uintptr_t a,b;
 return span(x,uint64_t(count),from,a)&&span(out,uint64_t(count),to,b)&&
        (a<=reinterpret_cast<uintptr_t>(out)||b<=reinterpret_cast<uintptr_t>(x));
}
}
#ifdef __HIPCC__
#ifdef __FAST_MATH__
#error "IEEE casts require nonfinite and signed-zero semantics"
#endif
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
namespace nrqi_cast {
__global__ void down(const float*x,int64_t n,__half*out){const int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;if(i<n)out[i]=__float2half_rn(x[i]);}
__global__ void up(const __half*x,int64_t n,float*out){const int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;if(i<n)out[i]=__half2float(x[i]);}
inline int down_launch(void*ctx,const float*x,int64_t n,void*out){
 if(!valid(ctx,x,n,out,4,2))return ns_fail("nrqi_f32_to_f16: invalid span/count");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 down<<<dim3(unsigned((n+255)/256)),256,0,g->q>>>(x,n,static_cast<__half*>(out));NS_LAUNCH_CHECK();return 0;
}
inline int up_launch(void*ctx,const void*x,int64_t n,float*out){
 if(!valid(ctx,x,n,out,2,4))return ns_fail("nrqi_f16_to_f32: invalid span/count");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 up<<<dim3(unsigned((n+255)/256)),256,0,g->q>>>(static_cast<const __half*>(x),n,out);NS_LAUNCH_CHECK();return 0;
}
}
#endif
