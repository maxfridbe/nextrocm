// Private original Euler primitive; not registered in normal builds.
// x[i] = round_f32(x[i] + round_f32(step * velocity[i])). Never contract to FMA.
#pragma once
#include <cmath>
#include <cstdint>
namespace nrqi_trajectory {
using I=int64_t;
constexpr I max_values=16777216; // private bounded ABI:64MiB per F32 input
struct Span{uintptr_t first,last;};
inline bool span(const void* p,I n,Span& s) {
    const uintptr_t a=reinterpret_cast<uintptr_t>(p);
    if(!p||n<1||n>max_values||a%4||uint64_t(n)>UINTPTR_MAX/4)return false;
    const uintptr_t bytes=uintptr_t(n)*4;if(bytes>UINTPTR_MAX-a)return false;
    s={a,a+bytes};return true;
}
inline bool valid(const void* g,const void* x,const void* v,I n,float step) {
    Span a,b;return g&&std::isfinite(step)&&span(x,n,a)&&span(v,n,b)&&(a.last<=b.first||b.last<=a.first);
}
}
#ifdef __HIPCC__
#ifdef __FAST_MATH__
#error "Trajectory requires separate IEEE F32 multiply and add"
#endif
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
namespace nrqi_trajectory {
__global__ void axpy(float* x,const float* v,I n,float step) {
    const I i=I(blockIdx.x)*256+threadIdx.x;
    if(i<n)x[i]=__fadd_rn(x[i],__fmul_rn(step,v[i]));
}
inline int launch(void* ctx,float* x,const float* v,I n,float step) {
    if(!valid(ctx,x,v,n,step))return ns_fail("Qwen trajectory: invalid count/step/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
    hipLaunchKernelGGL(axpy,dim3(unsigned((n+255)/256)),dim3(256),0,g->q,x,v,n,step);
    NS_LAUNCH_CHECK();return 0;
}
}
#endif
