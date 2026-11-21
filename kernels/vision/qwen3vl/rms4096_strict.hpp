// Private source-derived F32 RMS, D4096, eps1e-6. Strict TU only.
#pragma once
#include <cstddef>
#include <cstdint>
namespace qv_rms4096 {
inline bool valid(void*ctx,const float*x,int64_t rows,const float*w,float*out){
 if(!ctx||rows<1||rows>4096)return false;
 const auto span=[](const void*p,size_t n,uintptr_t&e){const auto b=reinterpret_cast<uintptr_t>(p);if(!p||b%4||n>SIZE_MAX/4||n*4>UINTPTR_MAX-b)return false;e=b+n*4;return true;};
 uintptr_t xe,we,oe;if(!span(x,size_t(rows)*4096,xe)||!span(w,4096,we)||!span(out,size_t(rows)*4096,oe))return false;
 const auto xb=reinterpret_cast<uintptr_t>(x),wb=reinterpret_cast<uintptr_t>(w),ob=reinterpret_cast<uintptr_t>(out);
 return (oe<=xb||xe<=ob)&&(oe<=wb||we<=ob);
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "Source RMS requires strict F32 boundaries"
#endif
#pragma clang fp contract(off) reassociate(off)
namespace qv_rms4096 {
__global__ void kernel(const float*x,const float*w,float*out){
 __shared__ float partial[32],inv;
 const unsigned t=threadIdx.x;const size_t base=size_t(blockIdx.x)*4096;
 // Four partial vectors of eight lanes, eight chunks of sixteen groups.
 if(t<32){float acc=0;for(unsigned chunk=0;chunk<8;++chunk){float z=0;for(unsigned g=chunk*16;g<(chunk+1)*16;++g){float v=x[base+g*32+t];z=__fadd_rn(z,__fmul_rn(v,v));}acc=__fadd_rn(acc,z);}partial[t]=acc;}
 __syncthreads();
 if(t==0){float total=0;for(unsigned lane=0;lane<8;++lane){float z=partial[lane];for(unsigned p=1;p<4;++p)z=__fadd_rn(z,partial[p*8+lane]);total=__fadd_rn(total,z);}inv=__fdiv_rn(1.f,sqrtf(__fadd_rn(__fdiv_rn(total,4096.f),1e-6f)));}
 __syncthreads();
 for(unsigned j=t;j<4096;j+=256)out[base+j]=__fmul_rn(__fmul_rn(x[base+j],inv),w[j]);
}
inline int launch(void*ctx,const float*x,int64_t rows,const float*w,float*out){
 if(!valid(ctx,x,rows,w,out))return ns_fail("nrqv_rms4096_strict_f32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));kernel<<<unsigned(rows),256,0,g->q>>>(x,w,out);NS_LAUNCH_CHECK();return 0;
}
}
#endif
