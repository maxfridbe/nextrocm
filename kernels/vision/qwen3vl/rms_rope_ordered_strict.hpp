// Private strict-TU correction: disable contraction and use accurate OCML sqrtf.
// Do not include this header from a default-compiled production translation unit.
// See reference/qwenimage21-turbo/te-headnorm-order-v1 for causal/source proof.
#pragma once
#include <cstddef>
#include <cstdint>
namespace qv_ordered_strict {
inline bool valid(void*ctx,float*x,int64_t rows,int64_t heads,int64_t stride,const float*w,const float*cs){
 if(!ctx||rows<1||rows>4096||(heads!=8&&heads!=32)||stride<heads*128||stride>6144)return false;
 const auto span=[](const void*p,size_t count,uintptr_t&end){
  const uintptr_t begin=reinterpret_cast<uintptr_t>(p);
  if(!p||begin%4||count>SIZE_MAX/4||count*4>UINTPTR_MAX-begin)return false;
  end=begin+count*4;return true;
 };
 uintptr_t xe,we,ce;
 if(!span(x,size_t((rows-1)*stride+heads*128),xe)||!span(w,128,we)||!span(cs,size_t(rows)*128,ce))return false;
 const uintptr_t xb=reinterpret_cast<uintptr_t>(x),wb=reinterpret_cast<uintptr_t>(w),cb=reinterpret_cast<uintptr_t>(cs);
 return (xe<=wb||we<=xb)&&(xe<=cb||ce<=xb);
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "Ordered RMS/RoPE requires separate F32 rounding"
#endif
// Required in addition to per-TU -ffp-contract=off; inspect generated IR.
#pragma clang fp contract(off) reassociate(off)
namespace qv_ordered_strict {
__global__ void kernel(float*x,int64_t stride,const float*w,const float*cs){
 __shared__ float values[128],inv;
 const unsigned t=threadIdx.x;const size_t row=blockIdx.x,base=row*stride+size_t(blockIdx.y)*128;
 values[t]=x[base+t];__syncthreads();
 if(t==0){
  float lane_sum[8]={};
  for(unsigned p=0;p<4;++p)for(unsigned lane=0;lane<8;++lane){
   float partial=0;
   for(unsigned g=0;g<4;++g){float v=values[g*32+p*8+lane];partial=__fadd_rn(partial,__fmul_rn(v,v));}
   lane_sum[lane]=__fadd_rn(lane_sum[lane],partial);
  }
  float total=0;for(unsigned lane=0;lane<8;++lane)total=__fadd_rn(total,lane_sum[lane]);
  const float mean=__fdiv_rn(total,128.f);
  inv=__fdiv_rn(1.f,sqrtf(__fadd_rn(mean,1e-6f)));
 }
 __syncthreads();
 if(t<64){
  const float a=__fmul_rn(__fmul_rn(values[t],inv),w[t]);
  const float b=__fmul_rn(__fmul_rn(values[t+64],inv),w[t+64]);
  const float co=cs[row*128+2*t],si=cs[row*128+2*t+1];
  x[base+t]=__fsub_rn(__fmul_rn(a,co),__fmul_rn(b,si));
  x[base+t+64]=__fadd_rn(__fmul_rn(b,co),__fmul_rn(a,si));
 }
}
inline int launch(void*ctx,float*x,int64_t rows,int64_t heads,int64_t stride,const float*w,const float*cs){
 if(!valid(ctx,x,rows,heads,stride,w,cs))return ns_fail("nrqv_rms_rope_ordered_strict_f32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 kernel<<<dim3(unsigned(rows),unsigned(heads)),128,0,g->q>>>(x,stride,w,cs);NS_LAUNCH_CHECK();return 0;
}
}
#endif
