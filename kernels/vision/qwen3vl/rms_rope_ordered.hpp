// Private opt-in F32 RMS/RoPE. Source-derived contiguous128 reduction.
// See reference/qwenimage21-turbo/te-headnorm-order-v1 for causal/source proof.
#pragma once
#include "int8_convrot.hpp"
namespace qv_ordered {
inline bool valid(void*ctx,float*x,int64_t rows,int64_t heads,int64_t stride,const float*w,const float*cs){
 if(!ctx||rows<1||rows>4096||(heads!=8&&heads!=32)||stride<heads*128||stride>6144)return false;
 qv_int8::Span a,b,c;
 return qv_int8::span(x,size_t((rows-1)*stride+heads*128),4,a)&&qv_int8::span(w,128,4,b)&&qv_int8::span(cs,size_t(rows)*128,4,c)&&qv_int8::separate(a,b)&&qv_int8::separate(a,c);
}
}
#ifdef __HIPCC__
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "Ordered RMS/RoPE requires separate F32 rounding"
#endif
namespace qv_ordered {
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
  inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(mean,1e-6f)));
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
 if(!valid(ctx,x,rows,heads,stride,w,cs))return ns_fail("nrqv_rms_rope_ordered_f32: invalid geometry/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 kernel<<<dim3(unsigned(rows),unsigned(heads)),128,0,g->q>>>(x,stride,w,cs);NS_LAUNCH_CHECK();return 0;
}
}
#endif
