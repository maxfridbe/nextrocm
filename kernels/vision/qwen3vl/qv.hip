#include "qv.h"
#include "int8_convrot.hpp"
#include "attention_prob.hpp"
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#ifdef __FAST_MATH__
#error "Qwen3-VL requires explicit F32 arithmetic, no fast math"
#endif
extern "C" int nrqv_attention_prob_f32(void*ctx,const float*qkv,int64_t rows,float*out){
 return qv_prob::launch(ctx,qkv,rows,out);
}
extern "C" int nrqv_int8_convrot_linear(void* ctx,const nrqv_int8_args* p) {
 if(!ctx||!p)return ns_fail("nrqv_int8: null context/arguments");
 qv_int8::Args a{p->x,p->weight,p->weight_scale,p->bias,p->out,p->rotated,p->quantized,
                p->row_scale,p->accum,p->status,p->m,p->n,p->k,p->scales,p->dtype};
 if(!qv_int8::valid(a))return ns_fail("nrqv_int8: invalid shape/spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 NS_CU(qv_int8::launch(a,g->q));return 0;
}
namespace {
__global__ void rms_rope(float*x,int64_t stride,const float*w,const float*cs) {
 __shared__ float values[128],inv;
 const unsigned t=threadIdx.x;
 const size_t row=blockIdx.x,base=row*stride+size_t(blockIdx.y)*128;
 values[t]=x[base+t];__syncthreads();
 if(t==0) {
  float sum=0;
  for(unsigned i=0;i<128;++i)sum=__fadd_rn(sum,__fmul_rn(values[i],values[i]));
  inv=rsqrtf(__fadd_rn(__fdiv_rn(sum,128.f),1e-6f));
 }
 __syncthreads();
 if(t<64) {
  const float a=__fmul_rn(__fmul_rn(values[t],inv),w[t]);
  const float b=__fmul_rn(__fmul_rn(values[t+64],inv),w[t+64]);
  const float co=cs[row*128+2*t],si=cs[row*128+2*t+1];
  x[base+t]=__fsub_rn(__fmul_rn(a,co),__fmul_rn(b,si));
  x[base+t+64]=__fadd_rn(__fmul_rn(b,co),__fmul_rn(a,si));
 }
}
}
extern "C" int nrqv_rms_rope_f32(void*ctx,float*x,int64_t rows,int64_t heads,int64_t stride,
                                 const float*w,const float*cs) {
 if(!ctx||rows<1||rows>4096||(heads!=8&&heads!=32)||stride<heads*128||stride>6144)
  return ns_fail("nrqv_rms_rope: invalid geometry");
 qv_int8::Span xs,ws,cs_span;
 if(!qv_int8::span(x,size_t((rows-1)*stride+heads*128),4,xs)||
    !qv_int8::span(w,128,4,ws)||!qv_int8::span(cs,size_t(rows)*128,4,cs_span)||
    !qv_int8::separate(xs,ws)||!qv_int8::separate(xs,cs_span))
  return ns_fail("nrqv_rms_rope: invalid spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 rms_rope<<<dim3(unsigned(rows),unsigned(heads)),128,0,g->q>>>(x,stride,w,cs);
 NS_LAUNCH_CHECK();return 0;
}

#ifndef NRQV_ORDERED_STRICT_TU
#include "rms_rope_ordered.hpp"
extern "C" int nrqv_rms_rope_ordered_f32(void*ctx,float*x,int64_t rows,int64_t heads,int64_t stride,const float*w,const float*cs){
 return qv_ordered::launch(ctx,x,rows,heads,stride,w,cs);
}

#endif // NRQV_ORDERED_STRICT_TU
