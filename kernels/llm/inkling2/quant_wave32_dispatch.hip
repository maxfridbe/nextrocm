// Test-only combined-runtime replacement for quant.hip. Do not link quant.hip
// or quant_wave32.hip separately with this TU. Both included sources stay frozen.
// Full-model validation required before any production selection.
#define ni2_quant_matvec ni2_quant_matvec_scalar_baseline
#include "quant.hip"
#undef ni2_quant_matvec
#include "quant_wave32.hip"
extern "C" int ni2_quant_matvec(ns_gpu* g,const void* w,int type,int rows,int cols,
    int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    return ni2_quant_matvec_wave32(g,w,type,rows,cols,ws,x,xs,scale,out,os);
}
