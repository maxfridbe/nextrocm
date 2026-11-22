// Explicit portable runtime alternative. Compile instead of quant.hip,
// quant_wave32.hip, or quant_wave32_dispatch.hip, never alongside them.
// Existing decoders and both arithmetic implementations remain unchanged.
// Wave64 uses the ascending-FMA scalar baseline; its results are not assumed
// bitwise equal to the wave32 reduction. Full-model gates remain mandatory.
#define ni2_quant_matvec ni2_quant_matvec_scalar_baseline
#include "quant.hip"
#undef ni2_quant_matvec
#include "quant_wave32.hip"

extern "C" int ni2_quant_matvec(ns_gpu* g,const void* w,int type,int rows,int cols,
    int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    if(!g)return ns_fail("inkling2 portable matvec: null context");
    int wave=0;
    NS_CU(hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,g->dev));
    if(wave==32)
        return ni2_quant_matvec_wave32(g,w,type,rows,cols,ws,x,xs,scale,out,os);
    if(wave==64)
        return ni2_quant_matvec_scalar_baseline(g,w,type,rows,cols,ws,x,xs,scale,out,os);
    return ns_fail("inkling2 portable matvec: unsupported hardware wave size");
}
