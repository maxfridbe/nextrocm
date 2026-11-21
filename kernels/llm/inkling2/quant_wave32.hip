#include "quant_wave32.h"
#include "quant.hpp"
#include "ns_internal.hpp"
// Separate private TU, strict -ffp-contract=off; explicit fmaf only.
namespace {
__global__ void wave32_matvec(const uint8_t* w,int type,int rows,int cols,
    int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    const int lane=threadIdx.x&31;
    const int row=int(blockIdx.x)*4+int(threadIdx.x)/32;
    if(row>=rows)return; // whole wave returns; no block barriers
    float sum=0.0f;
    for(int k=lane;k<cols;k+=32)
        sum=fmaf(ni2_quant::value(w+int64_t(row)*ws,type,k),x[int64_t(k)*xs],sum);
    for(int distance=16;distance;distance/=2) {
        const float other=__shfl_down(sum,distance,32);
        sum=__fadd_rn(sum,other);
    }
    if(lane==0) {
        if(scale)sum=__fmul_rn(sum,*scale);
        out[int64_t(row)*os]=sum;
    }
}
}
extern "C" int ni2_quant_matvec_wave32(ns_gpu* g,const void* w,int type,
    int rows,int cols,int64_t ws,const float* x,int64_t xs,const float* scale,
    float* out,int64_t os) {
    ni2_quant::Span a,b,c,d;
    if(!g||!ni2_quant::matrix(w,type,rows,cols,ws,a)||
       !ni2_quant::floats(x,cols,xs,1,b)||!ni2_quant::floats(out,rows,os,1,c)||
       !ni2_quant::disjoint(a,c)||!ni2_quant::disjoint(b,c)||
       (scale&&(!ni2_quant::span(scale,4,4,d)||!ni2_quant::disjoint(d,c))))
        return ns_fail("inkling2 wave32 matvec: invalid shape/span/alias");
    NS_CU(hipSetDevice(g->dev));
    int wave=0;NS_CU(hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,g->dev));
    if(wave!=32)return ns_fail("inkling2 wave32 matvec: hardware wave32 required");
    wave32_matvec<<<(rows+3)/4,128,0,g->q>>>(static_cast<const uint8_t*>(w),type,
        rows,cols,ws,x,xs,scale,out,os);
    NS_LAUNCH_CHECK();return 0;
}
