// Independent candidate. Compile INSTEAD of every other quant runtime TU.
// Physical wave32 keeps the frozen implementation. Physical wave64 runs two
// logical 32-lane rows per wave; all lanes stay active through every shuffle.
#define ni2_quant_matvec ni2_quant_matvec_scalar_baseline
#include "quant.hip"
#undef ni2_quant_matvec
#include "quant_wave32.hip"

namespace {
__global__ void logical32_matvec(const uint8_t* w,int type,int rows,int cols,
    int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    const int lane=threadIdx.x&31;
    const int row=int(blockIdx.x)*4+int(threadIdx.x)/32;
    const bool valid=row<rows;
    float sum=0.0f;
    if(valid) {
        for(int k=lane;k<cols;k+=32)
            sum=fmaf(ni2_quant::value(w+int64_t(row)*ws,type,k),x[int64_t(k)*xs],sum);
    }
    for(int distance=16;distance;distance/=2) {
        const float other=__shfl_down(sum,distance,32);
        sum=__fadd_rn(sum,other);
    }
    if(valid&&lane==0) {
        if(scale)sum=__fmul_rn(sum,*scale);
        out[int64_t(row)*os]=sum;
    }
}
}

extern "C" int ni2_quant_matvec_logical32(ns_gpu* g,const void* w,int type,
    int rows,int cols,int64_t ws,const float* x,int64_t xs,const float* scale,
    float* out,int64_t os) {
    ni2_quant::Span a,b,c,d;
    if(!g||!ni2_quant::matrix(w,type,rows,cols,ws,a)||
       !ni2_quant::floats(x,cols,xs,1,b)||!ni2_quant::floats(out,rows,os,1,c)||
       !ni2_quant::disjoint(a,c)||!ni2_quant::disjoint(b,c)||
       (scale&&(!ni2_quant::span(scale,4,4,d)||!ni2_quant::disjoint(d,c))))
        return ns_fail("inkling2 logical32 matvec: invalid shape/span/alias");
    NS_CU(hipSetDevice(g->dev));
    int wave=0;NS_CU(hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,g->dev));
    if(wave==32)
        return ni2_quant_matvec_wave32(g,w,type,rows,cols,ws,x,xs,scale,out,os);
    if(wave!=64)return ns_fail("inkling2 logical32 matvec: hardware wave32 or wave64 required");
    logical32_matvec<<<(rows+3)/4,128,0,g->q>>>(static_cast<const uint8_t*>(w),
        type,rows,cols,ws,x,xs,scale,out,os);
    NS_LAUNCH_CHECK();return 0;
}

extern "C" int ni2_quant_matvec(ns_gpu* g,const void* w,int type,int rows,int cols,
    int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    return ni2_quant_matvec_logical32(g,w,type,rows,cols,ws,x,xs,scale,out,os);
}
