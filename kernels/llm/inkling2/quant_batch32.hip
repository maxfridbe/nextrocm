// Private additive ABI. Compile INSTEAD of quant_typed32_dispatch.hip.
// Frozen typed/default/scalar/logical exports are retained unchanged.
#include "quant_typed32_dispatch.hip"
#include "quant_batch32.h"
namespace {
bool batch_span(const float* p,int n,int64_t stride,int64_t bs,int batch,
                ni2_quant::Span& span) {
    if(batch<1||batch>4||n<=0||stride<1||stride>1000000000LL||
       bs<1||bs>1000000000LL)return false;
    const uint64_t width=uint64_t(n-1)*uint64_t(stride)+1;
    if(uint64_t(bs)<width)return false;
    // Bounds above keep arithmetic below2^63; span separately enforces4GB/wrap.
    return ni2_quant::span(p,(uint64_t(batch-1)*uint64_t(bs)+width)*4,4,span);
}
template<int Type,int B>
__global__ void batch32_matvec(const uint8_t* w,int rows,int cols,int64_t ws,
    const float* x,int64_t xs,int64_t xbs,const float* scale,float* out,
    int64_t os,int64_t obs,int batch) {
    const int lane=threadIdx.x&31;
    const int row=int(blockIdx.x)*4+int(threadIdx.x)/32;
    const bool valid=row<rows;
    float sum[B]={};
    if(valid) {
        for(int k=lane;k<cols;k+=32) {
            const float weight=ni2_quant::value(w+int64_t(row)*ws,Type,k);
            #pragma unroll
            for(int b=0;b<B;++b) if(b<batch)
                sum[b]=fmaf(weight,x[int64_t(b)*xbs+int64_t(k)*xs],sum[b]);
        }
    }
    // All128 threads participate, including invalid rows and unused B3 slot.
    #pragma unroll
    for(int b=0;b<B;++b) {
        for(int distance=16;distance;distance/=2) {
            const float other=__shfl_down(sum[b],distance,32);
            sum[b]=__fadd_rn(sum[b],other);
        }
    }
    if(valid&&lane==0) {
        #pragma unroll
        for(int b=0;b<B;++b) if(b<batch) {
            float value=sum[b];
            if(scale)value=__fmul_rn(value,*scale);
            out[int64_t(b)*obs+int64_t(row)*os]=value;
        }
    }
}
template<int Type>
void launch_batch(ns_gpu* g,const uint8_t* w,int rows,int cols,int64_t ws,
    const float* x,int64_t xs,int64_t xbs,const float* scale,float* out,
    int64_t os,int64_t obs,int batch) {
    if(batch==2) batch32_matvec<Type,2><<<(rows+3)/4,128,0,g->q>>>(w,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);
    else batch32_matvec<Type,4><<<(rows+3)/4,128,0,g->q>>>(w,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);
}
}
extern "C" int ni2_quant_matvec_batch32(ns_gpu* g,const void* w,int type,
    int rows,int cols,int64_t ws,const float* x,int64_t xs,int64_t xbs,
    const float* scale,float* out,int64_t os,int64_t obs,int batch) {
    ni2_quant::Span a,b,c,d;
    if(!g||!ni2_quant::matrix(w,type,rows,cols,ws,a)||
       !batch_span(x,cols,xs,xbs,batch,b)||!batch_span(out,rows,os,obs,batch,c)||
       !ni2_quant::disjoint(a,c)||!ni2_quant::disjoint(b,c)||
       (scale&&(!ni2_quant::span(scale,4,4,d)||!ni2_quant::disjoint(d,c))))
        return ns_fail("inkling2 batch32 matvec: invalid shape/span/alias");
    NS_CU(hipSetDevice(g->dev));
    int wave=0;NS_CU(hipDeviceGetAttribute(&wave,hipDeviceAttributeWarpSize,g->dev));
    if(wave!=32&&wave!=64)return ns_fail("inkling2 batch32 matvec: hardware wave32 or wave64 required");
    if(batch==1||wave==32) {
        for(int i=0;i<batch;++i) {
            const int rc=ni2_quant_matvec_typed32(g,w,type,rows,cols,ws,
                x+int64_t(i)*xbs,xs,scale,out+int64_t(i)*obs,os);
            if(rc)return rc;
        }
        return 0;
    }
    const auto* bytes=static_cast<const uint8_t*>(w);
    switch(type) {
    case 0:launch_batch<0>(g,bytes,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);break;
    case 8:launch_batch<8>(g,bytes,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);break;
    case 13:launch_batch<13>(g,bytes,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);break;
    case 30:launch_batch<30>(g,bytes,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);break;
    case 40:launch_batch<40>(g,bytes,rows,cols,ws,x,xs,xbs,scale,out,os,obs,batch);break;
    default:return ns_fail("inkling2 batch32 matvec: invalid type");
    }
    NS_LAUNCH_CHECK();return 0;
}
