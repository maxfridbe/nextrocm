#include "quant.h"
#include "quant.hpp"
#include "ns_internal.hpp"
// Compile this private TU with -ffp-contract=off, no fast/reassociation flags.
namespace {
__global__ void decode(const uint8_t* w,int type,int rows,int cols,int64_t ws,float* out,int64_t os) {
    int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
    if(i<int64_t(rows)*cols) out[(i/cols)*os+i%cols]=ni2_quant::value(w+(i/cols)*ws,type,int(i%cols));
}
__global__ void matvec(const uint8_t* w,int type,int rows,int cols,int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    const int r=blockIdx.x*64+threadIdx.x;
    if(r>=rows)return;
    float sum=0;
    for(int k=0;k<cols;++k)sum=fmaf(ni2_quant::value(w+int64_t(r)*ws,type,k),x[int64_t(k)*xs],sum);
    if(scale)sum=__fmul_rn(sum,*scale);
    out[int64_t(r)*os]=sum;
}
}
extern "C" int ni2_quant_decode(ns_gpu* g,const void* w,int type,int rows,int cols,int64_t ws,float* out,int64_t os) {
    ni2_quant::Span a,b;
    if(!g||!ni2_quant::matrix(w,type,rows,cols,ws,a)||!ni2_quant::floats(out,rows,os,cols,b)||!ni2_quant::disjoint(a,b))return ns_fail("inkling2 quant decode: invalid shape/span/alias");
    NS_CU(hipSetDevice(g->dev));
    decode<<<unsigned((int64_t(rows)*cols+255)/256),256,0,g->q>>>(static_cast<const uint8_t*>(w),type,rows,cols,ws,out,os);
    NS_LAUNCH_CHECK();return 0;
}
extern "C" int ni2_quant_matvec(ns_gpu* g,const void* w,int type,int rows,int cols,int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    ni2_quant::Span a,b,c,d;
    if(!g||!ni2_quant::matrix(w,type,rows,cols,ws,a)||!ni2_quant::floats(x,cols,xs,1,b)||!ni2_quant::floats(out,rows,os,1,c)||!ni2_quant::disjoint(a,c)||!ni2_quant::disjoint(b,c)||(scale&&(!ni2_quant::span(scale,4,4,d)||!ni2_quant::disjoint(d,c))))return ns_fail("inkling2 quant matvec: invalid shape/span/alias");
    NS_CU(hipSetDevice(g->dev));
    matvec<<<(rows+63)/64,64,0,g->q>>>(static_cast<const uint8_t*>(w),type,rows,cols,ws,x,xs,scale,out,os);
    NS_LAUNCH_CHECK();return 0;
}
