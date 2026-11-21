// Test-only wrapper: compile the ORIGINAL speech translation unit unchanged.
// Its public wrapper caps SK at 4096; this private validation export invokes
// the very same att_kernel with checked SK <= 4608. No copied arithmetic.
// Build this TU INSTEAD OF speech.hip, never together with it.
#include "../../../ops/speech.hip"
#define NRQI_ATTENTION_1024_GUARDS_ONLY
#include "../attention_1024.hpp"
#undef NRQI_ATTENTION_1024_GUARDS_ONLY
extern "C" int nrqi_image_attention_1024_baseline(void* ctx,const void* q,const void* k,const void* v,
    int64_t sq,int64_t sk,int64_t h,int64_t d,int64_t qs,int64_t ks,void* out) {
    if(!ctx||!nrqi_attention_1024::arguments(q,k,v,sq,sk,h,d,qs,ks,out))
        return ns_fail("1024 test baseline: invalid shape/span/alias");
    auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
    hipLaunchKernelGGL(att_kernel,dim3(unsigned(sq),unsigned(h)),dim3(256),size_t(sk)*sizeof(float),g->q,
        q,k,v,1,0,sk,h,h,d,qs,ks,out,1,false);
    NS_LAUNCH_CHECK();return 0;
}
