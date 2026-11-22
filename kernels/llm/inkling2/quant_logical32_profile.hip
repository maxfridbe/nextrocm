// Observer-only runtime: exact logical32 source prefix, omitting only its public forwarding alias.
// The profiler below owns that alias. Original arithmetic sources remain frozen.
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

#define ni2_quant_matvec_profile_baseline ni2_quant_matvec_logical32
#include "quant_profile.h"
#include <array>
#include <mutex>
#include <cmath>
namespace {
constexpr int max_calls=2047;
struct Profile {
    ns_gpu* owner=nullptr;
    int capacity=0, count=0;
    bool failed=false;
    std::array<hipEvent_t,max_calls> start{},stop{};
    std::array<ni2_quant_profile_record,max_calls> records{};
};
Profile profile;
std::mutex profile_mutex;
// Always attempt every destroy; report the first failure.
hipError_t clear_events() {
    hipError_t first=hipSuccess;
    for(int i=0;i<profile.capacity;++i) {
        for(auto* event : {&profile.start[i],&profile.stop[i]}) {
            if(*event) { auto e=hipEventDestroy(*event);if(first==hipSuccess)first=e;*event=nullptr; }
        }
    }
    profile.owner=nullptr;profile.capacity=profile.count=0;profile.failed=false;
    return first;
}
}
extern "C" int ni2_quant_profile_begin(ns_gpu* g,int capacity) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!g||capacity<1||capacity>max_calls||profile.owner)
        return ns_fail("quant profile begin: invalid capacity/context or active session");
    NS_CU(hipSetDevice(g->dev));
    profile.owner=g;profile.capacity=capacity;profile.count=0;profile.failed=false;
    for(int i=0;i<capacity;++i) {
        auto e=hipEventCreate(&profile.start[i]);
        if(e==hipSuccess)e=hipEventCreate(&profile.stop[i]);
        if(e!=hipSuccess){clear_events();return ns_hip(e,"quant profile event create");}
    }
    return 0;
    NS_CATCH
}
extern "C" int ni2_quant_matvec(ns_gpu* g,const void* w,int type,int rows,int cols,
    int64_t ws,const float* x,int64_t xs,const float* scale,float* out,int64_t os) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!profile.owner)
        return ni2_quant_matvec_profile_baseline(g,w,type,rows,cols,ws,x,xs,scale,out,os);
    if(g!=profile.owner||profile.failed||profile.count>=profile.capacity) {
        profile.failed=true;return ns_fail("quant profile: wrong context, failed session or capacity exhausted");
    }
    auto e=hipSetDevice(g->dev);
    if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"quant profile device");}
    const int i=profile.count;
    e=hipEventRecord(profile.start[i],g->q);
    if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"quant profile start");}
    int rc=ni2_quant_matvec_profile_baseline(g,w,type,rows,cols,ws,x,xs,scale,out,os);
    if(rc){profile.failed=true;return rc;}
    e=hipEventRecord(profile.stop[i],g->q);
    if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"quant profile stop");}
    profile.records[i]={type,rows,cols,scale?1:0,0.0};++profile.count;
    return 0;
    NS_CATCH
}
extern "C" int ni2_quant_profile_finish(ns_gpu* g,ni2_quant_profile_record* records,
    int capacity,int* count) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!g||g!=profile.owner||!records||!count||capacity<profile.capacity||profile.failed)
        return ns_fail("quant profile finish: invalid session/output or failed capture");
    NS_CU(hipSetDevice(g->dev));
    auto e=hipStreamSynchronize(g->q);
    if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"quant profile synchronize");}
    for(int i=0;i<profile.count;++i) {
        float ms=0;e=hipEventElapsedTime(&ms,profile.start[i],profile.stop[i]);
        if(e!=hipSuccess||!std::isfinite(ms)||ms<0) {
            profile.failed=true;
            return e==hipSuccess?ns_fail("quant profile invalid elapsed time"):ns_hip(e,"quant profile elapsed");
        }
        profile.records[i].gpu_ms=ms;
    }
    const int n=profile.count;
    // Clear resources before publishing successful output.
    e=clear_events();if(e!=hipSuccess)return ns_hip(e,"quant profile event destroy");
    for(int i=0;i<n;++i)records[i]=profile.records[i];
    *count=n;return 0;
    NS_CATCH
}
extern "C" int ni2_quant_profile_cancel(ns_gpu* g) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!g||g!=profile.owner)return ns_fail("quant profile cancel: wrong/inactive context");
    NS_CU(hipSetDevice(g->dev));
    auto sync=hipStreamSynchronize(g->q);auto cleanup=clear_events();
    if(sync!=hipSuccess)return ns_hip(sync,"quant profile cancel synchronize");
    return cleanup==hipSuccess?0:ns_hip(cleanup,"quant profile cancel destroy");
    NS_CATCH
}

#undef ni2_quant_matvec_profile_baseline
