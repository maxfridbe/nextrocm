// Private current-wave profiler; original scalar and wave sources stay frozen.
#define ni2_quant_matvec ni2_quant_matvec_scalar_baseline
#include "quant.hip"
#undef ni2_quant_matvec
#define ni2_quant_matvec_wave32 ni2_quant_matvec_profile_baseline
#include "quant_wave32.hip"
#undef ni2_quant_matvec_wave32
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
