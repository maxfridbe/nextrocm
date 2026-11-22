// Private observer manager; lifecycle follows frozen quant profiler.
#include "all_profile_internal.hpp"
#include <array>
#include <mutex>
#include <cmath>
namespace {
constexpr int max_calls=4096;
struct Profile {
    ns_gpu* owner=nullptr;
    int capacity=0, count=0;
    bool failed=false;
    std::array<hipEvent_t,max_calls> start{},stop{};
    std::array<ni2_all_profile_record,max_calls> records{};
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
extern "C" int ni2_all_profile_begin(ns_gpu* g,int capacity) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!g||capacity<1||capacity>max_calls||profile.owner)
        return ns_fail("all profile begin: invalid capacity/context or active session");
    NS_CU(hipSetDevice(g->dev));
    profile.owner=g;profile.capacity=capacity;profile.count=0;profile.failed=false;
    for(int i=0;i<capacity;++i) {
        auto e=hipEventCreate(&profile.start[i]);
        if(e==hipSuccess)e=hipEventCreate(&profile.stop[i]);
        if(e!=hipSuccess){clear_events();return ns_hip(e,"all profile event create");}
    }
    return 0;
    NS_CATCH
}
int ni2_all_profile_invoke(ns_gpu* g,ni2_all_profile_record record,int(*invoke)(void*),void* opaque) {
 NS_TRY
 std::lock_guard<std::mutex> lock(profile_mutex);
 if(!profile.owner)return invoke(opaque);
 if(g!=profile.owner||profile.failed||profile.count>=profile.capacity) {
  profile.failed=true;return ns_fail("all profile: wrong context, failed session or capacity exhausted");
 }
 auto e=hipSetDevice(g->dev);
 if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"all profile device");}
 const int i=profile.count;
 e=hipEventRecord(profile.start[i],g->q);
 if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"all profile start");}
 int rc=invoke(opaque);
 if(rc){profile.failed=true;return rc;}
 e=hipEventRecord(profile.stop[i],g->q);
 if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"all profile stop");}
 profile.records[i]=record;++profile.count;return 0;
 NS_CATCH
}
extern "C" int ni2_all_profile_finish(ns_gpu* g,ni2_all_profile_record* records,
    int capacity,int* count) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!g||g!=profile.owner||!records||!count||capacity<profile.capacity||profile.failed)
        return ns_fail("all profile finish: invalid session/output or failed capture");
    NS_CU(hipSetDevice(g->dev));
    auto e=hipStreamSynchronize(g->q);
    if(e!=hipSuccess){profile.failed=true;return ns_hip(e,"all profile synchronize");}
    for(int i=0;i<profile.count;++i) {
        float ms=0;e=hipEventElapsedTime(&ms,profile.start[i],profile.stop[i]);
        if(e!=hipSuccess||!std::isfinite(ms)||ms<0) {
            profile.failed=true;
            return e==hipSuccess?ns_fail("all profile invalid elapsed time"):ns_hip(e,"all profile elapsed");
        }
        profile.records[i].gpu_ms=ms;
    }
    const int n=profile.count;
    // Clear resources before publishing successful output.
    e=clear_events();if(e!=hipSuccess)return ns_hip(e,"all profile event destroy");
    for(int i=0;i<n;++i)records[i]=profile.records[i];
    *count=n;return 0;
    NS_CATCH
}
extern "C" int ni2_all_profile_cancel(ns_gpu* g) {
    NS_TRY
    std::lock_guard<std::mutex> lock(profile_mutex);
    if(!g||g!=profile.owner)return ns_fail("all profile cancel: wrong/inactive context");
    NS_CU(hipSetDevice(g->dev));
    auto sync=hipStreamSynchronize(g->q);auto cleanup=clear_events();
    if(sync!=hipSuccess)return ns_hip(sync,"all profile cancel synchronize");
    return cleanup==hipSuccess?0:ns_hip(cleanup,"all profile cancel destroy");
    NS_CATCH
}
