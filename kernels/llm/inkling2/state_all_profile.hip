// Original state kernels and guards remain unchanged; split attention timing only.
#define ni2_shortconv ni2_shortconv_all_baseline
#define ni2_attention ni2_attention_all_baseline
#include "state.hip"
#undef ni2_shortconv
#undef ni2_attention
#include "all_profile_internal.hpp"
extern "C" int ni2_shortconv(ns_gpu* g,const float* x,const float* w,const float* d,int heads,int c,float* history,float* out) {
 return ni2_all_profile_call(g,8,0,heads,c,d?1:0,[&]()->int{return ni2_shortconv_all_baseline(g,x,w,d,heads,c,history,out);});
}
extern "C" int ni2_attention(ns_gpu* g,const float* q,const float* k,const float* v,
 const float* r,const float* w,const float* sink,int64_t pos,int cap,int window,
 float* out,float* scratch,int64_t count) {
    if(!g||!ni2_state::attention_args(q,k,v,r,w,sink,pos,cap,window,out,scratch,count)) return ns_fail("inkling2: invalid attention arguments");
    NS_CU(hipSetDevice(g->dev));
    int n=ni2_state::key_count(pos,cap,window);
    int rc=ni2_all_profile_call(g,9,0,24,n,0,[&]()->int {
      logits<<<dim3((n+255)/256,24),256,0,g->q>>>(q,k,r,w,pos,cap,n,scratch); NS_LAUNCH_CHECK();return 0;
    });if(rc)return rc;
    rc=ni2_all_profile_call(g,10,0,24,n,0,[&]()->int {
      probabilities<<<24,256,0,g->q>>>(scratch,sink,n); NS_LAUNCH_CHECK();return 0;
    });if(rc)return rc;
    return ni2_all_profile_call(g,11,0,24,n,0,[&]()->int {
      weighted<<<24,128,0,g->q>>>(scratch,v,pos,cap,n,out); NS_LAUNCH_CHECK();return 0;
    });
}
