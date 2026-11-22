// Observer aliases only; frozen implementation included unchanged.
#define ni2_rms ni2_rms_all_baseline
#define ni2_clip ni2_clip_all_baseline
#define ni2_situ ni2_situ_all_baseline
#define ni2_gate ni2_gate_all_baseline
#define ni2_softcap ni2_softcap_all_baseline
#include "primitives.hip"
#undef ni2_rms
#undef ni2_clip
#undef ni2_situ
#undef ni2_gate
#undef ni2_softcap
#include "all_profile_internal.hpp"
extern "C" int ni2_rms(ns_gpu* g,const float* x,const float* gamma,int rows,int width,float eps,float* out) {
 return ni2_all_profile_call(g,3,0,rows,width,0,[&]()->int{return ni2_rms_all_baseline(g,x,gamma,rows,width,eps,out);});
}
extern "C" int ni2_clip(ns_gpu* g,const float* x,const float* scale,int rows,int width,float threshold,float* out) {
 return ni2_all_profile_call(g,4,0,rows,width,0,[&]()->int{return ni2_clip_all_baseline(g,x,scale,rows,width,threshold,out);});
}
extern "C" int ni2_situ(ns_gpu* g,const float* a,const float* b,int n,float beta,float linear_beta,float* out) {
 return ni2_all_profile_call(g,5,0,1,n,0,[&]()->int{return ni2_situ_all_baseline(g,a,b,n,beta,linear_beta,out);});
}
extern "C" int ni2_gate(ns_gpu* g,const float* x,const float* gate_values,int n,float* out) {
 return ni2_all_profile_call(g,6,0,1,n,0,[&]()->int{return ni2_gate_all_baseline(g,x,gate_values,n,out);});
}
extern "C" int ni2_softcap(ns_gpu* g,const float* x,int n,float scale,float* out) {
 return ni2_all_profile_call(g,7,0,1,n,0,[&]()->int{return ni2_softcap_all_baseline(g,x,n,scale,out);});
}
