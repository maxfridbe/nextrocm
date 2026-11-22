// Observer aliases only; frozen implementation included unchanged.
#define ni2_depthmix ni2_depthmix_all_baseline
#define ni2_router ni2_router_all_baseline
#define ni2_loop_control ni2_loop_control_all_baseline
#define ni2_scale ni2_scale_all_baseline
#define ni2_add ni2_add_all_baseline
#define ni2_axpy ni2_axpy_all_baseline
#include "mix.hip"
#undef ni2_depthmix
#undef ni2_router
#undef ni2_loop_control
#undef ni2_scale
#undef ni2_add
#undef ni2_axpy
#include "all_profile_internal.hpp"
extern "C" int ni2_depthmix(ns_gpu* g,const float* states,const float* queries,int count,int nq,int width,float* out) {
 return ni2_all_profile_call(g,12,0,count,width,nq,[&]()->int{return ni2_depthmix_all_baseline(g,states,queries,count,nq,width,out);});
}
extern "C" int ni2_router(ns_gpu* g,const float* logits,const float* bias,int32_t* ids,float* weights) {
 return ni2_all_profile_call(g,13,0,1,128,0,[&]()->int{return ni2_router_all_baseline(g,logits,bias,ids,weights);});
}
extern "C" int ni2_loop_control(ns_gpu* g,const float* cur,const float* msg,const float* gain,const float* forget,int n,float* out) {
 return ni2_all_profile_call(g,14,0,1,n,0,[&]()->int{return ni2_loop_control_all_baseline(g,cur,msg,gain,forget,n,out);});
}
extern "C" int ni2_scale(ns_gpu* g,const float* x,int n,float factor,float* out) {
 return ni2_all_profile_call(g,15,0,1,n,0,[&]()->int{return ni2_scale_all_baseline(g,x,n,factor,out);});
}
extern "C" int ni2_add(ns_gpu* g,const float* a,const float* b,int n,float* out) {
 return ni2_all_profile_call(g,16,0,1,n,0,[&]()->int{return ni2_add_all_baseline(g,a,b,n,out);});
}
extern "C" int ni2_axpy(ns_gpu* g,const float* x,const float* weight,int n,float* out) {
 return ni2_all_profile_call(g,17,0,1,n,0,[&]()->int{return ni2_axpy_all_baseline(g,x,weight,n,out);});
}
