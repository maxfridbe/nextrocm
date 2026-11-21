// Diagnostic DSO ONLY. Include this TU instead of attention_batched_runtime.hip.
// Frozen accepted kernel bodies; event stamps only in the separate export.
#include "../attention_batched.hpp"
#include "../../../ns/ns.h"
#include <array>
#include <cmath>
extern "C" int nrqi_image_attention_batched(void*g,const void*q,const void*k,const void*v,
 int64_t sq,int64_t sk,int64_t h,int64_t d,int64_t qs,int64_t ks,void*out,void*scratch,int64_t bytes){
 return nrqi_attention_batched::launch(g,q,k,v,sq,sk,h,d,qs,ks,out,scratch,bytes);
}
extern "C" int nrqi_image_attention_batched_profile(void*ctx,const void*q,const void*k,const void*v,
 int64_t sq,int64_t sk,int64_t h,int64_t d,int64_t qs,int64_t ks,void*out,void*scratch,int64_t bytes,
 double* host_ms,int64_t* host_counts){
 using namespace nrqi_attention_batched;
 if(!ctx||!host_ms||!host_counts||!arguments(q,k,v,sq,sk,h,d,qs,ks,out,scratch,bytes))return ns_fail("attention profile: invalid arguments");
 auto*g=static_cast<ns_gpu*>(ctx);if(!g->stamps)return ns_fail("attention profile requires NS_PROFILE=gpu");
 NS_CU(hipSetDevice(g->dev));
 struct Span{I start=-1,end=-1;int group=0;};std::array<Span,48> spans{};size_t used=0;
 const I capacity=batch(sq);auto*s=static_cast<float*>(scratch);float*den=s+h*capacity*sk;
 auto begin=[&](int group)->int{if(used>=spans.size())return ns_fail("attention profile span cap");spans[used].group=group;return ns_stamp(g,&spans[used].start);};
 auto end=[&]()->int{int rc=ns_stamp(g,&spans[used].end);if(rc==0)++used;return rc;};
 for(I row0=0;row0<sq;row0+=capacity){const I rows=sq-row0<capacity?sq-row0:capacity;
  if(begin(0))return -1;
  qk<<<dim3(unsigned((sk+63)/64),unsigned((rows+63)/64),unsigned(h)),256,0,g->q>>>(static_cast<const __half*>(q),static_cast<const __half*>(k),row0,rows,sk,d,qs,ks,capacity,s);NS_LAUNCH_CHECK();if(end())return -1;
  if(begin(1))return -1;
  softmax<<<dim3(unsigned(rows),unsigned(h)),256,0,g->q>>>(s,den,sk,capacity);NS_LAUNCH_CHECK();if(end())return -1;
  if(begin(2))return -1;
  pv<<<dim3(2,unsigned((rows+63)/64),unsigned(h)),256,0,g->q>>>(s,den,static_cast<const __half*>(v),row0,rows,sk,h,ks,capacity,static_cast<__half*>(out));NS_LAUNCH_CHECK();if(end())return -1;
 }
 if(used!=size_t(3*((sq+capacity-1)/capacity)))return ns_fail("attention profile coverage");
 double ms[3]={};I counts[3]={};
 // All launches/stamps are already queued. No mid-operation synchronization.
 for(size_t i=0;i<used;++i){double ns=-1;if(ns_elapsed(g,spans[i].start,spans[i].end,&ns))return -1;if(!std::isfinite(ns)||ns<=0)return ns_fail("attention profile invalid elapsed");ms[spans[i].group]+=ns/1e6;++counts[spans[i].group];}
 for(int group=0;group<3;++group){host_ms[group]=ms[group];host_counts[group]=counts[group];}
 return 0;
}
