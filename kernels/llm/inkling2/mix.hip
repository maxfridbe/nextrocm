// Source: merged Inkling graph 883286 mix/control and pinned HF router.
// Strict private TU: -ffp-contract=off, no unsafe math/reassociation.
#include "mix.h"
#include "ns_internal.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
namespace {
struct Span {uintptr_t a,b;};
bool span(const void* p,size_t bytes,Span& s) {
 s.a=reinterpret_cast<uintptr_t>(p);
 if(!p||s.a%4||!bytes||bytes>std::numeric_limits<uintptr_t>::max()-s.a)return false;
 s.b=s.a+bytes;return true;
}
bool disjoint(Span a,Span b) {return a.b<=b.a||b.b<=a.a;}
bool output(const float* x,float* y,int n,bool exact=true) {
 Span a,b;return n>0&&n<=16*1024*1024&&span(x,size_t(n)*4,a)&&span(y,size_t(n)*4,b)&&((exact&&a.a==b.a)||disjoint(a,b));
}
__global__ void depth(const float* states,const float* queries,int count,int width,float* out) {
 __shared__ float tmp[256],inv[5],prob[5];const int t=threadIdx.x,q=blockIdx.x;
 // Distinct per-state materialized F32 squares and reduction; shared reuse barriers.
 for(int d=0;d<count;++d) {
  float sum=0;for(int k=t;k<width;k+=256){float v=states[d*width+k];sum+=v*v;}
  tmp[t]=sum;__syncthreads();
  for(int s=128;s;s/=2){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}
  if(t==0)inv[d]=1.0f/sqrtf(tmp[0]/float(width)+1e-6f);
  __syncthreads();
 }
 if(t==0) {
  float maximum=-INFINITY;
  for(int d=0;d<count;++d){float dot=0;for(int k=0;k<width;++k){float normalized=states[d*width+k]*inv[d];dot=fmaf(queries[q*width+k],normalized,dot);}prob[d]=dot*(1.0f/sqrtf(float(width)));maximum=fmaxf(maximum,prob[d]);}
  float sum=0;for(int d=0;d<count;++d){prob[d]=expf(prob[d]-maximum);sum+=prob[d];}
  for(int d=0;d<count;++d)prob[d]=prob[d]/sum;
 }
 __syncthreads();
 for(int k=t;k<width;k+=256){float sum=0;for(int d=0;d<count;++d){float normalized=states[d*width+k]*inv[d];sum=fmaf(normalized,prob[d],sum);}out[q*width+k]=sum;}
}
__global__ void router(const float* logits,const float* bias,int32_t* ids,float* weights) {
 __shared__ float score[128],rank[128];int t=threadIdx.x;
 score[t]=1.0f/(1.0f+expf(-logits[t]));rank[t]=score[t]+bias[t];__syncthreads();
 if(t==0){int selected[8];float sum=0;
  for(int k=0;k<8;++k){int best=-1;for(int j=0;j<128;++j){bool used=false;for(int p=0;p<k;++p)used|=selected[p]==j;if(!used&&(best<0||rank[j]>rank[best]))best=j;}selected[k]=best;sum+=score[best];}
  const float denom=sum+1e-20f;
  for(int k=0;k<8;++k){ids[k]=selected[k];weights[k]=(score[selected[k]]/denom)*8.0f;}
 }
}
__global__ void control(const float* cur,const float* msg,const float* gain,const float* forget,int n,float* out) {
 int i=blockIdx.x*256+threadIdx.x;if(i<n){float a=(1.0f-tanhf(forget[i]))*cur[i];float b=tanhf(gain[i])*msg[i];out[i]=a+b;}
}
__global__ void scaled(const float* x,int n,float factor,float* out) {int i=blockIdx.x*256+threadIdx.x;if(i<n)out[i]=x[i]*factor;}
__global__ void add(const float* a,const float* b,int n,float* out) {int i=blockIdx.x*256+threadIdx.x;if(i<n)out[i]=a[i]+b[i];}
__global__ void axpy(const float* x,const float* weight,int n,float* out) {int i=blockIdx.x*256+threadIdx.x;if(i<n){float product=x[i]*(*weight);out[i]=out[i]+product;}}
}
#define BEGIN() NS_CU(hipSetDevice(g->dev))
#define END() NS_LAUNCH_CHECK(); return 0
extern "C" int ni2_depthmix(ns_gpu* g,const float* states,const float* queries,int count,int nq,int width,float* out) {
 Span a,b,c;if(!g||count<1||count>5||nq<1||nq>4||width<1||width>8192||!span(states,size_t(count)*width*4,a)||!span(queries,size_t(nq)*width*4,b)||!span(out,size_t(nq)*width*4,c)||!disjoint(a,c)||!disjoint(b,c))return ns_fail("inkling depthmix: geometry/span/alias");
 BEGIN();depth<<<nq,256,0,g->q>>>(states,queries,count,width,out);END();
}
extern "C" int ni2_router(ns_gpu* g,const float* logits,const float* bias,int32_t* ids,float* weights) {
 Span a,b,c,d;if(!g||!span(logits,512,a)||!span(bias,512,b)||!span(ids,32,c)||!span(weights,32,d)||!disjoint(a,c)||!disjoint(a,d)||!disjoint(b,c)||!disjoint(b,d)||!disjoint(c,d))return ns_fail("inkling router: span/alias");
 BEGIN();router<<<1,128,0,g->q>>>(logits,bias,ids,weights);END();
}
extern "C" int ni2_loop_control(ns_gpu* g,const float* cur,const float* msg,const float* gain,const float* forget,int n,float* out) {
 if(!g||!output(cur,out,n)||!output(msg,out,n)||!output(gain,out,n)||!output(forget,out,n))return ns_fail("inkling control: span/alias");
 BEGIN();control<<<(n+255)/256,256,0,g->q>>>(cur,msg,gain,forget,n,out);END();
}
extern "C" int ni2_scale(ns_gpu* g,const float* x,int n,float factor,float* out) {
 if(!g||!std::isfinite(factor)||!output(x,out,n))return ns_fail("inkling scale: span/factor/alias");
 BEGIN();scaled<<<(n+255)/256,256,0,g->q>>>(x,n,factor,out);END();
}
extern "C" int ni2_add(ns_gpu* g,const float* a,const float* b,int n,float* out) {
 if(!g||!output(a,out,n)||!output(b,out,n))return ns_fail("inkling add: span/alias");
 BEGIN();add<<<(n+255)/256,256,0,g->q>>>(a,b,n,out);END();
}
extern "C" int ni2_axpy(ns_gpu* g,const float* x,const float* weight,int n,float* out) {
 Span a,b;if(!g||!output(x,out,n)||!span(weight,4,a)||!span(out,size_t(n)*4,b)||!disjoint(a,b))return ns_fail("inkling axpy: span/alias");
 BEGIN();axpy<<<(n+255)/256,256,0,g->q>>>(x,weight,n,out);END();
}
#undef BEGIN
#undef END
