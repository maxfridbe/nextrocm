// Independent depth candidate: parallel independent state dots, unchanged per-dot order.
// Compile INSTEAD of mix.hip. Retain all old operators and original depth export for validation.
#define ni2_depthmix ni2_depthmix_baseline
#include "mix.hip"
#undef ni2_depthmix
namespace {
__global__ void depth_parallel(const float* states,const float* queries,int count,int width,float* out) {
 __shared__ float tmp[256],inv[5],prob[5];const int t=threadIdx.x,q=blockIdx.x;
 // Distinct per-state materialized F32 squares and reduction; shared reuse barriers.
 for(int d=0;d<count;++d) {
  float sum=0;for(int k=t;k<width;k+=256){float v=states[d*width+k];sum+=v*v;}
  tmp[t]=sum;__syncthreads();
  for(int s=128;s;s/=2){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}
  if(t==0)inv[d]=1.0f/sqrtf(tmp[0]/float(width)+1e-6f);
  __syncthreads();
 }
 // Each state dot is independent. Preserve ascending-k FMA and normalized
 // F32 materialization exactly; only move distinct states to distinct lanes.
 if(t<count) {
  float dot=0;
  for(int k=0;k<width;++k){float normalized=states[t*width+k]*inv[t];dot=fmaf(queries[q*width+k],normalized,dot);}
  prob[t]=dot*(1.0f/sqrtf(float(width)));
 }
 __syncthreads();
 if(t==0) {
  float maximum=-INFINITY;
  for(int d=0;d<count;++d)maximum=fmaxf(maximum,prob[d]);
  float sum=0;for(int d=0;d<count;++d){prob[d]=expf(prob[d]-maximum);sum+=prob[d];}
  for(int d=0;d<count;++d)prob[d]=prob[d]/sum;
 }
 __syncthreads();
 for(int k=t;k<width;k+=256){float sum=0;for(int d=0;d<count;++d){float normalized=states[d*width+k]*inv[d];sum=fmaf(normalized,prob[d],sum);}out[q*width+k]=sum;}
}
}
extern "C" int ni2_depthmix(ns_gpu* g,const float* states,const float* queries,int count,int nq,int width,float* out) {
 Span a,b,c;if(!g||count<1||count>5||nq<1||nq>4||width<1||width>8192||!span(states,size_t(count)*width*4,a)||!span(queries,size_t(nq)*width*4,b)||!span(out,size_t(nq)*width*4,c)||!disjoint(a,c)||!disjoint(b,c))return ns_fail("inkling depthmix: geometry/span/alias");
 NS_CU(hipSetDevice(g->dev));depth_parallel<<<nq,256,0,g->q>>>(states,queries,count,width,out);NS_LAUNCH_CHECK(); return 0;
}
