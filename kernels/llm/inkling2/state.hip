// Source contracts: merged llama.cpp 883286... src/models/inkling2.cpp
// convolve(), and pinned HF modeling_inkling2.py eager_attention_forward.
// F32 baseline, not a claim of BF16 or dense GGML reduction-order parity.
#include "state.h"
#include "state.hpp"
#include "ns_internal.hpp"
#include <cmath>
namespace {
__global__ void shortconv(const float* x,const float* w,const float* dynamic,
 int heads,int channels,float* history,float* out) {
    int c=blockIdx.x*256+threadIdx.x;
    if(c>=channels) return;
    float input[4]={history[c*3],history[c*3+1],history[c*3+2],x[c]};
    float sum=0;
    for(int tap=0;tap<4;++tap) sum=fmaf(input[tap],w[c*4+tap],sum);
    if(dynamic) {
        int head=c/(channels/heads);
        for(int lag=0;lag<4;++lag)
            sum=__fadd_rn(sum,__fmul_rn(input[3-lag],dynamic[head*4+lag]));
    }
    out[c]=__fmul_rn(__fadd_rn(input[3],sum),dynamic ? 1.0f/3.0f : 1.0f/sqrtf(5.0f));
    history[c*3]=input[1]; history[c*3+1]=input[2]; history[c*3+2]=input[3];
}
__global__ void logits(const float* q,const float* k,const float* r,const float* w,
 int64_t pos,int cap,int n,float* scores) {
    int i=blockIdx.x*256+threadIdx.x,head=blockIdx.y;
    if(i>=n) return;
    int64_t key=pos-n+1+i;
    int row=int(key%cap),kv=head/3;
    float dot=0;
    for(int d=0;d<128;++d) dot=fmaf(q[head*128+d],k[(row*8+kv)*128+d],dot);
    int distance=n-1-i; float bias=0;
    if(distance<1024) for(int d=0;d<16;++d) bias=fmaf(r[head*16+d],w[distance*16+d],bias);
    scores[head*n+i]=__fadd_rn(__fmul_rn(dot,1.0f/sqrtf(128.0f)),bias);
}
__global__ void probabilities(float* scores,const float* sink,int n) {
    __shared__ float values[256];
    int t=threadIdx.x,head=blockIdx.x; float m=sink[head];
    for(int i=t;i<n;i+=256) m=fmaxf(m,scores[head*n+i]);
    values[t]=m; __syncthreads();
    for(int s=128;s;s/=2) { if(t<s) values[t]=fmaxf(values[t],values[t+s]); __syncthreads(); }
    m=values[0]; __syncthreads();
    float sum=t==0?expf(sink[head]-m):0.0f;
    for(int i=t;i<n;i+=256) { float e=expf(scores[head*n+i]-m); scores[head*n+i]=e; sum+=e; }
    values[t]=sum; __syncthreads();
    for(int s=128;s;s/=2) { if(t<s) values[t]+=values[t+s]; __syncthreads(); }
    float den=values[0];
    for(int i=t;i<n;i+=256) scores[head*n+i]/=den;
}
__global__ void weighted(const float* scores,const float* v,int64_t pos,int cap,int n,float* out) {
    int head=blockIdx.x,d=threadIdx.x,kv=head/3;
    float sum=0;
    for(int i=0;i<n;++i) {
        int row=int((pos-n+1+i)%cap);
        sum=fmaf(scores[head*n+i],v[(row*8+kv)*128+d],sum);
    }
    out[head*128+d]=sum;
}
}
extern "C" int ni2_shortconv(ns_gpu* g,const float* x,const float* w,const float* d,
 int heads,int c,float* history,float* out) {
    if(!g||!ni2_state::conv_args(x,w,d,heads,c,history,out)) return ns_fail("inkling2: invalid shortconv arguments");
    NS_CU(hipSetDevice(g->dev));
    shortconv<<<(c+255)/256,256,0,g->q>>>(x,w,d,heads,c,history,out);
    NS_LAUNCH_CHECK(); return 0;
}
extern "C" int ni2_attention(ns_gpu* g,const float* q,const float* k,const float* v,
 const float* r,const float* w,const float* sink,int64_t pos,int cap,int window,
 float* out,float* scratch,int64_t count) {
    if(!g||!ni2_state::attention_args(q,k,v,r,w,sink,pos,cap,window,out,scratch,count)) return ns_fail("inkling2: invalid attention arguments");
    NS_CU(hipSetDevice(g->dev));
    int n=ni2_state::key_count(pos,cap,window);
    logits<<<dim3((n+255)/256,24),256,0,g->q>>>(q,k,r,w,pos,cap,n,scratch); NS_LAUNCH_CHECK();
    probabilities<<<24,256,0,g->q>>>(scratch,sink,n); NS_LAUNCH_CHECK();
    weighted<<<24,128,0,g->q>>>(scratch,v,pos,cap,n,out); NS_LAUNCH_CHECK();
    return 0;
}
