// Scalar GGUF readers adapted from nextsycl kernels/llm/common/quants.hpp
// TQ4K/TQ6K. Float accumulation deliberately avoids activation requantization.
// Stream/error handling follows nextcuda's nh.cu and nextrocm's q3t.hip.
#include "stablelm.h"
#include "quants.hpp"
#include "ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
namespace {
__device__ float half_at(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }
struct HalfReader { __device__ float operator()(const uint8_t* p) const { return half_at(p); } };
__device__ float value(const void* w, int type, int i) {
    if (type == 0) return static_cast<const float*>(w)[i];
    if (type == 1) return __half2float(static_cast<const __half*>(w)[i]);
    return nslm::quant(w, type, i, HalfReader{});
}
// Fixed 256-thread reduction, independent of wave32/wave64.
__device__ float sum(float x) {
    __shared__ float tmp[256];
    const int t = threadIdx.x;
    tmp[t] = x; __syncthreads();
    for (int k=128; k; k/=2) { if (t<k) tmp[t]+=tmp[t+k]; __syncthreads(); }
    const float out=tmp[0]; __syncthreads(); return out;
}
__global__ void embed(const void* w, int type, int n, int token, float* out) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n) out[i]=value(w,type,token*n+i);
}
__global__ void mv(const void* w,int type,int cols,const float* x,float* y) {
    int row=blockIdx.x; float v=0;
    for(int i=threadIdx.x;i<cols;i+=256) v+=value(w,type,row*cols+i)*x[i];
    float s=sum(v); if(threadIdx.x==0)y[row]=s;
}
__global__ void norm(const float* x,const float* w,const float* b,int n,float eps,float* out) {
    float v=0; for(int i=threadIdx.x;i<n;i+=256)v+=x[i];
    float mean=sum(v)/n; v=0;
    for(int i=threadIdx.x;i<n;i+=256){float d=x[i]-mean;v+=d*d;}
    float inv=rsqrtf(sum(v)/n+eps);
    for(int i=threadIdx.x;i<n;i+=256)out[i]=(x[i]-mean)*inv*w[i]+b[i];
}
__global__ void add(float* x,const float* y,int n) {
    int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]+=y[i];
}
__global__ void swiglu(float* x,const float* y,int n) {
    int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]=x[i]/(1+expf(-x[i]))*y[i];
}
__global__ void rope(float* x,int heads,int head,int rot,int pos,float theta) {
    int i=blockIdx.x*256+threadIdx.x;if(i>=heads*rot/2)return;
    int h=i/(rot/2),j=i%(rot/2),a=h*head+j,b=a+rot/2;
    float angle=pos*powf(theta,-2.0f*j/rot),c=cosf(angle),s=sinf(angle),v=x[a],u=x[b];
    x[a]=v*c-u*s;x[b]=v*s+u*c;
}
__global__ void cache(const float* k,const float* v,__half* kc,__half* vc,int n,int pos) {
    int i=blockIdx.x*256+threadIdx.x;if(i<n){kc[pos*n+i]=__float2half(k[i]);vc[pos*n+i]=__float2half(v[i]);}
}
// One block/head; online softmax, channel-parallel dot and value accumulation.
__global__ void attn(const float* q,const __half* k,const __half* v,int heads,int d,int pos,float* out) {
    int h=blockIdx.x,t=threadIdx.x;float acc=0,den=0,m=-INFINITY;
    float qi=t<d?q[h*d+t]:0;
    for(int p=0;p<=pos;++p){
        int at=p*heads*d+h*d+t;
        float dot=sum(t<d?qi*__half2float(k[at]):0)*rsqrtf(float(d));
        float nm=fmaxf(m,dot),a=expf(m-nm),b=expf(dot-nm);
        acc=acc*a+(t<d?__half2float(v[at])*b:0);den=den*a+b;m=nm;
    }
    if(t<d)out[h*d+t]=acc/den;
}
bool type_ok(int t){return t==0||t==1||t==12||t==14;}
}
#define BEGIN() if(!g)return ns_fail("stablelm: null GPU"); NS_CU(hipSetDevice(g->dev))
#define END() NS_LAUNCH_CHECK(); return 0
extern "C" int nslm_embed(ns_gpu* g,const void* w,int t,int n,int token,float* out){
    BEGIN();if(!type_ok(t)||n<=0||token<0)return ns_fail("stablelm: invalid embedding");
    embed<<<(n+255)/256,256,0,g->q>>>(w,t,n,token,out);END();
}
extern "C" int nslm_mv(ns_gpu* g,const void* w,int t,int rows,int cols,const float* x,float* y){
    BEGIN();if(!type_ok(t)||rows<=0||cols<=0||(t>=12&&cols%256))return ns_fail("stablelm: invalid matrix");
    mv<<<rows,256,0,g->q>>>(w,t,cols,x,y);END();
}
extern "C" int nslm_norm(ns_gpu* g,const float* x,const float* w,const float* b,int n,float eps,float* out){
    BEGIN();if(n<=0||eps<=0)return ns_fail("stablelm: invalid norm");norm<<<1,256,0,g->q>>>(x,w,b,n,eps,out);END();
}
extern "C" int nslm_add(ns_gpu* g,float* x,const float* y,int n){BEGIN();if(n<=0)return ns_fail("stablelm: invalid add");add<<<(n+255)/256,256,0,g->q>>>(x,y,n);END();}
extern "C" int nslm_swiglu(ns_gpu* g,float* x,const float* y,int n){BEGIN();if(n<=0)return ns_fail("stablelm: invalid swiglu");swiglu<<<(n+255)/256,256,0,g->q>>>(x,y,n);END();}
extern "C" int nslm_rope(ns_gpu* g,float* x,int heads,int head,int rot,int pos,float theta){
    BEGIN();if(heads<=0||rot<=0||rot%2||rot>head||pos<0||theta<=0)return ns_fail("stablelm: invalid rope");
    rope<<<(heads*rot/2+255)/256,256,0,g->q>>>(x,heads,head,rot,pos,theta);END();
}
extern "C" int nslm_attn(ns_gpu* g,const float* q,const float* k,const float* v,void* kc,void* vc,int heads,int head,int pos,float* out){
    BEGIN();if(heads<=0||head<=0||head>256||pos<0)return ns_fail("stablelm: invalid attention");
    int n=heads*head;
    cache<<<(n+255)/256,256,0,g->q>>>(k,v,static_cast<__half*>(kc),static_cast<__half*>(vc),n,pos);NS_LAUNCH_CHECK();
    attn<<<heads,256,0,g->q>>>(q,static_cast<const __half*>(kc),static_cast<const __half*>(vc),heads,head,pos,out);END();
}
