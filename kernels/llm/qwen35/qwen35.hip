// Scalar GGUF readers adapted from nextsycl kernels/llm/common/quants.hpp
// TQ4K/TQ5K/TQ6K. Float accumulation deliberately avoids activation requantization.
// Stream/error handling follows nextcuda's nh.cu and nextrocm's q3t.hip.
#include "qwen35.h"
#include "quants.hpp"
#include "ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
#include <climits>
namespace {
__device__ float half_at(const uint8_t* p) { return __half2float(*reinterpret_cast<const __half*>(p)); }
struct HalfReader { __device__ float operator()(const uint8_t* p) const { return half_at(p); } };
__device__ float value(const void* w, int type, int i) {
    if (type == 0) return static_cast<const float*>(w)[i];
    if (type == 1) return __half2float(static_cast<const __half*>(w)[i]);
    return nq35::quant(w, type, i, HalfReader{});
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
__global__ void norm(const float* x,const float* w,int n,float eps,float* out) {
    float v=0; for(int i=threadIdx.x;i<n;i+=256)v+=x[i]*x[i];
    float inv=rsqrtf(sum(v)/n+eps);
    for(int i=threadIdx.x;i<n;i+=256)out[i]=x[i]*inv*w[i];
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
__global__ void attn(const float* q,const __half* k,const __half* v,int heads,int kh,int d,int pos,float* out) {
    int h=blockIdx.x,t=threadIdx.x;float acc=0,den=0,m=-INFINITY;
    float qi=t<d?q[h*d+t]:0;
    for(int p=0;p<=pos;++p){
        int at=p*kh*d+(h/(heads/kh))*d+t;
        float dot=sum(t<d?qi*__half2float(k[at]):0)*rsqrtf(float(d));
        float nm=fmaxf(m,dot),a=expf(m-nm),b=expf(dot-nm);
        acc=acc*a+(t<d?__half2float(v[at])*b:0);den=den*a+b;m=nm;
    }
    if(t<d)out[h*d+t]=acc/den;
}
__global__ void axpy(float* x,const float* y,int n,float scale) {
    int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]+=scale*y[i];
}
__global__ void qk(const float* x,const float* w,int d,int stride,int rot,int pos,float theta,float eps,float* y) {
    int h=blockIdx.x,t=threadIdx.x;float v=t<d?x[h*stride+t]:0;
    float inv=rsqrtf(sum(v*v)/d+eps);
    __shared__ float row[256];if(t<d)row[t]=v*inv*w[t];__syncthreads();
    if(t<d){float out=row[t];if(t<rot){int j=t%(rot/2);float a=pos*powf(theta,-2.0f*j/rot);
        out=t<rot/2 ? row[j]*cosf(a)-row[j+rot/2]*sinf(a) : row[j]*sinf(a)+row[j+rot/2]*cosf(a);}
        y[h*d+t]=out;}
}
__global__ void gate(float* x,const float* q,int n,int d) {
    int i=blockIdx.x*256+threadIdx.x;if(i<n)x[i]/=1+expf(-q[(i/d)*2*d+d+i%d]);
}
__global__ void conv(const float* x,const float* w,float* history,int channels,int taps,float* y) {
    int c=blockIdx.x*256+threadIdx.x;if(c>=channels)return;
    float v=x[c]*w[c*taps+taps-1];
    for(int j=0;j<taps-1;++j)v+=history[j*channels+c]*w[c*taps+j];
    for(int j=0;j<taps-2;++j)history[j*channels+c]=history[(j+1)*channels+c];
    history[(taps-2)*channels+c]=x[c];y[c]=v/(1+expf(-v));
}
// One block/value column: S[head][value][key]. All blocks read immutable q/k/v.
// Reference expansion is h % key_heads (not contiguous GQA grouping).
__global__ void delta(const float* co,const float* alpha,const float* beta,const float* dt,const float* a,
                      float* state,int hk,int hv,int d,float eps,float* y) {
    int h=blockIdx.x/d,col=blockIdx.x%d,t=threadIdx.x,kh=h%hk;
    float q=t<d?co[kh*d+t]:0,k=t<d?co[hk*d+kh*d+t]:0;
    q*=rsqrtf(sum(q*q)+1e-6f)*rsqrtf(float(d));k*=rsqrtf(sum(k*k)+1e-6f);
    float z=alpha[h]+dt[h],decay=expf(a[h]*(z>20?z:log1pf(expf(z))));
    int at=(h*d+col)*d+t;float s=t<d?state[at]*decay:0;
    float pred=sum(s*k),dv=(co[2*hk*d+h*d+col]-pred)/(1+expf(-beta[h]));
    s+=k*dv;if(t<d)state[at]=s;
    float out=sum(s*q);if(t==0)y[h*d+col]=out;
}
__global__ void gated_norm(const float* x,const float* z,const float* w,int d,float eps,float* out) {
    int h=blockIdx.x,t=threadIdx.x;float v=t<d?x[h*d+t]:0;
    float inv=rsqrtf(sum(v*v)/d+eps);
    if(t<d){float zz=z[h*d+t];out[h*d+t]=v*inv*w[t]*zz/(1+expf(-zz));}
}
bool type_ok(int t){return t==0||t==1||t==8||t==12||t==13||t==14;}
}

#define BEGIN() if(!g)return ns_fail("qwen35: null GPU"); NS_CU(hipSetDevice(g->dev))
#define END() NS_LAUNCH_CHECK(); return 0
extern "C" int nq35_embed(ns_gpu* g,const void* w,int t,int n,float* out){
 BEGIN();if(!w||!out||!type_ok(t)||n<=0)return ns_fail("qwen35: invalid embedding");
 embed<<<(n+255)/256,256,0,g->q>>>(w,t,n,0,out);END();}
extern "C" int nq35_mv(ns_gpu* g,const void* w,int t,int rows,int cols,const float* x,float* y){
 BEGIN();if(!w||!x||!y||!type_ok(t)||rows<=0||cols<=0||int64_t(rows)*cols>INT32_MAX||(t==8&&cols%32)||(t>=12&&cols%256))return ns_fail("qwen35: invalid matrix");
 mv<<<rows,256,0,g->q>>>(w,t,cols,x,y);END();}
extern "C" int nq35_norm(ns_gpu* g,const float* x,const float* w,int n,float eps,float* out){
 BEGIN();if(!x||!w||!out||n<=0||!(eps>0))return ns_fail("qwen35: invalid norm"); norm<<<1,256,0,g->q>>>(x,w,n,eps,out);END();}
extern "C" int nq35_axpy(ns_gpu* g,float* x,const float* y,int n,float scale){
 BEGIN();if(!x||!y||n<=0||!std::isfinite(scale))return ns_fail("qwen35: invalid axpy");axpy<<<(n+255)/256,256,0,g->q>>>(x,y,n,scale);END();}
extern "C" int nq35_swiglu(ns_gpu* g,float* x,const float* y,int n){
 BEGIN();if(!x||!y||n<=0)return ns_fail("qwen35: invalid swiglu");swiglu<<<(n+255)/256,256,0,g->q>>>(x,y,n);END();}
extern "C" int nq35_qk(ns_gpu* g,const float* x,const float* w,int heads,int d,int stride,int rot,int pos,float theta,float eps,float* y){
 BEGIN();if(!x||!w||!y||heads<=0||heads>256||d<=0||d>256||stride<d||rot<=0||rot>d||rot%2||pos<0||!(theta>0)||!(eps>0))return ns_fail("qwen35: invalid qk");
 qk<<<heads,256,0,g->q>>>(x,w,d,stride,rot,pos,theta,eps,y);END();}
extern "C" int nq35_attn(ns_gpu* g,const float* q,const float* k,const float* v,void* kc,void* vc,int heads,int kh,int d,int pos,float* out){
 BEGIN();if(!q||!k||!v||!kc||!vc||!out||heads<=0||heads>256||kh<=0||heads%kh||d<=0||d>256||pos<0||int64_t(pos+1)*kh*d>INT32_MAX)return ns_fail("qwen35: invalid attention");
 int n=kh*d;cache<<<(n+255)/256,256,0,g->q>>>(k,v,static_cast<__half*>(kc),static_cast<__half*>(vc),n,pos);NS_LAUNCH_CHECK();
 attn<<<heads,256,0,g->q>>>(q,static_cast<const __half*>(kc),static_cast<const __half*>(vc),heads,kh,d,pos,out);END();}
extern "C" int nq35_gate(ns_gpu* g,float* x,const float* q,int heads,int d){
 BEGIN();if(!x||!q||heads<=0||heads>256||d<=0||d>256)return ns_fail("qwen35: invalid gate");int n=heads*d;gate<<<(n+255)/256,256,0,g->q>>>(x,q,n,d);END();}
extern "C" int nq35_conv(ns_gpu* g,const float* x,const float* w,float* history,int channels,int taps,float* y){
 BEGIN();if(!x||!w||!history||!y||channels<=0||channels>196608||taps<2||taps>9)return ns_fail("qwen35: invalid conv");conv<<<(channels+255)/256,256,0,g->q>>>(x,w,history,channels,taps,y);END();}
extern "C" int nq35_delta(ns_gpu* g,const float* co,const float* alpha,const float* beta,const float* dt,const float* a,float* state,int hk,int hv,int d,float eps,float* y){
 BEGIN();if(!co||!alpha||!beta||!dt||!a||!state||!y||hk<=0||hv<=0||hv>256||hv%hk||d<=0||d>256||!(eps>0))return ns_fail("qwen35: invalid delta");delta<<<hv*d,256,0,g->q>>>(co,alpha,beta,dt,a,state,hk,hv,d,eps,y);END();}
extern "C" int nq35_gated_norm(ns_gpu* g,const float* x,const float* z,const float* w,int heads,int d,float eps,float* out){
 BEGIN();if(!x||!z||!w||!out||heads<=0||heads>256||d<=0||d>256||!(eps>0))return ns_fail("qwen35: invalid gated norm");gated_norm<<<heads,256,0,g->q>>>(x,z,w,d,eps,out);END();}
