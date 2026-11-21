// Source formulas: Transformers Inkling2 046bdb8 and Unsloth merged
// 883286076efe5b85fddb31d9cc7035a6b698ba9b, pinned under reference/inkling2-mini.
// Private scalar baseline; independent CPU fixtures must establish numeric parity.
#include "primitives.h"
#include "ns_internal.hpp"
#include <cmath>
#include <cstdint>
#include <limits>

namespace {
constexpr int MAX_WIDTH = 8192;
constexpr int MAX_ELEMENTS = 16 * 1024 * 1024;
bool extent(int rows, int width) {
    return rows > 0 && width > 0 && width <= MAX_WIDTH &&
        int64_t(rows) * width <= MAX_ELEMENTS;
}
bool finite_positive(float v) { return std::isfinite(v) && v > 0; }
bool span(const void* p, size_t n, uintptr_t& a, uintptr_t& b) {
    a = reinterpret_cast<uintptr_t>(p);
    if (!p || a % alignof(float) || n > std::numeric_limits<uintptr_t>::max() - a) return false;
    b = a + n; return true;
}
bool compatible(const float* x, const float* y, size_t count, bool exact_alias) {
    uintptr_t a,b,c,d;
    if (!span(x,count*sizeof(float),a,b) || !span(y,count*sizeof(float),c,d)) return false;
    return (exact_alias && a == c) || b <= c || d <= a;
}
// All threads take the same barriers; no wave-size assumptions.
__device__ float reduce_sum(float v) {
    __shared__ float values[256];
    int t=threadIdx.x; values[t]=v; __syncthreads();
    for(int s=128;s;s/=2) { if(t<s) values[t] += values[t+s]; __syncthreads(); }
    float out=values[0]; __syncthreads(); return out;
}
__global__ void rms(const float* x,const float* gamma,int width,float eps,float* out) {
    int row=blockIdx.x,t=threadIdx.x; float sum=0;
    for(int i=t;i<width;i+=256) { float v=x[row*width+i]; sum += v*v; }
    float inv=1.0f/sqrtf(reduce_sum(sum)/float(width)+eps);
    for(int i=t;i<width;i+=256) out[row*width+i]=(x[row*width+i]*inv)*(gamma?gamma[i]:1.0f);
}
__global__ void clip(const float* x,const float* scale,int width,float threshold,float* out) {
    int row=blockIdx.x,t=threadIdx.x; float sum=0;
    for(int i=t;i<width;i+=256) { int at=row*width+i; float v=x[at]*(scale?scale[at]:1.0f); sum += v*v; }
    float factor=fminf(1.0f,threshold/(sqrtf(reduce_sum(sum))+1e-6f));
    for(int i=t;i<width;i+=256) { int at=row*width+i; out[at]=(x[at]*(scale?scale[at]:1.0f))*factor; }
}
__global__ void situ(const float* gate,const float* up,int n,float beta,float linear_beta,float* out) {
    int i=blockIdx.x*256+threadIdx.x;
    if(i<n) { float g=gate[i],u=up[i];
        out[i]=((beta*tanhf(g/beta))*(1.0f/(1.0f+expf(-g))))*(linear_beta*tanhf(u/linear_beta));
    }
}
__global__ void gate(const float* x,const float* g,int n,float* out) {
    int i=blockIdx.x*256+threadIdx.x; if(i<n) out[i]=x[i]*(1.0f/(1.0f+expf(-g[i])));
}
__global__ void softcap(const float* x,int n,float scale,float* out) {
    int i=blockIdx.x*256+threadIdx.x; if(i<n) out[i]=scale*tanhf(x[i]/scale);
}
bool binary_args(const float* a,const float* b,int n,float* out) {
    return n>0 && n<=MAX_ELEMENTS && compatible(a,out,size_t(n),true) && compatible(b,out,size_t(n),true);
}
}
#define START() if(!g) return ns_fail("inkling2: null GPU"); NS_CU(hipSetDevice(g->dev))
#define DONE() NS_LAUNCH_CHECK(); return 0
extern "C" int ni2_rms(ns_gpu* g,const float* x,const float* gamma,int rows,int width,float eps,float* out) {
    if(!extent(rows,width)||!finite_positive(eps)||!compatible(x,out,size_t(rows)*width,true)) return ns_fail("inkling2: invalid RMS span/geometry");
    if(gamma) { uintptr_t a,b,c,d; if(!span(gamma,size_t(width)*4,a,b)||!span(out,size_t(rows)*width*4,c,d)||!(b<=c||d<=a)) return ns_fail("inkling2: RMS gamma aliases output"); }
    START(); rms<<<rows,256,0,g->q>>>(x,gamma,width,eps,out); DONE();
}
extern "C" int ni2_clip(ns_gpu* g,const float* x,const float* scale,int rows,int width,float threshold,float* out) {
    if(!extent(rows,width)||!finite_positive(threshold)||!compatible(x,out,size_t(rows)*width,true)||
       (scale&&!compatible(scale,out,size_t(rows)*width,false))) return ns_fail("inkling2: invalid clip span/geometry");
    START(); clip<<<rows,256,0,g->q>>>(x,scale,width,threshold,out); DONE();
}
extern "C" int ni2_situ(ns_gpu* g,const float* a,const float* b,int n,float beta,float linear_beta,float* out) {
    if(!binary_args(a,b,n,out)||!finite_positive(beta)||!finite_positive(linear_beta)) return ns_fail("inkling2: invalid SiTU arguments");
    START(); situ<<<(n+255)/256,256,0,g->q>>>(a,b,n,beta,linear_beta,out); DONE();
}
extern "C" int ni2_gate(ns_gpu* g,const float* x,const float* gate_values,int n,float* out) {
    if(!binary_args(x,gate_values,n,out)) return ns_fail("inkling2: invalid gate arguments");
    START(); gate<<<(n+255)/256,256,0,g->q>>>(x,gate_values,n,out); DONE();
}
extern "C" int ni2_softcap(ns_gpu* g,const float* x,int n,float scale,float* out) {
    if(n<=0||n>MAX_ELEMENTS||!finite_positive(scale)||!compatible(x,out,size_t(n),true)) return ns_fail("inkling2: invalid softcap arguments");
    START(); softcap<<<(n+255)/256,256,0,g->q>>>(x,n,scale,out); DONE();
}
