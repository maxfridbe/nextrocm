// Private DiT normalization/modulation/RMSRoPE; original nextsycl semantics.
// No shared binder changes, external runtime, global scratch, or default switch.
#pragma once
#include "dit_ops.h"
#include <cmath>
#include <cstdint>
namespace nrqi_dit {
using I=int64_t;
struct Span{uintptr_t lo=0,hi=0;};
inline bool span(const void*p,uint64_t n,unsigned bytes,unsigned align,Span&s){uintptr_t a=reinterpret_cast<uintptr_t>(p);if(!p||!n||a%align||n>UINTPTR_MAX/bytes||n*bytes>UINTPTR_MAX-a)return false;s={a,a+uintptr_t(n*bytes)};return true;}
inline bool overlap(Span a,Span b){return a.lo<b.hi&&b.lo<a.hi;}
inline bool geometry(I m,I c){return m>=1&&m<=65536&&c>=1&&c<=8192;}
inline bool epsilon(float e){return std::isfinite(e)&&e>0;}
inline bool valid_ln(void*g,const float*x,I m,I c,float eps,const float*scale,void*out){
 if(!g||!geometry(m,c)||!epsilon(eps))return false;
 Span a,b,s;
 if(!span(x,uint64_t(m)*c,4,4,a)||!span(out,uint64_t(m)*c,2,2,b)||overlap(a,b))return false;
 return !scale||(span(scale,c,4,4,s)&&!overlap(s,b));
}
inline bool valid_rms(void*g,const float*x,I m,I c,const float*w,float eps,float*out){
 if(!g||!geometry(m,c)||!epsilon(eps))return false;
 Span a,b,s;
 return span(x,uint64_t(m)*c,4,4,a)&&span(out,uint64_t(m)*c,4,4,b)&&span(w,c,4,4,s)&&(!overlap(a,b)||a.lo==b.lo)&&!overlap(s,b);
}
inline bool valid_rope(void*g,void*x,I m,I h,I d,I stride,const float*w,float eps,const float*cs,int rot){
 if(!g||m<1||m>65536||h<1||h>64||d<2||d>256||d%2||stride<h*d||stride>65536||rot<0||rot>d||rot%2||!epsilon(eps))return false;
 Span a,b,c;if(!span(x,uint64_t(m-1)*stride+h*d,2,2,a)||!span(w,d,4,4,b)||overlap(a,b))return false;
 return rot==0||(span(cs,uint64_t(m)*rot,4,4,c)&&!overlap(a,c));
}
inline bool valid_gate(void*g,float*x,const void*y,I m,I c,const float*gate){
 if(!g||!geometry(m,c))return false;
 Span a,b,s;
 if(!span(x,uint64_t(m)*c,4,4,a)||!span(y,uint64_t(m)*c,2,2,b)||overlap(a,b))return false;
 return !gate||(span(gate,c,4,4,s)&&!overlap(a,s));
}
}
#ifdef __HIPCC__
#ifdef __FAST_MATH__
#error "DiT operators require explicit F32 arithmetic; no fast math"
#endif
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
namespace nrqi_dit {
__device__ float reduce_sum(float v,float*tmp){int t=threadIdx.x;tmp[t]=v;__syncthreads();for(int n=128;n;n/=2){if(t<n)tmp[t]=__fadd_rn(tmp[t],tmp[t+n]);__syncthreads();}return tmp[0];}
__global__ void ln_kernel(const float*x,I c,float eps,const float*scale,__half*out){
 __shared__ float tmp[256];float v[32];float sum=0;I r=blockIdx.x;int t=threadIdx.x;
 #pragma unroll
 for(int j=0;j<32;++j){I i=I(j)*256+t;v[j]=i<c?x[r*c+i]:0.f;sum=__fadd_rn(sum,v[j]);}
 float mean=reduce_sum(sum,tmp)/float(c);__syncthreads();float var=0;
 #pragma unroll
 for(int j=0;j<32;++j){I i=I(j)*256+t;float d=i<c?__fsub_rn(v[j],mean):0.f;var=__fadd_rn(var,__fmul_rn(d,d));}
 float inv=rsqrtf(__fadd_rn(reduce_sum(var,tmp)/float(c),eps));
 #pragma unroll
 for(int j=0;j<32;++j){I i=I(j)*256+t;if(i<c){float z=__fmul_rn(__fsub_rn(v[j],mean),inv);z=__fmul_rn(z,scale?__fadd_rn(1.f,scale[i]):1.f);out[r*c+i]=__float2half_rn(z);}}
}
__global__ void rms_kernel(const float*x,I c,const float*w,float eps,float*out){
 __shared__ float tmp[256];I r=blockIdx.x;int t=threadIdx.x;float sum=0;
 for(I i=t;i<c;i+=256){float v=x[r*c+i];sum=__fadd_rn(sum,__fmul_rn(v,v));}
 float inv=rsqrtf(__fadd_rn(reduce_sum(sum,tmp)/float(c),eps));
 for(I i=t;i<c;i+=256)out[r*c+i]=__fmul_rn(__fmul_rn(x[r*c+i],inv),w[i]);
}
__global__ void rope_kernel(__half*x,I d,I stride,const float*w,float eps,const float*cs,int rot){
 __shared__ float values[256],inv;I r=blockIdx.x,head=blockIdx.y,base=r*stride+head*d;int t=threadIdx.x;
 for(I i=t;i<d;i+=128) { values[i]=__half2float(x[base+i]); }
 __syncthreads();
 if(t==0){float sum=0;for(I i=0;i<d;++i)sum=__fadd_rn(sum,__fmul_rn(values[i],values[i]));inv=rsqrtf(__fadd_rn(sum/float(d),eps));}__syncthreads();
 int half=rot/2;
 for(I j=t;j<d/2;j+=128){
  if(j<half){float a=__fmul_rn(__fmul_rn(values[j],inv),w[j]);float b=__fmul_rn(__fmul_rn(values[half+j],inv),w[half+j]);float co=cs[(r*half+j)*2],si=cs[(r*half+j)*2+1];x[base+j]=__float2half_rn(__fsub_rn(__fmul_rn(a,co),__fmul_rn(b,si)));x[base+half+j]=__float2half_rn(__fadd_rn(__fmul_rn(b,co),__fmul_rn(a,si)));}
  else{I i=rot+2*(j-half);x[base+i]=__float2half_rn(__fmul_rn(__fmul_rn(values[i],inv),w[i]));x[base+i+1]=__float2half_rn(__fmul_rn(__fmul_rn(values[i+1],inv),w[i+1]));}
 }
}
__global__ void gate_kernel(float*x,const __half*y,I n,I c,const float*gate){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n)x[i]=__fadd_rn(x[i],__fmul_rn(__half2float(y[i]),gate?gate[i%c]:1.f));}
inline int ln(void*ctx,const float*x,I m,I c,float eps,const float*scale,void*out){
 if(!valid_ln(ctx,x,m,c,eps,scale,out))return ns_fail("nrqi_ln_mod: invalid span/epsilon");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));ln_kernel<<<dim3(unsigned(m)),256,0,g->q>>>(x,c,eps,scale,static_cast<__half*>(out));NS_LAUNCH_CHECK();return 0;
}
inline int rms(void*ctx,const float*x,I m,I c,const float*w,float eps,float*out){
 if(!valid_rms(ctx,x,m,c,w,eps,out))return ns_fail("nrqi_text_rms_f32: invalid span/epsilon");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));rms_kernel<<<dim3(unsigned(m)),256,0,g->q>>>(x,c,w,eps,out);NS_LAUNCH_CHECK();return 0;
}
inline int rope(void*ctx,void*x,I m,I h,I d,I stride,const float*w,float eps,const float*cs,int rot){
 if(!valid_rope(ctx,x,m,h,d,stride,w,eps,cs,rot))return ns_fail("nrqi_rms_rope: invalid span/epsilon");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));rope_kernel<<<dim3(unsigned(m),unsigned(h)),128,0,g->q>>>(static_cast<__half*>(x),d,stride,w,eps,cs,rot);NS_LAUNCH_CHECK();return 0;
}
inline int gate(void*ctx,float*x,const void*y,I m,I c,const float*weight){
 if(!valid_gate(ctx,x,y,m,c,weight))return ns_fail("nrqi_gate_add: invalid span");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));gate_kernel<<<dim3(unsigned((m*c+255)/256)),256,0,g->q>>>(x,static_cast<const __half*>(y),m*c,c,weight);NS_LAUNCH_CHECK();return 0;
}
}
#endif
