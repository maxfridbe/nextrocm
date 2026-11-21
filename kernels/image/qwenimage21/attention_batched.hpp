// Private bounded-score image attention. Caller owns reusable workspace.
// Preserve ascending QK/PV FMA, original 256-lane softmax, divide-last half.
#pragma once
#ifdef __FAST_MATH__
#error "Qwen image attention requires baseline arithmetic"
#endif
#define NRQI_ATTENTION_1024_GUARDS_ONLY
#include "attention_1024.hpp"
#undef NRQI_ATTENTION_1024_GUARDS_ONLY
namespace nrqi_attention_batched {
using I=int64_t;
inline I batch(I sq){return sq<256?sq:256;}
inline I workspace(I sq,I sk,I h){
 if(sq<1||sq>4096||sk<1||sk>4608||h<1||h>32)return 0;
 return 4*h*batch(sq)*(sk+1);
}
inline bool arguments(const void*q,const void*k,const void*v,I sq,I sk,I h,I d,I qs,I ks,void*out,void* scratch,I bytes){
 if(!nrqi_attention_1024::arguments(q,k,v,sq,sk,h,d,qs,ks,out))return false;
 const I need=workspace(sq,sk,h);if(bytes<need||!scratch||reinterpret_cast<uintptr_t>(scratch)%4)return false;
 uintptr_t a,ae,b,be;
 if(!nrqi_attention_1024::span(scratch,uintptr_t(need),a,ae))return false;
 const void* p[]={q,k,v,out};const I sizes[]={((sq-1)*qs+h*d)*2,((sk-1)*ks+h*d)*2,((sk-1)*ks+h*d)*2,sq*h*d*2};
 for(int i=0;i<4;++i)if(!nrqi_attention_1024::span(p[i],uintptr_t(sizes[i]),b,be)||!(be<=a||ae<=b))return false;
 return true;
}
}
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include "../../ns/ns_internal.hpp"
namespace nrqi_attention_batched {
__global__ __launch_bounds__(256) void qk(const __half*q,const __half*k,I row0,I rows,I sk,I d,I qs,I ks,I capacity,float*score){
 __shared__ float x[64][33],w[64][33];
 const int t=threadIdx.x,r=t/16,c=t%16;const I qr=I(blockIdx.y)*64,kr=I(blockIdx.x)*64,head=blockIdx.z;
 float acc[4][4]={};
 for(int base=0;base<128;base+=32){
  for(int z=t;z<2048;z+=256){int a=z/32,b=z%32;x[a][b]=qr+a<rows?__half2float(q[(row0+qr+a)*qs+head*128+base+b]):0.f;w[a][b]=kr+a<sk?__half2float(k[(kr+a)*ks+head*128+base+b]):0.f;}
  __syncthreads();
  #pragma unroll
  for(int j=0;j<32;++j){
   #pragma unroll
   for(int a=0;a<4;++a){
    #pragma unroll
    for(int b=0;b<4;++b)acc[a][b]=fmaf(x[r+16*a][j],w[c+16*b][j],acc[a][b]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int a=0;a<4;++a){
  #pragma unroll
  for(int b=0;b<4;++b)if(qr+r+16*a<rows&&kr+c+16*b<sk)score[(head*capacity+qr+r+16*a)*sk+kr+c+16*b]=acc[a][b]*rsqrtf(float(d));
 }
}
__global__ __launch_bounds__(256) void softmax(float*score,float*den,I sk,I capacity){
 __shared__ float tmp[256];const int t=threadIdx.x;const I row=I(blockIdx.y)*capacity+blockIdx.x;float* s=score+row*sk;
 float best=-INFINITY;for(I j=t;j<sk;j+=256)best=fmaxf(best,s[j]);tmp[t]=best;__syncthreads();
 for(int step=128;step;step/=2){if(t<step)tmp[t]=fmaxf(tmp[t],tmp[t+step]);__syncthreads();}
 best=tmp[0];__syncthreads();float sum=0;
 for(I j=t;j<sk;j+=256){float z=expf(s[j]-best);s[j]=z;sum+=z;}tmp[t]=sum;__syncthreads();
 for(int step=128;step;step/=2){if(t<step)tmp[t]+=tmp[t+step];__syncthreads();}
 if(t==0)den[row]=tmp[0];
}
__global__ __launch_bounds__(256) void pv(const float*score,const float*den,const __half*v,I row0,I rows,I sk,I h,I ks,I capacity,__half*out){
 __shared__ float x[64][33],w[32][65];
 const int t=threadIdx.x,r=t/16,c=t%16;const I qr=I(blockIdx.y)*64,dc=I(blockIdx.x)*64,head=blockIdx.z;
 float acc[4][4]={};
 for(I base=0;base<sk;base+=32){
  for(int z=t;z<2048;z+=256){int a=z/32,b=z%32;x[a][b]=(qr+a<rows&&base+b<sk)?score[(head*capacity+qr+a)*sk+base+b]:0.f;int j=z/64,f=z%64;w[j][f]=base+j<sk?__half2float(v[(base+j)*ks+head*128+dc+f]):0.f;}
  __syncthreads();
  // Skip padded keys: even zero FMAs would change the declared operation order.
  const int count=int(sk-base<32?sk-base:32);
  for(int j=0;j<count;++j){
   #pragma unroll
   for(int a=0;a<4;++a){
    #pragma unroll
    for(int b=0;b<4;++b)acc[a][b]=fmaf(x[r+16*a][j],w[j][c+16*b],acc[a][b]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int a=0;a<4;++a)if(qr+r+16*a<rows){float divisor=den[head*capacity+qr+r+16*a];
  #pragma unroll
  for(int b=0;b<4;++b)out[((row0+qr+r+16*a)*h+head)*128+dc+c+16*b]=__float2half_rn(acc[a][b]/divisor);
 }
}
inline int launch(void* ctx,const void*q,const void*k,const void*v,I sq,I sk,I h,I d,I qs,I ks,void*out,void*scratch,I bytes){
 if(!ctx||!arguments(q,k,v,sq,sk,h,d,qs,ks,out,scratch,bytes))return ns_fail("Qwen batched attention: invalid shape/span/alias/workspace");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));const I capacity=batch(sq);auto*s=static_cast<float*>(scratch);float*den=s+h*capacity*sk;
 for(I row0=0;row0<sq;row0+=capacity){const I rows=sq-row0<capacity?sq-row0:capacity;
  qk<<<dim3(unsigned((sk+63)/64),unsigned((rows+63)/64),unsigned(h)),256,0,g->q>>>(static_cast<const __half*>(q),static_cast<const __half*>(k),row0,rows,sk,d,qs,ks,capacity,s);NS_LAUNCH_CHECK();
  softmax<<<dim3(unsigned(rows),unsigned(h)),256,0,g->q>>>(s,den,sk,capacity);NS_LAUNCH_CHECK();
  pv<<<dim3(2,unsigned((rows+63)/64),unsigned(h)),256,0,g->q>>>(s,den,static_cast<const __half*>(v),row0,rows,sk,h,ks,capacity,static_cast<__half*>(out));NS_LAUNCH_CHECK();
 }
 return 0;
}
}
#endif
