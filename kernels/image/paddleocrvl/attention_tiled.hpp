// Experimental PRIVATE F32 vision attention; no production dispatch/default.
// Coalesced 16-dimension x256-key LDS tile reused by two queries (one above5120).
// Exactly72 ascending FMAs, original rsqrtf scale, 256-lane max/sum trees,
// ascending-key PV FMAs and divide-after-PV. No online softmax/global scratch.
#pragma once
#ifdef __FAST_MATH__
#error "PaddleOCR attention requires original F32 arithmetic"
#endif
#include <cstdint>
#include <initializer_list>
namespace pocr_tiled {
using I=int64_t;
inline bool supported(I n,I h,I d,I stride){return n>=1&&n<=8192&&h>=1&&h<=16&&d==72&&stride==3*h*72;}
inline I queries(I n){return n<=5120?2:1;}
inline I shared_bytes(I n){return (queries(n)*n+16*257+queries(n)*256)*4;}
inline bool arguments(const float*q,const float*k,const float*v,I n,I h,I d,I stride,float*out){
 if(!supported(n,h,d,stride)||!q||!k||!v||!out)return false;
 const uintptr_t input=uintptr_t((n-1)*stride+h*d)*4,output=uintptr_t(n*h*d)*4,o=reinterpret_cast<uintptr_t>(out);
 if(o%4||o>UINTPTR_MAX-output)return false;
 for(const float*p:{q,k,v}){const uintptr_t a=reinterpret_cast<uintptr_t>(p);if(a%4||a>UINTPTR_MAX-input||!(a+input<=o||o+output<=a))return false;}
 return true;
}
}
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <cmath>
namespace pocr_tiled {
template<int Q>
__global__ __launch_bounds__(256) void tiled(const float*q,const float*k,const float*v,I n,I h,I d,I stride,float*out){
 extern __shared__ float scores[];
 __shared__ float keys[16][257],tmp[Q][256];
 const int t=threadIdx.x;const I row=I(blockIdx.x)*Q,head=blockIdx.y;
 for(I base=0;base<n;base+=256){
  float z[Q]={};
  for(int a0=0;a0<72;a0+=16){
   for(int i=t;i<256*16;i+=256){int j=i/16,a=i%16;keys[a][j]=(base+j<n&&a0+a<72)?k[(base+j)*stride+head*72+a0+a]:0.f;}
   __syncthreads();
   #pragma unroll
   for(int a=0;a<16;++a)if(a0+a<72){
    const float kv=keys[a][t];
    #pragma unroll
    for(int r=0;r<Q;++r)if(row+r<n)z[r]=fmaf(q[(row+r)*stride+head*72+a0+a],kv,z[r]);
   }
   __syncthreads();
  }
  if(base+t<n){
   #pragma unroll
   for(int r=0;r<Q;++r)scores[r*n+base+t]=z[r]*rsqrtf(float(d));
  }
 }
 float best[Q];
 #pragma unroll
 for(int r=0;r<Q;++r)best[r]=-INFINITY;
 for(I j=t;j<n;j+=256){
  #pragma unroll
  for(int r=0;r<Q;++r)best[r]=fmaxf(best[r],scores[r*n+j]);
 }
 #pragma unroll
 for(int r=0;r<Q;++r)tmp[r][t]=best[r];
 __syncthreads();
 for(int step=128;step;step/=2){if(t<step){
  #pragma unroll
  for(int r=0;r<Q;++r)tmp[r][t]=fmaxf(tmp[r][t],tmp[r][t+step]);
 }__syncthreads();}
 #pragma unroll
 for(int r=0;r<Q;++r)best[r]=tmp[r][0];
 __syncthreads();
 float sum[Q]={};
 for(I j=t;j<n;j+=256){
  #pragma unroll
  for(int r=0;r<Q;++r){float z=expf(scores[r*n+j]-best[r]);scores[r*n+j]=z;sum[r]+=z;}
 }
 #pragma unroll
 for(int r=0;r<Q;++r)tmp[r][t]=sum[r];
 __syncthreads();
 for(int step=128;step;step/=2){if(t<step){
  #pragma unroll
  for(int r=0;r<Q;++r)tmp[r][t]+=tmp[r][t+step];
 }__syncthreads();}
 const int r=t/72,a=t%72;
 if(r<Q&&row+r<n){float z=0;for(I j=0;j<n;++j)z=fmaf(scores[r*n+j],v[j*stride+head*72+a],z);out[((row+r)*h+head)*72+a]=z/tmp[r][0];}
}
inline hipError_t launch(const float*q,const float*k,const float*v,I n,I h,I d,I stride,float*out,hipStream_t stream){
 if(!arguments(q,k,v,n,h,d,stride,out))return hipErrorInvalidValue;
 if(n<=5120)tiled<2><<<dim3(unsigned((n+1)/2),unsigned(h)),256,size_t(2*n)*4,stream>>>(q,k,v,n,h,d,stride,out);
 else tiled<1><<<dim3(unsigned(n),unsigned(h)),256,size_t(n)*4,stream>>>(q,k,v,n,h,d,stride,out);
 return hipGetLastError();
}
}
#endif
