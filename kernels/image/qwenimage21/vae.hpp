// Private single-frame VAE kernels, original nextsycl QwenImage21 semantics.
#pragma once
#include "vae.h"
#include <cmath>
#include <cstdint>
namespace nrqi_vae {
using I=int64_t;
struct Span{uintptr_t a,b;};
inline bool span(const void*p,I n,unsigned width,Span&s){uintptr_t a=reinterpret_cast<uintptr_t>(p);if(!p||n<1||a%width||uint64_t(n)>UINTPTR_MAX/width||uint64_t(n)*width>UINTPTR_MAX-a)return false;s={a,a+uintptr_t(n)*width};return true;}
inline bool disjoint(Span a,Span b){return a.b<=b.a||b.b<=a.a;}
inline bool image(I h,I w,I c){return h>0&&h<=4096&&w>0&&w<=4096&&c>0&&c<=4096&&h*w*c<=INT32_MAX;}
inline bool pair(const void*x,I nx,void*y,I ny,unsigned bx=2,unsigned by=2){Span a,b;return span(x,nx,bx,a)&&span(y,ny,by,b)&&disjoint(a,b);}
inline bool conv_valid(void*g,const void*x,I h,I w,I ci,const void*weight,I co,int k,int stride,const float*bias,void*out){
 if(!g||!image(h,w,ci)||co<1||co>4096||(k!=1&&k!=3)||(stride!=1&&stride!=2)||(stride==2&&(h<k||w<k)))return false;
 const I oh=stride==1?h:(h-k)/2+1,ow=stride==1?w:(w-k)/2+1;
 if(!image(oh,ow,co))return false;
 Span a,b,c,d;return span(x,h*w*ci,2,a)&&span(weight,co*ci*k*k,2,b)&&span(out,oh*ow*co,2,c)&&disjoint(a,c)&&disjoint(b,c)&&(!bias||(span(bias,co,4,d)&&disjoint(d,c)));
}
}
#ifdef __HIPCC__
#ifdef __FAST_MATH__
#error "VAE requires original rounding boundaries; no fast math"
#endif
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
namespace nrqi_vae {
__global__ void conv(const __half*x,I h,I w,I ci,const __half*weight,I co,int k,int stride,const float*bias,__half*out,I oh,I ow){
 __shared__ float a[16][16],b[16][16];int tx=threadIdx.x,ty=threadIdx.y;I p=I(blockIdx.x)*16+ty,c=I(blockIdx.y)*16+tx;float sum=0;I reduction=ci*k*k;int pad=stride==1?k/2:0;
 for(I base=0;base<reduction;base+=16){I j=base+tx,ch=j/(k*k),iy=(p/ow)*stride-pad+(j/k)%k,ix=(p%ow)*stride-pad+j%k;
  a[ty][tx]=(p<oh*ow&&j<reduction&&iy>=0&&iy<h&&ix>=0&&ix<w)?__half2float(x[(iy*w+ix)*ci+ch]):0;
  I oc=I(blockIdx.y)*16+ty;b[ty][tx]=(oc<co&&j<reduction)?__half2float(weight[oc*reduction+j]):0;
  __syncthreads();for(int q=0;q<16;++q)sum=fmaf(a[ty][q],b[tx][q],sum);__syncthreads();
 }
 if(p<oh*ow&&c<co)out[p*co+c]=__float2half_rn(__fadd_rn(sum,bias?bias[c]:0.f));
}
__global__ void silu(__half*x,I n){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n){float v=__half2float(x[i]);x[i]=__float2half_rn(v/(1.f+expf(-v)));}}
__global__ void softmax(const float*x,I n,float scale,__half*out){
 __shared__ float tmp[256];int t=threadIdx.x;I r=blockIdx.x;float v=-INFINITY;
 for(I j=t;j<n;j+=256) { v=fmaxf(v,__fmul_rn(x[r*n+j],scale)); }
 tmp[t]=v;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)tmp[t]=fmaxf(tmp[t],tmp[t+s]);__syncthreads();}float mx=tmp[0];__syncthreads();v=0;
 for(I j=t;j<n;j+=256) { v=__fadd_rn(v,expf(__fsub_rn(__fmul_rn(x[r*n+j],scale),mx))); }
 tmp[t]=v;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)tmp[t]=__fadd_rn(tmp[t],tmp[t+s]);__syncthreads();}float inv=1.f/tmp[0];
 for(I j=t;j<n;j+=256)out[r*n+j]=__float2half_rn(__fmul_rn(expf(__fsub_rn(__fmul_rn(x[r*n+j],scale),mx)),inv));
}
__global__ void spatial(const __half*x,I h,I w,I ci,I co,int ft,__half*out,int op,I n){
 I i=I(blockIdx.x)*256+threadIdx.x;if(i>=n)return;
 if(op==0){I c=i%ci,xx=(i/ci)%(2*w),y=i/(ci*2*w);out[i]=x[((y/2)*w+xx/2)*ci+c];}
 else if(op==1){I c=i%co,xx=(i/co)%(2*w),y=i/(co*2*w),f=ft*4,r=co*f/ci,ic=(c*f+(ft-1)*4+(y%2)*2+xx%2)/r;out[i]=__float2half_rn(__fadd_rn(__half2float(out[i]),__half2float(x[((y/2)*w+xx/2)*ci+ic])));}
 else if(op==2){I c=i%ci,xx=(i/ci)%(w+1),y=i/(ci*(w+1));out[i]=(y<h&&xx<w)?x[(y*w+xx)*ci+c]:__float2half_rn(0.f);}
 else{I c=i%co,xx=(i/co)%(w/2),y=i/(co*(w/2)),f=ft*4,group=ci*f/co;float sum=0;
  for(I gi=0;gi<group;++gi){I j=c*group+gi,ch=j/f,r=j%f,t=r/4,hs=(r%4)/2,ws=r%2;if(ft==2&&t==0)continue;sum=__fadd_rn(sum,__half2float(x[((2*y+hs)*w+2*xx+ws)*ci+ch]));}
  out[i]=__float2half_rn(__fadd_rn(__half2float(out[i]),sum/float(group)));
 }
}
__global__ void rgba(const __half*x,I n,uint8_t*out){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n){float v=__fadd_rn(__fmul_rn(fminf(1.f,fmaxf(-1.f,__half2float(x[i]))),.5f),.5f);out[i]=uint8_t(fminf(255.f,fmaxf(0.f,roundf(__fmul_rn(v,255.f)))));}}
inline int conv_launch(void*ctx,const void*x,I h,I w,I ci,const void*wt,I co,int k,int stride,const float*bias,void*out){
 if(!conv_valid(ctx,x,h,w,ci,wt,co,k,stride,bias,out))return ns_fail("VAE conv: invalid geometry/span");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));I oh=stride==1?h:(h-k)/2+1,ow=stride==1?w:(w-k)/2+1;
 conv<<<dim3(unsigned((oh*ow+15)/16),unsigned((co+15)/16)),dim3(16,16),0,g->q>>>(static_cast<const __half*>(x),h,w,ci,static_cast<const __half*>(wt),co,k,stride,bias,static_cast<__half*>(out),oh,ow);NS_LAUNCH_CHECK();return 0;
}
inline int spatial_launch(void*ctx,const void*x,I h,I w,I ci,I co,int ft,void*out,int op){
 if(!ctx||!image(h,w,ci)||co<1||co>4096||(ft!=1&&ft!=2))return ns_fail("VAE spatial: invalid geometry");
 I n=0;
 if(op==0||op==1){if(!image(h*2,w*2,co)||(co*4*ft)%ci)return ns_fail("VAE up: invalid shape");n=h*w*4*co;}
 else if(op==2){if(!image(h+1,w+1,ci))return ns_fail("VAE pad: invalid shape");n=(h+1)*(w+1)*ci;}
 else{if(h%2||w%2||h<2||w<2||(ci*4*ft)%co||!image(h/2,w/2,co))return ns_fail("VAE down: invalid shape");n=(h/2)*(w/2)*co;}
 if(!pair(x,h*w*ci,out,n))return ns_fail("VAE spatial: overlapping/invalid spans");
 auto*g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));spatial<<<dim3(unsigned((n+255)/256)),256,0,g->q>>>(static_cast<const __half*>(x),h,w,ci,co,ft,static_cast<__half*>(out),op,n);NS_LAUNCH_CHECK();return 0;
}
}
#endif
