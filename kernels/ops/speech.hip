// Baseline speech operations. Rust owns the model pass; all arithmetic executes on its HIP stream.
#include "../ns/ns.h"
#include "../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>
using I = int64_t;
__device__ float rd(const void* p, int dt, I i) { return dt == 1 ? __half2float(((const __half*)p)[i]) : ((const float*)p)[i]; }
__device__ void wr(void* p, int dt, I i, float x) { if(dt==1)((__half*)p)[i]=__float2half_rn(x); else ((float*)p)[i]=x; }
extern "C" const char* nsd_last_error() { return ns_last_error(); }
extern "C" void* nsd_create(void* g) { return g; }
extern "C" void nsd_destroy(void*) {}
extern "C" int nsd_wait(void* g) { return ns_sync((ns_gpu*)g); }
#define START auto* g=(ns_gpu*)ctx; NS_CU(hipSetDevice(g->dev))
#define DONE NS_LAUNCH_CHECK(); return 0

// out[M,N] = X[M,K] W[N,K]^T. Float accumulation; no tensor-core assumption.
__global__ void linear_kernel(const void* x,const void* w,int dt,I m,I k,I n,const float* bias,void* out,int odt){
 __shared__ float a[16][16],b[16][16];
 int tx=threadIdx.x,ty=threadIdx.y;I r=blockIdx.y*16+ty,c=blockIdx.x*16+tx;float sum=0;
 for(I base=0;base<k;base+=16){
  a[ty][tx]=(r<m&&base+tx<k)?rd(x,dt,r*k+base+tx):0;
  b[ty][tx]=(blockIdx.x*16+ty<n&&base+tx<k)?rd(w,dt,(blockIdx.x*16+ty)*k+base+tx):0;
  __syncthreads();
  #pragma unroll
  for(int j=0;j<16;++j)sum=fmaf(a[ty][j],b[tx][j],sum);
  __syncthreads();
 }
 if(r<m&&c<n)wr(out,odt,r*n+c,sum+(bias?bias[c]:0));
}
extern "C" int nsd_linear(void* ctx,const void* x,int dt,I m,I k,const void* w,I n,const float* bias,void* out,int odt){
 START; if(dt>1||odt>1||m<1||k<1||n<1)return ns_fail("linear: invalid shape or dtype");
 hipLaunchKernelGGL(linear_kernel,dim3((n+15)/16,(m+15)/16),dim3(16,16),0,g->q,x,w,dt,m,k,n,bias,out,odt);DONE;
}
__global__ void norm_kernel(const void* x,int dt,I c,const float* w,const float* b,float eps,void* out,int odt){
 __shared__ float sums[256];int t=threadIdx.x;I r=blockIdx.x;float sum=0;
 for(I j=t;j<c;j+=256)sum+=rd(x,dt,r*c+j);sums[t]=sum;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)sums[t]+=sums[t+s];__syncthreads();}float mean=sums[0]/c;__syncthreads();
 sum=0;for(I j=t;j<c;j+=256){float z=rd(x,dt,r*c+j)-mean;sum+=z*z;}sums[t]=sum;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)sums[t]+=sums[t+s];__syncthreads();}float inv=rsqrtf(sums[0]/c+eps);
 for(I j=t;j<c;j+=256)wr(out,odt,r*c+j,(rd(x,dt,r*c+j)-mean)*inv*(w?w[j]:1)+(b?b[j]:0));
}
extern "C" int nsd_layer_norm(void* ctx,const void* x,int dt,I m,I c,const float* w,const float* b,float eps,void* out,int odt){
 START;hipLaunchKernelGGL(norm_kernel,dim3(m),dim3(256),0,g->q,x,dt,c,w,b,eps,out,odt);DONE;
}
__global__ void gelu_kernel(void* x,int dt,I n,int approx){I i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float z=rd(x,dt,i);float y=approx?0.5f*z*(1+tanhf(0.7978845608f*(z+0.044715f*z*z*z))):0.5f*z*(1+erff(z*0.7071067812f));wr(x,dt,i,y);}}
extern "C" int nsd_gelu(void* ctx,void* x,int dt,I n,int approx){START;hipLaunchKernelGGL(gelu_kernel,dim3((n+255)/256),dim3(256),0,g->q,x,dt,n,approx);DONE;}
__global__ void add_kernel(void* x,int dt,const void* y,int ydt,I n,I c,const int32_t* rows,const float* gate){I i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)wr(x,dt,i,rd(x,dt,i)+rd(y,ydt,i)*(gate?gate[(rows?rows[i/c]:0)*c+i%c]:1));}
extern "C" int nsd_gate_add(void* ctx,void* x,int dt,I m,I c,const void* y,int ydt,const int* rows,const float* gate){START;hipLaunchKernelGGL(add_kernel,dim3((m*c+255)/256),dim3(256),0,g->q,x,dt,y,ydt,m*c,c,rows,gate);DONE;}
// Stable softmax per query/head. Only scores for one row reside in shared memory.
__global__ void att_kernel(const void* q,const void* k,const void* v,int dt,I row0,I sk,I hq,I hkv,I d,I qs,I ks,void* out,int odt,bool causal){
 extern __shared__ float score[];__shared__ float tmp[256];int t=threadIdx.x;I r=blockIdx.x,h=blockIdx.y,qr=r+row0,kh=h/(hq/hkv);I end=causal?qr+1:sk;
 float best=-INFINITY;
 for(I j=t;j<end;j+=256){float z=0;for(I a=0;a<d;++a)z=fmaf(rd(q,dt,qr*qs+h*d+a),rd(k,dt,j*ks+kh*d+a),z);z*=rsqrtf((float)d);score[j]=z;best=fmaxf(best,z);}tmp[t]=best;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)tmp[t]=fmaxf(tmp[t],tmp[t+s]);__syncthreads();}best=tmp[0];__syncthreads();float sum=0;
 for(I j=t;j<end;j+=256){float z=expf(score[j]-best);score[j]=z;sum+=z;}tmp[t]=sum;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}float den=tmp[0];
 for(I a=t;a<d;a+=256){float z=0;for(I j=0;j<end;++j)z=fmaf(score[j],rd(v,dt,j*ks+kh*d+a),z);wr(out,odt,(r*hq+h)*d+a,z/den);}
}
static int attention(void* ctx,const void* q,const void* k,const void* v,int dt,I row0,I rows,I sk,I hq,I hkv,I d,I qs,I ks,void* out,int odt,bool causal){
 START;if(sk>4096||sk<1||d<1||hq<1||hkv<1||hq%hkv)return ns_fail("attention shape outside speech baseline");
 hipLaunchKernelGGL(att_kernel,dim3(rows,hq),dim3(256),sk*sizeof(float),g->q,q,k,v,dt,row0,sk,hq,hkv,d,qs,ks,out,odt,causal);DONE;
}
extern "C" int nsd_attention(void* ctx,const void* q,const void* k,const void* v,int dt,I s,I h,I d,I stride,void* out,int odt){return attention(ctx,q,k,v,dt,0,s,s,h,h,d,stride,stride,out,odt,false);}
extern "C" int nsd_attention_qk(void* ctx,const void* q,I sq,I qs,const void* k,const void* v,I sk,I ks,int dt,I h,I d,void* out,int odt){return attention(ctx,q,k,v,dt,0,sq,sk,h,h,d,qs,ks,out,odt,false);}
extern "C" int nsd_attention_causal_rows(void* ctx,const void* q,const void* k,const void* v,int dt,I row0,I rows,I hq,I hkv,I d,I qs,I ks,void* out){return attention(ctx,q,k,v,dt,row0,rows,row0+rows,hq,hkv,d,qs,ks,out,dt,true);}
__global__ void conv_kernel(const float* x,I ci,I l,const float* w,I co,I width,const float* bias,I stride,I dil,I pad,float* out,I lo,I count){
 I i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;I pos=i%lo,c=(i/lo)%co,b=i/(lo*co);float sum=0;
 for(I ch=0;ch<ci;++ch)for(I j=0;j<width;++j){I at=pos*stride-pad+j*dil;if(at>=0&&at<l)sum=fmaf(x[(b*ci+ch)*l+at],w[(c*ci+ch)*width+j],sum);}out[i]=sum+(bias?bias[c]:0);
}
extern "C" int nsd_conv1d(void* ctx,const float* x,I b,I ci,I l,const float* w,I co,I width,const float* bias,I stride,I dil,I pad,float* out,I lo){START;I count=b*co*lo;hipLaunchKernelGGL(conv_kernel,dim3((count+255)/256),dim3(256),0,g->q,x,ci,l,w,co,width,bias,stride,dil,pad,out,lo,count);DONE;}
__global__ void rms_kernel(const void* x,int dt,I c,const float* w,float eps,const int32_t* rows,const float* scale,const float* shift,void* out,int odt){
 __shared__ float sum[256];int t=threadIdx.x;I r=blockIdx.x;float z=0;
 for(I j=t;j<c;j+=256){float v=rd(x,dt,r*c+j);z+=v*v;}sum[t]=z;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)sum[t]+=sum[t+s];__syncthreads();}float inv=rsqrtf(sum[0]/c+eps);
 for(I j=t;j<c;j+=256){float v=rd(x,dt,r*c+j)*inv*(w?w[j]:1);if(rows&&scale&&shift&&rows[r]>=0){I mr=rows[r];v=v*(1+scale[mr*c+j])+shift[mr*c+j];}wr(out,odt,r*c+j,v);}
}
extern "C" int nsd_rms_norm_mod(void* ctx,const void* x,int dt,I m,I c,const float* w,float eps,const int* rows,const float* scale,const float* shift,void* out,int odt){START;hipLaunchKernelGGL(rms_kernel,dim3(m),dim3(256),0,g->q,x,dt,c,w,eps,rows,scale,shift,out,odt);DONE;}
__global__ void swiglu_kernel(const void* x,int dt,I c,I count,void* out,int odt){I i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){I row=i/c,col=i%c;float z=rd(x,dt,row*2*c+col);wr(out,odt,i,z/(1+expf(-z))*rd(x,dt,row*2*c+c+col));}}
extern "C" int nsd_swiglu(void* ctx,const void* x,int dt,I m,I c,void* out,int odt){START;hipLaunchKernelGGL(swiglu_kernel,dim3((m*c+255)/256),dim3(256),0,g->q,x,dt,c,m*c,out,odt);DONE;}
