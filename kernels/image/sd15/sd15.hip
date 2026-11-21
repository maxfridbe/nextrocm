#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include "sd15.h"
#include <hip/hip_fp16.h>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <utility>

#ifndef NR_SD15_DIAGNOSTIC_ATTN_CAPTURE
#define NR_SD15_DIAGNOSTIC_ATTN_CAPTURE 0
#endif
#if NR_SD15_DIAGNOSTIC_ATTN_CAPTURE != 0 && NR_SD15_DIAGNOSTIC_ATTN_CAPTURE != 1
#error "NR_SD15_DIAGNOSTIC_ATTN_CAPTURE must be 0 or 1"
#endif
#if NR_SD15_DIAGNOSTIC_ATTN_CAPTURE
#include "attention_capture_diagnostic.hpp"
#endif

namespace {
using I = int64_t;
// Limit indexing to signed 64-bit byte spans and portable HIP launch dimensions.
bool shape(std::initializer_list<I> dims) {
 I p=1;
 for (I d:dims) { if(d<=0 || p>INT64_MAX/4/d) return false; p*=d; }
 return true;
}
bool wt(nrsd_weight w) { return w.dtype==0 || w.dtype==1; }
bool grid(I n) { return n>0 && n<=2147483647; }
unsigned blocks(I n) { return unsigned((n+255)/256); }
__device__ float rd(nrsd_weight w,I i) {
 return w.dtype==1 ? __half2float(static_cast<const __half*>(w.data)[i]) : static_cast<const float*>(w.data)[i];
}
__device__ float bias(nrsd_weight b,I i) { return b.data?rd(b,i):0.f; }

// Tile rows=tokens/spatial positions, columns=output channels. Shared tiles
// expand only 16x16 values at a time: implicit im2col never reaches global memory.
__global__ void linear_k(const float* x,nrsd_weight w,nrsd_weight b,float* y,nrsd_linear_shape s) {
 __shared__ float a[16][16],v[16][16];
 int tx=threadIdx.x,ty=threadIdx.y;
 I row=I(blockIdx.y)*16+ty,col=I(blockIdx.x)*16+tx;
 float z=0;
 for(I base=0;base<s.input;base+=16) {
  a[ty][tx]=(row<s.rows&&base+tx<s.input)?x[row*s.x_stride+base+tx]:0;
  I wc=I(blockIdx.x)*16+ty;
  v[ty][tx]=(wc<s.output&&base+tx<s.input)?rd(w,wc*s.input+base+tx):0;
  __syncthreads();
  for(int j=0;j<16;++j) z=fmaf(a[ty][j],v[tx][j],z);
  __syncthreads();
 }
 if(row<s.rows&&col<s.output)y[row*s.out_stride+col]=z+bias(b,col);
}
__global__ void conv_k(const float* x,nrsd_weight w,nrsd_weight b,float* y,nrsd_conv_shape s) {
 __shared__ float a[16][16],v[16][16];
 int tx=threadIdx.x,ty=threadIdx.y;
 I p=I(blockIdx.x)*16+tx,co=I(blockIdx.y)*16+ty,nb=blockIdx.z;
 I reduction=s.ci*s.kh*s.kw;
 float z=0;
 for(I base=0;base<reduction;base+=16) {
  I r=base+ty,ic=r/(s.kh*s.kw),iy=(p/s.ow)*s.sh-s.ph+(r/s.kw)%s.kh,ix=(p%s.ow)*s.sw-s.pw+r%s.kw;
  a[ty][tx]=(p<s.oh*s.ow && r<reduction && iy>=0 && iy<s.height && ix>=0 && ix<s.width)?x[((nb*s.ci+ic)*s.height+iy)*s.width+ix]:0;
  v[ty][tx]=(co<s.co&&base+tx<reduction)?rd(w,co*reduction+base+tx):0;
  __syncthreads();
  for(int j=0;j<16;++j)z=fmaf(v[ty][j],a[j][tx],z);
  __syncthreads();
 }
 if(p<s.oh*s.ow&&co<s.co)y[(nb*s.co+co)*s.oh*s.ow+p]=z+bias(b,co);
}
#include "matmul_fast.hpp"

// Two-pass centered variance avoids E[x^2]-E[x]^2 cancellation.
__global__ void norm_k(const float* x,nrsd_weight w,nrsd_weight b,float* y,I count,I channels_per_group,I spatial,I groups,float eps,bool ln) {
 __shared__ float tmp[256];
 int t=threadIdx.x; I row=blockIdx.x,offset=row*count;
 float z=0;
 for(I i=t;i<count;i+=256)z+=x[offset+i];
 tmp[t]=z;__syncthreads();
 for(int j=128;j;j/=2){if(t<j)tmp[t]+=tmp[t+j];__syncthreads();}
 float mean=tmp[0]/float(count);__syncthreads();z=0;
 for(I i=t;i<count;i+=256){float d=x[offset+i]-mean;z=fmaf(d,d,z);}
 tmp[t]=z;__syncthreads();
 for(int j=128;j;j/=2){if(t<j)tmp[t]+=tmp[t+j];__syncthreads();}
 float inv=rsqrtf(tmp[0]/float(count)+eps);
 for(I i=t;i<count;i+=256){I c=ln?i:(row%groups)*channels_per_group+i/spatial;y[offset+i]=(x[offset+i]-mean)*inv*(w.data?rd(w,c):1.f)+bias(b,c);}
}
// Private diagnostic build only. Never enabled by the production build script.
#ifndef NR_SD15_DIAGNOSTIC_GN_F64_STATS
#define NR_SD15_DIAGNOSTIC_GN_F64_STATS 0
#endif
#if NR_SD15_DIAGNOSTIC_GN_F64_STATS != 0 && NR_SD15_DIAGNOSTIC_GN_F64_STATS != 1
#error "NR_SD15_DIAGNOSTIC_GN_F64_STATS must be 0 or 1"
#endif
#if NR_SD15_DIAGNOSTIC_GN_F64_STATS
#include "gn_f64_diagnostic.hpp"
#endif
__device__ float act(float x,int kind) {
 if(kind==0)return x/(1.f+expf(-x));
 if(kind==2)return x/(1.f+expf(-1.702f*x));
 if(kind==3)return 0.5f*x*(1.f+tanhf(0.7978845608028654f*(x+0.044715f*x*x*x)));
 return 0.5f*x*(1.f+erff(x*0.7071067811865475f));
}
__global__ void act_k(const float* x,float* y,I n,int kind) {I i=I(blockIdx.x)*256+threadIdx.x;if(i<n)y[i]=act(x[i],kind);}
__global__ void add_k(const float* x,const float* b,float* y,I n,float alpha,float beta,bool multiply) {
 I i=I(blockIdx.x)*256+threadIdx.x;if(i<n)y[i]=multiply?x[i]*b[i]:alpha*x[i]+(b?beta*b[i]:0.f);
}
__global__ void geglu_k(const float* x,float* y,I n,I c){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n){I j=(i/c)*2*c+i%c;y[i]=x[j]*act(x[j+c],3);}}
__global__ void time_k(const float* x,const float* e,float* y,I n,I spatial){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n)y[i]=x[i]+e[i/spatial];}
__global__ void layout_k(const float* x,float* y,I n,I c,I p,bool to_nhwc){
 I i=I(blockIdx.x)*256+threadIdx.x;if(i<n){I s=i%p,ch=(i/p)%c,b=i/(p*c),j=(b*p+s)*c+ch;if(to_nhwc)y[j]=x[i];else y[i]=x[j];}
}
__global__ void up_k(const float* x,float* y,I n,I h,I w){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n){I ox=i%(w*2),oy=(i/(w*2))%(h*2),bc=i/(4*h*w);y[i]=x[(bc*h+oy/2)*w+ox/2];}}
__global__ void concat_k(const float* a,const float* b,float* y,I n,I ca,I cb,I p){I i=I(blockIdx.x)*256+threadIdx.x;if(i<n){I nb=i/((ca+cb)*p),c=(i/p)%(ca+cb),s=i%p;y[i]=c<ca?a[(nb*ca+c)*p+s]:b[(nb*cb+c-ca)*p+s];}}

// One CTA per query/head. A fixed 256-key tile plus 256-value accumulator
// bounds shared memory at 2 KiB, irrespective of image size. Online log-sum-exp
// rescales prior numerator/denominator at each tile; never materializes QK^T.
// Each thread computes a key dot product, then a component of the weighted V.
__global__ void attention_k(const float* q,const float* k,const float* v,float* out,nrsd_attention_shape s){
 __shared__ float scores[256],tmp[256];
 __shared__ float old_scale,tile_max,den;
 int t=threadIdx.x;I qr=blockIdx.x,h=blockIdx.y,nb=blockIdx.z;
 I qo=nb*s.qb+qr*s.qt+h*s.qh,ko=nb*s.kb+h*s.kh,vo=nb*s.vb+h*s.vh;
 I end=s.causal?(qr+1<s.keys?qr+1:s.keys):s.keys;
 // Supported head dimensions <= 512: two components per lane (VAE head 512).
 float accum0=0,accum1=0;
 if(t==0){tile_max=-INFINITY;den=0;}
 __syncthreads();
 for(I base=0;base<end;base+=256){
  I key=base+t;float score=-INFINITY;
  if(key<end){score=0;for(I d=0;d<s.dim;++d)score=fmaf(q[qo+d],k[ko+key*s.kt+d],score);score*=s.scale;}
  scores[t]=score;tmp[t]=score;__syncthreads();
  for(int j=128;j;j/=2){if(t<j)tmp[t]=fmaxf(tmp[t],tmp[t+j]);__syncthreads();}
  if(t==0){float next=fmaxf(tile_max,tmp[0]);old_scale=expf(tile_max-next);tile_max=next;}
  __syncthreads();
  float prob=key<end?expf(scores[t]-tile_max):0; scores[t]=prob;tmp[t]=prob;__syncthreads();
  for(int j=128;j;j/=2){if(t<j)tmp[t]+=tmp[t+j];__syncthreads();}
  if(t==0)den=den*old_scale+tmp[0];
  accum0*=old_scale;accum1*=old_scale;
  I count=end-base<256?end-base:256;
  for(I j=0;j<count;++j){if(t<s.dim)accum0=fmaf(scores[j],v[vo+(base+j)*s.vt+t],accum0);if(t+256<s.dim)accum1=fmaf(scores[j],v[vo+(base+j)*s.vt+t+256],accum1);}
  __syncthreads();
 }
 I dst=nb*s.ob+qr*s.ot+h*s.oh;
 if(t<s.dim)out[dst+t]=accum0/den;
 if(t+256<s.dim)out[dst+t+256]=accum1/den;
}
// Four queries share each coalesced key tile and value load. Arithmetic within
// each output retains the baseline dot order and 256-key online-softmax tree.
template<int D>
__global__ void attention_coalesced_k(const float* q,const float* k,const float* v,float* out,nrsd_attention_shape s){
 constexpr int Q=4;
 const I dim=D?D:s.dim;
 __shared__ float keys[16][257],scores[Q][256],tmp[Q][256];
 __shared__ float old_scale[Q],tile_max[Q],den[Q];
 int t=threadIdx.x;I qr=I(blockIdx.x)*Q,h=blockIdx.y,nb=blockIdx.z;
 I ko=nb*s.kb+h*s.kh,vo=nb*s.vb+h*s.vh;
 I end=s.causal?(qr+Q<s.keys?qr+Q:s.keys):s.keys;
 float accum0[Q]={},accum1[Q]={};
 float packed_accum=0;
 if(t<Q){tile_max[t]=-INFINITY;den[t]=0;}
 __syncthreads();
 for(I base=0;base<end;base+=256){
  I key=base+t;float score[Q]={};
  for(I d0=0;d0<dim;d0+=16){
   for(int i=t;i<256*16;i+=256){
    int kr=i/16,dc=i%16;I ki=base+kr,di=d0+dc;
    keys[dc][kr]=(ki<end&&di<dim)?k[ko+ki*s.kt+di]:0.f;
   }
   __syncthreads();
   #pragma unroll
   for(int dc=0;dc<16&&d0+dc<dim;++dc){
    float kv=keys[dc][t];
    #pragma unroll
    for(int r=0;r<Q;++r)if(qr+r<s.queries)score[r]=fmaf(q[nb*s.qb+(qr+r)*s.qt+h*s.qh+d0+dc],kv,score[r]);
   }
   __syncthreads();
  }
  #pragma unroll
  for(int r=0;r<Q;++r){float z=(key<s.keys&&(!s.causal||key<=qr+r))?score[r]*s.scale:-INFINITY;scores[r][t]=z;tmp[r][t]=z;}
  __syncthreads();
  for(int j=128;j;j/=2){
   #pragma unroll
   for(int r=0;r<Q;++r)if(t<j)tmp[r][t]=fmaxf(tmp[r][t],tmp[r][t+j]);
   __syncthreads();
  }
  if(t<Q){float next=fmaxf(tile_max[t],tmp[t][0]);old_scale[t]=expf(tile_max[t]-next);tile_max[t]=next;}
  __syncthreads();
  #pragma unroll
  for(int r=0;r<Q;++r){float prob=(key<s.keys&&(!s.causal||key<=qr+r))?expf(scores[r][t]-tile_max[r]):0;scores[r][t]=prob;tmp[r][t]=prob;}
  __syncthreads();
  for(int j=128;j;j/=2){
   #pragma unroll
   for(int r=0;r<Q;++r)if(t<j)tmp[r][t]+=tmp[r][t+j];
   __syncthreads();
  }
  if(t<Q)den[t]=den[t]*old_scale[t]+tmp[t][0];
  if constexpr(D>0&&D<=64){
   // Map the four queries across lanes during PV rather than leaving most
   // lanes idle at small head dimensions. Each output retains key order.
   if(t<Q*D){
    int r=t/D,d=t%D;packed_accum*=old_scale[r];
    I count=end-base<256?end-base:256;
    for(I j=0;j<count;++j)if(!s.causal||base+j<=qr+r)
     packed_accum=fmaf(scores[r][j],v[vo+(base+j)*s.vt+d],packed_accum);
   }
  }else{
  #pragma unroll
  for(int r=0;r<Q;++r){accum0[r]*=old_scale[r];accum1[r]*=old_scale[r];}
  I count=end-base<256?end-base:256;
  for(I j=0;j<count;++j){
   float v0=t<dim?v[vo+(base+j)*s.vt+t]:0;
   float v1=t+256<dim?v[vo+(base+j)*s.vt+t+256]:0;
   #pragma unroll
   for(int r=0;r<Q;++r){
    // Causal rows must skip keys beyond their own end, as the baseline does.
    if(!s.causal||base+j<=qr+r){
     if(t<dim)accum0[r]=fmaf(scores[r][j],v0,accum0[r]);
     if(t+256<dim)accum1[r]=fmaf(scores[r][j],v1,accum1[r]);
    }
   }
  }
  }
  __syncthreads();
 }
 if constexpr(D>0&&D<=64){
  int r=t/D,d=t%D;
  if(t<Q*D&&qr+r<s.queries)out[nb*s.ob+(qr+r)*s.ot+h*s.oh+d]=packed_accum/den[r];
 }else{
 #pragma unroll
 for(int r=0;r<Q;++r)if(qr+r<s.queries){
  I dst=nb*s.ob+(qr+r)*s.ot+h*s.oh;
  if(t<dim)out[dst+t]=accum0[r]/den[r];
  if(t+256<dim)out[dst+t+256]=accum1[r]/den[r];
 }
 }
}

// Explicit compile-time arithmetic experiment; production build leaves this off.
#ifndef NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P
#define NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P 0
#endif
#if NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P != 0 && NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P != 1
#error "NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P must be 0 or 1"
#endif
#if NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P
#include "attention_normalized_diagnostic.hpp"
#endif

bool span(I b,I t,I h,I d,I bs,I ts,I hs){
 if(!shape({b,t,h,d})||bs<=0||ts<=0||hs<=0)return false;
 I total=d;
 for(auto p:{std::pair<I,I>{b-1,bs},{t-1,ts},{h-1,hs}}){if(p.first && p.second>(INT64_MAX/4-total)/p.first)return false;total+=p.first*p.second;}
 // Either token-major or head-major, allowing padding/interleaving, no alias.
 bool token=hs>=d && h<=INT64_MAX/hs && ts>=h*hs && t<=INT64_MAX/ts && (b==1||bs>=t*ts);
 bool head=ts>=d && t<=INT64_MAX/ts && hs>=t*ts && h<=INT64_MAX/hs && (b==1||bs>=h*hs);
 return token||head;
}
}
#define START if(!g)return ns_fail("nrsd: null GPU"); NS_CU(hipSetDevice(g->dev))
#define DONE NS_LAUNCH_CHECK();return 0
#define REQUIRE(test,msg) if(!(test))return ns_fail("nrsd: " msg)
extern "C" int nrsd_linear(ns_gpu* g,const float* x,nrsd_weight w,nrsd_weight b,float* y,nrsd_linear_shape s){
 REQUIRE(x&&y&&w.data&&wt(w)&&wt(b)&&shape({s.rows,s.x_stride})&&shape({s.rows,s.out_stride})&&shape({s.output,s.input})&&s.x_stride>=s.input&&s.out_stride>=s.output&&(s.rows+15)/16<=65535&&grid((s.output+15)/16),"invalid linear");START;
 if(s.rows>=16&&s.output>=16){hipLaunchKernelGGL(linear_fast_k,dim3((s.output+63)/64,(s.rows+63)/64),dim3(16,16),0,g->q,x,w,b,y,s);}else{hipLaunchKernelGGL(linear_k,dim3((s.output+15)/16,(s.rows+15)/16),dim3(16,16),0,g->q,x,w,b,y,s);}DONE;
}
extern "C" int nrsd_conv2d(ns_gpu* g,const float* x,nrsd_weight w,nrsd_weight b,float* y,nrsd_conv_shape s){
 REQUIRE(x&&y&&w.data&&wt(w)&&wt(b)&&shape({s.batch,s.ci,s.height,s.width})&&shape({s.batch,s.co,s.oh,s.ow})&&shape({s.co,s.ci,s.kh,s.kw})&&(s.kh==1||s.kh==3)&&(s.kw==1||s.kw==3)&&s.sh>0&&s.sw>0&&s.sh<=INT32_MAX&&s.sw<=INT32_MAX&&s.height<=INT32_MAX&&s.width<=INT32_MAX&&s.oh<=INT32_MAX&&s.ow<=INT32_MAX&&s.ph>=0&&s.pw>=0&&s.ph<=INT32_MAX&&s.pw<=INT32_MAX&&s.ph<=(INT64_MAX-s.height)/2&&s.pw<=(INT64_MAX-s.width)/2,"invalid conv2d");
 REQUIRE(s.height+2*s.ph>=s.kh&&s.width+2*s.pw>=s.kw&&s.oh==(s.height+2*s.ph-s.kh)/s.sh+1&&s.ow==(s.width+2*s.pw-s.kw)/s.sw+1&&grid((s.oh*s.ow+15)/16)&&(s.co+15)/16<=65535&&s.batch<=65535,"invalid conv2d output/grid");START;
 if(s.co>=16&&s.oh*s.ow>=16){hipLaunchKernelGGL(conv_fast_k,dim3((s.oh*s.ow+63)/64,(s.co+63)/64,s.batch),dim3(16,16),0,g->q,x,w,b,y,s);}else{hipLaunchKernelGGL(conv_k,dim3((s.oh*s.ow+15)/16,(s.co+15)/16,s.batch),dim3(16,16),0,g->q,x,w,b,y,s);}DONE;
}
extern "C" int nrsd_group_norm(ns_gpu* g,const float* x,nrsd_weight w,nrsd_weight b,float* y,I n,I c,I h,I width,I groups,float eps){
 REQUIRE(x&&y&&wt(w)&&wt(b)&&shape({n,c,h,width})&&groups>0&&c%groups==0&&grid(n*groups)&&std::isfinite(eps)&&eps>0,"invalid group norm");START;
#if NR_SD15_DIAGNOSTIC_GN_F64_STATS
 hipLaunchKernelGGL(gn_f64_stats_diagnostic_k,dim3(n*groups),dim3(256),0,g->q,x,w,b,y,c/groups*h*width,c/groups,h*width,groups,eps);
#else
 hipLaunchKernelGGL(norm_k,dim3(n*groups),dim3(256),0,g->q,x,w,b,y,c/groups*h*width,c/groups,h*width,groups,eps,false);
#endif
 DONE;
}
extern "C" int nrsd_layer_norm(ns_gpu* g,const float* x,nrsd_weight w,nrsd_weight b,float* y,I rows,I c,float eps){
 REQUIRE(x&&y&&wt(w)&&wt(b)&&shape({rows,c})&&grid(rows)&&std::isfinite(eps)&&eps>0,"invalid layer norm");START;
 hipLaunchKernelGGL(norm_k,dim3(rows),dim3(256),0,g->q,x,w,b,y,c,c,I(1),I(1),eps,true);DONE;
}
extern "C" int nrsd_activation(ns_gpu* g,const float* x,float* y,I n,int32_t kind){REQUIRE(x&&y&&shape({n})&&grid((n+255)/256)&&kind>=0&&kind<=3,"invalid activation");START;hipLaunchKernelGGL(act_k,dim3(blocks(n)),dim3(256),0,g->q,x,y,n,kind);DONE;}
extern "C" int nrsd_add(ns_gpu* g,const float* x,const float* b,float* y,I n,float alpha,float beta){REQUIRE(x&&y&&shape({n})&&grid((n+255)/256)&&std::isfinite(alpha)&&std::isfinite(beta),"invalid add");START;hipLaunchKernelGGL(add_k,dim3(blocks(n)),dim3(256),0,g->q,x,b,y,n,alpha,beta,false);DONE;}
extern "C" int nrsd_multiply(ns_gpu* g,const float* x,const float* b,float* y,I n){REQUIRE(x&&b&&y&&shape({n})&&grid((n+255)/256),"invalid multiply");START;hipLaunchKernelGGL(add_k,dim3(blocks(n)),dim3(256),0,g->q,x,b,y,n,1.f,1.f,true);DONE;}
extern "C" int nrsd_geglu(ns_gpu* g,const float* x,float* y,I rows,I c){REQUIRE(x&&y&&shape({rows,c,2})&&grid((rows*c+255)/256),"invalid geglu");START;hipLaunchKernelGGL(geglu_k,dim3(blocks(rows*c)),dim3(256),0,g->q,x,y,rows*c,c);DONE;}
extern "C" int nrsd_time_emb(ns_gpu* g,const float* x,const float* e,float* y,I n,I c,I h,I w){REQUIRE(x&&e&&y&&shape({n,c,h,w})&&grid((n*c*h*w+255)/256),"invalid time embedding");START;hipLaunchKernelGGL(time_k,dim3(blocks(n*c*h*w)),dim3(256),0,g->q,x,e,y,n*c*h*w,h*w);DONE;}
extern "C" int nrsd_layout(ns_gpu* g,const float* x,float* y,I n,I c,I h,I w,int32_t direction){REQUIRE(x&&y&&x!=y&&shape({n,c,h,w})&&grid((n*c*h*w+255)/256)&&(direction==0||direction==1),"invalid layout");START;hipLaunchKernelGGL(layout_k,dim3(blocks(n*c*h*w)),dim3(256),0,g->q,x,y,n*c*h*w,c,h*w,direction!=0);DONE;}
extern "C" int nrsd_up2(ns_gpu* g,const float* x,float* y,I n,I c,I h,I w){REQUIRE(x&&y&&x!=y&&shape({n,c,h,w,4})&&grid((n*c*h*w*4+255)/256),"invalid up2");START;hipLaunchKernelGGL(up_k,dim3(blocks(n*c*h*w*4)),dim3(256),0,g->q,x,y,n*c*h*w*4,h,w);DONE;}
extern "C" int nrsd_concat(ns_gpu* g,const float* x,const float* b,float* y,I n,I ca,I cb,I h,I w){REQUIRE(x&&b&&y&&ca>0&&cb>0&&ca<=INT64_MAX-cb&&shape({n,ca+cb,h,w})&&grid((n*(ca+cb)*h*w+255)/256),"invalid concat");START;hipLaunchKernelGGL(concat_k,dim3(blocks(n*(ca+cb)*h*w)),dim3(256),0,g->q,x,b,y,n*(ca+cb)*h*w,ca,cb,h*w);DONE;}
extern "C" int nrsd_attention(ns_gpu* g,const float* q,const float* k,const float* v,float* y,nrsd_attention_shape s){
 REQUIRE(q&&k&&v&&y&&s.dim<=512&&std::isfinite(s.scale)&&(s.causal==0||s.causal==1)&&grid(s.queries)&&s.heads<=65535&&s.batch<=65535&&span(s.batch,s.queries,s.heads,s.dim,s.qb,s.qt,s.qh)&&span(s.batch,s.keys,s.heads,s.dim,s.kb,s.kt,s.kh)&&span(s.batch,s.keys,s.heads,s.dim,s.vb,s.vt,s.vh)&&span(s.batch,s.queries,s.heads,s.dim,s.ob,s.ot,s.oh),"invalid attention");START;
#if NR_SD15_DIAGNOSTIC_ATTN_CAPTURE
 if(attention_capture_diagnostic(g,q,k,v,s))return -1;
#endif
#if NR_SD15_DIAGNOSTIC_ATTN_NORMALIZED_P
 return attention_normalized_diagnostic(g,q,k,v,y,s);
#else
 switch(s.dim){
#define ATTN(D) case D: hipLaunchKernelGGL((attention_coalesced_k<D>),dim3((s.queries+3)/4,s.heads,s.batch),dim3(256),0,g->q,q,k,v,y,s);break
 ATTN(40);ATTN(64);ATTN(80);ATTN(160);ATTN(512);
#undef ATTN
 default:hipLaunchKernelGGL((attention_coalesced_k<0>),dim3((s.queries+3)/4,s.heads,s.batch),dim3(256),0,g->q,q,k,v,y,s);break;
 }DONE;
#endif
}
