// Model-private 8192-position MQA/CLIP attention. Based on local native speech
// attention semantics, extended without changing its 4096-position contract.
// No global score scratch, <=33 KiB shared storage. Correctness baseline only.
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include "starvector.h"
#include <hip/hip_fp16.h>
#include <cmath>
namespace {
using I=int64_t;
__global__ void quick_gelu_f32(float* x,I count){
 const I i=I(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i<count){const float v=x[i];const float scaled=1.702f*v;
  x[i]=v*(1.0f/(1.0f+expf(-scaled)));}
}
__device__ float get(const void* p,int dt,I i){return dt?__half2float(static_cast<const __half*>(p)[i]):static_cast<const float*>(p)[i];}
__device__ void put(void* p,int dt,I i,float x){if(dt)static_cast<__half*>(p)[i]=__float2half_rn(x);else static_cast<float*>(p)[i]=x;}
__global__ void attention(const void* q,const void* k,const void* v,int dt,I row0,I keys,I hq,I hkv,I d,I qs,I ks,void* out,int odt,bool causal){
 extern __shared__ float score[];__shared__ float tmp[256];
 int t=threadIdx.x;I r=blockIdx.x,h=blockIdx.y,kh=h/(hq/hkv),end=causal?row0+r+1:keys;
 float best=-INFINITY;
 for(I j=t;j<end;j+=256){float z=0;for(I a=0;a<d;++a)z=fmaf(get(q,dt,r*qs+h*d+a),get(k,dt,j*ks+kh*d+a),z);z*=rsqrtf(float(d));score[j]=z;best=fmaxf(best,z);}
 tmp[t]=best;__syncthreads();
 for(int n=128;n;n/=2){if(t<n)tmp[t]=fmaxf(tmp[t],tmp[t+n]);__syncthreads();}
 best=tmp[0];__syncthreads();float sum=0;
 for(I j=t;j<end;j+=256){float z=expf(score[j]-best);score[j]=z;sum+=z;}
 tmp[t]=sum;__syncthreads();
 for(int n=128;n;n/=2){if(t<n)tmp[t]+=tmp[t+n];__syncthreads();}
 float den=tmp[0];
 for(I a=t;a<d;a+=256){float z=0;for(I j=0;j<end;++j)z=fmaf(score[j],get(v,dt,j*ks+kh*d+a),z);put(out,odt,(r*hq+h)*d+a,z/den);}
}
int launch(void* ctx,const void* q,const void* k,const void* v,int dt,I row0,I rows,I keys,I hq,I hkv,I d,I qs,I ks,void* out,int odt,bool causal){
 if(!ctx||!q||!k||!v||!out||(dt!=0&&dt!=1)||(odt!=0&&odt!=1)||row0<0||row0>=8192||rows<1||rows>8192-row0||keys<1||keys>8192||hq<1||hq>16||hkv<1||hkv>hq||hq%hkv||d<1||d>128||qs<hq*d||ks<hkv*d||qs>INT64_MAX/4/8192||ks>INT64_MAX/4/8192)
  return ns_fail("nrsv attention: invalid geometry/type (8192 position cap)");
 auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 hipLaunchKernelGGL(attention,dim3(rows,hq),dim3(256),size_t(keys)*4,g->q,q,k,v,dt,row0,keys,hq,hkv,d,qs,ks,out,odt,causal);
 NS_LAUNCH_CHECK();return 0;
}
}
extern "C" int nrsv_attention(void* g,const void* q,const void* k,const void* v,int dt,I rows,I heads,I dim,I stride,void* out,int odt){return launch(g,q,k,v,dt,0,rows,rows,heads,heads,dim,stride,stride,out,odt,false);}
extern "C" int nrsv_quick_gelu_f32(void* ctx,float* x,I count){
 if(!ctx||!x||count<1||count>257LL*4096)return ns_fail("nrsv QuickGELU: invalid pointer/count");
 auto* g=static_cast<ns_gpu*>(ctx);NS_CU(hipSetDevice(g->dev));
 hipLaunchKernelGGL(quick_gelu_f32,dim3((count+255)/256),dim3(256),0,g->q,x,count);
 NS_LAUNCH_CHECK();return 0;
}
extern "C" int nrsv_attention_causal_rows(void* g,const void* q,const void* k,const void* v,int dt,I row0,I rows,I hq,I hkv,I dim,I qs,I kvs,void* out){
 if(row0<0||row0>=8192||rows<1||rows>8192-row0)return ns_fail("nrsv attention: invalid causal extent");
 return launch(g,q,k,v,dt,row0,rows,row0+rows,hq,hkv,dim,qs,kvs,out,dt,true);
}

// Optional private experiment. Normal runtime builds export no candidate ABI.
#if defined(NR_STARVECTOR_FFN_SMALL_M) && NR_STARVECTOR_FFN_SMALL_M
#include "ffn_small_m.hpp"
extern "C" int nrsv_ffn_small_m_apply(void* g, const void* x, int dt, int64_t m,
                                int64_t k, const void* w, int64_t n,
                                const float* b, void* y, int odt) {
    return nrsv_ffn_small_m::try_launch(g, x, dt, m, k, w, n, b, y, odt);
}
#endif
