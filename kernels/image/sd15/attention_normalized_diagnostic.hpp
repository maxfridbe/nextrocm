// Private arithmetic diagnostic, included inside sd15.hip's anonymous namespace.
// Unlike the production ABI behavior, this path allocates tracked scratch and
// synchronizes on release. QK uses ascending-dimension fmaf, as production does.
// Softmax has a global row max/sum; normalized FP32 P is stored BEFORE PV.
// This isolates a different arithmetic path, not exact PyTorch reduction order.
constexpr size_t attention_diagnostic_cap = size_t(64) * 1024 * 1024;
static_assert(attention_diagnostic_cap <= size_t(1) * 1024 * 1024 * 1024);
// Paired control: identical global max/sum and QK/PV orders. Only division
// placement changes. This flag has no effect outside the diagnostic dispatch.
#ifndef NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV
#define NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV 1
#endif
#if NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV != 0 && NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV != 1
#error "NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV must be 0 or 1"
#endif

__global__ void attention_diagnostic_qk(const float* q,const float* k,float* p,
                                       nrsd_attention_shape s,I first) {
 I row=first+blockIdx.x,qr=row%s.queries,h=(row/s.queries)%s.heads;
 I b=row/(s.queries*s.heads),qo=b*s.qb+qr*s.qt+h*s.qh,ko=b*s.kb+h*s.kh;
 I end=s.causal?(qr+1<s.keys?qr+1:s.keys):s.keys;
 for(I j=threadIdx.x;j<s.keys;j+=256){
  float z=-INFINITY;
  if(j<end){z=0;for(I d=0;d<s.dim;++d)z=fmaf(q[qo+d],k[ko+j*s.kt+d],z);z*=s.scale;}
  p[I(blockIdx.x)*s.keys+j]=z;
 }
}
__global__ void attention_diagnostic_softmax(float* p,float* denominators,nrsd_attention_shape s,I first) {
 __shared__ float tmp[256];
 int t=threadIdx.x;I qr=(first+blockIdx.x)%s.queries;
 I end=s.causal?(qr+1<s.keys?qr+1:s.keys):s.keys;
 float* row=p+I(blockIdx.x)*s.keys;
 float z=-INFINITY;
 for(I j=t;j<end;j+=256)z=fmaxf(z,row[j]);
 tmp[t]=z;__syncthreads();
 for(int step=128;step;step/=2){if(t<step)tmp[t]=fmaxf(tmp[t],tmp[t+step]);__syncthreads();}
 float maximum=tmp[0];__syncthreads();z=0;
 for(I j=t;j<end;j+=256){float e=expf(row[j]-maximum);row[j]=e;z+=e;}
 tmp[t]=z;__syncthreads();
 for(int step=128;step;step/=2){if(t<step)tmp[t]+=tmp[t+step];__syncthreads();}
 float denominator=tmp[0];
 if(t==0)denominators[blockIdx.x]=denominator;
#if NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV
 for(I j=t;j<s.keys;j+=256)row[j]=j<end?row[j]/denominator:0.f;
#else
 for(I j=t;j<s.keys;j+=256)if(j>=end)row[j]=0.f;
#endif
}
__global__ void attention_diagnostic_pv(const float* p,const float* denominators,const float* v,float* y,
                                       nrsd_attention_shape s,I first) {
 I row=first+blockIdx.x,qr=row%s.queries,h=(row/s.queries)%s.heads;
 I b=row/(s.queries*s.heads),vo=b*s.vb+h*s.vh,yo=b*s.ob+qr*s.ot+h*s.oh;
 I end=s.causal?(qr+1<s.keys?qr+1:s.keys):s.keys;
 for(I d=threadIdx.x;d<s.dim;d+=256){
  float z=0;
  for(I j=0;j<end;++j)z=fmaf(p[I(blockIdx.x)*s.keys+j],v[vo+j*s.vt+d],z);
  y[yo+d]=NR_SD15_DIAGNOSTIC_ATTN_DIVIDE_BEFORE_PV?z:z/denominators[blockIdx.x];
 }
}
// Caller has already validated all tensor spans, shape products and the device.
// Chunk whole rows only: every softmax sees all keys, including across 256 tails.
int attention_normalized_diagnostic(ns_gpu* g,const float* q,const float* k,
                                   const float* v,float* y,nrsd_attention_shape s) {
 if(s.keys>=I(attention_diagnostic_cap/sizeof(float)))
  return ns_fail("nrsd: diagnostic attention row exceeds 64 MiB scratch cap");
 const I rows=s.batch*s.heads*s.queries;
 I chunk=I(attention_diagnostic_cap/((size_t(s.keys)+1)*sizeof(float)));
 if(chunk>rows)chunk=rows;
 void* storage=nullptr;
 if(ns_alloc(g,size_t(chunk)*(size_t(s.keys)+1)*sizeof(float),&storage))return -1;
 float* p=static_cast<float*>(storage);
 float* denominators=p+chunk*s.keys;
 hipError_t error=hipSuccess;
 for(I first=0;first<rows;first+=chunk){
  I count=rows-first<chunk?rows-first:chunk;
  hipLaunchKernelGGL(attention_diagnostic_qk,dim3(count),dim3(256),0,g->q,q,k,p,s,first);
  error=hipGetLastError();if(error!=hipSuccess)break;
  hipLaunchKernelGGL(attention_diagnostic_softmax,dim3(count),dim3(256),0,g->q,p,denominators,s,first);
  error=hipGetLastError();if(error!=hipSuccess)break;
  hipLaunchKernelGGL(attention_diagnostic_pv,dim3(count),dim3(256),0,g->q,p,denominators,v,y,s,first);
  error=hipGetLastError();if(error!=hipSuccess)break;
 }
 // ns_free synchronizes, frees, and updates runtime accounting. Always attempt
 // it after any launch failure. It can stop before hipFree on a sync error.
 int cleanup=ns_free(g,storage);
 std::string cleanup_error;
 if(cleanup){
  cleanup_error=ns_last_error();
  // Best effort after a failed synchronized release. Never untrack a pointer
  // unless HIP confirms it was freed, and never hide the original sync error.
  hipError_t release=hipSetDevice(g->dev);
  if(release==hipSuccess)release=hipFree(storage);
  if(release==hipSuccess){
   auto it=g->allocations.find(storage);
   if(it!=g->allocations.end()){g->allocated-=it->second;g->allocations.erase(it);}
  }else cleanup_error+=std::string("; scratch release failed: ")+hipGetErrorString(release);
 }
 if(error!=hipSuccess){
  std::string message=std::string("nrsd: diagnostic attention launch: ")+hipGetErrorString(error);
  if(cleanup)message+=std::string("; cleanup: ")+cleanup_error;
  return ns_fail(message);
 }
 return cleanup?ns_fail(cleanup_error):0;
}
