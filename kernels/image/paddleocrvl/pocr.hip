// Ported from nextcuda 5733ec695f53eede3700d43678228bd693f582c6.
// Explicit 16/32 lane shuffle widths preserve CUDA subgroup arithmetic.
// pocr.cu: the PaddleOCR-VL engine's kernels (pocr.h) - the projector's 2 x 2 merge, the decoder's cache copies and
// its one-row attention over grouped key / value heads; the linears, norms, RoPE, GELU and the prompt's attention are
// the shared diffusion kernels' (nsd.h). nextsycl's pocr.cpp in CUDA (the same names, arguments and arithmetic): its
// sub-groups of 16 are here 16 threads reduced with half-warp shuffles.
#include "ns.h"
#include "ns_internal.hpp"
#include "pocr.h"

#include <hip/hip_fp16.h>
#include <cmath>
#include "bounds.h"

namespace {

unsigned blocks(int64_t n, int t = 256) { return (unsigned) ((n + t - 1) / t); }

__global__ void to_half_k(const float* x, __half* o, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = __float2half_rn(x[i]);
}

__global__ void copy_rows_k(const float* src, int64_t ss, float* dst, int64_t ds, int64_t cols, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[(i / cols) * ds + i % cols] = src[(i / cols) * ss + i % cols];
}

__global__ void merge2_k(const float* x, int64_t W, int64_t C, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int64_t w2 = W / 2, c = i % (4 * C), r = i / (4 * C);
    const int64_t q = c / C, cc = c % C, y = (r / w2) * 2 + q / 2, xx = (r % w2) * 2 + q % 2;
    out[i] = x[(y * W + xx) * C + cc];
}

constexpr int64_t CHUNK = 256;

// 16 threads a (kv head, key), each D / 16 features, every query head of the group at once
__global__ void scores_k(const float* q, const float* k, int64_t n, int64_t kvs, int64_t rep, int64_t D, float scale, float* sc) {
    const int64_t w = blockIdx.x, gh = w / n, j = w % n;
    const int l = threadIdx.x;
    const int64_t per = D / 16;
    const float* kj = k + j * kvs + gh * D + l * per;
    float acc[16];
    for (int64_t r = 0; r < rep; ++r) acc[r] = 0.0f;
    for (int64_t e = 0; e < per; ++e) {
        const float kv = kj[e];
        for (int64_t r = 0; r < rep; ++r) acc[r] += q[(gh * rep + r) * D + l * per + e] * kv;
    }
    for (int64_t r = 0; r < rep; ++r) {
        float s = acc[r];
        for (int o = 8; o > 0; o >>= 1) s += __shfl_xor(s, o, 16);
        if (l == 0) sc[(gh * rep + r) * n + j] = s * scale;
    }
}

__device__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int i = 1; i < (int) (blockDim.x >> 5); ++i) t = fmaxf(t, red[i]);
    __syncthreads();
    return t;
}

__device__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float t = 0.0f;
    for (int i = 0; i < (int) (blockDim.x >> 5); ++i) t += red[i];
    __syncthreads();
    return t;
}

// each (head, chunk): its max, its sum of exp and its exp-weighted values, a thread a feature
__global__ void chunk_k(const float* sc, const float* v, int64_t n, int64_t kvs, int64_t rep, int64_t D, int64_t chunks, float* part) {
    __shared__ float red[8];
    const int64_t w = blockIdx.x, h = w / chunks, c = w % chunks;
    const int l = threadIdx.x, wg = blockDim.x;
    const int64_t j0 = c * CHUNK, j1 = min(n, j0 + CHUNK);
    const float* s = sc + h * n;
    float m = -INFINITY;
    for (int64_t j = j0 + l; j < j1; j += wg) m = fmaxf(m, s[j]);
    m = block_max(m, red);
    float sum = 0.0f;
    for (int64_t j = j0 + l; j < j1; j += wg) sum += expf(s[j] - m);
    sum = block_sum(sum, red);
    float* p = part + (h * chunks + c) * (D + 2);
    if (l < D) {
        const float* vh = v + (h / rep) * D + l;
        float acc = 0.0f;
        for (int64_t j = j0; j < j1; ++j) acc += expf(s[j] - m) * vh[j * kvs];
        p[2 + l] = acc;
    }
    if (l == 0) {
        p[0] = m;
        p[1] = sum;
    }
}

__global__ void combine_k(const float* part, int64_t D, int64_t chunks, float* out) {
    const int64_t h = blockIdx.x;
    const int l = threadIdx.x;
    if (l >= D) return;
    const float* p = part + h * chunks * (D + 2);
    float m = -INFINITY;
    for (int64_t c = 0; c < chunks; ++c) m = fmaxf(m, p[c * (D + 2)]);
    float sum = 0.0f, acc = 0.0f;
    for (int64_t c = 0; c < chunks; ++c) {
        const float f = expf(p[c * (D + 2)] - m);
        sum += p[c * (D + 2) + 1] * f;
        acc += p[c * (D + 2) + 2 + l] * f;
    }
    out[h * D + l] = acc / sum;
}

}  // namespace

extern "C" {

int ns_image_pocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n) {
    if (!g || !x || !out || n < 0 || n > POCR_MAX_ELEMENTS) return ns_fail("pocr_to_half: invalid span");
    if (n == 0) return 0;
    NS_CU(hipSetDevice(g->dev));
    to_half_k<<<blocks(n), 256, 0, g->q>>>(x, (__half*) out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_pocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols) {
    if (!g || !src || !dst || !pocr_span(rows, ss, cols) || !pocr_span(rows, ds, cols))
        return ns_fail("pocr_copy_rows: invalid span");
    if (rows == 0) return 0;
    NS_CU(hipSetDevice(g->dev));
    copy_rows_k<<<blocks(rows * cols), 256, 0, g->q>>>(src, ss, dst, ds, cols, rows * cols);
    NS_LAUNCH_CHECK();
    return 0;
}

int ns_image_pocr_merge2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    if (!g || !x || !out || H <= 0 || W <= 0 || C <= 0 || !pocr_product(H, W) || !pocr_product(H * W, C))
        return ns_fail("pocr_merge2: invalid span");
    if (H % 2 || W % 2) return ns_fail("ns_image_pocr_merge2: even sides");
    const int64_t n = H * W * C;
    if (n <= 0) return 0;
    NS_CU(hipSetDevice(g->dev));
    merge2_k<<<blocks(n), 256, 0, g->q>>>(x, W, C, out, n);
    NS_LAUNCH_CHECK();
    return 0;
}

int64_t ns_image_pocr_attn1_scratch(int64_t n, int64_t Hq, int64_t D) {
    if (n <= 0 || Hq <= 0 || D <= 0 || D > 256 || !pocr_product(Hq, n)) return -1;
    const int64_t chunks = (n + CHUNK - 1) / CHUNK;
    if (!pocr_product(Hq, chunks) || !pocr_product(Hq * chunks, D + 2)) return -1;
    const int64_t result = Hq * n + Hq * chunks * (D + 2);
    return result <= POCR_MAX_ELEMENTS ? result : -1;
}

int ns_image_pocr_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                        int64_t D, float* scratch, float* out) {
    if (!g || !q || !k || !v || !scratch || !out || n <= 0 || Hq <= 0 || Hkv <= 0 || D <= 0 ||
        !pocr_product(Hq, D) || !pocr_product(Hkv, D) || kvs < Hkv * D ||
        !pocr_span(n, kvs, Hkv * D) || ns_image_pocr_attn1_scratch(n, Hq, D) < 0)
        return ns_fail("pocr_attn1: invalid span");
    if (Hq % Hkv || D % 16 || D > 256 || Hq / Hkv > 16)
        return ns_fail("ns_image_pocr_attn1: Hq a multiple of Hkv (at most 16 a kv head), D a multiple of 16 up to 256");
    NS_CU(hipSetDevice(g->dev));
    const int64_t rep = Hq / Hkv, chunks = (n + CHUNK - 1) / CHUNK;
    float* sc = scratch;
    float* part = scratch + Hq * n;
    scores_k<<<(unsigned) (Hkv * n), 16, 0, g->q>>>(q, k, n, kvs, rep, D, 1.0f / sqrtf((float) D), sc);
    NS_LAUNCH_CHECK();
    const int wg = D <= 64 ? 64 : (D <= 128 ? 128 : 256);
    chunk_k<<<(unsigned) (Hq * chunks), wg, 0, g->q>>>(sc, v, n, kvs, rep, D, chunks, part);
    NS_LAUNCH_CHECK();
    combine_k<<<(unsigned) Hq, wg, 0, g->q>>>(part, D, chunks, out);
    NS_LAUNCH_CHECK();
    return 0;
}

}  // extern "C"

// Private subset of nextcuda nsd.cu dequant/RoPE arithmetic. The pinned model is
// BF16; no unused quantized formats or unimplemented shared ABI stubs are bound.
namespace {
// Exact-input controls show the shared single K-long accumulator contributes
// 2.4–4.3x the CPU dot-product error against F64 on decoder shapes. Shorten
// each chain by eight; retain original F16 operands and F32 accumulation/output.
// This private candidate leaves all vision and shared speech operations intact.
__global__ void pocr_linear_k(const __half* x,const __half* w,int64_t m,int64_t k,int64_t n,float* out){
 __shared__ float a[16][16],b[16][16];
 int tx=threadIdx.x,ty=threadIdx.y;int64_t r=int64_t(blockIdx.y)*16+ty,c=int64_t(blockIdx.x)*16+tx;
 float partial[8]={0,0,0,0,0,0,0,0};
 for(int64_t base=0;base<k;base+=16){
  a[ty][tx]=(r<m&&base+tx<k)?__half2float(x[r*k+base+tx]):0;
  b[ty][tx]=(int64_t(blockIdx.x)*16+ty<n&&base+tx<k)?__half2float(w[(int64_t(blockIdx.x)*16+ty)*k+base+tx]):0;
  __syncthreads();
  #pragma unroll
  for(int j=0;j<16;++j)partial[j%8]=fmaf(a[ty][j],b[tx][j],partial[j%8]);
  __syncthreads();
 }
 float sum=((partial[0]+partial[1])+(partial[2]+partial[3]))+((partial[4]+partial[5])+(partial[6]+partial[7]));
 if(r<m&&c<n)out[r*n+c]=sum;
}
__global__ void pocr_dequant_k(const void* src,int type,int64_t n,void* out,int odt){
 int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;if(i>=n)return;
 float x=type==30?__uint_as_float(uint32_t(((const uint16_t*)src)[i])<<16):type==1?__half2float(((const __half*)src)[i]):((const float*)src)[i];
 if(odt==0)((float*)out)[i]=x;else ((__half*)out)[i]=__float2half_rn(x);
}
__global__ void pocr_rope_k(float* x,int64_t heads,int64_t dim,int64_t stride,const float* cs){
 int64_t r=blockIdx.x,h=blockIdx.y,half=dim/2,base=r*stride+h*dim;
 for(int64_t i=threadIdx.x;i<half;i+=blockDim.x){float c=cs[(r*half+i)*2],s=cs[(r*half+i)*2+1],a=x[base+i],b=x[base+half+i];x[base+i]=a*c-b*s;x[base+half+i]=b*c+a*s;}
}
// Shared speech attention's F32 arithmetic/reduction, private launch ceiling
// extended to the original 8192-patch spotting cap. No global score matrix.
__global__ void pocr_attention_k(const float* q,const float* k,const float* v,int64_t rows,int64_t hq,int64_t hkv,int64_t d,int64_t qs,int64_t ks,float* out,bool causal,bool normalize_first){
 extern __shared__ float score[];__shared__ float tmp[256];
 int t=threadIdx.x;int64_t r=blockIdx.x,h=blockIdx.y,kh=h/(hq/hkv),end=causal?r+1:rows;float best=-INFINITY;
 for(int64_t j=t;j<end;j+=256){float z=0;for(int64_t a=0;a<d;++a)z=fmaf(q[r*qs+h*d+a],k[j*ks+kh*d+a],z);z*=rsqrtf(float(d));score[j]=z;best=fmaxf(best,z);}tmp[t]=best;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)tmp[t]=fmaxf(tmp[t],tmp[t+s]);__syncthreads();}best=tmp[0];__syncthreads();float sum=0;
 for(int64_t j=t;j<end;j+=256){float z=expf(score[j]-best);score[j]=z;sum+=z;}tmp[t]=sum;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}float den=tmp[0];
 // Match original nextcuda softmax_k and Transformers: rounded normalized
 // probabilities precede P@V. Opt-in isolates this order from other changes.
 if(normalize_first){float inv=1.0f/den;for(int64_t j=t;j<end;j+=256)score[j]*=inv;__syncthreads();}
 for(int64_t a=t;a<d;a+=256){float z=0;for(int64_t j=0;j<end;++j)z=fmaf(score[j],v[j*ks+kh*d+a],z);out[(r*hq+h)*d+a]=normalize_first?z:z/den;}
}
}
extern "C" int ns_image_pocr_linear(ns_gpu* g,const void* x,int64_t m,int64_t k,const void* w,int64_t n,float* out){
 if(!g||!x||!w||!out||m<=0||m>65535*16||k<=0||n<=0||!pocr_product(m,k)||!pocr_product(n,k)||!pocr_product(m,n))return ns_fail("pocr_linear: invalid span");
 NS_CU(hipSetDevice(g->dev));pocr_linear_k<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,g->q>>>((const __half*)x,(const __half*)w,m,k,n,out);NS_LAUNCH_CHECK();return 0;
}
extern "C" int ns_image_pocr_dequant(ns_gpu* g,const void* src,int type,int64_t n,void* out,int odt){
 if(!g||!src||!out||n<=0||n>POCR_MAX_ELEMENTS||(type!=0&&type!=1&&type!=30)||(odt!=0&&odt!=1))return ns_fail("pocr_dequant: invalid span/type");
 NS_CU(hipSetDevice(g->dev));pocr_dequant_k<<<blocks(n),256,0,g->q>>>(src,type,n,out,odt);NS_LAUNCH_CHECK();return 0;
}
extern "C" int ns_image_pocr_rope(ns_gpu* g,float* x,int64_t rows,int64_t heads,int64_t dim,int64_t stride,const float* cs){
 if(!g||!x||!cs||rows<=0||heads<=0||heads>65535||dim<=0||dim%2||dim>256||!pocr_product(heads,dim)||!pocr_span(rows,stride,heads*dim))return ns_fail("pocr_rope: invalid span");
 NS_CU(hipSetDevice(g->dev));pocr_rope_k<<<dim3(rows,heads),128,0,g->q>>>(x,heads,dim,stride,cs);NS_LAUNCH_CHECK();return 0;
}
extern "C" int ns_image_pocr_attention(ns_gpu* g,const float* q,const float* k,const float* v,int64_t rows,int64_t heads,int64_t kv,int64_t dim,int64_t qs,int64_t ks,float* out,int causal){
 if(!g||!q||!k||!v||!out||rows<=0||rows>8192||heads<=0||heads>256||kv<=0||heads%kv||dim<=0||dim>256||!pocr_span(rows,qs,heads*dim)||!pocr_span(rows,ks,kv*dim)||(causal<0||causal>3))return ns_fail("pocr_attention: invalid span");
 NS_CU(hipSetDevice(g->dev));pocr_attention_k<<<dim3(rows,heads),256,rows*sizeof(float),g->q>>>(q,k,v,rows,heads,kv,dim,qs,ks,out,(causal&1)!=0,(causal&2)!=0);NS_LAUNCH_CHECK();return 0;
}

// Default-off private vision-only candidate; preserve original attention ABI.
#include "attention_tiled.hpp"
extern "C" int ns_image_pocr_attention_tiled(ns_gpu* g,const float* q,const float* k,const float* v,int64_t rows,int64_t heads,int64_t dim,int64_t stride,float* out){
 if(!g||rows>5120||!pocr_tiled::arguments(q,k,v,rows,heads,dim,stride,out))return ns_fail("pocr_attention_tiled: invalid vision span");
 NS_CU(hipSetDevice(g->dev));
 NS_CU(pocr_tiled::launch(q,k,v,rows,heads,dim,stride,out,g->q));
 return 0;
}
