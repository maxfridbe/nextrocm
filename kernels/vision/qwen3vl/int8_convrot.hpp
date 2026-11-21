// Private experimental Qwen3-VL INT8 operators. No production dispatch.
// Arithmetic contract and independent oracle: tests/README.md.
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>

namespace qv_int8 {
using I = int64_t;
enum Dtype : int { F32 = 0, BF16 = 1 };
struct Span { uintptr_t begin, end; };
inline bool span(const void* p, size_t count, size_t width, Span& s) {
    if (!p || !count || !width || uintptr_t(p) % width ||
        count > std::numeric_limits<size_t>::max() / width) return false;
    const size_t bytes = count * width;
    s = {uintptr_t(p), uintptr_t(p)};
    if (bytes > UINTPTR_MAX - s.begin) return false;
    s.end += bytes;
    return true;
}
inline bool separate(Span a, Span b) { return a.end <= b.begin || b.end <= a.begin; }
inline bool geometry(I m, I n, I k, int dtype, I scales) {
    return m > 0 && m <= 4096 && n > 0 && n <= 24576 && k > 0 &&
        k <= 12288 && k % 256 == 0 && (dtype == F32 || dtype == BF16) &&
        (scales == 1 || scales == n);
}
// All storage supplied by caller, including diagnostic intermediate tensors.
// No allocation or synchronization inside launch; numerical status is mandatory.
struct Args {
    const void* x; const int8_t* weight; const float* weight_scale;
    const void* bias; void* out;
    float* rotated; int8_t* quantized; float* row_scale; int32_t* accum;
    uint32_t* status;
    I m, n, k, scales; int dtype;
};
inline bool valid(const Args& a) {
    if (!geometry(a.m,a.n,a.k,a.dtype,a.scales)) return false;
    const size_t mk=size_t(a.m)*size_t(a.k), nk=size_t(a.n)*size_t(a.k);
    const size_t mn=size_t(a.m)*size_t(a.n), width=a.dtype==F32 ? 4 : 2;
    Span reads[4], writes[6];
    if (!span(a.x,mk,width,reads[0]) || !span(a.weight,nk,1,reads[1]) ||
        !span(a.weight_scale,size_t(a.scales),4,reads[2]) ||
        (a.bias && !span(a.bias,size_t(a.n),width,reads[3])) ||
        !span(a.out,mn,width,writes[0]) || !span(a.rotated,mk,4,writes[1]) ||
        !span(a.quantized,mk,1,writes[2]) || !span(a.row_scale,size_t(a.m),4,writes[3]) ||
        !span(a.accum,mn,4,writes[4]) || !span(a.status,1,4,writes[5])) return false;
    for (int i=0;i<6;++i) {
        for (int j=0;j<(a.bias ? 4 : 3);++j) if (!separate(writes[i],reads[j])) return false;
        for (int j=0;j<i;++j) if (!separate(writes[i],writes[j])) return false;
    }
    return true;
}
// Exact regular H4 tensor product, not the Sylvester Hadamard convention.
inline int hadamard_sign(unsigned i, unsigned j) {
    unsigned parity=0;
    for (int digit=0;digit<4;++digit) {
        parity ^= unsigned(((i & 3u) + (j & 3u)) == 3u);
        i >>= 2; j >>= 2;
    }
    return parity ? -1 : 1;
}
} // namespace qv_int8

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
namespace qv_int8 {
__device__ inline float bf_round(float x) {
    unsigned u=__float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) return __uint_as_float(u | 0x00400000u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}
__device__ inline float load(const void* p,size_t i,int dtype) {
    return dtype==F32 ? static_cast<const float*>(p)[i]
        : __uint_as_float(unsigned(static_cast<const uint16_t*>(p)[i]) << 16);
}
__device__ inline void store(void* p,size_t i,float x,int dtype) {
    if (dtype==F32) static_cast<float*>(p)[i]=x;
    else static_cast<uint16_t*>(p)[i]=uint16_t(__float_as_uint(bf_round(x))>>16);
}
__global__ void rotate(Args a) {
    __shared__ float values[256];
    const unsigned t=threadIdx.x;
    const size_t base=size_t(blockIdx.x)*256;
    const float x=load(a.x,base+t,a.dtype);
    if (!isfinite(x)) atomicOr(a.status,1u);
    values[t]=x;
    __syncthreads();
    float sum=0;
    for (unsigned i=0;i<256;++i) {
        unsigned ii=i,jj=t,parity=0;
        #pragma unroll
        for (int digit=0;digit<4;++digit) {
            parity ^= unsigned(((ii&3u)+(jj&3u))==3u); ii>>=2; jj>>=2;
        }
        // Dense reference path. Do not substitute butterfly reduction before
        // quantization boundary comparisons against the pinned CPU oracle.
        sum=fmaf(values[i],parity ? -0.0625f : 0.0625f,sum);
    }
    if(a.dtype==BF16) sum=bf_round(sum);
    if(!isfinite(sum)) atomicOr(a.status,2u);
    a.rotated[base+t]=sum;
}
__global__ void quantize(Args a) {
    __shared__ float maxima[256];
    __shared__ unsigned bad[256];
    const unsigned t=threadIdx.x;
    const size_t base=size_t(blockIdx.x)*size_t(a.k);
    float best=0; unsigned invalid=0;
    for(I k=t;k<a.k;k+=256) {
        float x=a.rotated[base+k]; invalid |= !isfinite(x);
        best=fmaxf(best,fabsf(x));
    }
    maxima[t]=best; bad[t]=invalid; __syncthreads();
    for(unsigned d=128;d;d>>=1) {
        if(t<d) { maxima[t]=fmaxf(maxima[t],maxima[t+d]); bad[t]|=bad[t+d]; }
        __syncthreads();
    }
    const float scale=fmaxf(__fdiv_rn(maxima[0],127.0f),1e-30f);
    float math_scale=a.dtype==BF16 ? bf_round(scale) : scale;
    if(math_scale==0) math_scale=__uint_as_float(0x00800000u);
    if(t==0) a.row_scale[blockIdx.x]=scale;
    for(I k=t;k<a.k;k+=256) {
        float q=__fdiv_rn(a.rotated[base+k],math_scale);
        if(a.dtype==BF16) q=bf_round(q);
        a.quantized[base+k]=bad[0] ? int8_t(0) : int8_t(fminf(127.f,fmaxf(-128.f,rintf(q))));
    }
}
__global__ void gemm(Args a) {
    __shared__ int8_t xs[8][32], ws[32][32];
    const unsigned t=threadIdx.x, rr=t/32, cc=t%32;
    const I r=I(blockIdx.y)*8+rr, n=I(blockIdx.x)*32+cc;
    int32_t sum=0;
    for(I kb=0;kb<a.k;kb+=32) {
        xs[rr][cc]=r<a.m ? a.quantized[size_t(r)*a.k+kb+cc] : int8_t(0);
        for(unsigned j=t;j<1024;j+=256) {
            const I wn=I(blockIdx.x)*32+j/32;
            ws[j/32][j%32]=wn<a.n ? a.weight[size_t(wn)*a.k+kb+j%32] : int8_t(0);
        }
        __syncthreads();
        #pragma unroll
        for(unsigned j=0;j<32;++j) sum+=int(xs[rr][j])*int(ws[cc][j]);
        __syncthreads();
    }
    if(r<a.m && n<a.n) {
        const size_t at=size_t(r)*a.n+n;
        a.accum[at]=sum;
        const float wscale=a.weight_scale[a.scales==1 ? 0 : n];
        if(!isfinite(wscale) || wscale<=0) atomicOr(a.status,4u);
        const float scale=__fmul_rn(a.row_scale[r],wscale);
        float y=__fmul_rn(float(sum),scale);
        if(a.dtype==BF16) y=bf_round(y);
        if(a.bias) y=__fadd_rn(y,load(a.bias,size_t(n),a.dtype));
        if(a.dtype==BF16) y=bf_round(y);
        if(!isfinite(y)) atomicOr(a.status,8u);
        store(a.out,at,y,a.dtype);
    }
}
inline hipError_t launch(const Args& a,hipStream_t stream) {
    if(!valid(a)) return hipErrorInvalidValue;
    hipError_t e=hipMemsetAsync(a.status,0,4,stream);
    if(e!=hipSuccess) return e;
    rotate<<<dim3(unsigned(a.m*a.k/256)),256,0,stream>>>(a);
    if((e=hipGetLastError())!=hipSuccess) return e;
    quantize<<<dim3(unsigned(a.m)),256,0,stream>>>(a);
    if((e=hipGetLastError())!=hipSuccess) return e;
    gemm<<<dim3(unsigned((a.n+31)/32),unsigned((a.m+7)/8)),256,0,stream>>>(a);
    return hipGetLastError();
}
} // namespace qv_int8
#endif
