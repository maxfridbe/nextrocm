// Private Whisper F16 x F16 -> F16 candidate; no production dispatch.
// Contiguous x[M,K], w[N,K], optional F32 bias[N], out[M,N].
// Contract: native speech.hip linear_kernel ascending F32 fmaf through K16
// padding, separate bias-or-+0 addition, ONE final round-to-nearest-even half.
// CUDA cuBLAS then half bias may round twice; it is not the bitwise oracle.
// No split-K, reassociation, global scratch, or wave-width assumptions.
// Proposed ABI (integration owner adds only after validation):
// int ns_audio_whisper_linear_f16_tiled(ns_gpu*, const void* x, int64_t M,
//   int64_t K, const void* w, int64_t N, const float* bias, void* out);
// Validate GPU and arguments, set device, call launch on g->q, propagate errors.
// Proposed opt-in NS_WHISPER_LINEAR_F16_TILED=1, Whisper only, F16 in/out;
// unsupported shapes use original nsd.linear before selection. No defaults.
// See docs/whisper-performance.md for reference revisions and validation gates.
#pragma once
#if defined(__FAST_MATH__)
#error "Whisper linear requires ordered F32 FMA and separate bias rounding"
#endif
#include <cstdint>
#include <climits>
namespace whisper_linear_f16 {
using I=int64_t;
constexpr I rows=64, cols=64, depth=32;
constexpr I shared_bytes=(rows+cols)*(depth+1)*sizeof(float);
inline bool supported(I M,I K,I N) {
    if(M<1||M>4096||K<=0||N<=0) return false;
    // All individual spans and their sum fit the tracked 7.5 GB ceiling.
    constexpr I limit=1874999999LL;
    if(K>limit/M||N>limit/M||N>limit/K) return false;
    const I bytes=2*(M*K+N*K+M*N)+4*N;
    return bytes<7500000000LL && (N+cols-1)/cols<=INT_MAX;
}
inline bool arguments(const void* x,I M,I K,const void* w,I N,
                      const float* bias,void* out) {
    if(!supported(M,K,N)||!x||!w||!out) return false;
    const uintptr_t a=reinterpret_cast<uintptr_t>(x),b=reinterpret_cast<uintptr_t>(w),
                    c=reinterpret_cast<uintptr_t>(out),d=reinterpret_cast<uintptr_t>(bias);
    if(a%2||b%2||c%2||(bias&&d%alignof(float))) return false;
    const uintptr_t bytes=uintptr_t(M*N)*2;
    if(c>UINTPTR_MAX-bytes) return false;
    auto disjoint=[&](uintptr_t p,uintptr_t size) {
        return p<=UINTPTR_MAX-size&&(p+size<=c||c+bytes<=p);
    };
    return disjoint(a,uintptr_t(M*K)*2)&&disjoint(b,uintptr_t(N*K)*2)&&
           (!bias||disjoint(d,uintptr_t(N)*4));
}
} // namespace whisper_linear_f16

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
namespace whisper_linear_f16 {
// 256 threads, 4x4 F32 registers per thread, 16,896 bytes LDS. Adjacent
// loaders traverse K; 33-float strides avoid the power-of-two LDS bank stride.
// Exactly two barriers per 32 K values, no wave-width dependent operations.
__global__ void tiled(const __half* x,I M,I K,const __half* w,I N,
                      const float* bias,__half* out) {
    __shared__ float a[64][33],b[64][33];
    const int tx=threadIdx.x%16,ty=threadIdx.x/16,t=threadIdx.x;
    const I r0=I(blockIdx.y)*64,c0=I(blockIdx.x)*64;
    float acc[4][4]={};
    for(I base=0;base<K;base+=32) {
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int z=t+i*256,r=z/32,k=z%32;
            a[r][k]=(r0+r<M&&base+k<K)?__half2float(x[(r0+r)*K+base+k]):0.f;
        }
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int z=t+i*256,c=z/32,k=z%32;
            b[c][k]=(c0+c<N&&base+k<K)?__half2float(w[(c0+c)*K+base+k]):0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int half=0;half<2;++half) if(base+half*16<K) {
            #pragma unroll
            for(int k=0;k<16;++k) {
                float av[4],bv[4];
                #pragma unroll
                for(int i=0;i<4;++i) { av[i]=a[ty+i*16][half*16+k];bv[i]=b[tx+i*16][half*16+k]; }
                #pragma unroll
                for(int i=0;i<4;++i) {
                    #pragma unroll
                    for(int j=0;j<4;++j) acc[i][j]=fmaf(av[i],bv[j],acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;++i) {
        const I r=r0+ty+i*16;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const I c=c0+tx+j*16;
            if(r<M&&c<N) out[r*N+c]=__float2half_rn(acc[i][j]+(bias?bias[c]:0.f));
        }
    }
}
inline hipError_t launch(const void* x,I M,I K,const void* w,I N,
                         const float* bias,void* out,hipStream_t stream) {
    if(!arguments(x,M,K,w,N,bias,out)) return hipErrorInvalidValue;
    tiled<<<dim3(unsigned((N+63)/64),unsigned((M+63)/64)),256,0,stream>>>(
        static_cast<const __half*>(x),M,K,static_cast<const __half*>(w),N,bias,static_cast<__half*>(out));
    return hipGetLastError();
}
} // namespace whisper_linear_f16
#endif
