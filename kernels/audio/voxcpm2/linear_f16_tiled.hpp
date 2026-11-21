// Private VoxCPM2 wide-row F16 x F16 -> F32 candidate; no default dispatch.
// x[M,K], w[N,K], optional F32 bias[N], out[M,N], contiguous and nonaliasing.
// Preserve kernels/ops/speech.hip linear_kernel: ascending K F32 fmaf, zeros
// through ceil(K/16)*16 (not ceil(K/32)*32), then one bias-or-+0 addition.
// No split-K, half accumulation, reassociation, vendor library or global scratch.
//
// Reference inspection:
// nextcuda 5733ec695f53eede3700d43678228bd693f582c6 Mat::apply stages half
// inputs and calls nsd_linear -> cublasGemmEx with CUBLAS_COMPUTE_32F.
// Available nextsycl 98124afb83f10c9e0cfbabd1f2495d2417df0a78 nsd_linear
// caches oneDNN matmul and may stage output in input dtype before conversion;
// that rounding is NOT this native baseline's operator contract.
// Follows their private-kernel/support-check, matched workload, host-oracle
// methodology. Register reuse follows local SD15 matmul_fast.hpp, specialized
// for M=22 rather than its 64-row tile. Shared kernels remain untouched.
//
// Proposed private binding (parent/Maxwell owns integration after validation):
// int ns_audio_vcp_linear_f16_tiled(ns_gpu* g, const void* x, int64_t M,
//     int64_t K, const void* w, int64_t N, const float* bias, float* out);
// Wrapper: validate g and arguments(), hipSetDevice(g->dev), then translate
// launch(x,M,K,w,N,bias,out,g->q) via NS_CU. No synchronization or allocation.
// Proposed gate NS_VCP_LINEAR_F16_TILED=1; unset/unsupported uses nsd.linear.
// Keep Mat::apply's M<=8 and f32_rows branches ahead of this gate. Call only
// AFTER the original to_half/dequant_rows staging: never silently use F32
// activations, or raw int8 weights, or override the independent precision gate.
// Initial model gate: VoxCPM2 only, 9<=M<=64, F16 operands and F32 output.
// Errors after selection propagate; do not fall back after a failed launch.
#pragma once
#if defined(__FAST_MATH__)
#error "VoxCPM2 linear requires ordered F32 FMA and separate bias rounding"
#endif
#include <cstdint>
#include <climits>
namespace vcp_linear_f16 {
using I=int64_t;
constexpr I rows=32, cols=64, depth=32;
constexpr I shared_bytes=(rows+cols)*(depth+1)*sizeof(float);
inline bool supported(I M,I K,I N) {
    if(M<9||M>64||K<=0||N<=0) return false;
    // All individual spans and their sum fit the tracked 7.5 GB ceiling.
    constexpr I limit=1874999999LL;
    if(K>limit/M||N>limit/M||N>limit/K) return false;
    const I bytes=2*(M*K+N*K)+4*(M*N+N);
    return bytes<7500000000LL && (N+cols-1)/cols<=INT_MAX;
}
inline bool arguments(const void* x,I M,I K,const void* w,I N,
                      const float* bias,float* out) {
    if(!supported(M,K,N)||!x||!w||!out) return false;
    const uintptr_t a=reinterpret_cast<uintptr_t>(x),b=reinterpret_cast<uintptr_t>(w),
                    c=reinterpret_cast<uintptr_t>(out),d=reinterpret_cast<uintptr_t>(bias);
    if(a%2||b%2||c%alignof(float)||(bias&&d%alignof(float))) return false;
    const uintptr_t bytes=uintptr_t(M*N)*4;
    if(c>UINTPTR_MAX-bytes) return false;
    auto disjoint=[&](uintptr_t p,uintptr_t size) {
        return p<=UINTPTR_MAX-size&&(p+size<=c||c+bytes<=p);
    };
    return disjoint(a,uintptr_t(M*K)*2)&&disjoint(b,uintptr_t(N*K)*2)&&
           (!bias||disjoint(d,uintptr_t(N)*4));
}
} // namespace vcp_linear_f16

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
namespace vcp_linear_f16 {
// 128 threads, 4x4 F32 registers per thread, 12,672 bytes LDS. Adjacent
// loaders traverse K; 33-float strides avoid the power-of-two LDS bank stride.
// Exactly two barriers per 32 K values, no wave-width dependent operations.
__global__ void tiled(const __half* x,I M,I K,const __half* w,I N,
                      const float* bias,float* out) {
    __shared__ float a[32][33],b[64][33];
    const int tx=threadIdx.x%16,ty=threadIdx.x/16,t=threadIdx.x;
    const I r0=I(blockIdx.y)*32,c0=I(blockIdx.x)*64;
    float acc[4][4]={};
    for(I base=0;base<K;base+=32) {
        #pragma unroll
        for(int i=0;i<8;++i) {
            const int z=t+i*128,r=z/32,k=z%32;
            a[r][k]=(r0+r<M&&base+k<K)?__half2float(x[(r0+r)*K+base+k]):0.f;
        }
        #pragma unroll
        for(int i=0;i<16;++i) {
            const int z=t+i*128,c=z/32,k=z%32;
            b[c][k]=(c0+c<N&&base+k<K)?__half2float(w[(c0+c)*K+base+k]):0.f;
        }
        __syncthreads();
        #pragma unroll
        for(int half=0;half<2;++half) if(base+half*16<K) {
            #pragma unroll
            for(int k=0;k<16;++k) {
                float av[4],bv[4];
                #pragma unroll
                for(int i=0;i<4;++i) { av[i]=a[ty+i*8][half*16+k];bv[i]=b[tx+i*16][half*16+k]; }
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
        const I r=r0+ty+i*8;
        #pragma unroll
        for(int j=0;j<4;++j) {
            const I c=c0+tx+j*16;
            if(r<M&&c<N) out[r*N+c]=acc[i][j]+(bias?bias[c]:0.f);
        }
    }
}
inline hipError_t launch(const void* x,I M,I K,const void* w,I N,
                         const float* bias,float* out,hipStream_t stream) {
    if(!arguments(x,M,K,w,N,bias,out)) return hipErrorInvalidValue;
    tiled<<<dim3(unsigned((N+63)/64),unsigned((M+31)/32)),128,0,stream>>>(
        static_cast<const __half*>(x),M,K,static_cast<const __half*>(w),N,bias,out);
    return hipGetLastError();
}
} // namespace vcp_linear_f16
#endif
