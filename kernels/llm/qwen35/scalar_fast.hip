// Opt-in experiments: specialize type dispatch, replace only the final five
// shared reduction stages with the identical wave32 shuffle tree, or both.
// Weight decoding is the existing quants.hpp expression, not requantization.
#include "scalar_fast.h"
#include "quants.hpp"
#include "ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <climits>
#include <cstdint>

namespace {
struct HalfReader {
    __device__ float operator()(const uint8_t* p) const {
        return __half2float(*reinterpret_cast<const __half*>(p));
    }
};
__device__ float dynamic_value(const void* w, int type, int i) {
    if (type == 0) return static_cast<const float*>(w)[i];
    if (type == 1) return __half2float(static_cast<const __half*>(w)[i]);
    return nq35::quant(w, type, i, HalfReader{});
}
template<int Type>
__device__ float fixed_value(const void* w, int i) {
    if constexpr (Type == 0) return static_cast<const float*>(w)[i];
    else if constexpr (Type == 1) return __half2float(static_cast<const __half*>(w)[i]);
    else return nq35::quant(w, Type, i, HalfReader{});
}

template<bool Shuffle>
__device__ float reduce(float v) {
    __shared__ float tmp[256];
    const int t = threadIdx.x;
    tmp[t] = v;
    __syncthreads();
    if constexpr (Shuffle) {
        if (warpSize == 32) {
            // Preserve cross-wave additions BEFORE the within-wave tail:
            // reducing each wave independently first would change the tree.
            for (int k = 128; k >= 32; k /= 2) {
                if (t < k) tmp[t] += tmp[t + k];
                __syncthreads();
            }
            if (t < 32) {
                float s = tmp[t];
                for (int k = 16; k; k /= 2) {
                    // All 32 lanes participate; only the same active lanes as
                    // the shared tree update. No barrier follows this branch.
                    const float other = __shfl_down(s, k, 32);
                    if (t < k) s += other;
                }
                return s;
            }
            return 0; // Only thread zero's result is consumed.
        }
    }
    // Original shared tree, including its final read/broadcast barrier.
    for (int k = 128; k; k /= 2) {
        if (t < k) tmp[t] += tmp[t + k];
        __syncthreads();
    }
    const float out = tmp[0];
    __syncthreads();
    return out;
}

template<int Type, bool Shuffle>
__global__ void fast_mv(const void* w, int type, int cols, const float* x, float* y) {
    const int row = blockIdx.x;
    float v = 0;
    for (int i = threadIdx.x; i < cols; i += 256) {
        if constexpr (Type < 0) v += dynamic_value(w, type, row * cols + i) * x[i];
        else v += fixed_value<Type>(w, row * cols + i) * x[i];
    }
    const float s = reduce<Shuffle>(v);
    if (threadIdx.x == 0) y[row] = s;
}

template<bool Shuffle>
void specialized(ns_gpu* g, const void* w, int type, int rows, int cols,
                 const float* x, float* y) {
    // Type validated before dispatch; no default launch or silent fallback.
    switch (type) {
        case 0: fast_mv<0, Shuffle><<<rows, 256, 0, g->q>>>(w, type, cols, x, y); break;
        case 1: fast_mv<1, Shuffle><<<rows, 256, 0, g->q>>>(w, type, cols, x, y); break;
        case 8: fast_mv<8, Shuffle><<<rows, 256, 0, g->q>>>(w, type, cols, x, y); break;
        case 12: fast_mv<12, Shuffle><<<rows, 256, 0, g->q>>>(w, type, cols, x, y); break;
        case 13: fast_mv<13, Shuffle><<<rows, 256, 0, g->q>>>(w, type, cols, x, y); break;
        case 14: fast_mv<14, Shuffle><<<rows, 256, 0, g->q>>>(w, type, cols, x, y); break;
    }
}
}

extern "C" int nq35_mv_fast(ns_gpu* g, const void* w, int type, int rows,
                             int cols, const float* x, float* y, int variant) {
    if (!g || !w || !x || !y || rows <= 0 || cols <= 0 || cols > INT_MAX - 255 ||
        int64_t(rows) * cols > INT_MAX || variant < 1 || variant > 3)
        return ns_fail("qwen35 scalar fast: invalid shape/pointer/variant");
    const int64_t elems = int64_t(rows) * cols;
    int alignment = 2;
    switch (type) {
        case 0: alignment = 4; break;
        case 1: break;
        case 8:
            // Scalar reader's block-byte offset is a signed int.
            if (cols % 32 || (elems / 32) * 34 > INT_MAX)
                return ns_fail("qwen35 scalar fast: invalid Q8_0 geometry");
            break;
        case 12: case 13: case 14:
            if (cols % 256) return ns_fail("qwen35 scalar fast: invalid K-quant geometry");
            break;
        default: return ns_fail("qwen35 scalar fast: unsupported weight type");
    }
    if (reinterpret_cast<uintptr_t>(w) % alignment ||
        reinterpret_cast<uintptr_t>(x) % alignof(float) ||
        reinterpret_cast<uintptr_t>(y) % alignof(float))
        return ns_fail("qwen35 scalar fast: unaligned buffer");
    NS_CU(hipSetDevice(g->dev));
    if (variant == NQ35_MV_SPECIALIZE) specialized<false>(g, w, type, rows, cols, x, y);
    else if (variant == NQ35_MV_SHUFFLE_TAIL)
        fast_mv<-1, true><<<rows, 256, 0, g->q>>>(w, type, cols, x, y);
    else specialized<true>(g, w, type, rows, cols, x, y);
    NS_LAUNCH_CHECK();
    return 0;
}
