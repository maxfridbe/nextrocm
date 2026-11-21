// Private opt-in StarVector FFN candidate. Not included by a production TU.
// Same ascending K fmaf chain, post-dot float bias add and F16 rounding as
// speech.hip linear_kernel. No reduction tree, split K, global scratch or alloc.
#pragma once
#include "../../ns/ns.h"
#include "../../ns/ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>

namespace nrsv_ffn_small_m {
using I = int64_t;
constexpr int columns = 64;
constexpr int depth = 32;
constexpr int threads = 256;

// Load each weight row along contiguous K, transpose into padded LDS, then
// compute adjacent output columns without the baseline's 16-row arithmetic.
// Padding to 65 columns avoids the transpose's 32-bank stride conflict.
// Threads own one column and one/two rows: no inter-thread dot reduction.
template<int M, int FixedK = 0, int FixedN = 0>
__global__ void kernel(const __half* x, const __half* w, I dynamic_k,
                       I dynamic_n, const float* bias, __half* out) {
    const I k = FixedK ? FixedK : dynamic_k;
    const I n = FixedN ? FixedN : dynamic_n;
    const int t = int(threadIdx.x);
    const int col = t % columns;
    const int row = t / columns;
    const I first = I(blockIdx.x) * columns;
    __shared__ float weight[depth][columns + 1];
    __shared__ float input[M][depth];
    constexpr int owned = (M + 3) / 4;
    float sum[owned] = {};
    // Baseline includes zero-padded products through ceil(K/16)*16. Retain
    // exactly that extent even when the final 32-wide tile contains 16 lanes.
    const I rounded_k = ((k + 15) / 16) * 16;
    for (I base = 0; base < k; base += depth) {
        for (int a = t; a < columns * depth; a += threads) {
            const int c = a / depth, j = a % depth;
            weight[j][c] = (first + c < n && base + j < k)
                ? __half2float(w[(first + c) * k + base + j]) : 0.0f;
        }
        for (int a = t; a < M * depth; a += threads) {
            const int r = a / depth, j = a % depth;
            input[r][j] = base + j < k ? __half2float(x[I(r) * k + base + j]) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < owned; ++r) {
            if (row + 4 * r < M && first + col < n) {
                #pragma unroll
                for (int j = 0; j < depth; ++j) {
                    if (FixedK || base + j < rounded_k)
                        sum[r] = fmaf(input[row + 4 * r][j], weight[j][col], sum[r]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < owned; ++r) {
        if (row + 4 * r < M && first + col < n)
            out[I(row + 4 * r) * n + first + col] =
                __float2half_rn(sum[r] + (bias ? bias[first + col] : 0.0f));
    }
}

inline bool supported(I m, I k, I n, int dt, int odt) {
    return (m == 1 || m == 2 || m == 8) && dt == 1 && odt == 1 &&
        ((k == 2048 && n == 8192) || (k == 8192 && n == 2048));
}
struct Span { uintptr_t first = 0, end = 0; };
inline bool span(const void* p, I count, size_t element, Span& result) {
    const uintptr_t first = reinterpret_cast<uintptr_t>(p);
    if (!p || count <= 0 || first % element ||
        uint64_t(count) > UINTPTR_MAX / element) return false;
    const uintptr_t bytes = uintptr_t(count) * element;
    if (bytes > UINTPTR_MAX - first) return false;
    result = {first, first + bytes};
    return true;
}
inline bool overlaps(const Span& a, const Span& b) {
    return a.first < b.end && b.first < a.end;
}
inline bool valid(void* ctx, const void* x, const void* w, const float* bias,
                  void* out, I m, I k, I n, int dt, int odt) {
    if (!ctx || (dt != 0 && dt != 1) || (odt != 0 && odt != 1) ||
        m <= 0 || k <= 0 || n <= 0 ||
        m > INT64_MAX / 4 / k || n > INT64_MAX / 4 / k ||
        m > INT64_MAX / 4 / n) return false;
    Span xs, ws, ys, bs;
    if (!span(x, m * k, dt ? 2 : 4, xs) ||
        !span(w, n * k, dt ? 2 : 4, ws) ||
        !span(out, m * n, odt ? 2 : 4, ys) ||
        overlaps(ys, xs) || overlaps(ys, ws)) return false;
    return !bias || (span(bias, n, sizeof(float), bs) && !overlaps(ys, bs));
}

template<int M>
inline int dispatch(ns_gpu* g, const void* x, I k, const void* w,
                    I n, const float* bias, void* out, bool geometry_test) {
    const auto* hx = static_cast<const __half*>(x);
    const auto* hw = static_cast<const __half*>(w);
    auto* ho = static_cast<__half*>(out);
    if (geometry_test) {
        hipLaunchKernelGGL((kernel<M>), dim3((n + columns - 1) / columns),
                          dim3(threads), 0, g->q, hx, hw, k, n, bias, ho);
    } else if (k == 2048) {
        hipLaunchKernelGGL((kernel<M, 2048, 8192>), dim3(8192 / columns),
                          dim3(threads), 0, g->q, hx, hw, k, n, bias, ho);
    } else {
        hipLaunchKernelGGL((kernel<M, 8192, 2048>), dim3(2048 / columns),
                          dim3(threads), 0, g->q, hx, hw, k, n, bias, ho);
    }
    NS_LAUNCH_CHECK();
    return 0;
}

// 0 = launched, 1 = unsupported valid shape/type (caller uses baseline),
// -1 = invalid arguments or launch error. Unsupported dispatch never writes.
inline int try_launch(void* ctx, const void* x, int dt, I m, I k,
                      const void* w, I n, const float* bias, void* out, int odt) {
    if (!valid(ctx, x, w, bias, out, m, k, n, dt, odt))
        return ns_fail("nrsv FFN: invalid arguments/overflow");
    if (!supported(m, k, n, dt, odt)) return 1;
    auto* g = static_cast<ns_gpu*>(ctx);
    NS_CU(hipSetDevice(g->dev));
    if (m == 1) return dispatch<1>(g, x, k, w, n, bias, out, false);
    if (m == 2) return dispatch<2>(g, x, k, w, n, bias, out, false);
    return dispatch<8>(g, x, k, w, n, bias, out, false);
}

#ifdef NR_STARVECTOR_FFN_GEOMETRY_TEST
// Validator-only generic geometry; never part of exact-shape production gate.
inline int geometry_test(void* ctx, const void* x, I m, I k,
                         const void* w, I n, const float* bias, void* out) {
    if (!valid(ctx, x, w, bias, out, m, k, n, 1, 1) ||
        !(m == 1 || m == 2 || m == 8) || k > 8192 || n > 8192)
        return ns_fail("nrsv FFN test: invalid bounded geometry");
    auto* g = static_cast<ns_gpu*>(ctx);
    NS_CU(hipSetDevice(g->dev));
    if (m == 1) return dispatch<1>(g, x, k, w, n, bias, out, true);
    if (m == 2) return dispatch<2>(g, x, k, w, n, bias, out, true);
    return dispatch<8>(g, x, k, w, n, bias, out, true);
}
#endif
} // namespace nrsv_ffn_small_m
