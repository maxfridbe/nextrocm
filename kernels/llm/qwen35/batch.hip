// Small-batch weight reuse follows nextcuda q35_mmq.cu/MMVQ and nextsycl
// qwen35/silo/mmvq.cpp's multi-column products. Their Q8_1 activation/DP4A
// arithmetic is deliberately not used: this path retains nq35_mv's F32 inputs,
// per-lane i+=256 accumulation and 128..1 shared-memory reduction tree.
#include "batch.h"
#include "quants.hpp"
#include "ns_internal.hpp"
#include <hip/hip_fp16.h>
#include <cstddef>
#include <cstdint>
#include <climits>

namespace {
struct HalfReader {
    __device__ float operator()(const uint8_t* p) const {
        return __half2float(*reinterpret_cast<const __half*>(p));
    }
};
__device__ float value(const void* row, int type, int i) {
    if (type == 0) return static_cast<const float*>(row)[i];
    if (type == 1) return __half2float(static_cast<const __half*>(row)[i]);
    return nq35::quant(row, type, i, HalfReader{});
}

template<int B>
__global__ void batch_mv(const void* w, int type, int cols, uint64_t ws,
                         const float* x, int ninput, uint64_t xs,
                         float* y, uint64_t ys) {
    const int lane = threadIdx.x;
    const int first = blockIdx.y * B;
    const auto* row = static_cast<const uint8_t*>(w) + uint64_t(blockIdx.x) * ws;
    float acc[B] = {};
    // One packed-weight read/dequantization serves all B input rows.
    for (int i = lane; i < cols; i += 256) {
        const float wi = value(row, type, i);
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            if (first + b < ninput)
                acc[b] += wi * x[uint64_t(first + b) * xs + i];
        }
    }
    __shared__ float tmp[B][256];
    #pragma unroll
    for (int b = 0; b < B; ++b) tmp[b][lane] = acc[b];
    __syncthreads();
    for (int k = 128; k; k /= 2) {
        if (lane < k) {
            #pragma unroll
            for (int b = 0; b < B; ++b) tmp[b][lane] += tmp[b][lane + k];
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            if (first + b < ninput)
                y[uint64_t(first + b) * ys + blockIdx.x] = tmp[b][0];
        }
    }
}

// All pointer offsets fit ptrdiff_t, including the complete final row.
bool span_ok(uint64_t count, uint64_t stride, uint64_t width, uint64_t unit) {
    const uint64_t limit = uint64_t(PTRDIFF_MAX) / unit;
    return width <= limit && stride >= width &&
           (count <= 1 || stride <= (limit - width) / (count - 1));
}
}

extern "C" int nq35_mv_batch(ns_gpu* g, const void* w, int type, int rows,
                              int cols, uint64_t ws, const float* x, int ninput,
                              uint64_t xs, float* y, uint64_t ys) {
    if (!g || !w || !x || !y || rows <= 0 || cols <= 0 ||
        cols > INT_MAX - 255 || ninput < 1 || ninput > 64)
        return ns_fail("qwen35 batch: invalid pointer/shape");
    uint64_t row_bytes = 0, alignment = 2;
    switch (type) {
        case 0: row_bytes = uint64_t(cols) * 4; alignment = 4; break;
        case 1: row_bytes = uint64_t(cols) * 2; break;
        case 8:
            if (cols % 32) return ns_fail("qwen35 batch: Q8_0 row alignment");
            row_bytes = uint64_t(cols / 32) * 34; break;
        case 12: case 13: case 14:
            if (cols % 256) return ns_fail("qwen35 batch: K-quant row alignment");
            row_bytes = uint64_t(cols / 256) * (type == 12 ? 144 : type == 13 ? 176 : 210);
            break;
        default: return ns_fail("qwen35 batch: unsupported weight type");
    }
    // Existing scalar quant readers use signed-int packed-block byte offsets.
    if ((type >= 8 && row_bytes > INT_MAX) ||
        reinterpret_cast<uintptr_t>(w) % alignment || ws % alignment ||
        reinterpret_cast<uintptr_t>(x) % alignof(float) ||
        reinterpret_cast<uintptr_t>(y) % alignof(float) ||
        !span_ok(rows, ws, row_bytes, 1) ||
        !span_ok(ninput, xs, cols, sizeof(float)) ||
        !span_ok(ninput, ys, rows, sizeof(float)))
        return ns_fail("qwen35 batch: invalid stride/alignment/address span");
    NS_CU(hipSetDevice(g->dev));
    if (ninput == 1) {
        batch_mv<1><<<dim3(rows, 1), 256, 0, g->q>>>(w, type, cols, ws, x, ninput, xs, y, ys);
    } else {
        batch_mv<4><<<dim3(rows, (ninput + 3) / 4), 256, 0, g->q>>>(w, type, cols, ws, x, ninput, xs, y, ys);
    }
    NS_LAUNCH_CHECK();
    return 0;
}
