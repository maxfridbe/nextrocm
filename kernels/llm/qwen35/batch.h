// Private Qwen35 packed-weight/F32 small-batch ABI. No activation quantization.
#pragma once
#include "ns.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Y[input, output] = sum_i W[output, i] * X[input, i].
// ninput: 1..64. Types: F32=0, F16=1, Q8_0=8, Q4_K=12, Q5_K=13, Q6_K=14.
// Weight stride is BYTES; X/Y strides are FLOAT ELEMENTS. Positive strides
// must cover each logical row; packed weight rows must start on type alignment.
// Caller owns device buffers, validates their allocated extents and keeps them
// live until stream completion. Y must not overlap X or W. Required extents:
// W: (rows-1)*w_stride + packed_row_bytes; X: (ninput-1)*x_stride + cols
// floats; Y: (ninput-1)*y_stride + rows floats. Padding is not read/written.
// Enqueues on g's work stream, no allocations or synchronization. Existing
// nq35_mv remains the caller's independent numerical/performance fallback.
int nq35_mv_batch(ns_gpu* g, const void* w, int type, int rows, int cols,
                 uint64_t w_stride, const float* x, int ninput,
                 uint64_t x_stride, float* y, uint64_t y_stride);
#ifdef __cplusplus
}
#endif
