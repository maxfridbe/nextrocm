// Private Inkling quantized row decode and scalar F32 matvec. No allocations.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
// Supported GGUF codes: F32=0,Q8_0=8,Q5_K=13,BF16=30,NVFP4=40.
// rows<=201024,cols<=8192, rows*cols<=512Mi elements. Complete quant blocks per row.
// Weight stride is bytes; output/input strides below are FLOAT ELEMENTS.
// All pointers device resident, read/write spans disjoint; read/read alias allowed.
// Decode does NOT apply external expert scales. Padding remains untouched.
int ni2_quant_decode(ns_gpu*, const void* weights, int type, int rows, int cols,
                     int64_t weight_stride_bytes, float* out, int64_t out_stride);
// One vector / one expert matrix. Ascending-K explicit F32 FMA sum, then separate
// F32 multiply by *post_scale if nonnull. Caller selects the expert's scale.
// This is a F32 scalar baseline, NOT a claim of original backend BF16/MMQ parity.
// Caller supplies finite operands/scales; IEEE exceptional results propagate.
int ni2_quant_matvec(ns_gpu*, const void* weights, int type, int rows, int cols,
                     int64_t weight_stride_bytes, const float* x, int64_t x_stride,
                     const float* post_scale, float* out, int64_t out_stride);
#ifdef __cplusplus
}
#endif
