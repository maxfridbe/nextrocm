// Private, opt-in scalar matvec canaries. Original nq35_mv is unchanged.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
enum nq35_mv_fast_variant {
    NQ35_MV_SPECIALIZE = 1,
    NQ35_MV_SHUFFLE_TAIL = 2,
    NQ35_MV_COMBINED = 3
};
// Contiguous W[rows,cols], X[cols], Y[rows]; F32 activations/output.
// Types: 0 F32, 1 F16, 8 Q8_0, 12 Q4_K, 13 Q5_K, 14 Q6_K.
// Quantized cols must be block-aligned (32 for Q8_0, 256 for K).
// Caller validates allocated extents, nonaliasing Y against W/X, device
// ownership and lifetime through stream completion. No allocations or sync.
// Same i+=256 accumulation and addition tree as nq35_mv; identical compiler
// floating-point flags are required. Exact parity must pass GPU canaries before
// adoption. Shuffle variants use shared-memory fallback when warpSize != 32.
int nq35_mv_fast(ns_gpu* g, const void* w, int type, int rows, int cols,
                const float* x, float* y, int variant);
#ifdef __cplusplus
}
#endif
