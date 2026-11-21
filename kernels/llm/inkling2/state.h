// Private single-token F32 state operators. Caller owns all device storage.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
// x[C], weight[C,4] oldest->current, optional dynamic[heads,4] lag0=current.
// history[C,3] oldest->newest is updated with RAW x after computing out[C].
// C<=8192, heads divides C; all write spans disjoint from every other span.
// A failed launch poisons the caller's session; no transactional rollback.
int ni2_shortconv(ns_gpu*, const float* x, const float* weight,
                  const float* dynamic, int heads, int channels,
                  float* history, float* out);
// Q[24,128], R[24,16], rel_weight[1024,16], sink[24].
// KV[capacity,8,128] is a ring: absolute key position % capacity. Host appends
// current normalized K/V before calling, on this stream. Q/R already include
// the full-attention tau; this operator does not apply it again.
// window=0 (full, capacity>position) or1024 (local, capacity>=min(pos+1,1024)).
// 0<=position<=INT32_MAX, 1<=capacity<=65536. Output[24,128].
// scratch_count is floats, >=24*min(position+1,window?window:position+1).
// Scratch/output disjoint from each other and all inputs, no allocation.
// Direct HF sink logit (zero sink value); not dense GGML sink-subtraction order.
int ni2_attention(ns_gpu*, const float* q, const float* k, const float* v,
                  const float* relative, const float* rel_weight, const float* sink,
                  int64_t position, int capacity, int window, float* out,
                  float* scratch, int64_t scratch_count);
#ifdef __cplusplus
}
#endif
