// Private F32 Inkling depth/router/loop primitives. Caller owns all buffers.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
// One token: raw states[count,width], queries[nq,width], output[nq,width].
// Each state RMS-normalized (eps1e-6), scaled QK softmax, normalized states as V.
// count1..5, nq1..4, width1..8192. Caller supplies model's exact depth ordering.
// No output/read overlap; no allocation; finite inputs expected.
int ni2_depthmix(ns_gpu*,const float* states,const float* queries,int count,int nq,int width,float* out);
// One token, exactly128 experts/top8: input logits128, correction_bias128.
// IDs I32 descending biased sigmoid score; ties lower ID (merged GGML contract).
// Weights use UNBIASED sigmoid, divide(sum+1e-20), multiply8, matching ID order.
// All output spans disjoint from reads and each other. No projection included.
int ni2_router(ns_gpu*,const float* logits,const float* correction_bias,int32_t* ids,float* weights);
// Projected gain/forget logits: (1-tanh(forget))*current+tanh(gain)*message.
// n1..16Mi floats. Exact output/input alias allowed, partial overlap rejected.
// Controller RMS/concat/down/gain/forget projections belong to caller.
int ni2_loop_control(ns_gpu*,const float* current,const float* message,const float* gain,const float* forget,int n,float* out);
// Same vector bounds/alias rules. Separate RN operations, no fused axpy.
int ni2_scale(ns_gpu*,const float* x,int n,float factor,float* out);
int ni2_add(ns_gpu*,const float* a,const float* b,int n,float* out);
// out = out + x * (*device_weight). Weight may NOT overlap output.
int ni2_axpy(ns_gpu*,const float* x,const float* device_weight,int n,float* out);
#ifdef __cplusplus
}
#endif
