// Private family ABI, following nextcuda's ns_gpu/DevBuf dispatch pattern.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
// GGUF types: F32=0, F16=1, Q4_K=12, Q6_K=14; row-major matrices.
int nslm_embed(ns_gpu*, const void*, int type, int width, int token, float*);
int nslm_mv(ns_gpu*, const void*, int type, int rows, int cols, const float*, float*);
int nslm_norm(ns_gpu*, const float*, const float* weight, const float* bias, int width, float eps, float*);
int nslm_add(ns_gpu*, float*, const float*, int width);
int nslm_swiglu(ns_gpu*, float* gate, const float* up, int width);
// NeoX split-half RoPE on the first rot dimensions of each head, in place.
int nslm_rope(ns_gpu*, float*, int heads, int head, int rot, int pos, float theta);
// cache is [token, head, channel] f16; writes the new k/v at pos then attends [0,pos].
int nslm_attn(ns_gpu*, const float* q, const float* k, const float* v,
              void* kc, void* vc, int heads, int head, int pos, float* out);
#ifdef __cplusplus
}
#endif
