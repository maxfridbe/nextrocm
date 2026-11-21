#pragma once
#include "../../ns/ns.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Contiguous F16 x[M,K], F16 w[N,K], optional F32 bias[N], F16 out[M,N].
 * Private opt-in only. The checked header owns arithmetic and geometry. */
int ns_audio_whisper_linear_f16_tiled(ns_gpu* g, const void* x, int64_t M,
    int64_t K, const void* w, int64_t N, const float* bias, void* out);
/* Noncausal contiguous F16 Q/K/V and output [S,H,64], no global scratch. */
int ns_audio_whisper_attention_tiled(ns_gpu* g, const void* q, const void* k,
    const void* v, int64_t S, int64_t H, int64_t D, int64_t stride, void* out);
#ifdef __cplusplus
}
#endif
