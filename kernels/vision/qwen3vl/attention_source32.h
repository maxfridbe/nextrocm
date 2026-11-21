#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Private opt-in exact32 attention; caller synchronizes/checks mandatory status before consuming output.
int nrqv_attention_source32_f32(void*ctx,const float*qkv,int64_t rows,float*out,uint32_t*status);
// Diagnostic only: 163 floats/query/head: rawQK0..31, scaled32..63, softmax99 at64.
int nrqv_attention_source32_capture(void*ctx,const float*qkv,int64_t rows,float*out,float*records,uint32_t*status);
#ifdef __cplusplus
}
#endif
