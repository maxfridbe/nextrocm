#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Version1 caller-owned scratch ABI. All buffers live through stream completion.
// status must be read after synchronization before consuming any output.
typedef struct nrqv_int8_args {
 const void* x; const int8_t* weight; const float* weight_scale; const void* bias;
 void* out; float* rotated; int8_t* quantized; float* row_scale; int32_t* accum;
 uint32_t* status;
 int64_t m,n,k,scales; int dtype;
} nrqv_int8_args;
int nrqv_int8_convrot_linear(void* gpu,const nrqv_int8_args* args);
// In-place F32 per-head RMS + split-half RoPE, fixedD128, heads8/32.
int nrqv_rms_rope_f32(void* gpu,float* x,int64_t rows,int64_t heads,int64_t stride,
                     const float* weight,const float* cos_sin);
// Separate opt-in source-derived F32 head normalization; baseline geometry.
int nrqv_rms_rope_ordered_f32(void* gpu,float* x,int64_t rows,int64_t heads,int64_t stride,
                             const float* weight,const float* cos_sin);
// Experimental opt-in causal F32 GQA, interleaved QKV[rows,6144], out[rows,4096].
int nrqv_attention_prob_f32(void* gpu,const float* qkv,int64_t rows,float* out);
// Independent opt-ins: source D4096 RMS (eps1e-6), original attention scaling.
int nrqv_rms4096_strict_f32(void* gpu,const float* x,int64_t rows,const float* weight,float* out);
int nrqv_attention_divide_f32(void* gpu,const float* qkv,int64_t rows,float* out);
#ifdef __cplusplus
}
#endif
