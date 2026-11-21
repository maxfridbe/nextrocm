/* AUTHORITATIVE private DiT ABI. Do not infer names from crossed chat proposals.
 * F16 is IEEE binary16. Work uses the supplied ns_gpu stream. Caller synchronizes.
 * Returns0 or negative with ns_last_error. No shared Nsd binder changes. */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int nrqi_ln_mod(void* g,const float* x,int64_t M,int64_t C,float eps,const float* scale,void* f16out);
int nrqi_rms_rope(void* g,void* f16x,int64_t M,int64_t H,int64_t D,int64_t stride,const float* weight,float eps,const float* cs,int rot_dim);
int nrqi_text_rms_f32(void* g,const float* x,int64_t M,int64_t C,const float* weight,float eps,float* out);
int nrqi_gate_add_f32_f16(void* g,float* x,const void* f16other,int64_t M,int64_t C,const float* gate);
#ifdef __cplusplus
}
#endif
