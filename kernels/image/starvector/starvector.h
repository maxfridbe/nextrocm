// Private StarVector attention ABI. F16/F32 inputs; output dt explicit for CLIP.
// Causal queries are LOCAL rows [0,rows); row0 offsets the mask only. K/V are
// cached rows [0,row0+rows). No backwards/out-of-allocation query pointers.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int nrsv_attention(void* gpu,const void* q,const void* k,const void* v,int dt,
 int64_t rows,int64_t heads,int64_t dim,int64_t stride,void* out,int out_dt);
int nrsv_attention_causal_rows(void* gpu,const void* q,const void* k,const void* v,int dt,
 int64_t row0,int64_t rows,int64_t hq,int64_t hkv,int64_t dim,int64_t qs,int64_t kvs,void* out);
// Private in-place QuickGELU for the opt-in F32 CLIP branch; no scratch.
int nrsv_quick_gelu_f32(void* gpu,float* x,int64_t count);
// Optional ABI, exported only with -DNR_STARVECTOR_FFN_SMALL_M=1.
// 0 launched, 1 unsupported (caller baseline fallback), negative error.
int nrsv_ffn_small_m_apply(void* gpu,const void* x,int dt,int64_t m,int64_t k,
 const void* w,int64_t n,const float* bias,void* out,int out_dt);
#ifdef __cplusplus
}
#endif
