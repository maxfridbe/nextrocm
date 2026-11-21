// Private ABI v1. Matrices are packed GGUF row-major; float32 activations.
// KV: [token][kv_head][channel] fp16. All buffers owned by caller.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
int nq3m_embed(ns_gpu* g,const void* w,int t,int n,float* out);
int nq3m_mv(ns_gpu* g,const void* w,int t,int rows,int cols,const float* x,float* y);
int nq3m_norm(ns_gpu* g,const float* x,const float* w,int n,float eps,float* out);
int nq3m_axpy(ns_gpu* g,float* x,const float* y,int n,float scale);
int nq3m_swiglu(ns_gpu* g,float* x,const float* y,int n);
int nq3m_qk(ns_gpu* g,const float* x,const float* w,int heads,int d,int stride,int rot,int pos,float theta,float eps,float* y);
int nq3m_attn(ns_gpu* g,const float* q,const float* k,const float* v,void* kc,void* vc,int heads,int kh,int d,int pos,float* out);
#ifdef __cplusplus
}
#endif
