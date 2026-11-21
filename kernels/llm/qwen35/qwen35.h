// Private ABI v1. Matrices are packed GGUF row-major; float32 activations.
// KV: [token][kv_head][channel] fp16. Recurrent: [head][value][key] f32.
// Conv history: [oldest..newest][channel] f32. All buffers owned by caller.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
int nq35_embed(ns_gpu* g,const void* w,int t,int n,float* out);
int nq35_mv(ns_gpu* g,const void* w,int t,int rows,int cols,const float* x,float* y);
int nq35_norm(ns_gpu* g,const float* x,const float* w,int n,float eps,float* out);
int nq35_axpy(ns_gpu* g,float* x,const float* y,int n,float scale);
int nq35_swiglu(ns_gpu* g,float* x,const float* y,int n);
int nq35_qk(ns_gpu* g,const float* x,const float* w,int heads,int d,int stride,int rot,int pos,float theta,float eps,float* y);
int nq35_attn(ns_gpu* g,const float* q,const float* k,const float* v,void* kc,void* vc,int heads,int kh,int d,int pos,float* out);
int nq35_gate(ns_gpu* g,float* x,const float* q,int heads,int d);
int nq35_conv(ns_gpu* g,const float* x,const float* w,float* history,int channels,int taps,float* y);
int nq35_delta(ns_gpu* g,const float* co,const float* alpha,const float* beta,const float* dt,const float* a,float* state,int hk,int hv,int d,float eps,float* y);
int nq35_gated_norm(ns_gpu* g,const float* x,const float* z,const float* w,int heads,int d,float eps,float* out);
#ifdef __cplusplus
}
#endif
