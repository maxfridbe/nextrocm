/* Model-private SD1.5 ABI. All pointers are device pointers on g. Calls enqueue on
 * g->q, return 0 or ns_fail (-1; ns_last_error). Caller owns allocations and must
 * keep them alive until stream completion. No scratch allocations or fallback.
 * All activations/output f32. Packed weights independently F32=0, IEEE F16=1.
 * Sizes/strides count ELEMENTS, never bytes. No overlapping output/input except
 * exact in-place activation/add/multiply/time_emb/norm. Positive dimensions required. */
#ifndef NRSD_H
#define NRSD_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ns_gpu ns_gpu;
typedef struct { const void* data; int32_t dtype; } nrsd_weight;
typedef struct { int64_t rows, input, output, x_stride, out_stride; } nrsd_linear_shape;
typedef struct {
 int64_t batch, ci, height, width, co, kh, kw, sh, sw, ph, pw;
 int64_t oh, ow; /* must equal floor((input+2*pad-kernel)/stride)+1 */
} nrsd_conv_shape;
/* D is contiguous. Other strides support [B,T,H,D] or [B,H,T,D], including
 * separate Q/K/V projections and slices of interleaved projections. Output
 * has its own strides. Causal means key index <= query index (no KV cache). */
typedef struct {
 int64_t batch, queries, keys, heads, dim;
 int64_t qb, qt, qh, kb, kt, kh, vb, vt, vh, ob, ot, oh;
 float scale; int32_t causal;
} nrsd_attention_shape;
int nrsd_linear(ns_gpu*, const float*, nrsd_weight, nrsd_weight, float*, nrsd_linear_shape);
/* x/out NCHW contiguous; weight [co,ci,kh,kw]; optional bias [co]. */
int nrsd_conv2d(ns_gpu*, const float*, nrsd_weight, nrsd_weight, float*, nrsd_conv_shape);
/* GN NCHW, groups divides C (SD1.5: 32); affine [C], optional independently.
 * LN contiguous [rows,C]; affine [C]. Population variance, epsilon > 0. */
int nrsd_group_norm(ns_gpu*, const float*, nrsd_weight, nrsd_weight, float*, int64_t batch, int64_t channels, int64_t height, int64_t width, int64_t groups, float eps);
int nrsd_layer_norm(ns_gpu*, const float*, nrsd_weight, nrsd_weight, float*, int64_t rows, int64_t channels, float eps);
/* kind: 0 SiLU, 1 exact GELU(erf), 2 QuickGELU(1.702), 3 tanh GELU. */
int nrsd_activation(ns_gpu*, const float*, float*, int64_t count, int32_t kind);
/* out = alpha*x + beta*y; y NULL means scale x. */
int nrsd_add(ns_gpu*, const float*, const float*, float*, int64_t count, float alpha, float beta);
/* Elementwise out = x*y; exact in-place permitted. */
int nrsd_multiply(ns_gpu*, const float*, const float*, float*, int64_t count);
/* [rows,2C] -> [rows,C]: value first, tanh GELU gate second (pinned development-host ggml). No alias. */
int nrsd_geglu(ns_gpu*, const float*, float*, int64_t rows, int64_t channels);
/* out[B,C,H,W] = x + embedding[B,C] (already projected/activated by caller). */
int nrsd_time_emb(ns_gpu*, const float*, const float*, float*, int64_t batch, int64_t channels, int64_t height, int64_t width);
/* to_nhwc=1 NCHW->NHWC, 0 inverse. */
int nrsd_layout(ns_gpu*, const float*, float*, int64_t batch, int64_t channels, int64_t height, int64_t width, int32_t to_nhwc);
int nrsd_up2(ns_gpu*, const float*, float*, int64_t batch, int64_t channels, int64_t height, int64_t width);
/* contiguous NCHW channel concatenation, a then b. */
int nrsd_concat(ns_gpu*, const float*, const float*, float*, int64_t batch, int64_t ca, int64_t cb, int64_t height, int64_t width);
/* Stable softmax; D <= 512; bounded shared storage per block (~24.1 KiB),
 * no global scores/probabilities. Scale is explicit, typically 1/sqrt(D). */
int nrsd_attention(ns_gpu*, const float*, const float*, const float*, float*, nrsd_attention_shape);
#ifdef __cplusplus
}
#endif
#endif
