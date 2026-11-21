/* pocr.h: the PaddleOCR-VL engine's kernels (image/paddleocrvl) - its own C ABI, bound by its crate by name
 * (image/paddleocrvl/src/ffi.rs). Every buffer is device memory of g; the work is queued on g's queue (the caller
 * syncs). Each returns 0, or -1 and ns_last_error(). */
#ifndef NS_IMAGE_POCR_H
#define NS_IMAGE_POCR_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ns_gpu ns_gpu;

/* Optional default-off vision-only F32 candidate, rows<=5120, D72, stride3HD. */
int ns_image_pocr_attention_tiled(ns_gpu*, const float*, const float*, const float*, int64_t rows, int64_t heads, int64_t dim, int64_t stride, float*);

/* Decoder-only F16 matrices -> F32; eight F32 partial sums, balanced reduction. */
int ns_image_pocr_linear(ns_gpu* g, const void* x, int64_t m, int64_t k,
    const void* weight, int64_t n, float* out);

int ns_image_pocr_dequant(ns_gpu* g, const void* src, int type, int64_t n, void* out, int out_type);
int ns_image_pocr_rope(ns_gpu* g, float* x, int64_t rows, int64_t heads, int64_t dim, int64_t stride, const float* cs);
/* Last argument bit 0: causal; bit 1: normalize F32 probabilities before P@V. */
int ns_image_pocr_attention(ns_gpu* g, const float* q, const float* k, const float* v,
    int64_t rows, int64_t heads, int64_t kv_heads, int64_t dim, int64_t qs, int64_t ks, float* out, int causal);
/* float32 -> half, n values */
int ns_image_pocr_to_half(ns_gpu* g, const float* x, void* out, int64_t n);
/* dst [rows][..] (row stride ds) = src [rows][..] (row stride ss), cols values a row */
int ns_image_pocr_copy_rows(ns_gpu* g, const float* src, int64_t ss, float* dst, int64_t ds, int64_t rows, int64_t cols);
/* The projector's 2 x 2 merge: x [H][W][C] (H, W even) -> out [H/2][W/2][4 C], a merged row the patches (2y, 2x),
 * (2y, 2x + 1), (2y + 1, 2x), (2y + 1, 2x + 1) side by side */
int ns_image_pocr_merge2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out);
/* One query row over n cached rows: q [Hq][D], k / v [n][Hkv * D] (row stride kvs), head h reading kv head
 * h / (Hq / Hkv); out [Hq][D] = softmax(q.k / sqrt(D)) v. Split over chunks of 256 keys (flash decoding); scratch
 * float32 of ns_image_pocr_attn1_scratch(n, Hq, D) values. D a multiple of 16, at most 256; Hq / Hkv at most 16 */
int64_t ns_image_pocr_attn1_scratch(int64_t n, int64_t Hq, int64_t D);
int ns_image_pocr_attn1(ns_gpu* g, const float* q, const float* k, const float* v, int64_t n, int64_t kvs, int64_t Hq, int64_t Hkv,
                        int64_t D, float* scratch, float* out);

#ifdef __cplusplus
}
#endif
#endif
