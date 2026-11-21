/* rb.h: the background removers' own kernels (kernels/image/rembg), beside the shared vision kernels (vision.h) and
 * the diffusion kernels' GEMMs. Bound by name by image/rembg/src/ffi.rs. Pictures are float32 channels-last [H, W, C];
 * every call is queued on g's queue (the caller syncs) and returns 0, or -1 with the reason in ns_last_error().
 * Dimensions are positive; products/launch grids must fit. Pool/resize require
 * nonoverlapping input/output; resize out_stride >= C (padding is left untouched).
 * Bias may be NULL; relu is 0 or 1. Bias/sigmoid update x in place. */
#ifndef NS_IMAGE_RB_H
#define NS_IMAGE_RB_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ns_gpu ns_gpu;
/* Private opt-in F32 X[M,K] * W[N,K]^T -> out[M,N], no bias.
 * Output must not overlap either input. Increasing K, padded K16 FMAs, +0 epilogue. */
int ns_image_rb_linear_tiled(ns_gpu* g, const float* x, const float* w,
                            int64_t M, int64_t K, int64_t N, float* out);
/* BiRefNet variant with optional bias[N], added after reduction. Nonoverlapping output. */
int ns_image_rb_biref_linear_tiled(ns_gpu* g, const float* x, const float* w,
                                  int64_t M, int64_t K, int64_t N,
                                  const float* bias, float* out);
/* max-pool 2 x 2, stride 2, ceil mode (a last odd row or column pooled alone): x [H, W, C] -> out [ceil(H/2), ceil(W/2), C] */
int ns_image_rb_maxpool2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out);
/* bilinear resize, half-pixel centres (align_corners false: PyTorch's interpolate, ONNX's half_pixel), the source
 * coordinate clamped at the edges: x [Hi, Wi, C] -> out [Ho, Wo, C], out's rows `out_stride` values apart */
int ns_image_rb_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t out_stride);
/* x += bias[c] over [M, C]; then ReLU when relu */
int ns_image_rb_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int relu);
/* x = 1 / (1 + exp(-x)) over n values */
int ns_image_rb_sigmoid(ns_gpu* g, float* x, int64_t n);
#ifdef __cplusplus
}
#endif
#endif
