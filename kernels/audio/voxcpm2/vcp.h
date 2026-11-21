/* vcp.h: the VoxCPM2 engine's own kernels (its language models' decode shares the MiniMax Music engine's mm3 kernels):
 * rotary positions from a frequency table (MiniCPM4's long RoPE), attention within short groups of rows (the local
 * encoder's and the local DiT's sequences: every row sees every row of its group), the AudioVAE's Snake and dilated
 * causal depthwise convolution, a per-channel scale and bias, the scalar quantizer. Float32 on the device; 0 or -1
 * (the reason: ns_last_error). */
#pragma once
#include "ns.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Rotate-half rotary positions on rows of H heads x D (xs floats apart): row r at position p0 + r % S, pair (i, i + D/2)
 * turned by position * inv_freq[i] */
int ns_audio_vcp_rope(ns_gpu* g, float* x, int64_t xs, int64_t rows, int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0);
/* Attention within B groups of S rows (none masked): q rows of Hq heads x D qs floats apart, k / v rows of Hkv heads
 * kvs apart (query head h reads key head h / (Hq / Hkv)); out [B * S, Hq x D]; S <= 64, D <= 128 */
int ns_audio_vcp_local_attn(ns_gpu* g, const float* q, const float* k, const float* v, int64_t B, int64_t S, int64_t Hq, int64_t Hkv, int64_t D,
                            int64_t qs, int64_t kvs, float* out);
/* Snake on x [C, L]: x + sin^2(alpha x) / (alpha + 1e-9) */
int ns_audio_vcp_snake(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* alpha, float* out);
/* A causal depthwise convolution of x [C, L], dilation dil ((K - 1) * dil zeros ahead): w [C, K], bias [C] or NULL */
int ns_audio_vcp_dwconv(ns_gpu* g, const float* x, int64_t C, int64_t L, const float* w, const float* bias, int64_t K, int64_t dil, float* out);
/* x [C, L] = x * scale[c] + bias[c] */
int ns_audio_vcp_affine(ns_gpu* g, float* x, int64_t C, int64_t L, const float* scale, const float* bias);
/* The scalar quantizer's middle: x = rint(tanh(x) * s) / s over n, ties-to-even; s finite and positive */
int ns_audio_vcp_fsq(ns_gpu* g, float* x, int64_t n, float s);
/* Opt-in F32 transpose convolution: w[Ci,Co,K], out[B,Co,L*stride],
 * K=2*stride, stride in {2,5,6,8}; no scratch. Errors never imply fallback. */
int ns_audio_vcp_conv_up_tiled(ns_gpu* g, const float* x, int64_t B,
    int64_t Ci, int64_t L, const float* w, int64_t Co, int64_t K,
    const float* bias, int64_t stride, float* out);
/* Private F32-KV precision control. Same layouts, scratch and arithmetic as
 * mm3 cached attention, but kc/vc store F32; D=128, Hq/Hkv in {2,4,8}. */
int ns_audio_vcp_kv_store_f32(ns_gpu* g, const float* k, const float* v,
    int64_t ks, int64_t B, int64_t S, int64_t Hkv, int64_t D,
    int64_t T, int64_t p0, void* kc, void* vc);
int ns_audio_vcp_attn_f32(ns_gpu* g, const float* q, int64_t qs,
    const void* kc, const void* vc, int64_t B, int64_t S, int64_t Hq,
    int64_t Hkv, int64_t D, int64_t T, int64_t p0, float* out, float* part);

int ns_audio_vcp_rope_bf16_cache(ns_gpu* g, float* x, int64_t xs, int64_t rows,
    int64_t S, int64_t H, int64_t D, const float* inv_freq, int64_t p0);
int ns_audio_vcp_linear_f16_tiled(ns_gpu* g, const void* x, int64_t M,
    int64_t K, const void* w, int64_t N, const float* bias, float* out);

#ifdef __cplusplus
}
#endif
