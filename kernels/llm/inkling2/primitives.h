// Private float32 Inkling primitives. ns owns the device/stream and allocations.
#pragma once
#include "ns.h"
#ifdef __cplusplus
extern "C" {
#endif
// Contiguous [rows,width], optional shared [width] gamma; exact in-place x/out allowed.
int ni2_rms(ns_gpu*, const float* x, const float* gamma, int rows, int width, float eps, float* out);
// Contiguous [rows,width], optional [rows,width] scale applied BEFORE L2 norm.
int ni2_clip(ns_gpu*, const float* x, const float* scale, int rows, int width, float threshold, float* out);
// gate and up are separate contiguous vectors; beta=4, linear_beta=25 in Mini.
int ni2_situ(ns_gpu*, const float* gate, const float* up, int n, float beta, float linear_beta, float* out);
// y = x*sigmoid(gate); and y = scale*tanh(x/scale), respectively.
int ni2_gate(ns_gpu*, const float* x, const float* gate, int n, float* out);
int ni2_softcap(ns_gpu*, const float* x, int n, float scale, float* out);
#ifdef __cplusplus
}
#endif
