#pragma once
#include "ns.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Private candidate; same finite-input/shape/stride/span contract as quant.h.
// Decoder unchanged. Lane-strided FMA + fixed subgroup tree changes dot order.
// Requires actual hardware wave32. No allocation, no baseline dispatch change.
int ni2_quant_matvec_wave32(ns_gpu*,const void*,int,int,int,int64_t,
                           const float*,int64_t,const float*,float*,int64_t);
#ifdef __cplusplus
}
#endif
