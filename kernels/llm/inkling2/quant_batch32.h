#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct ns_gpu;
// ws bytes; all x/y strides F32 elements. One shared optional physical-expert scale.
// B=1..4; caller owns buffers until stream completion. No allocation or cache.
int ni2_quant_matvec_batch32(struct ns_gpu*,const void*,int,int,int,int64_t,
    const float*,int64_t,int64_t,const float*,float*,int64_t,int64_t,int);
#ifdef __cplusplus
}
#endif
