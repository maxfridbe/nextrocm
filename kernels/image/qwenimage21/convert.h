#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Native IEEE casts: RNE to F16; overflow to Inf; signed zero/Inf preserved.
// NaN category is preserved; payload/sign of NaNs is not contractual.
int nrqi_f32_to_f16(void* g,const float* x,int64_t count,void* out);
int nrqi_f16_to_f32(void* g,const void* x,int64_t count,float* out);
#ifdef __cplusplus
}
#endif
