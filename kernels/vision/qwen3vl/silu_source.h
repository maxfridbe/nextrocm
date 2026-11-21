#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Numeric IEEE special values are outputs, not rejected inputs. Host geometry
// rejection is synchronous/no-write. No status buffer or hidden allocations.
int nrqv_silu_source_f32(void*ctx,const float*x,int64_t rows,int64_t cols,float*out);
int nrqv_silu_source_capture(void*ctx,const float*x,int64_t rows,int64_t cols,float*out,float*records);
#ifdef __cplusplus
}
#endif
