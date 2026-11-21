#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Standalone diagnostics: output99 floats/row; caller-owned status reset every valid call.
// status1 invalid logit row, status2 arithmetic-domain failure; affected row output untouched.
// Check stream completion/status before consuming any output. Width must be32, rows1..4096.
int nrqv_softmax_source32(void*ctx,const float*logits,int64_t rows,int64_t width,float*records,uint32_t*status);
#ifdef __cplusplus
}
#endif
