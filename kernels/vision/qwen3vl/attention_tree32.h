#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
int nrqv_attention_tree32_f32(void*ctx,const float*qkv,int64_t rows,float*out);
#ifdef __cplusplus
}
#endif
