#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Single image NHWC F16, OIHW F16 weights, F32 bias. k1/3; stride1 same or stride2 valid.
int nrqi_vae_conv(void*,const void*,int64_t,int64_t,int64_t,const void*,int64_t,int,int,const float*,void*);
int nrqi_vae_silu(void*,void*,int64_t);
int nrqi_vae_softmax(void*,const float*,int64_t,int64_t,float,void*);
int nrqi_vae_up2(void*,const void*,int64_t,int64_t,int64_t,void*);
int nrqi_vae_dupup_add(void*,const void*,int64_t,int64_t,int64_t,int64_t,int,void*);
int nrqi_vae_pad_br(void*,const void*,int64_t,int64_t,int64_t,void*);
int nrqi_vae_avg_down_add(void*,const void*,int64_t,int64_t,int64_t,int64_t,int,void*);
int nrqi_vae_rgba8(void*,const void*,int64_t,int64_t,uint8_t*);
#ifdef __cplusplus
}
#endif
