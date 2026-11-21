// Strict new private TU; frozen softmax helper has a TU-specific namespace.
// Existing source32 TU may coexist; no standalone softmax object is needed.
#include "attention_order32.h"
#include "attention_order32.hpp"
extern "C" int nrqv_attention_order32_f32(void*c,const float*q,int64_t n,float*y,uint32_t*s){return qv_attention_order32::launch(c,q,n,y,nullptr,s);}
extern "C" int nrqv_attention_order32_capture(void*c,const float*q,int64_t n,float*y,float*r,uint32_t*s){if(!r)return ns_fail("capture records required");return qv_attention_order32::launch(c,q,n,y,r,s);}
