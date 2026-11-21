// Strict new private TU only. Do not also link qv_softmax_source32.o (included header kernel).
#include "attention_source32.h"
#include "attention_source32.hpp"
extern "C" int nrqv_attention_source32_f32(void*c,const float*q,int64_t n,float*y,uint32_t*s){return qv_attention_source32::launch(c,q,n,y,nullptr,s);}
extern "C" int nrqv_attention_source32_capture(void*c,const float*q,int64_t n,float*y,float*r,uint32_t*s){if(!r)return ns_fail("capture records required");return qv_attention_source32::launch(c,q,n,y,r,s);}
