#include "silu_source.h"
#include "silu_source.hpp"
extern "C" int nrqv_silu_source_f32(void*c,const float*x,int64_t r,int64_t n,float*y){return qv_silu_source::launch(c,x,r,n,y,nullptr);}
extern "C" int nrqv_silu_source_capture(void*c,const float*x,int64_t r,int64_t n,float*y,float*z){if(!z)return ns_fail("nrqv_silu_source_capture: null records");return qv_silu_source::launch(c,x,r,n,y,z);}
