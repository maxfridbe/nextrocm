// Compile this standalone TU with -ffp-contract=off -fno-fast-math; no LTO.
#include "softmax_source32.h"
#include "softmax_source32.hpp"
extern "C" int nrqv_softmax_source32(void*c,const float*x,int64_t n,int64_t w,float*y,uint32_t*s){return qv_softmax_source32::launch(c,x,n,w,y,s);}
