// Default HIP flags: preserve frozen probability-first exp/reduction/PV.
#include "qv.h"
#include "attention_divide.hpp"
extern "C" int nrqv_attention_divide_f32(void*ctx,const float*qkv,int64_t rows,float*out){return qv_divide::launch(ctx,qkv,rows,out);}
