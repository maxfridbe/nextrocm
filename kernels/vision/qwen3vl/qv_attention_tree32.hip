// Default HIP flags, matching frozen attention; inspect sum IR for reassociation.
#include "attention_tree32.h"
#include "attention_tree32.hpp"
extern "C" int nrqv_attention_tree32_f32(void*ctx,const float*qkv,int64_t rows,float*out){return qv_tree32::launch(ctx,qkv,rows,out);}
