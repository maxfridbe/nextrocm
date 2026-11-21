// Compile only this new TU with -ffp-contract=off -fno-unsafe-math-optimizations -fno-approx-func.
#include "qv.h"
#include "rms4096_strict.hpp"
extern "C" int nrqv_rms4096_strict_f32(void*ctx,const float*x,int64_t rows,const float*w,float*out){return qv_rms4096::launch(ctx,x,rows,w,out);}
