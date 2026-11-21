// Compile ONLY this TU with strict floating-point flags; existing TUs unchanged.
#ifndef NRQV_ORDERED_STRICT_TU
#error "Build qv.hip and this TU with NRQV_ORDERED_STRICT_TU=1"
#endif
#include "qv.h"
#include "rms_rope_ordered_strict.hpp"
extern "C" int nrqv_rms_rope_ordered_f32(void*ctx,float*x,int64_t rows,int64_t heads,int64_t stride,const float*w,const float*cs){
 return qv_ordered_strict::launch(ctx,x,rows,heads,stride,w,cs);
}
