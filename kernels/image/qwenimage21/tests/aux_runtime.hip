// Private validation DSO only; no normal build integration.
#include "../aux_f32.hpp"
extern "C" int nrqi_aux_f32_linear(void* g,const float* x,int64_t m,int64_t k,const void* w,
                                  int type,int64_t n,float* y,int64_t stride) {
    return nrqi_aux_f32::launch(g,x,m,k,w,type,n,y,stride);
}
