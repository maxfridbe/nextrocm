#include "../trajectory.hpp"
extern "C" int nrqi_axpy_f32(void* g,float* x,const float* velocity,int64_t n,float step) {
    return nrqi_trajectory::launch(g,x,velocity,n,step);
}
