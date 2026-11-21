#include "../convert.hpp"
extern "C" int nrqi_f32_to_f16(void*g,const float*x,int64_t n,void*out){return nrqi_cast::down_launch(g,x,n,out);}
extern "C" int nrqi_f16_to_f32(void*g,const void*x,int64_t n,float*out){return nrqi_cast::up_launch(g,x,n,out);}
