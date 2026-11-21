// Private operator DSO: compile with kernels/ns/ns.hip. Never registers a model.
#include "../dit_ops.hpp"
extern "C" int nrqi_ln_mod(void*g,const float*x,int64_t m,int64_t c,float eps,const float*s,void*out){return nrqi_dit::ln(g,x,m,c,eps,s,out);}
extern "C" int nrqi_rms_rope(void*g,void*x,int64_t m,int64_t h,int64_t d,int64_t stride,const float*w,float eps,const float*cs,int rot){return nrqi_dit::rope(g,x,m,h,d,stride,w,eps,cs,rot);}
extern "C" int nrqi_text_rms_f32(void*g,const float*x,int64_t m,int64_t c,const float*w,float eps,float*out){return nrqi_dit::rms(g,x,m,c,w,eps,out);}
extern "C" int nrqi_gate_add_f32_f16(void*g,float*x,const void*y,int64_t m,int64_t c,const float*gate){return nrqi_dit::gate(g,x,y,m,c,gate);}
