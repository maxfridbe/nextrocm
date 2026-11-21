// Replaces packed128_runtime.hip, preserving baseline exports.
#include "../packed_expanded.hpp"
extern "C" int nrqi_packed_linear64(void*g,const void*x,int64_t m,int64_t k,const void*w,int t,int64_t n,void*y,int o,int64_t s){return nrqi_packed64::launch(g,x,m,k,w,t,n,y,o,s);}
extern "C" int nrqi_packed_linear128(void*g,const void*x,int64_t m,int64_t k,const void*w,int t,int64_t n,void*y,int o,int64_t s){return nrqi_packed128::launch(g,x,m,k,w,t,n,y,o,s);}
extern "C" int nrqi_packed_linear_expanded(void*g,const void*x,int64_t m,int64_t k,const void*w,int t,int64_t n,void*y,int o,int64_t s,void*scratch,int64_t bytes){return nrqi_expanded::launch(g,x,m,k,w,t,n,y,o,s,scratch,bytes);}
