// Standalone paired DSO: compile INSTEAD OF packed64_runtime.hip.
#include "../packed128.hpp"
extern "C" int nrqi_packed_linear64(void*g,const void*x,int64_t m,int64_t k,const void*w,int type,int64_t n,void*y,int odt,int64_t stride){return nrqi_packed64::launch(g,x,m,k,w,type,n,y,odt,stride);}
extern "C" int nrqi_packed_linear128(void*g,const void*x,int64_t m,int64_t k,const void*w,int type,int64_t n,void*y,int odt,int64_t stride){return nrqi_packed128::launch(g,x,m,k,w,type,n,y,odt,stride);}
