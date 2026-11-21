// Combined-model ABI closure. Frozen operator implementation stays unchanged.
#include "packed_expanded_runtime.hip"
extern "C" int nrqi_packed_linear(void*g,const void*x,int64_t m,int64_t k,const void*w,int t,int64_t n,void*y,int o,int64_t s){
    return nrqi_packed::launch(g,x,m,k,w,t,n,y,o,s);
}
