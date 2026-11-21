// Standalone test DSO only. Compile with ns/ns.hip + ops/speech.hip.
#include "../packed_linear.hpp"
extern "C" int nrqi_packed_linear(void* g,const void* x,int64_t m,int64_t k,
                                 const void* w,int qtype,int64_t n,void* y,int odt,int64_t stride) {
    return nrqi_packed::launch(g,x,m,k,w,qtype,n,y,odt,stride);
}
