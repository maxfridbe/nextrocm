// Standalone validation DSO only; no production registration.
#include "../attention_1024.hpp"
extern "C" int nrqi_image_attention_1024(void* g,const void* q,const void* k,const void* v,
    int64_t sq,int64_t sk,int64_t h,int64_t d,int64_t qs,int64_t ks,void* out) {
    return nrqi_attention_1024::launch(g,q,k,v,sq,sk,h,d,qs,ks,out);
}
