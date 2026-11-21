#include "../attention_wide.hpp"
extern "C" int nrqi_image_attention_wide(void*g,const void*q,const void*k,const void*v,
 int64_t sq,int64_t sk,int64_t h,int64_t d,int64_t qs,int64_t ks,void*out,void*scratch,int64_t bytes){
 return nrqi_attention_wide::launch(g,q,k,v,sq,sk,h,d,qs,ks,out,scratch,bytes);
}
