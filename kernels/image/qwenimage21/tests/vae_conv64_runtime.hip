// Standalone candidate+original control. Compile INSTEAD OF vae.hip.
#include "../vae.hip"
#include "../vae_conv64.hpp"
extern "C" int nrqi_vae_conv64(void*g,const void*x,int64_t h,int64_t w,int64_t ci,const void*wt,int64_t co,int k,int s,const float*b,void*out){return nrqi_vae64::launch(g,x,h,w,ci,wt,co,k,s,b,out);}
