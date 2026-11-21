// CPU-only numerical hypothesis from retained mathematical behavior observations.
// No proprietary implementation source/instructions copied. Not production code.
// Contract limited to S32/H32/KV8/D128, N32 and even K128/32.
#pragma once
#include <cmath>
#include <cstddef>
namespace qkv_order {
inline float add(float a,float b){volatile float v=a+b;return v;}
inline float mul(float a,float b){volatile float v=a*b;return v;}
template<class A,class B> inline float dot(unsigned query,unsigned k,A a,B b){
 float sum=0.f;
 if(query<30){for(unsigned i=0;i<k;++i)sum=std::fmaf(a(i),b(i),sum);}
 else {
  float even=0.f,odd=0.f;
  for(unsigned i=0;i<k;i+=2){even=add(even,mul(a(i),b(i)));odd=add(odd,mul(a(i+1),b(i+1)));}
  sum=add(even,odd);
 }
 return mul(sum,1.f); // observed alpha1/beta0; no previous output read
}
inline void qk(const float*x,float*out){
 for(unsigned h=0;h<32;++h)for(unsigned q=0;q<32;++q)for(unsigned key=0;key<32;++key)
  out[(h*32+q)*32+key]=dot(q,128,[&](unsigned d){return x[q*6144+h*128+d];},[&](unsigned d){return x[key*6144+4096+(h/4)*128+d];});
}
inline void pv(const float*x,const float*p,float*out){
 for(unsigned q=0;q<32;++q)for(unsigned h=0;h<32;++h)for(unsigned d=0;d<128;++d)
  out[(q*32+h)*128+d]=dot(q,32,[&](unsigned key){return p[(h*32+q)*32+key];},[&](unsigned key){return x[key*6144+5120+(h/4)*128+d];});
}
}
