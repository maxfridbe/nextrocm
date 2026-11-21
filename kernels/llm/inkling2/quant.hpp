#pragma once
#include <cstdint>
#include <cstring>
#include <limits>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define NI2_HD __host__ __device__
#else
#define NI2_HD
#endif
namespace ni2_quant {
NI2_HD inline float bits(uint32_t u) { float f; memcpy(&f,&u,4); return f; }
NI2_HD inline uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1])<<8; }
NI2_HD inline float half(const uint8_t* p) {
    uint32_t h=u16(p), sign=(h&32768)<<16, e=(h>>10)&31, m=h&1023;
    if(e==0) {
        if(!m) return bits(sign);
        int shift=0; while((m&1024)==0) {m<<=1;++shift;}
        return bits(sign|uint32_t(113-shift)<<23|(m&1023)<<13);
    }
    return bits(sign | (e==31?255:e+112)<<23 | m<<13);
}
NI2_HD inline float value(const uint8_t* row,int type,int i) {
    if(type==0) { const auto* p=row+size_t(i)*4; return bits(uint32_t(p[0])|uint32_t(p[1])<<8|uint32_t(p[2])<<16|uint32_t(p[3])<<24); }
    if(type==30) return bits(uint32_t(u16(row+size_t(i)*2))<<16);
    if(type==8) { const auto* b=row+(i/32)*34; const unsigned q=b[2+i%32]; return half(b)*float(q<128?int(q):int(q)-256); }
    if(type==40) {
        const auto* b=row+(i/64)*36; const unsigned lane=i%64, g=lane/16, at=lane%16;
        const unsigned s=b[g], e=(s>>3)&15, m=s&7;
        const float d=(s==0||s==127)?0.0f:float(e?8+m:m)*bits(uint32_t((e?int(e)-11:-10)+127)<<23);
        constexpr int v[16]={0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
        const unsigned q=b[4+g*8+at%8]; return float(v[at<8?q&15:q>>4])*d;
    }
    // Q5_K source dequantization: d*scale*q - dmin*min; preserve operation order.
    const auto* b=row+(i/256)*176; const int g=(i%256)/32,l=i%32; const auto* s=b+4;
    const int scale=g<4?s[g]&63:(s[g+4]&15)|((s[g-4]>>6)<<4);
    const int min=g<4?s[g+4]&63:(s[g+4]>>4)|((s[g]>>6)<<4);
    const int q=((b[48+(g/2)*32+l]>>(4*(g%2)))&15)|(((b[16+l]>>g)&1)<<4);
    const float d=half(b)*float(scale), dm=half(b+2)*float(min);
    return d*float(q)-dm;
}
inline int row_bytes(int type,int cols) {
    int block=1,size=0;
    switch(type) {case 0:size=4;break;case 30:size=2;break;case 8:block=32;size=34;break;case 13:block=256;size=176;break;case 40:block=64;size=36;break;default:return 0;}
    return cols>0 && cols<=8192 && cols%block==0?cols/block*size:0;
}
struct Span {uintptr_t start,end;};
inline bool span(const void* p,uint64_t bytes,unsigned alignment,Span& s) {
    s.start=reinterpret_cast<uintptr_t>(p);
    if(!p||s.start%alignment||!bytes||bytes>4000000000ULL||bytes>std::numeric_limits<uintptr_t>::max()-s.start)return false;
    s.end=s.start+bytes;return true;
}
inline bool disjoint(Span a,Span b) {return a.end<=b.start||b.end<=a.start;}
inline bool matrix(const void* w,int type,int rows,int cols,int64_t stride,Span& s) {
    int bytes=row_bytes(type,cols);
    if(rows<=0||rows>201024||!bytes||int64_t(rows)*cols>512*1024*1024||stride<bytes||stride>4000000000LL)return false;
    return span(w,uint64_t(rows-1)*uint64_t(stride)+bytes,1,s);
}
inline bool floats(const float* p,int count,int64_t stride,int width,Span& s) {
    if(count<=0||width<=0||stride<width||stride>1000000000LL)return false;
    return span(p,(uint64_t(count-1)*uint64_t(stride)+unsigned(width))*4,4,s);
}
}
#undef NI2_HD
