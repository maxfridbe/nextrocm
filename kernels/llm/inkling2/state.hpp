// Host-only guards shared by the private HIP implementation and CPU tests.
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
namespace ni2_state {
struct Span { uintptr_t begin, end; };
inline bool span(const void* p, uint64_t floats, Span& s) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if (!p || a % alignof(float) || floats > (std::numeric_limits<uintptr_t>::max()-a)/sizeof(float)) return false;
    s={a,a+uintptr_t(floats*sizeof(float))}; return true;
}
inline bool apart(Span a,Span b) { return a.end<=b.begin || b.end<=a.begin; }
inline bool conv_args(const float* x,const float* w,const float* d,int heads,int c,float* h,float* o) {
    if(c<1||c>8192||heads<1||heads>c||c%heads) return false;
    Span reads[3],history,out;
    if(!span(x,c,reads[0])||!span(w,uint64_t(c)*4,reads[1])||
       !span(h,uint64_t(c)*3,history)||!span(o,c,out)||!apart(history,out)) return false;
    int n=2;
    if(d) { if(!span(d,uint64_t(heads)*4,reads[2])) return false; n=3; }
    for(int i=0;i<n;++i) if(!apart(reads[i],history)||!apart(reads[i],out)) return false;
    return true;
}
inline int key_count(int64_t pos,int cap,int window) {
    if(pos<0||pos>INT32_MAX||cap<1||cap>65536||(window!=0&&window!=1024)) return 0;
    const int64_t n=window&&pos>=window ? window : pos+1;
    return n<=cap ? int(n) : 0;
}
inline bool attention_args(const float* q,const float* k,const float* v,const float* r,
 const float* w,const float* sink,int64_t pos,int cap,int window,float* out,float* scratch,int64_t count) {
    int n=key_count(pos,cap,window);
    if(!n||count<int64_t(24)*n||count>int64_t(24)*65536) return false;
    Span reads[6],o,s;
    if(!span(q,24*128,reads[0])||!span(k,uint64_t(cap)*8*128,reads[1])||
       !span(v,uint64_t(cap)*8*128,reads[2])||!span(r,24*16,reads[3])||
       !span(w,1024*16,reads[4])||!span(sink,24,reads[5])||
       !span(out,24*128,o)||!span(scratch,uint64_t(count),s)||!apart(o,s)) return false;
    for(auto input:reads) if(!apart(input,o)||!apart(input,s)) return false;
    return true;
}
}
