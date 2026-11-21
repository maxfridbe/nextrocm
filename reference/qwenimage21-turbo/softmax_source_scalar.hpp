// CPU reference validation only; not production inference.
// Exponential arithmetic derived from SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68.
// Copyright Naoki Shibata and contributors 2010 - 2024.
// Distributed under the Boost Software License, Version 1.0, retained in
// te-softmax-complete-spec-v1/sleef-source/LICENSE.txt.
// Reduction semantics derived from pinned ATen; copyright/license retained in
// te-softmax-complete-spec-v1/sleef-source/Torch-LICENSE.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace source_softmax {
inline float from_bits(uint32_t b) { float x; std::memcpy(&x,&b,4); return x; }
inline uint32_t bits(float x) { uint32_t b; std::memcpy(&b,&x,4); return b; }
// Explicit F32 materializations disallow incidental algebraic contraction.
inline float add(float a,float b) { volatile float x=a+b; return x; }
inline float sub(float a,float b) { volatile float x=a-b; return x; }
inline float mul(float a,float b) { volatile float x=a*b; return x; }
inline float div(float a,float b) { volatile float x=a/b; return x; }
inline float exp_nonpositive(float d) {
    if (std::isnan(d)||d>0) throw std::runtime_error("exp domain: nonpositive finite or -Inf");
    // The source bit-clears these results; avoid undefined scalar float->int conversion.
    if(d < -104.0f) return 0.0f;
    const float z=mul(d,from_bits(0x3fb8aa3bu));
    const int q=static_cast<int>(std::lrintf(z)); // FE_TONEAREST checked by driver.
    float s=std::fmaf(static_cast<float>(q),-from_bits(0x3f317200u),d);
    s=std::fmaf(static_cast<float>(q),-from_bits(0x35bfbe8eu),s);
    float u=from_bits(0x39502bdau);
    for(uint32_t c: {0x3ab696c9u,0x3c0888a6u,0x3d2aaa7au,0x3e2aaaabu,0x3f000000u})
        u=std::fmaf(u,s,from_bits(c));
    u=add(1.0f,std::fmaf(mul(s,s),u,s));
    const int a=q>=0?q/2:-((-q+1)/2), b=q-a;
    if(a < -126 || a>127 || b < -126 || b>127) throw std::runtime_error("bounded exp exponent");
    return mul(mul(u,from_bits(uint32_t(a+127)<<23)),from_bits(uint32_t(b+127)<<23));
}
// AVX MAXPS chooses its second operand for equal (including +/-0) values.
// NaNs are rejected before this finite/-Inf-only reduction.
inline float maximum(float a,float b) {return a>b?a:b;}
template<class Op> inline float reduce32(const float* x,Op op) {
    std::array<float,8> a{};
    for(unsigned i=0;i<8;++i) a[i]=x[i];
    for(unsigned j=8;j<32;j+=8) for(unsigned i=0;i<8;++i) a[i]=op(a[i],x[j+i]);
    for(unsigned step: {4u,2u,1u}) {auto prior=a;for(unsigned i=0;i<8;++i)a[i]=op(prior[i],prior[i^step]);}
    return a[0];
}
inline void validate32(const float* x) {
    bool finite=false;
    for(unsigned i=0;i<32;++i){if(std::isnan(x[i])||x[i]==INFINITY)throw std::runtime_error("invalid logit");finite|=std::isfinite(x[i]);}
    if(!finite)throw std::runtime_error("all masked row");
}
struct Row {float maximum,sum,inverse;std::array<float,32> shifted,ex,p;};
inline Row row(const float* x) {
    validate32(x);Row r{};r.maximum=reduce32(x,maximum);
    for(unsigned i=0;i<32;++i){r.shifted[i]=sub(x[i],r.maximum);r.ex[i]=exp_nonpositive(r.shifted[i]);}
    r.sum=reduce32(r.ex.data(),add);r.inverse=div(1.0f,r.sum);
    if(!std::isfinite(r.sum)||r.sum<=0)throw std::runtime_error("invalid denominator");
    for(unsigned i=0;i<32;++i)r.p[i]=mul(r.ex[i],r.inverse);
    return r;
}
}
