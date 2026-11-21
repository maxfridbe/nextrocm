// CPU-only two-sided SiLU source proof; independent of frozen softmax helper.
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
#include <limits>
namespace source_silu {
inline float from_bits(uint32_t b) { float x; std::memcpy(&x,&b,4); return x; }
inline uint32_t bits(float x) { uint32_t b; std::memcpy(&b,&x,4); return b; }
// Explicit F32 materializations disallow incidental algebraic contraction.
inline float add(float a,float b) { volatile float x=a+b; return x; }
inline float sub(float a,float b) { volatile float x=a-b; return x; }
inline float mul(float a,float b) { volatile float x=a*b; return x; }
inline float div(float a,float b) { volatile float x=a/b; return x; }
inline float exp_two_sided(float d) {
    // NaN payload/sign are not part of the reference contract; only category.
    if (std::isnan(d)) return std::numeric_limits<float>::quiet_NaN();
    if (d > 100.0f) return INFINITY;
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
struct Value {float ex, denominator, silu, product;};
inline Value value(float gate,float up) {
    Value v{};
    v.ex=exp_two_sided(-gate);
    v.denominator=add(1.0f,v.ex);
    v.silu=div(gate,v.denominator);
    v.product=mul(v.silu,up);
    return v;
}
} // namespace source_silu
