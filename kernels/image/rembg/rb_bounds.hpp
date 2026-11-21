// Host-side bounds for the private float32 HWC ABI; no HIP dependency.
#pragma once
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace rb_bounds {
inline bool product(std::initializer_list<int64_t> dims, int64_t& out) {
    out = 1;
    for (int64_t d : dims) {
        if (d <= 0 || out > (INT64_MAX / 4) / d) return false;
        out *= d;
    }
    return true;
}
inline bool grid(int64_t n) {
    return n > 0 && (n / 256 + (n % 256 != 0)) <= INT32_MAX;
}
inline bool linear(int64_t m, int64_t k, int64_t n) {
    int64_t x, w, y;
    return product({m,k},x) && product({n,k},w) && product({m,n},y) &&
           (m/64 + (m%64 != 0)) <= 65535 &&
           (n/64 + (n%64 != 0)) <= INT32_MAX;
}
inline bool pool(int64_t h, int64_t w, int64_t c) {
    int64_t input, output;
    return product({h,w,c},input) &&
           product({h/2+h%2,w/2+w%2,c},output) && grid(output);
}
inline bool resize(int64_t hi,int64_t wi,int64_t c,int64_t ho,int64_t wo,int64_t stride) {
    int64_t input, output, storage;
    // Coordinate interpolation uses float32 then signed integer conversion.
    return hi <= INT32_MAX && wi <= INT32_MAX && ho <= INT32_MAX && wo <= INT32_MAX &&
           stride >= c && product({hi,wi,c},input) &&
           product({ho,wo,c},output) && product({ho,wo,stride},storage) && grid(output);
}
inline bool bias(int64_t m,int64_t c,int relu) {
    int64_t n;
    return (relu == 0 || relu == 1) && product({m,c},n) && grid(n);
}
inline bool sigmoid(int64_t n) {
    int64_t bytes;
    return product({n},bytes) && grid(n);
}
} // namespace rb_bounds
