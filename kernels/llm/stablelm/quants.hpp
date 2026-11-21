// Scalar adaptation of nextsycl kernels/llm/common/quants.hpp TQ4K/TQ6K.
// Shared verbatim by the HIP product and the host format test.
#pragma once
#include <cstdint>
#ifdef __HIPCC__
#define NSLM_DEVICE __device__
#else
#define NSLM_DEVICE
#endif
namespace nslm {
template<class Half> NSLM_DEVICE inline float quant(const void* w, int type, int i, Half half) {
    if (type == 12) {
        const uint8_t* b = static_cast<const uint8_t*>(w) + (i / 256) * 144;
        const int j = (i % 256) / 32, l = i % 32;
        const uint8_t* sc = b + 4;
        const int s = j < 4 ? sc[j] & 63 : (sc[j+4] & 15) | ((sc[j-4] >> 6) << 4);
        const int m = j < 4 ? sc[j+4] & 63 : (sc[j+4] >> 4) | ((sc[j] >> 6) << 4);
        const int q = (b[16 + (j/2)*32 + l] >> (4*(j%2))) & 15;
        return half(b) * s * q - half(b+2) * m;
    }
    const uint8_t* b = static_cast<const uint8_t*>(w) + (i / 256) * 210;
    const int n = (i%256)/128, j = (i%128)/32, l = i%32;
    const int q = ((b[64*n + 32*(j%2) + l] >> (4*(j/2))) & 15)
                | (((b[128+32*n+l] >> (2*j)) & 3) << 4);
    const int s = reinterpret_cast<const int8_t*>(b+192)[8*n+2*j+l/16];
    return half(b+208) * s * (q-32);
}
}
#undef NSLM_DEVICE
