// Qwen35-private scalar Q5_K/Q8_0 readers, after nextsycl common/quants.hpp TQ5K.
// Reuse established Q4_K/Q6_K readers without changing the StableLM family.
#pragma once
#include "../stablelm/quants.hpp"
#ifdef __HIPCC__
#define NQ35_DEVICE __device__
#else
#define NQ35_DEVICE
#endif
namespace nq35 {
template<class Half> NQ35_DEVICE inline float quant(const void* w, int type, int i, Half half) {
    if (type == 8) {
        // 32 signed values / 34 bytes: f16 d followed by int8 quants.
        const uint8_t* b = static_cast<const uint8_t*>(w) + (i / 32) * 34;
        return half(b) * reinterpret_cast<const int8_t*>(b + 2)[i % 32];
    }
    if (type != 13) return nslm::quant(w, type, i, half);
    // 256 values / 176 bytes: f16 d,dmin; 12 packed scale/min bytes;
    // 32 high-bit planes (one bit per 32-value group); 128 low-nibble bytes.
    const uint8_t* b = static_cast<const uint8_t*>(w) + (i / 256) * 176;
    const int group = (i % 256) / 32, lane = i % 32;
    const uint8_t* sc = b + 4;
    const int scale = group < 4 ? sc[group] & 63
        : (sc[group+4] & 15) | ((sc[group-4] >> 6) << 4);
    const int min = group < 4 ? sc[group+4] & 63
        : (sc[group+4] >> 4) | ((sc[group] >> 6) << 4);
    const int low = (b[48 + (group/2)*32 + lane] >> (4*(group%2))) & 15;
    const int high = ((b[16 + lane] >> group) & 1) << 4;
    return half(b) * scale * (low | high) - half(b+2) * min;
}
}
#undef NQ35_DEVICE
