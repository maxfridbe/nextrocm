#pragma once
#include <cstdint>
// Each private span remains below the runtime's global allocation ceiling.
constexpr int64_t POCR_MAX_ELEMENTS = 1000000000;
inline bool pocr_product(int64_t a, int64_t b) {
    return a >= 0 && b > 0 && a <= POCR_MAX_ELEMENTS / b;
}
inline bool pocr_span(int64_t rows, int64_t stride, int64_t cols) {
    return rows >= 0 && cols > 0 && cols <= POCR_MAX_ELEMENTS && stride >= cols &&
           (rows <= 1 || stride <= (POCR_MAX_ELEMENTS - cols) / (rows - 1));
}
