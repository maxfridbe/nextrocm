// SPDX-License-Identifier: GPL-3.0-only
#pragma once
#include <cstdint>
// Checked host geometry before multiplication or launch. The original ABI carries
// no row count for x/rowptr; graph indices must be validated by the Rust owner.
constexpr int64_t max_floats = 134217728; // 512 MiB per private buffer
inline bool product_ok(int64_t a, int64_t b) { return a >= 0 && b > 0 && a <= max_floats / b; }
inline bool span_ok(int64_t rows, int64_t stride, int64_t cols, int64_t off = 0) {
    return rows >= 0 && stride >= 0 && cols > 0 && off >= 0 && off <= max_floats - cols &&
           (rows <= 1 || stride <= (max_floats - off - cols) / (rows - 1));
}
