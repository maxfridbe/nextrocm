// SPDX-License-Identifier: GPL-3.0-only
#include "bounds.h"
#include <cassert>
#include <limits>
int main() {
    assert(product_ok(3,2));
    assert(product_ok(0,2));
    assert(!product_ok(-1,2));
    assert(!product_ok(1,0));
    assert(!product_ok(std::numeric_limits<int64_t>::max(),2));
    assert(product_ok(max_floats,1));
    assert(!product_ok(max_floats+1,1));
    assert(span_ok(3,4,2,1));
    assert(span_ok(3,0,2));
    assert(!span_ok(3,4,2,-1));
    assert(!span_ok(3,std::numeric_limits<int64_t>::max(),2));
    assert(!span_ok(1,0,max_floats+1));
}
