// Standalone validator DSO only; never included in runtime build scripts.
#define NR_STARVECTOR_FFN_GEOMETRY_TEST 1
#include "../ffn_small_m.hpp"
extern "C" int nsd_linear(void*, const void*, int, int64_t, int64_t,
                          const void*, int64_t, const float*, void*, int);
extern "C" int nrsv_ffn_try(void* g, const void* x, int dt, int64_t m, int64_t k,
                            const void* w, int64_t n, const float* b, void* y, int odt) {
    return nrsv_ffn_small_m::try_launch(g, x, dt, m, k, w, n, b, y, odt);
}
extern "C" int nrsv_ffn_test(void* g, const void* x, int dt, int64_t m, int64_t k,
                             const void* w, int64_t n, const float* b, void* y, int odt) {
    if (dt != 1 || odt != 1) return ns_fail("FFN geometry test expects half");
    return nrsv_ffn_small_m::geometry_test(g, x, m, k, w, n, b, y);
}
extern "C" int nrsv_ffn_with_fallback(void* g, const void* x, int dt, int64_t m,
                                     int64_t k, const void* w, int64_t n,
                                     const float* b, void* y, int odt) {
    const int status = nrsv_ffn_try(g, x, dt, m, k, w, n, b, y, odt);
    return status == 1 ? nsd_linear(g, x, dt, m, k, w, n, b, y, odt) : status;
}
