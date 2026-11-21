// Test-only allocations deliberately bypass the resident model allocator.
// Link this TU into libnextrocm-runtime.so; see tooling/check-lowvram-kernels.py.
#include "ops.h"
#include "../../ns/ns_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}
template<class T> struct Buffer {
    T* p = nullptr;
    explicit Buffer(size_t n) { check(hipMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { if (p) (void)hipFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void put(const std::vector<T>& v) { check(hipMemcpy(p, v.data(), v.size()*sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get(size_t n) {
        check(hipGetLastError());
        check(hipDeviceSynchronize());
        std::vector<T> v(n);
        check(hipMemcpy(v.data(), p, n*sizeof(T), hipMemcpyDeviceToHost));
        return v;
    }
};
// Integer hashing avoids platform-dependent RNG distributions and periodic lanes.
int fixture(uint32_t i) {
    i ^= i >> 16; i *= 0x7feb352du; i ^= i >> 15; i *= 0x846ca68bu; i ^= i >> 16;
    return int(i % 255) - 127;
}
// Fixtures are exactly representable normal binary16 values (or zero).
uint16_t exact_half(float f) {
    if (f == 0) return 0;
    uint32_t u; std::memcpy(&u, &f, sizeof(u));
    const int e = int((u >> 23) & 255) - 127 + 15;
    if (e <= 0 || e >= 31 || (u & 8191)) throw std::runtime_error("non-exact half fixture");
    return uint16_t(((u >> 16) & 0x8000) | (e << 10) | ((u >> 13) & 1023));
}
constexpr size_t guard = 16;
constexpr float sentinel = 12345.25f;
void compare(const std::string& name, const std::vector<float>& actual,
             const std::vector<double>& expected, double atol, double rtol) {
    for (size_t i = 0; i < expected.size(); ++i) {
        const double a = actual[i + guard], b = expected[i];
        if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > atol + rtol*std::abs(b))
            throw std::runtime_error(name + " index=" + std::to_string(i) +
                " actual=" + std::to_string(a) + " expected=" + std::to_string(b));
    }
    for (size_t i = 0; i < guard; ++i)
        if (actual[i] != sentinel || actual[guard + expected.size() + i] != sentinel)
            throw std::runtime_error(name + " output guard overwritten");
}
std::vector<float> output(size_t n, bool accumulate = false) {
    std::vector<float> v(n + 2*guard, sentinel);
    for (size_t i = 0; i < n; ++i)
        v[guard+i] = accumulate ? fixture(uint32_t(i+77))/32.f : std::numeric_limits<float>::quiet_NaN();
    return v;
}
void matrix(WT type, int M, int& completed) {
    constexpr int K = 4096, N = 17;
    std::vector<float> x(M*K), w(N*K), bias(N);
    std::vector<int8_t> q(N*K);
    std::vector<uint16_t> h(N*K), scales(N*K/32);
    for (size_t i = 0; i < x.size(); ++i) x[i] = fixture(uint32_t(i+101))/128.f;
    for (size_t b = 0; b < scales.size(); ++b) scales[b] = exact_half(float(1+b%7)/1024.f);
    for (size_t i = 0; i < w.size(); ++i) {
        q[i] = int8_t(fixture(uint32_t(i+900001)));
        w[i] = type == WT::Q8 ? float(q[i])*float(1+(i/32)%7)/1024.f : float(q[i])/128.f;
        h[i] = exact_half(w[i]);
    }
    for (int n = 0; n < N; ++n) bias[n] = fixture(n+71)/64.f;
    Buffer<float> dx(x.size()), dw(w.size()), db(N), dy(M*N+2*guard);
    Buffer<uint16_t> dh(h.size()), ds(scales.size()); Buffer<int8_t> dq(q.size());
    dx.put(x); dw.put(w); db.put(bias); dh.put(h); ds.put(scales); dq.put(q);
    Weight W; W.t = type; W.K = K; W.N = N; W.scales = ds.p;
    W.w = type == WT::F32 ? static_cast<void*>(dw.p) : type == WT::F16 ? static_cast<void*>(dh.p) : static_cast<void*>(dq.p);
    std::vector<double> dot(M*N);
    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) dot[m*N+n] += double(x[m*K+k])*w[n*K+k];
    // Overwrite with bias, accumulate with bias, overwrite without bias.
    for (int mode = 0; mode < 3; ++mode) {
        const bool acc = mode == 1;
        auto initial = output(M*N, acc); dy.put(initial);
        W.bias = mode == 2 ? nullptr : db.p;
        auto expected = dot;
        for (int i = 0; i < M*N; ++i) expected[i] += (W.bias ? bias[i%N] : 0) + (acc ? initial[guard+i] : 0);
        const std::string name = "matmul type=" + std::to_string(int(type)) + " M=" + std::to_string(M) + " K=4096 N=17 mode=" + std::to_string(mode);
        try {
            matmul(dx.p, M, W, dy.p+guard, acc);
            compare(name, dy.get(M*N+2*guard), expected, 2e-3, 2e-4);
        } catch (const std::exception& e) { throw std::runtime_error(name + ": " + e.what()); }
        ++completed;
    }
}
void reductions(int C, int& completed) {
    constexpr int M = 3;
    constexpr float eps = 1e-5f;
    std::vector<float> x(M*C), w(C), b(C);
    for (int c = 0; c < C; ++c) {
        w[c] = 1.f + fixture(c+81)/256.f; b[c] = fixture(c+91)/128.f;
        x[c] = fixture(c+1001)/32.f + 2.f;
        x[C+c] = 0.75f; // epsilon and zero-variance behavior
        x[2*C+c] = fixture(c+3001)/65536.f;
    }
    Buffer<float> dx(x.size()), dw(C), db(C), dy(x.size()+2*guard);
    dx.put(x); dw.put(w); db.put(b);
    for (int op = 0; op < 6; ++op) {
        auto initial = output(x.size());
        const bool softmax = op >= 4;
        const float scale = op == 4 ? 0.75f : -1.25f;
        if (softmax) for (size_t i = 0; i < x.size(); ++i)
            initial[guard+i] = x[i]*16.f + 1000.f; // overflow without max subtraction
        dy.put(initial);
        std::vector<double> expected(x.size());
        for (int m = 0; m < M; ++m) {
            double mean = 0, ss = 0;
            for (int c = 0; c < C; ++c) mean += x[m*C+c];
            mean /= C;
            for (int c = 0; c < C; ++c) {
                double v = x[m*C+c] - (op >= 2 ? mean : 0);
                ss += v*v;
            }
            if (softmax) {
                double mx = -std::numeric_limits<double>::infinity(), sum = 0;
                // Max uses rounded float products. HIP contracts the exp argument to FMA;
                // preserve that rounding before reference exp/sum in double.
                for (int c = 0; c < C; ++c) mx = std::max(mx, double(initial[guard+m*C+c]*scale));
                for (int c = 0; c < C; ++c) sum += expected[m*C+c] = std::exp(double(std::fma(initial[guard+m*C+c], scale, -float(mx))));
                for (int c = 0; c < C; ++c) expected[m*C+c] /= sum;
            } else for (int c = 0; c < C; ++c)
                expected[m*C+c] = (x[m*C+c] - (op >= 2 ? mean : 0))/std::sqrt(ss/C+eps)*(op == 1 ? 1.f : w[c]) + (op == 2 ? b[c] : 0);
        }
        const std::string name = "reduction op=" + std::to_string(op) + " C=" + std::to_string(C);
        try {
            if (op < 2) rmsnorm(dx.p, M, C, op == 1 ? nullptr : dw.p, eps, dy.p+guard);
            else if (op < 4) layernorm(dx.p, M, C, dw.p, op == 2 ? db.p : nullptr, eps, dy.p+guard);
            else softmax_rows(dy.p+guard, M, C, scale);
            auto actual = dy.get(x.size()+2*guard);
            compare(name, actual, expected, softmax ? 2e-7 : 2e-5, softmax ? 2e-5 : 3e-5);
            if (softmax) for (int m = 0; m < M; ++m) {
                double sum = 0;
                for (int c = 0; c < C; ++c) {
                    const float a = actual[guard+m*C+c];
                    if (a < 0 || a > 1) throw std::runtime_error("invalid probability");
                    sum += a;
                }
                if (std::abs(sum-1) > 2e-5) throw std::runtime_error("probabilities do not sum to one");
            }
        } catch (const std::exception& e) { throw std::runtime_error(name + ": " + e.what()); }
        ++completed;
    }
}
} // namespace

// Returns 0 only when all 60 numeric cases pass; errors use ns_last_error().
extern "C" int nr_lowvram_selftest(ns_gpu* g, int* completed) {
    if (!g || !completed) return ns_fail("lowvram selftest: null argument");
    *completed = 0;
    NS_TRY
        check(hipSetDevice(g->dev));
        for (WT type : {WT::F32, WT::F16, WT::Q8})
            for (int M : {1, 4, 24, 25}) matrix(type, M, *completed);
        for (int C : {17, 255, 257, 1025}) reductions(C, *completed);
        return 0;
    NS_CATCH
}
