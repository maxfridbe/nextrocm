#include "ns/ns.h"
#include "ns/ns_internal.hpp"
__global__ void add_vectors(float* out, const float* a, const float* b, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}
extern "C" int nr_test_add(ns_gpu* g, void* out, const void* a, const void* b, size_t n) {
    NS_CU(hipSetDevice(g->dev));
    if (!n) return 0;
    hipLaunchKernelGGL(add_vectors, dim3((n+255)/256), dim3(256), 0, g->q,
                      static_cast<float*>(out), static_cast<const float*>(a), static_cast<const float*>(b), n);
    NS_LAUNCH_CHECK();
    return 0;
}
