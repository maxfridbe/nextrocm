// rb.hip: the background removers' own kernels: IS-Net's ceil-mode max-pool, its half-pixel bilinear upsampling, a
// bias with ReLU, the sigmoid. Elementwise, on the GPU's stream.
#include "ns.h"
#include "ns_internal.hpp"
#include "rb.h"
#include "rb_bounds.hpp"

#include <cfloat>
#include <cmath>

namespace {

constexpr int WG = 256;
inline unsigned nblk(int64_t n) { return (unsigned) ((n + WG - 1) / WG); }

__global__ void maxpool2_k(const float* x, int64_t H, int64_t W, int64_t C, int64_t Ho, int64_t Wo, float* out) {
    const int64_t n = Ho * Wo * C;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, p = id / C, xo = p % Wo, yo = p / Wo;
    float m = -FLT_MAX;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            const int64_t y = 2 * yo + dy, xx = 2 * xo + dx;
            if (y < H && xx < W) m = fmaxf(m, x[(y * W + xx) * C + c]);
        }
    }
    out[id] = m;
}

__global__ void resize_k(const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t os) {
    const int64_t n = Ho * Wo * C;
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    const int64_t c = id % C, p = id / C, xo = p % Wo, yo = p / Wo;
    const float sy = fmaxf(((float) yo + 0.5f) * (float) Hi / (float) Ho - 0.5f, 0.0f);
    const float sx = fmaxf(((float) xo + 0.5f) * (float) Wi / (float) Wo - 0.5f, 0.0f);
    const int64_t y0 = min((int64_t) sy, Hi - 1), x0 = min((int64_t) sx, Wi - 1);
    const int64_t y1 = min(y0 + 1, Hi - 1), x1 = min(x0 + 1, Wi - 1);
    const float fy = sy - (float) y0, fx = sx - (float) x0;
    const float a = x[(y0 * Wi + x0) * C + c], b = x[(y0 * Wi + x1) * C + c];
    const float d = x[(y1 * Wi + x0) * C + c], e = x[(y1 * Wi + x1) * C + c];
    out[(yo * Wo + xo) * os + c] = (a * (1.0f - fx) + b * fx) * (1.0f - fy) + (d * (1.0f - fx) + e * fx) * fy;
}

__global__ void bias_act_k(float* x, int64_t n, int64_t C, const float* bias, int relu) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    float v = x[id] + (bias ? bias[id % C] : 0.0f);
    x[id] = relu ? fmaxf(v, 0.0f) : v;
}

__global__ void sigmoid_k(float* x, int64_t n) {
    const int64_t id = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) x[id] = 1.0f / (1.0f + expf(-x[id]));
}

// F32 X[M,K] * W[N,K]^T. Each lane owns rows ty+16*i and
// columns tx+16*j. Padding changes bank mapping, never logical K order.
__global__ void linear_tiled_k(const float* x, const float* w,
                              int64_t M, int64_t K, int64_t N, float* out) {
    __shared__ float a[64][17], b[64][17];
    const int tx = threadIdx.x, ty = threadIdx.y, lane = ty * 16 + tx;
    const int64_t row = (int64_t)blockIdx.y * 64, col = (int64_t)blockIdx.x * 64;
    float acc[4][4] = {};
    for (int64_t base = 0; base < K; base += 16) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int v = lane + i * 256, r = v / 16, k = v % 16;
            a[r][k] = row+r < M && base+k < K ? x[(row+r)*K+base+k] : 0.0f;
            b[r][k] = col+r < N && base+k < K ? w[(col+r)*K+base+k] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float av = a[ty+i*16][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] = fmaf(av, b[tx+j*16][k], acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            if (row+ty+i*16 < M && col+tx+j*16 < N)
                // Match nsd_linear's null-bias epilogue, including +0.
                out[(row+ty+i*16)*N+col+tx+j*16] = acc[i][j] + 0.0f;
    }
}

#include "biref_linear.hpp"

}  // namespace

#define RB_LAUNCH(k, n, ...) do { NS_CU(hipSetDevice(g->dev)); if ((n) > 0) { k<<<nblk(n), WG, 0, g->q>>>(__VA_ARGS__); NS_LAUNCH_CHECK(); } return 0; } while (0)

extern "C" {

int ns_image_rb_biref_linear_tiled(ns_gpu* g, const float* x, const float* w,
                                  int64_t M, int64_t K, int64_t N,
                                  const float* bias, float* out) {
    if (!g || !x || !w || !out || x == out || w == out || bias == out || !rb_bounds::linear(M,K,N))
        return ns_fail("rembg: invalid BiRefNet linear shape/pointers");
    NS_CU(hipSetDevice(g->dev));
    biref_linear_tiled_k<<<dim3((unsigned)((N+63)/64), (unsigned)((M+63)/64)), dim3(16,16), 0, g->q>>>(x,w,M,K,N,bias,out);
    NS_LAUNCH_CHECK();
    return 0;
}


int ns_image_rb_linear_tiled(ns_gpu* g, const float* x, const float* w,
                            int64_t M, int64_t K, int64_t N, float* out) {
    if (!g || !x || !w || !out || x == out || w == out || !rb_bounds::linear(M,K,N))
        return ns_fail("rembg: invalid linear shape/pointers");
    NS_CU(hipSetDevice(g->dev));
    linear_tiled_k<<<dim3((unsigned)((N+63)/64), (unsigned)((M+63)/64)), dim3(16,16), 0, g->q>>>(x,w,M,K,N,out);
    NS_LAUNCH_CHECK();
    return 0;
}


int ns_image_rb_maxpool2(ns_gpu* g, const float* x, int64_t H, int64_t W, int64_t C, float* out) {
    if (!g || !x || !out || x == out || !rb_bounds::pool(H,W,C))
        return ns_fail("rembg: invalid maxpool2 shape/pointers");
    const int64_t Ho = H / 2 + H % 2, Wo = W / 2 + W % 2;
    RB_LAUNCH(maxpool2_k, Ho * Wo * C, x, H, W, C, Ho, Wo, out);
}

int ns_image_rb_resize(ns_gpu* g, const float* x, int64_t Hi, int64_t Wi, int64_t C, int64_t Ho, int64_t Wo, float* out, int64_t out_stride) {
    if (!g || !x || !out || x == out || !rb_bounds::resize(Hi,Wi,C,Ho,Wo,out_stride))
        return ns_fail("rembg: invalid resize shape/stride/pointers");
    RB_LAUNCH(resize_k, Ho * Wo * C, x, Hi, Wi, C, Ho, Wo, out, out_stride);
}

int ns_image_rb_bias_act(ns_gpu* g, float* x, int64_t M, int64_t C, const float* bias, int relu) {
    if (!g || !x || !rb_bounds::bias(M,C,relu))
        return ns_fail("rembg: invalid bias_act shape/pointers");
    RB_LAUNCH(bias_act_k, M * C, x, M * C, C, bias, relu);
}

int ns_image_rb_sigmoid(ns_gpu* g, float* x, int64_t n) {
    if (!g || !x || !rb_bounds::sigmoid(n))
        return ns_fail("rembg: invalid sigmoid shape/pointers");
    RB_LAUNCH(sigmoid_k, n, x, n);
}

}  // extern "C"
