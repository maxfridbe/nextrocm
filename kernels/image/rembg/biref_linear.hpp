// Private BiRefNet F32 candidate; included inside rb.hip anonymous namespace.
// Same padded increasing-K arithmetic as nsd_linear; independent of ISNet dispatch.
#pragma once

__global__ void biref_linear_tiled_k(const float* x, const float* w,
                              int64_t M, int64_t K, int64_t N, const float* bias, float* out) {
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
                // Match nsd_linear: bias addition follows the complete FMA chain.
                out[(row+ty+i*16)*N+col+tx+j*16] = acc[i][j] + (bias ? bias[col+tx+j*16] : 0.0f);
    }
}
