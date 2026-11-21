// Exact bodies extracted from pinned merged release; see provenance.json and LICENSE.
#include <cstdint>
#include <cmath>
#include <cassert>
#define GGML_RESTRICT
#define QK_NVFP4 64
#define QK_NVFP4_SUB 16
#define GGML_TABLE_BEGIN(type,name,size) static const type name[size] = {
#define GGML_TABLE_END() };
#define kvalues_mxfp4 kvalues_fp4
struct block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; };
static_assert(sizeof(block_nvfp4)==36);
GGML_TABLE_BEGIN(int8_t, kvalues_fp4, 16)
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
GGML_TABLE_END()
static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    // TODO: reject invalid unsigned E4M3 scales instead of decoding NaN as zero or ignoring the sign bit.
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}
void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK_NVFP4;
    static const int qk_sub = QK_NVFP4_SUB;
    static const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        for (int s = 0; s < n_sub; s++) {
            const float d = ggml_ue4m3_to_fp32(x[i].d[s]);
            float * yb = y + i*qk + s*qk_sub;

            for (int j = 0; j < qk_sub/2; ++j) {
                const int8_t v0 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] & 0x0F];
                const int8_t v1 = kvalues_mxfp4[x[i].qs[s*(qk_sub/2) + j] >>   4];

                yb[j + 0       ] = v0*d;
                yb[j + qk_sub/2] = v1*d;
            }
        }
    }
}
