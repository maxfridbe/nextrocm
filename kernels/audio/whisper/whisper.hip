#include "whisper.h"
#include "../../ns/ns_internal.hpp"
#include "linear_f16_tiled.hpp"
#include "attention_tiled.hpp"

extern "C" int ns_audio_whisper_linear_f16_tiled(ns_gpu* g, const void* x,
    int64_t M, int64_t K, const void* w, int64_t N, const float* bias, void* out) {
    if (!g || !whisper_linear_f16::arguments(x,M,K,w,N,bias,out))
        return ns_fail("Whisper tiled linear: invalid arguments");
    NS_CU(hipSetDevice(g->dev));
    NS_CU(whisper_linear_f16::launch(x,M,K,w,N,bias,out,g->q));
    return 0;
}

extern "C" int ns_audio_whisper_attention_tiled(ns_gpu* g, const void* q,
    const void* k, const void* v, int64_t S, int64_t H, int64_t D,
    int64_t stride, void* out) {
    if (!g || !whisper_attention::arguments(q,k,v,S,H,D,stride,out))
        return ns_fail("Whisper tiled attention: invalid arguments");
    NS_CU(hipSetDevice(g->dev));
    NS_CU(whisper_attention::launch(q,k,v,S,H,D,stride,out,g->q));
    return 0;
}
