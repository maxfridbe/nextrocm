// Private observer ABI; no math changes. Single live-context session.
#pragma once
#include "ns.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ni2_all_profile_record {
 int32_t family,type,rows,cols,aux,reserved;
 double gpu_ms;
} ni2_all_profile_record;
// 1..4096 records / at most8192 events. Allocate before capture.
// Families:1matvec,2decode,3RMS,4clip,5SiTU,6gate,7softcap,8shortconv,
// 9attention logits,10probabilities,11weighted,12depthmix,13router,
// 14loopcontrol,15scale,16add,17axpy. Reserved MUST be zero.
// Metadata type=GGUF code for1/2 else0; rows/cols logical shape;
// aux=postscale for1, dynamic for8, depth query count for12, else0.
int ni2_all_profile_begin(ns_gpu*,int capacity);
// Output room must be >= reserved capacity. Synchronize, validate timings,
// destroy events, then publish. On failure no timing admission; call cancel.
int ni2_all_profile_finish(ns_gpu*,ni2_all_profile_record*,int capacity,int* count);
// Must finish/cancel BEFORE destroying context, including inference errors.
int ni2_all_profile_cancel(ns_gpu*);
#ifdef __cplusplus
}
#endif
