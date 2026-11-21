// Test-only host ABI. Replace quant.hip with quant_profile.hip; never link both.
#pragma once
#include "ns.h"
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ni2_quant_profile_record {
    int32_t type, rows, cols, post_scale;
    double gpu_ms;
} ni2_quant_profile_record;
// Single global profiling session, bound to this live context. 1..2047 calls
// (at most 4094 events). Events allocated before capture; no device tensor memory.
int ni2_quant_profile_begin(ns_gpu*, int capacity);
// Host records must have room for capacity entries. Synchronizes this stream,
// writes count records in invocation order, destroys events and closes session.
// Failure means NO timing admission. Caller must cancel a failed session.
int ni2_quant_profile_finish(ns_gpu*, ni2_quant_profile_record* host_records,
                             int capacity, int* host_count);
// Synchronize/destroy events before freeing context, also after any inference error.
int ni2_quant_profile_cancel(ns_gpu*);
#ifdef __cplusplus
}
#endif
