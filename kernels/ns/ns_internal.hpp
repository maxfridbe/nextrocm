// ns_internal.hpp: what the C ABI's translation units share (not part of the ABI).
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

struct ns_gpu {
    int dev = 0;               // the HIP device ordinal (HIP_DEVICE_ORDER=PCI_BUS_ID numbers them as nvidia-smi)
    hipStream_t q = nullptr;  // the work stream (in order)
    hipStream_t cq = nullptr; // copies that overlap q's work (ns_stream_copy); ordered against q on the device only
    hipStream_t cq2 = nullptr;// a second copy lane: the other PCIe direction runs beside the first
    std::vector<hipEvent_t> ev;   // tickets (ns_mark, ns_stream_copy, ns_stamp), % size
    int64_t next_ev = 0;
    uint64_t allocated = 0, peak = 0;
    std::unordered_map<void*, size_t> allocations;
    bool stamps = false;       // NS_PROFILE=gpu: events keep timing
};

// sets the thread's error (ns_last_error) and returns -1
int ns_fail(const std::string& what);

// the HIP call's error as ns_fail's, with where it was
int ns_hip(hipError_t e, const char* what);

#define NS_CU(call) do { hipError_t e_ = (call); if (e_ != hipSuccess) return ns_hip(e_, #call); } while (0)
#define NS_TRY try {
#define NS_CATCH } catch (const std::exception& e) { return ns_fail(e.what()); }
// a kernel launch's own error (bad configuration), checked right after it
#define NS_LAUNCH_CHECK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return ns_hip(e_, "kernel launch"); } while (0)
