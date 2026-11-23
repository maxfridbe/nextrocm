// ns.cu: the devices, memory and copies of the C ABI (ns.h) on HIP. Kernels are called through their own wrappers
// (kernels/<kind>/<arch>/*.cu), each taking the ns_gpu's stream.
#include "ns.h"
#include "ns_internal.hpp"

#include <mutex>
#include <memory>
#include <unordered_map>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {
thread_local std::string g_err;

int count() {
    int n = 0;
    const auto e = hipGetDeviceCount(&n);
    if (e != hipSuccess) { ns_hip(e, "hipGetDeviceCount"); return -1; }
    return n;
}

// make GPU g current for this thread (every call: threads are not tied to a GPU)
inline hipError_t use(ns_gpu* g) { return hipSetDevice(g->dev); }
}  // namespace

int ns_fail(const std::string& what) {
    g_err = what;
    return -1;
}

int ns_hip(hipError_t e, const char* what) {
    return ns_fail(std::string(what) + ": " + hipGetErrorName(e) + " (" + hipGetErrorString(e) + ")");
}

extern "C" {

const char* ns_last_error(void) { return g_err.c_str(); }
const char* ns_version(void) { return "nextrocm 0.1"; }

int ns_gpu_count(void) { return count(); }

int ns_gpu_name(int index, char* buf, size_t len) {
    hipDeviceProp_t p;
    NS_CU(hipGetDeviceProperties(&p, index));
    std::snprintf(buf, len, "%s", p.name);
    return 0;
}

// GPU `index`'s multiprocessors and their clock (MHz): how much it computes
int ns_gpu_units(int index, int* units, int* mhz) {
    int sm = 0, khz = 0;
    NS_CU(hipDeviceGetAttribute(&sm, hipDeviceAttributeMultiprocessorCount, index));
    NS_CU(hipDeviceGetAttribute(&khz, hipDeviceAttributeClockRate, index));
    *units = sm;
    *mhz = khz / 1000;
    return 0;
}

int ns_gpu_props(int index, uint64_t* total, int* wave, int* integrated,
                 char* gfx, size_t len) {
    if (index < 0 || !total || !wave || !integrated || !gfx || len == 0)
        return ns_fail("ns_gpu_props: invalid index or output");
    hipDeviceProp_t p{};
    NS_CU(hipGetDeviceProperties(&p, index));
    const size_t n = strnlen(p.gcnArchName, sizeof(p.gcnArchName));
    if (n == 0 || n == sizeof(p.gcnArchName) || n >= len || p.warpSize <= 0)
        return ns_fail("ns_gpu_props: incomplete properties or short gfx buffer");
    // Publish only after the query and every fallible check succeeds.
    std::memcpy(gfx, p.gcnArchName, n + 1);
    *total = static_cast<uint64_t>(p.totalGlobalMem);
    *wave = p.warpSize;
    *integrated = p.integrated ? 1 : 0;
    return 0;
}

// Legacy compatibility query: AMD gfx capabilities are exposed by ns_gpu_props.
int ns_gpu_cc(int index, int* cc) {
    // No NVIDIA SM capability on AMD. gfx targets are selected at build time.
    (void)index;
    *cc = 0;
    return 0;
}

// the PCI address of GPU `index` ("0000:09:00.0") into buf
int ns_gpu_pci(int index, char* buf, size_t len) {
    char id[32] = {0};
    NS_CU(hipDeviceGetPCIBusId(id, (int) sizeof id, index));
    std::string a(id);
    for (auto& c : a) c = (char) std::tolower((unsigned char) c);
    // "00000000:09:00.0" -> "0000:09:00.0"
    const auto colon = a.find(':');
    if (colon != std::string::npos && colon > 4) a = a.substr(colon - 4);
    std::snprintf(buf, len, "%s", a.c_str());
    return 0;
}

int ns_gpu_open(int index, ns_gpu** out) {
    NS_TRY
    const int n = count();
    if (index < 0 || index >= n) return ns_fail("no GPU " + std::to_string(index) + " (" + std::to_string(n) + " found)");
    NS_CU(hipSetDevice(index));
    auto owner = std::make_unique<ns_gpu>();
    auto* g = owner.get();
    g->dev = index;
    g->ev.resize(1 << 12);
    const char* pf = getenv("NS_PROFILE");
    g->stamps = pf && std::string(pf) == "gpu";
    if (hipStreamCreateWithFlags(&g->q, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&g->cq, hipStreamNonBlocking) != hipSuccess ||
        hipStreamCreateWithFlags(&g->cq2, hipStreamNonBlocking) != hipSuccess) {
        if (g->q) (void)hipStreamDestroy(g->q);
        if (g->cq) (void)hipStreamDestroy(g->cq);
        if (g->cq2) (void)hipStreamDestroy(g->cq2);
        return ns_fail("creating the GPU's streams failed");
    }
    // tickets are events kept in a ring (the host never waits on one: the device orders by them)

    for (auto& e : g->ev) {
        auto rc = hipEventCreateWithFlags(&e, g->stamps ? hipEventDefault : hipEventDisableTiming);
        if (rc != hipSuccess) {
            for (auto event : g->ev) if (event) (void)hipEventDestroy(event);
            (void)hipStreamDestroy(g->q); (void)hipStreamDestroy(g->cq); (void)hipStreamDestroy(g->cq2);
            return ns_hip(rc, "creating event ring");
        }
    }
    *out = owner.release();
    return 0;
    NS_CATCH
}

void ns_gpu_close(ns_gpu* g) {
    if (!g) return;
    (void)hipSetDevice(g->dev);
    (void)hipStreamSynchronize(g->q);
    (void)hipStreamSynchronize(g->cq);
    (void)hipStreamSynchronize(g->cq2);
    for (auto e : g->ev) (void)hipEventDestroy(e);
    (void)hipStreamDestroy(g->q);
    (void)hipStreamDestroy(g->cq);
    (void)hipStreamDestroy(g->cq2);
    delete g;
}

void* ns_gpu_queue(ns_gpu* g) { return (void*) g->q; }

int ns_gpu_memory(ns_gpu* g, uint64_t* total, int64_t* free_bytes) {
    NS_CU(use(g));
    size_t f = 0, t = 0;
    NS_CU(hipMemGetInfo(&f, &t));
    *total = t;
    *free_bytes = (int64_t) f;
    return 0;
}

int ns_alloc(ns_gpu* g, size_t bytes, void** out) {
    NS_CU(use(g));
    bytes = bytes ? bytes : 1;
    // Cap every model context, retaining headroom for the HIP runtime itself.
    if (bytes >= 7500000000ULL || g->allocated >= 7500000000ULL - bytes)
        return ns_fail("device allocation exceeds the 7.5 GB tensor budget");
    hipError_t e = hipMalloc(out, bytes);
    if (e != hipSuccess) {
        (void)hipGetLastError();
        return ns_fail("device allocation of " + std::to_string(bytes) + " bytes failed: " + hipGetErrorString(e));
    }
    g->allocations[out[0]] = bytes;
    g->allocated += bytes;
    g->peak = std::max(g->peak, g->allocated);
    return 0;
}

int nr_allocation_stats(ns_gpu* g, uint64_t* current, uint64_t* peak) {
    *current = g->allocated; *peak = g->peak; return 0;
}

int ns_free(ns_gpu* g, void* p) {
    NS_CU(use(g));
    // the stream is in order: work queued on p runs first
    NS_CU(hipStreamSynchronize(g->q));
    NS_CU(hipFree(p));
    auto it = g->allocations.find(p);
    if (it != g->allocations.end()) { g->allocated -= it->second; g->allocations.erase(it); }
    return 0;
}

int ns_alloc_host(ns_gpu* g, size_t bytes, void** out) {
    NS_CU(use(g));
    hipError_t e = hipHostMalloc(out, bytes ? bytes : 1);
    if (e != hipSuccess) {
        (void)hipGetLastError();
        return ns_fail("pinned host allocation of " + std::to_string(bytes) + " bytes failed: " + hipGetErrorString(e));
    }
    return 0;
}

int ns_free_host(ns_gpu* g, void* p) {
    NS_CU(use(g));
    NS_CU(hipStreamSynchronize(g->q));
    NS_CU(hipHostFree(p));
    return 0;
}

// queued on the GPU's stream (in order); a pageable host source is staged by the driver before the call returns
int ns_copy_to(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_CU(use(g));
    NS_CU(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, g->q));
    return 0;
}

int ns_copy_from(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_CU(use(g));
    NS_CU(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, g->q));
    return 0;
}

int ns_copy_dev(ns_gpu* g, void* dst, const void* src, size_t bytes) {
    NS_CU(use(g));
    // (hipMemcpyDefault: either side may be pinned host memory's device address - an upload slot - as well as VRAM)
    NS_CU(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, g->q));
    return 0;
}

int ns_copy_2d(ns_gpu* g, void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height) {
    NS_CU(use(g));
    NS_CU(hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice, g->q));
    return 0;
}

// Copies beside the GPU's work. The order between the streams is kept on the device (events, hipStreamWaitEvent):
// the host never waits on an event.
static int64_t keep(ns_gpu* g, hipStream_t s) {
    const int64_t t = g->next_ev++;
    const auto rc = hipEventRecord(g->ev[t % g->ev.size()], s);
    if (rc != hipSuccess) return ns_hip(rc, "recording ticket");
    return t;
}

// a ticket for everything submitted to the GPU's stream so far
int ns_mark(ns_gpu* g, int64_t* ticket) {
    NS_CU(use(g));
    *ticket = keep(g, g->q);
    return *ticket < 0 ? -1 : 0;
}

// bytes from src to dst on copy lane `lane` (0 or 1), after the `ndeps` tickets `after`; its own ticket
int ns_stream_copy(ns_gpu* g, void* dst, const void* src, size_t bytes, const int64_t* after, int ndeps, int lane, int64_t* ticket) {
    NS_CU(use(g));
    hipStream_t s = lane ? g->cq2 : g->cq;
    for (int i = 0; i < ndeps; ++i) NS_CU(hipStreamWaitEvent(s, g->ev[after[i] % g->ev.size()], 0));
    NS_CU(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, s));
    *ticket = keep(g, s);
    return *ticket < 0 ? -1 : 0;
}

// `n` copies (dsts[i] <- srcs[i], sizes[i] bytes) in one call on copy lane `lane`, after the tickets `after`; one ticket
// after them all (individual HIP copies, preserving dependencies on older ROCm versions)
int ns_stream_copy_batch(ns_gpu* g, void* const* dsts, const void* const* srcs, const size_t* sizes, size_t n, const int64_t* after, int ndeps, int lane, int64_t* ticket) {
    NS_CU(use(g));
    hipStream_t s = lane ? g->cq2 : g->cq;
    for (int i = 0; i < ndeps; ++i) NS_CU(hipStreamWaitEvent(s, g->ev[after[i] % g->ev.size()], 0));
    for (size_t i = 0; i < n; ++i) {
        NS_CU(hipMemcpyAsync(dsts[i], srcs[i], sizes[i], hipMemcpyDefault, s));
    }
    *ticket = keep(g, s);
    return *ticket < 0 ? -1 : 0;
}

// a timestamp: an event on the GPU's stream (its time is when the work before it ended); its ticket
int ns_stamp(ns_gpu* g, int64_t* ticket) {
    NS_CU(use(g));
    *ticket = keep(g, g->q);
    return *ticket < 0 ? -1 : 0;
}

// nanoseconds between tickets t0 and t1 (stamps; waits for them - profiling only, with NS_PROFILE=gpu)
int ns_elapsed(ns_gpu* g, int64_t t0, int64_t t1, double* ns) {
    NS_CU(use(g));
    if (!g->stamps) return ns_fail("ns_elapsed needs NS_PROFILE=gpu");
    float ms = 0;
    NS_CU(hipEventSynchronize(g->ev[t1 % g->ev.size()]));
    NS_CU(hipEventElapsedTime(&ms, g->ev[t0 % g->ev.size()], g->ev[t1 % g->ev.size()]));
    *ns = (double) ms * 1e6;
    return 0;
}

// Graphs: the work queued on the GPU's stream between ns_capture_begin and ns_capture_launch recorded (not run), then
// launched as one graph - a thousand kernels one entry in the stream's queue, so the host never waits on a full queue
// and queues the other GPU's work. Nothing between them may allocate or wait (the capture is the thread's own).
int ns_capture_begin(ns_gpu* g) {
    NS_CU(use(g));
    NS_CU(hipStreamBeginCapture(g->q, hipStreamCaptureModeThreadLocal));
    return 0;
}

int ns_capture_launch(ns_gpu* g) {
    NS_CU(use(g));
    hipGraph_t graph = nullptr;
    NS_CU(hipStreamEndCapture(g->q, &graph));
    hipGraphExec_t exec = nullptr;
    const hipError_t e = hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    (void)hipGraphDestroy(graph);
    if (e != hipSuccess) return ns_hip(e, "hipGraphInstantiate");
    const hipError_t l = hipGraphLaunch(exec, g->q);
    // (freed once it has run)
    (void)hipGraphExecDestroy(exec);
    if (l != hipSuccess) return ns_hip(l, "hipGraphLaunch");
    return 0;
}

// a capture that failed part way: ended, its work dropped (the stream usable again)
int ns_capture_abort(ns_gpu* g) {
    NS_CU(use(g));
    hipStreamCaptureStatus st = hipStreamCaptureStatusNone;
    NS_CU(hipStreamIsCapturing(g->q, &st));
    if (st != hipStreamCaptureStatusNone) {
        hipGraph_t graph = nullptr;
        (void)hipStreamEndCapture(g->q, &graph);
        if (graph) (void)hipGraphDestroy(graph);
    }
    (void)hipGetLastError();
    return 0;
}

// the host waits for ticket t (a buffer the work before it read may be written again)
int ns_ticket_wait(ns_gpu* g, int64_t t) {
    NS_CU(use(g));
    NS_CU(hipEventSynchronize(g->ev[t % g->ev.size()]));
    return 0;
}

// the GPU's stream waits (on the device) for ticket t
int ns_await(ns_gpu* g, int64_t t) {
    NS_CU(use(g));
    NS_CU(hipStreamWaitEvent(g->q, g->ev[t % g->ev.size()], 0));
    return 0;
}

int ns_fill(ns_gpu* g, void* dst, uint8_t value, size_t bytes) {
    NS_CU(use(g));
    NS_CU(hipMemsetAsync(dst, value, bytes, g->q));
    return 0;
}

int ns_sync(ns_gpu* g) {
    NS_CU(use(g));
    NS_CU(hipStreamSynchronize(g->q));
    // a kernel's asynchronous error surfaces here
    NS_CU(hipGetLastError());
    return 0;
}

int ns_copy_peer(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging) {
    NS_CU(use(from));
    NS_CU(hipMemcpyAsync(staging, src, bytes, hipMemcpyDeviceToHost, from->q));
    NS_CU(hipStreamSynchronize(from->q));
    NS_CU(use(to));
    NS_CU(hipMemcpyAsync(dst, staging, bytes, hipMemcpyHostToDevice, to->q));
    NS_CU(hipStreamSynchronize(to->q));
    return 0;
}

// The same through pinned `staging`, ordered on the devices only (the host does not wait): the read after the work
// queued on `from` so far, the write after the work queued on `to` so far (dst may be what that work reads); later work
// on `to` after the write, on `from` after the read (src may then be reused); a staging area's next read after its
// last write. Two GPUs then run side by side: `from` on the next rows while `to` is on these.
int ns_copy_peer_async(ns_gpu* to, void* dst, ns_gpu* from, const void* src, size_t bytes, void* staging) {
    // Conservative fallback until multi-device event ownership is ported and tested.
    return ns_copy_peer(to, dst, from, src, bytes, staging);
}

}  // extern "C"
