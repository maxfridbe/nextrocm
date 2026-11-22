# Idle-queue host-to-device diagnostic

Build from the silo root (host C++ only, no kernel changes):

```sh
c++ -std=c++17 -O2 -Wall -Wextra -Werror tests/wopr-llm/h2d/h2d.cpp -ldl -o /tmp/inkling-h2d-v1
```

Run under the existing exclusive GPU watchdog, pinning executable and runtime before/after. Pass the absolute typed runtime path as the sole argument. The tested runtime prerequisite is SHA256 `25a45d71d718cd92c2e2fec67dc4ffe08ab7f393bb3c8338e9e6becc0a877598`; this program does not itself hash files. Redirect stdout to a new JSON report and retain stderr and exit status. A failed run may leave incomplete JSON; only exit zero with final `status: PASS` passes.

Sizes are 2,448; 4,478,976; 6,469,632; 35,831,808; and 51,757,056 bytes. For each size, pageable then HIP-pinned sources run separately, with three warmup copies, a copy-plus-sync wall pass, then an event-instrumented pass. Each pass targets at least 512 MiB, capped at 4,096 repetitions (the small row therefore transfers only 10,027,008 bytes). Actual repetition and byte counts are reported. Repeated source/destination buffers are reused; this is a warm-buffer microbenchmark, not a streaming-model workload.

The wall-only pass times exactly `ns_copy_to` followed by `ns_sync`. No prior compute is queued. The event pass includes timestamp submission in its reported wall interval; elapsed-event retrieval happens afterward. Event intervals can include pageable staging or host enqueue gaps, and **are not pure DMA time**. Neither measurement establishes PCIe link bandwidth or model throughput.

Every source byte and destination byte, including two 64-byte guard regions, is checked before timing and after each pass. Sources remain alive until synchronization; pinned memory is freed through the runtime. Device allocation statistics must return to zero after every case with peak at most 64 MiB. Largest explicit source + readback + device buffers total 155,271,552 bytes; small timing arrays, runtime/context allocations, and driver memory are additional and must be observed by the external watchdog. No model weights are loaded.
