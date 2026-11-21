// Shared helpers for the vibe-audio engine: error handling, logging, timing.
#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define HIP_CHECK(expr)                                                                       \
    do {                                                                                      \
        hipError_t err_ = (expr);                                                             \
        if (err_ != hipSuccess) {                                                             \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err_) + \
                                     " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
        }                                                                                     \
    } while (0)

[[noreturn]] inline void fail(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    throw std::runtime_error(buf);
}

void nr_lowvram_progress(const char*, long, long);

// Log lines go to stderr. Lines starting with "PROGRESS" are parsed by the server.
inline void logf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    fflush(stderr);
}

inline void progress(const char *phase,long done,long total){nr_lowvram_progress(phase,done,total);}

inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
