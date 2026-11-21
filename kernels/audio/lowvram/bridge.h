#pragma once
#include <memory>
#include <string>
#include "models.h"
#include "wav.h"

// Resident native model backend. Rust owns the GPU, lifecycle, requests and progress.
struct SttBackend {
    virtual ~SttBackend() = default;
    virtual SttResult transcribe(Audio audio,const SttOptions& options)=0;
    virtual std::string info() const=0;
};
std::unique_ptr<SttBackend> make_canary(const SttOptions&);
std::unique_ptr<SttBackend> make_whisper(const SttOptions&);
void* nr_lowvram_allocate(size_t bytes);
void nr_lowvram_free(void* ptr);
