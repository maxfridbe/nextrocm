// Native packaging smoke test: no model inference or numerical admission.
// c++ -std=c++17 -O2 -Wall -Wextra -Werror library_smoke.cpp -ldl -o library-smoke
#include <dlfcn.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

struct Device {
    void* gpu = nullptr;
    void* buffer = nullptr;
    int (*release)(void*, void*) = nullptr;
    void (*close)(void*) = nullptr;
    ~Device() { if (buffer) release(gpu, buffer); if (gpu) close(gpu); }
};

int main(int argc, char** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: library-smoke RUNTIME SILO_DIRECTORY EXPECTED_NAMES_FILE");
        std::ifstream expected_file(argv[3]);
        if (!expected_file) throw std::runtime_error("cannot read expected library inventory");
        std::set<std::string> expected, actual;
        for (std::string name; std::getline(expected_file, name);) {
            if (name.empty() || std::filesystem::path(name).filename() != name ||
                std::filesystem::path(name).extension() != ".so" || !expected.insert(name).second)
                throw std::runtime_error("invalid or duplicate expected library name");
        }
        if (expected.empty() || expected_file.bad()) throw std::runtime_error("invalid expected library inventory");
        for (const auto& entry : std::filesystem::directory_iterator(argv[2]))
            if (entry.path().extension() == ".so") actual.insert(entry.path().filename().string());
        if (actual != expected) throw std::runtime_error("architecture library inventory differs from expected manifest");
        void* runtime = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (!runtime) throw std::runtime_error(dlerror());
        using Props = int (*)(int, uint64_t*, int*, int*, char*, size_t);
        auto props = reinterpret_cast<Props>(dlsym(runtime, "ns_gpu_props"));
        if (!props) throw std::runtime_error("missing ns_gpu_props");
        uint64_t total = 0;
        int wave = 0, integrated = -1;
        char gfx[256] = {};
        if (props(0, &total, &wave, &integrated, gfx, sizeof(gfx)) || !total ||
            (wave != 32 && wave != 64) || (integrated != 0 && integrated != 1) ||
            !std::memchr(gfx, 0, sizeof(gfx)) || std::string(gfx).rfind("gfx", 0) != 0)
            throw std::runtime_error("invalid reported AMD device properties");
        // Keep every handle alive: registration and shared dependencies must survive
        // the entire inspection just as they do in the Rust process-lifetime loader.
        std::vector<void*> handles;
        for (const auto& entry : std::filesystem::directory_iterator(argv[2])) {
            if (entry.path().extension() != ".so") continue;
            void* library = dlopen(entry.path().c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!library) throw std::runtime_error(entry.path().string() + ": " + dlerror());
            handles.push_back(library);
            std::cout << "LOADED " << entry.path().filename().string() << '\n';
        }
        if (handles.empty()) throw std::runtime_error("no architecture libraries found");
        // Keep a small real allocation visible to the external DRM sampler.
        // This is an allocation/lifetime smoke, not an engine numerical test.
        auto symbol = [&](const char* name) {
            void* p = dlsym(runtime, name);
            if (!p) throw std::runtime_error(std::string("missing ") + name);
            return p;
        };
        auto open = reinterpret_cast<int (*)(int, void**)>(symbol("ns_gpu_open"));
        auto alloc = reinterpret_cast<int (*)(void*, size_t, void**)>(symbol("ns_alloc"));
        auto sync = reinterpret_cast<int (*)(void*)>(symbol("ns_sync"));
        auto stats = reinterpret_cast<int (*)(void*, uint64_t*, uint64_t*)>(symbol("nr_allocation_stats"));
        Device device;
        device.close = reinterpret_cast<void (*)(void*)>(symbol("ns_gpu_close"));
        device.release = reinterpret_cast<int (*)(void*, void*)>(symbol("ns_free"));
        if (open(0, &device.gpu) || alloc(device.gpu, 65536, &device.buffer) || sync(device.gpu))
            throw std::runtime_error("allocation smoke failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(350));
        if (device.release(device.gpu, device.buffer)) throw std::runtime_error("free failed");
        device.buffer = nullptr;
        uint64_t live = 0, peak = 0;
        if (stats(device.gpu, &live, &peak) || live != 0 || peak != 65536)
            throw std::runtime_error("allocation lifetime counters differ");
        std::cout << "PASS libraries=" << handles.size() << " gfx=" << gfx
                  << " wave=" << wave << " integrated=" << integrated
                  << " total_bytes=" << total << " live=" << live << " peak=" << peak << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
