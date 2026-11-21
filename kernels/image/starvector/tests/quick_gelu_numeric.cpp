// Independent double QuickGELU oracle: signs, tails, saturation, guards and cleanup.
#include "../../../ns/ns.h"
#include "../starvector.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using I = int64_t;
using F = std::vector<float>;
using D = std::vector<double>;
static constexpr float sentinel = -12345.25f;
static constexpr size_t guard = 16;
static void require(bool yes, const std::string& what) {
    if (!yes) throw std::runtime_error(what);
}
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) X(nrsv_quick_gelu_f32)
struct Api {
    void* lib = nullptr;
    ns_gpu* gpu = nullptr;
#define FIELD(name) decltype(&::name) name = nullptr;
    SYMBOLS(FIELD)
#undef FIELD
    template<class T> void bind(T& fn, const char* name) {
        void* address = dlsym(lib, name);
        require(address != nullptr, std::string("missing native symbol: ") + name);
        static_assert(sizeof(fn) == sizeof(address), "POSIX function pointer size");
        std::memcpy(&fn, &address, sizeof(fn));
    }
    Api(const char* path, int index) {
        lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!lib) throw std::runtime_error(dlerror());
        try {
#define BIND(name) bind(name, #name);
            SYMBOLS(BIND)
#undef BIND
            check(ns_gpu_open(index, &gpu), "ns_gpu_open");
        } catch (...) { dlclose(lib); lib = nullptr; throw; }
    }
    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;
    ~Api() { if (gpu) { ns_sync(gpu); ns_gpu_close(gpu); } if (lib) dlclose(lib); }
    void check(int status, const char* where) {
        if (status) {
            const char* error = ns_last_error();
            throw std::runtime_error(std::string(where) + ": " + (error ? error : "unknown native error"));
        }
    }
    void sync() { check(ns_sync(gpu), "ns_sync"); }
};
struct Buffer {
    Api& a;
    void* allocation = nullptr;
    size_t bytes;
    Buffer(Api& api, size_t length) : a(api), bytes(length) {
        a.check(a.ns_alloc(a.gpu, bytes + 2*guard*sizeof(float), &allocation), "ns_alloc");
        try {
            std::vector<unsigned char> init(bytes + 2*guard*sizeof(float), 0);
            for (size_t i = 0; i < guard; ++i) {
                std::memcpy(init.data()+i*sizeof(float), &sentinel, sizeof(float));
                std::memcpy(init.data()+guard*sizeof(float)+bytes+i*sizeof(float), &sentinel, sizeof(float));
            }
            a.check(a.ns_copy_to(a.gpu, allocation, init.data(), init.size()), "guard upload");
            a.sync();
        } catch (...) { a.ns_free(a.gpu, allocation); allocation = nullptr; throw; }
    }
    Buffer(Api& api, const F& host) : Buffer(api, host.size()*sizeof(float)) { write(host.data(), bytes); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (allocation) { a.ns_sync(a.gpu); a.ns_free(a.gpu, allocation); } }
    void* ptr() const { return static_cast<unsigned char*>(allocation)+guard*sizeof(float); }
    float* fp() const { return static_cast<float*>(ptr()); }
    void write(const void* src, size_t n) {
        require(n == bytes, "host upload size mismatch");
        a.check(a.ns_copy_to(a.gpu, ptr(), src, n), "ns_copy_to");
        a.sync(); // Temporary upload sources must outlive the queued transfer.
    }
    F read() const {
        require(bytes % sizeof(float) == 0, "not an f32 buffer");
        F host(bytes/sizeof(float));
        a.check(a.ns_copy_from(a.gpu, host.data(), ptr(), bytes), "ns_copy_from");
        a.sync();
        return host;
    }
    void guards() const {
        F ends(guard*2);
        a.check(a.ns_copy_from(a.gpu, ends.data(), allocation, guard*sizeof(float)), "prefix read");
        auto* suffix = static_cast<unsigned char*>(ptr())+bytes;
        a.check(a.ns_copy_from(a.gpu, ends.data()+guard, suffix, guard*sizeof(float)), "suffix read");
        a.sync();
        for (float v : ends) require(v == sentinel, "device buffer guard overwritten");
    }
};

static void run_case(Api& a, size_t count) {
    F input(count);
    uint32_t seed=73;
    for(size_t i=0;i<count;++i) {
        seed=seed*1664525u+1013904223u;
        input[i]=(int((seed>>8)%200001)-100000)/4096.0f;
    }
    const float edge[]={0.0f,-0.0f,1e-30f,-1e-30f,0.5f,-0.5f,1,-1,8,-8,80,-80,1000,-1000};
    for(size_t i=0;i<std::min(count,sizeof(edge)/sizeof(edge[0]));++i)input[i]=edge[i];
    Buffer x(a,input);
    a.check(a.nrsv_quick_gelu_f32(a.gpu,x.fp(),I(count)),"QuickGELU");
    const F actual=x.read(); double maximum=0, squared=0;
    for(size_t i=0;i<count;++i) {
        const double v=input[i],z=1.702*v;
        const double sigmoid=z>=0?1/(1+std::exp(-z)):std::exp(z)/(1+std::exp(z));
        const double want=v*sigmoid,err=std::abs(actual[i]-want);
        require(std::isfinite(actual[i]),"nonfinite QuickGELU result");
        require(err<=2e-6+2e-6*std::abs(want),"independent QuickGELU mismatch");
        maximum=std::max(maximum,err);squared+=err*err;
    }
    x.guards();
    std::cout<<"PASS count="<<count<<" max_abs="<<maximum<<" rms="<<std::sqrt(squared/count)<<'\n';
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: starvector-quick-gelu-numeric ABS_LIBRARY GPU_INDEX");
        Api a(argv[1],std::stoi(argv[2]));
        for(size_t n:{size_t(1),size_t(14),size_t(255),size_t(256),size_t(257),size_t(4097),size_t(257*4096)})run_case(a,n);
        {
            Buffer x(a,F(16,sentinel));
            for(I n:{I(0),I(-1),I(257*4096+1),INT64_MAX})
                require(a.nrsv_quick_gelu_f32(a.gpu,x.fp(),n)!=0,"invalid extent accepted");
            require(a.nrsv_quick_gelu_f32(nullptr,x.fp(),1)!=0,"null context accepted");
            require(a.nrsv_quick_gelu_f32(a.gpu,nullptr,1)!=0,"null data accepted");
            for(float v:x.read())require(v==sentinel,"invalid call wrote data");
            x.guards();
        }
        using Stats=int (*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
        uint64_t live=0,peak=0;a.check(stats(a.gpu,&live,&peak),"stats");require(live==0,"tracked allocation leak");
        std::cout<<"PASS seven independent shapes + invalid calls; zero live bytes; peak="<<peak<<'\n';
        return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
