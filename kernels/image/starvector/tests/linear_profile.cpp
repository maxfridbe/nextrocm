// Actual StarVector TG shapes; independent double oracle, guards, repeat bitwise.
// Diagnostic baseline only, no candidate dispatch or production changes.
#include "../../../ns/ns.h"
#include "../starvector.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
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

extern "C" int nsd_linear(void*,const void*,int,int64_t,int64_t,const void*,int64_t,const float*,void*,int);
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
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) X(nsd_linear) X(ns_stamp) X(ns_elapsed)
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


// Test data are exact binary fractions. This encoder handles only our bounded
// integers / 256; the independent reader covers every finite half bit pattern.
static uint16_t half_fraction(int value) {
    const uint16_t sign=value<0?0x8000:0;
    unsigned v=unsigned(value<0?-value:value);
    if(!v)return sign;
    unsigned exponent=0;while((1u<<(exponent+1))<=v)++exponent;
    return uint16_t(sign|((exponent+7)<<10)|((v-(1u<<exponent))<<(10-exponent)));
}
static double from_half(uint16_t h) {
    const unsigned e=(h>>10)&31,m=h&1023;
    require(e!=31,"nonfinite half");
    const double v=e?std::ldexp(double(1024+m),int(e)-25):std::ldexp(double(m),-24);
    return h&0x8000?-v:v;
}
static int sample(uint32_t& seed) {seed=seed*1664525u+1013904223u;return int((seed>>9)%129)-64;}
static std::vector<unsigned char> bytes(const Buffer& b) {
    std::vector<unsigned char> out(b.bytes);
    b.a.check(b.a.ns_copy_from(b.a.gpu,out.data(),b.ptr(),b.bytes),"read output");b.a.sync();return out;
}
static double element(const std::vector<unsigned char>& b,size_t i,bool head) {
    if(head){float v;std::memcpy(&v,b.data()+i*4,4);return v;}
    uint16_t h;std::memcpy(&h,b.data()+i*2,2);return from_half(h);
}
static void run_case(Api& a,I m,I k,I n,bool head,unsigned repeats) {
    const size_t xb=size_t(m*k)*2,wb=size_t(n*k)*2,ob=size_t(m*n)*(head?4:2);
    require(xb+wb+ob+size_t(n)*4+512<=256u*1024u*1024u,"256 MiB microbenchmark cap");
    std::vector<uint16_t> x(size_t(m*k)),w(size_t(n*k));
    F wf(static_cast<size_t>(n*k)); F bias(static_cast<size_t>(n));uint32_t seed=uint32_t(71+m*3+k+n);
    for(auto& v:x)v=half_fraction(sample(seed));
    for(size_t i=0;i<w.size();++i){const int v=sample(seed);w[i]=half_fraction(v);wf[i]=float(v)/256.0f;}
    for(auto& v:bias)v=float(sample(seed))/256.0f;
    Buffer dx(a,xb),dw(a,wb),db(a,bias),out(a,ob);
    dx.write(x.data(),xb);dw.write(w.data(),wb);
    auto launch=[&](){a.check(a.nsd_linear(a.gpu,dx.ptr(),1,m,k,dw.ptr(),n,head?nullptr:db.fp(),out.ptr(),head?0:1),"linear");};
    launch();const auto baseline=bytes(out);
    D xd(x.size());
    for(size_t i=0;i<x.size();++i)xd[i]=from_half(x[i]);
    double maximum=0;
    for(I r=0;r<m;++r)for(I c=0;c<n;++c){
        double expected=head?0:double(bias[size_t(c)]);
        for(I j=0;j<k;++j)expected+=xd[size_t(r*k+j)]*double(wf[size_t(c*k+j)]);
        const double actual=element(baseline,size_t(r*n+c),head),error=std::abs(actual-expected);
        require(std::isfinite(actual),"nonfinite linear output");
        // Fixed a priori tolerances, including half output rounding.
        require(error <= (head?4e-5:0.002)+(head?5e-4:0.002)*std::abs(expected),"independent double linear mismatch");
        maximum=std::max(maximum,error);
    }
    for(unsigned i=0;i<4;++i) { launch(); }
    a.sync();
    std::vector<double> device,host;
    for(unsigned i=0;i<repeats;++i){
        int64_t begin=0,end=0;double ns=0;auto wall=std::chrono::steady_clock::now();
        a.check(a.ns_stamp(a.gpu,&begin),"stamp begin");launch();
        a.check(a.ns_stamp(a.gpu,&end),"stamp end");a.check(a.ns_elapsed(a.gpu,begin,end,&ns),"elapsed");
        require(std::isfinite(ns)&&ns>=0,"invalid event time");device.push_back(ns);
        host.push_back(std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-wall).count());
        require(bytes(out)==baseline,"repeat differs bitwise from baseline");
    }
    dx.guards();dw.guards();db.guards();out.guards();
    std::sort(device.begin(),device.end());std::sort(host.begin(),host.end());
    std::cout<<"PASS m="<<m<<" k="<<k<<" n="<<n<<" out="<<(head?"f32":"f16")<<" repeats="<<repeats
      <<" max_abs="<<maximum<<" gpu_p50_ns="<<device[(device.size()-1)/2]<<" gpu_p95_ns="<<device[(device.size()-1)*95/100]
      <<" host_p50_ns="<<host[(host.size()-1)/2]<<" full_cpu_oracle=pass baseline_bitwise=pass guards=pass\n"<<std::flush;
}
int main(int argc,char** argv) {
    try {
        require(argc==4,"usage: starvector-linear-profile ABS_LIBRARY GPU_INDEX REPEATS(1..32); NS_PROFILE=gpu required");
        require(argv[1][0]=='/',"runtime must be absolute");
        require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","requires NS_PROFILE=gpu");
        const int repeats=std::stoi(argv[3]);require(repeats>=1&&repeats<=32,"repeat bound");
        // Independent sanity checks of generated half fractions before using GPU.
        for(int i=-64;i<=64;++i)require(from_half(half_fraction(i))==double(i)/256,"half fraction self-test");
        Api a(argv[1],std::stoi(argv[2]));
        using Stats=int (*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
        uint64_t live=0,peak=0;a.check(stats(a.gpu,&live,&peak),"initial stats");require(live==0,"nonzero initial tracked bytes");
        for(unsigned i=0;i<8;++i){int64_t begin=0,end=0;double ns=0;
            a.check(a.ns_stamp(a.gpu,&begin),"empty begin");a.check(a.ns_stamp(a.gpu,&end),"empty end");
            a.check(a.ns_elapsed(a.gpu,begin,end,&ns),"empty elapsed");std::cout<<"calibration_empty_event_ns="<<ns<<'\n';}
        for(I m:{I(1),I(2),I(8)})for(const auto& shape:{std::pair<I,I>{2048,2048},{2048,128},{2048,8192},{8192,2048},{2048,49156}}){
            run_case(a,m,shape.first,shape.second,shape.second==49156,unsigned(repeats));
            a.check(stats(a.gpu,&live,&peak),"case stats");require(live==0,"tracked allocation leak");require(peak<=256u*1024u*1024u,"tracked peak exceeds cap");
        }
        std::cout<<"PASS 15 actual shapes; zero live bytes; peak="<<peak<<"; synchronized microbench only, no global performance claim\n";
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
