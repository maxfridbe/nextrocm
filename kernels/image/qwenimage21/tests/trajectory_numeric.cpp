// Standalone private Qwen packed linear validator; no production registration.
// Independent F64 dots, original-runtime bitwise parity, tails and paired events.
#include "../../../ns/ns.h"
#include "../trajectory.hpp"

#include <fstream>
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
 X(ns_copy_to) X(ns_copy_from) X(ns_sync)
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



using Axpy=int(*)(void*,float*,const float*,I,float);
static uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
static float from_bits(uint32_t x){float v;std::memcpy(&v,&x,4);return v;}
static float oracle(float x,float v,float step) {
    // A binary32 product fits exactly in binary64; each cast is a separate RNE.
    volatile float product=static_cast<float>(double(v)*double(step));
    return static_cast<float>(double(x)+double(product));
}
static void cpu() {
    const float step=from_bits(0xbf800001),v=from_bits(0x3f7ffffe),x=1.f;
    require(bits(oracle(x,v,step))==0,"separate rounding witness must be +0");
    require(bits(std::fma(step,v,x))!=bits(oracle(x,v,step)),"witness must distinguish fused FMA");
    std::vector<float>a(1024),b(1024);
    require(nrqi_trajectory::valid(a.data(),a.data(),b.data(),1024,step),"valid spans");
    require(!nrqi_trajectory::valid(a.data(),a.data(),a.data(),1024,step),"alias rejected");
    require(!nrqi_trajectory::valid(a.data(),a.data(),b.data(),1024,NAN),"NaN step rejected");
    require(!nrqi_trajectory::valid(a.data(),a.data(),b.data(),nrqi_trajectory::max_values+1,step),"cap rejected");
    nrqi_trajectory::Span s;
    require(!nrqi_trajectory::span(reinterpret_cast<void*>(UINTPTR_MAX-3),2,s),"span wrap rejected");
    std::cout<<"CPU PASS F64 two-round oracle and FMA distinction\n";
}
static void invalid(Api& a,Axpy fn) {
    Buffer x(a,F(1024,1)),v(a,F(1024,2));const auto before=x.read(),vb=v.read();
    auto call=[&](void* ctx,void* xp,const void* vp,I n,float step){
        require(fn(ctx,static_cast<float*>(xp),static_cast<const float*>(vp),n,step)!=0,"invalid accepted");a.sync();
        const auto after=x.read(),va=v.read();require(after==before&&va==vb,"invalid call changed storage");};
    call(nullptr,x.ptr(),v.ptr(),16,-.2f);call(a.gpu,nullptr,v.ptr(),16,-.2f);call(a.gpu,x.ptr(),nullptr,16,-.2f);
    call(a.gpu,x.ptr(),v.ptr(),0,-.2f);call(a.gpu,x.ptr(),v.ptr(),-1,-.2f);
    call(a.gpu,x.ptr(),v.ptr(),16,NAN);call(a.gpu,x.ptr(),v.ptr(),16,INFINITY);
    call(a.gpu,x.ptr(),x.ptr(),16,-.2f);call(a.gpu,x.ptr(),x.fp()+1,16,-.2f);
    call(a.gpu,static_cast<char*>(x.ptr())+1,v.ptr(),16,-.2f);call(a.gpu,x.ptr(),static_cast<char*>(v.ptr())+1,16,-.2f);
    call(a.gpu,reinterpret_cast<void*>(UINTPTR_MAX-3),v.ptr(),16,-.2f);
    x.guards();v.guards();std::cout<<"12 invalid entrypoints PASS\n";
}
static void run(Api& a,Axpy fn,I n,float step) {
    F x(size_t(n),0.f),v(size_t(n),0.f),expected(size_t(n),0.f);uint32_t seed=1234;
    for(I i=0;i<n;++i){seed=seed*1664525u+1013904223u;x[size_t(i)]=from_bits((seed&0x807fffff)|0x3f000000);seed=seed*1664525u+1013904223u;v[size_t(i)]=from_bits((seed&0x807fffff)|0x3e000000);}
    if(n>0){x[0]=1;v[0]=from_bits(0x3f7ffffe);} // detects fused arithmetic for witness step
    if(n>1){x[1]=from_bits(0x80000000);v[1]=from_bits(0x80000000);}
    if(n>2){x[2]=from_bits(1);v[2]=from_bits(2);} // subnormal behavior
    for(I i=0;i<n;++i)expected[size_t(i)]=oracle(x[size_t(i)],v[size_t(i)],step);
    Buffer dx(a,x),dv(a,v);a.check(fn(a.gpu,dx.fp(),dv.fp(),n,step),"axpy");a.sync();const auto got=dx.read();
    double maxerr=0;
    for(I i=0;i<n;++i){const size_t j=size_t(i);require(std::isfinite(got[j])&&bits(got[j])==bits(expected[j]),"full exact two-round mismatch");
        const double exact=double(x[j])+double(v[j])*double(step);const double error=std::abs(double(got[j])-exact);
        const double bound=std::ldexp(1.,-24)*(std::abs(double(v[j])*double(step))+std::abs(double(x[j])+double(oracle(0,v[j],step))))+std::ldexp(1.,-148);
        require(error<=bound,"F64 forward error bound");maxerr=std::max(maxerr,error);}
    const auto afterv=dv.read();require(std::memcmp(afterv.data(),v.data(),v.size()*4)==0,"velocity mutated");
    dx.guards();dv.guards();dx.write(x.data(),x.size()*4);a.check(fn(a.gpu,dx.fp(),dv.fp(),n,step),"repeat");a.sync();const auto repeat=dx.read();
    require(std::memcmp(repeat.data(),got.data(),got.size()*4)==0,"repeat differs");
    std::cout<<"PASS n="<<n<<" step_bits="<<bits(step)<<" maxF64="<<maxerr<<"\n";
}
int main(int argc,char** argv)try {
    cpu();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
    require(argc==3&&argv[1][0]=='/',"usage: trajectory_numeric ABS_RUNTIME GPU");Api a(argv[1],std::stoi(argv[2]));Axpy fn=nullptr;a.bind(fn,"nrqi_axpy_f32");
    using Stats=int(*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");uint64_t live=0,peak=0;a.check(stats(a.gpu,&live,&peak),"initial stats");require(live==0,"initial live");
    invalid(a,fn);
    for(I n:{I(1),I(255),I(256),I(257),I(16384),I(65536),I(262144)})
        for(float step:{0.f,-0.f,-.2f,-.1234567f,from_bits(0xbf800001),2.5f})run(a,fn,n,step);
    a.sync();a.check(stats(a.gpu,&live,&peak),"drop stats");require(live==0&&peak<4*1024*1024,"trajectory allocation leak/cap");
    std::cout<<"tracked_final="<<live<<" tracked_peak="<<peak<<"\nALL TRAJECTORY PASS (operator only)\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
