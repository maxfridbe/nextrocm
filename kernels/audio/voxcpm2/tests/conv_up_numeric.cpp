// Standalone candidate validator; no production integration needed.
// From nextrocm/ (parent owns all GPU/SSH execution):
// c++ -std=c++17 -O2 -Wall -Wextra -Werror kernels/audio/voxcpm2/tests/conv_up_numeric.cpp -o /tmp/vcp-up-cpu
// /tmp/vcp-up-cpu --self-test
// hipcc -std=c++17 -O2 --offload-arch=gfx1035 -x hip kernels/audio/voxcpm2/tests/conv_up_numeric.cpp -ldl -o /tmp/vcp-up-gpu
// /tmp/vcp-up-gpu /absolute/libnextrocm-runtime.so GPU_INDEX [--profile]
// --profile adds all six exact shapes from voxcpm-short-profile-v2.log.
// GPU mode compares the ACTUAL loaded mm3 baseline bitwise, independent CPU
// F32 and double oracles, all guards and input immutability; event timing gives
// five individual baseline/candidate pairs after warmup. No deployment switch.
#include "../conv_up_tiled.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using I=int64_t;
using F=std::vector<float>;
[[maybe_unused]] constexpr float sentinel=-12345.25f;
[[maybe_unused]] constexpr size_t guard=16;
static void require(bool ok,const std::string& msg) { if(!ok) throw std::runtime_error(msg); }
static F values(size_t n,uint32_t seed=7) {
    F v(n);
    for(float& f:v) { seed=1664525u*seed+1013904223u; f=float(int((seed>>8)%2001)-1000)/731.f; }
    return v;
}
struct Shape { I b,ci,l,co,k,s; };
// Independent scatter definition. Iterating ci then k retains each output's
// exact ordered FMA chain, while deriving destinations from input coordinates.
static F scatter(Shape d,const F& x,const F& w,const F* bias) {
    const I lo=d.l*d.s;
    F y(size_t(d.b*d.co*lo),0.f);
    for(I b=0;b<d.b;++b) for(I ci=0;ci<d.ci;++ci) for(I k=0;k<d.k;++k)
        for(I co=0;co<d.co;++co) for(I t=0;t<d.l;++t) {
            const I dest=t*d.s+k;
            if(dest<lo) {
                float& a=y[(b*d.co+co)*lo+dest];
                a=std::fma(x[(b*d.ci+ci)*d.l+t],w[(ci*d.co+co)*d.k+k],a);
            }
        }
    for(I b=0;b<d.b;++b) for(I co=0;co<d.co;++co) for(I t=0;t<lo;++t)
        y[(b*d.co+co)*lo+t]+=bias?(*bias)[co]:0.f;
    return y;
}
// CPU execution of candidate's index transformation, independent of scatter.
static F phases(Shape d,const F& x,const F& w,const F* bias) {
    F y(size_t(d.b*d.co*d.l*d.s));
    for(I b=0;b<d.b;++b) for(I co=0;co<d.co;++co) for(I t=0;t<d.l;++t)
        for(I p=0;p<d.s;++p) {
            float a=0;
            for(I ci=0;ci<d.ci;++ci) for(I j=0;j<2;++j) if(t>=j)
                a=std::fma(x[(b*d.ci+ci)*d.l+t-j],w[(ci*d.co+co)*d.k+p+j*d.s],a);
            y[((b*d.co+co)*d.l+t)*d.s+p]=a+(bias?(*bias)[co]:0.f);
        }
    return y;
}
static void exact(const F& a,const F& b,const std::string& label) {
    require(a.size()==b.size(),label+": length");
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),label+": nonfinite");
        require(std::memcmp(&a[i],&b[i],4)==0,label+": bitwise mismatch at "+std::to_string(i));
    }
}
// Double gather oracle checks every small-case output; profile cases sample
// 97 evenly spaced outputs plus the entire first temporal boundary per channel.
// Bound is declared from F32 gamma_n * sum(abs(products)), not tuned to results.
static void double_check(Shape d,const F& x,const F& w,const F* bias,const F& y,bool sample) {
    auto at=[&](size_t idx) {
        const I lo=d.l*d.s, pos=I(idx)%lo,co=(I(idx)/lo)%d.co,b=I(idx)/(lo*d.co);
        double ref=bias?(*bias)[co]:0., magnitude=std::abs(ref);
        for(I ci=0;ci<d.ci;++ci) for(I k=0;k<d.k;++k) {
            const I u=pos-k;
            if(u<0 || u%d.s || u/d.s>=d.l) continue;
            const double product=double(x[(b*d.ci+ci)*d.l+u/d.s])*w[(ci*d.co+co)*d.k+k];
            ref+=product; magnitude+=std::abs(product);
        }
        const double neps=(2*d.ci+1)*std::ldexp(1.,-24);
        const double bound=neps/(1-neps)*magnitude+1e-30;
        require(std::isfinite(y[idx])&&std::isfinite(ref)&&std::abs(double(y[idx])-ref)<=bound,"double oracle at "+std::to_string(idx));
    };
    if(!sample) for(size_t i=0;i<y.size();++i) at(i);
    else {
        for(size_t i=0;i<97;++i) at(i*(y.size()-1)/96);
        for(I b=0;b<d.b;++b) for(I c=0;c<d.co;++c) for(I p=0;p<d.s;++p)
            at(size_t((b*d.co+c)*d.l*d.s+p));
    }
}
static void self_test() {
    for(I s:{2,5,6,8}) for(I l:{1,2,31,32,33,63,64,65}) for(bool bias:{false,true}) {
        Shape d{2,3,l,17,2*s,s};
        F x=values(size_t(d.b*d.ci*l)),w=values(size_t(d.ci*d.co*d.k),19),b=values(size_t(d.co),31);
        const F y=scatter(d,x,w,bias?&b:nullptr);
        exact(y,phases(d,x,w,bias?&b:nullptr),"CPU scatter/phase");
        double_check(d,x,w,bias?&b:nullptr,y,false);
    }
    exact(scatter({1,1,2,1,4,2},{2,3},{1,2,3,4},nullptr),{2,4,9,14},"hand convolution");
    const float e=std::ldexp(1.f,-23);
    F w{1,1,0,0,1-e,1-e,0,0};
    const F y=scatter({1,2,1,1,4,2},{-1,1+e},w,nullptr);
    require(y[0]==-std::ldexp(1.f,-46),"fused rounding witness");
    const float inf=std::numeric_limits<float>::infinity();
    exact(scatter({1,1,1,1,4,2},{2},{1,2,inf,inf},nullptr),{2,4},"skipped infinite tap");
    require(vcp_conv_up::supported(1,2048,44,1024,16,8),"profile supported");
    require(!vcp_conv_up::supported(0,2,3,4,4,2),"zero batch");
    require(!vcp_conv_up::supported(1,2,-1,4,4,2),"negative length");
    require(!vcp_conv_up::supported(1,2,3,4,5,2),"non-double kernel");
    require(!vcp_conv_up::supported(1,2,3,4,6,3),"unsupported stride");
    require(!vcp_conv_up::supported(INT64_MAX,INT64_MAX,INT64_MAX,INT64_MAX,4,2),"overflow");
    require(!vcp_conv_up::supported(65536,1,1,1,4,2),"grid y overflow");
    F src(32),dst(32),weights(32);
    require(vcp_conv_up::arguments(src.data(),weights.data(),nullptr,dst.data(),1,1,2,1,4,2),"valid pointers");
    require(!vcp_conv_up::arguments(src.data(),weights.data(),nullptr,src.data()+1,1,1,2,1,4,2),"partial overlap");
    require(!vcp_conv_up::arguments(nullptr,weights.data(),nullptr,dst.data(),1,1,2,1,4,2),"null input");
    std::cout<<"PASS CPU self-test (64 cases, hand arithmetic, rounding, skipped taps, guards)\n";
}

#ifdef __HIPCC__
#include "../../../ns/ns.h"
#include "../../../ns/ns_internal.hpp"
#include "../../minimaxmusic3/mm3.h"
#include <dlfcn.h>
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) \
 X(ns_audio_mm3_conv1d_lr)
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
static void hipcheck(hipError_t e,const char* msg) {
    require(e==hipSuccess,std::string(msg)+": "+hipGetErrorString(e));
}
struct Events {
    hipEvent_t start=nullptr,stop=nullptr;
    Events() {
        hipcheck(hipEventCreate(&start),"create start event");
        const auto e=hipEventCreate(&stop);
        if(e!=hipSuccess) { (void)hipEventDestroy(start); hipcheck(e,"create stop event"); }
    }
    // Cleanup cannot throw; creation, recording and synchronization are checked.
    ~Events() { (void)hipEventDestroy(stop); (void)hipEventDestroy(start); }
};
static void gpu_case(Api& a,Shape d,bool has_bias,bool profile,int special=0) {
    const size_t count=size_t(d.b*d.co*d.l*d.s);
    F x=values(size_t(d.b*d.ci*d.l)),w=values(size_t(d.ci*d.co*d.k),19),bias=values(size_t(d.co),31);
    if(special==1) {
        // Only the valid tap is finite; zero-padding instead of skipping would
        // introduce 0*infinity NaNs at t=0.
        std::fill(w.begin(),w.end(),std::numeric_limits<float>::infinity());
        for(I ci=0;ci<d.ci;++ci) for(I co=0;co<d.co;++co) for(I k=0;k<d.s;++k)
            w[(ci*d.co+co)*d.k+k]=.25f;
    } else if(special==2) {
        const float e=std::ldexp(1.f,-23);
        x={-1.f,1.f+e};w={1,1,0,0,1-e,1-e,0,0};
    } else if(special==3) {
        std::fill(x.begin(),x.end(),-0.f); std::fill(bias.begin(),bias.end(),-0.f);
    }
    Buffer dx(a,x),dw(a,w),db(a,bias),base(a,F(count,sentinel)),out(a,F(count,sentinel));
    auto baseline=[&] {
        a.check(a.ns_audio_mm3_conv1d_lr(a.gpu,1,dx.fp(),d.b,d.ci,d.l,dw.fp(),d.co,d.k,
            has_bias?db.fp():nullptr,d.s,1,0,d.k-d.s,base.fp(),d.l*d.s),"actual baseline");
    };
    auto candidate=[&] {
        hipcheck(vcp_conv_up::launch(dx.fp(),d.b,d.ci,d.l,dw.fp(),d.co,d.k,
            has_bias?db.fp():nullptr,d.s,out.fp(),a.gpu->q),"candidate");
    };
    hipcheck(hipSetDevice(a.gpu->dev),"select device");
    baseline(); candidate(); a.sync();
    const F actual=out.read(),original=base.read();
    exact(actual,original,"GPU baseline/candidate");
    if(!profile) exact(actual,scatter(d,x,w,has_bias?&bias:nullptr),"GPU/independent CPU FMA");
    double_check(d,x,w,has_bias?&bias:nullptr,actual,profile);
    dx.guards(); dw.guards(); db.guards(); base.guards(); out.guards();
    exact(dx.read(),x,"input unchanged");
    // Special skipped-tap case intentionally has infinity in weights.
    const F after_w=dw.read();
    require(std::memcmp(after_w.data(),w.data(),w.size()*4)==0,"weight changed");
    exact(db.read(),bias,"bias unchanged");
    std::cout<<"PASS shape=["<<d.b<<','<<d.ci<<','<<d.l<<','<<d.co<<','<<d.k<<','<<d.s
             <<"] bias="<<has_bias<<" special="<<special<<" outputs="<<count<<" bitwise=1 cpu="
             <<(profile?"sampled-double":"full-FMA-and-double")<<'\n';
    if(profile) {
        Events ev;
        auto time=[&](bool tiled) {
            hipcheck(hipEventRecord(ev.start,a.gpu->q),"record start");
            if(tiled) candidate(); else baseline();
            hipcheck(hipEventRecord(ev.stop,a.gpu->q),"record stop");
            hipcheck(hipEventSynchronize(ev.stop),"sync stop");
            float ms=0; hipcheck(hipEventElapsedTime(&ms,ev.start,ev.stop),"elapsed"); return ms;
        };
        F bt,ct;
        for(int i=0;i<5;++i) {
            float b,c;
            if(i%2) { c=time(true);b=time(false); } else { b=time(false);c=time(true); }
            bt.push_back(b);ct.push_back(c);
            std::cout<<"TIMING pair="<<i<<" baseline_ms="<<b<<" candidate_ms="<<c<<'\n';
        }
        std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());
        std::cout<<"MEDIAN baseline_ms="<<bt[2]<<" candidate_ms="<<ct[2]<<" speedup="<<bt[2]/ct[2]<<'\n';
        exact(out.read(),original,"repeated candidate"); base.guards(); out.guards();
    }
}
static void gpu_tests(Api& a,bool profile) {
    std::cout<<std::setprecision(9);
    hipDeviceProp_t prop{}; int runtime=0,driver=0;
    hipcheck(hipGetDeviceProperties(&prop,a.gpu->dev),"device properties");
    hipcheck(hipRuntimeGetVersion(&runtime),"runtime version");
    hipcheck(hipDriverGetVersion(&driver),"driver version");
    std::cout<<"DEVICE name="<<prop.name<<" arch="<<prop.gcnArchName
             <<" wave="<<prop.warpSize<<" runtime="<<runtime<<" driver="<<driver
             <<" candidate_scratch_bytes=0\n";
    for(I s:{2,5,6,8}) for(I l:{1,2,31,32,33,63,64,65}) for(bool b:{false,true})
        gpu_case(a,{2,3,l,17,2*s,s},b,false);
    // Include full input reduction lengths in CPU coverage at small output size.
    for(I ci:{64,128,256,512,1024,2048}) gpu_case(a,{1,ci,3,5,16,8},true,false);
    gpu_case(a,{1,3,1,5,16,8},false,false,1);
    gpu_case(a,{1,2,1,1,4,2},false,false,2);
    gpu_case(a,{2,3,33,5,4,2},true,false,3);
    Buffer x(a,F(128,1.f)),w(a,F(128,1.f)),out(a,F(128,sentinel));
    auto reject=[&](I b,I ci,I l,I co,I k,I s,const float* src,float* dst) {
        const auto e=vcp_conv_up::launch(src,b,ci,l,w.fp(),co,k,nullptr,s,dst,a.gpu->q);
        require(e==hipErrorInvalidValue,"invalid call accepted");
    };
    reject(0,1,1,1,4,2,x.fp(),out.fp());
    reject(1,1,-1,1,4,2,x.fp(),out.fp());
    reject(1,1,1,1,5,2,x.fp(),out.fp());
    reject(1,1,1,1,6,3,x.fp(),out.fp());
    reject(INT64_MAX,INT64_MAX,1,1,4,2,x.fp(),out.fp());
    reject(1,1,1,1,4,2,nullptr,out.fp());
    reject(1,1,1,1,4,2,x.fp(),nullptr);
    reject(1,1,2,1,4,2,x.fp(),x.fp()+1);
    reject(1,1,2,1,4,2,x.fp(),w.fp()+1);
    a.sync(); exact(out.read(),F(128,sentinel),"rejected call output");
    exact(x.read(),F(128,1.f),"rejected call input");
    exact(w.read(),F(128,1.f),"rejected call weights");x.guards();w.guards();out.guards();
    if(profile) {
        for(Shape d:std::initializer_list<Shape>{{1,2048,44,1024,16,8},{1,1024,352,512,12,6},
            {1,512,2112,256,10,5},{1,256,10560,128,4,2},{1,128,21120,64,4,2},{1,64,42240,32,4,2}})
            gpu_case(a,d,true,true);
    }
}

#endif

int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--self-test") { self_test(); return 0; }
#ifdef __HIPCC__
        if(argc!=3&&argc!=4) throw std::runtime_error("usage: validator LIB GPU_INDEX [--profile] | --self-test");
        if(argc==4) require(std::string(argv[3])=="--profile","unknown option");
        size_t used=0; const int dev=std::stoi(argv[2],&used);
        require(dev>=0&&used==std::strlen(argv[2]),"GPU index");
        self_test(); Api api(argv[1],dev); gpu_tests(api,argc==4);
        std::cout<<"PASS GPU operator checks\n";
        return 0;
#else
        throw std::runtime_error("CPU-only build: use --self-test; compile with hipcc for GPU checks");
#endif
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
