// Standalone private Qwen packed linear validator; no production registration.
// Independent F64 dots, original-runtime bitwise parity, tails and paired events.
#include "../../../ns/ns.h"
#include "../packed_linear.hpp"
#include "oracle/ggml_rows.hpp"
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



using Packed=int(*)(void*,const void*,I,I,const void*,int,I,void*,int,I);
static uint32_t rnd(uint32_t& s){s=s*1664525u+1013904223u;return s;}
static std::vector<uint8_t> synthetic(int type,I k,I n) {
    const size_t bs=type==12?144:210;std::vector<uint8_t> b(size_t(k/256*n)*bs);
    uint32_t seed=1234;
    for(size_t at=0;at<b.size();at+=bs) {
        for(size_t j=0;j<bs;++j)b[at+j]=uint8_t(rnd(seed)>>24);
        // Finite non-power-of-two scales, signed Q6 scales, all quant/high bits.
        const uint16_t scale=uint16_t(0x1400+(rnd(seed)&1023));
        std::memcpy(b.data()+at+(type==12?0:208),&scale,2);
        if(type==12){const uint16_t mn=uint16_t(0x1000+(rnd(seed)&1023));std::memcpy(b.data()+at+2,&mn,2);}
    }
    return b;
}
static F decode(const std::vector<uint8_t>& b,int type,I k,I n) {
    F f(size_t(k*n));
    if(type==12) {
        std::vector<qi_oracle::block_q4_K> blocks(b.size()/144);std::memcpy(blocks.data(),b.data(),b.size());
        qi_oracle::dequantize_row_q4_K(blocks.data(),f.data(),k*n);
    } else {
        std::vector<qi_oracle::block_q6_K> blocks(b.size()/210);std::memcpy(blocks.data(),b.data(),b.size());
        qi_oracle::dequantize_row_q6_K(blocks.data(),f.data(),k*n);
    }
    for(float& x:f){x=qi_oracle::half_float(qi_oracle::half_bits(x));require(std::isfinite(x),"nonfinite decoded weight");}
    return f;
}
static void cpu_tests() {
    require(qi_oracle::half_float(0x3c00)==1 && qi_oracle::half_bits(1)==0x3c00,"half one");
    require(qi_oracle::half_bits(1.00048828125f)==0x3c00,"half RNE tie");
    for(int type:{12,14}) {
        auto b=synthetic(type,256,1);
        std::fill(b.begin(),b.end(),0);
        if(type==12) { // all q=3, scale=1,min=0,d=1 -> every output3
            uint16_t one=0x3c00;std::memcpy(b.data(),&one,2);
            for(int j=0;j<4;++j){b[4+j]=1;b[12+j]=1;}
            std::fill(b.begin()+16,b.end(),0x33);
        } else { // unsigned q=32, scales nonzero -> all zero
            uint16_t one=0x3c00;std::memcpy(b.data()+208,&one,2);
            std::fill(b.begin()+128,b.begin()+192,0xaa);
            std::fill(b.begin()+192,b.begin()+208,7);
        }
        const auto f=decode(b,type,256,1);for(float x:f)require(x==(type==12?3.f:0.f),"oracle known block");
    }
    std::vector<uint16_t>x(32*512),w(200000),y(32*512);
    auto valid=[&](I m,I k,int type,I n,int odt,I stride){return nrqi_packed::valid(x.data(),x.data(),m,k,w.data(),type,n,y.data(),odt,stride);};
    require(valid(17,256,12,19,1,64),"valid tail shape");
    for(auto bad:{0,-1,257,16385})require(!valid(17,bad,12,19,1,64),"bad K");
    require(!valid(17,256,13,19,1,64)&&!valid(17,256,12,19,2,64)&&!valid(17,256,12,19,1,18),"type stride guards");
    require(!nrqi_packed::valid(x.data(),x.data(),1,256,w.data(),12,1,x.data(),1,1),"output alias");
    nrqi_packed::Span span;
    require(!nrqi_packed::span(reinterpret_cast<void*>(UINTPTR_MAX-1),8,2,2,span),"wrap guard");
    require(!nrqi_packed::span(reinterpret_cast<void*>(uintptr_t(3)),8,2,2,span),"alignment guard");
    std::cout<<"CPU oracle known-block/half/geometry guards PASS; no GPU claim\n";
}
static std::vector<uint8_t> bytes(Api& a,const Buffer& b) {
    std::vector<uint8_t> v(b.bytes);a.check(a.ns_copy_from(a.gpu,v.data(),b.ptr(),v.size()),"read");a.sync();return v;
}
static double timed(Api& a,const std::function<void()>& f) {
    int64_t t0,t1;a.sync();a.check(a.ns_stamp(a.gpu,&t0),"stamp");f();a.check(a.ns_stamp(a.gpu,&t1),"stamp");a.sync();
    double ns;a.check(a.ns_elapsed(a.gpu,t0,t1,&ns),"elapsed");return ns/1e6;
}
static void run(Api& a,Packed fn,int type,I m,I k,I n,int odt,I stride,
                const std::vector<uint8_t>& packed,int repeats) {
    require(m*k+n*k+m*stride < 100000000,"bounded validator elements");
    auto wf=decode(packed,type,k,n);std::vector<uint16_t> hw(wf.size()),hx(size_t(m*k));
    for(size_t i=0;i<hw.size();++i)hw[i]=qi_oracle::half_bits(wf[i]);
    uint32_t seed=42;F xf(hx.size());for(size_t i=0;i<hx.size();++i){hx[i]=uint16_t((rnd(seed)&0x83ff)|0x3000);xf[i]=qi_oracle::half_float(hx[i]);}
    const size_t es=odt?2:4;
    Buffer x(a,hx.size()*2),w(a,packed.size()),expanded(a,hw.size()*2),out(a,size_t(m*stride)*es),base(a,size_t(m*n)*es);
    x.write(hx.data(),x.bytes);w.write(packed.data(),w.bytes);expanded.write(hw.data(),expanded.bytes);
    std::vector<uint8_t> initial(out.bytes,0x5a);out.write(initial.data(),initial.size());
    auto candidate=[&]{a.check(fn(a.gpu,x.ptr(),m,k,w.ptr(),type,n,out.ptr(),odt,stride),"packed linear");};
    auto baseline=[&]{a.check(a.nsd_linear(a.gpu,x.ptr(),1,m,k,expanded.ptr(),n,nullptr,base.ptr(),odt),"half baseline");};
    baseline();candidate();a.sync();auto y=bytes(a,out),ref=bytes(a,base);
    double worst=0;
    for(I r=0;r<m;++r)for(I c=0;c<stride;++c) {
        const size_t at=size_t(r*stride+c)*es;
        if(c>=n){require(std::memcmp(y.data()+at,initial.data()+at,es)==0,"stride hole modified");continue;}
        require(std::memcmp(y.data()+at,ref.data()+size_t(r*n+c)*es,es)==0,"full baseline bitwise mismatch");
        float value;if(odt){uint16_t h;std::memcpy(&h,y.data()+at,2);value=qi_oracle::half_float(h);}else std::memcpy(&value,y.data()+at,4);
        require(std::isfinite(value),"nonfinite output");
        double dot=0,absum=0;for(I j=0;j<k;++j){double v=double(xf[size_t(r*k+j)])*wf[size_t(c*k+j)];dot+=v;absum+=std::abs(v);}
        const double eps=std::ldexp(1.,-24),gamma=(k*eps)/(1-k*eps);
        const double bound=gamma*absum+(odt?std::abs(dot)*std::ldexp(1.,-10)+std::ldexp(1.,-24):0)+1e-12;
        require(std::abs(double(value)-dot)<=bound,"independent F64 forward-error bound failed");
        worst=std::max(worst,std::abs(double(value)-dot));
    }
    x.guards();w.guards();expanded.guards();out.guards();base.guards();
    candidate();a.sync();require(bytes(a,out)==y,"repeat mismatch");
    double bt=0,ct=0;for(int i=0;i<repeats;++i){if(i%2){ct+=timed(a,candidate);bt+=timed(a,baseline);}else{bt+=timed(a,baseline);ct+=timed(a,candidate);}}
    std::cout<<"PASS type="<<type<<" M="<<m<<" K="<<k<<" N="<<n<<" odt="<<odt<<" stride="<<stride
             <<" maxF64="<<worst<<" half_baseline_ms="<<bt/repeats<<" packed_ms="<<ct/repeats<<"\n";
}
static std::vector<uint8_t> fixture(const char* path,int type,I k,I n) {
    require((type==12||type==14)&&k>=256&&k<=16384&&k%256==0&&n>0&&n<=16384,"fixture geometry");
    const size_t length=size_t(k/256*n)*(type==12?144:210);require(length<=64*1024*1024,"packed fixture cap64MiB");
    std::ifstream f(path,std::ios::binary|std::ios::ate);require(bool(f)&&f.tellg()==std::streamoff(length),"fixture length");f.seekg(0);
    std::vector<uint8_t> b(length);f.read(reinterpret_cast<char*>(b.data()),std::streamsize(length));require(bool(f),"fixture read");return b;
}
static void invalid(Api& a,Packed fn) {
    Buffer x(a,1024),w(a,1024),y(a,1024);
    const auto original=bytes(a,y);
    auto check=[&](void* ctx,const void* xp,I m,I k,const void* wp,int type,I n,void* yp,int odt,I stride){
        require(fn(ctx,xp,m,k,wp,type,n,yp,odt,stride)!=0,"invalid call accepted");
        a.sync();require(bytes(a,y)==original,"invalid call wrote output");};
    check(nullptr,x.ptr(),1,256,w.ptr(),12,1,y.ptr(),0,1);
    check(a.gpu,nullptr,1,256,w.ptr(),12,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,nullptr,12,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),12,1,nullptr,0,1);
    check(a.gpu,x.ptr(),0,256,w.ptr(),12,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,255,w.ptr(),12,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),13,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),12,1,y.ptr(),2,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),12,2,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),12,1,x.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),12,1,w.ptr(),0,1);
    check(a.gpu,static_cast<uint8_t*>(x.ptr())+1,1,256,w.ptr(),12,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,static_cast<uint8_t*>(w.ptr())+1,12,1,y.ptr(),0,1);
    check(a.gpu,x.ptr(),1,256,w.ptr(),12,1,static_cast<uint8_t*>(y.ptr())+1,0,1);
    check(a.gpu,reinterpret_cast<void*>(UINTPTR_MAX-1),1,256,w.ptr(),12,1,y.ptr(),0,1);
    x.guards();w.guards();y.guards();std::cout<<"15 invalid entrypoint cases PASS\n";
}
int main(int argc,char** argv)try {
    if(argc==2 && std::string(argv[1])=="--self-test"){cpu_tests();return 0;}
    if(argc==6 && std::string(argv[1])=="--oracle-only") {
        const int type=std::stoi(argv[2]);const I k=std::stoll(argv[3]),n=std::stoll(argv[4]);
        const auto f=decode(fixture(argv[5],type,k,n),type,k,n);double sum=0;float maxabs=0;
        for(float v:f){sum+=v;maxabs=std::max(maxabs,std::abs(v));}
        std::cout<<"CPU decoded finite F16 weights="<<f.size()<<" sum="<<std::setprecision(17)<<sum<<" maxabs="<<maxabs<<" PASS\n";return 0;
    }
    require(argc==4||argc==8,"usage: packed_numeric ABS_RUNTIME GPU REPEATS [TYPE K N PACKED_FILE]; NS_PROFILE=gpu required");
    require(argv[1][0]=='/',"absolute runtime required");require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","NS_PROFILE=gpu required");
    int reps=std::stoi(argv[3]);require(reps>=1&&reps<=16,"bounded repeats1..16");
    Api a(argv[1],std::stoi(argv[2]));Packed fn=nullptr;a.bind(fn,"nrqi_packed_linear");
    cpu_tests();invalid(a,fn);
    if(argc==8){int type=std::stoi(argv[4]);I k=std::stoll(argv[5]),n=std::stoll(argv[6]);
        const auto b=fixture(argv[7],type,k,n);
        for(I m:{I(1),I(17),I(64)})for(int odt:{0,1})run(a,fn,type,m,k,n,odt,n+7,b,reps);
    } else for(int type:{12,14})for(I k:{I(256),I(512)})for(I m:{I(1),I(17)})for(I n:{I(1),I(19)})for(int odt:{0,1})
        run(a,fn,type,m,k,n,odt,n+7,synthetic(type,k,n),reps);
    a.sync();std::cout<<"ALL PASS (operator only; fixture identity must be pinned by runner)\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
