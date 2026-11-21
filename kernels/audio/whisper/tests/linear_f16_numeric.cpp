// Standalone private F16 matrix validator. Parent serializes all GPU/SSH work.
// From nextrocm/:
// c++ -std=c++17 -O2 -Wall -Wextra -Werror kernels/audio/whisper/tests/linear_f16_numeric.cpp -o /tmp/whisper-linear-cpu
// /tmp/whisper-linear-cpu --self-test
// hipcc -std=c++17 -O2 --offload-arch=gfx1035 -x hip kernels/audio/whisper/tests/linear_f16_numeric.cpp -ldl -o /tmp/whisper-linear-gpu
// /tmp/whisper-linear-gpu ABS_RUNTIME GPU_INDEX [--profile]
// --profile adds the three Whisper encoder linear shapes, five event pairs each. CPU
// oracles decode the exact input half bits independently of HIP. Actual loaded
// nsd_linear is the GPU baseline; every output must match bitwise, including
// signed zeros. No conversion/staging, model trajectory or load timing claimed.
#include "../linear_f16_tiled.hpp"
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
using H=std::vector<uint16_t>;
[[maybe_unused]] constexpr float sentinel=-12345.25f;
[[maybe_unused]] constexpr size_t guard=16;
static void require(bool ok,const std::string& msg) { if(!ok)throw std::runtime_error(msg); }
struct Shape { I m,n,k; };
static float decode(uint16_t h) {
    const unsigned e=(h>>10)&31,frac=h&1023;
    float v;
    if(e==0) v=std::ldexp(float(frac),-24);
    else if(e==31) v=frac?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity();
    else v=std::ldexp(float(1024+frac),int(e)-25);
    return h&0x8000?-v:v;
}
// Independent IEEE binary32 -> binary16 integer round-to-nearest-even.
// No HIP conversion or tile code is used by this oracle.
static uint16_t encode(float f) {
    uint32_t u;std::memcpy(&u,&f,4);
    const uint16_t sign=uint16_t((u>>16)&0x8000);
    const unsigned exp=(u>>23)&255,frac=u&0x7fffff;
    if(exp==255)return uint16_t(sign|0x7c00|(frac?0x200:0));
    if(exp<102)return sign;
    if(exp>142)return uint16_t(sign|0x7c00);
    unsigned mant=frac|0x800000,shift=exp<113?126-exp:13;
    unsigned rounded=mant>>shift,rem=mant&((1u<<shift)-1),mid=1u<<(shift-1);
    if(rem>mid||(rem==mid&&(rounded&1)))++rounded;
    if(exp<113)return uint16_t(sign|rounded);
    return uint16_t(sign|(((exp-112)<<10)+(rounded-1024)));
}
static float rounded(float f) {return decode(encode(f));}
static H halves(size_t n,uint32_t seed) {
    H v(n);
    for(auto& h:v) {
        seed=1664525u*seed+1013904223u;
        h=uint16_t((seed&0x8000)|((9+(seed>>16)%7)<<10)|(seed&1023));
    }
    return v;
}
static F values(size_t n,uint32_t seed=7) {
    F v(n);for(float& f:v) {seed=1664525u*seed+1013904223u;f=float(int((seed>>8)%2001)-1000)/731.f;}return v;
}
[[maybe_unused]] static void exact(const F& a,const F& b,const std::string& msg) {
    require(a.size()==b.size(),msg+": length");
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),msg+": nonfinite");
        require(std::memcmp(&a[i],&b[i],4)==0,msg+": mismatch at "+std::to_string(i));
    }
}
// Dot-product oracle uses no tile indexing. Extra zero FMAs are part of the
// original kernel's K16 contract, and bias is a separate final operation.
static float dot(Shape d,const H& x,const H& w,const F* bias,I r,I c) {
    float z=0;
    for(I k=0;k<d.k;++k) z=std::fma(decode(x[r*d.k+k]),decode(w[c*d.k+k]),z);
    for(I k=d.k;k%16;++k) z=std::fma(0.f,0.f,z);
    return z+(bias?(*bias)[c]:0.f);
}
static void oracle(Shape d,const H& x,const H& w,const F* bias,const F& out,bool sample) {
    auto at=[&](size_t idx) {
        const I r=I(idx)/d.n,c=I(idx)%d.n;
        float expected=rounded(dot(d,x,w,bias,r,c));
        require(std::isfinite(expected)&&std::isfinite(out[idx])&&
                std::memcmp(&expected,&out[idx],4)==0,"CPU ordered-FMA oracle at "+std::to_string(idx));
        double sum=bias?(*bias)[c]:0.,mag=std::abs(sum);
        for(I k=0;k<d.k;++k) {
            double p=double(decode(x[r*d.k+k]))*decode(w[c*d.k+k]);sum+=p;mag+=std::abs(p);
        }
        const double neps=(((d.k+15)/16)*16+1)*std::ldexp(1.,-24);
        require(neps<1,"oracle reduction bound");
        require(std::isfinite(sum)&&std::abs(double(out[idx])-sum)<=(1+std::ldexp(1.,-11))*neps/(1-neps)*mag+std::abs(sum)*std::ldexp(1.,-11)+std::ldexp(1.,-25),"CPU double oracle");
    };
    if(!sample) { for(size_t i=0;i<out.size();++i)at(i); }
    else {
        for(size_t i=0;i<97;++i)at(i*(out.size()-1)/96);
        for(I r=0;r<d.m;++r) {at(size_t(r*d.n));at(size_t((r+1)*d.n-1));}
    }
}
static void self_test() {
    for(unsigned h=0;h<65536;++h)if((h&0x7c00)!=0x7c00)
        require(encode(decode(uint16_t(h)))==h,"half exhaustive roundtrip");
    require(encode(1.f+std::ldexp(1.f,-11))==0x3c00,"half tie even down");
    require(encode(1.f+3*std::ldexp(1.f,-11))==0x3c02,"half tie even up");
    require(encode(std::ldexp(1.f,-25))==0,"half subnormal tie");
    require(encode(65520.f)==0x7c00,"half overflow tie");
    const float tie=1.f+std::ldexp(1.f,-11);
    require(encode(tie+std::ldexp(1.f,-13))==0x3c01 &&
            encode(rounded(tie)+std::ldexp(1.f,-13))==0x3c00,"single final rounding witness");
    require(decode(0x3c00)==1.f&&decode(0xc000)==-2.f,"half normals");
    require(decode(1)==std::ldexp(1.f,-24)&&decode(0x03ff)==1023*std::ldexp(1.f,-24),"half subnormals");
    require(decode(0x0400)==std::ldexp(1.f,-14)&&decode(0x7bff)==65504.f,"half limits");
    require(std::signbit(decode(0x8000))&&!std::signbit(decode(0)),"half signed zero");
    require(std::isinf(decode(0x7c00))&&std::isnan(decode(0x7c01)),"half specials");
    require(dot({1,1,3},{0x3c00,0x3c00,0x3c00},{0x7800,0x1400,0xf800},nullptr,0,0)==0.f,"K-order cancellation");
    const F bias{32768.f};
    require(dot({1,1,2},{0x3c00,0x3c00},{0xf800,0x1400},&bias,0,0)==0.f,"bias after reduction");
    require(dot({1,1,3},{0x3c00,0x4000,0x4200},{0x4400,0x4500,0x4600},nullptr,0,0)==32.f,"hand dot");
    // Independent CPU emulation of tile ownership/loading and K32/K16 tails.
    for(I m:{1,63,64,65})for(I n:{1,17,64,65})for(I k:{1,15,16,17,31,32,33,65}) {
        const Shape d{m,n,k};const H x=halves(size_t(m*k),7),w=halves(size_t(n*k),19);
        const F b=values(size_t(n));F y(size_t(m*n),sentinel);std::vector<int> hits(y.size());
        for(I gy=0;gy<(m+63)/64;++gy)for(I gx=0;gx<(n+63)/64;++gx)for(I t=0;t<256;++t)
            for(I i=0;i<4;++i)for(I j=0;j<4;++j) {
                const I r=gy*64+t/16+i*16,c=gx*64+t%16+j*16;
                if(r>=m||c>=n)continue;
                float z=0;
                for(I base=0;base<k;base+=32)for(I part=0;part<2;++part)if(base+part*16<k)
                    for(I kk=0;kk<16;++kk) {
                        const I at=base+part*16+kk;
                        z=std::fma(at<k?decode(x[r*k+at]):0.f,at<k?decode(w[c*k+at]):0.f,z);
                    }
                y[r*n+c]=rounded(z+b[c]);++hits[r*n+c];
            }
        require(std::all_of(hits.begin(),hits.end(),[](int h){return h==1;}),"tile output coverage");
        oracle(d,x,w,&b,y,false);
    }
    for(Shape d:std::initializer_list<Shape>{{1500,1280,1280},{1500,5120,1280},{1500,1280,5120}})
        require(whisper_linear_f16::supported(d.m,d.k,d.n),"profile supported");
    for(Shape d:std::initializer_list<Shape>{{0,1,1},{-1,1,1},{4097,1,1},{9,0,1},{9,1,0},{9,1,-1},{9,INT64_MAX,INT64_MAX}})
        require(!whisper_linear_f16::supported(d.m,d.k,d.n),"unsupported geometry");
    H x(128),w(128);F out(128);
    require(whisper_linear_f16::arguments(x.data(),9,1,w.data(),1,nullptr,out.data()),"valid arguments");
    require(!whisper_linear_f16::arguments(nullptr,9,1,w.data(),1,nullptr,out.data()),"null input");
    require(!whisper_linear_f16::arguments(x.data(),9,1,w.data(),1,out.data()+1,out.data()),"partial bias overlap");
    // A float offset is four bytes, while one F16 weight occupies only two.
    // Same allocation alone is not overlap: test both exact byte geometries.
    alignas(float) uint16_t weight_storage[64]={};
    auto* adjacent_out=reinterpret_cast<float*>(weight_storage+2);
    require(whisper_linear_f16::arguments(x.data(),9,1,weight_storage,1,nullptr,adjacent_out),
            "two-byte weight ends before output at byte four");
    require(!whisper_linear_f16::arguments(x.data(),9,3,weight_storage,1,nullptr,adjacent_out),
            "six-byte weight overlaps output at byte four");
    std::cout<<"PASS CPU half decoder, arithmetic witnesses, 128 tile cases and guards\n";
}
#ifdef __HIPCC__
#include "../../../ns/ns.h"
#include "../../../ns/ns_internal.hpp"
#include <dlfcn.h>
extern "C" void* nsd_create(void*);
extern "C" void nsd_destroy(void*);
extern "C" int nsd_linear(void*,const void*,int,I,I,const void*,I,const float*,void*,int);
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) \
 X(nsd_create) X(nsd_destroy) X(nsd_linear)
struct Api {
    void* lib = nullptr;
    ns_gpu* gpu = nullptr;
    void* ctx = nullptr;
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
            ctx=nsd_create(gpu); require(ctx!=nullptr,"nsd_create");
        } catch (...) { if(gpu) ns_gpu_close(gpu); dlclose(lib); lib = nullptr; throw; }
    }
    Api(const Api&) = delete;
    Api& operator=(const Api&) = delete;
    ~Api() { if(ctx) { ns_sync(gpu); nsd_destroy(ctx); } if (gpu) { ns_sync(gpu); ns_gpu_close(gpu); } if (lib) dlclose(lib); }
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
    F read_half() const {
        H bits(bytes/2);a.check(a.ns_copy_from(a.gpu,bits.data(),ptr(),bytes),"read half output");a.sync();
        F host(bits.size());for(size_t i=0;i<bits.size();++i)host[i]=decode(bits[i]);return host;
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
static void half_unchanged(Api& a,const Buffer& b,const H& expected) {
    H actual(expected.size());a.check(a.ns_copy_from(a.gpu,actual.data(),b.ptr(),actual.size()*2),"read half input");a.sync();
    require(actual==expected,"half input mutated");b.guards();
}
static void gpu_case(Api& a,Shape d,bool has_bias,bool profile,int special=0) {
    H x=halves(size_t(d.m*d.k),7),w=halves(size_t(d.n*d.k),19);
    F bias=values(size_t(d.n));
    if(special==1) {
        // Ascending K order witness: large + tiny - large, with K padding.
        std::fill(x.begin(),x.end(),0x3c00);
        for(I c=0;c<d.n;++c) {w[c*d.k]=0x7800;w[c*d.k+1]=0x1400;w[c*d.k+2]=0xf800;}
    } else if(special==2) {
        // Bias initialized in the accumulator would change the result.
        std::fill(x.begin(),x.end(),0x3c00);std::fill(bias.begin(),bias.end(),32768.f);
        for(I c=0;c<d.n;++c) {w[c*d.k]=0xf800;w[c*d.k+1]=0x1400;}
    } else if(special==3) {
        std::fill(x.begin(),x.end(),0x8000);std::fill(bias.begin(),bias.end(),-0.f);
    } else if(special==5) {
        std::fill(x.begin(),x.end(),0x3c00);
        std::fill(bias.begin(),bias.end(),std::ldexp(1.f,-13));
        for(I c=0;c<d.n;++c) {w[c*d.k]=0x3c00;w[c*d.k+1]=0x1000;}
    } else if(special==4) {
        const uint16_t edges[]={1,0x8001,0x03ff,0x0400,0x7bff,0xfbff,0,0x8000};
        for(size_t i=0;i<x.size();++i)x[i]=edges[i%8];
        for(size_t i=0;i<w.size();++i)w[i]=(i%2)?0x0400:0x8400;
    }
    Buffer dx(a,x.size()*2),dw(a,w.size()*2),db(a,bias);
    dx.write(x.data(),x.size()*2);dw.write(w.data(),w.size()*2);
    Buffer base(a,size_t(d.m*d.n)*2),out(a,size_t(d.m*d.n)*2);
    const H poison(size_t(d.m*d.n),0x7e00);base.write(poison.data(),poison.size()*2);out.write(poison.data(),poison.size()*2);
    auto baseline=[&] {a.check(a.nsd_linear(a.ctx,dx.ptr(),1,d.m,d.k,dw.ptr(),d.n,
        has_bias?db.fp():nullptr,base.ptr(),1),"actual nsd_linear");};
    auto candidate=[&] {hipcheck(whisper_linear_f16::launch(dx.ptr(),d.m,d.k,dw.ptr(),d.n,
        has_bias?db.fp():nullptr,out.fp(),a.gpu->q),"tiled linear");};
    hipcheck(hipSetDevice(a.gpu->dev),"set device");baseline();candidate();a.sync();
    const F original=base.read_half(),actual=out.read_half();
    exact(actual,original,"actual baseline/candidate");
    oracle(d,x,w,has_bias?&bias:nullptr,actual,profile);
    half_unchanged(a,dx,x);half_unchanged(a,dw,w);
    exact(db.read(),bias,"bias unchanged");db.guards();base.guards();out.guards();
    std::cout<<"PASS shape=["<<d.m<<','<<d.n<<','<<d.k<<"] bias="<<has_bias<<" special="<<special
             <<" outputs="<<actual.size()<<" bitwise=1 cpu="<<(profile?"sampled-FMA-and-double":"full-FMA-and-double")<<'\n';
    if(profile) {
        Events ev;F bt,ct;
        auto time=[&](bool tiled) {
            hipcheck(hipEventRecord(ev.start,a.gpu->q),"record start");
            if(tiled)candidate();else baseline();
            hipcheck(hipEventRecord(ev.stop,a.gpu->q),"record stop");
            hipcheck(hipEventSynchronize(ev.stop),"sync stop");
            float ms=0;hipcheck(hipEventElapsedTime(&ms,ev.start,ev.stop),"elapsed");return ms;
        };
        for(int i=0;i<5;++i) {
            float b,c;if(i%2){c=time(true);b=time(false);}else{b=time(false);c=time(true);}
            bt.push_back(b);ct.push_back(c);
            std::cout<<"TIMING pair="<<i<<" baseline_ms="<<b<<" candidate_ms="<<c<<'\n';
        }
        std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());
        std::cout<<"MEDIAN baseline_ms="<<bt[2]<<" candidate_ms="<<ct[2]<<" speedup="<<bt[2]/ct[2]<<'\n';
        exact(out.read_half(),original,"repeated output");out.guards();base.guards();
    }
}
static void gpu_tests(Api& a,bool profile) {
    std::cout<<std::setprecision(9);hipDeviceProp_t prop{};int runtime=0,driver=0;
    hipcheck(hipGetDeviceProperties(&prop,a.gpu->dev),"properties");
    hipcheck(hipRuntimeGetVersion(&runtime),"runtime");hipcheck(hipDriverGetVersion(&driver),"driver");
    std::cout<<"DEVICE name="<<prop.name<<" arch="<<prop.gcnArchName<<" wave="<<prop.warpSize
             <<" runtime="<<runtime<<" driver="<<driver<<" candidate_scratch_bytes=0 shared_bytes="
             <<whisper_linear_f16::shared_bytes<<'\n';
    const I ns[]={1,15,16,17,63,64,65};size_t step=0;
    for(I m:{1,9,63,64,65,127})for(I k:{1,15,16,17,31,32,33,65}) {
        const I n=ns[step++%7];for(bool b:{false,true})gpu_case(a,{m,n,k},b,false);
    }
    for(I k:{1024,2048,4096})gpu_case(a,{9,3,k},true,false);
    gpu_case(a,{9,3,3},false,false,1);gpu_case(a,{9,3,2},true,false,2);
    gpu_case(a,{22,17,16},true,false,3);gpu_case(a,{9,5,33},false,false,4);
    gpu_case(a,{65,65,2},true,false,5);
    Buffer x(a,F(128,1.f)),w(a,F(128,1.f)),out(a,F(128,sentinel));
    auto reject=[&](const char* label,I m,I k,I n,const void* src,const void* weights,const float* bias,float* dst) {
        require(whisper_linear_f16::launch(src,m,k,weights,n,bias,dst,a.gpu->q)==hipErrorInvalidValue,
                std::string("invalid call accepted: ")+label+" M="+std::to_string(m)+
                " K="+std::to_string(k)+" N="+std::to_string(n));
    };
    for(Shape d:std::initializer_list<Shape>{{0,1,1},{-1,1,1},{4097,1,1},{9,0,1},{9,1,-1},{9,INT64_MAX,INT64_MAX}})
        reject("geometry",d.m,d.k,d.n,x.ptr(),w.ptr(),nullptr,out.fp());
    reject("null input",9,1,1,nullptr,w.ptr(),nullptr,out.fp());
    reject("null weights",9,1,1,x.ptr(),nullptr,nullptr,out.fp());
    reject("null output",9,1,1,x.ptr(),w.ptr(),nullptr,nullptr);
    reject("input overlap",9,1,1,x.ptr(),w.ptr(),nullptr,x.fp()+1);
    // w[1,3] spans bytes [0,6); output starts at byte 4: two-byte overlap.
    reject("weight overlap",9,3,1,x.ptr(),w.ptr(),nullptr,w.fp()+1);
    reject("bias overlap",9,1,1,x.ptr(),w.ptr(),out.fp()+1,out.fp());
    reject("misaligned input",9,1,1,static_cast<const char*>(x.ptr())+1,w.ptr(),nullptr,out.fp());
    reject("misaligned output",9,1,1,x.ptr(),w.ptr(),nullptr,
           reinterpret_cast<float*>(static_cast<char*>(out.ptr())+1));
    reject("misaligned bias",9,1,1,x.ptr(),w.ptr(),
           reinterpret_cast<float*>(static_cast<char*>(out.ptr())+1),out.fp());
    reject("wrapped address",9,1,1,reinterpret_cast<void*>(UINTPTR_MAX-1),w.ptr(),nullptr,out.fp());
    a.sync();exact(x.read(),F(128,1.f),"rejected input");exact(w.read(),F(128,1.f),"rejected weights");
    exact(out.read(),F(128,sentinel),"rejected output");x.guards();w.guards();out.guards();
    if(profile)for(Shape d:std::initializer_list<Shape>{{1500,1280,1280},{1500,5120,1280},{1500,1280,5120}})
        for(bool b:{false,true})gpu_case(a,d,b,true);
}

#endif
int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--self-test") {self_test();return 0;}
#ifdef __HIPCC__
        if(argc!=3&&argc!=4)throw std::runtime_error("usage: validator LIB GPU_INDEX [--profile] | --self-test");
        if(argc==4)require(std::string(argv[3])=="--profile","unknown option");
        size_t used=0;int index=std::stoi(argv[2],&used);
        require(index>=0&&used==std::strlen(argv[2]),"GPU index");
        self_test();Api api(argv[1],index);gpu_tests(api,argc==4);
        std::cout<<"PASS GPU linear_f16 operator checks\n";return 0;
#else
        throw std::runtime_error("CPU build: use --self-test; compile with hipcc for GPU checks");
#endif
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
