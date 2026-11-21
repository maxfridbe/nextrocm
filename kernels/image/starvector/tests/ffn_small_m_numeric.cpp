// Standalone private FFN candidate validator. No production registration.
// Independent F64 dots, original-runtime bitwise parity, tails and paired events.
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



using Linear = int (*)(void*,const void*,int,I,I,const void*,I,const float*,void*,int);
struct Candidate {
    void* lib=nullptr;Linear attempt=nullptr,test=nullptr,fallback=nullptr;
    const char* (*error)()=nullptr;
    template<class T> void bind(T& fn,const char* name){
        void* p=dlsym(lib,name);require(p!=nullptr,std::string("missing candidate symbol ")+name);
        static_assert(sizeof(fn)==sizeof(p),"POSIX function pointer size");std::memcpy(&fn,&p,sizeof(fn));
    }
    explicit Candidate(const char* path){
        lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());
        try{bind(attempt,"nrsv_ffn_try");bind(test,"nrsv_ffn_test");bind(fallback,"nrsv_ffn_with_fallback");bind(error,"ns_last_error");}
        catch(...){dlclose(lib);lib=nullptr;throw;}
    }
    Candidate(const Candidate&)=delete;Candidate& operator=(const Candidate&)=delete;
    ~Candidate(){if(lib)dlclose(lib);}
    void check(int status){if(status){const char* why=error();throw std::runtime_error(std::string("candidate status ")+std::to_string(status)+": "+(why?why:"unknown"));}}
};
static uint32_t random_u32(uint32_t& s){s=s*1664525u+1013904223u;return s;}
static double half_value(uint16_t h){
    unsigned e=(h>>10)&31,m=h&1023;require(e!=31,"nonfinite half");
    const double a=e?std::ldexp(double(1024+m),int(e)-25):std::ldexp(double(m),-24);
    return h&0x8000?-a:a;
}
static uint16_t sample_half(uint32_t& seed,size_t i,int variant){
    uint32_t r=random_u32(seed);
    // Variable exponents and all mantissa bits exercise FMA rounding rather
    // than only exactly summable small integer fractions. Include subnormals,
    // signed zeros and cancellation. No NaN/Inf model inputs are assumed.
    if(i%127==0)return uint16_t((r>>16)&0x8000); // signed zero
    if(i%113==0)return uint16_t(((r>>16)&0x8000)|(r&1023));
    const uint16_t magnitude=uint16_t(((5+((r>>19)%10))<<10)|(r&1023));
    const uint16_t sign=variant?uint16_t((i&1)?0x8000:0):uint16_t((r>>16)&0x8000);
    return uint16_t(sign|magnitude);
}
static std::vector<unsigned char> read_bytes(const Buffer& b){
    std::vector<unsigned char> out(b.bytes);b.a.check(b.a.ns_copy_from(b.a.gpu,out.data(),b.ptr(),b.bytes),"read bytes");b.a.sync();return out;
}
static double value(const std::vector<unsigned char>& b,size_t i,int dt){
    if(dt==0){float v;std::memcpy(&v,b.data()+i*4,4);return v;}
    uint16_t h;std::memcpy(&h,b.data()+i*2,2);return half_value(h);
}
static double timed(Api& a,const std::function<void()>& f){
    I begin=0,end=0;double ns=0;
    a.check(a.ns_stamp(a.gpu,&begin),"begin");f();a.check(a.ns_stamp(a.gpu,&end),"end");
    a.check(a.ns_elapsed(a.gpu,begin,end,&ns),"elapsed");require(std::isfinite(ns)&&ns>=0,"invalid event time");return ns;
}
static double median(std::vector<double> v){std::sort(v.begin(),v.end());return v[(v.size()-1)/2];}
static void run_case(Api& a,Candidate& c,I m,I k,I n,int dt,int odt,int mode,int variant,bool bias_on,unsigned repeats){
    const size_t nx=size_t(m*k),nw=size_t(n*k),ny=size_t(m*n);
    const size_t xb=nx*(dt?2:4),wb=nw*(dt?2:4),yb=ny*(odt?2:4);
    require(xb+wb+2*yb+size_t(n)*4+640<=256u*1024u*1024u,"256 MiB allocation cap");
    std::vector<uint16_t> hx(nx),hw(nw);F xf(nx),wf(nw),bias(static_cast<size_t>(n));
    uint32_t seed=uint32_t(139+m*7+k*3+n+variant*97);
    for(size_t i=0;i<nx;++i){hx[i]=sample_half(seed,i,variant);xf[i]=float(half_value(hx[i]));}
    for(size_t i=0;i<nw;++i){hw[i]=sample_half(seed,i,variant);wf[i]=float(half_value(hw[i]));}
    for(auto& b:bias)b=float(int(random_u32(seed)%2049)-1024)/4096.0f;
    Buffer dx(a,xb),dw(a,wb),db(a,bias),base(a,yb),candidate(a,yb);
    dx.write(dt?static_cast<const void*>(hx.data()):xf.data(),xb);
    dw.write(dt?static_cast<const void*>(hw.data()):wf.data(),wb);
    const float* bp=bias_on?db.fp():nullptr;
    auto baseline=[&](){a.check(a.nsd_linear(a.gpu,dx.ptr(),dt,m,k,dw.ptr(),n,bp,base.ptr(),odt),"baseline linear");};
    Linear fn=mode==0?c.attempt:mode==1?c.test:c.fallback;
    auto selected=[&](){c.check(fn(a.gpu,dx.ptr(),dt,m,k,dw.ptr(),n,bp,candidate.ptr(),odt));};
    if(mode==2){const auto before=read_bytes(candidate);
        require(c.attempt(a.gpu,dx.ptr(),dt,m,k,dw.ptr(),n,bp,candidate.ptr(),odt)==1,"fallback geometry not declined");
        require(read_bytes(candidate)==before,"unsupported candidate wrote output");}
    baseline();selected();const auto expected_bits=read_bytes(base),actual_bits=read_bytes(candidate);
    require(actual_bits==expected_bits,"full output candidate/baseline bitwise mismatch");
    double max_abs=0;
    for(I row=0;row<m;++row)for(I col=0;col<n;++col){
        double expected=bias_on?double(bias[size_t(col)]):0;
        for(I j=0;j<k;++j)expected+=double(xf[size_t(row*k+j)])*double(wf[size_t(col*k+j)]);
        const double actual=value(actual_bits,size_t(row*n+col),odt),err=std::abs(actual-expected);
        require(std::isfinite(actual),"nonfinite output");
        require(err <= (odt?0.002:4e-5)+(odt?0.002:5e-4)*std::abs(expected),"independent F64 dot mismatch");
        max_abs=std::max(max_abs,err);
    }
    for(unsigned i=0;i<4;++i){baseline();selected();}a.sync();
    std::vector<double> base_ns,selected_ns;
    for(unsigned i=0;i<repeats;++i){
        if(i%2){selected_ns.push_back(timed(a,selected));base_ns.push_back(timed(a,baseline));}
        else{base_ns.push_back(timed(a,baseline));selected_ns.push_back(timed(a,selected));}
        require(read_bytes(base)==expected_bits,"baseline repeat changed bits");
        require(read_bytes(candidate)==expected_bits,"candidate repeat changed bits");
    }
    dx.guards();dw.guards();db.guards();base.guards();candidate.guards();
    std::cout<<"PASS mode="<<mode<<" m="<<m<<" k="<<k<<" n="<<n<<" dt="<<dt<<" odt="<<odt
       <<" variant="<<variant<<" bias="<<bias_on<<" max_abs="<<max_abs<<" pairs="<<repeats
       <<" baseline_p50_ns="<<median(base_ns)<<" candidate_p50_ns="<<median(selected_ns)
       <<" ratio="<<median(base_ns)/median(selected_ns)<<" bitwise=pass F64=pass guards=pass\n"<<std::flush;
}
static void invalid(Api& a,Candidate& c){
    Buffer x(a,F(8193,1)),w(a,F(300000,1)),out(a,F(8193,sentinel));const auto initial=read_bytes(out);
    auto reject=[&](void* g,const void* xp,const void* wp,void* yp,I m,I k,I n,int dt,int odt){
        require(c.attempt(g,xp,dt,m,k,wp,n,nullptr,yp,odt)<0,"invalid arguments accepted");};
    reject(nullptr,x.ptr(),w.ptr(),out.ptr(),1,17,65,1,1);
    reject(a.gpu,nullptr,w.ptr(),out.ptr(),1,17,65,1,1);
    reject(a.gpu,x.ptr(),nullptr,out.ptr(),1,17,65,1,1);
    reject(a.gpu,x.ptr(),w.ptr(),nullptr,1,17,65,1,1);
    for(I m:{I(0),I(-1),INT64_MAX})reject(a.gpu,x.ptr(),w.ptr(),out.ptr(),m,17,65,1,1);
    for(I k:{I(0),I(-1),INT64_MAX})reject(a.gpu,x.ptr(),w.ptr(),out.ptr(),1,k,65,1,1);
    for(I n:{I(0),I(-1),INT64_MAX})reject(a.gpu,x.ptr(),w.ptr(),out.ptr(),1,17,n,1,1);
    for(int dt:{-1,2})reject(a.gpu,x.ptr(),w.ptr(),out.ptr(),1,17,65,dt,1);
    for(int odt:{-1,2})reject(a.gpu,x.ptr(),w.ptr(),out.ptr(),1,17,65,1,odt);
    require(c.test(a.gpu,x.ptr(),1,3,17,w.ptr(),65,nullptr,out.ptr(),1)<0,"test row bound accepted");
    require(c.test(a.gpu,x.ptr(),1,1,8193,w.ptr(),65,nullptr,out.ptr(),1)<0,"test K bound accepted");
    require(c.test(a.gpu,x.ptr(),1,1,17,w.ptr(),8193,nullptr,out.ptr(),1)<0,"test N bound accepted");
    // No synthetic bad pointer below may reach HIP: common validation must
    // reject both exact dispatch and generic geometry before device access.
    auto guarded=[&](const void* xp,const void* wp,const float* bp,void* yp){
        require(c.attempt(a.gpu,xp,1,1,17,wp,65,bp,yp,1)<0,"pointer guard did not reject before fallback");
        require(c.test(a.gpu,xp,1,1,17,wp,65,bp,yp,1)<0,"generic pointer guard accepted");
    };
    auto plus=[](const void* p,uintptr_t n){return reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(p)+n);};
    guarded(plus(x.ptr(),1),w.ptr(),nullptr,out.ptr());
    guarded(x.ptr(),plus(w.ptr(),1),nullptr,out.ptr());
    guarded(x.ptr(),w.ptr(),nullptr,plus(out.ptr(),1));
    guarded(x.ptr(),w.ptr(),reinterpret_cast<const float*>(plus(x.ptr(),2)),out.ptr());
    const auto* wrapped=reinterpret_cast<const void*>(UINTPTR_MAX-1);
    guarded(wrapped,w.ptr(),nullptr,out.ptr());
    guarded(x.ptr(),wrapped,nullptr,out.ptr());
    guarded(x.ptr(),w.ptr(),nullptr,const_cast<void*>(wrapped));
    guarded(x.ptr(),w.ptr(),reinterpret_cast<const float*>(UINTPTR_MAX-3),out.ptr());
    guarded(x.ptr(),w.ptr(),nullptr,x.ptr());
    guarded(x.ptr(),w.ptr(),nullptr,plus(x.ptr(),2));
    guarded(x.ptr(),w.ptr(),nullptr,w.ptr());
    guarded(x.ptr(),w.ptr(),nullptr,plus(w.ptr(),2));
    guarded(x.ptr(),w.ptr(),out.fp(),out.ptr());
    guarded(x.ptr(),w.ptr(),out.fp(),plus(out.ptr(),2));
    require(read_bytes(out)==initial,"invalid call wrote output");x.guards();w.guards();out.guards();
    std::cout<<"PASS invalid pointers/alignment/spans/alias/types/dimensions/overflow/testcaps, no output writes\n";
}
int main(int argc,char** argv){
    try{
        require(argc==5,"usage: ffn-small-m-numeric ABS_BASELINE_RUNTIME ABS_CANDIDATE_DSO GPU_INDEX PAIRS(1..32); NS_PROFILE=gpu");
        require(argv[1][0]=='/'&&argv[2][0]=='/',"absolute library paths required");
        require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","requires NS_PROFILE=gpu");
        const int repeats=std::stoi(argv[4]);require(repeats>=1&&repeats<=32,"pair bound");
        require(half_value(0x3c00)==1&&half_value(0xbc00)==-1&&half_value(1)==std::ldexp(1.0,-24),"half oracle sanity");
        Api a(argv[1],std::stoi(argv[3]));Candidate c(argv[2]);
        using Stats=int(*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
        uint64_t live=0,peak=0;
        auto clean=[&](){a.check(stats(a.gpu,&live,&peak),"stats");require(live==0,"tracked allocation leak");require(peak<=256u*1024u*1024u,"tracked cap exceeded");};clean();
        for(int i=0;i<8;++i)std::cout<<"empty_event_ns="<<timed(a,[](){})<<'\n';
        unsigned cases=0;
        for(I m:{I(1),I(2),I(8)})for(const auto& shape:{std::pair<I,I>{2048,8192},{8192,2048}})for(int v:{0,1}){
            run_case(a,c,m,shape.first,shape.second,1,1,0,v,v==0,unsigned(repeats));clean();++cases;}
        for(I m:{I(1),I(2),I(8)})for(const auto& shape:{std::pair<I,I>{1,1},{15,31},{16,63},{17,64},{31,65},{32,127},{33,129},{63,7},{65,63},{127,65},{2049,127},{8191,65}}){
            run_case(a,c,m,shape.first,shape.second,1,1,1,int(cases%2),cases%2,1);clean();++cases;}
        struct Fallback{I m,k,n;int dt,odt;};
        for(const auto& f:{Fallback{3,33,65,1,1},Fallback{16,17,31,1,1},Fallback{1,2048,128,1,1},Fallback{2,65,63,1,0},Fallback{1,17,65,0,0},Fallback{8,33,31,0,1}}){
            run_case(a,c,f.m,f.k,f.n,f.dt,f.odt,2,0,true,1);clean();++cases;}
        invalid(a,c);clean();
        std::cout<<"PASS cases="<<cases<<" (12 actual+36 tails+6 fallback) independent F64/full baseline bitwise/guards; live="<<live<<" peak="<<peak<<"; paired operator timing only\n";
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
