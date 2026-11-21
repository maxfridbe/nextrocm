// Standalone private Whisper encoder attention validator; parent owns GPU/SSH.
// c++ -std=c++17 -O2 -Wall -Wextra -Werror kernels/audio/whisper/tests/attention_numeric.cpp -o /tmp/whisper-att-cpu
// /tmp/whisper-att-cpu --self-test
// hipcc -std=c++17 -O2 --offload-arch=gfx1035 -x hip kernels/audio/whisper/tests/attention_numeric.cpp -ldl -o /tmp/whisper-att-gpu
// /tmp/whisper-att-gpu ABS_RUNTIME 0 --profile
// Exact finite GPU comparison to actual loaded nsd_attention for EVERY output.
// Independent CPU double softmax oracle, sampled queries for larger shapes,
// full 64 output features per checked query. CPU exp and GPU exp may differ;
// the predeclared forward-error bound below is NOT the native bitwise gate.
#include "../attention_tiled.hpp"
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
[[maybe_unused]] static void exact(const F& a,const F& b,const std::string& msg) {
    require(a.size()==b.size(),msg+": length");
    for(size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),msg+": nonfinite");
        require(std::memcmp(&a[i],&b[i],4)==0,msg+": mismatch at "+std::to_string(i));
    }
}
struct Shape {I s,h;};
static size_t count(Shape d){return size_t(d.s*d.h*64);}
static size_t at(Shape d,I row,I head,I feature){return size_t((row*d.h+head)*64+feature);}
static float tree_sum(F values) {
    require(values.size()==256,"tree geometry");
    for(int s=128;s;s/=2)for(int i=0;i<s;++i)values[i]+=values[i+s];
    return values[0];
}
static F ordered_scores(Shape d,const H& q,const H& k,I row,I head) {
    F scores(size_t(d.s));
    for(I j=0;j<d.s;++j){float z=0;for(I a=0;a<64;++a)z=std::fma(decode(q[at(d,row,head,a)]),decode(k[at(d,j,head,a)]),z);scores[j]=z*.125f;}
    return scores;
}
// Full double dot/softmax/PV reference, independent of tile index/partition.
// Bound includes F32 dot and sum/PV roundoff, conservative exp/shift allowance,
// and final half rounding. No tolerance applies to actual native GPU equality.
static void oracle(Shape d,const H&q,const H&k,const H&v,const F&out,bool sample,bool uniform) {
    require(out.size()==count(d),"oracle output shape");
    std::vector<I> rows,heads;
    if(sample){rows={0,1,3,4,d.s/2,d.s-1};heads={0,d.h/2,d.h-1};}
    else{for(I r=0;r<d.s;++r)rows.push_back(r);for(I h=0;h<d.h;++h)heads.push_back(h);}
    for(I row:rows)if(row<d.s)for(I head:heads) {
        std::vector<double> scores(size_t(d.s));double maxscore=-INFINITY,maxmag=0,maxabs=0;
        for(I j=0;j<d.s;++j) {
            double z=0,mag=0;for(I a=0;a<64;++a){double p=double(decode(q[at(d,row,head,a)]))*decode(k[at(d,j,head,a)]);z+=p;mag+=std::abs(p);}
            z*=.125;scores[j]=z;maxscore=std::max(maxscore,z);maxabs=std::max(maxabs,std::abs(z));maxmag=std::max(maxmag,mag*.125);
        }
        double den=0;for(double& z:scores){z=std::exp(z-maxscore);den+=z;}
        const double u=std::ldexp(1.,-24),gdot=64*u/(1-64*u),gsum=double(d.s+256)*u/(1-double(d.s+256)*u);
        for(I a=0;a<64;++a) {
            double z=0,mv=0;float fsum=0;
            for(I j=0;j<d.s;++j){float value=decode(v[at(d,j,head,a)]);z+=scores[j]*value;mv=std::max(mv,std::abs(double(value)));if(uniform)fsum=std::fma(1.f,value,fsum);}
            z/=den;const float got=out[at(d,row,head,a)];
            const double arithmetic=mv*(2*gdot*maxmag+64*u*(1+maxabs)+3*gsum);
            const double tolerance=(1+std::ldexp(1.,-11))*arithmetic+std::abs(z)*std::ldexp(1.,-11)+std::ldexp(1.,-25);
            require(std::isfinite(got)&&std::isfinite(z)&&std::abs(got-z)<=tolerance,"independent double oracle");
            if(uniform){const float wanted=rounded(fsum/float(d.s));require(std::memcmp(&got,&wanted,4)==0,"uniform exact CPU mean");}
        }
    }
}
static void self_test() {
    for(unsigned h=0;h<65536;++h)if((h&0x7c00)!=0x7c00)require(encode(decode(uint16_t(h)))==h,"half roundtrip");
    require(rounded(1.f+std::ldexp(1.f,-11))==1.f,"half tie");
    size_t cases=0;
    // Independent emulation verifies cooperative load transpose and every
    // score's unique ownership for key/query tails; CPU fmaf dot is exact.
    for(I S:{1,3,4,5,63,64,65,257})for(I Hh:{1,3}) {
        Shape d{S,Hh};H q=halves(count(d),7),k=halves(count(d),19);std::vector<int>hits(size_t(S*S*Hh));F scores(hits.size(),sentinel);
        for(I h=0;h<Hh;++h)for(I r0=0;r0<S;r0+=4)for(I base=0;base<S;base+=64) {
            float kt[64][65]={};int writes[64][64]={};
            for(int t=0;t<256;++t)for(int z=t;z<4096;z+=256){int key=z/64,a=z%64;kt[a][key]=base+key<S?decode(k[at(d,base+key,h,a)]):0.f;++writes[a][key];}
            for(int a=0;a<64;++a)for(int j=0;j<64;++j)require(writes[a][j]==1,"cooperative load coverage");
            for(int t=0;t<256;++t){I r=r0+t/64,j=base+t%64;if(r>=S||j>=S)continue;float z=0;for(int a=0;a<64;++a)z=std::fma(decode(q[at(d,r,h,a)]),kt[a][t%64],z);size_t i=size_t((h*S+r)*S+j);scores[i]=z*.125f;++hits[i];}
        }
        require(std::all_of(hits.begin(),hits.end(),[](int h){return h==1;}),"score coverage");
        for(I h=0;h<Hh;++h)for(I r=0;r<S;++r){F ref=ordered_scores(d,q,k,r,h);for(I j=0;j<S;++j)require(std::memcmp(&ref[j],&scores[size_t((h*S+r)*S+j)],4)==0,"tiled CPU dot order");}
        F sums(256);for(I j=0;j<S;++j)sums[j%256]+=1.f;require(tree_sum(sums)==float(S),"denominator partition");
        ++cases;
    }
    Shape d{17,2};H q(count(d),0),k(count(d),0x3c00),v=halves(count(d),29);F out(count(d));
    for(I r=0;r<d.s;++r)for(I h=0;h<d.h;++h)for(I a=0;a<64;++a){float z=0;for(I j=0;j<d.s;++j)z=std::fma(1.f,decode(v[at(d,j,h,a)]),z);out[at(d,r,h,a)]=rounded(z/float(d.s));}
    oracle(d,q,k,v,out,false,true);
    H other(count(d));require(whisper_attention::arguments(q.data(),k.data(),v.data(),17,2,64,128,other.data()),"valid geometry");
    require(!whisper_attention::arguments(q.data(),k.data(),v.data(),17,2,64,128,q.data()+1),"overlap guard");
    require(!whisper_attention::arguments(q.data(),k.data(),v.data(),17,2,64,128,reinterpret_cast<char*>(other.data())+1),"alignment guard");
    require(!whisper_attention::supported(2049,20,64,1280)&&!whisper_attention::supported(1500,33,64,2112)&&!whisper_attention::supported(INT64_MAX,INT64_MAX,64,0),"shape caps");
    require(whisper_attention::shared_bytes(2048)<=65536,"LDS cap");
    std::cout<<"PASS CPU half decoder, "<<cases<<" cooperative score tiles, dot order, denominator partition, uniform oracle and guards\n";
}
#ifdef __HIPCC__
#include "../../../ns/ns.h"
#include "../../../ns/ns_internal.hpp"
#include <dlfcn.h>
extern "C" void* nsd_create(void*);
extern "C" void nsd_destroy(void*);
extern "C" int nsd_attention(void*,const void*,const void*,const void*,int,I,I,I,I,void*,int);
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) \
 X(nsd_create) X(nsd_destroy) X(nsd_attention)
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
static void gpu_case(Api&a,Shape d,bool profile,int special=0) {
    H q=halves(count(d),7),k=halves(count(d),19),v=halves(count(d),29);
    if(special==1){std::fill(q.begin(),q.end(),0);}
    if(special==2){const uint16_t edges[]={0,0x8000,1,0x8001,0x03ff,0x0400,0x3c00,0xbc00};for(size_t i=0;i<q.size();++i){q[i]=edges[i%8];k[i]=edges[(i*3+1)%8];v[i]=edges[(i*5+2)%8];}}
    if(special==3){std::fill(q.begin(),q.end(),0x3c00);for(I j=0;j<d.s;++j)for(I h=0;h<d.h;++h){for(I a0=0;a0<64;++a0)k[at(d,j,h,a0)]=0;k[at(d,j,h,0)]=0x7800;k[at(d,j,h,1)]=(j%2)?0xf800:0x1400;k[at(d,j,h,2)]=(j%2)?0x1400:0xf800;}}
    Buffer dq(a,q.size()*2),dk(a,k.size()*2),dv(a,v.size()*2),base(a,v.size()*2),out(a,v.size()*2);
    dq.write(q.data(),q.size()*2);dk.write(k.data(),k.size()*2);dv.write(v.data(),v.size()*2);
    H poison(v.size(),0x7e00);base.write(poison.data(),poison.size()*2);out.write(poison.data(),poison.size()*2);
    auto baseline=[&]{a.check(a.nsd_attention(a.ctx,dq.ptr(),dk.ptr(),dv.ptr(),1,d.s,d.h,64,d.h*64,base.ptr(),1),"actual nsd_attention");};
    auto candidate=[&]{hipcheck(whisper_attention::launch(dq.ptr(),dk.ptr(),dv.ptr(),d.s,d.h,64,d.h*64,out.ptr(),a.gpu->q),"tiled attention");};
    hipcheck(hipSetDevice(a.gpu->dev),"set device");baseline();candidate();a.sync();
    F original=base.read_half(),actual=out.read_half();exact(actual,original,"native attention bitwise");
    oracle(d,q,k,v,actual,d.s>65,special==1);
    half_unchanged(a,dq,q);half_unchanged(a,dk,k);half_unchanged(a,dv,v);base.guards();out.guards();
    std::cout<<"PASS shape=["<<d.s<<','<<d.h<<",64,"<<d.h*64<<"] special="<<special<<" outputs="<<actual.size()<<" bitwise=1 cpu="<<(d.s>65?"sampled-queries-all-features":"full")<<'\n';
    if(profile){Events ev;F bt,ct;
        auto time=[&](bool use){hipcheck(hipEventRecord(ev.start,a.gpu->q),"start");if(use)candidate();else baseline();hipcheck(hipEventRecord(ev.stop,a.gpu->q),"stop");hipcheck(hipEventSynchronize(ev.stop),"wait");float ms=0;hipcheck(hipEventElapsedTime(&ms,ev.start,ev.stop),"elapsed");return ms;};
        for(int i=0;i<5;++i){float b,c;if(i%2){c=time(true);b=time(false);}else{b=time(false);c=time(true);}bt.push_back(b);ct.push_back(c);std::cout<<"TIMING pair="<<i<<" baseline_ms="<<b<<" candidate_ms="<<c<<'\n';}
        std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());std::cout<<"MEDIAN baseline_ms="<<bt[2]<<" candidate_ms="<<ct[2]<<" speedup="<<bt[2]/ct[2]<<'\n';
        exact(out.read_half(),original,"repeated output");base.guards();out.guards();
    }
}
static void gpu_tests(Api&a,bool profile) {
    hipDeviceProp_t prop{};int runtime=0;hipcheck(hipGetDeviceProperties(&prop,a.gpu->dev),"properties");hipcheck(hipRuntimeGetVersion(&runtime),"runtime");
    std::cout<<std::setprecision(9)<<"DEVICE name="<<prop.name<<" arch="<<prop.gcnArchName<<" wave="<<prop.warpSize<<" runtime="<<runtime<<" scratch_bytes=0 actual_shared_bytes="<<whisper_attention::shared_bytes(1500)<<'\n';
    for(I s:{1,2,3,4,5,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257})gpu_case(a,{s,s%3+1},false);
    gpu_case(a,{17,2},false,1);gpu_case(a,{65,3},false,2);gpu_case(a,{257,1},false,3);
    Buffer q(a,F(4096,1.f)),k(a,F(4096,1.f)),v(a,F(4096,1.f)),out(a,F(4096,sentinel));
    auto reject=[&](const char*name,const void*qp,const void*kp,const void*vp,I S,I Hh,I D,I stride,void*dst){require(whisper_attention::launch(qp,kp,vp,S,Hh,D,stride,dst,a.gpu->q)==hipErrorInvalidValue,std::string("invalid accepted: ")+name);};
    for(I s:{0,-1,2049})reject("S",q.ptr(),k.ptr(),v.ptr(),s,1,64,64,out.ptr());
    for(I h:{0,-1,33})reject("H",q.ptr(),k.ptr(),v.ptr(),1,h,64,64,out.ptr());
    reject("D",q.ptr(),k.ptr(),v.ptr(),1,1,32,64,out.ptr());reject("stride",q.ptr(),k.ptr(),v.ptr(),1,1,64,65,out.ptr());
    reject("nullq",nullptr,k.ptr(),v.ptr(),1,1,64,64,out.ptr());reject("nullk",q.ptr(),nullptr,v.ptr(),1,1,64,64,out.ptr());reject("nullv",q.ptr(),k.ptr(),nullptr,1,1,64,64,out.ptr());reject("nullout",q.ptr(),k.ptr(),v.ptr(),1,1,64,64,nullptr);
    for(void*p:{q.ptr(),k.ptr(),v.ptr()})reject("alias",q.ptr(),k.ptr(),v.ptr(),1,1,64,64,static_cast<char*>(p)+2);
    reject("misaligned",q.ptr(),k.ptr(),v.ptr(),1,1,64,64,static_cast<char*>(out.ptr())+1);
    reject("wrapped",reinterpret_cast<void*>(UINTPTR_MAX-1),k.ptr(),v.ptr(),1,1,64,64,out.ptr());
    a.sync();exact(q.read(),F(4096,1.f),"reject q");exact(k.read(),F(4096,1.f),"reject k");exact(v.read(),F(4096,1.f),"reject v");exact(out.read(),F(4096,sentinel),"reject out");q.guards();k.guards();v.guards();out.guards();
    if(profile)gpu_case(a,{1500,20},true);
}
#endif
int main(int argc,char**argv){try{
    if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
#ifdef __HIPCC__
    require(argc==3||argc==4,"usage: attention_validator LIB GPU_INDEX [--profile]");if(argc==4)require(std::string(argv[3])=="--profile","unknown option");
    size_t used=0;int index=std::stoi(argv[2],&used);require(index>=0&&used==std::strlen(argv[2]),"GPU index");self_test();Api api(argv[1],index);gpu_tests(api,argc==4);std::cout<<"PASS GPU attention operator checks\n";return 0;
#else
    throw std::runtime_error("CPU build: use --self-test");
#endif
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
