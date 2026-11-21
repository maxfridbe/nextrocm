// Private rembg operator checks. Runtime/guard helpers follow the SD1.5 validator.
// Build: c++ -std=c++17 -O2 -Wall -Wextra -Werror native_numeric.cpp -ldl -o rb-numeric
// CPU only: rb-numeric --self-test (never loads the HIP library).
// Parent GPU window: rb-numeric /absolute/libnextrocm-runtime.so [GPU_INDEX]
#include "../../../ns/ns.h"
#include "../rb.h"
#include "../rb_bounds.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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
static F values(size_t n) {
    F v(n);
    for (size_t i=0;i<n;++i) v[i]=float(int((i*37+11)%257)-128)/31.f;
    return v;
}
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) \
 X(ns_image_rb_biref_linear_tiled) X(ns_image_rb_linear_tiled) X(ns_image_rb_maxpool2) X(ns_image_rb_resize) X(ns_image_rb_bias_act) X(ns_image_rb_sigmoid)
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

static void compare(const std::string& name,const F& actual,const D& expected,double atol) {
    require(!actual.empty() && actual.size()==expected.size(),name+": shape mismatch");
    double worst=0;
    for(size_t i=0;i<actual.size();++i){
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]),name+": nonfinite");
        worst=std::max(worst,std::abs(double(actual[i])-expected[i]));
    }
    std::cout<<name<<" max_abs="<<std::setprecision(10)<<worst<<'\n';
    require(worst<=atol,name+": tolerance exceeded");
}
static D pool_ref(const F& x,I h,I w,I c) {
    I ho=(h+1)/2,wo=(w+1)/2;D out(size_t(ho*wo*c));
    for(I y=0;y<ho;++y)for(I xx=0;xx<wo;++xx)for(I ch=0;ch<c;++ch){
        double best=-std::numeric_limits<double>::infinity();
        for(I yy=2*y;yy<std::min(2*y+2,h);++yy)
            for(I xxx=2*xx;xxx<std::min(2*xx+2,w);++xxx)
                best=std::max(best,double(x[size_t((yy*w+xxx)*c+ch)]));
        out[size_t((y*wo+xx)*c+ch)]=best;
    }
    return out;
}
static D resize_ref(const F& x,I hi,I wi,I c,I ho,I wo,I os) {
    D out(size_t(ho*wo*os),sentinel);
    for(I y=0;y<ho;++y)for(I xx=0;xx<wo;++xx){
        double sy=std::max((double(y)+.5)*double(hi)/double(ho)-.5,0.);
        double sx=std::max((double(xx)+.5)*double(wi)/double(wo)-.5,0.);
        I y0=std::min(I(sy),hi-1),x0=std::min(I(sx),wi-1);
        I y1=std::min(y0+1,hi-1),x1=std::min(x0+1,wi-1);
        double fy=sy-double(y0),fx=sx-double(x0);
        for(I ch=0;ch<c;++ch){
            auto at=[&](I yy,I xxx){return double(x[size_t((yy*wi+xxx)*c+ch)]);};
            out[size_t((y*wo+xx)*os+ch)]=(1-fy)*((1-fx)*at(y0,x0)+fx*at(y0,x1))+
                                                       fy*((1-fx)*at(y1,x0)+fx*at(y1,x1));
        }
    }
    return out;
}
static void self_test() {
    using namespace rb_bounds;
    require(linear(65,27,17) && linear(65535LL*64,1,1),"valid linear bounds");
    require(!linear(65535LL*64+1,1,1) && !linear(1,INT64_MAX,1) && !linear(1,1,0),"linear rejected bounds");
    require(pool(1,1,1)&&pool(7,9,3),"valid pool bounds");
    require(!pool(0,1,1)&&!pool(INT64_MAX,2,2),"pool overflow bounds");
    require(resize(3,5,7,9,11,12)&&!resize(3,5,7,9,11,6),"resize stride bounds");
    require(!resize(1,1,1,1,1,INT64_MAX)&&!resize(-1,1,1,1,1,1),"resize storage bounds");
    require(!bias(INT64_MAX,2,0)&&!bias(1,1,2)&&!sigmoid(0),"elementwise bounds");
    require(grid(I(INT32_MAX)*256)&&!grid(I(INT32_MAX)*256+1),"grid boundary");
    compare("pool oracle",F{4,6,8,9},pool_ref(F{1,2,3,4,0,6,7,8,9},3,3,1),0);
    compare("resize oracle",F{2,3.5f,6.5f,8},resize_ref(F{1,2,3,4,5,6,7,8,9},3,3,1,2,2,1),0);
    const auto one=resize_ref(F{2},1,1,1,3,5,2);
    for(size_t i=0;i<one.size();++i)require(one[i]==(i%2?sentinel:2),"constant/padding oracle");
    std::cout<<"PASS CPU bounds/oracles; no library or GPU opened\n";
}
static void pool_case(Api& a,I h,I w,I c) {
    F x=values(size_t(h*w*c));Buffer src(a,x),out(a,F(size_t(((h+1)/2)*((w+1)/2)*c),sentinel));
    a.check(a.ns_image_rb_maxpool2(a.gpu,src.fp(),h,w,c,out.fp()),"maxpool2");a.sync();
    compare("pool "+std::to_string(h)+"x"+std::to_string(w)+"x"+std::to_string(c),out.read(),pool_ref(x,h,w,c),0);
    src.guards();out.guards();require(src.read()==x,"pool modified input");
}
static void resize_case(Api& a,I hi,I wi,I c,I ho,I wo,I os) {
    F x=values(size_t(hi*wi*c));Buffer src(a,x),out(a,F(size_t(ho*wo*os),sentinel));
    a.check(a.ns_image_rb_resize(a.gpu,src.fp(),hi,wi,c,ho,wo,out.fp(),os),"resize");a.sync();
    compare("resize "+std::to_string(hi)+"x"+std::to_string(wi)+" -> "+std::to_string(ho)+"x"+std::to_string(wo),out.read(),resize_ref(x,hi,wi,c,ho,wo,os),1e-5);
    const auto got=out.read();
    for(I p=0;p<ho*wo;++p)for(I ch=c;ch<os;++ch)require(got[size_t(p*os+ch)]==sentinel,"resize touched concat padding");
    src.guards();out.guards();require(src.read()==x,"resize modified input");
}
static void bias_case(Api& a,I m,I c,bool has_bias,bool relu) {
    F x=values(size_t(m*c)),b=values(size_t(c));Buffer out(a,x),bias(a,b);D ref(x.size());
    for(size_t i=0;i<x.size();++i){
        float sum=x[i]+(has_bias?b[i%size_t(c)]:0.f);
        ref[i]=relu?std::max(sum,0.f):sum;
    }
    a.check(a.ns_image_rb_bias_act(a.gpu,out.fp(),m,c,has_bias?bias.fp():nullptr,relu),"bias_act");a.sync();
    compare("bias/relu",out.read(),ref,0);out.guards();bias.guards();require(bias.read()==b,"bias input modified");
}
static void sigmoid_case(Api& a) {
    F x={-100,-80,-20,-2,-.1f,0,.1f,2,20,80,100};D ref;
    for(float v:x)ref.push_back(1/(1+std::exp(-double(v))));
    Buffer out(a,x);a.check(a.ns_image_rb_sigmoid(a.gpu,out.fp(),I(x.size())),"sigmoid");a.sync();
    compare("sigmoid",out.read(),ref,2e-7);out.guards();
}
// Independent double oracle plus bitwise comparison against the actual shared
// nsd kernel. Dyadic inputs keep the double-oracle gate tight even at K=9216.
static void linear_case(Api& a,I m,I k,I n,bool rounded,int backend) {
    F x(size_t(m*k)),w(size_t(n*k));
    for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+11)%257)-128)/128.f;
    for(size_t i=0;i<w.size();++i)w[i]=float(int((i*19+7)%251)-125)/128.f;
    if(rounded) {
        for(float& v:x)v/=3.1f;
        for(float& v:w)v/=2.7f;
    }
    F bias(size_t(n),0.f);
    if(backend==2)for(size_t i=0;i<bias.size();++i)bias[i]=float(int(i%29)-14)/17.f;
    D expected(size_t(m*n),0);
    for(I r=0;r<m;++r)for(I c=0;c<n;++c)for(I j=0;j<k;++j)
        expected[size_t(r*n+c)]+=double(x[size_t(r*k+j)])*double(w[size_t(c*k+j)]);
    for(I r=0;r<m;++r)for(I c=0;c<n;++c)expected[size_t(r*n+c)]+=bias[size_t(c)];
    Buffer bb(a,bias);
    const float* bp=backend==2?bb.fp():nullptr;
    Buffer xb(a,x),wb(a,w),out(a,F(size_t(m*n),sentinel)),baseline(a,F(size_t(m*n),sentinel));
    using Create = void*(*)(void*);
    using Destroy = void(*)(void*);
    using Linear = int(*)(void*,const void*,int,I,I,const void*,I,const float*,void*,int);
    Create create;Destroy destroy;Linear linear;
    a.bind(create,"nsd_create");a.bind(destroy,"nsd_destroy");a.bind(linear,"nsd_linear");
    void* ctx=create(a.gpu);require(ctx!=nullptr,"nsd_create");
    const int status=linear(ctx,xb.ptr(),0,m,k,wb.ptr(),n,bp,baseline.ptr(),0);
    destroy(ctx);a.check(status,"nsd_linear baseline");
    if(backend==0)
        a.check(a.ns_image_rb_linear_tiled(a.gpu,xb.fp(),wb.fp(),m,k,n,out.fp()),"linear_tiled");
    else
        a.check(a.ns_image_rb_biref_linear_tiled(a.gpu,xb.fp(),wb.fp(),m,k,n,bp,out.fp()),"biref_linear_tiled");
    a.sync();
    const F got=out.read(),ref=baseline.read();
    compare("linear backend="+std::to_string(backend)+" "+std::to_string(m)+"x"+std::to_string(k)+"x"+std::to_string(n),got,expected,0.002);
    require(std::memcmp(got.data(),ref.data(),got.size()*sizeof(float))==0,"linear not bitwise baseline");
    xb.guards();wb.guards();out.guards();baseline.guards();bb.guards();
    require(bb.read()==bias,"linear modified bias");
    require(xb.read()==x && wb.read()==w,"linear modified inputs");
}
static void rejection_case(Api& a) {
    F init(32,sentinel);Buffer x(a,init),y(a,init);
    require(a.ns_image_rb_maxpool2(nullptr,x.fp(),2,2,1,y.fp())!=0,"null GPU accepted");
    require(a.ns_image_rb_maxpool2(a.gpu,x.fp(),0,2,1,y.fp())!=0,"zero pool accepted");
    require(a.ns_image_rb_maxpool2(a.gpu,x.fp(),INT64_MAX,2,1,y.fp())!=0,"overflow pool accepted");
    require(a.ns_image_rb_maxpool2(a.gpu,x.fp(),2,2,1,x.fp())!=0,"inplace pool accepted");
    require(a.ns_image_rb_resize(a.gpu,x.fp(),2,2,3,4,4,y.fp(),2)!=0,"short stride accepted");
    require(a.ns_image_rb_resize(a.gpu,x.fp(),2,2,1,4,4,y.fp(),INT64_MAX)!=0,"overflow stride accepted");
    require(a.ns_image_rb_resize(a.gpu,x.fp(),2,2,1,2,2,x.fp(),1)!=0,"inplace resize accepted");
    require(a.ns_image_rb_bias_act(a.gpu,x.fp(),4,0,nullptr,0)!=0,"zero channels accepted");
    require(a.ns_image_rb_bias_act(a.gpu,x.fp(),1,1,nullptr,2)!=0,"invalid relu accepted");
    require(a.ns_image_rb_sigmoid(a.gpu,x.fp(),-1)!=0,"negative count accepted");
    require(a.ns_image_rb_sigmoid(a.gpu,nullptr,1)!=0,"null tensor accepted");
    require(a.ns_image_rb_sigmoid(a.gpu,x.fp(),I(INT32_MAX)*256+1)!=0,"oversize grid accepted");
    require(a.ns_image_rb_linear_tiled(a.gpu,x.fp(),y.fp(),0,1,1,y.fp())!=0,"invalid linear accepted");
    require(a.ns_image_rb_linear_tiled(a.gpu,x.fp(),x.fp(),65535LL*64+1,1,1,y.fp())!=0,"linear grid overflow accepted");
    require(a.ns_image_rb_linear_tiled(a.gpu,x.fp(),x.fp(),1,INT64_MAX,1,y.fp())!=0,"linear storage overflow accepted");
    require(a.ns_image_rb_biref_linear_tiled(a.gpu,x.fp(),x.fp(),1,1,1,y.fp(),y.fp())!=0,"bias-output alias accepted");
    require(a.ns_image_rb_biref_linear_tiled(a.gpu,x.fp(),x.fp(),65535LL*64+1,1,1,nullptr,y.fp())!=0,"biref grid overflow accepted");
    a.sync();require(x.read()==init&&y.read()==init,"rejected call changed memory");x.guards();y.guards();
}
int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
        require(argc==2||argc==3,"usage: rb-numeric --self-test | LIBRARY [GPU_INDEX]");
        Api a(argv[1],argc==3?std::stoi(argv[2]):0);
        int passed=0,failed=0;
        auto run=[&](const std::function<void()>& f){try{f();++passed;}catch(const std::exception& e){++failed;std::cerr<<"FAIL "<<e.what()<<'\n';}};
        const std::array<I,3> shapes[] = {
            {1,1,1},{15,15,17},{16,16,16},{17,17,15},{63,27,65},
            {64,32,64},{65,33,63},{67,144,71},{3,288,16},{5,576,32},
            {7,1152,64},{3,2304,128},{2,4608,256},{2,9216,512}
        };
        for(bool rounded:{false,true})for(const auto& shape:shapes)
            run([&]{linear_case(a,shape[0],shape[1],shape[2],rounded,0);});
        for(int backend:{1,2})for(bool rounded:{false,true})for(const auto& shape:shapes)
            run([&]{linear_case(a,shape[0],shape[1],shape[2],rounded,backend);});
        // Encoder projection/MLP widths and decoder offset-mask odd outputs.
        const std::array<I,3> biref_shapes[] = {
            {17,768,2304},{17,768,3072},{17,3072,768},{65,3136,147},
            {17,1536,6144},{17,6144,1536},{65,4320,64},{17,384,1536}
        };
        for(int backend:{1,2})for(bool rounded:{false,true})for(const auto& shape:biref_shapes)
            run([&]{linear_case(a,shape[0],shape[1],shape[2],rounded,backend);});
        for(I c:{I(1),I(3),I(33)}){
            run([&]{pool_case(a,1,1,c);});run([&]{pool_case(a,5,7,c);});run([&]{pool_case(a,6,8,c);});
            run([&]{resize_case(a,1,1,c,3,5,c+2);});run([&]{resize_case(a,3,5,c,7,9,c+5);});
            run([&]{resize_case(a,7,9,c,3,2,c);});run([&]{resize_case(a,3,5,c,3,5,c);});
        }
        for(bool b:{false,true})for(bool r:{false,true})run([&]{bias_case(a,17,33,b,r);});
        run([&]{sigmoid_case(a);});run([&]{rejection_case(a);});
        // A valid operation after rejected requests proves recovery.
        run([&]{pool_case(a,3,3,1);});
        std::cout<<"SUMMARY passed="<<passed<<" failed="<<failed<<'\n';return failed?1:0;
    }catch(const std::exception& e){std::cerr<<"FATAL "<<e.what()<<'\n';return 1;}
}
