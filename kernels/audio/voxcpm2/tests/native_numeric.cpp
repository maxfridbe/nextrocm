// Standalone GPU numeric validator. No HIP headers or inference framework needed.
// Build: c++ -std=c++17 -O2 -Wall -Wextra -Werror -I../../../ns native_numeric.cpp -ldl -o /tmp/vcp-numeric
// Host only: /tmp/vcp-numeric --self-test (does not load any GPU library)
// GPU (parent serializes): /tmp/vcp-numeric /absolute/libnextrocm.so GPU_INDEX
// Every case has independent double CPU math, guard words, finite-value checks,
// explicit stream synchronization and a nonzero exit status on any failure.
#include "../../../ns/ns.h"
#include "../vcp.h"
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
static F values(size_t n, uint32_t seed = 7) {
    F result(n);
    for (float& v : result) {
        seed = seed * 1664525u + 1013904223u;
        v = float(int((seed >> 8) % 2001) - 1000) / 731.f;
    }
    return result;
}
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) \
 X(ns_audio_vcp_rope) X(ns_audio_vcp_local_attn) X(ns_audio_vcp_snake) \
 X(ns_audio_vcp_dwconv) X(ns_audio_vcp_affine) X(ns_audio_vcp_fsq)
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
static void compare(const std::string& label, const Buffer& out, const D& expected,
                    double atol=3e-5, double rtol=3e-4) {
    const F actual=out.read();
    out.guards();
    require(actual.size()==expected.size(), label+": reference size mismatch");
    double max_abs=0, squared=0; size_t worst=0, failures=0;
    for(size_t i=0;i<actual.size();++i) {
        require(std::isfinite(actual[i]) && std::isfinite(expected[i]),label+": nonfinite at "+std::to_string(i));
        double err=std::abs(double(actual[i])-expected[i]);
        if(err>max_abs){max_abs=err;worst=i;}
        squared+=err*err;
        if (expected[i]==sentinel) { if(actual[i]!=sentinel) ++failures; }
        else if(err>atol+rtol*std::abs(expected[i])) ++failures;
    }
    std::cout << label << " max_abs=" << max_abs << " rms=" << std::sqrt(squared/actual.size())
              << " worst=" << worst << " actual=" << actual[worst] << " expected=" << expected[worst] << '\n';
    require(!failures,label+": "+std::to_string(failures)+" elements exceed tolerance");
}
static void rope_case(Api& a) {
    I rows=15,seq=5,heads=2,dim=128,stride=heads*dim+7,p0=1003;
    F x=values(size_t(rows*stride)),freq(size_t(dim/2));
    D ref(x.begin(),x.end());
    for(I i=0;i<dim/2;++i)freq[i]=float(1/std::pow(10000.,2.*i/dim)/(1.+i*.01));
    for(I r=0;r<rows;++r)for(I h=0;h<heads;++h)for(I d=0;d<dim/2;++d){
        I j=r*stride+h*dim+d;float angle=float(p0+r%seq)*freq[d];
        double c=std::cos(double(angle)),s=std::sin(double(angle));
        ref[j]=double(x[j])*c-double(x[j+dim/2])*s;
        ref[j+dim/2]=double(x[j+dim/2])*c+double(x[j])*s;
    }
    Buffer dx(a,x),df(a,freq);
    a.check(a.ns_audio_vcp_rope(a.gpu,dx.fp(),stride,rows,seq,heads,dim,df.fp(),p0),"rope");
    compare("long_rope_rotate_half_strided_groups",dx,ref,4e-5,4e-4);
    for(float& f:freq)f=-f;
    df.write(freq.data(),freq.size()*4);
    a.check(a.ns_audio_vcp_rope(a.gpu,dx.fp(),stride,rows,seq,heads,dim,df.fp(),p0),"inverse rope");
    compare("rope_inverse",dx,D(x.begin(),x.end()),4e-5,4e-4);df.guards();
}
static void attention_case(Api& a,I seq,I heads,I kv) {
    const I batch=2,dim=128,qs=heads*dim+7,ks=kv*dim+11;
    F q=values(size_t(batch*seq*qs),5),k=values(size_t(batch*seq*ks),11),v=values(k.size(),19);
    D ref(size_t(batch*seq*heads*dim));
    const float scale=1.f/std::sqrt(float(dim));
    for(I b=0;b<batch;++b)for(I row=0;row<seq;++row)for(I h=0;h<heads;++h){
        D prob(size_t(seq),0.);I kh=h/(heads/kv);
        for(I key=0;key<seq;++key){for(I d=0;d<dim;++d)prob[key]+=double(q[(b*seq+row)*qs+h*dim+d])*k[(b*seq+key)*ks+kh*dim+d];prob[key]*=scale;}
        const double maximum=*std::max_element(prob.begin(),prob.end());double den=0;
        for(double& z:prob){z=std::exp(z-maximum);den+=z;}
        for(I d=0;d<dim;++d){double z=0;for(I key=0;key<seq;++key)z+=prob[key]*v[(b*seq+key)*ks+kh*dim+d];ref[((b*seq+row)*heads+h)*dim+d]=z/den;}
    }
    Buffer dq(a,q),dk(a,k),dv(a,v),out(a,F(ref.size(),sentinel));
    a.check(a.ns_audio_vcp_local_attn(a.gpu,dq.fp(),dk.fp(),dv.fp(),batch,seq,heads,kv,dim,qs,ks,out.fp()),"local attention");
    compare("local_gqa_s"+std::to_string(seq)+"_heads"+std::to_string(heads)+"_kv"+std::to_string(kv),out,ref);
    dq.guards();dk.guards();dv.guards();
}
static void snake_case(Api& a) {
    I c=3,l=263;F x=values(size_t(c*l)),alpha{.001f,.75f,2.3f};D ref;
    for(I ch=0;ch<c;++ch)for(I t=0;t<l;++t){double v=x[ch*l+t],s=std::sin(double(alpha[ch])*v);ref.push_back(v+s*s/(double(alpha[ch])+1e-9));}
    Buffer dx(a,x),da(a,alpha),out(a,F(x.size(),sentinel));
    a.check(a.ns_audio_vcp_snake(a.gpu,dx.fp(),c,l,da.fp(),out.fp()),"snake");compare("snake_nonmultiple263",out,ref);dx.guards();da.guards();
}
static void dwconv_case(Api& a,I dilation,bool has_bias) {
    I c=3,l=263,width=7;F x=values(size_t(c*l)),w=values(size_t(c*width),11),bias=values(size_t(c),9);D ref;
    for(I ch=0;ch<c;++ch)for(I t=0;t<l;++t){
        double z=has_bias?bias[ch]:0;
        for(I j=0;j<width;++j){I at=t-(width-1-j)*dilation;if(at>=0)z+=double(x[ch*l+at])*w[ch*width+j];}
        ref.push_back(z);
    }
    Buffer dx(a,x),dw(a,w),db(a,bias),out(a,F(x.size(),sentinel));
    a.check(a.ns_audio_vcp_dwconv(a.gpu,dx.fp(),c,l,dw.fp(),has_bias?db.fp():nullptr,width,dilation,out.fp()),"depthwise conv");
    compare("causal_dwconv_d"+std::to_string(dilation)+(has_bias?"_bias":"_nobias"),out,ref);dx.guards();dw.guards();db.guards();
}
static void affine_case(Api& a) {
    I c=3,l=263;F x=values(size_t(c*l)),scale{.25f,-2.f,0.f},bias{-.7f,.3f,2.f};D ref;
    for(I ch=0;ch<c;++ch)for(I t=0;t<l;++t)ref.push_back(double(x[ch*l+t])*scale[ch]+bias[ch]);
    Buffer dx(a,x),ds(a,scale),db(a,bias);
    a.check(a.ns_audio_vcp_affine(a.gpu,dx.fp(),c,l,ds.fp(),db.fp()),"affine");compare("affine_channels_inplace",dx,ref);ds.guards();db.guards();
}
static void fsq_case(Api& a,float scale) {
    F x=values(517);for(float& v:x)v*=3;
    // Saturated tanh and s=.5 give exactly representable +/-0.5 ties: both to 0.
    x[0]=100;x[1]=-100;x[2]=0;
    D ref;for(float v:x)ref.push_back(std::nearbyint(std::tanh(double(v))*scale)/scale);
    Buffer dx(a,x);
    a.check(a.ns_audio_vcp_fsq(a.gpu,dx.fp(),I(x.size()),scale),"FSQ");compare("fsq_ties_even_scale"+std::to_string(scale),dx,ref,1e-7,1e-7);
}
static void rejects_case(Api& a) {
    Buffer x(a,F(1024,1.f)),out(a,F(1024,sentinel));
    auto reject=[&](int status){require(status!=0,"invalid parameter accepted");const char* e=a.ns_last_error();require(e&&*e,"missing error text");};
    reject(a.ns_audio_vcp_local_attn(a.gpu,x.fp(),x.fp(),x.fp(),1,65,1,1,128,128,128,out.fp()));
    reject(a.ns_audio_vcp_local_attn(a.gpu,x.fp(),x.fp(),x.fp(),1,5,3,2,128,384,256,out.fp()));
    reject(a.ns_audio_vcp_rope(a.gpu,x.fp(),128,1,1,1,127,x.fp(),0));
    reject(a.ns_audio_vcp_dwconv(a.gpu,x.fp(),1,10,x.fp(),nullptr,7,0,out.fp()));
    reject(a.ns_audio_vcp_fsq(a.gpu,x.fp(),100,0));
    compare("invalid_calls_preserve_output",out,D(1024,sentinel),0,0);
    compare("invalid_inplace_calls_preserve_input",x,D(1024,1),0,0);
}
static void self_test() {
    require(std::nearbyint(.5)==0&&std::nearbyint(1.5)==2&&std::nearbyint(2.5)==2,"ties-even CPU rounding");
    require(std::nearbyint(std::tanh(100.)*.5)/.5==0.,"saturated half-scale tie");
    require(std::abs(std::sin(.7)*std::sin(.7)+std::cos(.7)*std::cos(.7)-1)<1e-14,"rotation reference");
    std::cout<<"PASS CPU reference self-test; no GPU library loaded\n";
}
int main(int argc,char** argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--self-test"){self_test();return 0;}
        if(argc!=3){std::cerr<<"usage: "<<argv[0]<<" --self-test | LIBRARY_PATH GPU_INDEX\n";return 2;}
        size_t used=0;int index=std::stoi(argv[2],&used);require(index>=0&&used==std::strlen(argv[2]),"invalid GPU index");
        Api api(argv[1],index);int pass=0,fail=0;
        std::cout<<std::setprecision(9);
        auto run=[&](const char* label,const std::function<void()>& fn){try{fn();api.sync();++pass;std::cout<<"PASS "<<label<<'\n';}catch(const std::exception& e){++fail;std::cerr<<"FAIL "<<label<<": "<<e.what()<<'\n';}};
        run("rope",[&]{rope_case(api);});
        for(I seq:{5,11,64})run("local attention",[&]{attention_case(api,seq,16,2);});
        run("local MHA",[&]{attention_case(api,5,2,2);});
        run("snake",[&]{snake_case(api);});
        for(I dilation:{1,3,9})run("depthwise",[&]{dwconv_case(api,dilation,true);});
        run("depthwise optional bias",[&]{dwconv_case(api,9,false);});
        run("affine",[&]{affine_case(api);});
        run("fsq9",[&]{fsq_case(api,9);});run("fsq exact ties",[&]{fsq_case(api,.5);});
        run("invalid parameters",[&]{rejects_case(api);});
        std::cout<<"SUMMARY passed="<<pass<<" failed="<<fail<<'\n';return fail?1:0;
    }catch(const std::exception& e){std::cerr<<"FATAL: "<<e.what()<<'\n';return 1;}
}
