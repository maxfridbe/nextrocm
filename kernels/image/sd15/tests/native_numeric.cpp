// Standalone GPU numeric validator. No HIP headers or inference framework needed.
// Build: c++ -std=c++17 -O2 -Wall -Wextra -Werror native_numeric.cpp -ldl -o /tmp/nrsd-numeric
// Host only: /tmp/nrsd-numeric --self-test (does not load any GPU library)
// GPU (parent serializes): /tmp/nrsd-numeric /absolute/libnextrocm.so GPU_INDEX
// Every case has independent double CPU math, guard words, finite-value checks,
// explicit stream synchronization and a nonzero exit status on any failure.
#include "../../../ns/ns.h"
#include "../sd15.h"
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
// Independent IEEE binary16 decoding; packed bits are sent unchanged to HIP.
static float half_value(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    double value = e == 0 ? std::ldexp(double(m), -24)
                         : e == 31 ? (m ? std::numeric_limits<double>::quiet_NaN()
                                        : std::numeric_limits<double>::infinity())
                                   : std::ldexp(double(1024 + m), e - 25);
    return float((h & 0x8000) ? -value : value);
}
static double activation(double x, int kind) {
    if (kind == 0) return x / (1 + std::exp(-x));
    if (kind == 2) return x / (1 + std::exp(-1.702 * x));
    if (kind == 3) return .5 * x * (1 + std::tanh(std::sqrt(2 / std::acos(-1.)) * (x + .044715*x*x*x)));
    return .5 * x * (1 + std::erf(x / std::sqrt(2.)));
}
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) \
 X(nrsd_linear) X(nrsd_conv2d) X(nrsd_group_norm) X(nrsd_layer_norm) \
 X(nrsd_activation) X(nrsd_add) X(nrsd_multiply) X(nrsd_geglu) \
 X(nrsd_time_emb) X(nrsd_layout) X(nrsd_up2) X(nrsd_concat) X(nrsd_attention)
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
struct Weight {
    int dtype;
    F decoded;
    std::vector<uint16_t> bits;
    Buffer device;
    Weight(Api& a, size_t count, int type, uint32_t seed) : dtype(type), decoded(values(count, seed)),
        bits(type ? count : 0), device(a, count*(type ? 2 : 4)) {
        if (type) {
            for (size_t i=0;i<count;++i) {
                seed=seed*1664525u+1013904223u;
                bits[i]=uint16_t(((seed>>16)&0x8000) | ((12+(seed%4))<<10) | ((seed>>4)&1023));
                // Include zeros, negative zero, and half subnormals in real calls.
                if (i%31==0) bits[i]=uint16_t(i%2 ? 0x8000 : 1);
                decoded[i]=half_value(bits[i]);
            }
            device.write(bits.data(), count*2);
        } else device.write(decoded.data(), count*4);
    }
    nrsd_weight raw() const { return {device.ptr(), dtype}; }
};
static nrsd_weight none() { return {nullptr,0}; }
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
static void linear_case(Api& a,int wt,int bt,bool has_bias) {
    nrsd_linear_shape s{19,23,17,29,21};
    F x=values(size_t(s.rows*s.x_stride));
    Weight w(a,size_t(s.output*s.input),wt,31),b(a,size_t(s.output),bt,9);
    Buffer dx(a,x),out(a,F(size_t(s.rows*s.out_stride),sentinel));
    D ref(size_t(s.rows*s.out_stride),sentinel);
    for(I row=0;row<s.rows;++row)for(I co=0;co<s.output;++co){
        double v=has_bias?b.decoded[co]:0;
        for(I ci=0;ci<s.input;++ci)v+=double(x[row*s.x_stride+ci])*w.decoded[co*s.input+ci];
        ref[row*s.out_stride+co]=v;
    }
    a.check(a.nrsd_linear(a.gpu,dx.fp(),w.raw(),has_bias?b.raw():none(),out.fp(),s),"linear");
    compare("linear_w"+std::to_string(wt)+"_b"+std::to_string(bt)+(has_bias?"":"_none"),out,ref);
    dx.guards();w.device.guards();b.device.guards();
}
static void conv_case(Api& a,int wt,int bt,I kernel,I stride,I pad,bool has_bias) {
    nrsd_conv_shape s{2,3,7,9,19,kernel,kernel,stride,stride,pad,pad,0,0};
    s.oh=(s.height+2*pad-kernel)/stride+1;s.ow=(s.width+2*pad-kernel)/stride+1;
    F x=values(size_t(s.batch*s.ci*s.height*s.width));
    Weight w(a,size_t(s.co*s.ci*kernel*kernel),wt,31),b(a,size_t(s.co),bt,9);
    Buffer dx(a,x),out(a,F(size_t(s.batch*s.co*s.oh*s.ow),sentinel));
    D ref;
    for(I nb=0;nb<s.batch;++nb)for(I co=0;co<s.co;++co)for(I oy=0;oy<s.oh;++oy)for(I ox=0;ox<s.ow;++ox){
        double z=has_bias?b.decoded[co]:0;
        for(I ci=0;ci<s.ci;++ci)for(I ky=0;ky<kernel;++ky)for(I kx=0;kx<kernel;++kx){
            I iy=oy*stride-pad+ky,ix=ox*stride-pad+kx;
            if(iy>=0&&iy<s.height&&ix>=0&&ix<s.width)
                z+=double(x[((nb*s.ci+ci)*s.height+iy)*s.width+ix])*w.decoded[((co*s.ci+ci)*kernel+ky)*kernel+kx];
        }
        ref.push_back(z);
    }
    a.check(a.nrsd_conv2d(a.gpu,dx.fp(),w.raw(),has_bias?b.raw():none(),out.fp(),s),"conv2d");
    compare("conv_k"+std::to_string(kernel)+"_s"+std::to_string(stride)+"_p"+std::to_string(pad)+"_w"+std::to_string(wt)+"_b"+std::to_string(bt)+(has_bias?"":"_none"),out,ref,8e-5,4e-4);
    dx.guards();w.device.guards();b.device.guards();
}
static void norm_case(Api& a,bool group,int wt,int bt,bool affine,bool in_place) {
    const I n=2,c=group?64:259,h=group?3:1,w=group?5:1,g=group?32:1;
    F x=values(size_t(n*c*h*w));
    // Centered variance with a substantial nonzero mean; LN width crosses 256.
    for(float& v:x)v+=16.f;
    Weight gamma(a,size_t(c),wt,51),beta(a,size_t(c),bt,61);
    Buffer dx(a,x),out(a,F(x.size(),sentinel));
    D ref(x.size()); const I count=c/g*h*w;
    const float eps=group?1e-6f:1e-5f;
    for(I batch=0;batch<n;++batch)for(I gr=0;gr<g;++gr){
        I offset=(batch*g+gr)*count;
        double mean=0,variance=0;
        for(I j=0;j<count;++j)mean+=x[offset+j];
        mean/=count;
        for(I j=0;j<count;++j){double d=x[offset+j]-mean;variance+=d*d;}
        double inv=1/std::sqrt(variance/count+eps);
        for(I j=0;j<count;++j){I channel=gr*(c/g)+j/(h*w);ref[offset+j]=(x[offset+j]-mean)*inv*(affine?gamma.decoded[channel]:1)+(affine?beta.decoded[channel]:0);}
    }
    Buffer& target=in_place?dx:out;
    if(group)a.check(a.nrsd_group_norm(a.gpu,dx.fp(),affine?gamma.raw():none(),affine?beta.raw():none(),target.fp(),n,c,h,w,g,eps),"group norm");
    else a.check(a.nrsd_layer_norm(a.gpu,dx.fp(),affine?gamma.raw():none(),affine?beta.raw():none(),target.fp(),n,c,eps),"layer norm");
    compare(std::string(group?"GN32":"LN")+"_w"+std::to_string(wt)+"_b"+std::to_string(bt)+(affine?"":"_noaffine")+(in_place?"_inplace":""),target,ref,1e-4,4e-4);
    dx.guards();gamma.device.guards();beta.device.guards();
}
static void activation_case(Api& a,int kind,bool in_place) {
    F x=values(517);for(float& v:x)v*=6;
    Buffer dx(a,x),out(a,F(x.size(),sentinel));D ref;
    for(float v:x)ref.push_back(activation(v,kind));
    Buffer& target=in_place?dx:out;
    a.check(a.nrsd_activation(a.gpu,dx.fp(),target.fp(),I(x.size()),kind),"activation");
    compare("activation"+std::to_string(kind)+(in_place?"_inplace":""),target,ref);dx.guards();
}
static void geglu_case(Api& a) {
    I rows=3,c=173;F x=values(size_t(rows*2*c));D ref;
    // Negative gate -2 separates tanh from erf beyond the test tolerance.
    x[0]=2.f;x[c]=-2.f;
    for(I row=0;row<rows;++row)for(I j=0;j<c;++j)ref.push_back(double(x[row*2*c+j])*activation(x[row*2*c+c+j],3));
    Buffer dx(a,x),out(a,F(size_t(rows*c),sentinel));
    a.check(a.nrsd_geglu(a.gpu,dx.fp(),out.fp(),rows,c),"geglu");compare("geglu_tanh_value_then_gate",out,ref,2e-5,3e-5);dx.guards();
}
static void elementwise_case(Api& a,int op,bool in_place) {
    F x=values(517),y=values(517,97);D ref;
    for(size_t i=0;i<x.size();++i)ref.push_back(op==2?double(x[i])*y[i]:.375*double(x[i])+(op==0?-.75*double(y[i]):0));
    Buffer dx(a,x),dy(a,y),out(a,F(x.size(),sentinel));Buffer& target=in_place?dx:out;
    if(op==2)a.check(a.nrsd_multiply(a.gpu,dx.fp(),dy.fp(),target.fp(),I(x.size())),"multiply");
    else a.check(a.nrsd_add(a.gpu,dx.fp(),op==0?dy.fp():nullptr,target.fp(),I(x.size()),.375f,-.75f),"add/scale");
    compare("elementwise"+std::to_string(op)+(in_place?"_inplace":""),target,ref);dx.guards();dy.guards();
}
static void spatial_case(Api& a) {
    const I n=2,c=3,h=5,w=7,cb=2;
    F x=values(size_t(n*c*h*w)),y=values(size_t(n*cb*h*w),41),emb=values(size_t(n*c),12);
    D hwc(x.size()),up(size_t(n*c*4*h*w)),cat(size_t(n*(c+cb)*h*w)),time(x.size());
    for(I nb=0;nb<n;++nb)for(I ch=0;ch<c;++ch)for(I iy=0;iy<h;++iy)for(I ix=0;ix<w;++ix){
        I src=((nb*c+ch)*h+iy)*w+ix;
        hwc[((nb*h+iy)*w+ix)*c+ch]=x[src];time[src]=double(x[src])+emb[nb*c+ch];
        for(I dy=0;dy<2;++dy)for(I dx=0;dx<2;++dx)up[((nb*c+ch)*(2*h)+2*iy+dy)*(2*w)+2*ix+dx]=x[src];
    }
    for(I nb=0;nb<n;++nb)for(I ch=0;ch<c+cb;++ch)for(I p=0;p<h*w;++p)
        cat[(nb*(c+cb)+ch)*h*w+p]=ch<c?x[(nb*c+ch)*h*w+p]:y[(nb*cb+ch-c)*h*w+p];
    Buffer dx(a,x),dy(a,y),de(a,emb),layout(a,F(x.size(),sentinel)),back(a,F(x.size(),sentinel)),
        du(a,F(up.size(),sentinel)),dc(a,F(cat.size(),sentinel)),dt(a,F(x.size(),sentinel));
    a.check(a.nrsd_layout(a.gpu,dx.fp(),layout.fp(),n,c,h,w,1),"NCHW -> NHWC");compare("NCHW_to_NHWC",layout,hwc,0,0);
    a.check(a.nrsd_layout(a.gpu,layout.fp(),back.fp(),n,c,h,w,0),"NHWC -> NCHW");compare("NHWC_to_NCHW",back,D(x.begin(),x.end()),0,0);
    a.check(a.nrsd_up2(a.gpu,dx.fp(),du.fp(),n,c,h,w),"up2");compare("nearest_up2",du,up,0,0);
    a.check(a.nrsd_concat(a.gpu,dx.fp(),dy.fp(),dc.fp(),n,c,cb,h,w),"concat");compare("channel_concat",dc,cat,0,0);
    a.check(a.nrsd_time_emb(a.gpu,dx.fp(),de.fp(),dt.fp(),n,c,h,w),"time embedding");compare("time_embedding",dt,time);
    a.check(a.nrsd_time_emb(a.gpu,dx.fp(),de.fp(),dx.fp(),n,c,h,w),"in-place time embedding");compare("time_embedding_inplace",dx,time);
    dx.guards();dy.guards();de.guards();
}
// Construct padded token-major or head-major layout, independently for Q/K/V/O.
static void strides(I tokens,I heads,I dim,bool head_major,I& bs,I& ts,I& hs) {
    if(head_major){ts=dim+3;hs=tokens*ts+5;bs=heads*hs+7;}
    else {hs=dim;ts=heads*dim+11;bs=tokens*ts+13;}
}
static void attention_case(Api& a,I queries,I keys,I heads,I dim,bool causal,bool mixed,bool extreme=false) {
    nrsd_attention_shape s{};s.batch=2;s.queries=queries;s.keys=keys;s.heads=heads;s.dim=dim;
    s.scale=float(1/std::sqrt(double(dim)));s.causal=causal;
    strides(queries,heads,dim,false,s.qb,s.qt,s.qh);
    strides(keys,heads,dim,mixed,s.kb,s.kt,s.kh);
    strides(keys,heads,dim,!mixed,s.vb,s.vt,s.vh);
    strides(queries,heads,dim,mixed,s.ob,s.ot,s.oh);
    F q(size_t(s.batch*s.qb),sentinel),k(size_t(s.batch*s.kb),sentinel),v(size_t(s.batch*s.vb),sentinel);
    F qvals=values(size_t(s.batch*queries*heads*dim),1),kvals=values(size_t(s.batch*keys*heads*dim),2),vvals=values(kvals.size(),3);
    size_t at=0;
    for(I b=0;b<s.batch;++b)for(I t=0;t<queries;++t)for(I h=0;h<heads;++h)for(I d=0;d<dim;++d)q[b*s.qb+t*s.qt+h*s.qh+d]=extreme?10.f:qvals[at++];
    at=0;
    for(I b=0;b<s.batch;++b)for(I t=0;t<keys;++t)for(I h=0;h<heads;++h)for(I d=0;d<dim;++d){
        k[b*s.kb+t*s.kt+h*s.kh+d]=extreme?(t<256?-10.f:10.f):kvals[at];
        v[b*s.vb+t*s.vt+h*s.vh+d]=vvals[at++];
    }
    D ref(size_t(s.batch*s.ob),sentinel);
    for(I b=0;b<s.batch;++b)for(I t=0;t<queries;++t)for(I h=0;h<heads;++h){
        I end=causal?std::min(keys,t+1):keys;D scores(size_t(end),0.);
        for(I j=0;j<end;++j){for(I d=0;d<dim;++d)scores[j]+=double(q[b*s.qb+t*s.qt+h*s.qh+d])*k[b*s.kb+j*s.kt+h*s.kh+d];scores[j]*=s.scale;}
        double max=*std::max_element(scores.begin(),scores.end()),den=0;
        for(double& z:scores){z=std::exp(z-max);den+=z;}
        for(I d=0;d<dim;++d){double z=0;for(I j=0;j<end;++j)z+=scores[j]*v[b*s.vb+j*s.vt+h*s.vh+d];ref[b*s.ob+t*s.ot+h*s.oh+d]=z/den;}
    }
    Buffer dq(a,q),dk(a,k),dv(a,v),out(a,F(ref.size(),sentinel));
    a.check(a.nrsd_attention(a.gpu,dq.fp(),dk.fp(),dv.fp(),out.fp(),s),"attention");
    compare("attention_Q"+std::to_string(queries)+"_K"+std::to_string(keys)+"_D"+std::to_string(dim)+(causal?"_causal":"")+(mixed?"_mixed":"_token")+(extreme?"_rescale":""),out,ref,4e-5,5e-4);
    dq.guards();dk.guards();dv.guards();
}
static void rejection_case(Api& a) {
    Buffer x(a,F(128,1.f)),out(a,F(128,sentinel));
    auto rejected=[&](int rc,const char* label){require(rc!=0,std::string(label)+": invalid input accepted");const char* error=a.ns_last_error();require(error&&*error,std::string(label)+": missing error text");};
    rejected(a.nrsd_activation(a.gpu,x.fp(),out.fp(),128,99),"bad activation");
    rejected(a.nrsd_layer_norm(a.gpu,x.fp(),none(),none(),out.fp(),1,128,-1.f),"bad epsilon");
    rejected(a.nrsd_group_norm(a.gpu,x.fp(),none(),none(),out.fp(),1,3,2,2,2,1e-6f),"bad groups");
    nrsd_linear_shape linear{1,4,4,4,4};
    rejected(a.nrsd_linear(a.gpu,x.fp(),{x.ptr(),2},none(),out.fp(),linear),"bad weight dtype");
    nrsd_conv_shape conv{1,1,3,3,1,3,3,1,1,0,0,3,3};
    rejected(a.nrsd_conv2d(a.gpu,x.fp(),{x.ptr(),0},none(),out.fp(),conv),"bad output geometry");
    nrsd_attention_shape s{};s.batch=1;s.queries=1;s.keys=1;s.heads=1;s.dim=513;s.scale=1;
    s.qb=s.qt=s.qh=s.kb=s.kt=s.kh=s.vb=s.vt=s.vh=s.ob=s.ot=s.oh=513;
    rejected(a.nrsd_attention(a.gpu,x.fp(),x.fp(),x.fp(),out.fp(),s),"unsupported head dimension");
    compare("invalid_calls_leave_output_unchanged",out,D(128,sentinel),0,0);
}
static void host_self_test() {
    require(half_value(0x3c00)==1.f,"half one");require(half_value(0xc000)==-2.f,"half negative");
    require(half_value(0x7bff)==65504.f,"half max");require(half_value(1)==std::ldexp(1.f,-24),"half subnormal");
    require(std::signbit(half_value(0x8000)),"half negative zero");require(std::isinf(half_value(0x7c00)),"half infinity");
    require(std::isnan(half_value(0x7e00)),"half NaN");
    require(std::abs(activation(1,1)-.8413447460685429)<1e-14,"GELU reference");
    require(std::abs(activation(1,2)-.8457957659328212)<1e-14,"QuickGELU reference");
    require(std::abs(activation(-2,3)-activation(-2,1))>9e-5,"tanh/erf distinction");
    std::cout<<"PASS host half-decoding and activation reference checks (no GPU library loaded)\n";
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--self-test"){host_self_test();return 0;}
        if(argc!=3){std::cerr<<"usage: "<<argv[0]<<" --self-test | /absolute/libnextrocm.so GPU_INDEX\n";return 2;}
        size_t used=0;int index=std::stoi(argv[2],&used);require(index>=0&&used==std::strlen(argv[2]),"invalid GPU index");
        Api a(argv[1],index);int passed=0,failed=0;
        std::cout<<std::setprecision(9);
        auto run=[&](const std::string& name,const std::function<void()>& f){
            try{f();a.sync();++passed;std::cout<<"PASS "<<name<<'\n';}
            catch(const std::exception& e){++failed;std::cerr<<"FAIL "<<name<<": "<<e.what()<<'\n';}
        };
        for(int wt=0;wt<2;++wt)for(int bt=0;bt<2;++bt){
            run("linear",[&]{linear_case(a,wt,bt,true);});
            run("conv1x1",[&]{conv_case(a,wt,bt,1,1,0,true);});
            run("conv3x3_stride1",[&]{conv_case(a,wt,bt,3,1,1,true);});
            run("conv3x3_stride2",[&]{conv_case(a,wt,bt,3,2,1,true);});
            run("groupnorm32",[&]{norm_case(a,true,wt,bt,true,false);});
            run("layernorm",[&]{norm_case(a,false,wt,bt,true,false);});
        }
        run("linear_no_bias",[&]{linear_case(a,1,0,false);});
        run("conv_valid_no_bias",[&]{conv_case(a,1,0,3,2,0,false);});
        for(bool group:{false,true}){
            run("norm_no_affine",[&]{norm_case(a,group,0,0,false,false);});
            run("norm_inplace",[&]{norm_case(a,group,1,0,true,true);});
        }
        for(int kind=0;kind<4;++kind)for(bool inplace:{false,true})run("activation",[&]{activation_case(a,kind,inplace);});
        run("GEGLU",[&]{geglu_case(a);});
        for(int op=0;op<3;++op)for(bool inplace:{false,true})run("elementwise",[&]{elementwise_case(a,op,inplace);});
        run("spatial",[&]{spatial_case(a);});
        for(I dim:{40,80,160}){
            run("attention_self257",[&]{attention_case(a,257,257,2,dim,false,false);});
            run("attention_cross77",[&]{attention_case(a,19,77,3,dim,false,true);});
        }
        run("causal_259",[&]{attention_case(a,259,259,2,40,true,false);});
        run("CLIP_causal",[&]{attention_case(a,77,77,3,64,true,true);});
        run("VAE_attention",[&]{attention_case(a,17,257,1,512,false,true);});
        run("attention_keys4096",[&]{attention_case(a,3,4096,1,40,false,true);});
        run("attention_online_rescale",[&]{attention_case(a,3,259,1,40,false,false,true);});
        run("invalid_parameters",[&]{rejection_case(a);});
        std::cout<<"SUMMARY passed="<<passed<<" failed="<<failed<<'\n';
        return failed?1:0;
    } catch(const std::exception& e){std::cerr<<"FATAL: "<<e.what()<<'\n';return 1;}
}
