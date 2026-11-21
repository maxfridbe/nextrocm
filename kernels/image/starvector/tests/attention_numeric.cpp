// Independent CPU double oracle for private StarVector attention; parent runs GPU.
#include "../../../ns/ns.h"
#include "../starvector.h"
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
// Independent IEEE binary16 decoding; packed bits are sent unchanged to HIP.
static float half_value(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    double value = e == 0 ? std::ldexp(double(m), -24)
                         : e == 31 ? (m ? std::numeric_limits<double>::quiet_NaN()
                                        : std::numeric_limits<double>::infinity())
                                   : std::ldexp(double(1024 + m), e - 25);
    return float((h & 0x8000) ? -value : value);
}
#define SYMBOLS(X) \
 X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) \
 X(ns_copy_to) X(ns_copy_from) X(ns_sync) X(nrsv_attention) X(nrsv_attention_causal_rows)
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
static uint16_t exact_half(float x){
 if(x==0)return 0;
 int e=0;float f=std::frexp(std::abs(x),&e);
 return uint16_t((x<0?0x8000:0)|((e+14)<<10)|int(f*2048-1024));
}
static void attention_case(Api& a,I row0,I rows,I heads,I kvheads,I dim,bool causal,int dt){
 I keys=causal?row0+rows:rows,qs=heads*dim+13,ks=kvheads*dim+17;
 if(!causal)ks=qs; // Vision ABI uses one interleaved stride for Q/K/V.
 F q(size_t(rows*qs)),k(size_t(keys*ks)),v(k.size());
 // Avoid periodic values correlated with strides (CLIP stride 1037 = 17*61
 // made a previous modulo-17 pattern constant across keys). Exactly half-
 // representable values keep input conversion independent of GPU arithmetic.
 auto fill=[](F& x,uint32_t seed){for(float& z:x){seed=1664525u*seed+1013904223u;z=float(int((seed>>16)%17)-8)/8;}};
 fill(q,1);fill(k,2);fill(v,3);
 size_t width=dt?2:4;
 Buffer dq(a,q.size()*width),dk(a,k.size()*width),dv(a,v.size()*width),out(a,size_t(rows*heads*dim)*width);
 auto upload=[&](Buffer& dst,const F& host){
  if(!dt){dst.write(host.data(),host.size()*4);return;}
  std::vector<uint16_t> bits;for(float x:host)bits.push_back(exact_half(x));dst.write(bits.data(),bits.size()*2);
 };
 upload(dq,q);upload(dk,k);upload(dv,v);
 if(causal)a.check(a.nrsv_attention_causal_rows(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),dt,row0,rows,heads,kvheads,dim,qs,ks,out.ptr()),"causal MQA");
 else a.check(a.nrsv_attention(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),dt,rows,heads,dim,qs,out.ptr(),dt),"vision attention");
 std::vector<unsigned char> raw(out.bytes);a.check(a.ns_copy_from(a.gpu,raw.data(),out.ptr(),raw.size()),"output read");a.sync();
 D reference(size_t(rows*heads*dim));
 for(I r=0;r<rows;++r)for(I h=0;h<heads;++h){
  I kh=h/(heads/kvheads),end=causal?row0+r+1:keys;D score(size_t(end),0.);
  for(I j=0;j<end;++j){for(I d=0;d<dim;++d)score[j]+=double(q[r*qs+h*dim+d])*k[j*ks+kh*dim+d];score[j]/=std::sqrt(double(dim));}
  double mx=*std::max_element(score.begin(),score.end()),den=0;
  for(double& x:score){x=std::exp(x-mx);den+=x;}
  for(double& x:score)x/=den;
  for(I d=0;d<dim;++d){double z=0;for(I j=0;j<end;++j)z+=score[j]*v[j*ks+kh*dim+d];reference[(r*heads+h)*dim+d]=z;}
 }
 double mx=0,e2=0;
 for(size_t i=0;i<reference.size();++i){
  float actual=0;if(dt){uint16_t x;std::memcpy(&x,raw.data()+2*i,2);actual=half_value(x);}else std::memcpy(&actual,raw.data()+4*i,4);
  double error=std::abs(actual-reference[i]);require(std::isfinite(actual),"nonfinite output");mx=std::max(mx,error);e2+=error*error;
  require(error<=(dt?0.002:4e-5)+(dt?0.002:5e-4)*std::abs(reference[i]),"attention independent oracle mismatch");
 }
 dq.guards();dk.guards();dv.guards();out.guards();
 std::cout<<"PASS row0="<<row0<<" rows="<<rows<<" keys="<<keys<<" hq="<<heads<<" hkv="<<kvheads<<" dim="<<dim<<" dt="<<dt<<" max_abs="<<mx<<" rms="<<std::sqrt(e2/reference.size())<<'\n';
}
int main(int argc,char** argv){
 try{
  require(argc==3,"usage: starvector-attention-numeric ABS_LIBRARY GPU_INDEX");
  Api a(argv[1],std::stoi(argv[2]));
  for(int dt:{0,1}){
   attention_case(a,0,257,16,16,64,false,dt);
   attention_case(a,0,259,16,1,128,true,dt);
   attention_case(a,4094,3,16,1,128,true,dt);
   attention_case(a,4096,1,16,1,128,true,dt);
   attention_case(a,8191,1,16,1,128,true,dt);
   attention_case(a,0,5,4,2,7,true,dt);
  }
  {
   Buffer x(a,F(16,1)),y(a,F(16,sentinel));
   require(a.nrsv_attention_causal_rows(a.gpu,x.ptr(),x.ptr(),x.ptr(),1,8192,1,16,1,128,2048,128,y.ptr())!=0,"position cap accepted");
   require(a.nrsv_attention_causal_rows(a.gpu,x.ptr(),x.ptr(),x.ptr(),1,INT64_MAX,INT64_MAX,16,1,128,2048,128,y.ptr())!=0,"overflow accepted");
   for(float z:y.read()){require(z==sentinel,"invalid call wrote output");}
   x.guards();y.guards();
  }
  using Stats=int (*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
  uint64_t live=0,peak=0;a.check(stats(a.gpu,&live,&peak),"stats");require(live==0,"tracked allocation leak");
  std::cout<<"PASS 12 independent cases + invalid geometry; zero live bytes\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
