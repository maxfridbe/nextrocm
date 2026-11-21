// Private 1024 image DiT attention: full native bitwise; full small-case F64
// and fixed 11-query x 5-head actual-shape F64 (all keys/features per row).
// CPU-only oracle/guard tests; GPU/SSH execution belongs to parent.
#include "../attention_1024.hpp"
#include "../../../ns/ns.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <cstdlib>
#include <future>
#include <array>
using I=int64_t;using F=std::vector<float>;using H=std::vector<uint16_t>;
constexpr float sentinel=-12345.25f;constexpr size_t guard=16;
static void require(bool ok,const std::string& s){if(!ok)throw std::runtime_error(s);}
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
struct Shape {I q,k,h,qs,ks;};
static size_t nout(Shape s){return size_t(s.q*s.h*128);}
static size_t at(I row,I head,I feature,I stride){return size_t(row*stride+head*128+feature);}
static F floats(const H& h){F out;out.reserve(h.size());for(auto v:h)out.push_back(decode(v));return out;}
// F64 dot, normalized softmax and full PV independent of kernel tiling/tree.
// Predeclared bound includes F32 QK/rsqrt/exp/PV/reduction and half rounding.
// The separate actual GPU baseline comparison remains exactly bitwise.
static void oracle(Shape s,const H& qh,const H& kh,const H& vh,const F& out,bool uniform,bool boundary_subset=false){
 const F q=floats(qh);require(out.size()==nout(s),"oracle shape");
 // Compact head-major K/V removes stride holes from the CPU working set.
 // Independent double arithmetic; inner feature loop vectorizes across outputs,
 // never across the ascending key reduction of any individual output.
 F k(size_t(s.h*s.k*128)),v(k.size());
 std::vector<std::array<double,128>> mv(size_t(s.h));
 std::vector<std::array<float,128>> uniform_sum(size_t(s.h));
 for(I h=0;h<s.h;++h)for(I j=0;j<s.k;++j)for(I d=0;d<128;++d){
  const size_t i=size_t((h*s.k+j)*128+d);
  k[i]=decode(kh[at(j,h,d,s.ks)]);v[i]=decode(vh[at(j,h,d,s.ks)]);
  mv[size_t(h)][size_t(d)]=std::max(mv[size_t(h)][size_t(d)],std::abs(double(v[i])));
  if(uniform)uniform_sum[size_t(h)][size_t(d)]=std::fma(1.f,v[i],uniform_sum[size_t(h)][size_t(d)]);
 }
 const double scale=1/std::sqrt(128.),u=std::ldexp(1.,-24),gd=128*u/(1-128*u),gs=(s.k+256)*u/(1-(s.k+256)*u);
 const int workers=s.q>=128?4:1;
 auto work=[&](int worker){
  for(I r=worker;r<s.q;r+=workers)for(I h=0;h<s.h;++h){
   if(boundary_subset){
    const I rows[]={0,1,2,63,64,65,255,256,257,4094,4095};
    const I heads[]={0,1,15,16,31};
    if(std::find(std::begin(rows),std::end(rows),r)==std::end(rows)||std::find(std::begin(heads),std::end(heads),h)==std::end(heads))continue;
   }
   std::vector<double> scores(size_t(s.k));double mx=-INFINITY,magmax=0,absmax=0;
   for(I j=0;j<s.k;++j){double z=0,mag=0;for(I d=0;d<128;++d){double p=double(q[at(r,h,d,s.qs)])*k[size_t((h*s.k+j)*128+d)];z+=p;mag+=std::abs(p);}z*=scale;scores[size_t(j)]=z;mx=std::max(mx,z);magmax=std::max(magmax,mag*scale);absmax=std::max(absmax,std::abs(z));}
   double den=0;for(auto& z:scores){z=std::exp(z-mx);den+=z;}
   std::array<double,128> z{};
   for(I j=0;j<s.k;++j)for(I d=0;d<128;++d)z[size_t(d)]+=scores[size_t(j)]*double(v[size_t((h*s.k+j)*128+d)]);
   for(I d=0;d<128;++d){const double value=z[size_t(d)]/den;
    double arithmetic=mv[size_t(h)][size_t(d)]*(2*gd*magmax+64*u*(1+absmax)+3*gs);
    double tolerance=(1+std::ldexp(1.,-11))*arithmetic+std::abs(value)*std::ldexp(1.,-11)+std::ldexp(1.,-25);
    float got=out[size_t((r*s.h+h)*128+d)];require(std::isfinite(got)&&std::isfinite(value)&&std::abs(got-value)<=tolerance,"full independent F64 oracle");
    if(uniform){float wanted=rounded(uniform_sum[size_t(h)][size_t(d)]/float(s.k));require(std::memcmp(&got,&wanted,4)==0,"uniform exact CPU mean");}
   }
  }
 };
 std::vector<std::future<void>> tasks;
 for(int w=0;w<workers;++w)tasks.push_back(std::async(std::launch::async,work,w));
 for(auto& task:tasks)task.get();
}
static void self_test(){
 for(unsigned h=0;h<65536;++h)if((h&0x7c00)!=0x7c00)require(encode(decode(uint16_t(h)))==h,"half roundtrip");
 for(I sq:{1,3,4,5,65})for(I sk:{1,63,64,65,257}){
  std::vector<int> hits(size_t(sq*sk)),outs(size_t(sq*128));
  for(I row=0;row<sq;row+=2){
   for(int t=0;t<256;++t)if(row+t/128<sq){++outs[size_t((row+t/128)*128+t%128)];}
   for(I base=0;base<sk;base+=64){int writes[128][64]={};
    for(int t=0;t<256;++t)for(int z=t;z<8192;z+=256)++writes[z%128][z/128];
    for(auto& line:writes)for(int n:line)require(n==1,"key transpose ownership");
    for(int t=0;t<256;++t)if(t%128<64&&row+t/128<sq&&base+t%128<sk)++hits[size_t((row+t/128)*sk+base+t%128)];
   }
  }
  require(std::all_of(hits.begin(),hits.end(),[](int n){return n==1;}),"score ownership");require(std::all_of(outs.begin(),outs.end(),[](int n){return n==1;}),"PV ownership");
  F sums(256);for(I j=0;j<sk;++j)sums[size_t(j%256)]+=1.f;for(int step=128;step;step/=2)for(int t=0;t<step;++t)sums[t]+=sums[t+step];require(sums[0]==float(sk),"lane reduction");
 }
 Shape s{3,17,2,768,768};H q(size_t(s.q*s.qs),0),k(size_t(s.k*s.ks),0x3c00),v=halves(size_t(s.k*s.ks),29);F out(nout(s));
 for(I r=0;r<s.q;++r)for(I h=0;h<s.h;++h)for(I d=0;d<128;++d){float z=0;for(I j=0;j<s.k;++j)z=std::fma(1.f,decode(v[at(j,h,d,s.ks)]),z);out[size_t((r*s.h+h)*128+d)]=rounded(z/float(s.k));}
 oracle(s,q,k,v,out,true);F wrong=out;wrong[0]+=1;bool rejected=false;try{oracle(s,q,k,v,wrong,true);}catch(const std::exception&){rejected=true;}require(rejected,"oracle negative control");
 H dest(nout(s));require(nrqi_attention_1024::arguments(q.data(),k.data(),v.data(),s.q,s.k,s.h,128,s.qs,s.ks,dest.data()),"valid guard");
 require(!nrqi_attention_1024::arguments(q.data(),k.data(),v.data(),s.q,s.k,s.h,128,s.qs,s.ks,q.data()+1),"alias guard");
 require(!nrqi_attention_1024::supported(INT64_MAX,274,32,128,12288,12288),"overflow cap");require(nrqi_attention_1024::shared_bytes(4114)==52624&&nrqi_attention_1024::shared_bytes(4608)==56576,"LDS cap");
 std::cout<<"CPU PASS half roundtrip,25 tile ownership/tail cases, reduction, full F64/uniform/negative control, guards\n";
}
extern "C" void* nsd_create(void*);
extern "C" void nsd_destroy(void*);
extern "C" int nsd_attention_qk(void*,const void*,I,I,const void*,const void*,I,I,int,I,I,void*,int);
extern "C" int nrqi_image_attention_1024_baseline(void*,const void*,const void*,const void*,I,I,I,I,I,I,void*);
extern "C" int nrqi_image_attention_1024(void*,const void*,const void*,const void*,I,I,I,I,I,I,void*);
extern "C" int nr_allocation_stats(ns_gpu*,uint64_t*,uint64_t*);
#define SYMS(X) X(ns_last_error) X(ns_gpu_open) X(ns_gpu_close) X(ns_alloc) X(ns_free) X(ns_copy_to) X(ns_copy_from) X(ns_sync) X(ns_stamp) X(ns_elapsed) X(nsd_create) X(nsd_destroy) X(nsd_attention_qk) X(nrqi_image_attention_1024) X(nrqi_image_attention_1024_baseline) X(nr_allocation_stats)
struct Api{
 void* lib=nullptr;ns_gpu* gpu=nullptr;void* ctx=nullptr;
#define FIELD(n) decltype(&::n) n=nullptr;
 SYMS(FIELD)
#undef FIELD
 template<class T>void bind(T& f,const char* n){void* p=dlsym(lib,n);require(p!=nullptr,std::string("missing ")+n);static_assert(sizeof(f)==sizeof(p));std::memcpy(&f,&p,sizeof(f));}
 void check(int rc,const char* label){if(rc)throw std::runtime_error(std::string(label)+": "+ns_last_error());}
 Api(const char* path,int index){lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!lib)throw std::runtime_error(dlerror());try{
#define BIND(n) bind(n,#n);
 SYMS(BIND)
#undef BIND
 check(ns_gpu_open(index,&gpu),"open");ctx=nsd_create(gpu);require(ctx!=nullptr,"context");
 }catch(...){if(gpu)ns_gpu_close(gpu);dlclose(lib);throw;}}
 ~Api(){if(gpu){ns_sync(gpu);if(ctx)nsd_destroy(ctx);ns_gpu_close(gpu);}if(lib)dlclose(lib);}
 Api(const Api&)=delete;Api&operator=(const Api&)=delete;
 void sync(){check(ns_sync(gpu),"sync");}
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
static void unchanged(Api&a,const Buffer& b,const H& h){H x(h.size());a.check(a.ns_copy_from(a.gpu,x.data(),b.ptr(),x.size()*2),"readonly input");a.sync();require(x==h,"input or stride padding mutated");b.guards();}
static void gpu_case(Api&a,Shape s,int special,int pairs,bool boundary_subset=false){
 H q=halves(size_t(s.q*s.qs),7),k=halves(size_t(s.k*s.ks),19),v=halves(size_t(s.k*s.ks),29);
 if(special==1)std::fill(q.begin(),q.end(),0);
 if(special==2){const uint16_t e[]={0,0x8000,1,0x8001,0x03ff,0x0400,0x3c00,0xbc00};for(size_t i=0;i<q.size();++i)q[i]=e[i%8];for(size_t i=0;i<k.size();++i){k[i]=e[(i*3+1)%8];v[i]=e[(i*5+2)%8];}}
 if(special==3){std::fill(q.begin(),q.end(),0x3c00);for(I j=0;j<s.k;++j)for(I h=0;h<s.h;++h){for(I d=0;d<128;++d)k[at(j,h,d,s.ks)]=0;k[at(j,h,0,s.ks)]=0x7800;k[at(j,h,1,s.ks)]=(j%2)?0xf800:0x1400;k[at(j,h,2,s.ks)]=(j%2)?0x1400:0xf800;}}
 Buffer dq(a,q.size()*2),dk(a,k.size()*2),dv(a,v.size()*2),base(a,nout(s)*2),out(a,nout(s)*2);
 dq.write(q.data(),q.size()*2);dk.write(k.data(),k.size()*2);dv.write(v.data(),v.size()*2);H poison(nout(s),0x7e00);base.write(poison.data(),poison.size()*2);out.write(poison.data(),poison.size()*2);
 auto old=[&]{a.check(a.nrqi_image_attention_1024_baseline(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),s.q,s.k,s.h,128,s.qs,s.ks,base.ptr()),"actual baseline");};
 auto candidate=[&]{a.check(a.nrqi_image_attention_1024(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),s.q,s.k,s.h,128,s.qs,s.ks,out.ptr()),"candidate");};
 old();candidate();a.sync();const F expected=base.read_half(),actual=out.read_half();exact(actual,expected,"complete baseline bitwise");
 // Bind the extended test wrapper back to the actual shared entrypoint where
 // the old public 4096-key guard permits it.
 if(s.k<=4096){a.check(a.nsd_attention_qk(a.ctx,dq.ptr(),s.q,s.qs,dk.ptr(),dv.ptr(),s.k,s.ks,1,s.h,128,base.ptr(),1),"shared baseline binding");a.sync();exact(base.read_half(),expected,"original shared baseline binding");}
 std::cout<<"NATIVE_BYTES_PASS outputs="<<nout(s)<<" F64_boundary_subset="<<boundary_subset<<"\n"<<std::flush;
 oracle(s,q,k,v,actual,special==1,boundary_subset);
 unchanged(a,dq,q);unchanged(a,dk,k);unchanged(a,dv,v);base.guards();out.guards();
 candidate();a.sync();exact(out.read_half(),expected,"repeat bitwise");
 std::cout<<"PASS shape=["<<s.q<<','<<s.k<<','<<s.h<<",128,"<<s.qs<<','<<s.ks<<"] special="<<special<<" outputs="<<nout(s)<<" full_bitwise=1 full_F64="<<(!boundary_subset)<<" F64_outputs="<<(boundary_subset?7040:nout(s))<<" LDS="<<nrqi_attention_1024::shared_bytes(s.k)<<'\n';
 if(pairs){std::vector<double> bt,ct;auto time=[&](bool use){int64_t st=-1,en=-1;double ns=-1;a.check(a.ns_stamp(a.gpu,&st),"stamp");if(use)candidate();else old();a.check(a.ns_stamp(a.gpu,&en),"stamp");a.check(a.ns_elapsed(a.gpu,st,en,&ns),"elapsed");require(std::isfinite(ns)&&ns>0,"timing finite");return ns/1e6;};
  for(int i=0;i<2;++i){old();candidate();}a.sync();
  for(int i=0;i<pairs;++i){double b,c;if(i%2){c=time(true);b=time(false);}else{b=time(false);c=time(true);}bt.push_back(b);ct.push_back(c);std::cout<<"TIMING pair="<<i<<" baseline_ms="<<b<<" candidate_ms="<<c<<'\n';}
  std::sort(bt.begin(),bt.end());std::sort(ct.begin(),ct.end());std::cout<<"MEDIAN baseline_ms="<<bt[bt.size()/2]<<" candidate_ms="<<ct[ct.size()/2]<<" speedup="<<bt[bt.size()/2]/ct[ct.size()/2]<<'\n';
  exact(out.read_half(),expected,"timed repeat bitwise");exact(base.read_half(),expected,"timed baseline stable");
 }
 unchanged(a,dq,q);unchanged(a,dk,k);unchanged(a,dv,v);base.guards();out.guards();
}
static void invalid(Api&a){
 // Every otherwise-valid declared span fits these independent allocations.
 Buffer q(a,F(524288,1)),k(a,F(524288,1)),v(a,F(524288,1)),out(a,F(524288,sentinel));size_t cases=0;
 auto bad=[&](const void*qp,const void*kp,const void*vp,I sq,I sk,I h,I d,I qs,I ks,void*dst){require(a.nrqi_image_attention_1024(a.gpu,qp,kp,vp,sq,sk,h,d,qs,ks,dst)!=0,"invalid accepted");++cases;};
 for(I x:{0,-1,4097})bad(q.ptr(),k.ptr(),v.ptr(),x,1,1,128,128,128,out.ptr());
 for(I x:{0,-1,4609})bad(q.ptr(),k.ptr(),v.ptr(),1,x,1,128,128,128,out.ptr());
 for(I x:{0,-1,33})bad(q.ptr(),k.ptr(),v.ptr(),1,1,x,128,12288,12288,out.ptr());
 bad(q.ptr(),k.ptr(),v.ptr(),1,1,1,64,128,128,out.ptr());
 for(I x:{127,12289}){bad(q.ptr(),k.ptr(),v.ptr(),2,2,1,128,x,128,out.ptr());bad(q.ptr(),k.ptr(),v.ptr(),2,2,1,128,128,x,out.ptr());}
 bad(nullptr,k.ptr(),v.ptr(),1,1,1,128,128,128,out.ptr());bad(q.ptr(),nullptr,v.ptr(),1,1,1,128,128,128,out.ptr());bad(q.ptr(),k.ptr(),nullptr,1,1,1,128,128,128,out.ptr());bad(q.ptr(),k.ptr(),v.ptr(),1,1,1,128,128,128,nullptr);
 for(void*p:{q.ptr(),k.ptr(),v.ptr()})bad(q.ptr(),k.ptr(),v.ptr(),1,1,1,128,128,128,static_cast<char*>(p)+2);
 bad(static_cast<char*>(q.ptr())+1,k.ptr(),v.ptr(),1,1,1,128,128,128,out.ptr());bad(q.ptr(),static_cast<char*>(k.ptr())+1,v.ptr(),1,1,1,128,128,128,out.ptr());bad(q.ptr(),k.ptr(),static_cast<char*>(v.ptr())+1,1,1,1,128,128,128,out.ptr());bad(q.ptr(),k.ptr(),v.ptr(),1,1,1,128,128,128,static_cast<char*>(out.ptr())+1);
 bad(reinterpret_cast<void*>(UINTPTR_MAX-1),k.ptr(),v.ptr(),1,1,1,128,128,128,out.ptr());bad(q.ptr(),reinterpret_cast<void*>(UINTPTR_MAX-1),v.ptr(),1,1,1,128,128,128,out.ptr());bad(q.ptr(),k.ptr(),reinterpret_cast<void*>(UINTPTR_MAX-1),1,1,1,128,128,128,out.ptr());bad(q.ptr(),k.ptr(),v.ptr(),1,1,1,128,128,128,reinterpret_cast<void*>(UINTPTR_MAX-1));
 require(a.nrqi_image_attention_1024(nullptr,q.ptr(),k.ptr(),v.ptr(),1,1,1,128,128,128,out.ptr())!=0,"nullctx accepted");++cases;
 a.sync();exact(q.read(),F(524288,1),"invalid q");exact(k.read(),F(524288,1),"invalid k");exact(v.read(),F(524288,1),"invalid v");exact(out.read(),F(524288,sentinel),"invalid output");q.guards();k.guards();v.guards();out.guards();std::cout<<cases<<" invalid entrypoints PASS\n";
}
int main(int argc,char**argv){try{
 self_test();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 require(argc==4,"usage: attention_1024_numeric ABS_RUNTIME GPU_INDEX small|full (NS_PROFILE=gpu)");require(argv[1][0]=='/',"absolute runtime");const char*p=std::getenv("NS_PROFILE");require(p&&std::string(p)=="gpu","NS_PROFILE=gpu required");
 size_t used=0;int index=std::stoi(argv[2],&used);require(index>=0&&used==std::strlen(argv[2]),"GPU index");Api a(argv[1],index);uint64_t live=0,peak=0;a.check(a.nr_allocation_stats(a.gpu,&live,&peak),"stats");require(live==0,"initial live");std::cout<<std::setprecision(9);
 const std::string mode=argv[3];require(mode=="small"||mode=="full","mode small|full");
 size_t cases=0;
 if(mode=="small"){
  for(I n:{1,2,3,4,5,15,16,17,63,64,65}){I h=n%3+1;gpu_case(a,{n,274,h,h*384,h*384},0,0);++cases;}
  for(I sk:{1,63,64,65,255,256,257,4095,4096,4097,4114,4608}){gpu_case(a,{3,sk,2,257,513},0,0);++cases;}
  for(int special:{1,2,3}){gpu_case(a,{3,65,2,768,768},special,0);++cases;}
 }else{gpu_case(a,{4096,4114,32,12288,12288},0,7,true);++cases;}
 invalid(a);
 a.check(a.nr_allocation_stats(a.gpu,&live,&peak),"stats");require(live==0&&peak<512*1024*1024,"allocation bound/drop");std::cout<<"tracked_final="<<live<<" tracked_peak="<<peak<<"\nALL IMAGE ATTENTION 1024 PASS cases="<<cases<<" (operator only)\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
