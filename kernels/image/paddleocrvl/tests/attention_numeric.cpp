// Standalone private candidate. CPU: c++ -std=c++17 -O2 FILE -o OUT; OUT --self-test
// GPU: hipcc -std=c++17 -O2 --offload-arch=gfx1035 -x hip FILE -ldl -o OUT
// OUT ORIGINAL_RUNTIME [--profile]; never uses a replacement baseline kernel.
#include "../attention_tiled.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using I=int64_t;using F=std::vector<float>;
static void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
struct Shape{I n,h;I stride()const{return 3*h*72;}size_t input()const{return size_t(n*stride());}size_t output()const{return size_t(n*h*72);}};
[[maybe_unused]] static F values(size_t n,uint32_t seed){F x(n);for(float&v:x){seed=1664525u*seed+1013904223u;v=float(int(seed>>8)%2049-1024)/1024.f;}return x;}
static void exact(const F&a,const F&b){require(a.size()==b.size(),"size differs");for(size_t i=0;i<a.size();++i)require(std::isfinite(a[i])&&std::isfinite(b[i])&&std::memcmp(&a[i],&b[i],4)==0,"nonfinite/bit mismatch at "+std::to_string(i));}
// Independent F64 full dot/softmax/PV. Frozen abs+rel2e-5 each; all channels,
// all queries forN<=65, five edge/central queries for largerN, every head.
static void oracle(Shape s,const F&q,const F&k,const F&v,const F&out){
 std::vector<I> rows;if(s.n<=65){for(I r=0;r<s.n;++r)rows.push_back(r);}else rows={0,1,s.n/2,s.n-2,s.n-1};
 for(I r:rows)for(I h=0;h<s.h;++h){std::vector<double> p(size_t(s.n),0.);double best=-INFINITY;
  for(I j=0;j<s.n;++j){double z=0;for(I a=0;a<72;++a)z+=double(q[r*s.stride()+h*72+a])*double(k[j*s.stride()+h*72+a]);p[j]=z/std::sqrt(72.);best=std::max(best,p[j]);}
  double den=0;for(double&z:p){z=std::exp(z-best);den+=z;}
  for(I a=0;a<72;++a){double z=0;for(I j=0;j<s.n;++j)z+=p[j]*double(v[j*s.stride()+h*72+a]);z/=den;const double actual=out[(r*s.h+h)*72+a];require(std::isfinite(actual)&&std::abs(actual-z)<=2e-5+2e-5*std::abs(z),"independent F64 gate failed");}
 }
}
static void host_tests(){
 using namespace pocr_tiled;
 for(I n:{1,2,3,31,64,65,255,256,257,576,1152,5119,5120,5121,8191,8192}){require(supported(n,16,72,3456),"valid rejected");require(shared_bytes(n)<=65536,"LDS overflow");}
 auto q=reinterpret_cast<const float*>(uintptr_t(0x100000));auto k=reinterpret_cast<const float*>(uintptr_t(0x200000));auto v=reinterpret_cast<const float*>(uintptr_t(0x300000));auto o=reinterpret_cast<float*>(uintptr_t(0x400000));
 require(arguments(q,k,v,3,1,72,216,o),"valid arguments");
 require(arguments(q,q+72,q+144,3,1,72,216,o),"packed read-only QKV overlap rejected");
 for(I n:{-1,0,8193})require(!arguments(q,k,v,n,1,72,216,o),"N accepted");
 for(I h:{-1,0,17})require(!arguments(q,k,v,1,h,72,3*h*72,o),"H accepted");
 for(I d:{-1,0,71,73})require(!arguments(q,k,v,1,1,d,216,o),"D accepted");
 for(I st:{-1,0,72,215,217})require(!arguments(q,k,v,1,1,72,st,o),"stride accepted");
 require(!arguments(nullptr,k,v,1,1,72,216,o)&&!arguments(q,nullptr,v,1,1,72,216,o)&&!arguments(q,k,nullptr,1,1,72,216,o)&&!arguments(q,k,v,1,1,72,216,nullptr),"null accepted");
 for(const float*p:{q,k,v})require(!arguments(q,k,v,3,1,72,216,const_cast<float*>(p)+1),"overlap accepted");
 auto bad=reinterpret_cast<const float*>(uintptr_t(q)+1);require(!arguments(bad,k,v,1,1,72,216,o)&&!arguments(q,bad,v,1,1,72,216,o)&&!arguments(q,k,bad,1,1,72,216,o),"input alignment");
 require(!arguments(q,k,v,1,1,72,216,reinterpret_cast<float*>(uintptr_t(o)+1)),"output alignment");
 auto wrap=reinterpret_cast<const float*>(UINTPTR_MAX-3);require(!arguments(wrap,k,v,1,1,72,216,o)&&!arguments(q,wrap,v,1,1,72,216,o)&&!arguments(q,k,wrap,1,1,72,216,o)&&!arguments(q,k,v,1,1,72,216,const_cast<float*>(wrap)),"wrapped span");
 Shape s{3,2};F zero(s.input(),0),v0(s.input(),.25f),expected(s.output(),.25f);oracle(s,zero,zero,v0,expected);
 bool rejected=false;expected[0]=.5f;try{oracle(s,zero,zero,v0,expected);}catch(const std::exception&){rejected=true;}require(rejected,"oracle fails to reject bad output");
 rejected=false;try{exact(F{0.f},F{-0.f});}catch(const std::exception&){rejected=true;}require(rejected,"bitwise check ignores signed zero");
 std::cout<<"CPU PASS geometry/alignment/overlap/wrap/LDS/oracle-rejection\n";
}
#ifdef __HIPCC__
#include <dlfcn.h>
static void hc(hipError_t e){require(e==hipSuccess,hipGetErrorString(e));}
struct Api{
 void*lib=nullptr;void*g=nullptr;hipStream_t stream=nullptr;
 int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;void*(*queue)(void*)=nullptr;
 int(*baseline)(void*,const float*,const float*,const float*,I,I,I,I,I,I,float*,int)=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));require(f!=nullptr,std::string("missing ")+n);}
 explicit Api(const char*path){lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);require(lib!=nullptr,"runtime dlopen failed");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(queue,"ns_gpu_queue");bind(baseline,"ns_image_pocr_attention");require(open(0,&g)==0,"GPU open");stream=static_cast<hipStream_t>(queue(g));require(stream!=nullptr,"GPU stream");}
 ~Api(){if(g){hipStreamSynchronize(stream);close(g);}if(lib)dlclose(lib);}
};
constexpr size_t guard=16;constexpr float sentinel=-12345.25f;
struct Buffer{
 float*base=nullptr;size_t n;hipStream_t stream;
 Buffer(const F&x,hipStream_t s):n(x.size()),stream(s){hc(hipMalloc(reinterpret_cast<void**>(&base),(n+2*guard)*4));F all(n+2*guard,sentinel);std::copy(x.begin(),x.end(),all.begin()+guard);hc(hipMemcpyAsync(base,all.data(),all.size()*4,hipMemcpyHostToDevice,stream));hc(hipStreamSynchronize(stream));}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 ~Buffer(){if(base){hipStreamSynchronize(stream);hipFree(base);}}
 float*ptr()const{return base+guard;}
 F read()const{F all(n+2*guard);hc(hipMemcpyAsync(all.data(),base,all.size()*4,hipMemcpyDeviceToHost,stream));hc(hipStreamSynchronize(stream));for(size_t i=0;i<guard;++i)require(all[i]==sentinel&&all[n+guard+i]==sentinel,"device guard clobbered");return F(all.begin()+guard,all.end()-guard);}
};
struct Events{hipEvent_t a=nullptr,b=nullptr;Events(){hc(hipEventCreate(&a));hc(hipEventCreate(&b));}~Events(){hipEventDestroy(a);hipEventDestroy(b);}};
static void gpu_case(Api&a,Shape s,bool profile,int mode=0){
 F q=values(s.input(),7),k=values(s.input(),19),v=values(s.input(),29);
 if(mode==1)std::fill(q.begin(),q.end(),0.f);
 if(mode==2)for(size_t i=0;i<q.size();++i){q[i]=(i%2)?0.f:-0.f;k[i]=(i%3)?std::ldexp(1.f,-20):-std::ldexp(1.f,-20);}
 if(mode==3){std::fill(q.begin(),q.end(),1.f);for(I j=0;j<s.n;++j)for(I h=0;h<s.h;++h){for(I c=0;c<72;++c)k[j*s.stride()+h*72+c]=0;k[j*s.stride()+h*72]=float(j%3-1)*64.f;}}
 if(mode==4){std::fill(k.begin(),k.end(),0);std::fill(v.begin(),v.end(),0);std::copy(q.begin()+s.h*72,q.end(),k.begin());std::copy(q.begin()+2*s.h*72,q.end(),v.begin());}
 Buffer dq(q,a.stream),dk(k,a.stream),dv(v,a.stream),base(F(s.output(),NAN),a.stream),out(F(s.output(),NAN),a.stream);Events ev;
 const float*qp=dq.ptr();const float*kp=mode==4?dq.ptr()+s.h*72:dk.ptr();const float*vp=mode==4?dq.ptr()+2*s.h*72:dv.ptr();
 auto run=[&](bool candidate){hc(hipEventRecord(ev.a,a.stream));if(candidate)hc(pocr_tiled::launch(qp,kp,vp,s.n,s.h,72,s.stride(),out.ptr(),a.stream));else require(a.baseline(a.g,qp,kp,vp,s.n,s.h,s.h,72,s.stride(),s.stride(),base.ptr(),0)==0,"original attention failed");hc(hipEventRecord(ev.b,a.stream));hc(hipEventSynchronize(ev.b));float ms=0;hc(hipEventElapsedTime(&ms,ev.a,ev.b));return ms;};
 float b0=run(false),c0=run(true);F original=base.read(),actual=out.read();exact(original,actual);oracle(s,q,k,v,actual);exact(q,dq.read());exact(k,dk.read());exact(v,dv.read());
 std::cout<<"PASS N="<<s.n<<" H="<<s.h<<" D=72 stride="<<s.stride()<<" mode="<<mode<<" bitwise_outputs="<<s.output()<<" F64_queries="<<(s.n<=65?s.n:5)<<" baseline_ms="<<b0<<" candidate_ms="<<c0<<" scratch=0 LDS="<<pocr_tiled::shared_bytes(s.n)<<std::endl;
 if(profile){float c1=run(true),b1=run(false);exact(original,out.read());exact(original,base.read());std::cout<<"PROFILE N="<<s.n<<" H="<<s.h<<" pair0_baseline_ms="<<b0<<" pair0_candidate_ms="<<c0<<" pair1_candidate_ms="<<c1<<" pair1_baseline_ms="<<b1<<" speedup="<<(b0+b1)/(c0+c1)<<std::endl;}
}
static void invalid_gpu(Api&a){
 Buffer q(F(4096,1),a.stream),k(F(4096,2),a.stream),v(F(4096,3),a.stream),out(F(4096,4),a.stream);
 auto reject=[&](const float*qp,const float*kp,const float*vp,I n,I h,I d,I st,float*op){require(pocr_tiled::launch(qp,kp,vp,n,h,d,st,op,a.stream)==hipErrorInvalidValue,"invalid launch accepted");};
 for(I n:{-1,0,8193})reject(q.ptr(),k.ptr(),v.ptr(),n,1,72,216,out.ptr());
 for(I h:{-1,0,17})reject(q.ptr(),k.ptr(),v.ptr(),1,h,72,3*h*72,out.ptr());
 for(I d:{-1,0,71,73})reject(q.ptr(),k.ptr(),v.ptr(),1,1,d,216,out.ptr());
 for(I st:{-1,0,72,215,217})reject(q.ptr(),k.ptr(),v.ptr(),1,1,72,st,out.ptr());
 reject(nullptr,k.ptr(),v.ptr(),1,1,72,216,out.ptr());reject(q.ptr(),nullptr,v.ptr(),1,1,72,216,out.ptr());reject(q.ptr(),k.ptr(),nullptr,1,1,72,216,out.ptr());reject(q.ptr(),k.ptr(),v.ptr(),1,1,72,216,nullptr);
 for(float*p:{q.ptr(),k.ptr(),v.ptr()}){reject(q.ptr(),k.ptr(),v.ptr(),3,1,72,216,p);reject(q.ptr(),k.ptr(),v.ptr(),3,1,72,216,p+1);reject(q.ptr(),k.ptr(),v.ptr(),3,1,72,216,p-1);}
 auto mis=[](float*p){return reinterpret_cast<float*>(reinterpret_cast<uintptr_t>(p)+1);};
 reject(mis(q.ptr()),k.ptr(),v.ptr(),1,1,72,216,out.ptr());reject(q.ptr(),mis(k.ptr()),v.ptr(),1,1,72,216,out.ptr());reject(q.ptr(),k.ptr(),mis(v.ptr()),1,1,72,216,out.ptr());reject(q.ptr(),k.ptr(),v.ptr(),1,1,72,216,mis(out.ptr()));
 auto wrap=reinterpret_cast<float*>(UINTPTR_MAX-3);
 reject(wrap,k.ptr(),v.ptr(),1,1,72,216,out.ptr());reject(q.ptr(),wrap,v.ptr(),1,1,72,216,out.ptr());reject(q.ptr(),k.ptr(),wrap,1,1,72,216,out.ptr());reject(q.ptr(),k.ptr(),v.ptr(),1,1,72,216,wrap);
 hc(hipStreamSynchronize(a.stream));exact(q.read(),F(4096,1));exact(k.read(),F(4096,2));exact(v.read(),F(4096,3));exact(out.read(),F(4096,4));std::cout<<"PASS invalid launch guard matrix; all buffers unchanged\n";
}
#endif
int main(int argc,char**argv){try{host_tests();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
#ifdef __HIPCC__
 require(argc==2||argc==3,"usage: validator ORIGINAL_RUNTIME [--profile|--supplement]");bool profile=argc==3&&std::string(argv[2])=="--profile";bool supplement=argc==3&&std::string(argv[2])=="--supplement";require(argc==2||profile||supplement,"invalid option");hc(hipSetDevice(0));Api a(argv[1]);invalid_gpu(a);
 if(supplement){gpu_case(a,{65,2},false,4);std::cout<<"SUPPLEMENT PASS isolated H17/all invalid guards and packed QKV only; no full rerun/admission\n";return 0;}
 for(I n:{1,2,3,15,16,17,31,32,33,63,64,65,127,128,129,255,256,257})gpu_case(a,{n,n%3+1},false);
 for(int mode:{1,2,3,4})gpu_case(a,{65,2},false,mode);
 for(I n:{576,1152,5119,5120,5121,8191,8192})gpu_case(a,{n,1},false);
 gpu_case(a,{5120,16},profile);std::cout<<"ALL PASS private candidate; no production admission\n";return 0;
#else
 throw std::runtime_error("GPU execution needs HIP build");
#endif
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
