// Parent-only standalone. CPU: c++ -std=c++17 -O2 -Wall -Wextra -Werror FILE -ldl -o OUT
// OUT --self-test; GPU: OUT PRIVATE_DSO ACTUAL_NATIVE_QKV_ROPE.bin
#include "../qv.h"
#include "../attention_divide.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using I=int64_t;using F=std::vector<float>;
static void need(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
static std::vector<double> oracle(const F&qkv,I rows){
 std::vector<double> out(size_t(rows)*4096);
 for(I r=0;r<rows;++r)for(I h=0;h<32;++h){std::vector<double> p(size_t(r+1),0.);double best=-INFINITY;
  for(I j=0;j<=r;++j){double z=0;for(I d=0;d<128;++d){z+=double(qkv[r*6144+h*128+d])*qkv[j*6144+4096+(h/4)*128+d];}p[j]=z/std::sqrt(128.);best=std::max(best,p[j]);}
  double den=0;for(double&v:p){v=std::exp(v-best);den+=v;}for(double&v:p){v/=den;}
  for(I d=0;d<128;++d){double v=0;for(I j=0;j<=r;++j){v+=p[j]*qkv[j*6144+5120+(h/4)*128+d];}out[r*4096+h*128+d]=v;}
 }return out;
}
static double check(const F&x,const std::vector<double>&ref){need(x.size()==ref.size(),"shape");double worst=0;for(size_t i=0;i<x.size();++i){double d=std::abs(double(x[i])-ref[i]);need(std::isfinite(x[i])&&std::isfinite(ref[i])&&d<=2e-5+2e-5*std::abs(ref[i]),"fixed F64 gate");worst=std::max(worst,d);}return worst;}
struct Api{
 void*lib=nullptr;void*g=nullptr;int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;
 int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqv_attention_divide_f32) candidate=nullptr;
 int(*baseline)(void*,const void*,const void*,const void*,int,I,I,I,I,I,I,I,void*)=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));need(f!=nullptr,n);}
 explicit Api(const char*p){lib=dlopen(p,RTLD_NOW|RTLD_LOCAL);need(lib!=nullptr,"dlopen");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(candidate,"nrqv_attention_divide_f32");bind(baseline,"nsd_attention_causal_rows");need(open(0,&g)==0,"open");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=16;
 Buffer(Api&api,const F&v):a(api),n(v.size()){F all(n+2*guard,-9999.25f);std::copy(v.begin(),v.end(),all.begin()+guard);need(a.alloc(a.g,all.size()*4,&base)==0,"alloc");need(a.to(a.g,base,all.data(),all.size()*4)==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 float*ptr()const{return static_cast<float*>(base)+guard;}
 F read(){F all(n+2*guard);need(a.from(a.g,all.data(),base,all.size()*4)==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i){need(all[i]==-9999.25f&&all[n+guard+i]==-9999.25f,"guards");}return F(all.begin()+guard,all.end()-guard);}
};
static void host(){
 void*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<float*>(uintptr_t(0x10000000));auto*y=reinterpret_cast<float*>(uintptr_t(0x20000000));
 for(I rows:{1,32,65,4096})need(qv_divide::valid(g,x,rows,y),"valid");
 for(I rows:{-1,0,4097})need(!qv_divide::valid(g,x,rows,y),"rows guard");
 need(!qv_divide::valid(nullptr,x,1,y)&&!qv_divide::valid(g,nullptr,1,y)&&!qv_divide::valid(g,x,1,nullptr),"null");
 need(!qv_divide::valid(g,x,1,x)&&!qv_divide::valid(g,x,1,x+1),"alias");
 need(!qv_divide::valid(g,reinterpret_cast<float*>(uintptr_t(x)+1),1,y),"alignment");
 need(!qv_divide::valid(g,reinterpret_cast<float*>(UINTPTR_MAX-3),1,y),"wrap");
 bool reject=false;try{check(F{1},std::vector<double>{0});}catch(const std::exception&){reject=true;}need(reject,"oracle negativecontrol");
 std::cout<<"CPU PASS geometry/span guards and independent oracle rejection\n";
}
static void run(Api&a,const F&qkv,I rows,bool uniform,const std::string&name,const F*expected=nullptr){
 Buffer x(a,qkv),y(a,F(size_t(rows)*4096,NAN)),b(a,F(size_t(rows)*4096,NAN));
 need(a.baseline(a.g,x.ptr(),x.ptr()+4096,x.ptr()+5120,0,0,rows,32,8,128,6144,6144,b.ptr())==0,"baseline");
 need(a.candidate(a.g,x.ptr(),rows,y.ptr())==0,"candidate");
 F actual=y.read(),baseline=b.read();auto reference=oracle(qkv,rows);double e=check(actual,reference),be=check(baseline,reference);
 if(uniform){for(I r=0;r<rows;++r)for(I h=0;h<32;++h)for(I d=0;d<128;++d){float v=0;const float p=1.f/float(r+1);for(I j=0;j<=r;++j){v=std::fma(p,qkv[j*6144+5120+(h/4)*128+d],v);}need(std::memcmp(&v,&actual[r*4096+h*128+d],4)==0,"uniform normalized-P F32 order differs");}}
 if(expected)need(actual.size()==expected->size()&&std::memcmp(actual.data(),expected->data(),actual.size()*4)==0,"actual diagnostic GPU division bits differ");
 F unchanged=x.read();need(std::memcmp(unchanged.data(),qkv.data(),qkv.size()*4)==0,"input mutated");
 std::cout<<"PASS "<<name<<" rows="<<rows<<" values="<<actual.size()<<" candidate_max_abs="<<e<<" baseline_max_abs="<<be<<" uniform_order="<<uniform<<'\n';
}
static void invalid(Api&a){F input(6144,1),output(4096,2);Buffer x(a,input),y(a,output);
 for(I rows:{-1,0,4097})need(a.candidate(a.g,x.ptr(),rows,y.ptr())!=0,"invalid rows launched");
 need(a.candidate(nullptr,x.ptr(),1,y.ptr())!=0&&a.candidate(a.g,nullptr,1,y.ptr())!=0&&a.candidate(a.g,x.ptr(),1,nullptr)!=0,"null launched");
 need(a.candidate(a.g,x.ptr(),1,x.ptr())!=0&&a.candidate(a.g,x.ptr(),1,x.ptr()+1)!=0,"alias launched");
 auto*mis=reinterpret_cast<float*>(uintptr_t(x.ptr())+1);need(a.candidate(a.g,mis,1,y.ptr())!=0,"unaligned launched");
 need(x.read()==input&&y.read()==output,"invalid calls changed buffers");
}
int main(int argc,char**argv){try{host();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 need(argc==3,"usage: attention_divide_numeric DSO PINNED_FIXTURES");Api a(argv[1]);invalid(a);
 for(I rows:{1,3,31,32,33,65,257})for(bool uniform:{false,true}){F qkv(size_t(rows)*6144);for(size_t i=0;i<qkv.size();++i){qkv[i]=float(int((i*73+11)%513)-256)/512.f;if(uniform&&i%6144<5120)qkv[i]=0.;}run(a,qkv,rows,uniform,uniform?"uniform":"synthetic");}
 const auto read=[](const std::string&p,size_t n){F v(n);std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()==std::streamoff(n*4),"actual length");f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*4));need(bool(f),"actual read");for(float x:v)need(std::isfinite(x),"actual finite");return v;};
 for(const std::string name:{"layer0_native","layer0_primary","block1_strict","block1_input_rms_strict"}){const std::string p=std::string(argv[2])+"/"+name;F q=read(p+".qkv.f32",32*6144),y=read(p+".expected.f32",32*4096);run(a,q,32,false,name,&y);}
 std::cout<<"GPU PASS18 division attention cases; four actual diagnostic outputs bitwise; no model admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
