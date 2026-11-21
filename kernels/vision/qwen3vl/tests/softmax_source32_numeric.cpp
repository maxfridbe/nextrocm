// Standalone CPU guard selftest and parent-only HIP operator validator.
#include "../softmax_source32.h"
#include "../softmax_source32.hpp"
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
static bool bits(const F&a,const F&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0;}
struct Api{
 void*lib=nullptr;void*g=nullptr;int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;
 int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqv_softmax_source32) candidate=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));need(f!=nullptr,n);}
 explicit Api(const char*p){lib=dlopen(p,RTLD_NOW|RTLD_LOCAL);need(lib!=nullptr,"dlopen");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(candidate,"nrqv_softmax_source32");need(open(0,&g)==0,"open");}
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
 void*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<float*>(uintptr_t(0x10000000));auto*y=reinterpret_cast<float*>(uintptr_t(0x20000000));auto*s=reinterpret_cast<uint32_t*>(uintptr_t(0x30000000));
 for(I n:{1,7,32,499,1024,4096})need(qv_softmax_source32::valid(g,x,n,32,y,s),"valid geometry");
 for(I n:{-1,0,4097})need(!qv_softmax_source32::valid(g,x,n,32,y,s),"row bound");
 for(I w:{-1,0,1,31,33,64})need(!qv_softmax_source32::valid(g,x,1,w,y,s),"width guard");
 need(!qv_softmax_source32::valid(nullptr,x,1,32,y,s)&&!qv_softmax_source32::valid(g,nullptr,1,32,y,s)&&!qv_softmax_source32::valid(g,x,1,32,nullptr,s)&&!qv_softmax_source32::valid(g,x,1,32,y,nullptr),"null guards");
 need(!qv_softmax_source32::valid(g,x,1,32,x,s)&&!qv_softmax_source32::valid(g,x,1,32,x+1,s),"input output overlap");
 need(!qv_softmax_source32::valid(g,x,1,32,y,reinterpret_cast<uint32_t*>(x+31))&&!qv_softmax_source32::valid(g,x,1,32,y,reinterpret_cast<uint32_t*>(y+98)),"status overlap");
 for(uintptr_t p:{uintptr_t(0x10000001),UINTPTR_MAX-3}){
  need(!qv_softmax_source32::valid(g,reinterpret_cast<float*>(p),1,32,y,s),"input alignment/wrap");
  need(!qv_softmax_source32::valid(g,x,1,32,reinterpret_cast<float*>(p),s),"output alignment/wrap");
 }
 need(!qv_softmax_source32::valid(g,x,1,32,y,reinterpret_cast<uint32_t*>(uintptr_t(0x30000001))),"status alignment");
 need(!qv_softmax_source32::valid(g,x,1,32,y,reinterpret_cast<uint32_t*>(UINTPTR_MAX-3)),"status wrap");
 std::cout<<"CPU PASS geometry/spans/overlap/wrap guards\n";
}
static double check64(const F&x,const F&y){
 const size_t rows=x.size()/32;need(y.size()==rows*99,"record shape");double worst=0;
 for(size_t r=0;r<rows;++r){double maximum=-INFINITY;bool finite=false;
  for(unsigned j=0;j<32;++j){float v=x[r*32+j];need(!std::isnan(v)&&v!=INFINITY,"oracle input domain");finite|=std::isfinite(v);maximum=std::max(maximum,double(v));}
  need(finite,"oracle all masked");double den=0;for(unsigned j=0;j<32;++j)den+=std::exp(double(x[r*32+j])-maximum);
  for(unsigned j=0;j<99;++j){const float v=y[r*99+j];need(std::isfinite(v)||((j>=1&&j<=32)&&v==-INFINITY),"record finite/domain");}
  need(y[r*99+65]>0&&y[r*99+66]>0,"sum/inv positive");
  for(unsigned j=0;j<32;++j){double ref=std::exp(double(x[r*32+j])-maximum)/den,actual=y[r*99+67+j];double delta=std::abs(actual-ref);need(std::isfinite(ref)&&actual>=0&&delta<=1e-6+1e-6*std::abs(ref),"independent fixed F64 gate");worst=std::max(worst,delta);}
 }return worst;
}
static F read(const std::string&p,size_t n){F v(n);std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()==std::streamoff(n*4),"fixture length: "+p);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),std::streamsize(n*4));need(bool(f),"fixture read");return v;}
static void run(Api&a,const F&x,const F*expected,const std::string&name){
 const I rows=I(x.size()/32);Buffer input(a,x),output(a,F(size_t(rows)*99,-777.f)),status(a,F{777.f});
 need(a.candidate(a.g,input.ptr(),rows,32,output.ptr(),reinterpret_cast<uint32_t*>(status.ptr()))==0,"launch");
 F y=output.read(),st=status.read();uint32_t code;std::memcpy(&code,st.data(),4);need(code==0,"numeric status");
 const double error=check64(x,y);if(expected)need(bits(y,*expected),"all CPU intermediate bits differ: "+name);
 need(bits(input.read(),x),"input changed");
 std::cout<<"PASS "<<name<<" rows="<<rows<<" all_intermediate_bits="<<(expected!=nullptr)<<" F64_max="<<error<<'\n';
}
static void invalid(Api&a){
 F xv(32,1),yv(99,2),sv{777};Buffer x(a,xv),y(a,yv),s(a,sv);auto*sp=reinterpret_cast<uint32_t*>(s.ptr());
 auto call=[&](void*c,const float*i,I n,I w,float*o,uint32_t*st){need(a.candidate(c,i,n,w,o,st)!=0,"invalid call accepted");};
 for(I n:{-1,0,4097})call(a.g,x.ptr(),n,32,y.ptr(),sp);
 for(I w:{-1,0,1,31,33,64})call(a.g,x.ptr(),1,w,y.ptr(),sp);
 call(nullptr,x.ptr(),1,32,y.ptr(),sp);call(a.g,nullptr,1,32,y.ptr(),sp);call(a.g,x.ptr(),1,32,nullptr,sp);call(a.g,x.ptr(),1,32,y.ptr(),nullptr);
 call(a.g,x.ptr(),1,32,x.ptr(),sp);call(a.g,x.ptr(),1,32,x.ptr()+1,sp);call(a.g,x.ptr(),1,32,y.ptr(),reinterpret_cast<uint32_t*>(x.ptr()+31));call(a.g,x.ptr(),1,32,y.ptr(),reinterpret_cast<uint32_t*>(y.ptr()+98));
 call(a.g,reinterpret_cast<float*>(uintptr_t(x.ptr())+1),1,32,y.ptr(),sp);call(a.g,x.ptr(),1,32,reinterpret_cast<float*>(uintptr_t(y.ptr())+1),sp);call(a.g,x.ptr(),1,32,y.ptr(),reinterpret_cast<uint32_t*>(uintptr_t(sp)+1));
 call(a.g,reinterpret_cast<float*>(UINTPTR_MAX-3),1,32,y.ptr(),sp);call(a.g,x.ptr(),1,32,reinterpret_cast<float*>(UINTPTR_MAX-3),sp);call(a.g,x.ptr(),1,32,y.ptr(),reinterpret_cast<uint32_t*>(UINTPTR_MAX-3));
 need(bits(x.read(),xv)&&bits(y.read(),yv)&&bits(s.read(),sv),"invalid call wrote buffers/status");
}
static void domain(Api&a){
 F xv(4*32,0),poison(4*99,-777.f);xv[0]=NAN;xv[32]=INFINITY;std::fill(xv.begin()+64,xv.begin()+96,-INFINITY);
 Buffer x(a,xv),y(a,poison),s(a,F{777});auto*sp=reinterpret_cast<uint32_t*>(s.ptr());
 need(a.candidate(a.g,x.ptr(),4,32,y.ptr(),sp)==0,"domain launch");F actual=y.read(),st=s.read();uint32_t code;std::memcpy(&code,st.data(),4);need(code==1,"invalid row status");
 need(std::equal(actual.begin(),actual.begin()+3*99,poison.begin()),"invalid rows written");need(bits(x.read(),xv),"invalid input changed");
 for(unsigned j=0;j<32;++j)need(actual[3*99+67+j]==1.f/32,"valid row lost");
 F valid(4*32,0);need(a.to(a.g,x.ptr(),valid.data(),valid.size()*4)==0,"recovery upload");
 need(a.candidate(a.g,x.ptr(),4,32,y.ptr(),sp)==0,"recovery launch");actual=y.read();st=s.read();std::memcpy(&code,st.data(),4);need(code==0,"status not reset on recovery");check64(valid,actual);need(bits(x.read(),valid),"recovery input changed");
 std::cout<<"PASS invalid NaN/+Inf/allmasked no-write rows + recovery/status reset\n";
}
int main(int argc,char**argv){try{
 host();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 need(argc==3,"usage: softmax_source32_numeric PRIVATE_DSO FIXTURE_DIR");Api a(argv[1]);invalid(a);domain(a);
 for(I n:{1,4096}){F x(size_t(n)*32,0);F expected(size_t(n)*99);for(I r=0;r<n;++r){expected[r*99]=0;for(unsigned j=0;j<32;++j){expected[r*99+1+j]=0;expected[r*99+33+j]=1;expected[r*99+67+j]=1.f/32;}expected[r*99+65]=32;expected[r*99+66]=1.f/32;}run(a,x,&expected,"uniform-boundary");}
 const std::string root=argv[2];std::ifstream list(root+"/cases.txt");need(bool(list),"cases list");std::string name;I rows;unsigned count=0;
 while(list>>name>>rows){need(name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos&&rows>=1&&rows<=4096,"case descriptor");F x=read(root+"/"+name+".input.f32",size_t(rows)*32),ref=read(root+"/"+name+".expected.f32",size_t(rows)*99);run(a,x,&ref,name);++count;}
 need(list.eof()&&count==14,"complete14 corpus cases required");std::cout<<"GPU PASS source S32 softmax14 pinnedCPU cases+2rowbounds/domain/guards; no model admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
