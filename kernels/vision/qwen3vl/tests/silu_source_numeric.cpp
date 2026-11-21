// Standalone CPU guard selftest and parent-only HIP operator validator.
#include "../silu_source.h"
#include "../silu_source.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <cfenv>
#include <xmmintrin.h>
#include <set>
#include "../../../../reference/qwenimage21-turbo/silu_source_scalar.hpp"
#include <stdexcept>
#include <string>
#include <vector>
using I=int64_t;using F=std::vector<float>;
static void need(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
static bool bits(const F&a,const F&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0;}
struct Api{
 void*lib=nullptr;void*g=nullptr;int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;
 int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqv_silu_source_f32) candidate=nullptr;
 decltype(&nrqv_silu_source_capture) capture=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));need(f!=nullptr,n);}
 explicit Api(const char*p){lib=dlopen(p,RTLD_NOW|RTLD_LOCAL);need(lib!=nullptr,"dlopen");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(candidate,"nrqv_silu_source_f32");bind(capture,"nrqv_silu_source_capture");need(open(0,&g)==0,"open");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=16;
 Buffer(Api&api,const F&v):a(api),n(v.size()){F all(n+2*guard,-9999.25f);std::copy(v.begin(),v.end(),all.begin()+guard);need(a.alloc(a.g,all.size()*4,&base)==0,"alloc");need(a.to(a.g,base,all.data(),all.size()*4)==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 float*ptr()const{return static_cast<float*>(base)+guard;}
 F read(){F all(n+2*guard);need(a.from(a.g,all.data(),base,all.size()*4)==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i){need(all[i]==-9999.25f&&all[n+guard+i]==-9999.25f,"guards");}return F(all.begin()+guard,all.end()-guard);}
};
static bool numeric(const F&a,const F&b){if(a.size()!=b.size())return false;for(size_t i=0;i<a.size();++i)if(!(std::isnan(a[i])&&std::isnan(b[i]))&&std::memcmp(&a[i],&b[i],4))return false;return true;}
static float rn64(double x){
 if(std::isnan(x))return NAN;
 constexpr double midpoint=0x1.ffffffp127;
 if(x>=midpoint)return INFINITY;
 if(x<=-midpoint)return -INFINITY;
 const float maximum=std::numeric_limits<float>::max();
 if(x>double(maximum))return maximum;
 if(x< -double(maximum))return -maximum;
 return float(x);
}
static void host(){
 need(std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&((3u<<13)|(1u<<15)|(1u<<6)))==0,"CPU RNE/noFTZ/noDAZ required");
 constexpr double midpoint=0x1.ffffffp127;
 need(rn64(std::nextafter(midpoint,0.))==std::numeric_limits<float>::max()&&rn64(midpoint)==INFINITY&&rn64(std::nextafter(midpoint,INFINITY))==INFINITY,"positive RN overflow midpoint");
 need(rn64(std::nextafter(-midpoint,0.))==-std::numeric_limits<float>::max()&&rn64(-midpoint)==-INFINITY&&rn64(std::nextafter(-midpoint,-INFINITY))==-INFINITY,"negative RN overflow midpoint");
 auto*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<float*>(uintptr_t(0x10000000));auto*y=reinterpret_cast<float*>(uintptr_t(0x20000000));auto*z=reinterpret_cast<float*>(uintptr_t(0x30000000));
 for(I c:{1,7,8,9,255,256,257,12288})need(qv_silu_source::valid(g,x,1,c,y,z),"positive geometry");
 need(qv_silu_source::valid(g,x,4096,1024,y,z),"maximum count");
 for(I r:{-1,0,4097})need(!qv_silu_source::valid(g,x,r,8,y,z),"rows");
 for(I c:{-1,0,12289})need(!qv_silu_source::valid(g,x,1,c,y,z),"cols");
 need(!qv_silu_source::valid(g,x,4096,1025,y,z),"count cap");
 need(!qv_silu_source::valid(nullptr,x,1,8,y,z)&&!qv_silu_source::valid(g,nullptr,1,8,y,z)&&!qv_silu_source::valid(g,x,1,8,nullptr,z),"null");
 need(qv_silu_source::valid(g,x,1,8,y,nullptr),"optional records");
 need(!qv_silu_source::valid(g,x,1,8,x,z)&&!qv_silu_source::valid(g,x,1,8,x+1,z)&&!qv_silu_source::valid(g,x,1,8,y,x+15)&&!qv_silu_source::valid(g,x,1,8,y,y+7),"alias");
 for(uintptr_t p:{uintptr_t(0x40000001),UINTPTR_MAX-3}){need(!qv_silu_source::valid(g,reinterpret_cast<float*>(p),1,8,y,z),"x span");need(!qv_silu_source::valid(g,x,1,8,reinterpret_cast<float*>(p),z),"y span");need(!qv_silu_source::valid(g,x,1,8,y,reinterpret_cast<float*>(p)),"z span");}
 std::cout<<"CPU PASS geometry/aliases/wrap; no numeric GPU claim\n";
}

static double f64(const F&x,const F&y,I rows,I cols){
 double worst=0;size_t finite=0;
 for(I r=0;r<rows;++r)for(I c=0;c<cols;++c){float g=x[size_t(r)*2*cols+c],u=x[size_t(r)*2*cols+cols+c];
  // Independent double exp/div/mul, with declared F32 boundaries retained.
  // Ideal real SiLU is not equivalent when its F32 exponential overflows.
  float ex=rn64(std::exp(-double(g))),den=rn64(1.+double(ex)),si=rn64(double(g)/double(den)),ref=rn64(double(si)*double(u));float got=y[size_t(r)*cols+c];
  if(std::isfinite(ref)&&std::isfinite(got)){double err=std::abs(double(ref)-got);need(err<=2e-5+2e-5*std::abs(double(ref)),"independent rounded-boundary F64 gate");worst=std::max(worst,err);++finite;}
  else need((std::isnan(ref)&&std::isnan(got))||(std::isinf(ref)&&ref==got),"F64 classification");
 }
 need(finite>0,"F64 finite scope empty");return worst;
}
static F read(const std::string&p,size_t n){F v(n);std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()==std::streamoff(n*4),"fixture size");f.seekg(0);need(bool(f.read(reinterpret_cast<char*>(v.data()),n*4)),"fixture read");return v;}
static void run(Api&a,const F&x,I rows,I cols,const F&expected,const std::string&name){
 size_t n=size_t(rows)*size_t(cols);need(x.size()==n*2&&expected.size()==n*4,"case shape");Buffer input(a,x),output(a,F(n,-777)),records(a,F(n*4,-777)),primary(a,F(n,-777));
 need(a.capture(a.g,input.ptr(),rows,cols,output.ptr(),records.ptr())==0,"capture launch");F out=output.read(),rec=records.read();need(numeric(rec,expected),"CPU four-intermediate bits/category mismatch: "+name);
 F product(n);for(size_t i=0;i<n;++i)product[i]=expected[i*4+3];need(numeric(out,product),"captured product");
 need(a.candidate(a.g,input.ptr(),rows,cols,primary.ptr())==0,"primary launch");need(numeric(primary.read(),out),"primary/capture mismatch");need(bits(input.read(),x),"input changed");double err=f64(x,out,rows,cols);
 std::cout<<"PASS "<<name<<" rows="<<rows<<" cols="<<cols<<" all4intermediate_bits_exceptNaNpayload F64rounded_max="<<err<<'\n';
}
static void invalid(Api&a){
 F xv(16,1),yv(8,2),zv(32,3);Buffer x(a,xv),y(a,yv),z(a,zv);
 auto call=[&](void*c,const float*i,I r,I n,float*o,float*rec){need(a.capture(c,i,r,n,o,rec)!=0,"invalid capture accepted");need(a.candidate(c,i,r,n,o)!=0,"invalid primary accepted");};
 // Primary-specific and capture-specific guards tested separately below.
 for(I r:{-1,0,4097})call(a.g,x.ptr(),r,8,y.ptr(),z.ptr());
 for(I c:{-1,0,12289})call(a.g,x.ptr(),1,c,y.ptr(),z.ptr());
 call(a.g,x.ptr(),4096,1025,y.ptr(),z.ptr());
 call(nullptr,x.ptr(),1,8,y.ptr(),z.ptr());call(a.g,nullptr,1,8,y.ptr(),z.ptr());call(a.g,x.ptr(),1,8,nullptr,z.ptr());
 call(a.g,x.ptr(),1,8,x.ptr(),z.ptr());call(a.g,x.ptr(),1,8,x.ptr()+1,z.ptr());
 for(uintptr_t p:{uintptr_t(x.ptr())+1,UINTPTR_MAX-3}){call(a.g,reinterpret_cast<float*>(p),1,8,y.ptr(),z.ptr());call(a.g,x.ptr(),1,8,reinterpret_cast<float*>(p),z.ptr());}
 for(float*p:{static_cast<float*>(nullptr),x.ptr()+15,y.ptr()+7,reinterpret_cast<float*>(uintptr_t(z.ptr())+1),reinterpret_cast<float*>(UINTPTR_MAX-3)})need(a.capture(a.g,x.ptr(),1,8,y.ptr(),p)!=0,"invalid record accepted");
 need(bits(x.read(),xv)&&bits(y.read(),yv)&&bits(z.read(),zv),"invalid call wrote");std::cout<<"PASS invalid host arguments no-write\n";
}
int main(int argc,char**argv){try{host();if(argc==2&&std::string(argv[1])=="--self-test")return 0;need(argc==3,"usage: silu_source_numeric PRIVATE_DSO FIXTURE_DIR");Api a(argv[1]);invalid(a);
 for(I cols:{1,7,8,9,255,256,257,12288}){I rows=3;F x(size_t(rows)*2*cols),ref(size_t(rows)*cols*4);for(size_t i=0;i<x.size();++i)x[i]=(float(int(i%127)-63))*.03125f;for(I r=0;r<rows;++r)for(I c=0;c<cols;++c){auto v=source_silu::value(x[size_t(r)*2*cols+c],x[size_t(r)*2*cols+cols+c]);size_t i=(size_t(r)*cols+c)*4;ref[i]=v.ex;ref[i+1]=v.denominator;ref[i+2]=v.silu;ref[i+3]=v.product;}run(a,x,rows,cols,ref,"synthetic-tail");}
 std::string root=argv[2],name;I rows,cols;std::ifstream list(root+"/cases.txt");need(bool(list),"case list");std::set<std::string> seen;
 while(list>>name>>rows>>cols){need(name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos&&seen.insert(name).second&&rows>=1&&rows<=4096&&cols>=1&&cols<=12288&&rows*cols<=4194304,"descriptor");run(a,read(root+"/"+name+".input.f32",size_t(rows)*cols*2),rows,cols,read(root+"/"+name+".expected.f32",size_t(rows)*cols*4),name);}
 need(list.eof()&&seen.size()==14,"all14cases required");std::cout<<"GPU PASS source SiLU13actual+edges+8tails/guards; no model claim\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
