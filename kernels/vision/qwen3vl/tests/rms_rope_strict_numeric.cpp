// CPU: c++ -std=c++17 -O2 -Wall -Wextra -Werror FILE -ldl -o OUT
// OUT --self-test; parent-only OUT PRIVATE_RUNTIME.so (GPU0).
#include "../qv.h"
#include "../rms_rope_ordered_strict.hpp"
#include <fstream>
#include <limits>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using I=int64_t;using F=std::vector<float>;
static void req(bool ok,const std::string&s){if(!ok)throw std::runtime_error(s);}
static F values(size_t n){F v(n);for(size_t i=0;i<n;++i)v[i]=float(int((i*73+11)%513)-256)/128.f;return v;}
static std::vector<double> oracle(const F&x,I rows,I heads,I stride,I offset,const F&w,const F&cs){
 std::vector<double> y(x.begin(),x.end());
 for(I r=0;r<rows;++r)for(I h=0;h<heads;++h){I base=offset+r*stride+h*128;double sum=0;
  for(I i=0;i<128;++i){sum+=double(x[base+i])*x[base+i];}
  double inv=1/std::sqrt(sum/128+double(1e-6f));
  for(I j=0;j<64;++j){double a=double(x[base+j])*inv*w[j],b=double(x[base+j+64])*inv*w[j+64],co=cs[r*128+j*2],si=cs[r*128+j*2+1];y[base+j]=a*co-b*si;y[base+j+64]=b*co+a*si;}
 }return y;
}
static double compare(const F&x,const std::vector<double>&y){req(x.size()==y.size(),"shape");double worst=0;for(size_t i=0;i<x.size();++i){double d=std::abs(double(x[i])-y[i]);req(std::isfinite(x[i])&&std::isfinite(y[i])&&d<=2e-5+2e-5*std::abs(y[i]),"fixed F64 absolute+relative gate at "+std::to_string(i));worst=std::max(worst,d);}return worst;}
struct Api{
 void*lib=nullptr;void*g=nullptr;
 int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;
 int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;
 int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqv_rms_rope_ordered_f32) rope=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));req(f!=nullptr,n);}
 explicit Api(const char*p){lib=dlopen(p,RTLD_NOW|RTLD_LOCAL);req(lib!=nullptr,"dlopen");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(rope,"nrqv_rms_rope_ordered_f32");req(open(0,&g)==0,"open GPU");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=16;
 Buffer(Api&api,const F&v):a(api),n(v.size()){F all(n+2*guard,-9999.25f);std::copy(v.begin(),v.end(),all.begin()+guard);req(a.alloc(a.g,all.size()*4,&base)==0,"alloc");req(a.to(a.g,base,all.data(),all.size()*4)==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
 ~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 float*ptr()const{return static_cast<float*>(base)+guard;}
 F read(){F all(n+2*guard);req(a.from(a.g,all.data(),base,all.size()*4)==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i)req(all[i]==-9999.25f&&all[n+guard+i]==-9999.25f,"allocation guards");return F(all.begin()+guard,all.end()-guard);}
};
static void run_case(Api&a,I rows,I heads,I stride,I offset,bool zero){
 F x=values(size_t(rows*stride)),w=values(128),cs(size_t(rows)*128);
 if(zero)std::fill(x.begin(),x.end(),0.f);
 for(float&v:w)v+=2.f;
 for(I r=0;r<rows;++r)for(I j=0;j<64;++j){double angle=double(r)*.137+double(j)*.019;cs[r*128+2*j]=float(std::cos(angle));cs[r*128+2*j+1]=float(std::sin(angle));}
 auto expected=oracle(x,rows,heads,stride,offset,w,cs);Buffer dx(a,x),dw(a,w),dc(a,cs);
 req(a.rope(a.g,dx.ptr()+offset,rows,heads,stride,dw.ptr(),dc.ptr())==0,"launch");F actual=dx.read();
 std::vector<bool> touched(x.size(),false);for(I r=0;r<rows;++r)for(I i=0;i<heads*128;++i)touched[offset+r*stride+i]=true;
 for(size_t i=0;i<x.size();++i)if(!touched[i])req(std::memcmp(&x[i],&actual[i],4)==0,"untouched QKV lane");
 req(dw.read()==w&&dc.read()==cs,"readonly metadata");double error=compare(actual,expected);
 std::cout<<"PASS F32 RMSRoPE rows="<<rows<<" heads="<<heads<<" stride="<<stride<<" offset="<<offset<<" zero="<<zero<<" max_abs="<<error<<'\n';
}
static void invalid(Api&a){F x(6144,1),w(128,1),cs(128,0);Buffer dx(a,x),dw(a,w),dc(a,cs);
 req(a.rope(nullptr,dx.ptr(),1,32,6144,dw.ptr(),dc.ptr())!=0,"null context accepted");
 auto call=[&](float*p,I m,I h,I stride,const float*wp,const float*cp){req(a.rope(a.g,p,m,h,stride,wp,cp)!=0,"invalid accepted");};
 for(I m:{-1,0,4097})call(dx.ptr(),m,32,6144,dw.ptr(),dc.ptr());
 for(I h:{-1,0,7,9,33})call(dx.ptr(),1,h,6144,dw.ptr(),dc.ptr());
 for(I s:{-1,0,4095,6145})call(dx.ptr(),1,32,s,dw.ptr(),dc.ptr());
 call(nullptr,1,32,6144,dw.ptr(),dc.ptr());call(dx.ptr(),1,32,6144,nullptr,dc.ptr());call(dx.ptr(),1,32,6144,dw.ptr(),nullptr);
 call(dx.ptr(),1,32,6144,dx.ptr(),dc.ptr());call(dx.ptr(),1,32,6144,dw.ptr(),dx.ptr());
 call(dx.ptr(),1,32,6144,dx.ptr()+1,dc.ptr());call(dx.ptr(),1,32,6144,dw.ptr(),dx.ptr()+2);
 auto bad=reinterpret_cast<float*>(uintptr_t(dx.ptr())+1);call(bad,1,32,6144,dw.ptr(),dc.ptr());
 auto wrap=reinterpret_cast<float*>(UINTPTR_MAX-3);call(wrap,1,32,6144,dw.ptr(),dc.ptr());
 req(dx.read()==x&&dw.read()==w&&dc.read()==cs,"invalid calls mutated data");std::cout<<"PASS invalid spans/shape, buffers unchanged\n";
}
static F read_raw(const std::string&p,size_t count){F out(count);std::ifstream f(p,std::ios::binary|std::ios::ate);req(bool(f)&&f.tellg()==std::streamoff(count*4),"fixture bytes "+p);f.seekg(0);f.read(reinterpret_cast<char*>(out.data()),std::streamsize(count*4));req(bool(f),"fixture read");for(float v:out)req(std::isfinite(v),"fixture finite");return out;}
static void actual_case(Api&a,const std::string&dir){
 F x=read_raw(dir+"/qkv.raw.bin",32*6144),wq=read_raw(dir+"/qnorm.weight.bin",128),wk=read_raw(dir+"/knorm.weight.bin",128),cs=read_raw(dir+"/cos_sin.bin",32*128),expected=read_raw(dir+"/expected.bin",32*6144);
 auto f64=oracle(x,32,32,6144,0,wq,cs),k64=oracle(x,32,8,6144,4096,wk,cs);
 for(I r=0;r<32;++r)for(I j=4096;j<5120;++j)f64[r*6144+j]=k64[r*6144+j];
 Buffer dx(a,x),dq(a,wq),dk(a,wk),dc(a,cs);
 req(a.rope(a.g,dx.ptr(),32,32,6144,dq.ptr(),dc.ptr())==0,"actualQ");req(a.rope(a.g,dx.ptr()+4096,32,8,6144,dk.ptr(),dc.ptr())==0,"actualK");
 F y=dx.read();req(std::memcmp(y.data(),expected.data(),y.size()*4)==0,"actual full CPU Torch QKV bits differ");
 req(dq.read()==wq&&dk.read()==wk&&dc.read()==cs,"actual metadata mutated");
 double e=compare(y,f64);std::cout<<"PASS actual "<<dir<<" full_QKV_bitwise=196608 F64_max_abs="<<e<<'\n';
}
static void host_guards(){
 auto*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<float*>(uintptr_t(0x10000000));auto*w=reinterpret_cast<float*>(uintptr_t(0x20000000));auto*cs=reinterpret_cast<float*>(uintptr_t(0x30000000));
 req(qv_ordered_strict::valid(g,x,4096,32,6144,w,cs),"host max geometry");
 req(!qv_ordered_strict::valid(g,x,4097,32,6144,w,cs)&&!qv_ordered_strict::valid(nullptr,x,1,8,1024,w,cs),"host rejected geometry");
 req(!qv_ordered_strict::valid(g,x,1,32,6144,x+1,cs)&&!qv_ordered_strict::valid(g,x,1,32,6144,w,x+2),"partial metadata alias");
 req(!qv_ordered_strict::valid(g,x,1,32,6144,reinterpret_cast<float*>(uintptr_t(w)+1),cs),"weight alignment");
 req(!qv_ordered_strict::valid(g,x,1,32,6144,w,reinterpret_cast<float*>(UINTPTR_MAX-3)),"metadata wrap");
}
static void special(Api&a){
 F w(128,1),cs(128,0);for(I j=0;j<64;++j)cs[2*j]=1;
 for(int mode=0;mode<6;++mode){F x(1024,0);
  if(mode==1)std::fill(x.begin(),x.end(),-0.f);
  if(mode==2)std::fill(x.begin(),x.end(),std::numeric_limits<float>::denorm_min());
  if(mode==3)std::fill(x.begin(),x.end(),1e20f);
  if(mode>=4)for(I h=0;h<8;++h)x[h*128]=mode==4?std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity();
  Buffer dx(a,x),dw(a,w),dc(a,cs);req(a.rope(a.g,dx.ptr(),1,8,1024,dw.ptr(),dc.ptr())==0,"special launch");F y=dx.read();
  for(I h=0;h<8;++h)for(I d=0;d<128;++d){float v=y[h*128+d];
   if(mode==4||(mode==5&&(d==0||d==64)))req(std::isnan(v),"NaN propagation");
   else if(mode==2)req(v==std::numeric_limits<float>::denorm_min()*1000.f,"subnormal F32 result");
   else {req(v==0,"zero/overflow/Inf unaffected components");if(mode==1)req(std::signbit(v)==(d>=64),"signed zero RNE propagation");}
  }
  req(dw.read()==w&&dc.read()==cs,"special metadata");std::cout<<"PASS special mode="<<mode<<'\n';
 }
}
int main(int argc,char**argv){try{
 host_guards();F x(128,0),w(128,1),cs(128,0);for(size_t j=0;j<64;++j)cs[j*2]=1;
 compare(x,oracle(x,1,1,128,0,w,cs));bool rejects=false;try{compare(F{1.f},std::vector<double>{0.});}catch(const std::exception&){rejects=true;}req(rejects,"oracle negative control");
 if(argc==2&&std::string(argv[1])=="--self-test"){std::cout<<"CPU PASS independent F64 zero+negative gate and guards\n";return 0;}
 req(argc==3,"usage: rms_rope_strict_numeric --self-test | PRIVATE_RUNTIME.so PINNED_FIXTURES");Api a(argv[1]);invalid(a);
 for(I rows:{1,3,32,33}){run_case(a,rows,32,6144,0,false);run_case(a,rows,8,6144,4096,false);}
 run_case(a,3,8,1024,0,true);special(a);actual_case(a,std::string(argv[2])+"/layer0");actual_case(a,std::string(argv[2])+"/block1");
 std::cout<<"GPU PASS9 F64 geometries +6 special +2 actual CPU-bitwise cases; not model admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
