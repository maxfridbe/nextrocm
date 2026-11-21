// CPU: c++ -std=c++17 -O2 -Wall -Wextra -Werror FILE -ldl -o OUT
// OUT --self-test; parent-only OUT PRIVATE_RUNTIME.so (GPU0).
#include "../qv.h"
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
 decltype(&nrqv_rms_rope_f32) rope=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));req(f!=nullptr,n);}
 explicit Api(const char*p){lib=dlopen(p,RTLD_NOW|RTLD_LOCAL);req(lib!=nullptr,"dlopen");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(rope,"nrqv_rms_rope_f32");req(open(0,&g)==0,"open GPU");}
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
 auto bad=reinterpret_cast<float*>(uintptr_t(dx.ptr())+1);call(bad,1,32,6144,dw.ptr(),dc.ptr());
 auto wrap=reinterpret_cast<float*>(UINTPTR_MAX-3);call(wrap,1,32,6144,dw.ptr(),dc.ptr());
 req(dx.read()==x&&dw.read()==w&&dc.read()==cs,"invalid calls mutated data");std::cout<<"PASS invalid spans/shape, buffers unchanged\n";
}
int main(int argc,char**argv){try{
 F x(128,0),w(128,1),cs(128,0);for(size_t j=0;j<64;++j)cs[j*2]=1;
 compare(x,oracle(x,1,1,128,0,w,cs));bool rejects=false;try{compare(F{1.f},std::vector<double>{0.});}catch(const std::exception&){rejects=true;}req(rejects,"oracle negative control");
 if(argc==2&&std::string(argv[1])=="--self-test"){std::cout<<"CPU PASS independent F64 zero+negative gate\n";return 0;}
 req(argc==2,"usage: rms_rope_numeric --self-test | PRIVATE_RUNTIME.so");Api a(argv[1]);invalid(a);
 for(I rows:{1,3,32}){run_case(a,rows,32,6144,0,false);run_case(a,rows,8,6144,4096,false);}
 run_case(a,3,8,1024,0,true);std::cout<<"GPU PASS7 F32 RMSRoPE cases; not model admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
