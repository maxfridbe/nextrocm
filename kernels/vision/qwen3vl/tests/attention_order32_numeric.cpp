// Private whole-attention CPU/source-order and independent F64 operator validator.
#include "../attention_order32.h"
#include "../attention_order32.hpp"
#include "../../../../reference/qwenimage21-turbo/softmax_source_scalar.hpp"
#include "../../../../reference/qwenimage21-turbo/qkv_order_scalar.hpp"
#include <set>
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __SSE__
#include <xmmintrin.h>
#endif
using I=int64_t;using F=std::vector<float>;
static void need(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
static bool bits(const F&a,const F&b){return a.size()==b.size()&&std::memcmp(a.data(),b.data(),a.size()*4)==0;}
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
 decltype(&nrqv_attention_order32_f32) candidate=nullptr;
 decltype(&nrqv_attention_order32_capture) capture=nullptr;
 template<class T>void bind(T&f,const char*n){f=reinterpret_cast<T>(dlsym(lib,n));need(f!=nullptr,n);}
 explicit Api(const char*p){lib=dlopen(p,RTLD_NOW|RTLD_LOCAL);need(lib!=nullptr,"dlopen");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(candidate,"nrqv_attention_order32_f32");bind(capture,"nrqv_attention_order32_capture");need(open(0,&g)==0,"open");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=16;
 Buffer(Api&api,const F&v):a(api),n(v.size()){F all(n+2*guard,-9999.25f);std::copy(v.begin(),v.end(),all.begin()+guard);need(a.alloc(a.g,all.size()*4,&base)==0,"alloc");need(a.to(a.g,base,all.data(),all.size()*4)==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 float*ptr()const{return static_cast<float*>(base)+guard;}
 F read(){F all(n+2*guard);need(a.from(a.g,all.data(),base,all.size()*4)==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i){need(all[i]==-9999.25f&&all[n+guard+i]==-9999.25f,"guards");}return F(all.begin()+guard,all.end()-guard);}
};
struct Expected{F out,records;};
static Expected source(const F&qkv){
 Expected e{F(32*4096),F(32*32*163)};
 for(size_t r=0;r<32;++r)for(size_t h=0;h<32;++h){float*rec=e.records.data()+(r*32+h)*163;float logits[32];
  for(size_t j=0;j<32;++j){float raw=qkv_order::dot(unsigned(r),128,[&](unsigned d){return qkv[r*6144+h*128+d];},[&](unsigned d){return qkv[j*6144+4096+(h/4)*128+d];});logits[j]=j<=r?source_softmax::div(raw,source_softmax::from_bits(0x413504f3u)):-INFINITY;rec[j]=raw;rec[32+j]=logits[j];}
  auto s=source_softmax::row(logits);rec[64]=s.maximum;rec[129]=s.sum;rec[130]=s.inverse;
  for(size_t j=0;j<32;++j){rec[65+j]=s.shifted[j];rec[97+j]=s.ex[j];rec[131+j]=s.p[j];}
  for(size_t d=0;d<128;++d)e.out[(r*32+h)*128+d]=qkv_order::dot(unsigned(r),32,[&](unsigned j){return s.p[j];},[&](unsigned j){return qkv[j*6144+5120+(h/4)*128+d];});
 }return e;
}
static void host(){
 need(std::fegetround()==FE_TONEAREST,"CPU oracle RNE required");
#ifdef __SSE__
 need((_mm_getcsr()&((1u<<15)|(1u<<6)))==0,"CPU oracle FTZ/DAZ disabled required");
#endif
 void*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<float*>(uintptr_t(0x10000000));auto*y=reinterpret_cast<float*>(uintptr_t(0x20000000));auto*r=reinterpret_cast<float*>(uintptr_t(0x30000000));auto*s=reinterpret_cast<uint32_t*>(uintptr_t(0x40000000));
 need(qv_attention_order32::valid(g,x,32,y,r,s)&&qv_attention_order32::valid(g,x,32,y,nullptr,s),"valid");
 for(I n:{-1,0,1,31,33,4096})need(!qv_attention_order32::valid(g,x,n,y,r,s),"rows");
 need(!qv_attention_order32::valid(nullptr,x,32,y,r,s)&&!qv_attention_order32::valid(g,nullptr,32,y,r,s)&&!qv_attention_order32::valid(g,x,32,nullptr,r,s)&&!qv_attention_order32::valid(g,x,32,y,r,nullptr),"null");
 need(!qv_attention_order32::valid(g,x,32,x+1,r,s)&&!qv_attention_order32::valid(g,x,32,y,y+1,s)&&!qv_attention_order32::valid(g,x,32,y,const_cast<float*>(x)+1,s),"float aliases");
 for(const float*p:{x+1,y+1,r+1})need(!qv_attention_order32::valid(g,x,32,y,r,reinterpret_cast<uint32_t*>(const_cast<float*>(p))),"status alias");
 for(uintptr_t p:{uintptr_t(0x10000001),UINTPTR_MAX-3}){
  need(!qv_attention_order32::valid(g,reinterpret_cast<float*>(p),32,y,r,s),"input align/wrap");need(!qv_attention_order32::valid(g,x,32,reinterpret_cast<float*>(p),r,s),"out align/wrap");need(!qv_attention_order32::valid(g,x,32,y,reinterpret_cast<float*>(p),s),"records align/wrap");need(!qv_attention_order32::valid(g,x,32,y,r,reinterpret_cast<uint32_t*>(p)),"status align/wrap");
 }
 F zero(32*6144);auto e=source(zero);need(check(e.out,oracle(zero,32))==0,"independent zero oracle");
 std::cout<<"CPU PASS guards/source-order/F64zero; GPU/originalfullarray proof pending\n";
}
static void run(Api&a,const F&qkv,const std::string&name,const F*oqk=nullptr,const F*op=nullptr,const F*opv=nullptr){
 Buffer x(a,qkv),y(a,F(32*4096,-777)),plain(a,F(32*4096,-777)),rec(a,F(32*32*163,-777)),status(a,F{777});
 auto*sp=reinterpret_cast<uint32_t*>(status.ptr());
 need(a.capture(a.g,x.ptr(),32,y.ptr(),rec.ptr(),sp)==0,"capture launch");F got=y.read(),records=rec.read(),st=status.read();uint32_t code;std::memcpy(&code,st.data(),4);need(code==0,"capture numeric status");
 need(a.candidate(a.g,x.ptr(),32,plain.ptr(),sp)==0,"primary launch");F primary=plain.read();st=status.read();std::memcpy(&code,st.data(),4);need(code==0&&bits(primary,got),"primary/capture differ");
 Expected expected=source(qkv);need(bits(records,expected.records),"CPU ordered-QK/source-softmax intermediate bits: "+name);need(bits(got,expected.out),"CPU ordered-PV output bits: "+name);
 if(oqk){need(op&&opv&&oqk->size()==32768&&op->size()==32768&&opv->size()==131072,"original shapes");
  for(size_t h=0;h<32;++h)for(size_t q=0;q<32;++q)for(size_t k=0;k<32;++k){const float*rec=records.data()+(q*32+h)*163;size_t i=(h*32+q)*32+k;float scaled=k<=q?source_softmax::div((*oqk)[i],source_softmax::from_bits(0x413504f3u)):-INFINITY;need(std::memcmp(rec+k,&(*oqk)[i],4)==0,"full original QK bits");need(std::memcmp(rec+32+k,&scaled,4)==0,"original scaled/masked bits");need(std::memcmp(rec+131+k,&(*op)[i],4)==0,"full original P bits");}
  need(bits(got,*opv),"full original Torch PV bits");
 }
 double error=check(got,oracle(qkv,32));need(bits(x.read(),qkv),"input mutated");
 std::cout<<"PASS "<<name<<" QK/scale/allsoftmax/PVbits exact records="<<records.size()<<" F64max="<<error<<'\n';
}
static void invalid(Api&a){
 F xv(32*6144,1),yv(32*4096,2),rv(32*32*163,3),sv{777};Buffer x(a,xv),y(a,yv),r(a,rv),s(a,sv);auto*sp=reinterpret_cast<uint32_t*>(s.ptr());
 for(I n:{-1,0,1,31,33,4096}){need(a.candidate(a.g,x.ptr(),n,y.ptr(),sp)!=0,"primary rows");need(a.capture(a.g,x.ptr(),n,y.ptr(),r.ptr(),sp)!=0,"capture rows");}
 auto call=[&](void*c,const float*i,float*o,float*rec,uint32_t*st){need(a.capture(c,i,32,o,rec,st)!=0,"invalid capture accepted");};
 call(nullptr,x.ptr(),y.ptr(),r.ptr(),sp);call(a.g,nullptr,y.ptr(),r.ptr(),sp);call(a.g,x.ptr(),nullptr,r.ptr(),sp);call(a.g,x.ptr(),y.ptr(),r.ptr(),nullptr);call(a.g,x.ptr(),y.ptr(),nullptr,sp);
 call(a.g,x.ptr(),x.ptr()+1,r.ptr(),sp);call(a.g,x.ptr(),y.ptr(),y.ptr()+1,sp);call(a.g,x.ptr(),y.ptr(),x.ptr()+1,sp);
 for(float*p:{x.ptr()+1,y.ptr()+1,r.ptr()+1})call(a.g,x.ptr(),y.ptr(),r.ptr(),reinterpret_cast<uint32_t*>(p));
 for(uintptr_t p:{uintptr_t(x.ptr())+1,UINTPTR_MAX-3}){call(a.g,reinterpret_cast<float*>(p),y.ptr(),r.ptr(),sp);call(a.g,x.ptr(),reinterpret_cast<float*>(p),r.ptr(),sp);call(a.g,x.ptr(),y.ptr(),reinterpret_cast<float*>(p),sp);call(a.g,x.ptr(),y.ptr(),r.ptr(),reinterpret_cast<uint32_t*>(p));}
 need(bits(x.read(),xv)&&bits(y.read(),yv)&&bits(r.read(),rv)&&bits(s.read(),sv),"invalid call wrote output/status/input");
 // Entire tensor invalid: every query/head observes invalid Q; no output record may be written.
 F bad(32*6144,NAN);need(a.to(a.g,x.ptr(),bad.data(),bad.size()*4)==0,"bad upload");need(a.capture(a.g,x.ptr(),32,y.ptr(),r.ptr(),sp)==0,"domain launch");F st=s.read();uint32_t code;std::memcpy(&code,st.data(),4);need(code!=0&&bits(y.read(),yv)&&bits(r.read(),rv)&&bits(x.read(),bad),"domain status/no-write/input");
 F zero(32*6144,0);need(a.to(a.g,x.ptr(),zero.data(),zero.size()*4)==0,"recovery upload");need(a.capture(a.g,x.ptr(),32,y.ptr(),r.ptr(),sp)==0,"recovery launch");F got=y.read();st=s.read();std::memcpy(&code,st.data(),4);need(code==0&&bits(got,F(32*4096,0))&&bits(x.read(),zero),"recovery/reset");
 std::cout<<"PASS invalid arguments/domain no-write and same-buffer recovery\n";
}
static F read(const std::string&p,size_t n=32*6144){F q(n);std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f)&&f.tellg()==std::streamoff(q.size()*4),"QKV geometry");f.seekg(0);f.read(reinterpret_cast<char*>(q.data()),std::streamsize(q.size()*4));need(bool(f),"QKV read");for(float v:q)need(std::isfinite(v),"QKV finite");return q;}
int main(int argc,char**argv){try{
 host();if(argc==2&&std::string(argv[1])=="--self-test")return 0;need(argc==3,"usage: attention_order32_numeric PRIVATE_DSO PINNED_CORPUS");Api a(argv[1]);invalid(a);
 for(unsigned mode=0;mode<3;++mode){F q(32*6144);for(size_t i=0;i<q.size();++i){q[i]=float(int((i*73+11)%513)-256)/512.f;if(mode==1&&i%6144<5120)q[i]=0;if(mode==2)q[i]=0;}run(a,q,"synthetic"+std::to_string(mode));}
 std::string root=argv[2];std::ifstream list(root+"/cases.txt");need(bool(list),"cases");std::string name;std::set<std::string>names;
 while(list>>name){need(name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos&&names.insert(name).second,"case name");F q=read(root+"/"+name+".qkv.f32"),oq=read(root+"/"+name+".qk.f32",32768),op=read(root+"/"+name+".probability.f32",32768),ov=read(root+"/"+name+".pv.f32",131072);run(a,q,name,&oq,&op,&ov);}
 need(list.eof()&&names.size()==12,"complete12actual original corpus");
 std::cout<<"GPU PASS wholeattention12actual+3synthetic, fulloriginalTorch QK/scale/P/PV bits plus source-softmax intermediates; no fullmodel claim\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
