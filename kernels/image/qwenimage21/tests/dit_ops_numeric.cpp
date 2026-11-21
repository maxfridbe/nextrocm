// Private DiT operators independent CPU/F64 validator. --self-test is CPU only.
// g++ -std=c++17 -O2 -Wall -Wextra -Werror dit_ops_numeric.cpp -ldl -o OUT
// OUT --self-test; parent only: OUT PATH_TO_PRIVATE_OPERATOR_DSO
#include "../dit_ops.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
using I=int64_t;using F=std::vector<float>;using H=std::vector<uint16_t>;using B=std::vector<unsigned char>;
static void require(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
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
static uint16_t encode_double(double x) {
 const uint16_t sign=std::signbit(x)?0x8000:0;double v=std::abs(x);
 if(!std::isfinite(v))return uint16_t(sign|0x7c00|(std::isnan(v)?0x200:0));
 if(v>=65520.)return uint16_t(sign|0x7c00);
 unsigned lo=0,hi=0x7bff;
 while(lo<hi){unsigned mid=(lo+hi+1)/2;if(double(decode(uint16_t(mid)))<=v)lo=mid;else hi=mid-1;}
 if(lo==0x7bff)return uint16_t(sign|lo);
 double a=decode(uint16_t(lo)),b=decode(uint16_t(lo+1));
 if(v-a>b-v||(v-a==b-v&&(lo&1)))++lo;
 return uint16_t(sign|lo);
}
static F values(size_t n,uint32_t seed,float scale=1){F out(n);for(auto&v:out){seed=1664525u*seed+1013904223u;v=float(int(seed>>8)%2049-1024)/1024.f*scale;}return out;}
static H halves(const F&x){H o;for(float v:x)o.push_back(encode(v));return o;}
template<class T>static B bytes(const std::vector<T>&x){B b(x.size()*sizeof(T));std::memcpy(b.data(),x.data(),b.size());return b;}
template<class T>static std::vector<T> unpack(const B&b){require(b.size()%sizeof(T)==0,"typed bytes");std::vector<T> x(b.size()/sizeof(T));std::memcpy(x.data(),b.data(),b.size());return x;}
static double half_ulp(uint16_t h){h&=0x7fff;if(h==0)return std::ldexp(1.,-24);double v=decode(h),lower=decode(uint16_t(h-1)),upper=h<0x7bff?decode(uint16_t(h+1)):v;return std::max(v-lower,upper-v);}
static int half_order(uint16_t h){return h&0x8000?0x8000-int(h&0x7fff):0x8000+int(h);}
struct Metric{double max_abs=0,diff2=0,ref2=0;int ulp=0;size_t count=0;};
static Metric check_half(const H&actual,const std::vector<double>&reference){
 require(actual.size()==reference.size(),"half shape");Metric m;
 for(size_t i=0;i<actual.size();++i){uint16_t rh=encode_double(reference[i]);double r=decode(rh),a=decode(actual[i]),diff=std::abs(a-r);require(std::isfinite(a)&&std::isfinite(r),"half nonfinite");require(diff<=half_ulp(rh)+2e-5,"frozen halfULP+2e-5 gate at "+std::to_string(i));m.max_abs=std::max(m.max_abs,diff);m.diff2+=diff*diff;m.ref2+=r*r;m.ulp=std::max(m.ulp,std::abs(half_order(actual[i])-half_order(rh)));++m.count;}return m;
}
static Metric check_float(const F&actual,const std::vector<double>&reference){
 require(actual.size()==reference.size(),"float shape");Metric m;
 for(size_t i=0;i<actual.size();++i){double a=actual[i],r=reference[i],diff=std::abs(a-r);require(std::isfinite(a)&&std::isfinite(r),"float nonfinite");require(diff<=2e-5+2e-5*std::abs(r),"frozen F32 absolute+relative gate");m.max_abs=std::max(m.max_abs,diff);m.diff2+=diff*diff;m.ref2+=r*r;++m.count;}return m;
}
static void report(const std::string&name,Metric m){std::cout<<"PASS "<<name<<" values="<<m.count<<" max_abs="<<m.max_abs<<" relative_L2="<<std::sqrt(m.diff2)/std::max(std::sqrt(m.ref2),1e-30)<<" max_half_ULP="<<m.ulp<<'\n';}
// Reference formulas use double reductions, not the HIP tree or serial-F32 implementation.
static std::vector<double> ln_ref(const F&x,I m,I c,float eps,const F*scale){std::vector<double> o(x.size());for(I r=0;r<m;++r){double mean=0;for(I i=0;i<c;++i)mean+=x[r*c+i];mean/=double(c);double var=0;for(I i=0;i<c;++i){double z=double(x[r*c+i])-mean;var+=z*z;}double inv=1/std::sqrt(var/double(c)+double(eps));for(I i=0;i<c;++i)o[r*c+i]=(double(x[r*c+i])-mean)*inv*(scale?1+double((*scale)[i]):1);}return o;}
static std::vector<double> rms_ref(const F&x,I m,I c,const F&w,float eps){std::vector<double> o(x.size());for(I r=0;r<m;++r){double s=0;for(I i=0;i<c;++i)s+=double(x[r*c+i])*x[r*c+i];double inv=1/std::sqrt(s/double(c)+eps);for(I i=0;i<c;++i)o[r*c+i]=double(x[r*c+i])*inv*w[i];}return o;}
static std::vector<double> rope_ref(const H&x,I m,I h,I d,I stride,I offset,const F&w,float eps,const F&cs,int rot){
 std::vector<double> o;for(uint16_t v:x)o.push_back(decode(v));
 for(I r=0;r<m;++r)for(I head=0;head<h;++head){I base=offset+r*stride+head*d;double sum=0;for(I i=0;i<d;++i){double v=decode(x[base+i]);sum+=v*v;}double inv=1/std::sqrt(sum/double(d)+eps);for(I i=0;i<d;++i)o[base+i]=double(decode(x[base+i]))*inv*w[i];int half=rot/2;for(int i=0;i<half;++i){double a=o[base+i],b=o[base+half+i],co=cs[(r*half+i)*2],si=cs[(r*half+i)*2+1];o[base+i]=a*co-b*si;o[base+half+i]=b*co+a*si;}}
 return o;
}
static void self_test(){
 using namespace nrqi_dit;void*g=reinterpret_cast<void*>(uintptr_t(1));auto x=reinterpret_cast<float*>(uintptr_t(0x100000000));auto y=reinterpret_cast<void*>(uintptr_t(0x200000000));auto w=reinterpret_cast<float*>(uintptr_t(0x300000000));auto cs=reinterpret_cast<float*>(uintptr_t(0x400000000));
 require(valid_ln(g,x,3,4096,1e-6f,w,y)&&valid_ln(g,x,3,4096,1e-6f,nullptr,y),"LN valid");require(valid_rms(g,x,3,4096,w,1e-6f,x),"RMS exact alias");require(valid_rope(g,x,3,32,128,12288,w,1e-6f,cs,128),"RMSRoPE actual geometry");require(valid_gate(g,x,y,3,4096,w),"gate valid");
 for(I c:{-1,0,8193}){require(!valid_ln(g,x,1,c,1e-6f,w,y),"LN invalidC");require(!valid_rms(g,x,1,c,w,1e-6f,reinterpret_cast<float*>(y)),"RMS invalidC");require(!valid_gate(g,x,y,1,c,w),"gate invalidC");}
 for(I m:{-1,0,65537}){require(!valid_ln(g,x,m,4096,1e-6f,w,y),"LN invalidM");require(!valid_rope(g,x,m,32,128,12288,w,1e-6f,cs,128),"rope invalidM");}
 for(float e:{0.f,-1.f,INFINITY,NAN}){require(!valid_ln(g,x,1,4096,e,w,y)&&!valid_rms(g,x,1,4096,w,e,reinterpret_cast<float*>(y))&&!valid_rope(g,x,1,32,128,12288,w,e,cs,128),"invalid epsilon");}
 require(!valid_ln(g,x,1,4096,1e-6f,w,x)&&!valid_rms(g,x,1,4096,w,1e-6f,x+1)&&!valid_gate(g,x,x,1,4096,w),"overlap guard");
 require(!valid_rope(g,x,1,32,128,4095,w,1e-6f,cs,128)&&!valid_rope(g,x,1,32,128,12288,w,1e-6f,cs,129),"rope stride/rotation guards");
 require(!valid_ln(nullptr,x,1,1,1e-6f,w,y)&&!valid_ln(g,nullptr,1,1,1e-6f,w,y)&&!valid_rms(g,x,1,1,nullptr,1e-6f,reinterpret_cast<float*>(y))&&!valid_rope(g,x,1,1,128,384,w,1e-6f,nullptr,128),"null guards");
 auto mis=reinterpret_cast<float*>(uintptr_t(x)+1);require(!valid_ln(g,mis,1,1,1e-6f,w,y)&&!valid_rope(g,mis,1,1,128,384,w,1e-6f,cs,128),"alignment guards");
 auto wrap=reinterpret_cast<float*>(UINTPTR_MAX-3);require(!valid_ln(g,wrap,1,2,1e-6f,w,y)&&!valid_rope(g,wrap,2,1,128,384,w,1e-6f,cs,128),"address wrap");
 for(unsigned i=0;i<65536;++i){uint16_t h=uint16_t(i);if((h&0x7c00)!=0x7c00)require(encode(decode(h))==h && encode_double(double(decode(h)))==h,"independent half conversion roundtrip");}
 for(unsigned h=0;h<0x7bff;++h){double mid=(double(decode(uint16_t(h)))+decode(uint16_t(h+1)))*.5;require(encode_double(mid)==(h+(h&1)),"double-half midpoint ties-even");require(encode_double(std::nextafter(mid,0.))==h&&encode_double(std::nextafter(mid,INFINITY))==h+1,"double-half midpoint neighbors");}
 F constant(3*257,1.25f),scale(257,.5f);auto l=ln_ref(constant,3,257,1e-6f,&scale);require(std::all_of(l.begin(),l.end(),[](double x){return x==0;}),"constantLN oracle");
 bool reject=false;try{check_float(F{.1f},std::vector<double>{0});}catch(const std::exception&){reject=true;}require(reject,"F32 negative gate control");reject=false;try{check_half(H{encode(.1f)},std::vector<double>{0});}catch(const std::exception&){reject=true;}require(reject,"F16 negative gate control");
 std::cout<<"CPU PASS allfinite half roundtrips, independent references, negative tolerance controls, host geometry/alias/null/epsilon/alignment/wrap guards\n";
}
struct Api{
 void*lib=nullptr;void*g=nullptr;int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqi_ln_mod) ln=nullptr;decltype(&nrqi_rms_rope) rope=nullptr;decltype(&nrqi_text_rms_f32) rms=nullptr;decltype(&nrqi_gate_add_f32_f16) gate=nullptr;
 template<class T>void bind(T&fn,const char*name){fn=reinterpret_cast<T>(dlsym(lib,name));require(fn!=nullptr,std::string("missing ")+name);}
 explicit Api(const char*path){lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);require(lib!=nullptr,"operator DSO load");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(ln,"nrqi_ln_mod");bind(rope,"nrqi_rms_rope");bind(rms,"nrqi_text_rms_f32");bind(gate,"nrqi_gate_add_f32_f16");require(open(0,&g)==0,"GPU open");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=64;
 Buffer(Api&api,const B&x):a(api),n(x.size()){require(a.alloc(a.g,n+2*guard,&base)==0,"alloc");B all(n+2*guard,0xa5);std::copy(x.begin(),x.end(),all.begin()+guard);require(a.to(a.g,base,all.data(),all.size())==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 void*ptr()const{return static_cast<unsigned char*>(base)+guard;}float*fp()const{return static_cast<float*>(ptr());}
 B read()const{B all(n+2*guard);require(a.from(a.g,all.data(),base,all.size())==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i)require(all[i]==0xa5&&all[n+guard+i]==0xa5,"allocation guard changed");return B(all.begin()+guard,all.end()-guard);}
};
static void norm_case(Api&a,I m,I c,int special=0){
 F x=values(size_t(m*c),7,2),s=values(c,19,.5f),w=values(c,29,.5f);for(float&v:w)v+=1.f;
 if(special==1)std::fill(x.begin(),x.end(),1.25f);
 if(special==2)for(size_t i=0;i<x.size();++i)x[i]=float(int(i%2)*2-1)*std::ldexp(1.f,-12);
 Buffer dx(a,bytes(x)),ds(a,bytes(s)),dw(a,bytes(w)),halfout(a,bytes(H(x.size(),0x7e00))),floatout(a,bytes(F(x.size(),NAN)));
 for(bool mod:{false,true}){require(a.ln(a.g,dx.fp(),m,c,1e-6f,mod?ds.fp():nullptr,halfout.ptr())==0,"LN launch");report("ln m="+std::to_string(m)+" c="+std::to_string(c)+" mod="+std::to_string(mod),check_half(unpack<uint16_t>(halfout.read()),ln_ref(x,m,c,1e-6f,mod?&s:nullptr)));}
 require(a.rms(a.g,dx.fp(),m,c,dw.fp(),1e-6f,floatout.fp())==0,"RMS launch");report("text RMS",check_float(unpack<float>(floatout.read()),rms_ref(x,m,c,w,1e-6f)));
 require(dx.read()==bytes(x)&&ds.read()==bytes(s)&&dw.read()==bytes(w),"normalization input mutated");
 require(a.rms(a.g,dx.fp(),m,c,dw.fp(),1e-6f,dx.fp())==0,"RMS inplace launch");report("text RMS exact alias",check_float(unpack<float>(dx.read()),rms_ref(x,m,c,w,1e-6f)));
}
static void rope_case(Api&a,I m,I h,I d,I stride,I offset,int rot){
 H x=halves(values(size_t(m*stride),41,2));F w=values(d,47,.5f);for(float&v:w)v+=1;F cs(size_t(m*std::max(rot,1)),0);
 for(I r=0;r<m;++r)for(int j=0;j<rot/2;++j){double angle=(r+1)*.137+(j+1)*.019;cs[r*rot+2*j]=float(std::cos(angle));cs[r*rot+2*j+1]=float(std::sin(angle));}
 auto expected=rope_ref(x,m,h,d,stride,offset,w,1e-6f,cs,rot);Buffer dx(a,bytes(x)),dw(a,bytes(w)),dc(a,bytes(cs));auto*p=static_cast<uint16_t*>(dx.ptr())+offset;
 require(a.rope(a.g,p,m,h,d,stride,dw.fp(),1e-6f,rot?dc.fp():nullptr,rot)==0,"RMSRoPE launch");H got=unpack<uint16_t>(dx.read());std::vector<bool> touched(x.size(),false);
 for(I r=0;r<m;++r)for(I head=0;head<h;++head)for(I i=0;i<d;++i)touched[offset+r*stride+head*d+i]=true;
 for(size_t i=0;i<x.size();++i)if(!touched[i])require(x[i]==got[i],"unused QKV/padding lane overwritten");
 require(dw.read()==bytes(w)&&dc.read()==bytes(cs),"rope metadata mutated");report("RMSRoPE m="+std::to_string(m)+" h="+std::to_string(h)+" d="+std::to_string(d)+" stride="+std::to_string(stride)+" offset="+std::to_string(offset)+" rot="+std::to_string(rot),check_half(got,expected));
}
static void gate_case(Api&a,I m,I c,bool withgate){F x=values(m*c,53,2),gate=values(c,59,.75f);H y=halves(values(m*c,61,2));std::vector<double> expected(x.size());for(size_t i=0;i<x.size();++i)expected[i]=double(x[i])+double(decode(y[i]))*(withgate?double(gate[i%size_t(c)]):1.);Buffer dx(a,bytes(x)),dy(a,bytes(y)),dg(a,bytes(gate));require(a.gate(a.g,dx.fp(),dy.ptr(),m,c,withgate?dg.fp():nullptr)==0,"gate launch");report("gate residual",check_float(unpack<float>(dx.read()),expected));require(dy.read()==bytes(y)&&dg.read()==bytes(gate),"gate input mutated");}
static void invalid_device(Api&a){Buffer x(a,bytes(F(16384,1))),y(a,bytes(H(16384,encode(2)))),w(a,bytes(F(8192,1))),o(a,bytes(H(16384,encode(3))));B xb=x.read(),yb=y.read(),wb=w.read(),ob=o.read();
 for(I c:{-1,0,8193}){require(a.ln(a.g,x.fp(),1,c,1e-6f,w.fp(),o.ptr())!=0,"badC LN launched");require(a.rms(a.g,x.fp(),1,c,w.fp(),1e-6f,x.fp())!=0,"badC RMS launched");require(a.gate(a.g,x.fp(),y.ptr(),1,c,w.fp())!=0,"badC gate launched");}
 require(a.ln(a.g,x.fp(),1,4096,NAN,w.fp(),o.ptr())!=0&&a.ln(a.g,x.fp(),1,4096,1e-6f,w.fp(),x.ptr())!=0,"LN invalid epsilon/overlap");
 require(a.rope(a.g,y.ptr(),1,32,128,4095,w.fp(),1e-6f,w.fp(),128)!=0&&a.rope(a.g,y.ptr(),1,32,128,12288,w.fp(),1e-6f,w.fp(),129)!=0,"rope invalid shape");
 require(a.rms(a.g,x.fp(),1,4096,nullptr,1e-6f,x.fp())!=0&&a.rms(a.g,x.fp(),1,4096,w.fp(),1e-6f,x.fp()+1)!=0,"RMS null/partialalias");
 require(a.gate(a.g,x.fp(),x.ptr(),1,4096,w.fp())!=0&&a.gate(nullptr,x.fp(),y.ptr(),1,4096,w.fp())!=0,"gate alias/nullcontext");
 require(x.read()==xb&&y.read()==yb&&w.read()==wb&&o.read()==ob,"invalid call changed device memory");std::cout<<"PASS rejected device calls preserve all buffers and allocation guards\n";
}
int main(int argc,char**argv){try{std::cout<<std::setprecision(9);self_test();if(argc==2&&std::string(argv[1])=="--self-test")return 0;require(argc==2,"usage: dit_ops_numeric --self-test | PRIVATE_RUNTIME.so");Api a(argv[1]);invalid_device(a);
 for(I c:{1,3,31,255,256,257,4096,8192}) { norm_case(a,3,c); }
 norm_case(a,2,4096,1);norm_case(a,2,4096,2);
 rope_case(a,3,32,128,12288,0,128);rope_case(a,3,32,128,12288,4096,128);rope_case(a,3,3,10,48,0,6);rope_case(a,2,1,128,384,0,0);rope_case(a,1,1,256,768,0,256);
 gate_case(a,3,4096,true);gate_case(a,3,4096,false);gate_case(a,1,257,true);std::cout<<"ALL PASS private DiT operators; no model/GPU-fit admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
