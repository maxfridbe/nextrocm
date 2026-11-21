// Independent IEEE conversion oracle, test only; no frozen DiT file dependency.
#include "../convert.hpp"
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
template<class T>static B bytes(const std::vector<T>&x){B b(x.size()*sizeof(T));std::memcpy(b.data(),x.data(),b.size());return b;}
template<class T>static std::vector<T> unpack(const B&b){require(b.size()%sizeof(T)==0,"typed bytes");std::vector<T> x(b.size()/sizeof(T));std::memcpy(x.data(),b.data(),b.size());return x;}
struct Api{
 void*lib=nullptr;void*g=nullptr;int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqi_f32_to_f16) down=nullptr;decltype(&nrqi_f16_to_f32) up=nullptr;
 template<class T>void bind(T&fn,const char*name){fn=reinterpret_cast<T>(dlsym(lib,name));require(fn!=nullptr,std::string("missing ")+name);}
 explicit Api(const char*path){lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);require(lib!=nullptr,"operator DSO load");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(down,"nrqi_f32_to_f16");bind(up,"nrqi_f16_to_f32");require(open(0,&g)==0,"GPU open");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=64;
 Buffer(Api&api,const B&x):a(api),n(x.size()){require(a.alloc(a.g,n+2*guard,&base)==0,"alloc");B all(n+2*guard,0xa5);std::copy(x.begin(),x.end(),all.begin()+guard);require(a.to(a.g,base,all.data(),all.size())==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 void*ptr()const{return static_cast<unsigned char*>(base)+guard;}float*fp()const{return static_cast<float*>(ptr());}
 B read()const{B all(n+2*guard);require(a.from(a.g,all.data(),base,all.size())==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i)require(all[i]==0xa5&&all[n+guard+i]==0xa5,"allocation guard changed");return B(all.begin()+guard,all.end()-guard);}
};
static uint32_t bits(float x){uint32_t b;std::memcpy(&b,&x,4);return b;}
static F samples(){
 F x;
 for(unsigned h=0;h<65536;++h)x.push_back(decode(uint16_t(h)));
 for(unsigned h=0;h<0x7bff;++h){float mid=float((double(decode(uint16_t(h)))+decode(uint16_t(h+1)))*.5);for(float v:{std::nextafter(mid,0.f),mid,std::nextafter(mid,INFINITY)}){x.push_back(v);x.push_back(-v);}}
 for(float v:{65504.f,65519.f,65520.f,65536.f,std::numeric_limits<float>::max(),std::numeric_limits<float>::denorm_min(),INFINITY,NAN}){x.push_back(v);x.push_back(-v);}
 return x;
}
static void cpu(){
 void*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<void*>(uintptr_t(0x100000000));auto*y=reinterpret_cast<void*>(uintptr_t(0x200000000));
 for(auto sizes: {std::pair<unsigned,unsigned>{4,2},{2,4}}){
 auto a=sizes.first,b=sizes.second;
 require(nrqi_cast::valid(g,x,257,y,a,b),"valid geometry");
 for(I n:{I(-1),I(0),I(INT32_MAX)+1,I(INT64_MAX)})require(!nrqi_cast::valid(g,x,n,y,a,b),"count guard");
 require(!nrqi_cast::valid(nullptr,x,1,y,a,b)&&!nrqi_cast::valid(g,nullptr,1,y,a,b)&&!nrqi_cast::valid(g,x,1,nullptr,a,b),"null guard");
 require(!nrqi_cast::valid(g,x,257,x,a,b)&&!nrqi_cast::valid(g,x,257,static_cast<char*>(x)+4,a,b),"alias guard");
 require(!nrqi_cast::valid(g,static_cast<char*>(x)+1,1,y,a,b)&&!nrqi_cast::valid(g,x,1,static_cast<char*>(y)+1,a,b),"alignment");
 require(!nrqi_cast::valid(g,reinterpret_cast<void*>(UINTPTR_MAX-(a-1)),2,y,a,b)&&!nrqi_cast::valid(g,x,2,reinterpret_cast<void*>(UINTPTR_MAX-(b-1)),a,b),"wrap");
 }
 for(unsigned h=0;h<65536;++h){float f=decode(uint16_t(h));if(std::isfinite(f))require(encode(f)==h,"all finite halves roundtrip");}
 for(float f:samples())if(!std::isnan(f))require(encode(f)==encode_double(double(f)),"integer vs direct-double oracle");
 std::cout<<"CPU PASS guards and independent conversion oracles\n";
}
static void down_case(Api&a,const F&x){
 Buffer src(a,bytes(x)),dst(a,bytes(H(x.size(),0x3555)));
 require(a.down(a.g,src.fp(),I(x.size()),dst.ptr())==0,"down launch");H got=unpack<uint16_t>(dst.read());
 for(size_t i=0;i<x.size();++i){if(std::isnan(x[i]))require(std::isnan(decode(got[i])),"down NaN category");else require(got[i]==encode(x[i]),"down exact RNE bits at "+std::to_string(i));}
 require(src.read()==bytes(x),"down input changed");std::cout<<"PASS F32_to_F16 exact_bits_or_NaN_category values="<<x.size()<<'\n';
}
static void up_case(Api&a,const H&x){
 Buffer src(a,bytes(x)),dst(a,bytes(F(x.size(),1.25f)));
 require(a.up(a.g,src.ptr(),I(x.size()),dst.fp())==0,"up launch");F got=unpack<float>(dst.read());
 for(size_t i=0;i<x.size();++i){float ref=decode(x[i]);if(std::isnan(ref))require(std::isnan(got[i]),"up NaN category");else require(bits(got[i])==bits(ref),"up exact bits");}
 require(src.read()==bytes(x),"up input changed");std::cout<<"PASS F16_to_F32 exact_bits_or_NaN_category values="<<x.size()<<'\n';
}
static void invalid(Api&a){
 Buffer x(a,bytes(F(1024,1))),y(a,bytes(F(1024,2)));B before=x.read(),after=y.read();
 for(I n:{I(-1),I(0),I(INT32_MAX)+1})require(a.down(a.g,x.fp(),n,y.ptr())!=0&&a.up(a.g,x.ptr(),n,y.fp())!=0,"invalid count launched");
 require(a.down(a.g,x.fp(),257,x.ptr())!=0&&a.up(a.g,x.ptr(),257,x.fp())!=0,"alias launched");
 require(a.down(a.g,x.fp(),257,x.fp()+1)!=0&&a.up(a.g,x.ptr(),257,x.fp()+1)!=0,"partial alias launched");
 require(a.down(nullptr,x.fp(),1,y.ptr())!=0&&a.up(nullptr,x.ptr(),1,y.fp())!=0,"null ctx launched");
 require(a.down(a.g,nullptr,1,y.ptr())!=0&&a.up(a.g,nullptr,1,y.fp())!=0,"null input launched");
 require(a.down(a.g,x.fp(),1,nullptr)!=0&&a.up(a.g,x.ptr(),1,nullptr)!=0,"null output launched");
 auto*mis=static_cast<char*>(x.ptr())+1;
 require(a.down(a.g,reinterpret_cast<float*>(mis),1,y.ptr())!=0&&a.up(a.g,mis,1,y.fp())!=0,"misaligned input launched");
 require(x.read()==before&&y.read()==after,"invalid changed device storage");std::cout<<"PASS invalid calls preserve buffers/guards\n";
}
int main(int argc,char**argv){try{cpu();if(argc==2&&std::string(argv[1])=="--self-test")return 0;require(argc==2,"usage: convert_numeric --self-test | PRIVATE_DSO");Api a(argv[1]);invalid(a);
 for(size_t n:{1,2,3,255,256,257,4095,4096,4097}){F f(n);H h(n);for(size_t i=0;i<n;++i){f[i]=float(int(i%1025)-512)/31.f;h[i]=uint16_t((i*379+17)%65536);}down_case(a,f);up_case(a,h);}
 H all(65536);for(size_t i=0;i<all.size();++i)all[i]=uint16_t(i);up_case(a,all);down_case(a,samples());std::cout<<"ALL PASS private casts; no model admission\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
