// Independent F64 VAE operators, test only; fixed 1halfULP+2e-5 arithmetic gate.
#include "../vae.hpp"
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
static void report(const std::string&name,Metric m){std::cout<<"PASS "<<name<<" values="<<m.count<<" max_abs="<<m.max_abs<<" relative_L2="<<std::sqrt(m.diff2)/std::max(std::sqrt(m.ref2),1e-30)<<" max_half_ULP="<<m.ulp<<'\n';}
struct Api{
 void*lib=nullptr;void*g=nullptr;int(*open)(int,void**)=nullptr;void(*close)(void*)=nullptr;int(*sync)(void*)=nullptr;int(*alloc)(void*,size_t,void**)=nullptr;int(*free)(void*,void*)=nullptr;int(*to)(void*,void*,const void*,size_t)=nullptr;int(*from)(void*,void*,const void*,size_t)=nullptr;
 decltype(&nrqi_vae_conv) conv=nullptr;decltype(&nrqi_vae_silu) silu=nullptr;decltype(&nrqi_vae_softmax) softmax=nullptr;
 decltype(&nrqi_vae_up2) up=nullptr;decltype(&nrqi_vae_dupup_add) dup=nullptr;decltype(&nrqi_vae_pad_br) pad=nullptr;decltype(&nrqi_vae_avg_down_add) down=nullptr;decltype(&nrqi_vae_rgba8) rgba=nullptr;
 template<class T>void bind(T&fn,const char*name){fn=reinterpret_cast<T>(dlsym(lib,name));require(fn!=nullptr,std::string("missing ")+name);}
 explicit Api(const char*path){lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);require(lib!=nullptr,"operator DSO load");bind(open,"ns_gpu_open");bind(close,"ns_gpu_close");bind(sync,"ns_sync");bind(alloc,"ns_alloc");bind(free,"ns_free");bind(to,"ns_copy_to");bind(from,"ns_copy_from");bind(conv,"nrqi_vae_conv");bind(silu,"nrqi_vae_silu");bind(softmax,"nrqi_vae_softmax");bind(up,"nrqi_vae_up2");bind(dup,"nrqi_vae_dupup_add");bind(pad,"nrqi_vae_pad_br");bind(down,"nrqi_vae_avg_down_add");bind(rgba,"nrqi_vae_rgba8");require(open(0,&g)==0,"GPU open");}
 ~Api(){if(g){sync(g);close(g);}if(lib)dlclose(lib);}
};
struct Buffer{
 Api&a;void*base=nullptr;size_t n;static constexpr size_t guard=64;
 Buffer(Api&api,const B&x):a(api),n(x.size()){require(a.alloc(a.g,n+2*guard,&base)==0,"alloc");B all(n+2*guard,0xa5);std::copy(x.begin(),x.end(),all.begin()+guard);require(a.to(a.g,base,all.data(),all.size())==0&&a.sync(a.g)==0,"upload");}
 Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;~Buffer(){if(base){a.sync(a.g);a.free(a.g,base);}}
 void*ptr()const{return static_cast<unsigned char*>(base)+guard;}float*fp()const{return static_cast<float*>(ptr());}
 B read()const{B all(n+2*guard);require(a.from(a.g,all.data(),base,all.size())==0&&a.sync(a.g)==0,"read");for(size_t i=0;i<guard;++i)require(all[i]==0xa5&&all[n+guard+i]==0xa5,"allocation guard changed");return B(all.begin()+guard,all.end()-guard);}
};
static std::vector<double> conv_ref(const H&x,const H&wt,const F&bias,I h,I w,I ci,I co,int k,int stride){
 I oh=stride==1?h:(h-k)/stride+1,ow=stride==1?w:(w-k)/stride+1,pad=stride==1?k/2:0;std::vector<double> y(oh*ow*co);
 for(I yy=0;yy<oh;++yy)for(I xx=0;xx<ow;++xx)for(I o=0;o<co;++o){double sum=bias[o];for(I c=0;c<ci;++c)for(int ky=0;ky<k;++ky)for(int kx=0;kx<k;++kx){I iy=yy*stride-pad+ky,ix=xx*stride-pad+kx;if(iy>=0&&iy<h&&ix>=0&&ix<w)sum+=double(decode(x[(iy*w+ix)*ci+c]))*decode(wt[((o*ci+c)*k+ky)*k+kx]);}y[(yy*ow+xx)*co+o]=sum;}
 return y;
}
static void conv_case(Api&a,I h,I w,I ci,I co,int k,int stride){
 H x=halves(values(h*w*ci,31,.5f)),wt=halves(values(co*ci*k*k,53,.125f));F bias=values(co,61,.25f);auto ref=conv_ref(x,wt,bias,h,w,ci,co,k,stride);
 Buffer dx(a,bytes(x)),dw(a,bytes(wt)),db(a,bytes(bias)),out(a,bytes(H(ref.size(),0x7e00)));
 require(a.conv(a.g,dx.ptr(),h,w,ci,dw.ptr(),co,k,stride,db.fp(),out.ptr())==0,"conv launch");report("conv h="+std::to_string(h)+" w="+std::to_string(w)+" ci="+std::to_string(ci)+" co="+std::to_string(co)+" k="+std::to_string(k)+" stride="+std::to_string(stride),check_half(unpack<uint16_t>(out.read()),ref));
 require(dx.read()==bytes(x)&&dw.read()==bytes(wt)&&db.read()==bytes(bias),"conv readonly inputs");
}
static void element_case(Api&a){
 H x=halves(values(4097,73,8.f));std::vector<double> ref;for(auto h:x){double v=decode(h);ref.push_back(v/(1+std::exp(-v)));}Buffer dx(a,bytes(x));require(a.silu(a.g,dx.ptr(),x.size())==0,"silu launch");report("silu",check_half(unpack<uint16_t>(dx.read()),ref));
 // Every finite half encoding exercises RGBA clamp/half-integer rounding.
 H all;B expected;for(unsigned i=0;i<65536;++i){float v=decode(uint16_t(i));if(!std::isfinite(v))continue;all.push_back(uint16_t(i));expected.push_back(uint8_t(std::round((std::clamp(double(v),-1.,1.)*.5+.5)*255.)));}
 require(all.size()%4==0,"rgba fixture");Buffer da(a,bytes(all)),out(a,B(all.size(),0xcd));require(a.rgba(a.g,da.ptr(),I(all.size()/4),1,static_cast<uint8_t*>(out.ptr()))!=0,"RGBA oversized height rejected");
 // Reformat all finite values as a bounded 2D image, keeping every channel.
 require(all.size()%256==0,"rgba factor");require(a.rgba(a.g,da.ptr(),I(all.size()/256),64,static_cast<uint8_t*>(out.ptr()))==0,"rgba launch");require(out.read()==expected&&da.read()==bytes(all),"RGBA exact output/immutable input");std::cout<<"PASS RGBA allfinitehalf exact values="<<all.size()<<'\n';
}
static void softmax_case(Api&a,I m,I n){F x=values(m*n,83,4);float scale=.371f;std::vector<double> ref(m*n);for(I r=0;r<m;++r){double mx=-INFINITY;for(I j=0;j<n;++j)mx=std::max(mx,double(x[r*n+j])*scale);double sum=0;for(I j=0;j<n;++j){ref[r*n+j]=std::exp(double(x[r*n+j])*scale-mx);sum+=ref[r*n+j];}for(I j=0;j<n;++j)ref[r*n+j]/=sum;}Buffer dx(a,bytes(x)),out(a,bytes(H(m*n,0x7e00)));require(a.softmax(a.g,dx.fp(),m,n,scale,out.ptr())==0,"softmax launch");report("softmax n="+std::to_string(n),check_half(unpack<uint16_t>(out.read()),ref));require(dx.read()==bytes(x),"softmax input unchanged");}
static void spatial_case(Api&a,I h,I w,I ci,I co,int ft){
 H x=halves(values(h*w*ci,93,2)),base=halves(values(h*w*4*co,97,1));Buffer dx(a,bytes(x));
 H expected(h*w*4*ci);for(I y=0;y<2*h;++y)for(I xx=0;xx<2*w;++xx)for(I c=0;c<ci;++c)expected[(y*2*w+xx)*ci+c]=x[((y/2)*w+xx/2)*ci+c];Buffer up(a,bytes(H(expected.size(),0x7e00)));require(a.up(a.g,dx.ptr(),h,w,ci,up.ptr())==0,"up launch");require(unpack<uint16_t>(up.read())==expected,"up exact");
 expected.assign((h+1)*(w+1)*ci,0);for(I y=0;y<h;++y)for(I xx=0;xx<w;++xx)for(I c=0;c<ci;++c)expected[(y*(w+1)+xx)*ci+c]=x[(y*w+xx)*ci+c];Buffer pad(a,bytes(H(expected.size(),0x7e00)));require(a.pad(a.g,dx.ptr(),h,w,ci,pad.ptr())==0,"pad launch");require(unpack<uint16_t>(pad.read())==expected,"pad exact");
 // Independent shortcut oracle: expand channel repeats into [Co,ft,2,2], retain finaltime.
 I repeat=co*ft*4/ci;H expanded;for(I y=0;y<h;++y)for(I xx=0;xx<w;++xx)for(I c=0;c<ci;++c)for(I r=0;r<repeat;++r)expanded.push_back(x[(y*w+xx)*ci+c]);
 expected=base;for(I y=0;y<2*h;++y)for(I xx=0;xx<2*w;++xx)for(I c=0;c<co;++c){I d=(y*2*w+xx)*co+c,j=(((y/2)*w+xx/2)*co+c)*ft*4+(ft-1)*4+(y%2)*2+xx%2;expected[d]=encode(float(decode(base[d])+decode(expanded[j])));}Buffer dup(a,bytes(base));require(a.dup(a.g,dx.ptr(),h,w,ci,co,ft,dup.ptr())==0,"dup launch");require(unpack<uint16_t>(dup.read())==expected,"dup exact");
 // Down shortcut materializes [Ci,ft,2,2] then groups to Co before mean.
 H downbase=halves(values((h/2)*(w/2)*co,101,1));expected=downbase;I group=ci*ft*4/co;
 for(I y=0;y<h/2;++y)for(I xx=0;xx<w/2;++xx){std::vector<double> gather;for(I c=0;c<ci;++c)for(int t=0;t<ft;++t)for(int dy=0;dy<2;++dy)for(int dd=0;dd<2;++dd)gather.push_back(ft==2&&t==0?0:decode(x[((2*y+dy)*w+2*xx+dd)*ci+c]));for(I c=0;c<co;++c){double sum=0;for(I q=0;q<group;++q)sum+=gather[c*group+q];I i=(y*(w/2)+xx)*co+c;expected[i]=encode_double(double(decode(downbase[i]))+sum/double(group));}}
 Buffer down(a,bytes(downbase));require(a.down(a.g,dx.ptr(),h,w,ci,co,ft,down.ptr())==0,"down launch");require(unpack<uint16_t>(down.read())==expected,"down exact");require(dx.read()==bytes(x),"spatial input immutable");std::cout<<"PASS spatial4 h="<<h<<" w="<<w<<" ci="<<ci<<" co="<<co<<" ft="<<ft<<" exact\n";
}
static void cpu(){using namespace nrqi_vae;void*g=reinterpret_cast<void*>(uintptr_t(1));void*x=reinterpret_cast<void*>(uintptr_t(0x100000000));void*wt=reinterpret_cast<void*>(uintptr_t(0x200000000));void*out=reinterpret_cast<void*>(uintptr_t(0x300000000));auto*b=reinterpret_cast<float*>(uintptr_t(0x400000000));require(conv_valid(g,x,3,5,1152,wt,1152,3,1,b,out),"conv actual valid");require(conv_valid(g,x,1,1,3,wt,4,3,1,b,out),"same pad small valid");for(I d:{-1,0,4097})require(!conv_valid(g,x,d,5,3,wt,4,3,1,b,out),"badheight");require(!conv_valid(g,x,3,5,3,wt,4,2,1,b,out)&&!conv_valid(g,x,3,5,3,wt,4,3,3,b,out),"badkernelstride");require(!conv_valid(g,x,3,5,3,wt,4,3,1,b,x),"convalias");require(!conv_valid(nullptr,x,3,5,3,wt,4,3,1,b,out),"nullctx");for(unsigned h=0;h<65536;++h)if(std::isfinite(decode(uint16_t(h))))require(encode_double(double(decode(uint16_t(h))))==h,"half oracle");bool rejected=false;try{check_half(H{encode(.1f)},std::vector<double>{0});}catch(...){rejected=true;}require(rejected,"negative tolerance control");std::cout<<"CPU PASS conv geometry and independent half oracle\n";}
static void invalid(Api&a){Buffer x(a,bytes(H(8192,encode(1)))),wt(a,bytes(H(8192,encode(1)))),b(a,bytes(F(128,0))),out(a,bytes(H(8192,encode(2))));auto xb=x.read(),wb=wt.read(),bb=b.read(),ob=out.read();require(a.conv(a.g,x.ptr(),3,5,3,wt.ptr(),4,2,1,b.fp(),out.ptr())!=0,"badkernel");require(a.conv(a.g,x.ptr(),3,5,3,wt.ptr(),4,3,1,b.fp(),x.ptr())!=0,"alias");require(a.up(a.g,x.ptr(),2,2,3,x.ptr())!=0,"up alias");require(a.down(a.g,x.ptr(),3,2,3,3,1,out.ptr())!=0,"down odd");require(a.dup(a.g,x.ptr(),2,2,3,4,1,out.ptr())!=0,"dup indivisible");require(a.softmax(a.g,b.fp(),1,128,NAN,out.ptr())!=0,"badscale");require(a.silu(nullptr,x.ptr(),8)!=0,"silu nullctx");require(a.pad(a.g,x.ptr(),2,2,3,x.ptr())!=0,"pad alias");require(x.read()==xb&&wt.read()==wb&&b.read()==bb&&out.read()==ob,"invalid calls mutated storage");std::cout<<"PASS invalid device cases unchanged guards\n";}
int main(int argc,char**argv){try{std::cout<<std::setprecision(9);cpu();if(argc==2&&std::string(argv[1])=="--self-test")return 0;require(argc==2,"usage vae_numeric --self-test | PRIVATE_DSO");Api a(argv[1]);invalid(a);conv_case(a,1,1,3,4,3,1);conv_case(a,3,5,17,19,3,1);conv_case(a,5,7,17,19,3,2);conv_case(a,3,5,1152,19,3,1);conv_case(a,2,3,768,17,1,1);conv_case(a,3,5,64,1152,1,1);element_case(a);for(I n:{1,3,255,256,257,4096})softmax_case(a,3,n);spatial_case(a,2,4,3,3,1);spatial_case(a,4,6,4,2,2);spatial_case(a,2,2,96,192,1);std::cout<<"ALL PASS VAE operators; no whole-VAE/model claim\n";return 0;}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
