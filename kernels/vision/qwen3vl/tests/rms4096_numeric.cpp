// Standalone independent F64 + pinned actual CPU bitwise operator validator.
#define main ordered_validator_main
#include "rms_rope_ordered_numeric.cpp"
#undef main
#include "../rms4096_strict.hpp"
using Rms=int(*)(void*,const float*,I,const float*,float*);
static std::vector<double> rms_oracle(const F&x,const F&w){
 std::vector<double> y(x.size());for(size_t r=0;r<x.size()/4096;++r){double sum=0;for(size_t j=0;j<4096;++j){const double v=x[r*4096+j];sum+=v*v;}const double inv=1/std::sqrt(sum/4096+double(1e-6f));for(size_t j=0;j<4096;++j)y[r*4096+j]=double(x[r*4096+j])*inv*w[j];}return y;
}
static void host_rms(){
 void*g=reinterpret_cast<void*>(uintptr_t(1));auto*x=reinterpret_cast<float*>(uintptr_t(0x10000000));auto*y=reinterpret_cast<float*>(uintptr_t(0x30000000));auto*w=reinterpret_cast<float*>(uintptr_t(0x50000000));
 for(I n:{1,3,32,257,4096})req(qv_rms4096::valid(g,x,n,w,y),"host valid");
 for(I n:{-1,0,4097})req(!qv_rms4096::valid(g,x,n,w,y),"row guard");
 req(!qv_rms4096::valid(nullptr,x,1,w,y)&&!qv_rms4096::valid(g,nullptr,1,w,y)&&!qv_rms4096::valid(g,x,1,nullptr,y)&&!qv_rms4096::valid(g,x,1,w,nullptr),"null guards");
 req(!qv_rms4096::valid(g,x,1,w,x)&&!qv_rms4096::valid(g,x,1,w,x+1)&&!qv_rms4096::valid(g,x,1,w,w+1),"write alias");
 req(qv_rms4096::valid(g,x,1,x,y),"read alias allowed");
 for(auto bad:{reinterpret_cast<float*>(uintptr_t(x)+1),reinterpret_cast<float*>(UINTPTR_MAX-3)}){req(!qv_rms4096::valid(g,bad,1,w,y)&&!qv_rms4096::valid(g,x,1,bad,y)&&!qv_rms4096::valid(g,x,1,w,bad),"alignment/wrap");}
 F zero(4096,0),ones(4096,1);auto z=rms_oracle(zero,ones);for(double v:z)req(v==0,"zero oracle");
 bool rejected=false;try{compare(F{1},std::vector<double>{0});}catch(const std::exception&){rejected=true;}req(rejected,"oracle negative");std::cout<<"CPU PASS RMS4096 geometry/spans/independent oracle\n";
}
static F execute(Api&a,Rms f,const F&x,const F&w,I rows){
 Buffer dx(a,x),dw(a,w),dy(a,F(x.size(),NAN));req(f(a.g,dx.ptr(),rows,dw.ptr(),dy.ptr())==0,"RMS launch");F y=dy.read(),xx=dx.read(),ww=dw.read();req(std::memcmp(xx.data(),x.data(),x.size()*4)==0&&std::memcmp(ww.data(),w.data(),w.size()*4)==0,"readonly inputs");return y;
}
static void invalid_rms(Api&a,Rms f){F x(4096,1),w(4096,2),y(4096,3);Buffer dx(a,x),dw(a,w),dy(a,y);
 for(I n:{-1,0,4097})req(f(a.g,dx.ptr(),n,dw.ptr(),dy.ptr())!=0,"row rejection");
 req(f(nullptr,dx.ptr(),1,dw.ptr(),dy.ptr())!=0&&f(a.g,nullptr,1,dw.ptr(),dy.ptr())!=0&&f(a.g,dx.ptr(),1,nullptr,dy.ptr())!=0&&f(a.g,dx.ptr(),1,dw.ptr(),nullptr)!=0,"null rejection");
 req(f(a.g,dx.ptr(),1,dw.ptr(),dx.ptr())!=0&&f(a.g,dx.ptr(),1,dw.ptr(),dx.ptr()+1)!=0&&f(a.g,dx.ptr(),1,dw.ptr(),dw.ptr()+1)!=0,"alias rejection");
 for(auto bad:{reinterpret_cast<float*>(uintptr_t(dx.ptr())+1),reinterpret_cast<float*>(UINTPTR_MAX-3)})req(f(a.g,bad,1,dw.ptr(),dy.ptr())!=0&&f(a.g,dx.ptr(),1,bad,dy.ptr())!=0&&f(a.g,dx.ptr(),1,dw.ptr(),bad)!=0,"unaligned/wrap rejection");
 req(dx.read()==x&&dw.read()==w&&dy.read()==y,"invalid mutation");
}
int main(int argc,char**argv){try{
 host_rms();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 req(argc==3,"usage: rms4096_numeric PRIVATE_DSO PINNED_FIXTURES");Api a(argv[1]);Rms f=nullptr;a.bind(f,"nrqv_rms4096_strict_f32");invalid_rms(a,f);
 for(I rows:{1,3,31,32,33,65,257}){F x=values(size_t(rows)*4096),w=values(4096);for(float&v:w)v+=2;const auto y=execute(a,f,x,w,rows);double e=compare(y,rms_oracle(x,w));std::cout<<"PASS F64 RMS rows="<<rows<<" max_abs="<<e<<'\n';}
 for(int mode=0;mode<5;++mode){F x(4096,mode==0?-0.f:mode==1?1e20f:mode==4?std::numeric_limits<float>::denorm_min():1.f),w(4096,1.f);if(mode==2)x[0]=INFINITY;if(mode==3)x[0]=NAN;F y=execute(a,f,x,w,1);
  for(size_t j=0;j<y.size();++j){if(mode==3||(mode==2&&j==0))req(std::isnan(y[j]),"NaN category");else if(mode==4)req(y[j]==1000.f*std::numeric_limits<float>::denorm_min(),"subnormal");else{req(y[j]==0,"zero/overflow");if(mode==0)req(std::signbit(y[j]),"signedzero");}}
  std::cout<<"PASS special RMS mode="<<mode<<'\n';
 }
 for(const std::string name:{"layer0_input","layer0_post","block1_primary_input","block1_primary_post","block1_native_input","block1_native_post"}){
  std::string p=std::string(argv[2])+"/"+name;F x=read_raw(p+".input.f32",32*4096),w=read_raw(p+".weight.f32",4096),expected=read_raw(p+".expected.f32",32*4096);F y=execute(a,f,x,w,32);compare(y,rms_oracle(x,w));req(std::memcmp(y.data(),expected.data(),y.size()*4)==0,"actual full CPU bits differ: "+name);std::cout<<"PASS actual RMS "<<name<<" bitwise="<<y.size()<<'\n';
 }
 std::cout<<"GPU PASS RMS4096 7F64+5special+6actual; no model admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
