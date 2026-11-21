// New validator; frozen original oracle/validator remain unchanged.
#define main frozen_vae_operator_main
#include "vae_numeric.cpp"
#undef main
#include <cstdlib>
using Conv=decltype(&nrqi_vae_conv);
static void ownership(){
 for(I m:{1,15,16,17,63,64,65,127,128,129})for(I n:{1,4,15,16,17,63,64,65}){
  std::vector<int> hits(size_t(m*n));
  for(I p0=0;p0<m;p0+=64)for(I c0=0;c0<n;c0+=64)for(int t=0;t<256;++t)for(int a=0;a<4;++a)for(int b=0;b<4;++b){I p=p0+t/16+16*a,c=c0+t%16+16*b;if(p<m&&c<n)++hits[size_t(p*n+c)];}
  require(std::all_of(hits.begin(),hits.end(),[](int n){return n==1;}),"conv64 output ownership");
 }
 std::cout<<"CPU PASS80 conv64 ownership cases\n";
}
static void negatives(Api&a,Conv fn){
 Buffer x(a,bytes(H(8192,encode(1)))),w(a,bytes(H(8192,encode(1)))),b(a,bytes(F(128,0))),y(a,bytes(H(8192,encode(2))));auto xx=x.read(),ww=w.read(),bb=b.read(),yy=y.read();
 auto call=[&](void*g,const void*in,I h,I width,I ci,const void*wt,I co,int k,int s,const float*bias,void*out){require(fn(g,in,h,width,ci,wt,co,k,s,bias,out)!=0,"invalid conv64 accepted");};
 call(nullptr,x.ptr(),3,5,3,w.ptr(),4,3,1,b.fp(),y.ptr());
 call(a.g,nullptr,3,5,3,w.ptr(),4,3,1,b.fp(),y.ptr());
 call(a.g,x.ptr(),0,5,3,w.ptr(),4,3,1,b.fp(),y.ptr());
 call(a.g,x.ptr(),3,5,3,w.ptr(),4,2,1,b.fp(),y.ptr());
 call(a.g,x.ptr(),3,5,3,w.ptr(),4,3,3,b.fp(),y.ptr());
 call(a.g,x.ptr(),2,5,3,w.ptr(),4,3,2,b.fp(),y.ptr());
 call(a.g,x.ptr(),3,5,3,w.ptr(),4,3,1,b.fp(),x.ptr());
 call(a.g,x.ptr(),3,5,3,w.ptr(),4,3,1,b.fp(),static_cast<char*>(x.ptr())+2);
 call(a.g,x.ptr(),3,5,3,w.ptr(),4,3,1,b.fp(),w.ptr());
 call(a.g,x.ptr(),3,5,3,w.ptr(),4,3,1,b.fp(),b.ptr());
 call(a.g,static_cast<char*>(x.ptr())+1,3,5,3,w.ptr(),4,3,1,b.fp(),y.ptr());
 call(a.g,reinterpret_cast<void*>(UINTPTR_MAX-1),3,5,3,w.ptr(),4,3,1,b.fp(),y.ptr());
 require(x.read()==xx&&w.read()==ww&&b.read()==bb&&y.read()==yy,"invalid modified input/output");
 std::cout<<"PASS12 invalid conv64 no-write cases\n";
}
static void run(Api&a,Conv fn,I h,I w,I ci,I co,int k,int stride,bool full,bool bias_on){
 require(h>=1&&h<=1024&&w>=1&&w<=1024&&ci>=1&&ci<=1152&&co>=1&&co<=1152,"bounded case geometry");
 require((k==1||k==3)&&(stride==1||stride==2)&&(stride==1||(h>=k&&w>=k)),"case kernel/stride");
 const I oh=stride==1?h:(h-k)/2+1,ow=stride==1?w:(w-k)/2+1;
 require(2*(h*w*ci+co*ci*k*k+2*oh*ow*co)+co*4 < 3LL*1024*1024*1024,"bounded device buffers below3GiB");
 H x=halves(values(size_t(h*w*ci),31,.5f)),wt=halves(values(size_t(co*ci*k*k),53,.125f));F bias=bias_on?values(size_t(co),61,.25f):F(size_t(co),0);
 Buffer dx(a,bytes(x)),dw(a,bytes(wt)),db(a,bytes(bias)),base(a,bytes(H(size_t(oh*ow*co),0x7e00))),out(a,bytes(H(size_t(oh*ow*co),0x7e00)));
 auto invoke=[&](Conv f,Buffer&y){require(f(a.g,dx.ptr(),h,w,ci,dw.ptr(),co,k,stride,bias_on?db.fp():nullptr,y.ptr())==0,"conv launch");};
 invoke(a.conv,base);invoke(fn,out);B expected=base.read(),got=out.read();require(got==expected,"all baseline output bits");H actual=unpack<uint16_t>(got);for(auto v:actual)require(std::isfinite(decode(v)),"full output finite");
 if(full)report("conv64 full independent F64",check_half(actual,conv_ref(x,wt,bias,h,w,ci,co,k,stride)));
 else {
  // Explicit bounded samples only; all outputs already compared bitwise above.
  H selected;std::vector<double> ref;
  for(I p:{I(0),std::min(I(1),oh*ow-1),oh*ow/2,oh*ow-1})for(I c:{I(0),co/2,co-1}){
   double sum=bias[size_t(c)];I pad=stride==1?k/2:0;
   for(I ch=0;ch<ci;++ch)for(int yy=0;yy<k;++yy)for(int xx=0;xx<k;++xx){I iy=(p/ow)*stride-pad+yy,ix=(p%ow)*stride-pad+xx;if(iy>=0&&iy<h&&ix>=0&&ix<w)sum+=double(decode(x[size_t((iy*w+ix)*ci+ch)]))*decode(wt[size_t(((c*ci+ch)*k+yy)*k+xx)]);}
   selected.push_back(actual[size_t(p*co+c)]);ref.push_back(sum);
  }
  report("conv64 sampled12 independent F64 (not full)",check_half(selected,ref));
 }
 invoke(fn,out);require(out.read()==got,"repeat baseline bits");
 int(*stamp)(void*,int64_t*)=nullptr;int(*elapsed)(void*,int64_t,int64_t,double*)=nullptr;a.bind(stamp,"ns_stamp");a.bind(elapsed,"ns_elapsed");
 auto time=[&](Conv f,Buffer&y){int64_t s=-1,e=-1;double ns=-1;require(stamp(a.g,&s)==0,"stamp start");invoke(f,y);require(stamp(a.g,&e)==0&&elapsed(a.g,s,e,&ns)==0&&std::isfinite(ns)&&ns>0,"elapsed");return ns/1e6;};
 double bt[2],ct[2];bt[0]=time(a.conv,base);ct[0]=time(fn,out);ct[1]=time(fn,out);bt[1]=time(a.conv,base);
 require(out.read()==got&&base.read()==expected,"timed output changed");require(dx.read()==bytes(x)&&dw.read()==bytes(wt)&&db.read()==bytes(bias),"inputs unchanged");
 std::cout<<"PASS conv64 h="<<h<<" w="<<w<<" ci="<<ci<<" co="<<co<<" k="<<k<<" stride="<<stride<<" bias="<<bias_on<<" full_baseline_bits=1 full_F64="<<full<<" baseline_ms="<<bt[0]<<","<<bt[1]<<" candidate_ms="<<ct[0]<<","<<ct[1]<<"\n";
}
int main(int argc,char**argv)try{
 cpu();ownership();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 require(argc==2||argc==8,"usage: vae_conv64_numeric DSO [H W CI CO K STRIDE]");require(argv[1][0]=='/',"absolute DSO");require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","NS_PROFILE=gpu required");
 Api a(argv[1]);
 using Stats=int(*)(void*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
 uint64_t initial=0,peak=0;require(stats(a.g,&initial,&peak)==0&&initial==0,"initial tracked allocation must be zero");
 Conv candidate=nullptr;a.bind(candidate,"nrqi_vae_conv64");negatives(a,candidate);
 if(argc==8)run(a,candidate,std::stoll(argv[2]),std::stoll(argv[3]),std::stoll(argv[4]),std::stoll(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),false,true);
 else for(auto shape:std::vector<std::vector<I>>{{1,1,4,4,1,1},{1,1,3,4,3,1},{3,5,17,19,3,1},{5,7,17,19,3,2},{9,9,17,65,1,1},{3,5,1152,19,3,1},{3,5,64,1152,1,1}})for(bool bias:{false,true})run(a,candidate,shape[0],shape[1],shape[2],shape[3],int(shape[4]),int(shape[5]),true,bias);
 uint64_t live=0;require(a.sync(a.g)==0&&stats(a.g,&live,&peak)==0&&live==0&&peak<3ull*1024*1024*1024+5*2*Buffer::guard,"tracked live/drop/peak cap");
 std::cout<<"tracked_initial="<<initial<<" tracked_final="<<live<<" tracked_peak="<<peak<<"\nALL conv64 operator PASS; no model promotion\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
