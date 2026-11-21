// Diagnostic shape microbenchmark; timings include launch and synchronization.
#define main numeric_validation_main
#include "native_numeric.cpp"
#undef main
#include <chrono>
int main(int argc,char** argv) {
 try {
  require(argc==2,"usage: performance LIBRARY");Api a(argv[1],0);
  for(I dim:{40,80,160,512}) {
   I tokens=dim==40?4096:dim==80?1024:dim==160?256:4096;
   I heads=dim==512?1:8,c=heads*dim;
   Buffer q(a,values(size_t(tokens*c))),k(a,values(size_t(tokens*c),9)),v(a,values(size_t(tokens*c),11)),y(a,F(size_t(tokens*c),sentinel));
   nrsd_attention_shape s{};s.batch=1;s.queries=tokens;s.keys=tokens;s.heads=heads;s.dim=dim;s.scale=1.f/std::sqrt(float(dim));
   s.qb=s.kb=s.vb=s.ob=tokens*c;s.qt=s.kt=s.vt=s.ot=c;s.qh=s.kh=s.vh=s.oh=dim;
   auto start=std::chrono::steady_clock::now();
   a.check(a.nrsd_attention(a.gpu,q.fp(),k.fp(),v.fp(),y.fp(),s),"attention");a.sync();
   std::cout<<"attention tokens="<<tokens<<" dim="<<dim<<" heads="<<heads<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
  }
  for(auto dims:std::vector<std::vector<I>>{{320,320,64,3},{640,640,32,3},{1280,1280,16,3},{128,128,256,3},{320,320,64,1}}) {
   I ci=dims[0],co=dims[1],h=dims[2],k=dims[3];
   nrsd_conv_shape s{1,ci,h,h,co,k,k,1,1,k/2,k/2,h,h};
   Buffer x(a,values(size_t(ci*h*h))),y(a,F(size_t(co*h*h),sentinel));
   Weight w(a,size_t(co*ci*k*k),1,31),b(a,size_t(co),0,9);
   for(int run=0;run<3;++run){
    auto start=std::chrono::steady_clock::now();
    a.check(a.nrsd_conv2d(a.gpu,x.fp(),w.raw(),b.raw(),y.fp(),s),"conv");a.sync();
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::cout<<"conv ci="<<ci<<" co="<<co<<" h="<<h<<" k="<<k<<" run="<<run<<" seconds="<<seconds<<std::endl;
   }
   y.guards();
  }
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
