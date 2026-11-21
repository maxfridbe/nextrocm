// Replay captured actual inputs against independent double softmax/PV.
// Separate processes/libraries for each placement avoid cross-library symbols.
#define main sd15_existing_validator_main
#include "native_numeric.cpp"
#undef main
#include <cstdio>
#include <fstream>

int main(int argc,char** argv){
 try{
  require(argc==5,"usage: attention-capture-numeric LIBRARY GPU_INDEX CAPTURE NEW_OUTPUT_F32");
  std::ifstream file(argv[3],std::ios::binary);
  require(bool(file),"open capture");
  char magic[8];uint64_t m[7]{};float scale=0;
  file.read(magic,8);file.read(reinterpret_cast<char*>(m),sizeof(m));
  file.read(reinterpret_cast<char*>(&scale),sizeof(scale));
  require(bool(file)&&std::memcmp(magic,"NRQKV001",8)==0,"bad capture header");
  constexpr uint64_t cap=64*1024*1024/4;
  for(int i=0;i<5;++i)require(m[i]>0&&m[i]<=cap,"invalid captured dimension");
  require(m[4]<=512&&m[5]<=1&&std::isfinite(scale),"invalid captured semantics");
  auto count=[&](uint64_t tokens){
   uint64_t n=1;for(uint64_t d:{m[0],m[3],tokens,m[4]}){require(d<=cap/n,"capture size overflow/cap");n*=d;}return n;
  };
  uint64_t nq=count(m[1]),nk=count(m[2]);
  require(nq<=cap&&nk<=(cap-nq)/2,"capture payload exceeds cap");
  F allq(static_cast<size_t>(nq)),k(static_cast<size_t>(nk)),v(static_cast<size_t>(nk));
  for(F* x:{&allq,&k,&v}){file.read(reinterpret_cast<char*>(x->data()),std::streamsize(x->size()*4));require(bool(file),"truncated payload");for(float z:*x)require(std::isfinite(z),"nonfinite captured input");}
  require(file.peek()==std::ifstream::traits_type::eof(),"trailing capture payload");
  nrsd_attention_shape s{};
  s.batch=I(m[0]);s.queries=I(std::min(uint64_t(8),m[1]));s.keys=I(m[2]);s.heads=I(m[3]);s.dim=I(m[4]);s.causal=int32_t(m[5]);s.scale=scale;
  s.qt=s.kt=s.vt=s.ot=s.dim;
  s.qh=s.oh=s.queries*s.dim;s.kh=s.vh=s.keys*s.dim;
  s.qb=s.ob=s.heads*s.qh;s.kb=s.vb=s.heads*s.kh;
  F q(size_t(s.batch*s.qb));
  for(I b=0;b<s.batch;++b)for(I h=0;h<s.heads;++h)for(I t=0;t<s.queries;++t)for(I d=0;d<s.dim;++d)
   q[b*s.qb+h*s.qh+t*s.qt+d]=allq[((b*s.heads+h)*I(m[1])+t)*s.dim+d];
  D ref(q.size());
  for(I b=0;b<s.batch;++b)for(I h=0;h<s.heads;++h)for(I t=0;t<s.queries;++t){
   I end=s.causal?std::min(s.keys,t+1):s.keys;D prob(size_t(end),0.);
   for(I j=0;j<end;++j){double z=0;for(I d=0;d<s.dim;++d)z+=double(q[b*s.qb+h*s.qh+t*s.qt+d])*k[b*s.kb+h*s.kh+j*s.kt+d];prob[j]=z*double(s.scale);}
   double mx=*std::max_element(prob.begin(),prob.end()),sum=0;
   for(double& z:prob){z=std::exp(z-mx);sum+=z;}
   for(double& z:prob)z/=sum;
   for(I d=0;d<s.dim;++d){double z=0;for(I j=0;j<end;++j)z+=prob[j]*double(v[b*s.vb+h*s.vh+j*s.vt+d]);ref[b*s.ob+h*s.oh+t*s.ot+d]=z;}
  }
  size_t used=0;int index=std::stoi(argv[2],&used);
  require(index>=0&&used==std::strlen(argv[2]),"invalid GPU index");
  Api a(argv[1],index);
  using Stats=int (*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
  F actual;
  {
   Buffer dq(a,q),dk(a,k),dv(a,v),out(a,F(ref.size(),sentinel));
   uint64_t before=0,after=0,peak=0;a.check(stats(a.gpu,&before,&peak),"stats before");
   a.check(a.nrsd_attention(a.gpu,dq.fp(),dk.fp(),dv.fp(),out.fp(),s),"captured attention");a.sync();
   a.check(stats(a.gpu,&after,&peak),"stats after");require(before==after,"scratch allocation leak");
   actual=out.read();dq.guards();dk.guards();dv.guards();out.guards();
   // Retain output even on numerical failure. Exclusive creation, no overwrite.
   FILE* output=std::fopen(argv[4],"wbx");require(output!=nullptr,"cannot create NEW output");
   bool ok=std::fwrite(actual.data(),4,actual.size(),output)==actual.size();
   if(std::fclose(output))ok=false;
   if(!ok){std::remove(argv[4]);throw std::runtime_error("write output failed");}
   std::cout<<"capture_call="<<m[6]<<" original_queries="<<m[1]<<" tested_first_queries="<<s.queries
    <<" B="<<s.batch<<" H="<<s.heads<<" K="<<s.keys<<" D="<<s.dim<<" causal="<<s.causal<<" scale="<<scale<<'\n';
   compare("captured_actual_inputs",out,ref,4e-5,5e-4);
  }
  uint64_t live=0,peak=0;a.check(stats(a.gpu,&live,&peak),"final stats");require(live==0,"final tracked allocations");
  std::cout<<"PASS independent double oracle; zero live bytes; paired comparison requires BOTH runs\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
