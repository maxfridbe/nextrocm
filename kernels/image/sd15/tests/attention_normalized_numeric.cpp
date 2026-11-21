// Independent double-precision oracle and guarded dlopen buffers are shared
// with the existing standalone validator; no implementation header is included.
// Build/run commands are in attention_normalized_commands.md.
#define main sd15_existing_validator_main
#include "native_numeric.cpp"
#undef main

int main(int argc,char** argv) {
 try {
  require(argc==3,"usage: attention-normalized-numeric ABS_DIAGNOSTIC_LIBRARY GPU_INDEX");
  size_t used=0;int index=std::stoi(argv[2],&used);
  require(index>=0&&used==std::strlen(argv[2]),"invalid GPU index");
  Api a(argv[1],index);
  using Stats=int (*)(ns_gpu*,uint64_t*,uint64_t*);
  Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
  auto current=[&](){uint64_t live=0,peak=0;a.check(stats(a.gpu,&live,&peak),"allocation stats");return live;};
  int cases=0;
  auto check=[&](I queries,I keys,I heads,I dim,bool causal,bool mixed,bool extreme=false){
   uint64_t before=current();
   attention_case(a,queries,keys,heads,dim,causal,mixed,extreme);
   a.sync();require(current()==before,"attention leaked tracked allocations");++cases;
  };
  // Full tensors + padding + guards checked against independent CPU double math.
  // Fixed pre-existing attention thresholds: atol 4e-5, rtol 5e-4; report RMS.
  for(I dim:{1,7,40,64,80,160,257,512})for(bool mixed:{false,true}){
   check(5,77,2,dim,false,mixed);
   check(5,259,2,dim,true,mixed);
  }
  for(I keys:{1,255,256,257,511,512,513,4096}){
   check(3,keys,1,40,false,true);
   check(259,keys,1,7,true,false);
  }
  check(3,259,1,40,false,false,true); // Large logits, max changes after tile 0.
  check(77,77,2,64,true,true);       // CLIP causal self attention.
  check(257,257,2,160,false,false);
  // 2*2051 rows *4096 keys exceeds the 64 MiB cap: cross a chunk AND batch
  // boundary with a short final chunk, while keeping CPU oracle cost modest.
  check(2051,4096,1,1,false,true);
  check(2051,4096,1,1,true,false);
  {
   Buffer x(a,F(16,1.f)),out(a,F(16,sentinel));
   nrsd_attention_shape s{};
   s.batch=s.queries=s.heads=s.dim=1;s.keys=64*1024*1024/4+1;s.scale=1;
   s.qt=s.qh=s.qb=s.ot=s.oh=s.ob=1;
   s.kt=s.vt=1;s.kb=s.kh=s.vb=s.vh=s.keys;
   uint64_t before=current();
   int rc=a.nrsd_attention(a.gpu,x.fp(),x.fp(),x.fp(),out.fp(),s);
   require(rc!=0,"oversize row must be rejected before touching inputs");
   require(std::string(a.ns_last_error()).find("scratch cap")!=std::string::npos,
           "expected diagnostic scratch cap error (wrong build flag?)");
   require(current()==before,"cap rejection allocated scratch");
   compare("cap_rejection_output_untouched",out,D(16,sentinel));x.guards();
   s.keys=1;s.kb=s.kh=s.vb=s.vh=1;
   a.check(a.nrsd_attention(a.gpu,x.fp(),x.fp(),x.fp(),out.fp(),s),"recovery after rejection");
   D expected(16,sentinel);expected[0]=1;
   compare("recovery",out,expected);require(current()==before,"recovery scratch leak");
  }
  rejection_case(a);
  require(current()==0,"nonzero final tracked allocations");
  std::cout<<"PASS "<<cases<<" independent attention cases + cap/recovery/invalid checks; zero live bytes\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
