// Timing-only companion to the frozen v1 correctness validator.
// Reuse its exact deterministic inputs, half conversion, guards, and loader.
// No baseline launch and no new independent numerical-parity claim here.
#define main attention_1024_validation_entrypoint
#include "attention_1024_numeric.cpp"
#undef main

static void timing_only(Api& a) {
 const Shape s{4096,4114,32,12288,12288};
 H q=halves(size_t(s.q*s.qs),7),k=halves(size_t(s.k*s.ks),19),v=halves(size_t(s.k*s.ks),29);
 Buffer dq(a,q.size()*2),dk(a,k.size()*2),dv(a,v.size()*2),out(a,nout(s)*2);
 dq.write(q.data(),q.size()*2);dk.write(k.data(),k.size()*2);dv.write(v.data(),v.size()*2);
 H poison(nout(s),0x7e00);out.write(poison.data(),poison.size()*2);
 auto candidate=[&]{a.check(a.nrqi_image_attention_1024(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),s.q,s.k,s.h,128,s.qs,s.ks,out.ptr()),"timing-only candidate");};
 std::cout<<"TIMING_ONLY initial_candidate_begin shape=[4096,4114,32,128,12288,12288] seeds=[7,19,29]\n"<<std::flush;
 int64_t initial_start=-1,initial_end=-1;double initial_ns=-1;
 a.check(a.ns_stamp(a.gpu,&initial_start),"initial start stamp");candidate();a.check(a.ns_stamp(a.gpu,&initial_end),"initial end stamp");
 std::cout<<"TIMING_ONLY initial_candidate_enqueued sync_begin\n"<<std::flush;
 a.sync();std::cout<<"TIMING_ONLY initial_candidate_sync_complete\n"<<std::flush;
 a.check(a.ns_elapsed(a.gpu,initial_start,initial_end,&initial_ns),"initial elapsed");require(std::isfinite(initial_ns)&&initial_ns>0,"initial finite timing");
 std::cout<<"TIMING_ONLY initial_candidate_ms="<<initial_ns/1e6<<" initial_only_not_median=1\n"<<std::flush;
 const F expected=out.read_half();exact(expected,expected,"initial complete output finite");
 std::cout<<"TIMING_ONLY initial_candidate_complete outputs="<<nout(s)<<" baseline_parity_claim=0 independent_F64_claim=0\n"<<std::flush;
 unchanged(a,dq,q);unchanged(a,dk,k);unchanged(a,dv,v);out.guards();
 std::vector<double> elapsed;
 for(int i=0;i<3;++i){
  std::cout<<"TIMING_ONLY candidate_begin repeat="<<i<<'\n'<<std::flush;
  int64_t start=-1,end=-1;double ns=-1;
  a.check(a.ns_stamp(a.gpu,&start),"start stamp");candidate();a.check(a.ns_stamp(a.gpu,&end),"end stamp");
  std::cout<<"TIMING_ONLY candidate_enqueued sync_begin repeat="<<i<<'\n'<<std::flush;
  a.sync();std::cout<<"TIMING_ONLY candidate_sync_complete repeat="<<i<<'\n'<<std::flush;
  a.check(a.ns_elapsed(a.gpu,start,end,&ns),"candidate elapsed");
  require(std::isfinite(ns)&&ns>0,"positive finite timing");elapsed.push_back(ns/1e6);
  exact(out.read_half(),expected,"complete timed repeat bitwise");out.guards();
  std::cout<<"TIMING_ONLY repeat="<<i<<" candidate_ms="<<elapsed.back()<<" repeat_bitwise=1\n"<<std::flush;
 }
 unchanged(a,dq,q);unchanged(a,dk,k);unchanged(a,dv,v);out.guards();
 std::sort(elapsed.begin(),elapsed.end());
 std::cout<<"TIMING_ONLY median_candidate_ms="<<elapsed[1]<<" baseline_launches=0 timed_repeats=3\n"<<std::flush;
}
int main(int argc,char**argv){try{
 self_test();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 require(argc==3,"usage: attention_1024_timing ABS_RUNTIME GPU_INDEX (NS_PROFILE=gpu)");
 require(argv[1][0]=='/',"absolute runtime");const char* profile=std::getenv("NS_PROFILE");require(profile&&std::string(profile)=="gpu","NS_PROFILE=gpu required");
 size_t used=0;int index=std::stoi(argv[2],&used);require(index>=0&&used==std::strlen(argv[2]),"GPU index");
 Api a(argv[1],index);uint64_t live=0,peak=0;a.check(a.nr_allocation_stats(a.gpu,&live,&peak),"initial stats");require(live==0,"initial allocation zero");
 std::cout<<std::setprecision(9);timing_only(a);invalid(a);
 a.check(a.nr_allocation_stats(a.gpu,&live,&peak),"final stats");require(live==0&&peak<512*1024*1024,"timing-only allocation bound/drop");
 std::cout<<"tracked_final="<<live<<" tracked_peak="<<peak<<"\nTIMING_ONLY PASS full_shape=1 repeat_stable=1 baseline_parity_claim=0 independent_F64_claim=0\n"<<std::flush;
 return 0;
}catch(const std::exception& e){std::cerr<<"TIMING_ONLY FAIL "<<e.what()<<'\n';return 1;}}
