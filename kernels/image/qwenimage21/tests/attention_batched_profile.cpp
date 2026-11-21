// Diagnostic subevents for accepted B256 only: unchanged complete output required.
#define main attention_batched_validation_entrypoint
#include "attention_batched_numeric.cpp"
#undef main
using Profile=int(*)(void*,const void*,const void*,const void*,I,I,I,I,I,I,void*,void*,I,double*,I*);
static void run_profile(Api&a){
 const Shape s{4096,4114,32,12288,12288};
 H q=halves(size_t(s.q*s.qs),7),k=halves(size_t(s.k*s.ks),19),v=halves(size_t(s.k*s.ks),29);
 Buffer dq(a,q.size()*2),dk(a,k.size()*2),dv(a,v.size()*2),plain(a,nout(s)*2),profiled(a,nout(s)*2),scratch(a,size_t(nrqi_attention_batched::workspace(s.q,s.k,s.h)));
 dq.write(q.data(),q.size()*2);dk.write(k.data(),k.size()*2);dv.write(v.data(),v.size()*2);
 H poison(nout(s),0x7e00);plain.write(poison.data(),poison.size()*2);profiled.write(poison.data(),poison.size()*2);
 Profile profile=nullptr;a.bind(profile,"nrqi_image_attention_batched_profile");
 a.check(a.nrqi_image_attention_batched(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),s.q,s.k,s.h,128,s.qs,s.ks,plain.ptr(),scratch.ptr(),I(scratch.bytes)),"plain accepted attention");a.sync();const F expected=plain.read_half();exact(expected,expected,"plain finite");
 double ms[3]={};I counts[3]={};
 std::cout<<"PROFILE_BEGIN shape=[4096,4114,32,128,12288,12288] expected_spans=48\n"<<std::flush;
 a.check(profile(a.gpu,dq.ptr(),dk.ptr(),dv.ptr(),s.q,s.k,s.h,128,s.qs,s.ks,profiled.ptr(),scratch.ptr(),I(scratch.bytes),ms,counts),"profile accepted attention");a.sync();
 exact(profiled.read_half(),expected,"profile complete output bitwise");
 for(int group=0;group<3;++group){require(counts[group]==16&&std::isfinite(ms[group])&&ms[group]>0,"profile coverage/elapsed");std::cout<<"GROUP name="<<(group==0?"QK":group==1?"softmax":"PV")<<" count="<<counts[group]<<" milliseconds="<<ms[group]<<'\n';}
 unchanged(a,dq,q);unchanged(a,dk,k);unchanged(a,dv,v);plain.guards();profiled.guards();scratch.guards();
 std::cout<<"PROFILE_OUTPUT_BITWISE outputs="<<nout(s)<<" event_sum_ms="<<ms[0]+ms[1]+ms[2]<<" new_independent_numeric_claim=0\n";
}
int main(int argc,char**argv){try{
 self_test();batched_self_test();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
 require(argc==3,"usage: attention_batched_profile ABS_DIAGNOSTIC_DSO GPU_INDEX");require(argv[1][0]=='/',"absolute DSO");const char* env=std::getenv("NS_PROFILE");require(env&&std::string(env)=="gpu","NS_PROFILE=gpu required");
 size_t used=0;int index=std::stoi(argv[2],&used);require(index>=0&&used==std::strlen(argv[2]),"GPU index");Api a(argv[1],index);uint64_t live=0,peak=0;
 a.check(a.nr_allocation_stats(a.gpu,&live,&peak),"initial stats");require(live==0,"initial live");std::cout<<std::setprecision(9);run_profile(a);
 a.check(a.nr_allocation_stats(a.gpu,&live,&peak),"final stats");require(live==0&&peak<512*1024*1024,"profile bound/drop");std::cout<<"tracked_final="<<live<<" tracked_peak="<<peak<<"\nPROFILE PASS (synthetic-input attribution, not throughput or new numerical admission)\n";return 0;
}catch(const std::exception&e){std::cerr<<"PROFILE FAIL "<<e.what()<<'\n';return 1;}}
