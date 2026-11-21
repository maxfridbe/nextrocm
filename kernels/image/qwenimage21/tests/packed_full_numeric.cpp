// Full-N operator extension; preserve the frozen slice validator and kernel.
// Each process validates every output/F64 dot for exactly one explicit M/odt.
#define main packed_slice_entry
#include "packed_numeric.cpp"
#undef main
int main(int argc,char** argv)try {
    require(argc==10,"usage: packed_full_numeric ABS_RUNTIME GPU REPEATS TYPE K N PACKED_FILE M ODT");
    require(argv[1][0]=='/',"absolute runtime required");
    require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","NS_PROFILE=gpu required");
    const int reps=std::stoi(argv[3]),type=std::stoi(argv[4]),odt=std::stoi(argv[9]);
    const I k=std::stoll(argv[5]),n=std::stoll(argv[6]),m=std::stoll(argv[8]);
    require(reps>=1&&reps<=16&&(odt==0||odt==1),"bounded repeats/output type");
    require(m==1||m==16||m==64||m==256,"bounded representative M");
    require((k==4096&&(n==4096||n==12288))||(k==12288&&n==4096),"block full geometry");
    require(type==12||(type==14&&k==4096&&n==4096),"actual block quantization");
    const auto b=fixture(argv[7],type,k,n);
    Api a(argv[1],std::stoi(argv[2]));Packed fn=nullptr;a.bind(fn,"nrqi_packed_linear");
    using Stats=int(*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
    uint64_t initial=0,peak=0;a.check(stats(a.gpu,&initial,&peak),"initial stats");require(initial==0,"expected empty test allocator");
    cpu_tests();invalid(a,fn);
    run(a,fn,type,m,k,n,odt,n+7,b,reps);
    a.sync();uint64_t live=0;a.check(stats(a.gpu,&live,&peak),"drop stats");
    require(live==initial&&peak<512ull*1024*1024,"allocation leak or operator cap512MiB");
    std::cout<<"tracked_initial="<<initial<<" tracked_final="<<live<<" tracked_peak="<<peak<<"\n";
    std::cout<<"FULL SHAPE PASS (one operator; no model admission)\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
