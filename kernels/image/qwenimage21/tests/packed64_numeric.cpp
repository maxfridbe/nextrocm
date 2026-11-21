// Separate paired candidate validator; frozen v1 source remains unchanged.
#define main packed_slice_entry
#include "packed_numeric.cpp"
#undef main
static void run64(Api& a,Packed original,Packed fn,int type,I m,I k,I n,int odt,I stride,
                const std::vector<uint8_t>& packed,int repeats) {
    require(m*k+n*k+m*stride < 100000000,"bounded validator elements");
    auto wf=decode(packed,type,k,n);std::vector<uint16_t> hw(wf.size()),hx(size_t(m*k));
    for(size_t i=0;i<hw.size();++i)hw[i]=qi_oracle::half_bits(wf[i]);
    uint32_t seed=42;F xf(hx.size());for(size_t i=0;i<hx.size();++i){hx[i]=uint16_t((rnd(seed)&0x83ff)|0x3000);xf[i]=qi_oracle::half_float(hx[i]);}
    const size_t es=odt?2:4;
    Buffer x(a,hx.size()*2),w(a,packed.size()),expanded(a,hw.size()*2),out(a,size_t(m*stride)*es),base(a,size_t(m*n)*es),prior(a,size_t(m*n)*es);
    x.write(hx.data(),x.bytes);w.write(packed.data(),w.bytes);expanded.write(hw.data(),expanded.bytes);
    std::vector<uint8_t> initial(out.bytes,0x5a);out.write(initial.data(),initial.size());
    auto candidate=[&]{a.check(fn(a.gpu,x.ptr(),m,k,w.ptr(),type,n,out.ptr(),odt,stride),"packed linear");};
    auto baseline=[&]{a.check(a.nsd_linear(a.gpu,x.ptr(),1,m,k,expanded.ptr(),n,nullptr,base.ptr(),odt),"half baseline");};
    auto previous=[&]{a.check(original(a.gpu,x.ptr(),m,k,w.ptr(),type,n,prior.ptr(),odt,n),"packed16");};
    baseline();previous();candidate();a.sync();auto y=bytes(a,out),ref=bytes(a,base);
    require(bytes(a,prior)==ref,"validated packed16 differs from half baseline");
    double worst=0;
    for(I r=0;r<m;++r)for(I c=0;c<stride;++c) {
        const size_t at=size_t(r*stride+c)*es;
        if(c>=n){require(std::memcmp(y.data()+at,initial.data()+at,es)==0,"stride hole modified");continue;}
        require(std::memcmp(y.data()+at,ref.data()+size_t(r*n+c)*es,es)==0,"full baseline bitwise mismatch");
        float value;if(odt){uint16_t h;std::memcpy(&h,y.data()+at,2);value=qi_oracle::half_float(h);}else std::memcpy(&value,y.data()+at,4);
        require(std::isfinite(value),"nonfinite output");
        double dot=0,absum=0;for(I j=0;j<k;++j){double v=double(xf[size_t(r*k+j)])*wf[size_t(c*k+j)];dot+=v;absum+=std::abs(v);}
        const double eps=std::ldexp(1.,-24),gamma=(k*eps)/(1-k*eps);
        const double bound=gamma*absum+(odt?std::abs(dot)*std::ldexp(1.,-10)+std::ldexp(1.,-24):0)+1e-12;
        require(std::abs(double(value)-dot)<=bound,"independent F64 forward-error bound failed");
        worst=std::max(worst,std::abs(double(value)-dot));
    }
    x.guards();w.guards();expanded.guards();out.guards();base.guards();prior.guards();
    candidate();a.sync();require(bytes(a,out)==y,"repeat mismatch");
    double bt=0,ct=0;for(int i=0;i<repeats;++i){if(i%2){ct+=timed(a,candidate);bt+=timed(a,previous);}else{bt+=timed(a,previous);ct+=timed(a,candidate);}}
    std::cout<<"PASS type="<<type<<" M="<<m<<" K="<<k<<" N="<<n<<" odt="<<odt<<" stride="<<stride
             <<" maxF64="<<worst<<" packed16_ms="<<bt/repeats<<" packed64_ms="<<ct/repeats<<"\n";
}
int main(int argc,char** argv)try {
    if(argc==2&&std::string(argv[1])=="--self-test"){cpu_tests();return 0;}
    require(argc==4||argc==10,"usage: packed64_numeric ABS_RUNTIME GPU REPEATS [TYPE K N FILE M ODT]");
    require(argv[1][0]=='/',"absolute runtime required");require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","NS_PROFILE=gpu required");
    const int reps=std::stoi(argv[3]);require(reps>=1&&reps<=16,"bounded repetitions");
    Api a(argv[1],std::stoi(argv[2]));Packed previous=nullptr,fn=nullptr;a.bind(previous,"nrqi_packed_linear");a.bind(fn,"nrqi_packed_linear64");
    using Stats=int(*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
    uint64_t initial=0,peak=0;a.check(stats(a.gpu,&initial,&peak),"initial stats");require(initial==0,"initial allocation nonzero");
    cpu_tests();invalid(a,previous);invalid(a,fn);
    if(argc==10){const int type=std::stoi(argv[4]),odt=std::stoi(argv[9]);const I k=std::stoll(argv[5]),n=std::stoll(argv[6]),m=std::stoll(argv[8]);
        require(m==1||m==16||m==64||m==256,"bounded representative M");require(odt==0||odt==1,"output type");
        require((k==4096&&(n==4096||n==12288))||(k==12288&&n==4096),"actual block geometry");
        require(type==12||(type==14&&k==4096&&n==4096),"actual block quantization");
        run64(a,previous,fn,type,m,k,n,odt,n+7,fixture(argv[7],type,k,n),reps);
    }else for(int type:{12,14})for(I m:{I(1),I(17),I(65)})for(I n:{I(1),I(19),I(65)})for(int odt:{0,1})
        run64(a,previous,fn,type,m,256,n,odt,n+7,synthetic(type,256,n),reps);
    a.sync();uint64_t live=0;a.check(stats(a.gpu,&live,&peak),"drop stats");require(live==0&&peak<512ull*1024*1024,"allocation leak/cap");
    std::cout<<"tracked_initial="<<initial<<" tracked_final="<<live<<" tracked_peak="<<peak<<"\nALL PACKED64 PASS (operator only)\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
