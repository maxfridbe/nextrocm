// Auxiliary F32 path validation. Reuse test-only loader/buffers and pinned GGML.
#define main packed_slice_entry
#include "packed_numeric.cpp"
#undef main
#include "../aux_f32.hpp"
using Aux=int(*)(void*,const float*,I,I,const void*,int,I,float*,I);
static F decode_f32(const std::vector<uint8_t>& b,I k,I n) {
    F f(size_t(k*n));std::vector<qi_oracle::block_q4_K> blocks(b.size()/144);
    std::memcpy(blocks.data(),b.data(),b.size());qi_oracle::dequantize_row_q4_K(blocks.data(),f.data(),k*n);
    for(float v:f) require(std::isfinite(v),"nonfinite F32 decoded weight");
    return f;
}
static void aux_cpu_tests() {
    cpu_tests();const auto b=synthetic(12,256,1);const auto full=decode_f32(b,256,1),half=decode(b,12,256,1);
    require(full!=half,"fixture must distinguish F32 and half-rounded weights");
    std::vector<float>x(1024),y(1024);std::vector<uint16_t>w(1024);
    require(nrqi_aux_f32::valid(x.data(),x.data(),1,256,w.data(),12,1,y.data(),1),"F32 valid");
    require(!nrqi_aux_f32::valid(x.data(),x.data(),1,256,w.data(),14,1,y.data(),1),"aux rejects Q6");
    require(!nrqi_aux_f32::valid(x.data(),reinterpret_cast<char*>(x.data())+2,1,256,w.data(),12,1,y.data(),1),"F32 alignment");
    std::cout<<"Auxiliary F32 precision distinction/guards PASS\n";
}
static void aux_invalid(Api& a,Aux fn) {
    Buffer x(a,2048),w(a,2048),y(a,2048);const auto original=bytes(a,y);
    auto call=[&](void* g,const void* xp,I m,I k,const void* wp,int type,I n,void* yp,I stride){
        require(fn(g,static_cast<const float*>(xp),m,k,wp,type,n,static_cast<float*>(yp),stride)!=0,"aux invalid accepted");
        a.sync();require(bytes(a,y)==original,"aux invalid wrote output");};
    call(nullptr,x.ptr(),1,256,w.ptr(),12,1,y.ptr(),1);
    call(a.gpu,nullptr,1,256,w.ptr(),12,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,nullptr,12,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,w.ptr(),12,1,nullptr,1);
    call(a.gpu,x.ptr(),0,256,w.ptr(),12,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,255,w.ptr(),12,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,w.ptr(),14,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,w.ptr(),12,2,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,w.ptr(),12,1,x.ptr(),1);
    call(a.gpu,x.ptr(),1,256,w.ptr(),12,1,w.ptr(),1);
    call(a.gpu,static_cast<char*>(x.ptr())+2,1,256,w.ptr(),12,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,static_cast<char*>(w.ptr())+1,12,1,y.ptr(),1);
    call(a.gpu,x.ptr(),1,256,w.ptr(),12,1,static_cast<char*>(y.ptr())+2,1);
    call(a.gpu,reinterpret_cast<void*>(UINTPTR_MAX-3),1,256,w.ptr(),12,1,y.ptr(),1);
    x.guards();w.guards();y.guards();std::cout<<"14 auxiliary invalid cases PASS\n";
}
static void aux_run(Api& a,Aux fn,I m,I k,I n,const std::vector<uint8_t>& packed,int reps) {
    const I stride=n+7;const auto wf=decode_f32(packed,k,n);F xf(size_t(m*k));uint32_t seed=42;
    // General finite F32 inputs, deliberately retain bits lost by half conversion.
    for(float& v:xf){uint32_t bits=(rnd(seed)&0x807fffffu)|0x3e000000u;std::memcpy(&v,&bits,4);}
    Buffer x(a,xf),w(a,packed.size()),expanded(a,wf),out(a,size_t(m*stride)*4),base(a,size_t(m*n)*4);
    w.write(packed.data(),packed.size());std::vector<uint8_t> initial(out.bytes,0x5a);out.write(initial.data(),initial.size());
    auto candidate=[&]{a.check(fn(a.gpu,x.fp(),m,k,w.ptr(),12,n,out.fp(),stride),"aux F32");};
    auto baseline=[&]{a.check(a.nsd_linear(a.gpu,x.ptr(),0,m,k,expanded.ptr(),n,nullptr,base.ptr(),0),"F32 baseline");};
    baseline();candidate();a.sync();const auto y=out.read(),ref=base.read();const auto raw=bytes(a,out);double worst=0;
    for(I r=0;r<m;++r)for(I c=0;c<stride;++c) {
        const size_t at=size_t(r*stride+c);
        if(c>=n){require(std::memcmp(raw.data()+at*4,initial.data()+at*4,4)==0,"aux stride hole");continue;}
        require(std::memcmp(&y[at],&ref[size_t(r*n+c)],4)==0,"aux full F32 baseline bitwise mismatch");
        require(std::isfinite(y[at]),"aux nonfinite");double dot=0,absum=0;
        for(I j=0;j<k;++j){const double v=double(xf[size_t(r*k+j)])*wf[size_t(c*k+j)];dot+=v;absum+=std::abs(v);}
        const double u=std::ldexp(1.,-24),bound=(k*u)/(1-k*u)*absum+1e-12;
        require(std::abs(double(y[at])-dot)<=bound,"aux F64 forward bound");worst=std::max(worst,std::abs(double(y[at])-dot));
    }
    x.guards();w.guards();expanded.guards();out.guards();base.guards();candidate();a.sync();require(bytes(a,out)==raw,"aux repeat mismatch");
    double bt=0,ct=0;for(int i=0;i<reps;++i){if(i%2){ct+=timed(a,candidate);bt+=timed(a,baseline);}else{bt+=timed(a,baseline);ct+=timed(a,candidate);}}
    std::cout<<"PASS AUX_F32 M="<<m<<" K="<<k<<" N="<<n<<" maxF64="<<worst<<" baseline_ms="<<bt/reps<<" packed_ms="<<ct/reps<<"\n";
}
int main(int argc,char** argv)try {
    if(argc==2&&std::string(argv[1])=="--self-test"){aux_cpu_tests();return 0;}
    require(argc==4||argc==8,"usage: aux_numeric ABS_RUNTIME GPU REPEATS [K N PACKED_FILE M]");
    require(argv[1][0]=='/',"absolute runtime required");require(std::getenv("NS_PROFILE")&&std::string(std::getenv("NS_PROFILE"))=="gpu","NS_PROFILE=gpu required");
    const int reps=std::stoi(argv[3]);require(reps>=1&&reps<=16,"bounded repetitions");Api a(argv[1],std::stoi(argv[2]));Aux fn=nullptr;a.bind(fn,"nrqi_aux_f32_linear");
    using Stats=int(*)(ns_gpu*,uint64_t*,uint64_t*);Stats stats=nullptr;a.bind(stats,"nr_allocation_stats");
    uint64_t initial=0,peak=0;a.check(stats(a.gpu,&initial,&peak),"initial stats");require(initial==0,"empty initial allocator");
    aux_cpu_tests();aux_invalid(a,fn);
    if(argc==8){const I k=std::stoll(argv[4]),n=std::stoll(argv[5]),m=std::stoll(argv[7]);
        require((k==256||k==4096)&&(n==8||n==64||n==4096||n==16384),"aux fixture geometry");
        require(m==1||m==2||m==17||m==64||m==256,"bounded aux M");aux_run(a,fn,m,k,n,fixture(argv[6],12,k,n),reps);
    }else for(I k:{I(256),I(512)})for(I m:{I(1),I(2),I(17)})for(I n:{I(1),I(19)})aux_run(a,fn,m,k,n,synthetic(12,k,n),reps);
    a.sync();uint64_t live=0;a.check(stats(a.gpu,&live,&peak),"drop stats");require(live==0&&peak<512ull*1024*1024,"aux allocation leak/cap");
    std::cout<<"tracked_initial="<<initial<<" tracked_final="<<live<<" tracked_peak="<<peak<<"\nALL AUX F32 PASS (operator only)\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
