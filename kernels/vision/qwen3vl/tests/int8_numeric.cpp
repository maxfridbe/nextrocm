// CPU: c++ -std=c++17 -O2 -Wall -Wextra -Werror FILE -o OUT; OUT --self-test
// HIP: hipcc -std=c++17 -O2 --offload-arch=gfx1035 -x hip FILE -o OUT
// Run through run_reference.py to bind fixture and executable hashes.
#include "../int8_convrot.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
using namespace qv_int8;
using Bytes=std::vector<uint8_t>;
static void require(bool ok,const std::string& s) { if(!ok) throw std::runtime_error(s); }
static void host_tests() {
    for(unsigned i=0;i<256;++i) for(unsigned j=0;j<256;++j) {
        int dot=0;
        for(unsigned k=0;k<256;++k) dot+=hadamard_sign(i,k)*hadamard_sign(j,k);
        require(dot==(i==j ? 256 : 0),"regular H256 orthogonality");
    }
    // Independent tensor-product construction, including the regular H4 signs.
    std::vector<int> h{1}; unsigned size=1;
    const int h4[4][4]={{1,1,1,-1},{1,1,-1,1},{1,-1,1,1},{-1,1,1,1}};
    for(int digit=0;digit<4;++digit) {
        std::vector<int> next(size*size*16);
        for(unsigned i=0;i<size;++i) for(unsigned j=0;j<size;++j)
            for(unsigned a=0;a<4;++a) for(unsigned b=0;b<4;++b)
                next[(i*4+a)*(size*4)+j*4+b]=h[i*size+j]*h4[a][b];
        h=std::move(next); size*=4;
    }
    for(unsigned i=0;i<256;++i) for(unsigned j=0;j<256;++j)
        require(h[i*256+j]==hadamard_sign(i,j),"regular versus Sylvester signs");
    auto ptr=[](unsigned index) { return reinterpret_cast<void*>(uintptr_t(index)*0x10000000u); };
    Args a{ptr(1),static_cast<int8_t*>(ptr(2)),static_cast<float*>(ptr(3)),ptr(4),ptr(5),
           static_cast<float*>(ptr(6)),static_cast<int8_t*>(ptr(7)),static_cast<float*>(ptr(8)),
           static_cast<int32_t*>(ptr(9)),static_cast<uint32_t*>(ptr(10)),1,17,256,17,F32};
    require(valid(a),"valid fixture rejected");
    for(I m:{I(-1),I(0),I(4097)}) { auto b=a;b.m=m;require(!valid(b),"M guard"); }
    for(I n:{I(-1),I(0),I(24577)}) { auto b=a;b.n=n;require(!valid(b),"N guard"); }
    for(I k:{I(-1),I(0),I(255),I(257),I(12544),INT64_MAX}) {
        auto b=a;b.k=k;require(!valid(b),"K/divisibility/overflow guard");
    }
    for(int dtype:{-1,2}) { auto b=a;b.dtype=dtype;require(!valid(b),"dtype guard"); }
    for(I s:{I(0),I(2),I(18)}) { auto b=a;b.scales=s;require(!valid(b),"scale count guard"); }
    auto b=a;b.out=const_cast<void*>(a.x);require(!valid(b),"output alias");
    b=a;b.rotated=static_cast<float*>(a.out);require(!valid(b),"scratch alias");
    b=a;b.status=reinterpret_cast<uint32_t*>(UINTPTR_MAX-3);require(!valid(b),"wrap guard");
    b=a;b.x=reinterpret_cast<void*>(uintptr_t(a.x)+1);require(!valid(b),"alignment guard");
    b=a;b.bias=nullptr;require(valid(b),"nullable bias");
    b=a;b.x=nullptr;require(!valid(b),"null input");
    b=a;b.dtype=BF16;require(valid(b),"BF16 geometry");
    // I8 dot cannot overflow I32 for any admitted K, even -128*-128.
    require(int64_t(12288)*128*128<INT32_MAX,"I32 bound");
    std::cout<<"CPU PASS H256 independent Kronecker/orthogonality, geometry/spans/overflow\n";
}
#ifdef __HIPCC__
namespace fs=std::filesystem;
static void hc(hipError_t e) { require(e==hipSuccess,hipGetErrorString(e)); }
static Bytes read(const fs::path& path,size_t count) {
    require(fs::file_size(path)==count,"file size: "+path.string());
    Bytes bytes(count);std::ifstream f(path,std::ios::binary);
    f.read(reinterpret_cast<char*>(bytes.data()),std::streamsize(count));
    require(bool(f),"read: "+path.string());return bytes;
}
static void exact(const Bytes& a,const Bytes& b,const std::string& stage) {
    require(a.size()==b.size(),stage+" length");
    for(size_t i=0;i<a.size();++i) require(a[i]==b[i],stage+" bit mismatch byte "+std::to_string(i));
}
struct Buffer {
    static constexpr size_t guard=32;
    uint8_t* base=nullptr; size_t bytes;
    explicit Buffer(const Bytes& data):bytes(data.size()) {
        hc(hipMalloc(reinterpret_cast<void**>(&base),bytes+2*guard));
        Bytes all(bytes+2*guard,0xa5);std::copy(data.begin(),data.end(),all.begin()+guard);
        hc(hipMemcpy(base,all.data(),all.size(),hipMemcpyHostToDevice));
    }
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(base) (void)hipFree(base);}
    void* ptr() const{return base+guard;}
    Bytes get() const {
        Bytes all(bytes+2*guard);hc(hipMemcpy(all.data(),base,all.size(),hipMemcpyDeviceToHost));
        for(size_t i=0;i<guard;++i) require(all[i]==0xa5&&all[bytes+guard+i]==0xa5,"buffer guard clobbered");
        return Bytes(all.begin()+guard,all.end()-guard);
    }
    void put(const Bytes& x){require(x.size()==bytes,"put size");hc(hipMemcpy(ptr(),x.data(),bytes,hipMemcpyHostToDevice));}
};
static void gpu_case(const fs::path& path,bool negative) {
    Args a{}; int bias;
    std::ifstream shape(path/"shape.txt");shape>>a.m>>a.n>>a.k>>a.scales>>a.dtype>>bias;
    require(bool(shape)&&geometry(a.m,a.n,a.k,a.dtype,a.scales)&&(bias==0||bias==1),"fixture geometry");
    std::string trailing;require(!(shape>>trailing),"trailing geometry");
    size_t mk=size_t(a.m*a.k),nk=size_t(a.n*a.k),mn=size_t(a.m*a.n),width=a.dtype==F32?4:2;
    // A single fixture must fit an explicit bounded test allocation plan.
    size_t planned=mk*(width+5)+nk+size_t(a.scales)*4+size_t(a.n)*width+mn*(width+4)+size_t(a.m)*4+4+640;
    require(planned<1000000000,"fixture exceeds 1GB test plan");
    auto x=read(path/"x.bin",mk*width),w=read(path/"weight.bin",nk);
    auto ws=read(path/"weight_scale.bin",size_t(a.scales)*4);
    Bytes bias_data=bias?read(path/"bias.bin",size_t(a.n)*width):Bytes(width,0);
    Buffer dx(x),dw(w),ds(ws),db(bias_data),out(Bytes(mn*width,0xcd)),xr(Bytes(mk*4,0xcd)),
           xq(Bytes(mk,0xcd)),rs(Bytes(size_t(a.m)*4,0xcd)),acc(Bytes(mn*4,0xcd)),status(Bytes(4,0xcd));
    a.x=dx.ptr();a.weight=static_cast<int8_t*>(dw.ptr());a.weight_scale=static_cast<float*>(ds.ptr());
    a.bias=bias?db.ptr():nullptr;a.out=out.ptr();a.rotated=static_cast<float*>(xr.ptr());
    a.quantized=static_cast<int8_t*>(xq.ptr());a.row_scale=static_cast<float*>(rs.ptr());
    a.accum=static_cast<int32_t*>(acc.ptr());a.status=static_cast<uint32_t*>(status.ptr());
    auto run=[&](){hc(launch(a,nullptr));hc(hipDeviceSynchronize());auto bytes=status.get();uint32_t value;std::memcpy(&value,bytes.data(),4);return value;};
    require(run()==0,"numerical status nonzero");
    exact(xr.get(),read(path/"rotated.bin",mk*4),"rotation");
    exact(xq.get(),read(path/"quantized.bin",mk),"quantization");
    exact(rs.get(),read(path/"row_scale.bin",size_t(a.m)*4),"row scales");
    exact(acc.get(),read(path/"accum.bin",mn*4),"I32 accumulators");
    exact(out.get(),read(path/"out.bin",mn*width),"scaled/bias output");
    exact(dx.get(),x,"input unchanged");exact(dw.get(),w,"weight unchanged");exact(ds.get(),ws,"scale unchanged");exact(db.get(),bias_data,"bias unchanged");
    if(negative) {
        // GPU content invalidity is reported asynchronously, not silently cast.
        auto bad=x;
        if(a.dtype==F32){uint32_t nan=0x7fc00000u;std::memcpy(bad.data(),&nan,4);}
        else {uint16_t nan=0x7fc0u;std::memcpy(bad.data(),&nan,2);}
        dx.put(bad);require((run()&3u)==3u,"NaN input not rejected");dx.put(x);
        if(a.dtype==F32){uint32_t inf=0x7f800000u;std::memcpy(bad.data(),&inf,4);}
        else {uint16_t inf=0x7f80u;std::memcpy(bad.data(),&inf,2);}
        dx.put(bad);require((run()&3u)==3u,"infinite input not rejected");dx.put(x);
        auto bad_scale=ws;std::fill_n(bad_scale.begin(),4,0);ds.put(bad_scale);
        require(run()&4u,"zero weight scale not rejected");ds.put(ws);
        uint32_t inf=0x7f800000u;std::memcpy(bad_scale.data(),&inf,4);ds.put(bad_scale);
        require(run()&4u,"infinite weight scale not rejected");ds.put(ws);
        if(bias) {
            auto bad_bias=bias_data;
            if(a.dtype==F32)std::memcpy(bad_bias.data(),&inf,4);
            else {uint16_t inf16=0x7f80u;std::memcpy(bad_bias.data(),&inf16,2);}
            db.put(bad_bias);require(run()&8u,"nonfinite bias/output not rejected");db.put(bias_data);
        }
        auto invalid=a;invalid.out=const_cast<void*>(a.x);
        require(launch(invalid,nullptr)==hipErrorInvalidValue,"GPU overlap guard");
        invalid=a;invalid.k=257;require(launch(invalid,nullptr)==hipErrorInvalidValue,"GPU unsupported K guard");
        require(run()==0,"recovery status");exact(out.get(),read(path/"out.bin",mn*width),"recovery output");
    }
    std::cout<<"PASS "<<path.filename().string()<<" M="<<a.m<<" N="<<a.n<<" K="<<a.k
             <<" dtype="<<a.dtype<<" planned_bytes="<<planned<<" all_stages_bitwise\n";
}
#endif
int main(int argc,char**argv) {
    try {
        host_tests();if(argc==2&&std::string(argv[1])=="--self-test")return 0;
#ifdef __HIPCC__
        require(argc==3,"usage: int8_numeric REFERENCE_DIRECTORY GPU_INDEX");
        std::string gpu_arg=argv[2];size_t end=0;int gpu=std::stoi(gpu_arg,&end);
        require(end==gpu_arg.size()&&gpu>=0,"GPU index");hc(hipSetDevice(gpu));
        std::vector<fs::path> cases;
        for(const auto& entry:fs::directory_iterator(argv[1]))if(entry.is_directory())cases.push_back(entry.path());
        std::sort(cases.begin(),cases.end());require(!cases.empty(),"empty fixture set");
        for(const auto& path:cases)gpu_case(path,true);
        std::cout<<"GPU PASS "<<cases.size()<<" pinned CPU fixtures\n";
#else
        (void)argv;throw std::runtime_error("CPU build: --self-test only");
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
