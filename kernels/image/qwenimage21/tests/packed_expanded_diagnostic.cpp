// Diagnostic only: preserve failed kernel/validator and expose the first Q6 case.
#define main expanded_validator_entry
#include "packed_numeric.cpp"
#undef main
#include <filesystem>
using Expanded=int(*)(void*,const void*,I,I,const void*,int,I,void*,int,I,void*,I);
static Expanded expand_fn=nullptr;
int main(int argc,char**argv)try{
 require(argc==4,"usage: packed_expanded_diagnostic ABS_RUNTIME GPU NEW_OUTPUT_DIR");
 require(argv[1][0]=='/'&&argv[3][0]=='/',"absolute paths");
 const std::filesystem::path dir(argv[3]);require(!std::filesystem::exists(dir),"output exists");
 Api a(argv[1],std::stoi(argv[2]));Packed old=nullptr;a.bind(old,"nrqi_packed_linear64");a.bind(expand_fn,"nrqi_packed_linear_expanded");
 auto p=synthetic(14,256,1);auto f=decode(p,14,256,1);std::vector<uint16_t> hw(256),hx(256);
 uint32_t seed=42;for(size_t i=0;i<256;++i){hw[i]=qi_oracle::half_bits(f[i]);hx[i]=uint16_t((rnd(seed)&0x83ff)|0x3000);}
 Buffer x(a,512),w(a,p.size()),scratch(a,512),out(a,4),base(a,4);
 x.write(hx.data(),512);w.write(p.data(),p.size());
 a.check(old(a.gpu,x.ptr(),1,256,w.ptr(),14,1,base.ptr(),0,1),"baseline");
 a.check(expand_fn(a.gpu,x.ptr(),1,256,w.ptr(),14,1,out.ptr(),0,1,scratch.ptr(),512),"expanded");a.sync();
 auto got=bytes(a,scratch);require(bytes(a,base)==bytes(a,out),"failed case baseline output no longer reproduced");
 for(Buffer*b:{&x,&w,&scratch,&out,&base})b->guards();
 require(bytes(a,w)==p,"packed mutated");require(bytes(a,x)==std::vector<uint8_t>(reinterpret_cast<uint8_t*>(hx.data()),reinterpret_cast<uint8_t*>(hx.data())+512),"input mutated");
 std::filesystem::create_directory(dir);
 auto dump=[&](const char*name,const void*data,size_t n){std::ofstream file(dir/name,std::ios::binary);file.write(static_cast<const char*>(data),std::streamsize(n));require(bool(file),"dump failed");};
 dump("packed.bin",p.data(),p.size());dump("cpu-half.bin",hw.data(),512);dump("gpu-half.bin",got.data(),512);dump("cpu-decoded.f32",f.data(),1024);
 auto output=bytes(a,out);dump("output.f32",output.data(),output.size());
 size_t mismatches=0,zero_sign=0;for(size_t i=0;i<256;++i){uint16_t g;std::memcpy(&g,got.data()+2*i,2);if(g!=hw[i]){++mismatches;if((g&0x7fff)==0&&(hw[i]&0x7fff)==0)++zero_sign;std::cout<<"index="<<i<<" gpu_half=0x"<<std::hex<<g<<" cpu_half=0x"<<hw[i]<<std::dec<<"\n";}}
 std::cout<<"DIAGNOSTIC_CAPTURE_COMPLETE baseline_bits=1 mismatches="<<mismatches<<" signed_zero_only="<<(mismatches==zero_sign)<<" numeric_pass="<<(mismatches==0)<<std::endl;
 return mismatches?1:0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
