#include "quant.hpp"
#include "../../../reference/inkling2-mini/nvfp4-decoder-v1/quant_upstream_oracle.hpp"
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <cfenv>
#include <string>
#include <cmath>
static void require(bool b,const char* m) {if(!b)throw std::runtime_error(m);}
static std::vector<float> oracle(const std::vector<uint8_t>& b,int type,int cols) {
    std::vector<float> y(cols);
    if(type==40)dequantize_row_nvfp4(reinterpret_cast<const block_nvfp4*>(b.data()),y.data(),cols);
    else if(type==13)dequantize_row_q5_K(reinterpret_cast<const block_q5_K*>(b.data()),y.data(),cols);
    else if(type==8)dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(b.data()),y.data(),cols);
    else if(type==0)memcpy(y.data(),b.data(),cols*4);
    else {require(type==30,"dtype");for(int i=0;i<cols;++i){uint16_t h;memcpy(&h,b.data()+i*2,2);uint32_t u=uint32_t(h)<<16;memcpy(&y[i],&u,4);}}
    for(int i=0;i<cols;++i) {float v=ni2_quant::value(b.data(),type,i);require(std::isfinite(v)&&memcmp(&v,&y[i],4)==0,"CPU decoder bits");}
    return y;
}
static void write(const std::string& path,const std::vector<float>& x) {std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(x.data()),x.size()*4);require(bool(f),"write");}
static void selftest() {
    for(unsigned h=0;h<65536;++h) {uint8_t p[2]={uint8_t(h),uint8_t(h>>8)};float a=ni2_quant::half(p),b=oracle_half(uint16_t(h));require((std::isnan(a)&&std::isnan(b))||memcmp(&a,&b,4)==0,"half encoding");}
    for(int type:{0,8,13,30,40})for(int cols:{256,512,2304,8192}) {
        std::vector<uint8_t> b(ni2_quant::row_bytes(type,cols));for(size_t j=0;j<b.size();++j)b[j]=uint8_t(j*37+19);
        if(type==8)for(size_t j=0;j<b.size();j+=34){b[j]=0;b[j+1]=0x38;}
        if(type==13)for(size_t j=0;j<b.size();j+=176){b[j]=0;b[j+1]=0x38;b[j+2]=0;b[j+3]=0x34;}
        if(type==0)for(int i=0;i<cols;++i){float f=float(i%37-18)/16;memcpy(b.data()+i*4,&f,4);}
        if(type==30)for(int i=0;i<cols;++i){uint16_t h=uint16_t(0x3e00+(i%128));memcpy(b.data()+i*2,&h,2);}
        oracle(b,type,cols);
    }
    ni2_quant::Span a,b;
    require(!ni2_quant::matrix(nullptr,40,1,64,36,a),"null");
    require(!ni2_quant::matrix(reinterpret_cast<void*>(4096),40,1,63,36,a),"block tail");
    require(!ni2_quant::matrix(reinterpret_cast<void*>(UINTPTR_MAX-15),40,1,64,36,a),"wrap");
    require(ni2_quant::matrix(reinterpret_cast<void*>(4096),40,3,64,40,a),"stride");
    require(ni2_quant::floats(reinterpret_cast<float*>(8192),3,65,64,b)&&ni2_quant::disjoint(a,b),"out stride");
    std::puts("CPU_SELF_TEST_PASS all65536halfencodings/20mixedquantcases/hostbounds");
}
int main(int argc,char** argv) {try {
    require(std::fegetround()==FE_TONEAREST,"round mode");
    if(argc==2&&std::string(argv[1])=="--self-test"){selftest();return 0;}
    require(argc==5,"usage: TYPE COLS INPUT_ROW OUTPUT_PREFIX");int type=std::stoi(argv[1]),cols=std::stoi(argv[2]);int n=ni2_quant::row_bytes(type,cols);require(n>0,"geometry");
    std::vector<uint8_t> b(n);std::ifstream f(argv[3],std::ios::binary);f.read(reinterpret_cast<char*>(b.data()),n);require(f.gcount()==n&&f.peek()==EOF,"row bytes");auto y=oracle(b,type,cols);
    std::vector<float> x(cols);float dot=0;double d=0;for(int i=0;i<cols;++i){x[i]=float(i%23-11)/32;dot=std::fma(y[i],x[i],dot);d+=double(y[i])*x[i];}
    require(std::isfinite(dot)&&std::isfinite(d),"finite dot");std::string p=argv[4];write(p+".decoded.f32",y);write(p+".x.f32",x);write(p+".dot.f32",{dot});
    std::printf("ROW_CPU_PASS cols=%d f64_dot=%.17g f32_dot=%.9g\n",cols,d,double(dot));return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
