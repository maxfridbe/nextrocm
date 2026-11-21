// Test-only raw F32 GGML decode; no half rounding, HIP, or production dependency.
#include "oracle/ggml_rows.hpp"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <string>
static void need(bool ok,const char*msg){if(!ok)throw std::runtime_error(msg);}
template<class T> static std::vector<float> decode(const char*file,size_t count){
 std::ifstream in(file,std::ios::binary);need(bool(in),"open input");
 std::vector<T> blocks(count/256);in.read(reinterpret_cast<char*>(blocks.data()),blocks.size()*sizeof(T));
 need(size_t(in.gcount())==blocks.size()*sizeof(T)&&in.peek()==EOF,"exact packed byte count");
 std::vector<float> out(count);
 if constexpr(sizeof(T)==144)qi_oracle::dequantize_row_q4_K(reinterpret_cast<const qi_oracle::block_q4_K*>(blocks.data()),out.data(),count);
 else qi_oracle::dequantize_row_q6_K(reinterpret_cast<const qi_oracle::block_q6_K*>(blocks.data()),out.data(),count);
 for(float x:out) { need(std::isfinite(x),"nonfinite F32 decode"); }
 return out;
}
int main(int argc,char**argv){try{
 need(argc==6,"usage: dump TYPE K ROWS INPUT NEW_F32_OUTPUT");
 auto number=[](const char*s){size_t used;auto v=std::stoll(s,&used);need(used==std::strlen(s),"trailing integer");return v;};
 auto type=number(argv[1]),k=number(argv[2]),rows=number(argv[3]);
 need((type==12||type==14)&&k>0&&k<=12288&&k%256==0&&rows>0&&rows<=64,"bounded slice geometry");
 uint16_t one=1;need(*reinterpret_cast<unsigned char*>(&one)==1,"requires little endian");
 need(!std::ifstream(argv[5]).good(),"output exists");
 auto out=type==12?decode<qi_oracle::block_q4_K>(argv[4],size_t(k*rows)):decode<qi_oracle::block_q6_K>(argv[4],size_t(k*rows));
 std::ofstream dst(argv[5],std::ios::binary);dst.write(reinterpret_cast<const char*>(out.data()),out.size()*4);dst.close();need(bool(dst),"output write");
 std::cout<<"PASS raw finite F32 values="<<out.size()<<" no half rounding\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
