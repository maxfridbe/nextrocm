// Test-only full-row references. Original pinned decode, independent F64 dot,
// plus explicitly labeled scalar/native-order and new wave-order F32 controls.
#include "../nvfp4-decoder-v1/quant_upstream_oracle.hpp"
#include "../../../kernels/llm/inkling2/quant.hpp"
#include <array>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cfenv>
#include <cstdio>
#include <xmmintrin.h>
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T>static void save(const std::string& p,const std::vector<T>& v){
 std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));need(bool(f),"write");
}
int main(int argc,char**argv){try{
 need(argc==7,"TYPE ROWS COLS WEIGHTS SCALE_F32_OR_DASH NEW_PREFIX");
 need(fegetround()==FE_TONEAREST&&(_mm_getcsr()&((1u<<15)|(1u<<6)))==0,"RNE/noFTZ/noDAZ");
 const int type=std::stoi(argv[1]),rows=std::stoi(argv[2]),cols=std::stoi(argv[3]);
 const auto rb=ni2_quant::row_bytes(type,cols);
 need(rows>0&&rows<=201024&&cols>0&&cols<=8192&&rb>0&&int64_t(rows)*cols<=512ll*1024*1024,"geometry");
 std::ifstream f(argv[4],std::ios::binary);need(bool(f),"weight open");
 float scale=1;
 if(std::string(argv[5])!="-"){std::ifstream s(argv[5],std::ios::binary);s.read(reinterpret_cast<char*>(&scale),4);need(s.gcount()==4&&s.peek()==EOF&&std::isfinite(scale),"scale");}
 const std::string prefix=argv[6];
 std::vector<float>x(4*cols),w(cols),scalar(4*rows),wave(4*rows);
 for(int k=0;k<cols;++k){ x[k]=float((k*17+13)%257-128)/128.0f; x[cols+k]=float((k*31+7)%509-254)/256.0f; x[2*cols+k]=float((k*7+91)%127-63)/32.0f; x[3*cols+k]=(k%5==0)?-0.0f:float((k*43+19)%251-125)/128.0f; }
 std::vector<double>ref(4*rows),bound(4*rows);std::vector<uint8_t>b(size_t(rb),0);
 const double u=std::ldexp(1.0,-24),n=(cols+31)/32+7,gamma=n*u/(1-n*u);
 for(int r=0;r<rows;++r){
  f.read(reinterpret_cast<char*>(b.data()),rb);need(f.gcount()==rb,"weight row bytes");
  if(type==40)dequantize_row_nvfp4(reinterpret_cast<const block_nvfp4*>(b.data()),w.data(),cols);
  else if(type==13)dequantize_row_q5_K(reinterpret_cast<const block_q5_K*>(b.data()),w.data(),cols);
  else if(type==8)dequantize_row_q8_0(reinterpret_cast<const block_q8_0*>(b.data()),w.data(),cols);
  else if(type==0)memcpy(w.data(),b.data(),size_t(cols)*4);
  else {need(type==30,"type");for(int k=0;k<cols;++k){uint16_t h;memcpy(&h,b.data()+k*2,2);uint32_t v=uint32_t(h)<<16;memcpy(&w[k],&v,4);}}
  for(int token=0;token<4;++token){
  double sum=0,ab=0;float s=0;std::array<float,32>lanes{};
  for(int k=0;k<cols;++k){
   float decoded=ni2_quant::value(b.data(),type,k);
   need(std::isfinite(w[k])&&memcmp(&decoded,&w[k],4)==0,"full CPU original decoder bits");
   const double product=double(w[k])*double(x[token*cols+k]);sum+=product;ab+=std::abs(product);
   s=std::fmaf(w[k],x[token*cols+k],s);lanes[k%32]=std::fmaf(w[k],x[token*cols+k],lanes[k%32]);
   need(std::isfinite(s)&&std::isfinite(lanes[k%32]),"F32 accumulation overflow");
  }
  for(int distance=16;distance;distance/=2)for(int lane=0;lane<distance;++lane){lanes[lane]=lanes[lane]+lanes[lane+distance];need(std::isfinite(lanes[lane]),"tree overflow");}
  scalar[token*rows+r]=s*scale;wave[token*rows+r]=lanes[0]*scale;ref[token*rows+r]=sum*double(scale);
  bound[token*rows+r]=gamma*ab*std::abs(double(scale))+4.0*(cols+1)*std::ldexp(1.0,-149)*std::max(1.0,std::abs(double(scale)));
  need(std::isfinite(ref[token*rows+r])&&std::isfinite(bound[token*rows+r])&&std::isfinite(scalar[token*rows+r])&&std::isfinite(wave[token*rows+r]),"finite outputs");
  need(std::abs(double(wave[token*rows+r])-ref[token*rows+r])<=bound[token*rows+r],"prospective F64 bound");
  }
 }
 need(f.peek()==EOF,"trailing weight bytes");
 save(prefix+".x.f32",x);save(prefix+".scalar.f32",scalar);save(prefix+".wave.f32",wave);save(prefix+".ref.f64",ref);save(prefix+".bound.f64",bound);
 printf("FOUR_VECTOR_FULL_ROW_ORACLE_PASS type=%d rows=%d cols=%d decoded_values=%lld\n",type,rows,cols,(long long)rows*cols);return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
