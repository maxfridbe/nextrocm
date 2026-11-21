// Compare deterministic native outputs bit-for-bit; independent CPU gates are separate.
#define main numeric_validation_main
#include "native_numeric.cpp"
#undef main
static std::vector<F> execute(const char* lib){
 Api a(lib,0);std::vector<F> outputs;
 for(I dim:{40,64,80,160,512,37})for(bool causal:{false,true}){
  I n=259,h=2,c=dim*h;
  Buffer q(a,values(n*c)),k(a,values(n*c,8)),v(a,values(n*c,9)),y(a,F(n*c,sentinel));
  nrsd_attention_shape s{};s.batch=1;s.queries=n;s.keys=n;s.heads=h;s.dim=dim;s.scale=1/std::sqrt(float(dim));s.causal=causal;
  s.qb=s.kb=s.vb=s.ob=n*c;s.qt=s.kt=s.vt=s.ot=c;s.qh=s.kh=s.vh=s.oh=dim;
  a.check(a.nrsd_attention(a.gpu,q.fp(),k.fp(),v.fp(),y.fp(),s),"attention");outputs.push_back(y.read());y.guards();
 }
 for(I kernel:{1,3})for(I stride:{1,2})for(int type:{0,1}){
  I ci=35,co=67,h=19,pad=kernel/2,oh=(h+2*pad-kernel)/stride+1;
  nrsd_conv_shape s{2,ci,h,h,co,kernel,kernel,stride,stride,pad,pad,oh,oh};
  Buffer x(a,values(2*ci*h*h)),y(a,F(2*co*oh*oh,sentinel));Weight w(a,co*ci*kernel*kernel,type,31),b(a,co,type,9);
  a.check(a.nrsd_conv2d(a.gpu,x.fp(),w.raw(),b.raw(),y.fp(),s),"conv");outputs.push_back(y.read());y.guards();
 }
 for(I rows:{1,19,77,257})for(int type:{0,1}){
  nrsd_linear_shape s{rows,81,131,85,139};
  Buffer x(a,values(rows*s.x_stride)),y(a,F(rows*s.out_stride,sentinel));Weight w(a,s.output*s.input,type,31),b(a,s.output,type,9);
  a.check(a.nrsd_linear(a.gpu,x.fp(),w.raw(),b.raw(),y.fp(),s),"linear");outputs.push_back(y.read());y.guards();
 }
 return outputs;
}
int main(int argc,char**argv){try{
 require(argc==3,"usage: parity BASELINE CANDIDATE");auto baseline=execute(argv[1]),candidate=execute(argv[2]);require(baseline.size()==candidate.size(),"count");size_t floats=0;
 for(size_t i=0;i<baseline.size();++i){require(baseline[i].size()==candidate[i].size(),"size");require(std::memcmp(baseline[i].data(),candidate[i].data(),baseline[i].size()*4)==0,"bitwise mismatch case "+std::to_string(i));floats+=baseline[i].size();}
 std::cout<<"PASS bitwise parity cases="<<baseline.size()<<" floats="<<floats<<'\n';
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
