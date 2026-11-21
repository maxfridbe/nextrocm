// Diagnostic only. Reuse frozen standalone buffer/API helpers, not its main.
#define main ordered_validator_main
#include "rms_rope_ordered_numeric.cpp"
#undef main
#include <filesystem>
#include <iomanip>
// Slots/head: squares0:128, partials128:160, lanes160:168,
// total168 mean169 epsilon170 sqrt171 inverse172, normalized192:320.
__global__ void probe(float*x,int64_t stride,const float*w,const float*cs,float*records,unsigned head_offset){
 __shared__ float values[128],inv;
 const unsigned t=threadIdx.x;const size_t row=blockIdx.x,base=row*stride+size_t(blockIdx.y)*128;
 float*rec=records+(row*40+head_offset+blockIdx.y)*320;
 values[t]=x[base+t];__syncthreads();
 rec[t]=__fmul_rn(values[t],values[t]);
 if(t==0){
  float lane_sum[8]={};
  for(unsigned p=0;p<4;++p)for(unsigned lane=0;lane<8;++lane){
   float partial=0;
   for(unsigned g=0;g<4;++g){float v=values[g*32+p*8+lane];partial=__fadd_rn(partial,__fmul_rn(v,v));}
   rec[128+p*8+lane]=partial;
   lane_sum[lane]=__fadd_rn(lane_sum[lane],partial);
  }
  float total=0;for(unsigned lane=0;lane<8;++lane){rec[160+lane]=lane_sum[lane];total=__fadd_rn(total,lane_sum[lane]);}
  const float mean=__fdiv_rn(total,128.f);
  inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(mean,1e-6f)));
  rec[168]=total;rec[169]=mean;rec[170]=__fadd_rn(mean,1e-6f);
  rec[171]=__fsqrt_rn(__fadd_rn(mean,1e-6f));rec[172]=inv;
 }
 __syncthreads();
 if(t<64){
  const float a=__fmul_rn(__fmul_rn(values[t],inv),w[t]);
  const float b=__fmul_rn(__fmul_rn(values[t+64],inv),w[t+64]);
  rec[192+t]=a;rec[192+t+64]=b;
  const float co=cs[row*128+2*t],si=cs[row*128+2*t+1];
  x[base+t]=__fsub_rn(__fmul_rn(a,co),__fmul_rn(b,si));
  x[base+t+64]=__fadd_rn(__fmul_rn(b,co),__fmul_rn(a,si));
 }
}
static void write_raw(const std::filesystem::path&p,const F&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*4));req(bool(f),"write diagnostic");}
static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static uint32_t ordered(float f){uint32_t u=bits(f);return u&0x80000000u?~u:u|0x80000000u;}
static void mismatch(const F&actual,const F&expected,const std::string&label){
 size_t count=0;uint32_t maxulp=0;
 for(size_t i=0;i<actual.size();++i)if(bits(actual[i])!=bits(expected[i])){
  req(std::isfinite(actual[i])&&std::isfinite(expected[i]),"nonfinite actual fixture output");
  uint32_t a=ordered(actual[i]),b=ordered(expected[i]),ulp=a>b?a-b:b-a;maxulp=std::max(maxulp,ulp);
  if(count<12)std::cout<<label<<" index="<<i<<" row="<<i/6144<<" lane="<<i%6144<<" actual="<<std::setprecision(10)<<actual[i]<<" expected="<<expected[i]<<" bits="<<std::hex<<bits(actual[i])<<'/'<<bits(expected[i])<<std::dec<<" ulp="<<ulp<<'\n';
  ++count;
 }
 std::cout<<label<<" mismatch_count="<<count<<" max_ulp="<<maxulp<<'\n';
}
int main(int argc,char**argv){try{
 req(argc==4,"usage: ordered_probe FROZEN_DSO PINNED_FIXTURES NEW_OUT");
 std::filesystem::path out(argv[3]);req(!std::filesystem::exists(out),"output exists");std::filesystem::create_directories(out);
 Api a(argv[1]);auto*g=static_cast<ns_gpu*>(a.g);req(hipSetDevice(g->dev)==hipSuccess,"setdevice");
 for(const std::string name:{"layer0","block1"}){
  const std::string dir=std::string(argv[2])+"/"+name;
  F x=read_raw(dir+"/qkv.raw.bin",32*6144),wq=read_raw(dir+"/qnorm.weight.bin",128),wk=read_raw(dir+"/knorm.weight.bin",128),cs=read_raw(dir+"/cos_sin.bin",32*128),expected=read_raw(dir+"/expected.bin",32*6144);
  Buffer original(a,x),instrumented(a,x),dq(a,wq),dk(a,wk),dc(a,cs),records(a,F(32*40*320,0));
  req(a.rope(a.g,original.ptr(),32,32,6144,dq.ptr(),dc.ptr())==0,"frozenQ");req(a.rope(a.g,original.ptr()+4096,32,8,6144,dk.ptr(),dc.ptr())==0,"frozenK");
  probe<<<dim3(32,32),128,0,g->q>>>(instrumented.ptr(),6144,dq.ptr(),dc.ptr(),records.ptr(),0);req(hipGetLastError()==hipSuccess,"probeQ");
  probe<<<dim3(32,8),128,0,g->q>>>(instrumented.ptr()+4096,6144,dk.ptr(),dc.ptr(),records.ptr(),32);req(hipGetLastError()==hipSuccess,"probeK");
  F y=original.read(),z=instrumented.read(),r=records.read();
  req(dq.read()==wq&&dk.read()==wk&&dc.read()==cs,"metadata changed");
  write_raw(out/(name+".frozen.f32"),y);write_raw(out/(name+".probe.f32"),z);write_raw(out/(name+".records.f32"),r);
  mismatch(y,expected,name+" frozen_vs_CPU");mismatch(z,y,name+" probe_vs_frozen");
  req(std::memcmp(y.data(),z.data(),y.size()*4)==0,"instrumentation changed output; intermediates not equivalent to frozen kernel");
 }
 std::cout<<"CAPTURE COMPLETE original failure preserved; no numeric admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
