// Diagnostic-only fixed captured QK intervention; no production dispatch.
// Default HIP flags keep exp/tree/PV identical to attention_stage_probe.hip.
#define main ordered_validator_main
#include "rms_rope_ordered_numeric.cpp"
#undef main
#include <filesystem>
__global__ void attention_scale_probe(const float*qkv,const float*captured,int64_t d,int divide,float divisor,float*out,float*records){
 extern __shared__ float score[];__shared__ float tmp[256];
 const unsigned t=threadIdx.x;const int64_t r=blockIdx.x,h=blockIdx.y,kh=h/4,end=r+1;
 float*rec=records+(r*32+h)*130;
 float best=-INFINITY;
 for(int64_t j=t;j<end;j+=256){
  float z=captured[(r*32+h)*130+j];
  rec[j]=z;if(divide)z=__fdiv_rn(z,divisor);else z*=rsqrtf(float(d));score[j]=z;rec[32+j]=z;best=fmaxf(best,z);
 }
 tmp[t]=best;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s)tmp[t]=fmaxf(tmp[t],tmp[t+s]);__syncthreads();}
 best=tmp[0];if(t==0)rec[128]=best;__syncthreads();float sum=0;
 for(int64_t j=t;j<end;j+=256){float z=expf(score[j]-best);score[j]=z;rec[64+j]=z;sum+=z;}
 tmp[t]=sum;__syncthreads();
 for(unsigned s=128;s;s>>=1){if(t<s)tmp[t]+=tmp[t+s];__syncthreads();}
 const float den=tmp[0];if(t==0)rec[129]=den;
 for(int64_t j=t;j<end;j+=256){score[j]=__fdiv_rn(score[j],den);rec[96+j]=score[j];}
 __syncthreads();
 for(int64_t a=t;a<d;a+=256){float z=0;for(int64_t j=0;j<end;++j)z=fmaf(score[j],qkv[j*6144+5120+kh*d+a],z);out[(r*32+h)*d+a]=z;}
}

static void save(const std::filesystem::path&p,const F&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*4));req(bool(f),"save");}
int main(int argc,char**argv){try{
 req(argc==5,"usage: attention_scale_probe FROZEN_DSO PINNED_FIXTURES PINNED_CAPTURE NEW_OUT");
 std::filesystem::path out(argv[4]);req(!std::filesystem::exists(out),"output exists");std::filesystem::create_directories(out);
 Api a(argv[1]);decltype(&nrqv_attention_prob_f32) attention=nullptr;a.bind(attention,"nrqv_attention_prob_f32");
 auto*g=static_cast<ns_gpu*>(a.g);req(hipSetDevice(g->dev)==hipSuccess,"setdevice");
 const float divisor=std::sqrt(128.0f);uint32_t db=0;std::memcpy(&db,&divisor,4);req(db==0x413504f3u,"rounded sqrt128 divisor");
 for(const std::string name:{"native","primary"}){
  const std::string dir=argv[2],cap=argv[3];F x=read_raw(dir+"/"+name+".qkv.f32",32*6144);
  F captured=read_raw(cap+"/"+name+".records.f32",32*32*130),expected=read_raw(cap+"/"+name+".probe.f32",32*4096);
  for(int r=0;r<32;++r)for(int h=0;h<32;++h)for(int j=r+1;j<32;++j)for(int s=0;s<4;++s)req(captured[(r*32+h)*130+s*32+j]==0,"masked record nonzero");
  Buffer dx(a,x),dc(a,captured),original(a,F(32*4096,NAN));
  req(attention(a.g,dx.ptr(),32,original.ptr())==0,"frozen attention");F y=original.read();
  req(std::memcmp(y.data(),expected.data(),y.size()*4)==0,"DSO differs from frozen capture");
  if(name=="native"){F strict=read_raw(dir+"/native.expected_gpu.f32",32*4096);req(std::memcmp(y.data(),strict.data(),y.size()*4)==0,"DSO differs from strict trace");}
  // Baseline full records/output must pass BEFORE the changed branch is launched.
  for(int divide=0;divide<2;++divide){
   Buffer output(a,F(32*4096,NAN)),records(a,F(32*32*130,0));
   attention_scale_probe<<<dim3(32,32),256,32*4,g->q>>>(dx.ptr(),dc.ptr(),128,divide,divisor,output.ptr(),records.ptr());req(hipGetLastError()==hipSuccess,"scale probe");
   F z=output.read(),r=records.read();for(float v:z)req(std::isfinite(v),"nonfinite output");for(float v:r)req(std::isfinite(v),"nonfinite record");
   if(!divide){req(std::memcmp(y.data(),z.data(),y.size()*4)==0,"baseline output differs; intervention invalid");req(std::memcmp(captured.data(),r.data(),r.size()*4)==0,"baseline records differ; intervention invalid");}
   else for(int q=0;q<32;++q)for(int h=0;h<32;++h)for(int j=0;j<=q;++j){
    const size_t i=size_t(q*32+h)*130+j;const float scaled=captured[i]/divisor;
    req(std::memcmp(&r[i],&captured[i],4)==0,"raw QK changed");req(std::memcmp(&r[i+32],&scaled,4)==0,"division differs from host RN division");
   }
   const std::string tag=name+(divide?".divide":".baseline");save(out/(tag+".f32"),z);save(out/(tag+".records.f32"),r);
   std::cout<<"CAPTURE "<<tag<<(divide?" fixed rawQK/division-only":" output+records bitwise frozen")<<'\n';
  }
  req(dx.read()==x,"QKV changed");req(dc.read()==captured,"captured input changed");
 }
 std::cout<<"CAPTURE COMPLETE scaling-only intervention; no numeric admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
