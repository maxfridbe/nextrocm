// Diagnostic only. Default HIP flags preserve the frozen division kernel.
#define main ordered_validator_main
#include "rms_rope_ordered_numeric.cpp"
#undef main
#include <filesystem>
// Records/query/head: rawQK0:32, scaled32:64, exp64:96, P96:128,max128,sum129.
// Unused causal entries stay zero; no mask sentinel is treated as a valid score.
__global__ void attention_division_stage_probe(const float*qkv,int64_t d,float divisor,float*out,float*records){
 extern __shared__ float score[];__shared__ float tmp[256];
 const unsigned t=threadIdx.x;const int64_t r=blockIdx.x,h=blockIdx.y,kh=h/4,end=r+1;
 float*rec=records+(r*32+h)*130;
 float best=-INFINITY;
 for(int64_t j=t;j<end;j+=256){
  float z=0;for(int64_t a=0;a<d;++a)z=fmaf(qkv[r*6144+h*d+a],qkv[j*6144+4096+kh*d+a],z);
  rec[j]=z;z=__fdiv_rn(z,divisor);score[j]=z;rec[32+j]=z;best=fmaxf(best,z);
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
 req(argc==4,"usage: attention_division_stage_probe FROZEN_DSO PINNED_FIXTURES NEW_OUT");
 std::filesystem::path out(argv[3]);req(!std::filesystem::exists(out),"output exists");
 const std::string dir=argv[2];F x=read_raw(dir+"/qkv.f32",32*6144),expected=read_raw(dir+"/expected_gpu.f32",32*4096);
 for(float v:x)req(std::isfinite(v),"nonfinite input");
 for(float v:expected)req(std::isfinite(v),"nonfinite expected");
 Api a(argv[1]);decltype(&nrqv_attention_divide_f32) attention=nullptr;a.bind(attention,"nrqv_attention_divide_f32");
 auto*g=static_cast<ns_gpu*>(a.g);req(hipSetDevice(g->dev)==hipSuccess,"setdevice");
 Buffer dx(a,x),original(a,F(32*4096,NAN)),instrumented(a,F(32*4096,NAN)),records(a,F(32*32*130,0));
 req(attention(a.g,dx.ptr(),32,original.ptr())==0,"frozen attention");
 F y=original.read();
 req(std::memcmp(y.data(),expected.data(),y.size()*4)==0,"frozen DSO differs from actual layer1 preimage; no valid capture");
 attention_division_stage_probe<<<dim3(32,32),256,32*4,g->q>>>(dx.ptr(),128,0x1.6a09e6p+3f,instrumented.ptr(),records.ptr());req(hipGetLastError()==hipSuccess,"probe attention");
 F z=instrumented.read(),r=records.read(),input_after=dx.read();
 req(std::memcmp(input_after.data(),x.data(),x.size()*4)==0,"QKV input changed");
 req(std::memcmp(y.data(),z.data(),y.size()*4)==0,"instrumentation changed output; records invalid");
 for(float v:z)req(std::isfinite(v),"nonfinite attention");
 for(float v:r)req(std::isfinite(v),"nonfinite record");
 for(int q=0;q<32;++q)for(int h=0;h<32;++h)for(int j=q+1;j<32;++j)for(int slot=0;slot<4;++slot)
  req(r[(q*32+h)*130+slot*32+j]==0,"masked record nonzero");
 // No records are written until both full-output bitwise prerequisites pass.
 std::filesystem::create_directories(out);
 save(out/"frozen.f32",y);save(out/"probe.f32",z);save(out/"records.f32",r);
 std::cout<<"CAPTURE COMPLETE layer1 division stages; frozen/actual/probe bitwise=131072; no numeric admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
