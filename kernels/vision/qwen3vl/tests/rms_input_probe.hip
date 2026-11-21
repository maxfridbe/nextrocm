// Diagnostic only: compile with the original default HIP flags, not strict flags.
#define main ordered_validator_main
#include "rms_rope_ordered_numeric.cpp"
#undef main
#include <hip/hip_fp16.h>
#include <filesystem>
__device__ float probe_rd(const void*p,int dt,I i){return dt==1?__half2float(((const __half*)p)[i]):((const float*)p)[i];}
__device__ void probe_wr(void*p,int dt,I i,float x){if(dt==1)((__half*)p)[i]=__float2half_rn(x);else((float*)p)[i]=x;}
// Exact shared RMS arithmetic plus independent records: partials0:256,
// total256 mean257 epsilon258 inverse259, squared264:4360.
__global__ void rms_probe(const void*x,int dt,I c,const float*w,float eps,const int32_t*rows,const float*scale,const float*shift,void*out,int odt,float*records){
 __shared__ float sum[256];int t=threadIdx.x;I r=blockIdx.x;float z=0;
 for(I j=t;j<c;j+=256){float v=probe_rd(x,dt,r*c+j);z+=v*v;records[r*4360+264+j]=v*v;}
 sum[t]=z;records[r*4360+t]=z;__syncthreads();
 for(int s=128;s;s/=2){if(t<s)sum[t]+=sum[t+s];__syncthreads();}
 float inv=rsqrtf(sum[0]/c+eps);
 if(t==0){records[r*4360+256]=sum[0];records[r*4360+257]=sum[0]/c;records[r*4360+258]=sum[0]/c+eps;records[r*4360+259]=inv;}
 for(I j=t;j<c;j+=256){float v=probe_rd(x,dt,r*c+j)*inv*(w?w[j]:1);if(rows&&scale&&shift&&rows[r]>=0){I mr=rows[r];v=v*(1+scale[mr*c+j])+shift[mr*c+j];}probe_wr(out,odt,r*c+j,v);}
}
static void save(const std::filesystem::path&p,const F&v){std::ofstream f(p,std::ios::binary);f.write(reinterpret_cast<const char*>(v.data()),std::streamsize(v.size()*4));req(bool(f),"save");}
int main(int argc,char**argv){try{
 req(argc==4,"usage: rms_input_probe FROZEN_DSO PINNED_LAYER0_TRACE NEW_OUT");
 std::filesystem::path out(argv[3]);req(!std::filesystem::exists(out),"output exists");std::filesystem::create_directories(out);
 const std::string dir=argv[2];F x=read_raw(dir+"/embed.bin",32*4096),w=read_raw(dir+"/input_norm.weight.bin",4096),expected=read_raw(dir+"/input_norm.bin",32*4096);
 Api a(argv[1]);int(*rms)(void*,const void*,int,I,I,const float*,float,const int32_t*,const float*,const float*,void*,int)=nullptr;a.bind(rms,"nsd_rms_norm_mod");
 auto*g=static_cast<ns_gpu*>(a.g);req(hipSetDevice(g->dev)==hipSuccess,"setdevice");
 Buffer dx(a,x),dw(a,w),original(a,F(x.size(),NAN)),instrumented(a,F(x.size(),NAN)),records(a,F(32*4360,0));
 req(rms(a.g,dx.ptr(),0,32,4096,dw.ptr(),1e-6f,nullptr,nullptr,nullptr,original.ptr(),0)==0,"frozen RMS");
 rms_probe<<<32,256,0,g->q>>>(dx.ptr(),0,4096,dw.ptr(),1e-6f,nullptr,nullptr,nullptr,instrumented.ptr(),0,records.ptr());req(hipGetLastError()==hipSuccess,"probe RMS");
 F y=original.read(),z=instrumented.read(),r=records.read();
 save(out/"frozen.f32",y);save(out/"probe.f32",z);save(out/"records.f32",r);
 req(dx.read()==x&&dw.read()==w,"readonly input/weight changed");
 req(std::memcmp(y.data(),expected.data(),y.size()*4)==0,"DSO RMS differs from captured input_norm");
 req(std::memcmp(y.data(),z.data(),y.size()*4)==0,"instrumentation changed output; records invalid for frozen path");
 std::cout<<"CAPTURE COMPLETE RMS matches captured and frozen bits=131072; no numeric admission\n";return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
