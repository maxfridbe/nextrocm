// Diagnostic only: idle-queue upload+sync; no model throughput inference.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "../../../kernels/ns/ns.h"
struct Api {
 void* lib=nullptr; ns_gpu* g=nullptr;
 decltype(&ns_gpu_open) open; decltype(&ns_gpu_close) close;
 decltype(&ns_alloc) alloc; decltype(&ns_free) free;
 decltype(&ns_alloc_host) host; decltype(&ns_free_host) unhost;
 decltype(&ns_copy_to) to; decltype(&ns_copy_from) from;
 decltype(&ns_sync) sync; decltype(&ns_fill) fill;
 decltype(&ns_stamp) stamp; decltype(&ns_elapsed) elapsed;
 int (*stats)(ns_gpu*,uint64_t*,uint64_t*);
 template<class T> T sym(const char* n) {auto p=dlsym(lib,n); if(!p)throw std::runtime_error(n); return reinterpret_cast<T>(p);}
 static void check(int rc) {if(rc)throw std::runtime_error("runtime operation failed");}
 explicit Api(const char* path) {
  lib=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!lib)throw std::runtime_error(dlerror());
#define BIND(field,name) field=sym<decltype(field)>(#name)
  BIND(open,ns_gpu_open);BIND(close,ns_gpu_close);BIND(alloc,ns_alloc);BIND(free,ns_free);
  BIND(host,ns_alloc_host);BIND(unhost,ns_free_host);BIND(to,ns_copy_to);BIND(from,ns_copy_from);
  BIND(sync,ns_sync);BIND(fill,ns_fill);BIND(stamp,ns_stamp);BIND(elapsed,ns_elapsed);BIND(stats,nr_allocation_stats);
#undef BIND
  check(open(0,&g));
 }
 ~Api(){if(g)close(g);if(lib)dlclose(lib);}
};
struct Buffers {
 Api& a; size_t n; bool pinned; void* device=nullptr; void* source=nullptr; std::vector<uint8_t> readback;
 Buffers(Api& api,size_t bytes,bool pin):a(api),n(bytes),pinned(pin),readback(bytes+128) {
  Api::check(a.alloc(a.g,n+128,&device));
  try {if(pinned)Api::check(a.host(a.g,n+128,&source));else {source=std::malloc(n+128);if(!source)throw std::bad_alloc();}}
  catch(...) {a.free(a.g,device);throw;}
 }
 ~Buffers(){if(device)a.free(a.g,device);if(source){if(pinned)a.unhost(a.g,source);else std::free(source);}}
 void release(){Api::check(a.sync(a.g));Api::check(a.free(a.g,device));device=nullptr;
  if(pinned)Api::check(a.unhost(a.g,source));else std::free(source);source=nullptr;}
 uint8_t* src(){return static_cast<uint8_t*>(source)+64;}
 uint8_t* dst(){return static_cast<uint8_t*>(device)+64;}
};
static uint8_t pattern(size_t i){return uint8_t((i*131u+(i>>9u)*17u+29u)&255u);}
static void verify(Buffers& b){
 Api::check(b.a.from(b.a.g,b.readback.data(),b.device,b.n+128));Api::check(b.a.sync(b.a.g));
 auto s=static_cast<uint8_t*>(b.source);
 for(size_t i=0;i<b.n+128;++i){uint8_t want=i<64||i>=b.n+64?0xa5:pattern(i-64);
  if(s[i]!=want||b.readback[i]!=want)throw std::runtime_error("full byte or guard mismatch");}
}
using Clock=std::chrono::steady_clock;
static double ns(Clock::time_point start){return std::chrono::duration<double,std::nano>(Clock::now()-start).count();}
static double median(std::vector<double> x){std::sort(x.begin(),x.end());return (x[(x.size()-1)/2]+x[x.size()/2])/2;}
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: h2d ABS_RUNTIME_SO (JSON stdout; external identity/watchdog required)");
 if(setenv("NS_PROFILE","gpu",1))throw std::runtime_error("setenv");
 Api a(argv[1]);uint64_t live=0,peak=0;Api::check(a.stats(a.g,&live,&peak));if(live)throw std::runtime_error("initial live allocations");
 std::cout<<std::setprecision(17)<<"{\"scope\":\"idle queue; wall pass copy+sync; event pass interval may include pageable staging/host submission gaps, NOT pure DMA\",\"cases\":[";
 bool first=true;
 for(size_t bytes:{size_t(2448),size_t(4478976),size_t(6469632),size_t(35831808),size_t(51757056)})for(bool pinned:{false,true}){
  Buffers b(a,bytes,pinned);auto s=static_cast<uint8_t*>(b.source);
  std::memset(s,0xa5,bytes+128);for(size_t i=0;i<bytes;++i)b.src()[i]=pattern(i);
  Api::check(a.fill(a.g,b.device,0xa5,bytes+128));Api::check(a.sync(a.g));
  for(int i=0;i<3;++i){Api::check(a.to(a.g,b.dst(),b.src(),bytes));Api::check(a.sync(a.g));}verify(b);
  size_t reps=std::min(size_t(4096),(size_t(512)*1024*1024+bytes-1)/bytes);
  for(bool events:{false,true}){
   std::vector<double> walls,device_times;walls.reserve(reps);device_times.reserve(reps);double total=0, event_total=0;
   for(size_t r=0;r<reps;++r){int64_t begin=0,end=0;
    auto start=Clock::now();if(events)Api::check(a.stamp(a.g,&begin));
    Api::check(a.to(a.g,b.dst(),b.src(),bytes));if(events)Api::check(a.stamp(a.g,&end));Api::check(a.sync(a.g));
    double wall=ns(start);if(!std::isfinite(wall)||wall<=0)throw std::runtime_error("invalid wall");walls.push_back(wall);total+=wall;
    if(events){double elapsed=0;Api::check(a.elapsed(a.g,begin,end,&elapsed));if(!std::isfinite(elapsed)||elapsed<0)throw std::runtime_error("invalid event");device_times.push_back(elapsed);event_total+=elapsed;}
   }
   verify(b);if(!first)std::cout<<',';first=false;
   std::cout<<"{\"bytes\":"<<bytes<<",\"pinned\":"<<(pinned?"true":"false")<<",\"events\":"<<(events?"true":"false")<<",\"repetitions\":"<<reps<<",\"transferred_bytes\":"<<bytes*reps<<",\"wall_ns_sum\":"<<total<<",\"wall_ns_median\":"<<median(walls)<<",\"wall_GB_per_s\":"<<double(bytes)*double(reps)/total<<",\"event_ns_sum\":"<<event_total<<",\"event_ns_median\":"<<(events?median(device_times):0)<<",\"full_bytes_and_guards\":true}"<<std::flush;
  }
  b.release();Api::check(a.stats(a.g,&live,&peak));if(live||peak>64*1024*1024)throw std::runtime_error("allocation gate");
 }
 std::cout<<"],\"status\":\"PASS\",\"live_bytes\":"<<live<<",\"peak_device_bytes\":"<<peak<<",\"max_explicit_host_plus_device_bytes\":"<<3*(51757056+128)<<"}\n";
 return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
