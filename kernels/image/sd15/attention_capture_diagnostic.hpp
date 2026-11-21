// Private, compile-time opt-in capture. Included at file scope, not a namespace.
// Capture one call in a fresh process. Stores packed B,H,T,D actual FP32 inputs.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace {
int attention_capture_diagnostic(ns_gpu* g,const float* q,const float* k,
                                 const float* v,nrsd_attention_shape s) {
 const char* path=std::getenv("NR_SD15_ATTN_CAPTURE_PATH");
 if(!path||!*path)return 0;
 const char* which=std::getenv("NR_SD15_ATTN_CAPTURE_CALL");
 uint64_t wanted=0;
 if(which){
  if(!*which)return ns_fail("attention capture: empty call ordinal");
  for(const char* c=which;*c;++c){
   if(*c<'0'||*c>'9'||wanted>(UINT64_MAX-9)/10)
    return ns_fail("attention capture: invalid call ordinal");
   wanted=wanted*10+uint64_t(*c-'0');
  }
 }
 static std::atomic<uint64_t> calls{0};
 uint64_t ordinal=calls.fetch_add(1);
 if(ordinal!=wanted)return 0;
 constexpr uint64_t cap=64*1024*1024/sizeof(float);
 uint64_t nq=uint64_t(s.batch)*s.heads*s.queries*s.dim;
 uint64_t nk=uint64_t(s.batch)*s.heads*s.keys*s.dim;
 if(nq>cap||nk>(cap-nq)/2)return ns_fail("attention capture: payload exceeds 64 MiB cap");
 try {
  std::vector<float> packed;packed.reserve(size_t(nq+2*nk));
  auto copy=[&](const float* input,int64_t tokens,int64_t bs,int64_t ts,int64_t hs){
   uint64_t span=uint64_t(s.batch-1)*bs+uint64_t(tokens-1)*ts+uint64_t(s.heads-1)*hs+s.dim;
   if(span>cap)return ns_fail("attention capture: source span exceeds 64 MiB cap");
   std::vector<float> host(static_cast<size_t>(span));
   if(ns_copy_from(g,host.data(),input,host.size()*sizeof(float)))return -1;
   if(ns_sync(g))return -1;
   for(int64_t b=0;b<s.batch;++b)for(int64_t h=0;h<s.heads;++h)
    for(int64_t t=0;t<tokens;++t)for(int64_t d=0;d<s.dim;++d)
     packed.push_back(host[b*bs+t*ts+h*hs+d]);
   return 0;
  };
  if(copy(q,s.queries,s.qb,s.qt,s.qh)||copy(k,s.keys,s.kb,s.kt,s.kh)||copy(v,s.keys,s.vb,s.vt,s.vh))return -1;
  // Native little-endian Linux test artifact: magic + seven uint64 + f32 scale.
  // Exclusive creation avoids overwriting evidence. Remove partial writes.
  FILE* file=std::fopen(path,"wbx");
  if(!file)return ns_fail("attention capture: cannot create new artifact");
  uint64_t meta[]={uint64_t(s.batch),uint64_t(s.queries),uint64_t(s.keys),uint64_t(s.heads),uint64_t(s.dim),uint64_t(s.causal),ordinal};
  bool ok=std::fwrite("NRQKV001",1,8,file)==8
   &&std::fwrite(meta,sizeof(uint64_t),7,file)==7
   &&std::fwrite(&s.scale,sizeof(float),1,file)==1
   &&std::fwrite(packed.data(),sizeof(float),packed.size(),file)==packed.size();
  if(std::fclose(file)!=0)ok=false;
  if(!ok){std::remove(path);return ns_fail("attention capture: write failed");}
  std::fprintf(stderr,"attention capture call=%llu B=%lld Q=%lld K=%lld H=%lld D=%lld bytes=%zu path=%s\n",
    static_cast<unsigned long long>(ordinal),static_cast<long long>(s.batch),static_cast<long long>(s.queries),static_cast<long long>(s.keys),static_cast<long long>(s.heads),static_cast<long long>(s.dim),packed.size()*sizeof(float)+68,path);
  return 0;
 }catch(const std::exception& e){return ns_fail(std::string("attention capture: ")+e.what());}
}
}
