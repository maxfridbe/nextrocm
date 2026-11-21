#include "bridge.h"
#include "common.h"
#include "ns.h"
#include "ns_internal.hpp"
#include <unordered_set>
#include <sstream>
#include <cmath>
#include <locale>

using Progress=int(*)(void*,const char*,int64_t,int64_t);
struct Resident;
thread_local Resident* active=nullptr;
struct Resident {
    ns_gpu* gpu;
    std::unordered_set<void*> allocations;
    std::unique_ptr<SttBackend> model;
    std::string info, result;
    Progress callback=nullptr;
    void* user=nullptr;
    explicit Resident(ns_gpu* g):gpu(g){}
    ~Resident() noexcept {
        auto* previous=active;active=this;
        (void)hipSetDevice(gpu->dev);
        (void)hipDeviceSynchronize();
        model.reset();
        for(void* p:allocations)(void)ns_free(gpu,p);
        active=previous;
    }
};
struct Scope {
    Resident* previous;
    explicit Scope(Resident* p):previous(active){HIP_CHECK(hipSetDevice(p->gpu->dev));active=p;}
    ~Scope(){active=previous;}
};
void* nr_lowvram_allocate(size_t bytes){
    if(!active)fail("native model allocation outside its owning context");
    void* p=nullptr;
    if(ns_alloc(active->gpu,bytes,&p)!=0)fail("%s",ns_last_error());
    try {active->allocations.insert(p);}catch(...){ns_free(active->gpu,p);throw;}
    return p;
}
void nr_lowvram_free(void* p){
    if(!p)return;
    if(!active||!active->allocations.count(p))fail("native model freed an unowned allocation");
    if(ns_free(active->gpu,p)!=0)fail("%s",ns_last_error());
    active->allocations.erase(p);
}
void nr_lowvram_progress(const char* phase,long done,long total){
    if(active&&active->callback&&active->callback(active->user,phase,done,total))fail("request cancelled");
}
static std::string escape(const std::string& s){
    std::string out;
    for(unsigned char c:s){
        if(c=='"'||c=='\\'){out+='\\';out+=c;}
        else if(c<32){char b[8];snprintf(b,sizeof b,"\\u%04x",c);out+=b;}
        else out+=c;
    }
    return out;
}
extern "C" int ns_audio_lowvram_open(ns_gpu* gpu,int kind,const char* path,int q8,void** out){
    if(!gpu||!path||!out)return ns_fail("invalid native model arguments");
    *out=nullptr;
    try{
        auto h=std::make_unique<Resident>(gpu);Scope scope(h.get());
        SttOptions options;options.model=path;options.q8=q8!=0;
        if(kind==0)h->model=make_canary(options);
        else if(kind==1)h->model=make_whisper(options);
        else fail("unknown native speech model kind");
        h->info=h->model->info();
        *out=h.release();return 0;
    }catch(const std::exception& e){return ns_fail(e.what());}
    catch(...){return ns_fail("unknown native model load failure");}
}
extern "C" const char* ns_audio_lowvram_info(void* handle){return handle?static_cast<Resident*>(handle)->info.c_str():nullptr;}
extern "C" int ns_audio_lowvram_transcribe(void* handle,const float* pcm,int64_t count,int rate,const char* language,int translate,Progress callback,void* user,const char** result){
    if(!handle||!pcm||count<=0||rate<=0||!result)return ns_fail("invalid native transcription input");
    auto* h=static_cast<Resident*>(handle);*result=nullptr;
    try{
        Scope scope(h);h->callback=callback;h->user=user;
        Audio audio;audio.sample_rate=rate;audio.samples.assign(pcm,pcm+count);
        for(float v:audio.samples)if(!std::isfinite(v))fail("non-finite audio sample");
        SttOptions options;options.language=language?language:"auto";options.translate=translate!=0;
        auto r=h->model->transcribe(std::move(audio),options);
        HIP_CHECK(hipDeviceSynchronize());
        std::ostringstream s;s.imbue(std::locale::classic());s.precision(15);
        s<<"{\"text\":\""<<escape(r.text)<<"\",\"language\":\""<<escape(r.language)<<"\",\"segments\":[";
        for(size_t i=0;i<r.segments.size();i++){auto& seg=r.segments[i];if(i)s<<',';s<<"{\"start\":"<<seg.start<<",\"end\":"<<seg.end<<",\"text\":\""<<escape(seg.text)<<"\"}";}
        s<<"]}";h->result=s.str();*result=h->result.c_str();h->callback=nullptr;h->user=nullptr;return 0;
    }catch(const std::exception& e){hipDeviceSynchronize();h->callback=nullptr;h->user=nullptr;return ns_fail(e.what());}
    catch(...){hipDeviceSynchronize();h->callback=nullptr;h->user=nullptr;return ns_fail("unknown native transcription failure");}
}
extern "C" void ns_audio_lowvram_close(void* handle){delete static_cast<Resident*>(handle);}
