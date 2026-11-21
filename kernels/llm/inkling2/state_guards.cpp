// CPU-only argument/ownership tests; no device or numerical-parity claim.
#include "state.hpp"
#include <cassert>
#include <cstdio>
#include <vector>
int main() {
    using namespace ni2_state;
    std::vector<float> x(8192),w(8192*4),d(8192*4),history(8192*3),out(8192);
    assert(conv_args(x.data(),w.data(),nullptr,1,2304,history.data(),out.data()));
    assert(conv_args(x.data(),w.data(),d.data(),8,1024,history.data(),out.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),7,1024,history.data(),out.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),0,1024,history.data(),out.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),1,8193,history.data(),out.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),1,0,history.data(),out.data()));
    assert(!conv_args(nullptr,w.data(),d.data(),1,1024,history.data(),out.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),1,1024,history.data(),x.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),1,1024,w.data(),out.data()));
    assert(!conv_args(x.data(),w.data(),d.data(),1,1024,history.data(),history.data()+1));
    assert(!conv_args(x.data(),w.data(),d.data(),1,1024,history.data(),d.data()));
    assert(key_count(0,1,0)==1);
    assert(key_count(1023,1024,1024)==1024);
    assert(key_count(1024,1024,1024)==1024);
    assert(key_count(INT32_MAX,1024,1024)==1024);
    assert(key_count(65535,65536,0)==65536);
    assert(!key_count(65536,65536,0));
    assert(!key_count(-1,1024,0));
    assert(!key_count(int64_t(INT32_MAX)+1,1024,1024));
    assert(!key_count(1024,1023,1024));
    assert(!key_count(0,1,1));
    std::vector<float> q(3072),k(1024*8*128),v(k.size()),r(384),rw(16384),sink(24),y(3072),scratch(24*1024);
    auto valid=[&](float* dst,float* work,int64_t size) {
        return attention_args(q.data(),k.data(),v.data(),r.data(),rw.data(),sink.data(),
                              2048,1024,1024,dst,work,size);
    };
    assert(valid(y.data(),scratch.data(),scratch.size()));
    assert(!valid(y.data(),scratch.data(),scratch.size()-1));
    assert(!valid(y.data(),scratch.data(),int64_t(24)*65536+1));
    assert(!valid(y.data(),nullptr,scratch.size()));
    for(float* input:{q.data(),k.data(),v.data(),r.data(),rw.data(),sink.data()}) {
        assert(!valid(input,scratch.data(),scratch.size()));
        assert(!valid(y.data(),input,scratch.size()));
    }
    assert(!valid(scratch.data(),scratch.data(),scratch.size()));
    Span s;
    assert(!span(reinterpret_cast<void*>(uintptr_t(1)),1,s));
    assert(!span(reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-3),2,s));
    assert(!span(nullptr,1,s));
    std::puts("PASS state host geometry/span/ownership guards; no GPU numerical claim");
}
