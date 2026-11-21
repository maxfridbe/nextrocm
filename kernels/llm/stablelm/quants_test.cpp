// Host format test only: this does not validate HIP launches or GPU math.
#include "quants.hpp"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
struct Half {
    float operator()(const uint8_t* p) const {
        const unsigned h=p[0]|unsigned(p[1])<<8;
        const int exp=(h>>10)&31, mant=h&1023;
        assert(exp!=31);
        const float v=exp?std::ldexp(float(1024+mant),exp-25):std::ldexp(float(mant),-24);
        return h&32768?-v:v;
    }
};
int main() {
    // Pack independent logical scale/min and quant arrays, then read every lane
    // across two blocks (including negative Q6 scales and high scale bits).
    std::array<uint8_t,288> q4{};
    for(int block=0;block<2;++block){
        uint8_t* b=q4.data()+144*block;b[1]=0x38;b[3]=0x34; // d=.5, min=.25
        int scales[8],mins[8];
        for(int j=0;j<8;++j){scales[j]=(j*11+block*7)%64;mins[j]=(63-j*7+block)%64;}
        for(int j=0;j<4;++j){b[4+j]=scales[j]|((scales[j+4]>>4)<<6);b[8+j]=mins[j]|((mins[j+4]>>4)<<6);b[12+j]=(scales[j+4]&15)|((mins[j+4]&15)<<4);}
        for(int i=0;i<256;++i){int j=i/32,l=i%32,q=(i*7+block)%16;b[16+(j/2)*32+l]|=q<<(4*(j%2));}
        for(int i=0;i<256;++i){float expected=.5f*scales[i/32]*((i*7+block)%16)-.25f*mins[i/32];assert(nslm::quant(q4.data(),12,block*256+i,Half{})==expected);}
    }
    std::array<uint8_t,420> q6{};
    for(int block=0;block<2;++block){
        uint8_t* b=q6.data()+210*block;b[209]=0x38;
        for(int j=0;j<16;++j)b[192+j]=uint8_t(j*13-100+block);
        for(int n=0;n<2;++n)for(int l=0;l<32;++l){
            int q[4];for(int j=0;j<4;++j)q[j]=((128*n+32*j+l)*3+block)%64;
            b[64*n+l]=(q[0]&15)|((q[2]&15)<<4);b[64*n+32+l]=(q[1]&15)|((q[3]&15)<<4);
            b[128+32*n+l]=(q[0]>>4)|((q[1]>>4)<<2)|((q[2]>>4)<<4)|((q[3]>>4)<<6);
        }
        for(int i=0;i<256;++i){float expected=.5f*int8_t(b[192+i/16])*(((i*3+block)%64)-32);assert(nslm::quant(q6.data(),14,block*256+i,Half{})==expected);}
    }
    std::puts("Q4_K/Q6_K: 1024 scalar values matched independent packed fixtures");
}
