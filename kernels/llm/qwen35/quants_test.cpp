// Independent logical Q5_K pack -> scalar decode test, CPU only.
// Build: c++ -std=c++17 -O2 quants_test.cpp -o /tmp/qwen35-quants-test
#include "quants.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
struct Half {
    float operator()(const uint8_t* p) const {
        unsigned bits = unsigned(p[0]) | (unsigned(p[1]) << 8);
        unsigned exp = (bits >> 10) & 31, mant = bits & 1023;
        float v = exp ? std::ldexp(float(1024 + mant), int(exp) - 25)
                      : std::ldexp(float(mant), -24);
        return bits & 0x8000 ? -v : v;
    }
};
// Encode scales by bit position, independently of the decoder's extraction.
static void pack_scale(uint8_t* header, int group, int scale, int min) {
    for (int bit=0;bit<6;++bit) {
        int sb, so, mb, mo;
        if (group<4) {sb=group;so=bit;mb=group+4;mo=bit;}
        else if (bit<4) {sb=group+4;so=bit;mb=group+4;mo=bit+4;}
        else {sb=group-4;so=bit+2;mb=group;mo=bit+2;}
        header[sb] |= ((scale>>bit)&1)<<so;
        header[mb] |= ((min>>bit)&1)<<mo;
    }
}
int main() {
    // Q8_0: every signed byte, block stride 34 and two distinct scales.
    std::array<uint8_t,16*34> q8{};
    for(int block=0;block<16;++block) {
        q8[block*34+1]=block<8?0x38:0x3c;
        for(int lane=0;lane<32;++lane) {
            int q=(block%8)*32+lane-128;
            q8[block*34+2+lane]=static_cast<uint8_t>(q);
        }
    }
    for(int i=0;i<512;++i) {
        float expected=(i<256?.5f:1.f)*((i%256)-128);
        if(nq35::quant(q8.data(),8,i,Half{})!=expected) return EXIT_FAILURE;
    }
    std::puts("Q8_0: 512 values matched, all signed bytes and block boundaries");
    size_t checked=0;
    // Every combination of six-bit scale/min, all 32 unsigned Q5 values,
    // every lane/group/high-bit plane, and both sides of a block boundary.
    for(int scale=0;scale<64;++scale) for(int min=0;min<64;++min) {
        std::array<uint8_t,352> packed{};
        std::array<float,512> expected{};
        for(int block=0;block<2;++block) {
            auto* b=packed.data()+block*176;
            b[1]=block?0x3c:0x38; // d = 1 or .5
            b[3]=block?0x38:0x34; // dmin = .5 or .25
            const float d=block?1.f:.5f, dm=block?.5f:.25f;
            for(int group=0;group<8;++group) {
                const int s=(scale+7*group+block)%64,m=(min+13*group+block)%64;
                pack_scale(b+4,group,s,m);
                for(int lane=0;lane<32;++lane) {
                    const int q=(lane+group+block+scale+min)%32;
                    // Encode each bit separately into its plane/nibble slot.
                    for(int bit=0;bit<5;++bit) {
                        const int byte=bit==4?16+lane:48+32*(group/2)+lane;
                        const int shift=bit==4?group:4*(group%2)+bit;
                        b[byte]|=((q>>bit)&1)<<shift;
                    }
                    expected[block*256+group*32+lane]=d*s*q-dm*m;
                }
            }
        }
        for(int i=0;i<512;++i) {
            const float got=nq35::quant(packed.data(),13,i,Half{});
            if(got!=expected[i]) {
                std::fprintf(stderr,"Q5_K mismatch scale=%d min=%d index=%d: %g != %g\n",scale,min,i,got,expected[i]);
                return EXIT_FAILURE;
            }
            ++checked;
        }
    }
    std::printf("Q5_K: %zu independently packed values matched (all scales/mins/quants, two blocks)\n",checked);
}
