// Include inside sd15.hip's anonymous namespace after conv_k: I, rd and bias
// must already be defined. Launch exactly dim3(16,16), with 64-wide grid tiles:
// linear: (ceil(output/64), ceil(rows/64)); conv: (ceil(oh*ow/64), ceil(co/64), batch).
// The existing public shape/stride/weight validation remains the caller's job.
// Each CTA owns 64x64 outputs; each thread owns 4x4 FP32 accumulators.
// K16 deliberately matches the original final zero-padded tile, including its
// extra fmaf(0,0,z) operations. No split-K, reassociation, or fused bias epilogue.
// Arithmetic order is preserved in source; GPU bitwise parity still needs testing.
// Pad row-major K strides to avoid stride-16 LDS bank conflicts during reuse.
// Shared storage: linear 8704 bytes, conv 8448 bytes per CTA; no global scratch.
#pragma once

__global__ void linear_fast_k(const float* x,nrsd_weight w,nrsd_weight b,float* y,nrsd_linear_shape s) {
 __shared__ float a[64][17],v[64][17];
 const int tx=threadIdx.x,ty=threadIdx.y,t=ty*16+tx;
 const I row0=I(blockIdx.y)*64,col0=I(blockIdx.x)*64;
 float z[4][4]={};
 for(I base=0;base<s.input;base+=16) {
  // Four loads per thread per operand; adjacent lanes traverse contiguous K.
  #pragma unroll
  for(int n=0;n<4;++n) {
   const int i=t+n*256,r=i/16,k=i%16;
   a[r][k]=(row0+r<s.rows&&base+k<s.input)?x[(row0+r)*s.x_stride+base+k]:0.f;
   v[r][k]=(col0+r<s.output&&base+k<s.input)?rd(w,(col0+r)*s.input+base+k):0.f;
  }
  __syncthreads();
  #pragma unroll
  for(int k=0;k<16;++k) {
   float av[4],bv[4];
   #pragma unroll
   for(int i=0;i<4;++i){av[i]=a[ty+i*16][k];bv[i]=v[tx+i*16][k];}
   #pragma unroll
   for(int i=0;i<4;++i) {
    #pragma unroll
    for(int j=0;j<4;++j)z[i][j]=fmaf(av[i],bv[j],z[i][j]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<4;++i) {
  const I row=row0+ty+i*16;
  #pragma unroll
  for(int j=0;j<4;++j) {
   const I col=col0+tx+j*16;
   if(row<s.rows&&col<s.output)y[row*s.out_stride+col]=z[i][j]+bias(b,col);
  }
 }
}

__global__ void conv_fast_k(const float* x,nrsd_weight w,nrsd_weight b,float* y,nrsd_conv_shape s) {
 __shared__ float a[16][64],v[64][17];
 const int tx=threadIdx.x,ty=threadIdx.y,t=ty*16+tx;
 const I p0=I(blockIdx.x)*64,co0=I(blockIdx.y)*64,nb=blockIdx.z;
 const I spatial=s.oh*s.ow,reduction=s.ci*s.kh*s.kw;
 float z[4][4]={};
 for(I base=0;base<reduction;base+=16) {
  #pragma unroll
  for(int n=0;n<4;++n) {
   const int i=t+n*256,kr=i/64,pc=i%64;
   const I r=base+kr,p=p0+pc;
   // K flattens [ci,kh,kw], exactly as in conv_k. Only the guarded branch
   // forms an input address: padded spatial/reduction tails never read x.
   float value=0.f;
   if(p<spatial&&r<reduction) {
    const I ic=r/(s.kh*s.kw);
    const I iy=(p/s.ow)*s.sh-s.ph+(r/s.kw)%s.kh;
    const I ix=(p%s.ow)*s.sw-s.pw+r%s.kw;
    if(iy>=0&&iy<s.height&&ix>=0&&ix<s.width)
     value=x[((nb*s.ci+ic)*s.height+iy)*s.width+ix];
   }
   a[kr][pc]=value;
   const int co=i/16,k=i%16;
   v[co][k]=(co0+co<s.co&&base+k<reduction)?rd(w,(co0+co)*reduction+base+k):0.f;
  }
  __syncthreads();
  #pragma unroll
  for(int k=0;k<16;++k) {
   float wv[4],av[4];
   #pragma unroll
   for(int i=0;i<4;++i){wv[i]=v[ty+i*16][k];av[i]=a[k][tx+i*16];}
   #pragma unroll
   for(int i=0;i<4;++i) {
    #pragma unroll
    for(int j=0;j<4;++j)z[i][j]=fmaf(wv[i],av[j],z[i][j]);
   }
  }
  __syncthreads();
 }
 #pragma unroll
 for(int i=0;i<4;++i) {
  const I co=co0+ty+i*16;
  #pragma unroll
  for(int j=0;j<4;++j) {
   const I p=p0+tx+j*16;
   if(p<spatial&&co<s.co)y[(nb*s.co+co)*spatial+p]=z[i][j]+bias(b,co);
  }
 }
}
