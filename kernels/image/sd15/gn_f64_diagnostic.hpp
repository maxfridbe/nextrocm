// Included only with NR_SD15_DIAGNOSTIC_GN_F64_STATS=1 inside sd15.hip.
// Hypothesis control, not an oracle-equivalent algorithm or production fix.
// Same 256 lanes/indexing and centered two-pass algorithm as norm_k, widening
// only statistics. Cast mean/inv to F32 before the unchanged F32 affine formula.
// 2048 bytes shared scratch; no global allocation; public entry/validation intact.
#pragma once
__global__ void gn_f64_stats_diagnostic_k(
 const float* x,nrsd_weight w,nrsd_weight b,float* y,I count,
 I channels_per_group,I spatial,I groups,float eps) {
 __shared__ double tmp[256];
 const int t=threadIdx.x;
 const I row=blockIdx.x,offset=row*count;
 double z=0.0;
 for(I i=t;i<count;i+=256)z+=double(x[offset+i]);
 tmp[t]=z;__syncthreads();
 for(int j=128;j;j/=2){if(t<j)tmp[t]+=tmp[t+j];__syncthreads();}
 const double mean64=tmp[0]/double(count);
 __syncthreads();z=0.0;
 for(I i=t;i<count;i+=256){const double d=double(x[offset+i])-mean64;z=fma(d,d,z);}
 tmp[t]=z;__syncthreads();
 for(int j=128;j;j/=2){if(t<j)tmp[t]+=tmp[t+j];__syncthreads();}
 const float mean=float(mean64);
 const float inv=float(1.0/sqrt(tmp[0]/double(count)+double(eps)));
 for(I i=t;i<count;i+=256){
  const I c=(row%groups)*channels_per_group+i/spatial;
  y[offset+i]=(x[offset+i]-mean)*inv*(w.data?rd(w,c):1.f)+bias(b,c);
 }
}
