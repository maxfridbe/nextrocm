//! Standalone validation harness, not a runtime fallback. No HIP library loaded.
//! rustc --edition=2021 --test host_wrapper.rs -o /tmp/nrsd-host && /tmp/nrsd-host
extern crate self as nextrocm_engine_core;
use std::ffi::c_void;
use std::sync::Arc;
#[derive(Debug)] pub struct Error(pub String);
pub type Result<T> = std::result::Result<T,Error>;
pub struct Gpu;
impl Gpu {pub fn raw(&self)->*mut c_void {std::ptr::null_mut()}}
pub struct DevBuf {pub len:usize,gpu:Arc<Gpu>}
impl DevBuf {
 pub fn ptr(&self)->*mut c_void {std::ptr::null_mut()}
 pub fn fp(&self)->*mut f32 {self.ptr().cast()}
 pub fn gpu(&self)->&Arc<Gpu> {&self.gpu}
 pub fn bounds(&self,at:usize,n:usize)->Result<()> {if at.checked_add(n).is_some_and(|n|n<=self.len){Ok(())}else{Err(Error("bounds".into()))}}
}
pub struct Api;
impl Api {pub fn symbol(&self,_:&str)->*mut c_void {std::ptr::null_mut()}}
pub fn api()->Result<&'static Api>{Err(Error("host test must not load HIP".into()))}
#[path="../../../../image/sd15/src/kernels.rs"] mod kernels;
#[test]fn detached_gpu_rejected(){assert!(kernels::Kernels::new(Arc::new(Gpu)).is_err());}
