//! Native Qwen-Image Turbo port. Service adapter deployment is user-authorized
//! despite the retained original numerical-gate failure; see engine reporting.
pub mod auxiliary;
pub mod block;
pub mod ffi;
pub mod plan;
pub mod sched;
pub mod weights;
pub mod denoiser;
pub mod vae;

#[cfg(feature = "operator-profile")]
pub mod profiler;

pub mod engine;
pub use engine::kind;
