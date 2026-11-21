//! Inkling 2 Mini validated plans, bounded storage and native operator probes.
//! No complete inference engine, model registration, or measured fit claim yet.
//! Official config: thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7.
//! Forward semantics: transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3.
pub mod config;
pub mod plan;
pub type Result<T> = std::result::Result<T, String>;

pub mod weights;

pub mod store;

pub mod memory;
pub mod primitives;
pub mod quant;
pub mod session;
pub mod device;
mod ops;
pub mod first_layer;
pub mod text_graph;

/// Private integration draft; no engine registration or service admission.
pub mod adapter;

pub mod output;

pub mod request;

mod registration;
pub use registration::{info, kind};

pub mod expert_cache;
