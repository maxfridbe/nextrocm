//! nextrocm-serve: what turns an engine into a service - an OpenAI-compatible HTTP API (and a control socket) over a
//! kind's engine contract, the prompt cache, the GPUs' telemetry. It knows engines only through their kind's trait
//! (`nextrocm_llm::Engine`, `nextrocm_image::ImageEngine` in `image`, `nextrocm_audio::AudioEngine` in `audio`), never one
//! engine.

pub mod audio;
pub mod cache;
pub mod cancel;
pub mod carry;
pub mod drives;
pub mod files;
pub mod gpustat;
pub mod gpulock;
pub mod home;
pub mod http;
pub mod idle;
pub mod image;
pub mod maintenance;
pub mod model3d;
pub mod page;
pub mod serve;
pub mod switch;
pub mod telemetry;
pub mod tls;
pub mod video;
pub mod voices;
pub mod wfe;

pub use serve::Server;

/// The release (yy.mmdd.###, from version.sh), reported by the API
pub const VERSION: &str = match option_env!("NS_VERSION") {
    Some(v) => v,
    None => "dev",
};
