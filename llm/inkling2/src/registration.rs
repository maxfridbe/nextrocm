//! Explicit text-only registration; shared caller validates raw JSON before Ask.
use crate::{adapter, config::MiniConfig, plan::ExecutionPlan, weights::Weights};
use nextrocm_gguf::Gguf;
use nextrocm_llm::{ChatTemplate, EngineKind};
pub fn kind() -> EngineKind {
    EngineKind {
        archs: &["inkling2"],
        name: "Inkling-2-Mini native F32 text (one session, bounded expert cache)",
        load: adapter::load,
        info,
        kernels: None,
        chat: ChatTemplate::Checked(crate::request::checked_request),
        options: &[],
        prepare: None,
    }
}
pub fn info(file: &Gguf) -> Result<String, String> {
    let weights = Weights::read(file)?;
    let config = MiniConfig::from_official_json(include_bytes!("../assets/mini-config.json"))?;
    let plan = ExecutionPlan::new(&config, 4096)?;
    Ok(format!("Inkling-2-Mini selected root layout:24 physical layers/48 applications,2304 hidden,128 experts/top8; packed dense{} bytes, F32 KV at4096={} bytes; one session, context<=4096, default1GiB packed expert LRU with direct pinned staging; expert_bytes=0 explicitly streams, positive budgets<=1GiB checked against current memory with no automatic fallback; text-only/no tools or draft; native kernel/reference gates bounded, arbitrary-prompt quality not guaranteed",weights.non_expert_bytes(),plan.kv_storage_bytes(4)?))
}
