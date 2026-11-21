//! Strict fixed-checkpoint contract; other Inkling variants need separate validation.
use crate::Result;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct MiniConfig {
    _validated: (),
}
impl MiniConfig {
    /// Validate official JSON without allocating weights or assuming GGUF aliases.
    pub fn from_official_json(bytes: &[u8]) -> Result<Self> {
        let v: Value = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if v["model_type"] != "inkling2_loop"
            || v["architectures"] != serde_json::json!(["Inkling2LoopForConditionalGeneration"])
        {
            return Err("not the fixed Inkling 2 Mini architecture".into());
        }
        if v["dtype"] != "bfloat16" || v["eos_token_id"].as_u64() != Some(200006) {
            return Err("checkpoint dtype/EOS mismatch".into());
        }
        let t = &v["text_config"];
        const KEYS: &[&str] = &[
            "attn_res_normalize_values",
            "d_rel",
            "final_logit_softcapping",
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "local_layer_ids",
            "log_scaling_alpha",
            "log_scaling_n_floor",
            "loop_config",
            "mlp_gate_softcap",
            "mlp_up_softcap",
            "model_max_length",
            "model_type",
            "n_layers_per_attn_res_block",
            "n_routed_experts",
            "n_shared_experts",
            "num_attention_heads",
            "num_experts_per_tok",
            "num_hidden_layers",
            "num_key_value_heads",
            "o_bias",
            "q_bias",
            "qk_l2_clip_threshold",
            "rel_extent",
            "rel_l2_clip_threshold",
            "rms_norm_eps",
            "route_scale",
            "sconv_kernel_size",
            "sliding_window_size",
            "swa_head_dim",
            "swa_num_attention_heads",
            "swa_num_key_value_heads",
            "torch_dtype",
            "unpadded_vocab_size",
            "use_gate_bias",
            "use_value_norm_no_gamma",
            "vocab_size",
        ];
        let object = t.as_object().ok_or("text_config must be an object")?;
        if object.len() != KEYS.len() || object.keys().any(|k| !KEYS.contains(&k.as_str())) {
            return Err(
                "unknown or missing text config keys: behavior overrides are not supported".into(),
            );
        }
        if t["torch_dtype"] != "bfloat16" {
            return Err("text dtype mismatch".into());
        }

        if t["model_type"] != "inkling2_loop_text" {
            return Err("text architecture mismatch".into());
        }
        for (k, n) in [
            ("hidden_size", 2304),
            ("num_hidden_layers", 24),
            ("vocab_size", 201024),
            ("unpadded_vocab_size", 200058),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("swa_num_attention_heads", 24),
            ("swa_num_key_value_heads", 8),
            ("swa_head_dim", 128),
            ("model_max_length", 1048576),
            ("intermediate_size", 1152),
            ("n_routed_experts", 128),
            ("num_experts_per_tok", 8),
            ("n_shared_experts", 0),
            ("sconv_kernel_size", 4),
            ("sliding_window_size", 1024),
            ("d_rel", 16),
            ("rel_extent", 1024),
            ("log_scaling_n_floor", 8192),
            ("n_layers_per_attn_res_block", 6),
        ] {
            if t[k].as_u64() != Some(n) {
                return Err(format!("unsupported text_config.{k}"));
            }
        }
        for (k, x) in [
            ("rms_norm_eps", 1e-6),
            ("route_scale", 8.0),
            ("mlp_gate_softcap", 4.0),
            ("mlp_up_softcap", 25.0),
            ("final_logit_softcapping", 30.0),
            ("qk_l2_clip_threshold", 22.627416997969522),
            ("rel_l2_clip_threshold", 8.0),
            ("log_scaling_alpha", 0.1),
        ] {
            if t[k].as_f64() != Some(x) {
                return Err(format!("unsupported text_config.{k}"));
            }
        }
        for (k, b) in [
            ("q_bias", false),
            ("o_bias", false),
            ("use_value_norm_no_gamma", true),
            ("attn_res_normalize_values", true),
            ("use_gate_bias", true),
        ] {
            if t[k].as_bool() != Some(b) {
                return Err(format!("unsupported text_config.{k}"));
            }
        }
        let locals: Vec<_> = (0..24).filter(|i| (i + 1) % 6 != 0).collect();
        if t["local_layer_ids"] != serde_json::json!(locals) {
            return Err("local/global layer schedule mismatch".into());
        }
        let l = &t["loop_config"];
        let loop_object = l.as_object().ok_or("loop_config must be an object")?;
        let loop_keys = [
            "core_start",
            "core_end",
            "num_passes",
            "gate_rank",
            "is_varloop",
        ];
        if loop_object.len() != loop_keys.len()
            || loop_object.keys().any(|k| !loop_keys.contains(&k.as_str()))
        {
            return Err("loop behavior overrides unsupported".into());
        }
        for (k, n) in [
            ("core_start", 6),
            ("core_end", 18),
            ("num_passes", 3),
            ("gate_rank", 64),
        ] {
            if l[k].as_u64() != Some(n) {
                return Err(format!("unsupported loop_config.{k}"));
            }
        }
        if l["is_varloop"] != false || t.get("loop_share_cache").is_some_and(|x| x != false) {
            return Err("variable loops or shared pass caches unsupported".into());
        }
        Ok(Self { _validated: () })
    }
    pub const fn width(&self) -> usize {
        2304
    }
    pub const fn physical_layers(&self) -> usize {
        24
    }
    pub const fn applications(&self) -> usize {
        48
    }
    pub const fn kv_heads(&self) -> usize {
        8
    }
    pub const fn head_dim(&self) -> usize {
        128
    }
    pub const fn window(&self) -> usize {
        1024
    }
    pub const fn context_limit(&self) -> usize {
        1048576
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CONFIG: &[u8] = include_bytes!("../tests/mini-config.json");
    #[test]
    fn pinned_config_and_reject_other_geometry() {
        assert!(MiniConfig::from_official_json(CONFIG).is_ok());
        for (k, bad) in [
            ("num_hidden_layers", 48),
            ("num_experts_per_tok", 4),
            ("swa_head_dim", 64),
        ] {
            let mut v: Value = serde_json::from_slice(CONFIG).unwrap();
            v["text_config"][k] = bad.into();
            assert!(MiniConfig::from_official_json(&serde_json::to_vec(&v).unwrap()).is_err());
        }
    }
    #[test]
    fn behavior_overrides_cannot_bypass_validation() {
        for (key, value) in [
            ("scoring_func", serde_json::json!("softmax")),
            ("layer_types", serde_json::json!(["hybrid"])),
            ("conv_kernel_size", serde_json::json!(8)),
            ("moe_intermediate_size", serde_json::json!(4096)),
            ("routed_scaling_factor", serde_json::json!(2.0)),
            ("loop_share_cache", serde_json::json!(true)),
        ] {
            let mut v: Value = serde_json::from_slice(CONFIG).unwrap();
            v["text_config"][key] = value;
            assert!(MiniConfig::from_official_json(&serde_json::to_vec(&v).unwrap()).is_err());
        }
    }
    #[test]
    fn fixed_loop_and_mask_are_required() {
        let mut v: Value = serde_json::from_slice(CONFIG).unwrap();
        v["text_config"]["loop_config"]["is_varloop"] = true.into();
        assert!(MiniConfig::from_official_json(&serde_json::to_vec(&v).unwrap()).is_err());
        let mut v: Value = serde_json::from_slice(CONFIG).unwrap();
        v["text_config"]["local_layer_ids"][0] = 5.into();
        assert!(MiniConfig::from_official_json(&serde_json::to_vec(&v).unwrap()).is_err());
    }
}
