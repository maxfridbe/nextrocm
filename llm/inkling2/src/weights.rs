//! Exact selected root-GGUF directory contract. Derived from release merged
//! source 883286076efe5b85fddb31d9cc7035a6b698ba9b, not its base tag.
//! A physical layer owns its expert scales as well as weights on EVERY visit.
use crate::{plan::Visit, Result};
use nextrocm_gguf::{GType, Gguf, Tensor, Value};
use std::collections::BTreeMap;

pub struct Weights<'a> {
    tensors: BTreeMap<&'a str, &'a Tensor>,
    packed_bytes: u64,
    non_expert_bytes: u64,
    largest_expert_pair_bytes: u64,
}
pub struct Experts<'a> {
    pub gate_up: &'a Tensor,
    pub down: &'a Tensor,
    /// Post-matmul per-expert scales, in addition to NVFP4 block scales.
    pub gate_up_scale: Option<&'a Tensor>,
    pub down_scale: Option<&'a Tensor>,
}
impl<'a> Weights<'a> {
    pub fn read(f: &'a Gguf) -> Result<Self> {
        if f.architecture() != "inkling2" {
            return Err("selected root GGUF requires inkling2, not a mainline alias".into());
        }
        for (k, n) in [
            ("block_count", 48),
            ("embedding_length", 2304),
            ("context_length", 1048576),
            ("feed_forward_length", 1152),
            ("expert_feed_forward_length", 1152),
            ("expert_count", 128),
            ("expert_used_count", 8),
            ("attention.key_length", 128),
            ("attention.value_length", 128),
            ("attention.key_length_swa", 128),
            ("attention.value_length_swa", 128),
            ("num_loops", 3),
            ("loop.core_start", 6),
            ("loop.core_end", 18),
            ("loop.gate_rank", 64),
            ("vocab_size", 201024),
            ("vocab_size_unpadded", 200058),
            ("rope.dimension_count", 0),
            ("attention.sliding_window", 1024),
            ("shortconv.l_cache", 4),
            ("attn_res.block_size", 6),
            ("attention.relative_length", 16),
            ("attention.relative_extent", 1024),
            ("attention.log_scale_floor", 8192),
        ] {
            if f.arch_meta(k).and_then(Value::as_u64) != Some(n) {
                return Err(format!("unsupported inkling2.{k}"));
            }
        }
        for (k, x) in [
            ("attention.layer_norm_rms_epsilon", 1e-6f32),
            ("expert_weights_scale", 8.0),
            ("activation.situ_beta", 4.0),
            ("activation.situ_linear_beta", 25.0),
            ("final_logit_softcapping", 30.0),
            ("attention.qk_clip", 22.627416997969522f32),
            ("attention.relative_clip", 8.0),
            ("attention.log_scale_alpha", 0.1),
        ] {
            if f.arch_meta(k)
                .and_then(Value::as_f64)
                .map(|v| (v as f32).to_bits())
                != Some(x.to_bits())
            {
                return Err(format!("unsupported inkling2.{k}"));
            }
        }
        for (k, n) in [("attention.head_count", 24), ("attention.head_count_kv", 8)] {
            let a = f
                .arch_meta(k)
                .and_then(Value::as_array)
                .ok_or_else(|| format!("missing {k}"))?;
            if a.len() != 48 || a.iter().any(|v| v.as_u64() != Some(n)) {
                return Err(format!("invalid per-visit {k}"));
            }
        }
        let a = f
            .arch_meta("attention.sliding_window_pattern")
            .and_then(Value::as_array)
            .ok_or("missing SWA pattern")?;
        if a.len() != 48
            || a.iter()
                .enumerate()
                .any(|(i, v)| !matches!(v,Value::Bool(b) if *b==((i+1)%6!=0)))
        {
            return Err("SWA visit pattern mismatch".into());
        }
        Self::validate_directory(&f.tensors)
    }
    pub fn validate_directory(tensors: &'a [Tensor]) -> Result<Self> {
        let mut expected = BTreeMap::<String, (Vec<u64>, GType)>::new();
        let mut add = |name: String, shape: &[u64], ty: GType| {
            expected.insert(name, (shape.to_vec(), ty));
        };
        for name in ["token_embd.weight", "output.weight"] {
            add(name.into(), &[201024, 2304], GType::Q8_0)
        }
        for name in [
            "token_embd_norm.weight",
            "output_norm.weight",
            "output_res_query.weight",
        ] {
            add(name.into(), &[2304], GType::F32)
        }
        for name in ["loop_res_query.weight", "loop_kv_res_query.weight"] {
            add(name.into(), &[24, 2304], GType::F32)
        }
        for l in 0..24 {
            let mut t =
                |name: &str, shape: &[u64], ty: GType| add(format!("blk.{l}.{name}"), shape, ty);
            for (name, shape) in [
                ("attn_gate.weight", vec![3072, 2304]),
                ("attn_k.weight", vec![1024, 2304]),
                ("attn_v.weight", vec![1024, 2304]),
                ("attn_q.weight", vec![3072, 2304]),
                ("attn_output.weight", vec![2304, 3072]),
                ("attn_rel.weight", vec![384, 2304]),
            ] {
                t(name, &shape, GType::Q8_0)
            }
            for (name, shape) in [
                ("attn_k_conv.weight", vec![1024, 4]),
                ("attn_v_conv.weight", vec![1024, 4]),
                ("attn_post_conv.weight", vec![2304, 4]),
                ("ffn_post_conv.weight", vec![2304, 4]),
                ("attn_q_clip.weight", vec![24, 128]),
                ("attn_rel_clip.weight", vec![24, 16]),
                ("attn_rel_bias.weight", vec![1024, 16]),
                ("ffn_norm.weight", vec![2304]),
                ("ffn_res_query.weight", vec![2304]),
                ("ffn_gate_inp.weight", vec![128, 2304]),
                ("exp_probs_b.bias", vec![128]),
            ] {
                t(name, &shape, GType::F32)
            }
            t("attn_sink_proj.weight", &[24, 2304], GType::BF16);
            if (l + 1) % 6 == 0 {
                for x in ["q", "k", "v"] {
                    t(&format!("attn_{x}_inp_norm.weight"), &[2304], GType::F32)
                }
                for x in ["k", "v"] {
                    t(
                        &format!("attn_{x}_conv_proj.weight"),
                        &[32, 2304],
                        GType::BF16,
                    )
                }
            } else {
                t("attn_norm.weight", &[2304], GType::F32)
            }
            if l > 0 {
                t(
                    "attn_res_query.weight",
                    &if (l + 1) % 6 == 0 {
                        vec![3, 2304]
                    } else {
                        vec![2304]
                    },
                    GType::F32,
                )
            }
            if l < 20 {
                t("ffn_gate_up_exps.weight", &[128, 2304, 2304], GType::NVFP4);
                t("ffn_down_exps.weight", &[128, 2304, 1152], GType::NVFP4);
                t("ffn_gate_up_exps.scale", &[128], GType::F32);
                t("ffn_down_exps.scale", &[128], GType::F32);
            } else {
                t("ffn_gate_up_exps.weight", &[128, 2304, 2304], GType::Q5K);
                t("ffn_down_exps.weight", &[128, 2304, 1152], GType::Q8_0);
            }
            if (6..18).contains(&l) {
                for m in ["attn", "k", "v", "mlp"] {
                    for op in ["down", "gain", "forget"] {
                        t(
                            &format!("loop_{m}_{op}.weight"),
                            &if op == "down" {
                                vec![64, 4608]
                            } else {
                                vec![2304, 64]
                            },
                            GType::F32,
                        );
                    }
                }
            }
        }
        if tensors.len() != expected.len() {
            return Err(format!(
                "expected {} tensors, got {}",
                expected.len(),
                tensors.len()
            ));
        }
        let mut map = BTreeMap::new();
        let mut total = 0u64;
        let mut non_expert = 0u64;
        for tensor in tensors {
            if map.insert(tensor.name.as_str(), tensor).is_some() {
                return Err("duplicate tensor name".into());
            }
            let (shape, ty) = expected
                .get(&tensor.name)
                .ok_or_else(|| format!("unexpected tensor {}", tensor.name))?;
            if &tensor.shape != shape || tensor.ty != *ty {
                return Err(format!("shape/type mismatch {}", tensor.name));
            }
            let elements = shape
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(*d))
                .ok_or("tensor size overflow")?;
            let (block, bytes) = ty.block();
            let required = elements
                .checked_div(block as u64)
                .and_then(|n| n.checked_mul(bytes as u64))
                .ok_or("tensor bytes overflow")?;
            if elements % (block as u64) != 0 || required != tensor.bytes {
                return Err(format!("tensor bytes mismatch {}", tensor.name));
            }
            total = total.checked_add(required).ok_or("weight bytes overflow")?;
            if !tensor.name.contains("_exps.weight") {
                non_expert = non_expert
                    .checked_add(required)
                    .ok_or("resident bytes overflow")?
            }
        }
        let mut largest = 0;
        for l in 0..24 {
            let gu = map[format!("blk.{l}.ffn_gate_up_exps.weight").as_str()].bytes / 128;
            let down = map[format!("blk.{l}.ffn_down_exps.weight").as_str()].bytes / 128;
            largest = largest.max(gu + down);
        }
        Ok(Self {
            tensors: map,
            packed_bytes: total,
            non_expert_bytes: non_expert,
            largest_expert_pair_bytes: largest,
        })
    }
    pub fn packed_bytes(&self) -> u64 {
        self.packed_bytes
    }
    /// Candidate residency accounting only; this is not an allocation/admission plan.
    pub fn non_expert_bytes(&self) -> u64 {
        self.non_expert_bytes
    }
    pub fn largest_expert_pair_bytes(&self) -> u64 {
        self.largest_expert_pair_bytes
    }
    pub fn tensor(&self, name: &str) -> Option<&'a Tensor> {
        self.tensors.get(name).copied()
    }
    pub fn experts_for_visit(&self, visit: &Visit) -> Result<Experts<'a>> {
        let l = visit.physical_layer;
        if l >= 24 {
            return Err("invalid physical layer".into());
        }
        let get = |suffix: &str| self.tensor(&format!("blk.{l}.{suffix}"));
        Ok(Experts {
            gate_up: get("ffn_gate_up_exps.weight").ok_or("missing gate/up")?,
            down: get("ffn_down_exps.weight").ok_or("missing down")?,
            gate_up_scale: get("ffn_gate_up_exps.scale"),
            down_scale: get("ffn_down_exps.scale"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Vec<Tensor> {
        let values: serde_json::Value =
            serde_json::from_str(include_str!("../tests/root-tensors.json")).unwrap();
        values
            .as_array()
            .unwrap()
            .iter()
            .map(|v| Tensor {
                name: v["name"].as_str().unwrap().into(),
                ty: GType::from_code(v["type_id"].as_u64().unwrap() as u32).unwrap(),
                shape: v["shape_outer_first"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap())
                    .collect(),
                shard: 0,
                offset: v["file_start"].as_u64().unwrap(),
                bytes: v["bytes"].as_u64().unwrap(),
            })
            .collect()
    }
    #[test]
    fn host_embedding_budget_keeps_margin_and_checks_free() {
        use crate::{device::EmbeddingMode, expert_cache::ExpertMode, memory::Estimate};
        let tensors = fixture();
        let w = Weights::validate_directory(&tensors).unwrap();
        let c = crate::config::MiniConfig::from_official_json(include_bytes!(
            "../tests/mini-config.json"
        ))
        .unwrap();
        let plan = crate::plan::ExecutionPlan::new(&c, 4096).unwrap();
        let mode = ExpertMode::DeviceLru {
            budget_bytes: 896 * 1024 * 1024,
        };
        let base = Estimate::text_with_policies(
            &w,
            &plan,
            1,
            48 << 30,
            8_000_000_000,
            mode,
            EmbeddingMode::Resident,
        )
        .unwrap();
        let host = Estimate::text_with_policies(
            &w,
            &plan,
            1,
            48 << 30,
            8_000_000_000,
            mode,
            EmbeddingMode::HostRow,
        )
        .unwrap();
        assert_eq!(base.tracked - host.tracked, 492_106_752 - 2448);
        assert_eq!(host.embedding_row, 2448);
        assert_eq!(base.host_staging, host.host_staging);
        assert_eq!(host.driver_margin, 1 << 30);
        assert_eq!(host.device_total, 3_972_302_320);
        assert!(Estimate::text_with_policies(
            &w,
            &plan,
            1,
            48 << 30,
            host.device_total - 1,
            mode,
            EmbeddingMode::HostRow
        )
        .is_err());
        assert!(Estimate::text_with_policies(
            &w,
            &plan,
            1,
            48 << 30,
            host.device_total,
            mode,
            EmbeddingMode::HostRow
        )
        .is_ok());
    }
    #[test]
    fn actual_mixed_directory_and_repeated_scales() {
        let tensors = fixture();
        let w = Weights::validate_directory(&tensors).unwrap();
        assert_eq!(tensors.len(), 734);
        assert_eq!(w.packed_bytes(), 16598564864);
        let c = crate::config::MiniConfig::from_official_json(include_bytes!(
            "../tests/mini-config.json"
        ))
        .unwrap();
        let p = crate::plan::ExecutionPlan::new(&c, 4096).unwrap();
        for visit in p.visits() {
            let e = w.experts_for_visit(visit).unwrap();
            assert_eq!(e.gate_up_scale.is_some(), visit.physical_layer < 20);
            assert_eq!(e.down_scale.is_some(), visit.physical_layer < 20);
        }
        let first = w.experts_for_visit(&p.visits()[6]).unwrap();
        for i in [18, 30] {
            let e = w.experts_for_visit(&p.visits()[i]).unwrap();
            assert!(std::ptr::eq(first.gate_up, e.gate_up));
            assert!(std::ptr::eq(
                first.gate_up_scale.unwrap(),
                e.gate_up_scale.unwrap()
            ));
        }
    }
    #[test]
    fn rejects_missing_scale_wrong_dtype_and_storage() {
        let mut t = fixture();
        let i = t
            .iter()
            .position(|t| t.name == "blk.6.ffn_gate_up_exps.scale")
            .unwrap();
        t.remove(i);
        assert!(Weights::validate_directory(&t).is_err());
        let mut t = fixture();
        t[0].ty = GType::F16;
        assert!(Weights::validate_directory(&t).is_err());
        let mut t = fixture();
        t[0].bytes -= 1;
        assert!(Weights::validate_directory(&t).is_err());
        let mut t = fixture();
        t[0].name = t[1].name.clone();
        assert!(Weights::validate_directory(&t).is_err());
    }
}
