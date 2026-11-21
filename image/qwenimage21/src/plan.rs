//! Complete pinned DiT tensor contract before any device allocation. GGUF shape
//! order here is the repository reader's outermost-first order, not disk order.
use std::collections::BTreeSet;

use nextrocm_engine_core::{Error, Result};
use nextrocm_gguf::{Gguf, Tensor};

pub const WIDTH: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const FFN: usize = 12288;
pub const LAYERS: usize = 32;
pub const LATENT: usize = 64;

/// The decoder's output precision is part of each operation's contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightArithmetic {
    /// Block linear: dequantized weights round to F16 before F32 accumulation.
    HalfRounded,
    /// Auxiliary linear or norm: no intermediate half rounding of weights.
    Float,
}

#[derive(Debug)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub quant_type: u32,
    pub arithmetic: WeightArithmetic,
    pub permute_rope_rows: bool,
}

pub fn tensor_contract() -> Vec<TensorSpec> {
    let mut result = Vec::with_capacity(297);
    let mut add = |name: String, shape: &[u64], quant_type, arithmetic, permute_rope_rows| {
        result.push(TensorSpec {
            name,
            shape: shape.to_vec(),
            quant_type,
            arithmetic,
            permute_rope_rows,
        });
    };
    use WeightArithmetic::{Float, HalfRounded};
    for (name, shape, ty) in [
        ("img_in.weight", [4096, 64], 30),
        ("modulation.1.weight", [16384, 4096], 12),
        ("norm_out.linear.weight", [4096, 4096], 12),
        ("proj_out.weight", [64, 4096], 12),
        (
            "time_text_embed.timestep_embedder.linear_1.weight",
            [4096, 256],
            12,
        ),
        (
            "time_text_embed.timestep_embedder.linear_2.weight",
            [4096, 4096],
            12,
        ),
        ("txt_in.in_layer.weight", [4096, 4096], 30),
        ("txt_in.out_layer.weight", [4096, 4096], 30),
    ] {
        add(name.into(), &shape, ty, Float, false);
    }
    add("txt_in.text_norm.weight".into(), &[4096], 0, Float, false);
    for layer in 0..LAYERS {
        for (name, shape, ty, permute) in [
            ("attn.to_q.weight", [4096, 4096], 12, true),
            ("attn.to_k.weight", [4096, 4096], 12, true),
            ("attn.to_v.weight", [4096, 4096], 14, false),
            ("attn.to_out.0.weight", [4096, 4096], 12, false),
            ("img_mlp.gate_layer.weight", [12288, 4096], 12, false),
            ("img_mlp.proj.weight", [12288, 4096], 12, false),
            ("img_mlp.out.weight", [4096, 12288], 12, false),
        ] {
            add(
                format!("transformer_blocks.{layer}.{name}"),
                &shape,
                ty,
                HalfRounded,
                permute,
            );
        }
        for name in ["attn.norm_q.weight", "attn.norm_k.weight"] {
            add(
                format!("transformer_blocks.{layer}.{name}"),
                &[128],
                0,
                Float,
                true,
            );
        }
    }
    result
}

pub struct DitPlan {
    /// Stored tensor bytes, excluding GGUF header/alignment padding.
    pub packed_bytes: u64,
    pub block_bytes: u64,
    pub auxiliary_bytes: u64,
}

impl DitPlan {
    pub fn validate(file: &Gguf) -> Result<Self> {
        if file.architecture() != "qwen_image" {
            return Err(Error("expected qwen_image GGUF architecture".into()));
        }
        Self::validate_tensors(&file.tensors)
    }

    /// This initial port accepts the pinned Q4_K_M artifact only. Other quants
    /// require their own tensor contract and numerical validation.
    pub fn validate_tensors(tensors: &[Tensor]) -> Result<Self> {
        let specs = tensor_contract();
        if tensors.len() != specs.len() {
            return Err(Error(
                "DiT tensor count differs from pinned checkpoint".into(),
            ));
        }
        let mut names = BTreeSet::new();
        for t in tensors {
            if !names.insert(t.name.as_str()) {
                return Err(Error(format!("duplicate DiT tensor {}", t.name)));
            }
        }
        let (mut block_bytes, mut auxiliary_bytes) = (0u64, 0u64);
        for spec in specs {
            let t = tensors
                .iter()
                .find(|t| t.name == spec.name)
                .ok_or_else(|| Error(format!("missing {}", spec.name)))?;
            if t.shape != spec.shape || t.ty.code() != spec.quant_type {
                return Err(Error(format!(
                    "{}: shape/type differs from pinned contract",
                    t.name
                )));
            }
            let elements = spec
                .shape
                .iter()
                .try_fold(1u64, |n, &d| n.checked_mul(d))
                .ok_or_else(|| Error("tensor geometry overflow".into()))?;
            if t.ty.bytes(elements) != Some(t.bytes) {
                return Err(Error(format!("{}: invalid stored byte count", t.name)));
            }
            let total = if spec.arithmetic == WeightArithmetic::HalfRounded {
                &mut block_bytes
            } else {
                &mut auxiliary_bytes
            };
            *total = total
                .checked_add(t.bytes)
                .ok_or_else(|| Error("weight byte sum overflow".into()))?;
        }
        Ok(Self {
            packed_bytes: block_bytes
                .checked_add(auxiliary_bytes)
                .ok_or_else(|| Error("weight byte sum overflow".into()))?,
            block_bytes,
            auxiliary_bytes,
        })
    }
}

/// Reference live activation buffers plus cached prefix projections. Caller
/// adds packed weights, operator scratch, other live tensors and driver margin.
pub fn activation_bytes(image_rows: u64, prefix_rows: u64) -> Result<u64> {
    fn step(n: u64, p: u64) -> Result<u64> {
        let d = WIDTH as u64;
        let f = FFN as u64;
        let rows = n
            .checked_add(p)
            .ok_or_else(|| Error("token count overflow".into()))?;
        let mut total = 0u64;
        for (count, bytes) in [
            (n, d * 4 * 2),
            (n, d * 2 * 4),
            (rows, 3 * d * 2),
            (n, f * 2 * 3),
            (n, (HEAD_DIM as u64 + 1 + LATENT as u64) * 4),
        ] {
            total = count
                .checked_mul(bytes)
                .and_then(|v| total.checked_add(v))
                .ok_or_else(|| Error("activation byte count overflow".into()))?;
        }
        Ok(total)
    }
    let live = step(image_rows, prefix_rows)?.max(step(prefix_rows, 0)?);
    prefix_rows
        .checked_mul(3 * WIDTH as u64 * 2 * LAYERS as u64)
        .and_then(|cache| live.checked_add(cache))
        .ok_or_else(|| Error("prefix cache byte count overflow".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextrocm_gguf::GType;

    fn fixture() -> Vec<Tensor> {
        tensor_contract()
            .into_iter()
            .map(|s| {
                let ty = GType::from_code(s.quant_type).unwrap();
                let bytes = ty.bytes(s.shape.iter().product()).unwrap();
                Tensor {
                    name: s.name,
                    ty,
                    shape: s.shape,
                    shard: 0,
                    offset: 0,
                    bytes,
                }
            })
            .collect()
    }

    #[test]
    fn pinned_header_totals_and_rejects() {
        let mut tensors = fixture();
        let plan = DitPlan::validate_tensors(&tensors).unwrap();
        // Independently parsed byte counts from the real pinned GGUF prefix.
        assert_eq!(plan.packed_bytes, 4_189_323_264);
        assert_eq!(tensors.len(), 297);
        tensors[0].shape.swap(0, 1);
        assert!(DitPlan::validate_tensors(&tensors).is_err());
        let mut tensors = fixture();
        tensors[0].name = tensors[1].name.clone();
        assert!(DitPlan::validate_tensors(&tensors).is_err());
    }

    #[test]
    fn activation_estimate_is_checked() {
        assert_eq!(activation_bytes(4096, 512).unwrap(), 1_089_486_848);
        assert_eq!(
            activation_bytes(1, 512).unwrap(),
            activation_bytes(0, 512).unwrap()
        );
        assert!(activation_bytes(u64::MAX, 1).is_err());
        assert!(activation_bytes(1, u64::MAX / 2).is_err());
    }
}
