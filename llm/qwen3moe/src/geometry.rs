use super::{error, Result};
use nextrocm_gguf::{GType, Gguf, Tensor};

/// Validated full-attention Qwen3-MoE geometry; shapes are outermost first.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub arch: &'static str,
    pub layers: usize,
    pub width: usize,
    pub ff: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head: usize,
    pub rotary: usize,
    pub vocab: usize,
    pub context: usize,
    pub experts: usize,
    pub used: usize,
    pub eps: f32,
    pub theta: f32,
}
impl Geometry {
    /// Read metadata and validate every required tensor without allocating GPU memory.
    pub fn read(f: &Gguf) -> Result<Self> {
        if f.architecture() != "qwen3moe" {
            return Err(error("expected qwen3moe GGUF"));
        }
        let n = |k: &str| -> Result<usize> {
            usize::try_from(
                f.arch_meta(k)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| error(format!("missing qwen3moe.{k}")))?,
            )
            .map_err(|_| error("metadata integer overflow"))
        };
        let scalar = |k: &str| -> Result<f32> {
            let v = f
                .arch_meta(k)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| error(format!("missing qwen3moe.{k}")))? as f32;
            if !v.is_finite() || v <= 0. {
                return Err(error(format!("invalid {k}")));
            }
            Ok(v)
        };
        let g = Self {
            arch: "qwen3moe",
            layers: n("block_count")?,
            width: n("embedding_length")?,
            ff: n("expert_feed_forward_length")?,
            heads: n("attention.head_count")?,
            kv_heads: n("attention.head_count_kv")?,
            head: n("attention.key_length")?,
            rotary: if f.arch_meta("rope.dimension_count").is_some() {
                n("rope.dimension_count")?
            } else {
                n("attention.key_length")?
            },
            vocab: f
                .tensor("token_embd.weight")
                .and_then(|t| t.shape.first())
                .and_then(|v| usize::try_from(*v).ok())
                .ok_or_else(|| error("missing vocabulary"))?,
            context: n("context_length")?,
            experts: n("expert_count")?,
            used: n("expert_used_count")?,
            eps: scalar("attention.layer_norm_rms_epsilon")?,
            theta: scalar("rope.freq_base")?,
        };
        g.validate()?;
        if f.arch_meta("attention.value_length").is_some() && n("attention.value_length")? != g.head
            || f.arch_meta("expert_shared_feed_forward_length").is_some()
                && n("expert_shared_feed_forward_length")? != 0
            || f.arch_meta("rope.scaling.type")
                .and_then(|v| v.as_str())
                .is_some_and(|v| v != "none")
            || f.arch_meta("rope.scaling.factor")
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v != 1.)
        {
            return Err(error(
                "unsupported value head, shared expert or scaled RoPE",
            ));
        }
        for (name, shape) in g.specs() {
            if name == "output.weight" && f.tensor(&name).is_none() {
                continue;
            }
            validate_tensor(
                f.tensor(&name)
                    .ok_or_else(|| error(format!("missing {name}")))?,
                &shape,
            )?;
        }
        Ok(g)
    }
    pub fn validate(&self) -> Result<()> {
        if self.layers == 0
            || self.layers > 256
            || self.width == 0
            || self.width > 32768
            || self.ff == 0
            || self.ff > 131072
            || self.heads == 0
            || self.heads > 256
            || self.kv_heads == 0
            || !self.heads.is_multiple_of(self.kv_heads)
            || self.head == 0
            || self.head > 256
            || self.rotary == 0
            || self.rotary > self.head
            || !self.rotary.is_multiple_of(2)
            || self.vocab == 0
            || self.vocab > 1_048_576
            || self.context == 0
            || self.context > 1_048_576
            || self.experts == 0
            || self.experts > 1024
            || self.used == 0
            || self.used > self.experts
            || !self.eps.is_finite()
            || self.eps <= 0.
            || !self.theta.is_finite()
            || self.theta <= 0.
        {
            return Err(error("invalid Qwen3-MoE geometry"));
        }
        self.cache_bytes(self.context)?;
        Ok(())
    }
    pub(crate) fn layer_bytes(&self, _layer: usize, ctx: usize) -> Result<(usize, usize)> {
        let n = ctx
            .checked_mul(self.kv_heads)
            .and_then(|n| n.checked_mul(self.head))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| error("KV size overflow"))?;
        Ok((n, n))
    }
    /// FP16 key and value cache bytes for all layers at the requested capacity.
    pub fn cache_bytes(&self, ctx: usize) -> Result<usize> {
        self.layer_bytes(0, ctx)?
            .0
            .checked_mul(2)
            .and_then(|n| n.checked_mul(self.layers))
            .ok_or_else(|| error("KV size overflow"))
    }
    pub(crate) fn specs(&self) -> Vec<(String, Vec<u64>)> {
        let mut out = Vec::new();
        let mut push =
            |n: String, s: Vec<usize>| out.push((n, s.into_iter().map(|v| v as u64).collect()));
        let d = self.width;
        let q = self.heads * self.head;
        let kv = self.kv_heads * self.head;
        push("token_embd.weight".into(), vec![self.vocab, d]);
        push("output.weight".into(), vec![self.vocab, d]);
        push("output_norm.weight".into(), vec![d]);
        for l in 0..self.layers {
            let mut p = |n: &str, s: Vec<usize>| push(format!("blk.{l}.{n}"), s);
            for n in ["attn_norm.weight", "ffn_norm.weight"] {
                p(n, vec![d]);
            }
            for (n, r, c) in [
                ("attn_q", q, d),
                ("attn_k", kv, d),
                ("attn_v", kv, d),
                ("attn_output", d, q),
            ] {
                p(&format!("{n}.weight"), vec![r, c]);
            }
            for n in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                p(n, vec![self.head]);
            }
            p("ffn_gate_inp.weight", vec![self.experts, d]);
            for n in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
                p(n, vec![self.experts, self.ff, d]);
            }
            p("ffn_down_exps.weight", vec![self.experts, d, self.ff]);
        }
        out
    }
}
fn validate_tensor(t: &Tensor, shape: &[u64]) -> Result<()> {
    if t.shape != shape {
        return Err(error(format!(
            "{} shape {:?}, expected {shape:?}",
            t.name, t.shape
        )));
    }
    if shape.len() == 1 && t.ty != GType::F32
        || !matches!(t.ty, GType::F32 | GType::F16 | GType::Q4K | GType::Q6K)
    {
        return Err(error(format!("unsupported {} type {:?}", t.name, t.ty)));
    }
    let elems = shape
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| error("tensor extent overflow"))?;
    let cols = *shape.last().ok_or_else(|| error("empty shape"))?;
    if cols % t.ty.block().0 as u64 != 0 || t.ty.bytes(elems) != Some(t.bytes) {
        return Err(error("invalid packed row length"));
    }
    Ok(())
}
