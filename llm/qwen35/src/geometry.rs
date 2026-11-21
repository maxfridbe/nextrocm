use super::{error, Result};
use nextrocm_gguf::{GType, Gguf, Tensor};

/// Validated text geometry. GGUF shapes here are outermost first.
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
    pub k_heads: usize,
    pub v_heads: usize,
    pub state: usize,
    pub conv: usize,
    pub interval: usize,
    pub vocab: usize,
    pub context: usize,
    pub experts: usize,
    pub used: usize,
    pub shared: usize,
    pub eps: f32,
    pub theta: f32,
}
pub(crate) fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("geometry size overflow"))
}
impl Geometry {
    pub fn read(f: &Gguf) -> Result<Self> {
        let arch = match f.architecture() {
            "qwen35" => "qwen35",
            "qwen35moe" => "qwen35moe",
            _ => {
                return Err(error(
                    "expected qwen35 or qwen35moe GGUF (HF qwen3_5_moe must be converted)",
                ))
            }
        };
        let n = |k: &str| -> Result<usize> {
            usize::try_from(
                f.arch_meta(k)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| error(format!("missing {arch}.{k}")))?,
            )
            .map_err(|_| error("metadata integer overflow"))
        };
        let scalar = |k: &str| -> Result<f32> {
            let v = f
                .arch_meta(k)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| error(format!("missing {arch}.{k}")))? as f32;
            if !v.is_finite() || v <= 0.0 {
                return Err(error(format!("invalid {k}")));
            }
            Ok(v)
        };
        let nextn = if f.arch_meta("nextn_predict_layers").is_some() {
            n("nextn_predict_layers")?
        } else {
            0
        };
        let moe = arch == "qwen35moe";
        let g = Self {
            arch,
            layers: n("block_count")?
                .checked_sub(nextn)
                .ok_or_else(|| error("invalid nextn layer count"))?,
            width: n("embedding_length")?,
            ff: if moe {
                n("expert_feed_forward_length")?
            } else {
                n("feed_forward_length")?
            },
            heads: n("attention.head_count")?,
            kv_heads: n("attention.head_count_kv")?,
            head: n("attention.key_length")?,
            rotary: n("rope.dimension_count")?,
            k_heads: n("ssm.group_count")?,
            v_heads: n("ssm.time_step_rank")?,
            state: n("ssm.state_size")?,
            conv: n("ssm.conv_kernel")?,
            interval: if f.arch_meta("full_attention_interval").is_some() {
                n("full_attention_interval")?
            } else {
                4
            },
            vocab: f
                .tensor("token_embd.weight")
                .and_then(|t| t.shape.first())
                .and_then(|v| usize::try_from(*v).ok())
                .ok_or_else(|| error("missing embedding vocabulary"))?,
            context: n("context_length")?,
            experts: if moe { n("expert_count")? } else { 0 },
            used: if moe { n("expert_used_count")? } else { 0 },
            shared: if moe {
                n("expert_shared_feed_forward_length")?
            } else {
                0
            },
            eps: scalar("attention.layer_norm_rms_epsilon")?,
            theta: scalar("rope.freq_base")?,
        };
        g.validate()?;
        if n("ssm.inner_size")? != g.inner() {
            return Err(error("ssm.inner_size != value heads * state"));
        }
        if f.arch_meta("attention.value_length").is_some() && n("attention.value_length")? != g.head
        {
            return Err(error("unequal full-attention key/value dimensions"));
        }
        if f.arch_meta("rope.scaling.type")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v != "none")
            || f.arch_meta("rope.scaling.factor")
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v != 1.0)
        {
            return Err(error("scaled RoPE is unsupported"));
        }
        for (name, shape) in g.specs() {
            if name == "output.weight" && f.tensor(&name).is_none() {
                continue;
            }
            let t = f
                .tensor(&name)
                .ok_or_else(|| error(format!("missing {name}")))?;
            validate_tensor(t, &shape)?;
        }
        Ok(g)
    }
    pub fn validate(&self) -> Result<()> {
        for v in [
            self.layers,
            self.width,
            self.ff,
            self.heads,
            self.kv_heads,
            self.head,
            self.rotary,
            self.k_heads,
            self.v_heads,
            self.state,
            self.interval,
            self.vocab,
            self.context,
        ] {
            if v == 0 || v > 1_048_576 {
                return Err(error("zero or excessive geometry dimension"));
            }
        }
        if self.layers > 256
            || self.width > 32768
            || self.ff > 131072
            || self.head > 256
            || self.state > 256
            || self.heads > 256
            || self.kv_heads > self.heads
            || self.v_heads > 256
            || !self.v_heads.is_multiple_of(self.k_heads)
            || !self.heads.is_multiple_of(self.kv_heads)
            || self.rotary > self.head
            || !self.rotary.is_multiple_of(2)
            || !(2..=9).contains(&self.conv)
        {
            return Err(error("unsupported attention/DeltaNet geometry"));
        }
        if self.arch == "qwen35moe"
            && (self.experts == 0
                || self.experts > 1024
                || self.used == 0
                || self.used > self.experts
                || self.shared == 0
                || self.shared > 131072)
        {
            return Err(error("invalid expert geometry"));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 || !self.theta.is_finite() || self.theta <= 0.0
        {
            return Err(error("invalid norm epsilon or RoPE base"));
        }
        self.cache_bytes(self.context)?;
        Ok(())
    }
    pub fn full(&self, l: usize) -> bool {
        (l + 1).is_multiple_of(self.interval)
    }
    pub fn inner(&self) -> usize {
        self.v_heads * self.state
    }
    pub fn channels(&self) -> usize {
        2 * self.k_heads * self.state + self.inner()
    }
    pub(crate) fn layer_bytes(&self, l: usize, ctx: usize) -> Result<(usize, usize)> {
        if self.full(l) {
            let n = mul(mul(ctx, self.kv_heads * self.head)?, 2)?;
            Ok((n, n))
        } else {
            Ok((
                mul(self.inner(), self.state * 4)?,
                mul(self.conv - 1, self.channels() * 4)?,
            ))
        }
    }
    pub fn cache_bytes(&self, ctx: usize) -> Result<usize> {
        (0..self.layers).try_fold(0usize, |s, l| {
            let (a, b) = self.layer_bytes(l, ctx)?;
            s.checked_add(a)
                .and_then(|s| s.checked_add(b))
                .ok_or_else(|| error("cache overflow"))
        })
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
            p("attn_norm.weight", vec![d]);
            p("post_attention_norm.weight", vec![d]);
            if self.full(l) {
                for (n, r, c) in [
                    ("attn_q", 2 * q, d),
                    ("attn_k", kv, d),
                    ("attn_v", kv, d),
                    ("attn_output", d, q),
                ] {
                    p(&format!("{n}.weight"), vec![r, c]);
                }
                p("attn_q_norm.weight", vec![self.head]);
                p("attn_k_norm.weight", vec![self.head]);
            } else {
                for (n, r, c) in [
                    ("attn_qkv", self.channels(), d),
                    ("attn_gate", self.inner(), d),
                    ("ssm_beta", self.v_heads, d),
                    ("ssm_alpha", self.v_heads, d),
                    ("ssm_out", d, self.inner()),
                ] {
                    p(&format!("{n}.weight"), vec![r, c]);
                }
                p("ssm_conv1d.weight", vec![self.channels(), self.conv]);
                p("ssm_dt.bias", vec![self.v_heads]);
                p("ssm_a", vec![self.v_heads]);
                p("ssm_norm.weight", vec![self.state]);
            }
            if self.experts == 0 {
                for n in ["ffn_gate", "ffn_up"] {
                    p(&format!("{n}.weight"), vec![self.ff, d]);
                }
                p("ffn_down.weight", vec![d, self.ff]);
            } else {
                p("ffn_gate_inp.weight", vec![self.experts, d]);
                for n in ["ffn_gate_exps", "ffn_up_exps"] {
                    p(&format!("{n}.weight"), vec![self.experts, self.ff, d]);
                }
                p("ffn_down_exps.weight", vec![self.experts, d, self.ff]);
                for n in ["ffn_gate_shexp", "ffn_up_shexp"] {
                    p(&format!("{n}.weight"), vec![self.shared, d]);
                }
                p("ffn_down_shexp.weight", vec![d, self.shared]);
                p("ffn_gate_inp_shexp.weight", vec![d]);
            }
        }
        out
    }
}
pub(crate) fn validate_tensor(t: &Tensor, shape: &[u64]) -> Result<()> {
    if t.shape != shape {
        return Err(error(format!(
            "{} shape {:?}, expected {shape:?}",
            t.name, t.shape
        )));
    }
    let small = shape.len() == 1 || t.name.ends_with("ssm_conv1d.weight");
    if (small && t.ty != GType::F32)
        || (!small
            && !matches!(
                t.ty,
                GType::F32 | GType::F16 | GType::Q4K | GType::Q5K | GType::Q6K | GType::Q8_0
            ))
    {
        return Err(error(format!("unsupported {} type {:?}", t.name, t.ty)));
    }
    let elems = shape
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| error("tensor extent overflow"))?;
    let cols = *shape.last().ok_or_else(|| error("empty shape"))?;
    if cols % t.ty.block().0 as u64 != 0 || t.ty.bytes(elems) != Some(t.bytes) {
        return Err(error(format!(
            "invalid packed row/byte length for {}",
            t.name
        )));
    }
    Ok(())
}
