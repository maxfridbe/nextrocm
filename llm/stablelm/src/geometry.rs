use nextrocm_gguf::{GType, Gguf, Tensor};
use nextrocm_llm::{Error, Result};

/// Deliberately limited to the original 3B sequential-residual StableLM family.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub layers: usize,
    pub width: usize,
    pub ff: usize,
    pub heads: usize,
    pub head: usize,
    pub rotary: usize,
    pub vocab: usize,
    pub context: usize,
    pub theta: f32,
    pub eps: f32,
}
fn err(s: impl Into<String>) -> Error {
    Error(s.into())
}
impl Geometry {
    pub fn read(f: &Gguf) -> Result<Self> {
        if f.architecture() != "stablelm" {
            return Err(err("expected general.architecture=stablelm"));
        }
        let n = |key: &str| -> Result<usize> {
            let v = f
                .arch_meta(key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| err(format!("missing stablelm.{key}")))?;
            usize::try_from(v).map_err(|_| err(format!("stablelm.{key} overflows usize")))
        };
        let scalar = |key: &str| -> Result<f32> {
            let v = f
                .arch_meta(key)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| err(format!("missing stablelm.{key}")))? as f32;
            if !v.is_finite() || v <= 0.0 {
                return Err(err(format!("invalid stablelm.{key}")));
            }
            Ok(v)
        };
        if f.arch_meta("rope.scaling.type")
            .and_then(|v| v.as_str())
            .is_some_and(|v| v != "none")
            || f.arch_meta("rope.scaling.factor")
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v != 1.0)
            || f.arch_meta("rope.freq_scale")
                .and_then(|v| v.as_f64())
                .is_some_and(|v| v != 1.0)
        {
            return Err(err("scaled RoPE is outside this StableLM backend's scope"));
        }
        let width = n("embedding_length")?;
        let heads = n("attention.head_count")?;
        let layers = n("block_count")?;
        let ff = n("feed_forward_length")?;
        let rotary = n("rope.dimension_count")?;
        let context = n("context_length")?;
        if (layers, width, ff, heads, rotary) != (32, 2560, 6912, 32, 20)
            || f.arch_meta("attention.head_count_kv")
                .and_then(|v| v.as_u64())
                .unwrap_or(heads as u64)
                != heads as u64
        {
            return Err(err("only StableLM 3B (32 layers, width 2560, FF 6912, 32 MHA heads, 20 rotary channels) is supported"));
        }
        if !(1..=16384).contains(&context) {
            return Err(err("StableLM context must be 1..=16384"));
        }
        // The installed Stable Code GGUFs omit theta. Do not silently use Zephyr's 1e4.
        // Match the exact inventory filenames (or general.name), not arbitrary geometry.
        let filename = f
            .paths
            .first()
            .and_then(|p| p.file_name())
            .and_then(|p| p.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let name = f
            .meta("general.name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let code = filename.starts_with("stable-code-3b.")
            || filename.starts_with("stable-code-instruct-3b.")
            || filename.starts_with("stable-code-instruct-3b-")
            || name.contains("stable-code");
        let zephyr =
            filename.starts_with("stablelm-zephyr-3b.") || name.contains("stablelm-zephyr");
        let theta = if f.arch_meta("rope.freq_base").is_some() {
            scalar("rope.freq_base")?
        } else if code {
            1_000_000.0
        } else if zephyr {
            10_000.0
        } else {
            return Err(err("missing stablelm.rope.freq_base; cannot identify an inventory model for the documented fallback"));
        };
        let vocab = f
            .tensor("token_embd.weight")
            .and_then(|t| t.shape.first())
            .copied()
            .ok_or_else(|| err("missing token_embd.weight"))?;
        if vocab != 50304 {
            return Err(err(
                "only the 50304-row StableLM 3B vocabulary is supported",
            ));
        }
        let g = Self {
            layers,
            width,
            ff,
            heads,
            head: width / heads,
            rotary,
            vocab: vocab as usize,
            context,
            theta,
            eps: scalar("attention.layer_norm_epsilon")?,
        };
        g.validate_tensors(f)?;
        Ok(g)
    }
    pub(crate) fn specs(&self) -> Vec<(String, Vec<u64>)> {
        let d = self.width as u64;
        let ff = self.ff as u64;
        let v = self.vocab as u64;
        let mut out = vec![
            ("token_embd.weight".into(), vec![v, d]),
            ("output.weight".into(), vec![v, d]),
            ("output_norm.weight".into(), vec![d]),
            ("output_norm.bias".into(), vec![d]),
        ];
        for l in 0..self.layers {
            for n in ["attn_norm", "ffn_norm"] {
                for suffix in ["weight", "bias"] {
                    out.push((format!("blk.{l}.{n}.{suffix}"), vec![d]));
                }
            }
            for n in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                out.push((format!("blk.{l}.{n}.weight"), vec![d, d]));
            }
            for n in ["ffn_gate", "ffn_up"] {
                out.push((format!("blk.{l}.{n}.weight"), vec![ff, d]));
            }
            out.push((format!("blk.{l}.ffn_down.weight"), vec![d, ff]));
        }
        out
    }
    fn validate_tensors(&self, f: &Gguf) -> Result<()> {
        let specs = self.specs();
        for (name, shape) in &specs {
            let t = f
                .tensor(name)
                .ok_or_else(|| err(format!("missing {name}")))?;
            validate_tensor(t, shape)?;
        }
        // No silently ignored QKV biases, q/k norms, tied output or parallel residual variants.
        for t in &f.tensors {
            if !specs.iter().any(|(n, _)| n == &t.name) {
                return Err(err(format!("unsupported StableLM tensor {}", t.name)));
            }
        }
        Ok(())
    }
    pub(crate) fn cache_bytes(&self, ctx: usize) -> Result<usize> {
        ctx.checked_mul(self.layers)
            .and_then(|n| n.checked_mul(self.width))
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| err("KV size overflow"))
    }
}
fn validate_tensor(t: &Tensor, shape: &[u64]) -> Result<()> {
    if t.shape != shape {
        return Err(err(format!(
            "{} shape {:?}, expected {shape:?}",
            t.name, t.shape
        )));
    }
    let allowed = if shape.len() == 1 {
        t.ty == GType::F32
    } else {
        matches!(t.ty, GType::F32 | GType::F16 | GType::Q4K | GType::Q6K)
    };
    if !allowed {
        return Err(err(format!("unsupported {} type {:?}", t.name, t.ty)));
    }
    let elements = shape
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| err("tensor size overflow"))?;
    if t.ty.bytes(elements) != Some(t.bytes) {
        return Err(err(format!("{} has invalid byte length", t.name)));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_wrong_shape_and_type() {
        let mut t = Tensor {
            name: "test".into(),
            ty: GType::Q4K,
            shape: vec![2, 256],
            shard: 0,
            offset: 0,
            bytes: 288,
        };
        assert!(validate_tensor(&t, &[2, 256]).is_ok());
        assert!(validate_tensor(&t, &[256, 2]).is_err());
        t.ty = GType::Q5K;
        assert!(validate_tensor(&t, &[2, 256]).is_err());
        t.ty = GType::Q4K;
        t.bytes = 1;
        assert!(validate_tensor(&t, &[2, 256]).is_err());
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    use nextrocm_gguf::{writer::Writer, Value};
    #[test]
    fn inventory_geometry_and_missing_code_theta() {
        let geom = Geometry {
            layers: 32,
            width: 2560,
            ff: 6912,
            heads: 32,
            head: 80,
            rotary: 20,
            vocab: 50304,
            context: 16384,
            theta: 1e6,
            eps: 1e-5,
        };
        let path = std::env::temp_dir().join(format!(
            "stablelm-geometry-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let meta = vec![
            ("general.architecture", Value::Str("stablelm".into())),
            ("general.name", Value::Str("stable-code-3b".into())),
            ("stablelm.block_count", Value::U(32)),
            ("stablelm.embedding_length", Value::U(2560)),
            ("stablelm.feed_forward_length", Value::U(6912)),
            ("stablelm.attention.head_count", Value::U(32)),
            ("stablelm.attention.head_count_kv", Value::U(32)),
            ("stablelm.rope.dimension_count", Value::U(20)),
            ("stablelm.context_length", Value::U(16384)),
            ("stablelm.attention.layer_norm_epsilon", Value::F(1e-5)),
        ];
        let tensors: Vec<_> = geom
            .specs()
            .into_iter()
            .map(|(n, s)| {
                let ty = if s.len() == 1 {
                    GType::F32
                } else if n == "output.weight" {
                    GType::Q6K
                } else {
                    GType::Q4K
                };
                (n, ty, s)
            })
            .collect();
        // Only the header and sparse file length are written; no 3B weights allocated/read.
        drop(Writer::create(&path, &meta, &tensors).unwrap());
        let header = std::fs::metadata(&path).unwrap().len();
        let size: u64 = tensors
            .iter()
            .map(|(_, t, s)| t.bytes(s.iter().product()).unwrap().div_ceil(32) * 32)
            .sum();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(header + size)
            .unwrap();
        let mut f = Gguf::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let g = Geometry::read(&f).unwrap();
        assert_eq!(g.theta, 1e6);
        assert_eq!(g.cache_bytes(1).unwrap(), 327680);
        f.metadata.insert(
            "general.name".into(),
            Value::Str("stablelm-zephyr-3b".into()),
        );
        assert_eq!(Geometry::read(&f).unwrap().theta, 1e4);
        f.metadata
            .insert("stablelm.rope.freq_base".into(), Value::F(2e4));
        assert_eq!(Geometry::read(&f).unwrap().theta, 2e4);
        f.metadata.insert(
            "stablelm.rope.scaling.type".into(),
            Value::Str("linear".into()),
        );
        assert!(Geometry::read(&f).is_err());
        f.metadata.remove("stablelm.rope.scaling.type");
        f.metadata
            .insert("stablelm.attention.head_count_kv".into(), Value::U(8));
        assert!(Geometry::read(&f).is_err());
    }
}
