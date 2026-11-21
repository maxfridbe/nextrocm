use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_gguf::safetensors::{SafeTensors, StTensor};
use std::{fs::File, os::unix::fs::FileExt, path::Path, sync::Arc};
pub const HIDDEN: usize = 4096;
pub const FFN: usize = 12288;
pub const LAYERS: usize = 36;
pub const VOCAB: usize = 151936;
// I8 matrices + F32 channel scales + four expanded F32 norm vectors.
pub const BLOCK_BYTES: usize = 192_937_984 + 155_648 + 33_792;
fn ge(e: nextrocm_gguf::Error) -> Error {
    Error(e.0)
}
pub struct Weights {
    file: SafeTensors,
}
pub struct Matrix {
    pub w: DevBuf,
    pub scales: DevBuf,
    pub n: usize,
    pub k: usize,
}
pub struct Block {
    pub input: DevBuf,
    pub post: DevBuf,
    pub qnorm: DevBuf,
    pub knorm: DevBuf,
    pub qkv: Matrix,
    pub out: Matrix,
    pub gu: Matrix,
    pub down: Matrix,
}
// Check all absolute spans before invoking the shared format reader. No shared
// parser changes; malformed offsets cannot wrap inside that reader.
fn preflight(path: &Path) -> Result<()> {
    let file = File::open(path).map_err(|e| Error(e.to_string()))?;
    let len = file.metadata().map_err(|e| Error(e.to_string()))?.len();
    let mut prefix = [0; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| Error(e.to_string()))?;
    let header = u64::from_le_bytes(prefix);
    if header == 0
        || header > 4 * 1024 * 1024
        || header.checked_add(8).filter(|b| *b <= len).is_none()
    {
        return Err(Error("TE header bound".into()));
    }
    let base = header + 8;
    let mut bytes = vec![0; header as usize];
    file.read_exact_at(&mut bytes, 8)
        .map_err(|e| Error(e.to_string()))?;
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| Error(e.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| Error("TE header object".into()))?;
    let mut spans = Vec::new();
    for (name, t) in object {
        if name == "__metadata__" {
            continue;
        }
        let offsets = t["data_offsets"]
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or_else(|| Error(format!("{name}: offsets")))?;
        let a = offsets[0]
            .as_u64()
            .ok_or_else(|| Error(format!("{name}: offset")))?;
        let b = offsets[1]
            .as_u64()
            .ok_or_else(|| Error(format!("{name}: offset")))?;
        if a > b || b > len - base {
            return Err(Error(format!("{name}: file bounds")));
        }
        let shape = t["shape"]
            .as_array()
            .ok_or_else(|| Error(format!("{name}: shape")))?;
        let mut elements = 1u64;
        for d in shape {
            elements = elements
                .checked_mul(
                    d.as_u64()
                        .ok_or_else(|| Error(format!("{name}: dimension")))?,
                )
                .ok_or_else(|| Error("TE shape overflow".into()))?;
        }
        let width = match t["dtype"].as_str() {
            Some("I8" | "U8" | "F8_E4M3") => 1,
            Some("BF16" | "F16") => 2,
            Some("F32" | "I32") => 4,
            Some("I64") => 8,
            _ => return Err(Error(format!("{name}: unsupported dtype"))),
        };
        if elements.checked_mul(width) != Some(b - a) {
            return Err(Error(format!("{name}: dtype/shape byte mismatch")));
        }
        spans.push((a, b));
    }
    spans.sort_unstable();
    if spans.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err(Error("TE overlapping tensor spans".into()));
    }
    Ok(())
}
impl Weights {
    pub fn open(path: &Path) -> Result<Self> {
        preflight(path)?;
        let this = Self {
            file: SafeTensors::open(path).map_err(ge)?,
        };
        this.tensor("model.embed_tokens.weight", "I8", &[VOCAB, HIDDEN])?;
        this.tensor("model.embed_tokens.weight_scale", "F32", &[VOCAB, 1])?;
        this.quant("model.embed_tokens")?;
        for l in 0..LAYERS {
            let p = format!("model.layers.{l}.");
            for (suffix, n, k) in [
                ("self_attn.q_proj", 4096, 4096),
                ("self_attn.k_proj", 1024, 4096),
                ("self_attn.v_proj", 1024, 4096),
                ("self_attn.o_proj", 4096, 4096),
                ("mlp.gate_proj", 12288, 4096),
                ("mlp.up_proj", 12288, 4096),
                ("mlp.down_proj", 4096, 12288),
            ] {
                let name = format!("{p}{suffix}");
                this.tensor(&format!("{name}.weight"), "I8", &[n, k])?;
                this.tensor(&format!("{name}.weight_scale"), "F32", &[n, 1])?;
                this.quant(&name)?;
            }
            for (name, n) in [
                ("input_layernorm", 4096),
                ("post_attention_layernorm", 4096),
                ("self_attn.q_norm", 128),
                ("self_attn.k_norm", 128),
            ] {
                this.tensor(&format!("{p}{name}.weight"), "BF16", &[n])?;
            }
        }
        if this
            .file
            .tensor("model.layers.36.input_layernorm.weight")
            .is_some()
        {
            return Err(Error("TE layer count differs from36".into()));
        }
        Ok(this)
    }
    fn tensor(&self, name: &str, dtype: &str, shape: &[usize]) -> Result<&StTensor> {
        let t = self.file.need(name).map_err(ge)?;
        if t.dtype != dtype || t.shape != shape.iter().map(|v| *v as u64).collect::<Vec<_>>() {
            return Err(Error(format!("{name}: unexpected dtype/shape")));
        }
        Ok(t)
    }
    fn quant(&self, name: &str) -> Result<()> {
        let t = self.file.need(&format!("{name}.comfy_quant")).map_err(ge)?;
        if t.dtype != "U8" || t.bytes > 1024 {
            return Err(Error(format!("{name}: quant metadata bound")));
        }
        let value: serde_json::Value = serde_json::from_slice(&self.file.read(t).map_err(ge)?)
            .map_err(|e| Error(e.to_string()))?;
        if value["format"] != "int8_tensorwise"
            || value["convrot"] != true
            || value["convrot_groupsize"] != 256
        {
            return Err(Error(format!(
                "{name}: expected stored INT8 ConvRot256 (no requantization)"
            )));
        }
        Ok(())
    }
    fn norm(&self, gpu: &Arc<Gpu>, name: &str) -> Result<DevBuf> {
        let v = self.file.f32(name).map_err(ge)?;
        if v.iter().any(|v| !v.is_finite()) {
            return Err(Error(format!("{name}: nonfinite norm")));
        }
        DevBuf::from_f32(gpu, &v)
    }
    fn matrix(&self, gpu: &Arc<Gpu>, names: &[String], k: usize) -> Result<Matrix> {
        let n = names.iter().try_fold(0usize, |n, name| {
            self.file
                .need(&format!("{name}.weight"))
                .map(|t| n + t.shape[0] as usize)
                .map_err(ge)
        })?;
        let w = DevBuf::new(gpu, n * k)?;
        let mut scales = Vec::with_capacity(n);
        let mut offset = 0;
        for name in names {
            let t = self.file.need(&format!("{name}.weight")).map_err(ge)?;
            // Each source matrix <=48MiB; positional read and synchronous upload before
            // releasing that host vector. No concatenated host copy or whole-file mmap.
            let raw = self.file.read(t).map_err(ge)?;
            w.write(offset, &raw)?;
            offset += raw.len();
            scales.extend(self.file.f32(&format!("{name}.weight_scale")).map_err(ge)?);
        }
        if scales.len() != n || scales.iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(Error("TE invalid channel scales".into()));
        }
        Ok(Matrix {
            w,
            scales: DevBuf::from_f32(gpu, &scales)?,
            n,
            k,
        })
    }
    pub fn block(&self, gpu: &Arc<Gpu>, layer: usize, allowance: usize) -> Result<Block> {
        if layer >= LAYERS || allowance < BLOCK_BYTES {
            return Err(Error("TE block exceeds index/weight allowance".into()));
        }
        let p = format!("model.layers.{layer}.");
        let mat = |names: &[&str], k| {
            self.matrix(
                gpu,
                &names.iter().map(|n| format!("{p}{n}")).collect::<Vec<_>>(),
                k,
            )
        };
        Ok(Block {
            input: self.norm(gpu, &format!("{p}input_layernorm.weight"))?,
            post: self.norm(gpu, &format!("{p}post_attention_layernorm.weight"))?,
            qnorm: self.norm(gpu, &format!("{p}self_attn.q_norm.weight"))?,
            knorm: self.norm(gpu, &format!("{p}self_attn.k_norm.weight"))?,
            qkv: mat(
                &["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                HIDDEN,
            )?,
            out: mat(&["self_attn.o_proj"], HIDDEN)?,
            gu: mat(&["mlp.gate_proj", "mlp.up_proj"], HIDDEN)?,
            down: mat(&["mlp.down_proj"], FFN)?,
        })
    }
    pub fn embeddings(&self, ids: &[u32]) -> Result<Vec<f32>> {
        if ids.is_empty() || ids.len() > 4096 || ids.iter().any(|i| *i as usize >= VOCAB) {
            return Err(Error("TE token bounds".into()));
        }
        let t = self.file.need("model.embed_tokens.weight").map_err(ge)?;
        let s = self
            .file
            .need("model.embed_tokens.weight_scale")
            .map_err(ge)?;
        let mut out = vec![0.; ids.len() * HIDDEN];
        let mut raw = vec![0; HIDDEN];
        let mut scale = [0; 4];
        for (row, id) in ids.iter().enumerate() {
            self.file
                .read_into(t, u64::from(*id) * HIDDEN as u64, &mut raw)
                .map_err(ge)?;
            self.file
                .read_into(s, u64::from(*id) * 4, &mut scale)
                .map_err(ge)?;
            let scale = f32::from_le_bytes(scale);
            if !scale.is_finite() || scale <= 0. {
                return Err(Error("TE embedding scale".into()));
            }
            for group in 0..HIDDEN / 256 {
                let mut src = [0.; 256];
                for k in 0..256 {
                    src[k] = raw[group * 256 + k] as i8 as f32 * scale;
                }
                for j in 0..256 {
                    let mut sum = 0.;
                    for (k, x) in src.iter().enumerate() {
                        sum += x * hadamard(j, k);
                    }
                    out[row * HIDDEN + group * 256 + j] = sum;
                }
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error("TE nonfinite embedding".into()));
        }
        Ok(out)
    }
}
pub fn hadamard(mut i: usize, mut j: usize) -> f32 {
    let mut sign = 1.;
    for _ in 0..4 {
        if (i & 3) + (j & 3) == 3 {
            sign = -sign;
        }
        i >>= 2;
        j >>= 2;
    }
    sign / 16.
}
