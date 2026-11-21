//! Checked original safetensors, with bounded host staging and one resident component.
use crate::kernels::{Weight, WeightType};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_gguf::safetensors::{SafeTensors, StTensor};
use nextrocm_image::{Error, Result};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path, sync::Arc};

/// Maximum payload chunk resident on the host while uploading a tensor.
pub const READ_CHUNK: usize = 64 << 20;
/// Hard ceiling for one resident component's packed weights.
pub const WEIGHT_CAP: usize = 2 << 30;

/// A validated file; the original utility owns all payload I/O.
pub struct Checkpoint {
    pub(crate) file: SafeTensors,
    external: bool,
}

impl Checkpoint {
    /// Validate every shape, dtype, byte range and architecture key before any upload.
    /// `external` selects the unprefixed external VAE namespace.
    pub fn open(path: &Path, external: bool) -> Result<Self> {
        // The general utility is permissive about JSON; validate raw values first.
        let mut f = File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let size = f.metadata().map_err(|e| Error(e.to_string()))?.len();
        let mut n = [0; 8];
        f.read_exact(&mut n).map_err(|e| Error(e.to_string()))?;
        let header = u64::from_le_bytes(n);
        if header == 0 || header > 4 << 20 || header.checked_add(8).is_none_or(|x| x > size) {
            return Err(Error(
                "SD1.5 invalid or oversized safetensors header".into(),
            ));
        }
        let mut bytes = vec![0; header as usize];
        f.read_exact(&mut bytes).map_err(|e| Error(e.to_string()))?;
        let raw: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| Error(e.to_string()))?;
        let map = raw
            .as_object()
            .ok_or_else(|| Error("tensor header must be an object".into()))?;
        let schema: BTreeMap<String, Vec<u64>> =
            serde_json::from_str(include_str!("shapes.json")).map_err(|e| Error(e.to_string()))?;
        let mut ranges = Vec::new();
        for (name, v) in map {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<u64> = serde_json::from_value(v["shape"].clone())
                .map_err(|e| Error(format!("{name}: invalid shape: {e}")))?;
            let dtype = v["dtype"]
                .as_str()
                .ok_or_else(|| Error(format!("{name}: missing dtype")))?;
            let width = match dtype {
                "F16" => 2,
                "F32" | "I32" => 4,
                "I64" => 8,
                _ => return Err(Error(format!("{name}: unsupported {dtype}"))),
            };
            let len = shape
                .iter()
                .try_fold(width, |n: u64, &d| n.checked_mul(d))
                .ok_or_else(|| Error(format!("{name}: shape overflow")))?;
            let offsets: Vec<u64> = serde_json::from_value(v["data_offsets"].clone())
                .map_err(|e| Error(e.to_string()))?;
            if offsets.len() != 2
                || offsets[1].checked_sub(offsets[0]) != Some(len)
                || offsets[1] > size - header - 8
            {
                return Err(Error(format!("{name}: invalid tensor byte range")));
            }
            let key = if external {
                format!("first_stage_model.{name}")
            } else {
                name.clone()
            };
            if let Some(expected) = schema.get(&key) {
                if &shape != expected {
                    return Err(Error(format!(
                        "{name}: shape {shape:?}, expected {expected:?}"
                    )));
                }
                let valid = if name.ends_with("position_ids") {
                    dtype == "I64"
                } else {
                    matches!(dtype, "F16" | "F32")
                };
                if !valid {
                    return Err(Error(format!("{name}: invalid parameter dtype {dtype}")));
                }
            } else if !(external
                && matches!(name.as_str(), "model_ema.decay" | "model_ema.num_updates")
                && len == 4)
            {
                return Err(Error(format!("{name}: not a supported SD1.5 tensor")));
            }
            ranges.push((offsets[0], offsets[1]));
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (a, b) in ranges {
            if a != end {
                return Err(Error("tensor ranges overlap or have gaps".into()));
            }
            end = b;
        }
        if end != size - header - 8 {
            return Err(Error("unaccounted safetensors payload".into()));
        }
        for name in schema
            .keys()
            .filter(|k| !external || k.starts_with("first_stage_model."))
        {
            let key = if external {
                name.trim_start_matches("first_stage_model.")
            } else {
                name
            };
            if !map.contains_key(key) {
                return Err(Error(format!("missing SD1.5 tensor {key}")));
            }
        }
        let file = SafeTensors::open(path).map_err(|e| Error(e.0))?;
        Ok(Self { file, external })
    }
    pub(crate) fn name(&self, name: &str) -> String {
        if self.external {
            name.trim_start_matches("first_stage_model.").into()
        } else {
            name.into()
        }
    }
    pub(crate) fn tensor(&self, name: &str) -> Result<&StTensor> {
        self.file.need(&self.name(name)).map_err(|e| Error(e.0))
    }
    pub(crate) fn floats(&self, name: &str) -> Result<Vec<f32>> {
        self.file.f32(&self.name(name)).map_err(|e| Error(e.0))
    }
    /// Load only this component. VAE loading deliberately excludes the encoder/quant_conv.
    pub(crate) fn stage(&self, gpu: &Arc<Gpu>, stage: Stage) -> Result<Weights> {
        let mut out = Weights {
            tensors: BTreeMap::new(),
            bytes: 0,
        };
        for t in &self.file.tensors {
            let name = if self.external {
                format!("first_stage_model.{}", t.name)
            } else {
                t.name.clone()
            };
            if !stage.contains(&name) || name.ends_with("position_ids") {
                continue;
            }
            // Embedding gather is host-side and only reads the selected rows.
            if name.contains(".embeddings.") {
                continue;
            }
            let bytes = usize::try_from(t.bytes).map_err(|_| Error("tensor too large".into()))?;
            out.bytes = out
                .bytes
                .checked_add(bytes)
                .filter(|n| *n <= WEIGHT_CAP)
                .ok_or_else(|| Error("SD1.5 component exceeds weight budget".into()))?;
            let buf = DevBuf::new(gpu, bytes)?;
            let mut chunk = vec![0; bytes.min(READ_CHUNK)];
            for at in (0..bytes).step_by(READ_CHUNK) {
                let n = (bytes - at).min(READ_CHUNK);
                self.file
                    .read_into(t, at as u64, &mut chunk[..n])
                    .map_err(|e| Error(e.0))?;
                buf.write(at, &chunk[..n])?;
            }
            let dtype = match t.dtype.as_str() {
                "F16" => WeightType::F16,
                "F32" => WeightType::F32,
                _ => return Err(Error("nonfloat model weight".into())),
            };
            out.tensors.insert(
                name,
                Parameter {
                    buf,
                    dtype,
                    shape: t.shape.iter().map(|x| *x as usize).collect(),
                },
            );
        }
        gpu.sync()?;
        Ok(out)
    }
    /// Gather 77 text embedding rows without holding a second complete embedding table.
    pub(crate) fn embeddings(&self, ids: &[u32; 77]) -> Result<Vec<f32>> {
        let name = "cond_stage_model.transformer.text_model.embeddings.token_embedding.weight";
        let t = self.tensor(name)?;
        let width = if t.dtype == "F16" { 2 } else { 4 };
        let mut row = vec![0; 768 * width];
        let mut out = self.floats(
            "cond_stage_model.transformer.text_model.embeddings.position_embedding.weight",
        )?;
        for (i, &id) in ids.iter().enumerate() {
            if id >= 49408 {
                return Err(Error("CLIP token outside vocabulary".into()));
            }
            self.file
                .read_into(t, id as u64 * row.len() as u64, &mut row)
                .map_err(|e| Error(e.0))?;
            for (j, b) in row.chunks_exact(width).enumerate() {
                let x = if width == 2 {
                    nextrocm_gguf::safetensors::f16_to_f32(u16::from_le_bytes([b[0], b[1]]))
                } else {
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                };
                out[i * 768 + j] += x;
            }
        }
        Ok(out)
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Stage {
    Clip,
    Unet,
    Vae,
}
impl Stage {
    fn contains(self, name: &str) -> bool {
        match self {
            Self::Clip => name.starts_with("cond_stage_model."),
            Self::Unet => name.starts_with("model.diffusion_model."),
            Self::Vae => {
                name.starts_with("first_stage_model.decoder.")
                    || name.starts_with("first_stage_model.post_quant_conv.")
            }
        }
    }
}
pub(crate) struct Parameter {
    buf: DevBuf,
    dtype: WeightType,
    pub shape: Vec<usize>,
}
impl Parameter {
    pub fn weight(&self) -> Weight<'_> {
        Weight::new(&self.buf, self.dtype)
    }
}
pub(crate) struct Weights {
    tensors: BTreeMap<String, Parameter>,
    pub bytes: usize,
}
impl Weights {
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Weight<'_>> {
        let p = self
            .tensors
            .get(name)
            .ok_or_else(|| Error(format!("missing resident parameter {name}")))?;
        if p.shape != shape {
            return Err(Error(format!("{name}: {:?}, expected {shape:?}", p.shape)));
        }
        Ok(p.weight())
    }
    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.tensors
            .get(name)
            .map(|p| p.shape.as_slice())
            .ok_or_else(|| Error(format!("missing {name}")))
    }
}
