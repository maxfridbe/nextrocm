//! CPU-only checkpoint and residency validation, before device allocation.
use nextrocm_engine_core::{Error, Result};
use nextrocm_gguf::safetensors::SafeTensors;
use std::path::Path;

#[derive(Debug)]
pub struct VaePlan {
    pub resident_bytes: u64,
    pub largest_upload_bytes: u64,
    pub encoder_bytes: u64,
    pub decoder_bytes: u64,
}
pub fn validate_dimensions(h: usize, w: usize) -> Result<()> {
    if h < 16 || w < 16 || h > 4096 || w > 4096 || h % 16 != 0 || w % 16 != 0 {
        return Err(Error(
            "VAE dimensions must be multiples16 in16..4096".into(),
        ));
    }
    Ok(())
}
pub fn default_stats() -> Result<(Vec<f32>, Vec<f32>)> {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("config.json")).map_err(|e| Error(e.to_string()))?;
    let get = |key: &str| -> Result<Vec<f32>> {
        v[key]
            .as_array()
            .ok_or_else(|| Error("VAE stats missing".into()))?
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|x| x as f32)
                    .ok_or_else(|| Error("VAE stats type".into()))
            })
            .collect()
    };
    let a = get("latents_mean")?;
    let b = get("latents_std")?;
    if a.len() != 64
        || b.len() != 64
        || a.iter().any(|v| !v.is_finite())
        || b.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(Error("invalid default VAE statistics".into()));
    }
    Ok((a, b))
}
impl VaePlan {
    pub fn read(path: &Path, allowance: usize) -> Result<Self> {
        let file = SafeTensors::open(path).map_err(|e| Error(e.0))?;
        let table: serde_json::Value =
            serde_json::from_str(include_str!("tensors.json")).map_err(|e| Error(e.to_string()))?;
        let table = table
            .as_object()
            .ok_or_else(|| Error("invalid embedded VAE tensor manifest".into()))?;
        if file.tensors.len() != table.len() {
            return Err(Error("VAE tensor inventory mismatch".into()));
        }
        let length = std::fs::metadata(path)
            .map_err(|e| Error(e.to_string()))?
            .len();
        let mut out = Self {
            resident_bytes: 0,
            largest_upload_bytes: 0,
            encoder_bytes: 0,
            decoder_bytes: 0,
        };
        let mut previous_end = 0;
        for t in &file.tensors {
            let expected = table
                .get(&t.name)
                .ok_or_else(|| Error(format!("unexpected VAE tensor {}", t.name)))?;
            let shape: Vec<u64> = serde_json::from_value(expected["shape"].clone())
                .map_err(|e| Error(e.to_string()))?;
            if t.dtype != "BF16" || t.shape != shape {
                return Err(Error(format!("VAE shape/dtype mismatch {}", t.name)));
            }
            let n = shape
                .iter()
                .try_fold(1u64, |a, &b| a.checked_mul(b))
                .ok_or_else(|| Error("VAE element overflow".into()))?;
            let end = t
                .offset
                .checked_add(t.bytes)
                .ok_or_else(|| Error("VAE range overflow".into()))?;
            if n.checked_mul(2) != Some(t.bytes) || t.offset < previous_end || end > length {
                return Err(Error("VAE tensor ranges invalid".into()));
            }
            previous_end = end;
            if t.name.contains(".time_conv.") {
                continue;
            }
            let bytes = n * if t.name.ends_with(".weight") { 2 } else { 4 };
            out.resident_bytes += bytes;
            out.largest_upload_bytes = out.largest_upload_bytes.max(n * 4);
            if t.name.starts_with("encoder.") || t.name.starts_with("conv1.") {
                out.encoder_bytes += bytes;
            } else {
                out.decoder_bytes += bytes;
            }
        }
        if out.resident_bytes != 662_158_672
            || out.resident_bytes + out.largest_upload_bytes > allowance as u64
            || allowance >= 8_000_000_000
        {
            return Err(Error(
                "VAE weights/transient exceed strict stage allowance".into(),
            ));
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_stats_and_geometry() {
        let (a, b) = default_stats().unwrap();
        assert_eq!((a.len(), b.len()), (64, 64));
        for (h, w) in [(16, 16), (256, 256), (1024, 768), (4096, 4096)] {
            assert!(validate_dimensions(h, w).is_ok());
        }
        for (h, w) in [(0, 16), (15, 16), (17, 16), (4097, 16), (usize::MAX, 16)] {
            assert!(validate_dimensions(h, w).is_err());
        }
    }
    #[test]
    fn immutable_inventory_residency() {
        let t: serde_json::Value = serde_json::from_str(include_str!("tensors.json")).unwrap();
        let t = t.as_object().unwrap();
        assert_eq!(t.len(), 238);
        let bytes: u64 = t
            .iter()
            .filter(|(n, _)| !n.contains(".time_conv."))
            .map(|(n, v)| {
                let s: Vec<u64> = serde_json::from_value(v["shape"].clone()).unwrap();
                s.iter().product::<u64>() * if n.ends_with(".weight") { 2 } else { 4 }
            })
            .sum();
        assert_eq!(bytes, 662_158_672);
    }
    #[test]
    fn checkpoint_ranges_and_allowance_before_allocation() {
        use std::io::Write;
        let table: serde_json::Value = serde_json::from_str(include_str!("tensors.json")).unwrap();
        let mut header = serde_json::Map::new();
        let mut offset = 0u64;
        for (name, value) in table.as_object().unwrap() {
            let shape: Vec<u64> = serde_json::from_value(value["shape"].clone()).unwrap();
            let bytes = shape.iter().product::<u64>() * 2;
            header.insert(name.clone(),serde_json::json!({"dtype":"BF16","shape":shape,"data_offsets":[offset,offset+bytes]}));
            offset += bytes;
        }
        let bytes = serde_json::to_vec(&header).unwrap();
        let path = std::env::temp_dir().join(format!(
            "qi-vae-plan-{}-{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        f.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        f.set_len(8 + bytes.len() as u64 + offset).unwrap();
        let plan = VaePlan::read(&path, 2_000_000_000).unwrap();
        assert_eq!(plan.resident_bytes, 662_158_672);
        assert_eq!(plan.largest_upload_bytes, 47_775_744);
        assert!(VaePlan::read(&path, 662_158_672).is_err());
        assert!(VaePlan::read(&path, 8_000_000_000).is_err());
        f.set_len(8 + bytes.len() as u64 + offset - 1).unwrap();
        assert!(VaePlan::read(&path, 2_000_000_000).is_err());
        drop(f);
        std::fs::remove_file(path).unwrap();
    }
}
