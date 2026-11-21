//! Bounded file reads: no whole-model host mirror or pinned 16.6 GB allocation.
//! The caller must finish consuming each borrowed chunk before the next read.
use crate::{plan::Visit, weights::Weights, Result};
use nextrocm_gguf::{Gguf, Tensor};
use std::path::Path;

pub const STAGING_BYTES: usize = 16 * 1024 * 1024;
pub struct HostStore {
    file: Gguf,
    staging: Vec<u8>,
    identity: std::sync::Arc<()>,
}
#[derive(Clone, Debug)]
pub struct ExpertSpan {
    pub tensor: String,
    pub offset: u64,
    pub bytes: u64,
    pub output_scale: f32,
}
impl HostStore {
    pub fn open(path: &Path, available_host_bytes: u64) -> Result<Self> {
        // Reclaimable file cache is not a second resident model. Leave a host
        // reserve here; the eventual device/session plan must budget UMA too.
        if available_host_bytes < 4 * 1024 * 1024 * 1024 + STAGING_BYTES as u64 {
            return Err("insufficient host reserve for staging".into());
        }
        let file = Gguf::open(path).map_err(|e| e.0)?;
        Weights::read(&file)?;
        let mut staging = Vec::new();
        staging
            .try_reserve_exact(STAGING_BYTES)
            .map_err(|e| e.to_string())?;
        staging.resize(STAGING_BYTES, 0);
        Ok(Self {
            file,
            staging,
            identity: std::sync::Arc::new(()),
        })
    }
    pub(crate) fn identity(&self) -> std::sync::Arc<()> {
        self.identity.clone()
    }
    pub fn file(&self) -> &Gguf {
        &self.file
    }
    /// Uses the physical layer from the validated visit; scales travel with
    /// that layer on every pass, never with an uninitialized virtual layer.
    pub fn expert_spans(&self, visit: &Visit, expert: usize) -> Result<[ExpertSpan; 2]> {
        if expert >= 128 || visit.physical_layer >= 24 {
            return Err("expert or physical layer out of bounds".into());
        }
        let layer = visit.physical_layer;
        let span = |part: &str| -> Result<ExpertSpan> {
            let name = format!("blk.{layer}.ffn_{part}_exps.weight");
            let t = self.file.tensor(&name).ok_or("missing expert tensor")?;
            let bytes = t.bytes / 128;
            let offset = bytes
                .checked_mul(expert as u64)
                .ok_or("expert offset overflow")?;
            let scale_name = format!("blk.{layer}.ffn_{part}_exps.scale");
            let output_scale = if layer < 20 {
                let scale = self
                    .file
                    .tensor(&scale_name)
                    .ok_or("missing NVFP4 expert scale")?;
                let mut b = [0u8; 4];
                self.file
                    .read_into(scale, expert as u64 * 4, &mut b)
                    .map_err(|e| e.0)?;
                f32::from_le_bytes(b)
            } else {
                1.0
            };
            if !output_scale.is_finite() || output_scale <= 0.0 {
                return Err("nonfinite or nonpositive expert scale".into());
            }
            Ok(ExpertSpan {
                tensor: name,
                offset,
                bytes,
                output_scale,
            })
        };
        Ok([span("gate_up")?, span("down")?])
    }
    /// Read one checked, at-most-16MiB span into caller-owned staging.
    /// No intermediate copy or asynchronous borrow; caller checks cancellation
    /// immediately before each bounded read and retains the destination.
    pub(crate) fn read_span_into(&self, name: &str, offset: u64, dst: &mut [u8]) -> Result<()> {
        if dst.len() > STAGING_BYTES {
            return Err("direct read exceeds bounded staging span".into());
        }
        let t = self.file.tensor(name).ok_or("missing tensor")?;
        validate_span(t, offset, dst.len() as u64)?;
        self.file.read_into(t, offset, dst).map_err(|e| e.0)
    }
    pub fn stream(
        &mut self,
        name: &str,
        offset: u64,
        bytes: u64,
        mut consume: impl FnMut(u64, &[u8]) -> Result<()>,
    ) -> Result<()> {
        let t = self.file.tensor(name).ok_or("missing tensor")?.clone();
        validate_span(&t, offset, bytes)?;
        let mut done = 0u64;
        while done < bytes {
            let n = (bytes - done).min(STAGING_BYTES as u64) as usize;
            self.file
                .read_into(&t, offset + done, &mut self.staging[..n])
                .map_err(|e| e.0)?;
            consume(done, &self.staging[..n])?;
            done += n as u64;
        }
        Ok(())
    }
}
fn validate_span(t: &Tensor, offset: u64, bytes: u64) -> Result<()> {
    if bytes == 0 || offset.checked_add(bytes).is_none_or(|end| end > t.bytes) {
        return Err("tensor range overflow/out of bounds".into());
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_reject_wrap_and_empty() {
        let t = Tensor {
            name: "x".into(),
            ty: nextrocm_gguf::GType::F32,
            shape: vec![16],
            shard: 0,
            offset: 0,
            bytes: 64,
        };
        assert!(validate_span(&t, 0, 64).is_ok());
        assert!(validate_span(&t, 64, 1).is_err());
        assert!(validate_span(&t, u64::MAX, 2).is_err());
        assert!(validate_span(&t, 0, 0).is_err());
    }
}
