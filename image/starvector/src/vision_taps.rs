//! Default-off, bounded diagnostic readback. No additional device allocations.
use nextrocm_engine_core::DevBuf;
use nextrocm_gguf::safetensors::f16_to_f32;
use nextrocm_image::{Error, Result};
use std::io::Write;
use std::path::PathBuf;

pub(crate) struct Taps {
    dir: PathBuf,
    bytes: usize,
    entries: Vec<serde_json::Value>,
}

impl Taps {
    pub(crate) fn open() -> Result<Option<Self>> {
        let Some(dir) = std::env::var_os("NR_STARVECTOR_VISION_TAPS") else {
            return Ok(None);
        };
        let dir = PathBuf::from(dir);
        std::fs::create_dir(&dir).map_err(|e| Error(format!("new tap directory: {e}")))?;
        Ok(Some(Self {
            dir,
            bytes: 0,
            entries: Vec::new(),
        }))
    }

    pub(crate) fn device(
        &mut self,
        name: &str,
        buf: &DevBuf,
        offset: usize,
        rows: usize,
        cols: usize,
        half: bool,
    ) -> Result<()> {
        let count = rows
            .checked_mul(cols)
            .filter(|&n| n <= 2_000_000)
            .ok_or_else(|| Error("tap tensor exceeds bounded size".into()))?;
        let mut raw = vec![0u8; count * if half { 2 } else { 4 }];
        buf.read(offset, &mut raw)?;
        let values: Vec<f32> = if half {
            raw.chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect()
        } else {
            raw.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        self.host(name, &values, rows, cols)
    }

    pub(crate) fn host(
        &mut self,
        name: &str,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows.checked_mul(cols) != Some(values.len()) || values.iter().any(|x| !x.is_finite()) {
            return Err(Error("invalid/nonfinite tap".into()));
        }
        let bytes = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error("tap size overflow".into()))?;
        if bytes > 64 * 1024 * 1024 - self.bytes {
            return Err(Error("tap capture exceeds 64 MiB".into()));
        }
        let filename = format!("{name}.f32");
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.dir.join(&filename))
            .map_err(|e| Error(e.to_string()))?;
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        file.write_all(&raw).map_err(|e| Error(e.to_string()))?;
        self.bytes += bytes;
        self.entries.push(
            serde_json::json!({"name":name,"file":filename,"shape":[rows,cols],"bytes":bytes}),
        );
        let manifest = serde_json::json!({"format":"little-endian f32","bytes":self.bytes,"tensors":self.entries});
        std::fs::write(self.dir.join("manifest.json"), manifest.to_string())
            .map_err(|e| Error(e.to_string()))
    }
}
