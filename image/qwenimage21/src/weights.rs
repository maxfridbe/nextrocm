//! Packed DiT matrix ownership. Adapted from nextsycl's QwenImage21 loader
//! (98124afb83f10c9e0cfbabd1f2495d2417df0a78), preserving its Q/K row permutation
//! while keeping each Q4_K/Q6_K part in its stored format.
use std::sync::Arc;

use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_gguf::{Gguf, Tensor};

const HEAD: usize = 128;
// Largest current quantized matrix: modulation [16384,4096], 37,748,736 bytes.
// Bound host transient storage independently of the complete packed model.
const MAX_MATRIX_BYTES: usize = 64 * 1024 * 1024;

pub struct PackedPart {
    pub(crate) weights: DevBuf,
    pub(crate) quant_type: u32,
    pub rows: usize,
    pub columns: usize,
}

/// A fused output may contain differently quantized parts (Q4_K Q/K, Q6_K V).
/// Each part writes into a column range with the same output row stride.
pub struct PackedMatrix {
    pub(crate) parts: Vec<PackedPart>,
    pub rows: usize,
    pub columns: usize,
    pub bytes: usize,
}

fn geometry(t: &Tensor) -> Result<(usize, usize, usize)> {
    let [rows, columns] = t.shape.as_slice() else {
        return Err(Error(format!("{}: expected a matrix", t.name)));
    };
    let rows = usize::try_from(*rows).map_err(|_| Error("matrix row overflow".into()))?;
    let columns = usize::try_from(*columns).map_err(|_| Error("matrix column overflow".into()))?;
    let block_bytes = match t.ty.code() {
        12 => 144,
        14 => 210,
        _ => {
            return Err(Error(format!(
                "{}: unsupported packed type {}",
                t.name,
                t.ty.name()
            )))
        }
    };
    if rows == 0 || columns == 0 || columns % 256 != 0 {
        return Err(Error(format!("{}: invalid packed matrix geometry", t.name)));
    }
    let row_bytes = (columns / 256)
        .checked_mul(block_bytes)
        .ok_or_else(|| Error("packed row size overflow".into()))?;
    let bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Error("packed matrix size overflow".into()))?;
    if bytes > MAX_MATRIX_BYTES || u64::try_from(bytes).ok() != Some(t.bytes) {
        return Err(Error(format!(
            "{}: packed byte size differs or exceeds per-matrix bound",
            t.name
        )));
    }
    Ok((rows, columns, row_bytes))
}

/// New split-half row reads old interleaved row, within each 128-channel head.
fn rope_row(row: usize) -> usize {
    let within = row % HEAD;
    row / HEAD * HEAD + (within % (HEAD / 2)) * 2 + within / (HEAD / 2)
}

impl PackedMatrix {
    /// Validate every part and the caller's remaining weight allowance before
    /// allocating any device buffer. The allowance excludes scratch/activations;
    /// the engine's stage planner must reserve those independently.
    pub fn load(
        file: &Gguf,
        gpu: &Arc<Gpu>,
        names: &[(&str, bool)],
        allowance: usize,
    ) -> Result<Self> {
        if names.is_empty() {
            return Err(Error("empty packed matrix".into()));
        }
        let mut plan = Vec::with_capacity(names.len());
        let (mut rows, mut columns, mut bytes) = (0usize, 0usize, 0usize);
        for &(name, permute) in names {
            let t = file
                .tensor(name)
                .ok_or_else(|| Error(format!("missing {name}")))?;
            let (n, k, row_bytes) = geometry(t)?;
            if columns != 0 && columns != k {
                return Err(Error("packed parts have different input dimensions".into()));
            }
            if permute && n % HEAD != 0 {
                return Err(Error(format!("{name}: partial RoPE head")));
            }
            columns = k;
            rows = rows
                .checked_add(n)
                .ok_or_else(|| Error("fused row overflow".into()))?;
            bytes = bytes
                .checked_add(n * row_bytes)
                .ok_or_else(|| Error("fused byte overflow".into()))?;
            plan.push((t, n, k, row_bytes, permute));
        }
        if bytes > allowance {
            return Err(Error(format!(
                "packed matrix needs {bytes} bytes, allowance {allowance}"
            )));
        }
        let mut parts = Vec::with_capacity(plan.len());
        for (t, n, k, row_bytes, permute) in plan {
            // At most two bounded matrix buffers on host; no full-model copy.
            let raw = file.read(t).map_err(|e| Error(e.0))?;
            let reordered = if permute {
                let mut dst = vec![0u8; raw.len()];
                for row in 0..n {
                    let src = rope_row(row) * row_bytes;
                    dst[row * row_bytes..(row + 1) * row_bytes]
                        .copy_from_slice(&raw[src..src + row_bytes]);
                }
                Some(dst)
            } else {
                None
            };
            let weights = DevBuf::new(gpu, raw.len())?;
            // DevBuf::write is synchronous; temporary host storage stays live
            // until this copy returns. RAII frees earlier parts on any failure.
            weights.write(0, reordered.as_deref().unwrap_or(&raw))?;
            parts.push(PackedPart {
                weights,
                quant_type: t.ty.code(),
                rows: n,
                columns: k,
            });
        }
        Ok(Self {
            parts,
            rows,
            columns,
            bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextrocm_gguf::GType;

    #[test]
    fn rope_permutation_matches_interleaved_pairs_across_heads() {
        let rows: Vec<_> = (0..256).map(rope_row).collect();
        assert_eq!(&rows[..4], &[0, 2, 4, 6]);
        assert_eq!(&rows[64..68], &[1, 3, 5, 7]);
        assert_eq!(&rows[128..132], &[128, 130, 132, 134]);
        let mut sorted = rows;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..256).collect::<Vec<_>>());
    }

    #[test]
    fn actual_quantized_geometry_and_bad_descriptors() {
        let mut t = Tensor {
            name: "q".into(),
            ty: GType::from_code(12).unwrap(),
            shape: vec![4096, 4096],
            shard: 0,
            offset: 0,
            bytes: 9437184,
        };
        assert_eq!(geometry(&t).unwrap(), (4096, 4096, 2304));
        t.ty = GType::from_code(14).unwrap();
        t.bytes = 13762560;
        assert_eq!(geometry(&t).unwrap(), (4096, 4096, 3360));
        t.bytes -= 1;
        assert!(geometry(&t).is_err());
        t.shape = vec![u64::MAX, 4096];
        assert!(geometry(&t).is_err());
        t.ty = GType::from_code(0).unwrap();
        assert!(geometry(&t).is_err());
    }
}
