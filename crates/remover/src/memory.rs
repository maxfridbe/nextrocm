//! Allocation admission for the fixed original Swin-L graph, before every device buffer.
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result, api};
use nextrocm_gguf::safetensors::SafeTensors;
use std::{collections::BTreeMap, sync::Arc};
const LIMIT: u64 = 8_000_000_000;
const RESERVE: u64 = 256 << 20;

pub(crate) fn product(dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d)
            .filter(|_| d > 0)
            .ok_or_else(|| Error("remover: zero or overflowing dimensions".into()))
    })
}

pub(crate) fn stats(gpu: &Gpu) -> Result<(u64, u64)> {
    let api = api()?;
    let p = api.symbol("nr_allocation_stats");
    if p.is_null() {
        return Err(Error("remover requires allocation telemetry".into()));
    }
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: retained runtime symbol, matching ABI; handle and output pointers remain live.
    let query: Stats = unsafe { std::mem::transmute(p) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: valid GPU handle and writable outputs for the runtime stats ABI.
    if unsafe { query(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err(Error(api.error()));
    }
    Ok((current, peak))
}

fn fits(current: u64, bytes: u64) -> bool {
    current
        .checked_add(bytes)
        .and_then(|n| n.checked_add(RESERVE))
        .is_some_and(|n| n < LIMIT)
}
fn admit(gpu: &Gpu, bytes: usize) -> Result<()> {
    let current = stats(gpu)?.0;
    let bytes = u64::try_from(bytes).map_err(|_| Error("remover allocation overflow".into()))?;
    if !fits(current, bytes) {
        return Err(Error(
            "remover device allocation exceeds 8 GB with reserve".into(),
        ));
    }
    let (total, free) = gpu.memory()?;
    if bytes
        .checked_add(RESERVE)
        .is_none_or(|n| n > free.unwrap_or(total))
    {
        return Err(Error(
            "remover: insufficient free device memory with reserve".into(),
        ));
    }
    Ok(())
}
pub(crate) fn floats(gpu: &Arc<Gpu>, n: usize) -> Result<DevBuf> {
    admit(gpu, product(&[n, 4])?)?;
    DevBuf::f32(gpu, n)
}
pub(crate) fn from_f32(gpu: &Arc<Gpu>, values: &[f32]) -> Result<DevBuf> {
    if values.iter().any(|x| !x.is_finite()) {
        return Err(Error(
            "remover: nonfinite device input or folded weights".into(),
        ));
    }
    admit(gpu, product(&[values.len(), 4])?)?;
    DevBuf::from_f32(gpu, values)
}

fn validate(st: &SafeTensors) -> Result<usize> {
    // Original revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 shapes.
    // Pin architecture, not numeric values: F16/BF16/F32 versions are accepted.
    let expected: BTreeMap<String, Vec<u64>> = serde_json::from_str(include_str!("geometry.json"))
        .map_err(|e| Error(format!("remover geometry: {e}")))?;
    let size = std::fs::metadata(&st.path)
        .map_err(|e| Error(e.to_string()))?
        .len();
    let mut bytes = 0usize;
    for (name, shape) in expected {
        let t = st.need(&name).map_err(|e| Error(e.0))?;
        if t.shape != shape {
            return Err(Error(format!(
                "{name}: unsupported BiRefNet geometry {:?}, expected {shape:?}",
                t.shape
            )));
        }
        let dims = t
            .shape
            .iter()
            .map(|&d| usize::try_from(d).map_err(|_| Error("tensor dimension overflow".into())))
            .collect::<Result<Vec<_>>>()?;
        let n = product(&dims)?;
        let width = match t.dtype.as_str() {
            "F16" | "BF16" => 2,
            "F32" => 4,
            _ => return Err(Error(format!("{name}: unsupported {}", t.dtype))),
        };
        if t.bytes != product(&[n, width])? as u64
            || t.offset.checked_add(t.bytes).is_none_or(|end| end > size)
        {
            return Err(Error(format!("{name}: invalid tensor extent")));
        }
        bytes = bytes
            .checked_add(product(&[n, 4])?)
            .ok_or_else(|| Error("weight size overflow".into()))?;
    }
    // Expanded relative biases [heads,144,144] in all 24 Swin blocks. Their
    // original tables are already counted above, deliberately overestimating.
    bytes
        .checked_add((2 * 6 + 2 * 12 + 18 * 24 + 2 * 48) * 144 * 144 * 4)
        .ok_or_else(|| Error("relative bias size overflow".into()))
}
pub(crate) fn checkpoint(st: &SafeTensors, gpu: &Gpu) -> Result<()> {
    admit(gpu, validate(st)?)
}

pub(crate) fn picture(pic: &nextrocm_diffusion::Picture) -> Result<()> {
    let n = product(&[
        pic.width as usize,
        pic.height as usize,
        pic.channels as usize,
    ])?;
    // Host preparation and returned masks stay bounded too (16 megapixels).
    if !matches!(pic.channels, 1 | 3 | 4)
        || n != pic.data.len()
        || u64::from(pic.width) * u64::from(pic.height) > 16 * 1024 * 1024
    {
        return Err(Error(
            "remover: invalid picture geometry or over 16 megapixels".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_allocation_limit_and_overflow() {
        assert!(fits(1_000_000_000, 256 << 20));
        assert!(!fits(LIMIT - RESERVE - 1, 1));
        assert!(!fits(u64::MAX, 1));
        assert!(product(&[usize::MAX, 4]).is_err());
        assert!(product(&[0, 4]).is_err());
    }
    #[test]
    fn rejects_malformed_pictures() {
        let mut p = nextrocm_diffusion::Picture::new(2, 3, 3);
        assert!(picture(&p).is_ok());
        p.data.pop();
        assert!(picture(&p).is_err());
    }
}
