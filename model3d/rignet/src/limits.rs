//! Private admission bounds, independent of a GPU. These are conservative port limits,
//! not a measured VRAM claim. The runtime's aggregate allocation limit still applies.
use nextrocm_model3d::{Error, Result};
pub const MAX_VERTICES: usize = 8192;
pub const MAX_FACES: usize = 8000;
pub const MAX_EDGES: usize = 131072;
pub const MAX_JOINTS: usize = 128;
pub const MAX_CHANNELS: usize = 4096;
pub const MAX_BUFFER_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_WEIGHT_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_INPUT_VERTICES: usize = 100000;
pub const MAX_INPUT_FACES: usize = 200000;
pub fn product(dims: &[usize], cap: usize, what: &str) -> Result<usize> {
    let n = dims
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| Error(format!("{what}: size overflow")))?;
    if n > cap {
        return Err(Error(format!("{what}: {n} exceeds {cap}")));
    }
    Ok(n)
}
pub fn buffer(n: usize) -> Result<usize> {
    product(&[n.max(1), 4], MAX_BUFFER_BYTES, "RigNet buffer")
}
pub fn mesh(pos: &[[f32; 3]], faces: &[[u32; 3]], input: bool) -> Result<()> {
    let (nv, nf) = if input {
        (MAX_INPUT_VERTICES, MAX_INPUT_FACES)
    } else {
        (MAX_VERTICES, MAX_FACES)
    };
    if pos.len() < 4
        || pos.len() > nv
        || faces.len() < 2
        || faces.len() > nf
        || pos.iter().flatten().any(|v| !v.is_finite())
        || faces.iter().flatten().any(|v| *v as usize >= pos.len())
    {
        return Err(Error("RigNet: invalid or oversized mesh".into()));
    }
    Ok(())
}
pub fn graph(n: usize, edges: &[(u32, u32)]) -> Result<()> {
    if n == 0
        || n > MAX_VERTICES
        || edges.len().checked_add(n).is_none_or(|e| e > MAX_EDGES)
        || edges
            .iter()
            .any(|&(s, d)| s as usize >= n || d as usize >= n)
    {
        return Err(Error("RigNet: invalid or oversized graph".into()));
    }
    Ok(())
}
pub fn joints(j: &[[f64; 3]]) -> Result<()> {
    if j.is_empty() || j.len() > MAX_JOINTS || j.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error("RigNet: invalid or oversized skeleton".into()));
    }
    Ok(())
}
pub fn settings(s: &crate::Settings) -> Result<()> {
    if !(100..=MAX_FACES).contains(&s.faces)
        || !s.threshold.is_finite()
        || s.threshold < 0.0
        || s.bandwidth.is_some_and(|b| !b.is_finite() || b <= 0.0)
    {
        return Err(Error(
            "RigNet: invalid faces, bandwidth or threshold".into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_overflow_and_excess() {
        assert!(product(&[usize::MAX, 2], usize::MAX, "test").is_err());
        assert!(buffer(MAX_BUFFER_BYTES / 4 + 1).is_err());
        assert!(graph(4, &[(0, 4)]).is_err());
        assert!(graph(MAX_VERTICES + 1, &[]).is_err());
        assert!(joints(&[[f64::NAN, 0., 0.]]).is_err());
    }
}
