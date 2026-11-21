//! CPU-only archive identity and checkpoint geometry inspection. Never opens a GPU.
use crate::{limits, zip};
use nextrocm_gguf::pth::Pth;
use nextrocm_model3d::{Error, Result};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path, time::Instant};
pub const ARCHIVE_BYTES: u64 = 195677735;
pub const ARCHIVE_SHA256: &str = "97fe5ae10c55b417acb3fe715547a7fa984e4d583a24bc158f30462a1b855212";
pub const NETWORKS: &[&str] = &["gcn_meanshift", "rootnet", "bonenet", "skinnet"];
pub fn inspect(path: &Path) -> Result<serde_json::Value> {
    let start = Instant::now();
    let mut f = File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
    if f.metadata().map_err(|e| Error(e.to_string()))?.len() != ARCHIVE_BYTES {
        return Err(Error(
            "RigNet archive size differs from pinned manifest".into(),
        ));
    }
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| Error(e.to_string()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    let hash = format!("{:x}", h.finalize());
    if hash != ARCHIVE_SHA256 {
        return Err(Error(
            "RigNet archive SHA-256 differs from pinned manifest".into(),
        ));
    }
    let hash_s = start.elapsed().as_secs_f64();
    let scratch = zip::Scratch::new()?;
    let mut networks = Vec::new();
    let mut total = 0usize;
    for net in NETWORKS {
        let t = Instant::now();
        let member = format!("checkpoints/{net}/model_best.pth.tar");
        let p = scratch.extract(path, &member, &format!("{net}.pth.tar"))?;
        let ck = Pth::open(&p).map_err(|e| Error(e.0))?;
        let probe = if *net == "gcn_meanshift" {
            "jointnet.gcu_1.mlp.0.0.weight"
        } else if *net == "skinnet" {
            "gcu1.mlp.0.0.weight"
        } else {
            "shape_encoder.gcu_1.mlp.0.0.weight"
        };
        // The per-network native loader remains authoritative for complete topology.
        if ck.prefix_of(probe).is_none() {
            return Err(Error(format!("{net}: missing network probe {probe}")));
        }
        let mut tensors = Vec::new();
        for (name, tensor) in &ck.tensors {
            // Optimizer tensors are serialized in the archive too; only the state dict is resident.
            if !name.starts_with("state_dict.") {
                continue;
            }
            let n = limits::product(&tensor.shape, limits::MAX_WEIGHT_BYTES / 4, name)?;
            total = total
                .checked_add(
                    n.checked_mul(4)
                        .ok_or_else(|| Error("weight size overflow".into()))?,
                )
                .filter(|n| *n <= limits::MAX_WEIGHT_BYTES)
                .ok_or_else(|| Error("RigNet expanded weights exceed bound".into()))?;
            if tensor.shape.len() > 2
                || tensor
                    .shape
                    .iter()
                    .any(|&d| d == 0 || d > limits::MAX_CHANNELS)
            {
                return Err(Error(format!(
                    "{name}: unsupported tensor geometry {:?}",
                    tensor.shape
                )));
            }
            let data = ck.f32(name).map_err(|e| Error(e.0))?;
            if data.len() != n || data.iter().any(|v| !v.is_finite()) {
                return Err(Error(format!("{name}: invalid tensor values")));
            }
            if name.ends_with("running_var") && data.iter().any(|v| *v < 0.) {
                return Err(Error(format!("{name}: negative variance")));
            }
            tensors.push(serde_json::json!({"name":name,"shape":tensor.shape,"elements":n}));
        }
        if tensors.is_empty() {
            return Err(Error(format!("{net}: missing state_dict")));
        }
        networks.push(serde_json::json!({"network":net,"member":member,"tensors":tensors,"inspect_s":t.elapsed().as_secs_f64()}));
    }
    Ok(
        serde_json::json!({"archive":path,"archive_bytes":ARCHIVE_BYTES,"sha256":hash,"hash_s":hash_s,
        "cpu_preflight_s":start.elapsed().as_secs_f64(),"expanded_state_dict_bytes":total,"networks":networks,
        "device_peak_bytes":null,"device_fit_validated":false}),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn archive_identity_rejects_wrong_size_without_device() {
        let scratch = crate::zip::Scratch::new().unwrap();
        let path = scratch.0.join("not-weights.zip");
        std::fs::write(&path, b"not a checkpoint").unwrap();
        assert!(super::inspect(&path)
            .unwrap_err()
            .0
            .contains("size differs"));
    }
}
