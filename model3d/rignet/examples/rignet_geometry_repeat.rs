//! CPU-only repeated mesh preprocessing: isolate randomness before seeded geometry.
use nextrocm_model3d::mesh::{tools, Mesh};
use serde_json::json;
use sha2::{Digest, Sha256};

fn digest(mesh: &Mesh) -> String {
    let mut hash = Sha256::new();
    hash.update((mesh.positions.len() as u64).to_le_bytes());
    hash.update((mesh.indices.len() as u64).to_le_bytes());
    for value in mesh.positions.iter().flatten() {
        hash.update(value.to_le_bytes());
    }
    for index in &mesh.indices {
        hash.update(index.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let full = std::env::args().any(|arg| arg == "--full");
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: rignet_geometry_repeat MESH")?;
    let bytes = std::fs::read(path)?;
    let mut rows = Vec::new();
    for repeat in 0..8 {
        let mesh = nextrocm_model3d::read_mesh(&bytes)?;
        let (mut welded, _) = tools::weld(&mesh, 1e-7);
        tools::compact(&mut welded);
        let mut decimated = tools::decimate(&welded, 4000);
        tools::compact(&mut decimated);
        let mut row = json!({"repeat":repeat,"input":digest(&mesh),"weld":digest(&welded),
            "decimated":digest(&decimated),"vertices":decimated.positions.len(),"triangles":decimated.triangles()});
        if full {
            let p = nextrocm_model3d_rignet::prepare_mesh_cpu(&mesh, 4000, 0).map_err(|e| e.0)?;
            let floats = |values: &[f32]| {
                let mut hash = Sha256::new();
                for value in values {
                    hash.update(value.to_le_bytes());
                }
                format!("{:x}", hash.finalize())
            };
            let json_hash = |value: serde_json::Value| -> Result<String, serde_json::Error> {
                Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
            };
            row["prepared"] = json!({
                "pos":floats(&p.pos.iter().flatten().copied().collect::<Vec<_>>()),
                "faces":json_hash(json!(p.faces))?, "tpl":json_hash(json!(p.tpl))?,
                "geo":json_hash(json!(p.geo))?, "geodesic_dist":floats(&p.geodesic.dist),
                "geodesic_nn":json_hash(json!(p.geodesic.nn))?, "samples":p.geodesic.s,
                "vox":format!("{:x}",Sha256::digest(p.vox.data.iter().map(|v|u8::from(*v)).collect::<Vec<_>>())),
                "vox_meta":[p.vox.dims as f32,p.vox.scale,p.vox.translate[0],p.vox.translate[1],p.vox.translate[2]],
                "pivot":p.pivot,"scale":p.scale,"seed":0
            });
        }
        rows.push(row);
    }
    let same = |key: &str| rows.iter().all(|row| row[key] == rows[0][key]);
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
        "gpu_opened":false,"seed_scope":"decimation happens before request-seeded sampling",
        "input_identical":same("input"),"weld_identical":same("weld"),
        "decimation_identical":same("decimated"),"full_prepared_identical":if full {Some(same("prepared"))} else {None},"runs":rows}))?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_cost_planar_decimation_repeats() {
        let mut mesh = Mesh::default();
        for y in 0..5 {
            for x in 0..5 {
                mesh.positions.push([x as f32, y as f32, 0.0]);
            }
        }
        for y in 0..4u32 {
            for x in 0..4u32 {
                let a = y * 5 + x;
                mesh.indices.extend([a, a + 1, a + 5, a + 1, a + 6, a + 5]);
            }
        }
        let expected = digest(&tools::decimate(&mesh, 12));
        for _ in 0..16 {
            assert_eq!(
                digest(&tools::decimate(&mesh, 12)),
                expected,
                "equal-cost decimation depends on randomized hash iteration"
            );
        }
    }
}
