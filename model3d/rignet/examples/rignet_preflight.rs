//! CPU-only preflight. Usage: rignet_preflight ARCHIVE.zip [MESH] or --mesh-only MESH.
use nextrocm_model3d_rignet::{preflight, prepare_mesh_cpu};
use std::{path::Path, time::Instant};
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.len() > 3 {
        return Err(
            "usage: rignet_preflight ARCHIVE.zip [MESH [DUMP_DIR]] | --mesh-only MESH [DUMP_DIR]"
                .into(),
        );
    }
    let start = Instant::now();
    let mut report = if args[0] == "--mesh-only" {
        serde_json::json!({"weights_validated":false})
    } else {
        preflight::inspect(Path::new(&args[0])).map_err(|e| e.0)?
    };
    if let Some(path) = args.get(1) {
        let data = std::fs::read(path).map_err(|e| e.to_string())?;
        let mesh = nextrocm_model3d::read_mesh(&data)?;
        let t = Instant::now();
        let p = prepare_mesh_cpu(&mesh, 4000, 0).map_err(|e| e.0)?;
        if let Some(dir) = args.get(2) {
            nextrocm_model3d_rignet::check::write_inputs(&p, Path::new(dir)).map_err(|e| e.0)?;
        }
        report["geometry"] = serde_json::json!({"vertices":p.pos.len(),"faces":p.faces.len(),
            "topology_edges":p.tpl.len(),"geodesic_edges":p.geo.len(),"prepare_s":t.elapsed().as_secs_f64()});
    } else if args[0] == "--mesh-only" {
        return Err("--mesh-only requires a mesh".into());
    }
    report["total_s"] = start.elapsed().as_secs_f64().into();
    report["gpu_opened"] = false.into();
    report["device_fit_validated"] = false.into();
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
