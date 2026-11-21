//! Parent-run GPU controls. No production registration is needed.
use nextrocm_engine_core::Gpu;
use nextrocm_image::{ModelFiles, Picture};
use nextrocm_image_paddleocrvl::{PaddleOcrVl};
use std::{path::Path, time::Instant};
fn tracked(gpu: &std::sync::Arc<Gpu>) -> Result<(u64, u64), String> {
    type Stats = unsafe extern "C" fn(*mut std::ffi::c_void, *mut u64, *mut u64) -> std::ffi::c_int;
    let api = nextrocm_engine_core::api().map_err(|e| e.0)?;
    let symbol = api.symbol("nr_allocation_stats");
    if symbol.is_null() {
        return Err("runtime lacks nr_allocation_stats".into());
    }
    // SAFETY: kernels/ns/ns.hip exports nr_allocation_stats with this exact signature.
    let stats = unsafe { std::mem::transmute::<*mut std::ffi::c_void, Stats>(symbol) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: live GPU handle and writable local u64 output slots.
    if unsafe { stats(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err("allocation stats failed".into());
    }
    Ok((current, peak))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let a: Vec<String> = std::env::args().collect();
    if a.len()!=5 { return Err("usage: paddleocr_profile WEIGHTS IMAGE IMAGE_SHA SETTINGS_JSON".into()); }
    let image=std::fs::read(&a[2])?;
    if nextrocm_image_paddleocrvl::strict::sha256(&image)!=a[3] { return Err("image SHA mismatch".into()); }
    let picture=Picture::decode(&image)?;
    let patches=match (picture.width,picture.height) { (896,1120)=>5120, (336,336)=>576, _=>return Err("profiler allows only declared 5120/576 grids".into()) };
    let settings_bytes=std::fs::read(&a[4])?;
    if !["67c6de119209846aa13a14ad387357bd88f549ef57bb25ee9734d2bc1fc2e7e9", "5b8a4bb0dff81078b0ff7b5ba7649fb35e41a87cdd29145d4f8ccdc1772857e5"].contains(&nextrocm_image_paddleocrvl::strict::sha256(&settings_bytes).as_str()) {return Err("settings SHA mismatch".into());}
    let options=nextrocm_image::LoadOptions {settings:serde_json::from_slice(&settings_bytes)?,..Default::default()};
    let g=Gpu::open(0)?;
    let before=tracked(&g)?;
    let dir=Path::new(&a[1]);
    let pins: serde_json::Value =
        serde_json::from_str(include_str!("../../../reference/paddleocr/weights.json"))?;
    let mut files = ModelFiles::new();
    for role in ["model", "mmproj", "tokenizer"] {
        let pin = &pins["files"][role];
        let name = Path::new(pin["path"].as_str().ok_or("weight path")?)
            .file_name()
            .ok_or("weight name")?;
        let path = dir.join(name);
        // Verify bytes before loading. Streaming hashing bounds host memory.
        use sha2::{Digest, Sha256};
        use std::io::Read;
        let mut file = std::fs::File::open(&path)?;
        if file.metadata()?.len() != pin["size"].as_u64().ok_or("weight size")? {
            return Err(format!("{role}: size mismatch").into());
        }
        let mut hash = Sha256::new();
        let mut b = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut b)?;
            if n == 0 {
                break;
            }
            hash.update(&b[..n]);
        }
        if format!("{:x}", hash.finalize()) != pin["sha256"].as_str().ok_or("weight hash")? {
            return Err(format!("{role}: hash mismatch").into());
        }
        files.insert(role.into(), path);
    }

    let start=Instant::now();
    let m=PaddleOcrVl::load_with_options(&files,&g,&options,&mut |s| println!("{s}"))?;
    g.sync()?;
    println!("{}",serde_json::json!({"diagnostic_only":true,"load_seconds":start.elapsed().as_secs_f64(),"patches":patches,"stop_after":"v_layer-1","admission":false}));
    let mut stages=Vec::new();
    let mut previous=Instant::now();
    let result=m.image_tokens(&picture,1_003_520,&mut |name,_| {
        g.sync()?;
        let elapsed=previous.elapsed().as_secs_f64();previous=Instant::now();
        println!("{}",serde_json::json!({"stage":name,"synchronized_seconds":elapsed,"patches":patches}));
        std::io::stdout().flush().map_err(|e| nextrocm_engine_core::Error(e.to_string()))?;
        stages.push(name.to_owned());
        if name=="v_layer-1" {return Err(nextrocm_engine_core::Error(STOP.into()));}
        Ok(())
    });
    let stopped=matches!(result,Err(ref e) if e.0==STOP);
    drop(result);
    drop(m);g.sync()?;
    let after=tracked(&g)?;
    if !stopped || stages!=["v_embd","v_layer-0","v_layer-1"] {return Err("profiler did not reach exact intentional two-block stop".into());}
    if after.0!=before.0 || before.0!=0 {return Err("tracked allocation leak".into());}
    println!("{}",serde_json::json!({"diagnostic_complete":true,"intentional_stop":true,"stages":stages,"tracked_before":before,"tracked_after_drop":after,"full_model_pass":false,"device_fit_validated":false,"admission":false}));
    Ok(())
}
const STOP:&str="paddleocr-profile:intentional-two-block-stop";
