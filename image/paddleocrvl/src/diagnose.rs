//! Diagnostic controls only: never a substitute for strict full-model admission.
use crate::{
    positions,
    strict::{metrics, sha256, Fixture},
    PaddleOcrVl,
};
use nextrocm_engine_core::{DevBuf, Error, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};
fn write(path: &Path, values: &[f32]) -> Result<String> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, &bytes).map_err(|e| Error(e.to_string()))?;
    Ok(sha256(&bytes))
}
pub fn run(m: &PaddleOcrVl, f: &Fixture, out: &Path, log: &mut dyn FnMut(String)) -> Result<Value> {
    f.validate_precision(m.decoder.is_f32())?;
    if out.exists() {
        return Err(Error("diagnostic output already exists".into()));
    }
    std::fs::create_dir_all(out).map_err(|e| Error(e.to_string()))?;
    let grid = f.nums("grid.txt")?;
    if grid.len() != 3 {
        return Err(Error("diagnostic grid".into()));
    }
    let (gh, gw) = (grid[1] as usize, grid[2] as usize);
    let ids = f.nums("tokens.txt")?;
    let pos = positions(&ids, gh / 2, gw / 2);
    let last = m.decoder.g.layers - 1;
    let mut results = BTreeMap::new();
    for case in ["native", "cpu_projected", "cpu_layer17"] {
        let dir = out.join(case);
        std::fs::create_dir(&dir).map_err(|e| Error(e.to_string()))?;
        let (x, start) = if case == "cpu_layer17" {
            (
                DevBuf::from_f32(&m.nsd.gpu, &f.floats(&format!("l_out-{}", last - 1))?)?,
                last,
            )
        } else {
            let img = if case == "native" {
                m.vision
                    .see(&m.nsd, &f.floats("pixel_values")?, gh, gw, &mut |_, _| {
                        Ok(())
                    })?
            } else {
                DevBuf::from_f32(&m.nsd.gpu, &f.floats("projected")?)?
            };
            let x = m.embed(&ids, &img)?;
            m.nsd.wait()?;
            (x, 0)
        };
        let mut files = BTreeMap::new();
        let mut stage_metrics = BTreeMap::new();
        let mut cache = m.decoder.cache(ids.len() + 1)?;
        let logits=m.decoder.diagnostic_pass(&m.nsd,&mut cache,&x,&pos,start,last,&mut |name,buf|{
            let a=buf.to_f32()?;let hash=write(&dir.join(format!("{name}.f32")),&a)?;
            files.insert(name.to_string(),json!({"values":a.len(),"sha256":hash}));
            if name.starts_with("l_out-"){
                let metric=metrics(&a,&f.floats(name)?)?;
                log(format!("DIAGNOSTIC {case} {name} {metric:?}"));
                stage_metrics.insert(name.to_string(),json!({"max_relative":metric.max_relative,"rms_relative":metric.rms_relative,"cosine":metric.cosine}));
            }Ok(())
        })?;
        let metric = metrics(&logits, &f.floats("logits_last")?)?;
        log(format!("DIAGNOSTIC {case} logits_last {metric:?}"));
        files.insert(
            "logits_last".to_string(),
            json!({"values":logits.len(),"sha256":write(&dir.join("logits_last.f32"),&logits)?}),
        );
        stage_metrics.insert("logits_last".to_string(),json!({"max_relative":metric.max_relative,"rms_relative":metric.rms_relative,"cosine":metric.cosine}));
        results.insert(
            case,
            json!({"first_layer":start,"traced_layer":last,"files":files,"metrics":stage_metrics}),
        );
    }
    let report = json!({"diagnostic_only":true,"admission_pass":false,"cases":results});
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).map_err(|e| Error(e.to_string()))?,
    )
    .map_err(|e| Error(e.to_string()))?;
    Ok(report)
}
