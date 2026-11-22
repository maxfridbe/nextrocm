//! All-family typed GPU event intervals: token0 exact49-output parity against pinned full98 [0,1] capture.
use nextrocm_engine_core::{api, Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_llm_inkling2::{config::MiniConfig, text_graph::TextGraph};
use nextrocm_tok::{inkling2_chat, Tokenizer};
use std::{
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg("--").arg(p).output()?;
    if !o.status.success() {
        return Err("hash command failed".into());
    }
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("hash missing")?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn available() -> Result<u64> {
    let s = fs::read_to_string("/proc/meminfo")?;
    let n = s
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|l| l.split_whitespace().next())
        .ok_or("MemAvailable missing")?
        .parse::<u64>()?;
    Ok(n.checked_mul(1024).ok_or("memory overflow")?)
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Record {
    family: i32,
    ty: i32,
    rows: i32,
    cols: i32,
    aux: i32,
    reserved: i32,
    gpu_ms: f64,
}
const _: () = {
    assert!(std::mem::size_of::<Record>() == 32);
    assert!(std::mem::offset_of!(Record, gpu_ms) == 24);
};
type Begin = unsafe extern "C" fn(*mut c_void, i32) -> i32;
type Finish = unsafe extern "C" fn(*mut c_void, *mut Record, i32, *mut i32) -> i32;
type Cancel = unsafe extern "C" fn(*mut c_void) -> i32;
struct Guard {
    gpu: std::sync::Arc<Gpu>,
    cancel: Cancel,
    active: bool,
    errors: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}
impl Drop for Guard {
    fn drop(&mut self) {
        if self.active {
            let rc = unsafe { (self.cancel)(self.gpu.raw()) };
            if rc != 0 {
                if let Ok(mut e) = self.errors.lock() {
                    e.push(format!("profile cancel rc={rc}"));
                }
            }
        }
    }
}
fn expected_records(header: &Gguf, config: &MiniConfig) -> Result<Vec<(i32, i32, i32, i32)>> {
    use nextrocm_llm_inkling2::plan::{Attention, ExecutionPlan};
    let plan = ExecutionPlan::new(config, 1)?;
    let mut e = Vec::new();
    let matrix = |e: &mut Vec<_>, name: String| -> Result<()> {
        let t = header.tensor(&name).ok_or("missing profile matrix")?;
        e.push((t.ty.code() as i32, t.shape[0] as i32, t.shape[1] as i32, 0));
        Ok(())
    };
    for v in plan.visits() {
        let l = v.physical_layer;
        if v.previous_pass_messages {
            for role in ["attn", "k", "v"] {
                for part in ["down", "gain", "forget"] {
                    matrix(&mut e, format!("blk.{l}.loop_{role}_{part}.weight"))?;
                }
            }
        }
        for role in ["attn_q", "attn_k", "attn_v", "attn_rel"] {
            matrix(&mut e, format!("blk.{l}.{role}.weight"))?;
        }
        if v.attention != Attention::Sliding {
            for role in ["attn_k_conv_proj", "attn_v_conv_proj"] {
                matrix(&mut e, format!("blk.{l}.{role}.weight"))?;
            }
        }
        for role in ["attn_sink_proj", "attn_gate", "attn_output"] {
            matrix(&mut e, format!("blk.{l}.{role}.weight"))?;
        }
        if v.previous_pass_messages {
            for part in ["down", "gain", "forget"] {
                matrix(&mut e, format!("blk.{l}.loop_mlp_{part}.weight"))?;
            }
        }
        matrix(&mut e, format!("blk.{l}.ffn_gate_inp.weight"))?;
        for _ in 0..8 {
            e.push((
                if l < 20 { 40 } else { 13 },
                2304,
                2304,
                if l < 20 { 1 } else { 0 },
            ));
            e.push((
                if l < 20 { 40 } else { 8 },
                2304,
                1152,
                if l < 20 { 1 } else { 0 },
            ));
        }
    }
    matrix(&mut e, "output.weight".into())?;
    if e.len() != 1457 {
        return Err("profile expected call count changed".into());
    }
    Ok(e)
}

fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 10 {
        return Err("usage: wopr_all_profile GGUF CONFIG GPU NEW_OUT token 0 0 BASELINE EXPECTED_REPORT_SHA256; NEXTROCM_LIB required".into());
    }
    let max_new: usize = a[7].parse()?;
    if max_new != 0 || a[5] != "token" || a[6] != "0" {
        return Err("profile requires token0 with no generation".into());
    }
    let model = fs::canonicalize(&a[1])?;
    let cfg = fs::canonicalize(&a[2])?;
    let exe = fs::canonicalize(std::env::current_exe()?)?;
    let runtime = fs::canonicalize(PathBuf::from(
        std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB required")?,
    ))?;
    let pins = [hash(&cfg)?, hash(&exe)?, hash(&runtime)?];
    let meta = fs::metadata(&model)?;
    let identity = (meta.len(), meta.modified()?);
    let config = MiniConfig::from_official_json(&fs::read(&cfg)?)?;
    let header = Gguf::open(&model)?;
    let tokenizer = Tokenizer::from_gguf(&header)?;
    if tokenizer.eos != Some(200006) || tokenizer.stop != [200006] {
        return Err("Inkling exact EOS contract required".into());
    }
    let (input, prompt) = match a[5].as_str() {
        "token" => (vec![a[6].parse::<u32>()?], None),
        "prompt" => {
            let rendered = inkling2_chat(
                &[serde_json::json!({"role":"user","content":a[6]})],
                None,
                &[],
            )?;
            (tokenizer.encode(&rendered), Some(rendered))
        }
        _ => return Err("mode must be token or prompt".into()),
    };
    if input.is_empty() || input.len() > 128 || input.iter().any(|&t| t >= 200058 || t == 200054) {
        return Err("input must contain1..128 text-only tokens".into());
    }
    let context = input.len().checked_add(max_new).ok_or("context overflow")?;
    let out = PathBuf::from(&a[4]);
    fs::create_dir(&out)?;
    fs::write(
        out.join("input.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"ids":input,"rendered_prompt":prompt,"max_new_tokens":max_new,"sampler":"greedy, ties lowerID","mode":a[5]}),
        )?,
    )?;
    nextrocm_engine_core::use_kind("llm");
    let gpu = Gpu::open(a[3].parse()?)?;
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    let ptr = api()?.symbol("nr_allocation_stats");
    if ptr.is_null() {
        return Err("allocation stats missing".into());
    }
    // SAFETY: exact private ns stats ABI, permanent runtime and live GPU.
    let stats: Stats = unsafe { std::mem::transmute(ptr) };
    let stat = || -> Result<(u64, u64)> {
        let (mut live, mut peak) = (0, 0);
        if unsafe { stats(gpu.raw(), &mut live, &mut peak) } != 0 {
            return Err("stats failed".into());
        }
        Ok((live, peak))
    };
    if stat()?.0 != 0 {
        return Err("diagnostic requires empty device".into());
    }
    let baseline = fs::canonicalize(&a[8])?;
    if a[9] != "8657467641a8652bbce0a9f8b11738a8336ace6d00a89f066cd33b00443d0785"
        || hash(&baseline.join("report.json"))? != a[9]
    {
        return Err("baseline report pin mismatch".into());
    }
    let bm: serde_json::Value = serde_json::from_slice(&fs::read(baseline.join("report.json"))?)?;
    if bm["status"] != "DIAGNOSTIC_COMPLETE"
        || bm["context"] != 2
        || bm["input_ids"] != serde_json::json!([0, 1])
        || bm["generated_ids"] != serde_json::json!([])
        || bm["config_sha256"] != serde_json::json!(pins[0])
        || !matches!(
            bm["runtime_sha256"].as_str(),
            Some("25a45d71d718cd92c2e2fec67dc4ffe08ab7f393bb3c8338e9e6becc0a877598")
        )
        || bm["files"].as_object().ok_or("baseline files")?.len() != 98
    {
        return Err("baseline inventory mismatch".into());
    }
    let mut baseline_inventory = std::collections::BTreeMap::new();
    let baseline_plan = nextrocm_llm_inkling2::plan::ExecutionPlan::new(&config, 2)?;
    for token in 0..2 {
        for visit in baseline_plan.visits() {
            baseline_inventory.insert(
                format!(
                    "token-{token:03}/visit-{:02}-physical-{:02}.f32",
                    visit.ordinal, visit.physical_layer
                ),
                (2304usize, Some(visit.physical_layer)),
            );
        }
        baseline_inventory.insert(format!("token-{token:03}/logits.f32"), (200058usize, None));
    }
    if baseline_inventory.len() != 98
        || baseline_inventory
            .keys()
            .any(|name| bm["files"].get(name).is_none())
    {
        return Err("baseline full98 scheduled inventory mismatch".into());
    }
    let mut baseline_pins = std::collections::BTreeMap::new();
    for (name, v) in bm["files"].as_object().unwrap() {
        let path = baseline.join(name);
        let (count, physical) = baseline_inventory
            .get(name)
            .ok_or("unexpected baseline path")?;
        if v["elements"].as_u64() != Some(*count as u64)
            || v["shape"] != serde_json::json!([count])
            || physical.is_some_and(|layer| v["physical_layer"].as_u64() != Some(layer as u64))
        {
            return Err("baseline shape/physical layer mismatch".into());
        }
        let raw = fs::read(&path)?;
        if raw.len() != count * 4
            || raw
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
        {
            return Err("baseline bytes/nonfinite mismatch".into());
        }
        let expected = v["sha256"].as_str().ok_or("baseline SHA")?;
        if hash(&path)? != expected {
            return Err("baseline artifact pin mismatch".into());
        }
        baseline_pins.insert(path, expected.to_string());
    }
    let expected = expected_records(&header, &config)?;
    let begin_ptr = api()?.symbol("ni2_all_profile_begin");
    let finish_ptr = api()?.symbol("ni2_all_profile_finish");
    let cancel_ptr = api()?.symbol("ni2_all_profile_cancel");
    if begin_ptr.is_null() || finish_ptr.is_null() || cancel_ptr.is_null() {
        return Err("profile ABI missing".into());
    }
    let begin_profile: Begin = unsafe { std::mem::transmute(begin_ptr) };
    let finish_profile: Finish = unsafe { std::mem::transmute(finish_ptr) };
    let cancel_profile: Cancel = unsafe { std::mem::transmute(cancel_ptr) };
    let profile_errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let mut profile_json = serde_json::Value::Null;
    let mut files = serde_json::Map::new();
    let mut timings = Vec::new();
    let mut generated = Vec::new();
    let mut load_seconds = None;
    let start = Instant::now();
    let result = (|| -> Result<()> {
        let begin = Instant::now();
        let mut graph = TextGraph::load(&model, &config, &gpu, context, available()?, || Ok(()))?;
        load_seconds = Some(begin.elapsed().as_secs_f64());
        println!(
            "loaded {:.6}s, input_tokens={}",
            load_seconds.unwrap(),
            input.len()
        );
        let prepare_start = Instant::now();
        if unsafe { begin_profile(gpu.raw(), 4096) } != 0 {
            return Err("profile begin failed".into());
        }
        // Declared after graph: cancellation destroys events BEFORE graph/context on errors.
        let mut guard = Guard {
            gpu: gpu.clone(),
            cancel: cancel_profile,
            active: true,
            errors: profile_errors.clone(),
        };
        let profile_prepare_seconds = prepare_start.elapsed().as_secs_f64();
        let mut evaluate = |id: u32, position: usize, graph: &mut TextGraph| -> Result<Vec<f32>> {
            let dir = out.join(format!("token-{position:03}"));
            fs::create_dir(&dir)?;
            let begin = Instant::now();
            let mut visits = 0usize;
            let logits=graph.token(id,||Ok(()),|visit,b|{
    if visit.ordinal!=visits||b.len!=2304*4{return Err("visit order/output shape changed".into())}
    let mut raw=vec![0u8;b.len];b.read(0,&mut raw).map_err(|e|e.0)?;
    if raw.chunks_exact(4).any(|x|!f32::from_le_bytes(x.try_into().unwrap()).is_finite()){return Err(format!("nonfinite visit{}",visit.ordinal))}
    let name=format!("visit-{:02}-physical-{:02}.f32",visit.ordinal,visit.physical_layer);let path=dir.join(&name);
    fs::write(&path,raw).map_err(|e|e.to_string())?;files.insert(format!("token-{position:03}/{name}"),serde_json::json!({"sha256":hash(&path).map_err(|e|e.to_string())?,"elements":2304,"shape":[2304],"physical_layer":visit.physical_layer,"core_pass":visit.core_pass}));visits+=1;Ok(())
   })?;
            if visits != 48 || logits.len() != 200058 || logits.iter().any(|x| !x.is_finite()) {
                return Err("full graph capture shape/nonfinite".into());
            }
            let path = dir.join("logits.f32");
            fs::write(
                &path,
                logits
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            files.insert(
                format!("token-{position:03}/logits.f32"),
                serde_json::json!({"sha256":hash(&path)?,"elements":200058,"shape":[200058]}),
            );
            let seconds = begin.elapsed().as_secs_f64();
            timings.push(seconds);
            println!("token {position} id={id} visits=48 seconds={seconds:.6}");
            Ok(logits)
        };
        let mut logits = Vec::new();
        for (i, &id) in input.iter().enumerate() {
            logits = evaluate(id, i, &mut graph)?;
        }
        for i in 0..max_new {
            let mut best = 0usize;
            for j in 1..logits.len() {
                if logits[j] > logits[best] {
                    best = j;
                }
            }
            generated.push(best as u32);
            println!("greedy {i}: {best}");
            if tokenizer.stop.contains(&(best as u32)) || i + 1 == max_new {
                break;
            }
            logits = evaluate(best as u32, input.len() + i, &mut graph)?;
        }
        let mut records = vec![Record::default(); 4096];
        let mut count = 0i32;
        if unsafe { finish_profile(gpu.raw(), records.as_mut_ptr(), 4096, &mut count) } != 0 {
            return Err("profile finish failed".into());
        }
        guard.active = false;
        const COUNTS: [usize; 17] = [
            1457, 1, 434, 144, 384, 48, 1, 192, 48, 48, 48, 144, 48, 96, 16, 88, 384,
        ];
        if count != 3581 || COUNTS.iter().sum::<usize>() != count as usize {
            return Err("all-family profile count mismatch".into());
        }
        records.truncate(count as usize);
        let mut observed = [0usize; 17];
        let mut matrix_index = 0usize;
        let mut groups =
            std::collections::BTreeMap::<(i32, i32, i32, i32, i32), (usize, f64)>::new();
        let mut ordered = Vec::new();
        let mut total_ms = 0.0;
        for r in &records {
            if !(1..=17).contains(&r.family)
                || r.reserved != 0
                || r.rows <= 0
                || r.cols <= 0
                || !r.gpu_ms.is_finite()
                || r.gpu_ms < 0.0
            {
                return Err("invalid family/shape/reserved/time".into());
            }
            observed[r.family as usize - 1] += 1;
            if r.family == 1 {
                let key = (r.ty, r.rows, r.cols, r.aux);
                if expected.get(matrix_index) != Some(&key) {
                    return Err("ordered matrix subsequence changed".into());
                }
                matrix_index += 1;
            } else if r.family == 2 {
                if (r.ty, r.rows, r.cols, r.aux) != (8, 1, 2304, 0) {
                    return Err("embedding decode metadata changed".into());
                }
            } else {
                if r.ty != 0 {
                    return Err("nonquant family type must be zero".into());
                }
                match r.family {
                    8 if !(0..=1).contains(&r.aux) => {
                        return Err("shortconv dynamic metadata".into())
                    }
                    12 if !(1..=5).contains(&r.rows)
                        || r.cols != 2304
                        || !(1..=4).contains(&r.aux) =>
                    {
                        return Err("depthmix metadata".into())
                    }
                    8 | 12 => (),
                    _ if r.aux != 0 => return Err("unused aux must be zero".into()),
                    _ => (),
                }
                if [5, 6, 7, 14, 15, 16, 17].contains(&r.family) && r.rows != 1 {
                    return Err("pointwise row metadata".into());
                }
                if [9, 10, 11].contains(&r.family) && (r.rows != 24 || r.cols != 1) {
                    return Err("token0 attention metadata".into());
                }
                if r.family == 13 && (r.rows != 1 || r.cols != 128) {
                    return Err("router metadata".into());
                }
            }
            let key = (r.family, r.ty, r.rows, r.cols, r.aux);
            let entry = groups.entry(key).or_default();
            entry.0 += 1;
            entry.1 += r.gpu_ms;
            total_ms += r.gpu_ms;
            ordered.push(serde_json::json!({"family":r.family,"type":r.ty,"rows":r.rows,"cols":r.cols,"aux":r.aux,"reserved":r.reserved,"gpu_ms":r.gpu_ms}));
        }
        if observed != COUNTS || matrix_index != expected.len() || !total_ms.is_finite() {
            return Err("family counts/matrix coverage/time sum mismatch".into());
        }
        if files.len() != 49 {
            return Err("profile output inventory mismatch".into());
        }
        for name in files.keys() {
            if fs::read(out.join(name))? != fs::read(baseline.join(name))? {
                return Err(format!("profile numerical bytes differ:{name}").into());
            }
        }
        for (path, pin) in &baseline_pins {
            if hash(path)? != *pin {
                return Err("baseline changed".into());
            }
        }
        if hash(&baseline.join("report.json"))? != a[9] {
            return Err("baseline report changed".into());
        }
        profile_json = serde_json::json!({"status":"PROFILE_AND49_BYTE_PARITY_PASS","record_count":count,"expected_count":3581,"expected_family_counts":COUNTS,"observed_family_counts":observed,"matvec_subsequence_count":matrix_index,"summed_gpu_ms":total_ms,"event_preparation_seconds":profile_prepare_seconds,"records":ordered,"groups":groups.iter().map(|(k,v)|serde_json::json!({"family":k.0,"type":k.1,"rows":k.2,"cols":k.3,"aux":k.4,"count":v.0,"gpu_ms":v.1})).collect::<Vec<_>>(),"scope":"events bracket all 17 typed operator families, with attention split into logits/probabilities/weighted intervals; exclude file reads/uploads/host sync; not occupancy or end-to-end throughput; token0 wall includes49readbacks/hashes; full98 baseline artifacts pinned before/after, only49 token0 counterparts compared","baseline_report_sha256":a[9],"baseline_input_ids":[0,1],"baseline_artifacts_pinned":98});
        fs::write(
            out.join("all-profile.json"),
            serde_json::to_vec_pretty(&profile_json)?,
        )?;
        drop(graph);
        gpu.sync()?;
        Ok(())
    })();
    // Preserve cleanup/identity errors in the report, rather than losing the
    // numerical capture on a post-execution GPU synchronization failure.
    let mut post_errors = profile_errors
        .lock()
        .map_err(|_| "profile cleanup errors mutex")?
        .clone();
    if let Err(e) = gpu.sync() {
        post_errors.push(e.to_string());
    }
    let allocation = match stat() {
        Ok(v) => Some(v),
        Err(e) => {
            post_errors.push(e.to_string());
            None
        }
    };
    let unchanged = match (|| -> Result<bool> {
        let after = fs::metadata(&model)?;
        Ok((after.len(), after.modified()?) == identity
            && [hash(&cfg)?, hash(&exe)?, hash(&runtime)?] == pins)
    })() {
        Ok(v) => v,
        Err(e) => {
            post_errors.push(e.to_string());
            false
        }
    };
    let complete = result.is_ok()
        && post_errors.is_empty()
        && allocation.is_some_and(|(live, peak)| live == 0 && peak < 4_026_531_840)
        && unchanged;
    let report = serde_json::json!({"status":if complete{"DIAGNOSTIC_COMPLETE"}else{"INCOMPLETE"},"numeric_validation_pass":false,"all_profile":profile_json,"model_admission":false,"scope":"WOPR typed all-family profile diagnostic; token0 exact49 byte parity against real full98 chronological[0,1] baseline, no derived report; event profiling is not throughput","error":result.as_ref().err().map(ToString::to_string),"post_errors":post_errors,"model":model,"model_bytes":identity.0,"config":cfg,"config_sha256":pins[0],"executable":exe,"executable_sha256":pins[1],"runtime":runtime,"runtime_sha256":pins[2],"input_ids":input,"generated_ids":generated,"generated_text":tokenizer.decode(&generated),"context":context,"files":files,"load_seconds":load_seconds,"token_seconds":timings,"seconds":start.elapsed().as_secs_f64(),"allocation_after_drop":allocation,"identities_unchanged":unchanged,"timing_scope":"includes48 synchronized per-visit readbacks and artifact hashing; not throughput benchmark"});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    if !complete {
        return Err("full text diagnostic incomplete; see report".into());
    }
    println!("DIAGNOSTIC_COMPLETE numerical validation pending");
    Ok(())
}
