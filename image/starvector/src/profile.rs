//! Private serialized diagnostic spans, compiled out of normal builds.
//! No GPU allocations; <=8192 samples and 512 groups. Not throughput timing.
use nextrocm_engine_core::{api, Error, Gpu, Result};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

type Stamp = unsafe extern "C" fn(*mut c_void, *mut i64) -> i32;
type Elapsed = unsafe extern "C" fn(*mut c_void, i64, i64, *mut f64) -> i32;
struct Api {
    stamp: Stamp,
    elapsed: Elapsed,
}
fn events() -> Result<&'static Api> {
    static EVENTS: OnceLock<std::result::Result<Api, String>> = OnceLock::new();
    EVENTS
        .get_or_init(|| {
            let lib = api().map_err(|e| e.0)?;
            let stamp = lib.symbol("ns_stamp");
            let elapsed = lib.symbol("ns_elapsed");
            if stamp.is_null() || elapsed.is_null() {
                return Err("missing profiling event ABI".into());
            }
            // SAFETY: exact ns.h signatures; library is retained by the process.
            Ok(unsafe {
                Api {
                    stamp: std::mem::transmute::<*mut c_void, Stamp>(stamp),
                    elapsed: std::mem::transmute::<*mut c_void, Elapsed>(elapsed),
                }
            })
        })
        .as_ref()
        .map_err(|e| Error(e.clone()))
}
#[derive(Default)]
struct State {
    active: bool,
    phase: String,
    count: usize,
    dropped: usize,
    rows: BTreeMap<String, Vec<(f64, f64)>>,
}
thread_local! { static STATE: RefCell<State> = RefCell::new(State::default()); }
pub fn begin() -> Result<()> {
    if std::env::var("NS_PROFILE").as_deref() != Ok("gpu") {
        return Err(Error(
            "profiler requires NS_PROFILE=gpu before GPU open".into(),
        ));
    }
    STATE.with(|s| {
        *s.borrow_mut() = State {
            active: true,
            phase: "calibration".into(),
            ..State::default()
        }
    });
    Ok(())
}
pub fn phase(name: &str) {
    STATE.with(|s| s.borrow_mut().phase = name.into());
}
fn reserve(label: &str) -> Option<String> {
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if !s.active {
            return None;
        }
        let key = format!("{}/{}", s.phase, label);
        if s.count >= 8192 || (!s.rows.contains_key(&key) && s.rows.len() >= 512) {
            s.dropped += 1;
            return None;
        }
        s.count += 1;
        s.rows.entry(key.clone()).or_default();
        Some(key)
    })
}
fn record(key: String, gpu_ns: f64, host_ns: f64) {
    STATE.with(|s| {
        s.borrow_mut()
            .rows
            .entry(key)
            .or_default()
            .push((gpu_ns, host_ns))
    });
}
pub fn measure<T>(gpu: &Arc<Gpu>, label: String, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let Some(key) = reserve(&label) else {
        return f();
    };
    let api = events()?;
    let mut start = 0;
    let mut end = 0;
    let mut ns = 0.0;
    let wall = Instant::now();
    // SAFETY: live GPU and valid local output pointers; elapsed consumes each
    // pair immediately so the runtime's 4096-event ring cannot wrap this span.
    if unsafe { (api.stamp)(gpu.raw(), &mut start) } != 0 {
        return Err(Error("profile start stamp failed".into()));
    }
    let result = f();
    if unsafe { (api.stamp)(gpu.raw(), &mut end) } != 0
        || unsafe { (api.elapsed)(gpu.raw(), start, end, &mut ns) } != 0
    {
        return Err(Error("profile end/elapsed failed".into()));
    }
    if !ns.is_finite() || ns < 0.0 {
        return Err(Error("invalid GPU span".into()));
    }
    record(key, ns, wall.elapsed().as_secs_f64() * 1e9);
    result
}
pub struct HostTimer {
    key: String,
    start: Instant,
}
impl HostTimer {
    pub fn start(label: &str) -> Option<Self> {
        reserve(&format!("host:{label}")).map(|key| Self {
            key,
            start: Instant::now(),
        })
    }
}
impl Drop for HostTimer {
    fn drop(&mut self) {
        record(
            self.key.clone(),
            0.0,
            self.start.elapsed().as_secs_f64() * 1e9,
        );
    }
}
pub fn finish(path: &Path) -> Result<()> {
    let mut state = STATE.with(|s| std::mem::take(&mut *s.borrow_mut()));
    state.active = false;
    let rows:Vec<_>=state.rows.into_iter().map(|(name,values)|{
        let mut device:Vec<_>=values.iter().map(|v|v.0).collect();device.sort_by(f64::total_cmp);
        let mut host:Vec<_>=values.iter().map(|v|v.1).collect();host.sort_by(f64::total_cmp);
        let pct=|v:&[f64],p:usize| if v.is_empty(){0.0}else{v[((v.len()-1)*p)/100]};
        serde_json::json!({"category":name,"count":values.len(),"gpu_total_ns":device.iter().sum::<f64>(),"gpu_p50_ns":pct(&device,50),"gpu_p95_ns":pct(&device,95),"host_total_ns":host.iter().sum::<f64>(),"host_p50_ns":pct(&host,50),"host_p95_ns":pct(&host,95)})
    }).collect();
    let report = serde_json::json!({"scope":"Serialized per-operation event spans plus host spans; synchronization/instrumentation perturbs scheduling; not end-to-end throughput","samples":state.count,"dropped":state.dropped,"groups":rows});
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| Error(e.to_string()))?;
    serde_json::to_writer_pretty(file, &report).map_err(|e| Error(e.to_string()))?;
    if state.dropped > 0 {
        return Err(Error(
            "profile sample/group cap reached; report retained".into(),
        ));
    }
    Ok(())
}
