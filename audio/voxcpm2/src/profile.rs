//! Opt-in synchronized operation wall profiling, never pure GPU timing.
use nextrocm_audio::Result;
use nextrocm_engine_core::Gpu;
use serde_json::json;
use std::{
    cell::Cell, collections::BTreeMap, marker::PhantomData, rc::Rc, sync::Mutex, time::Instant,
};

thread_local! { static STAGE: Cell<&'static str> = const { Cell::new("load_or_unscoped") }; }
pub struct Stage {
    previous: &'static str,
    _local: PhantomData<Rc<()>>,
}
impl Stage {
    pub fn enter(name: &'static str) -> Self {
        Self {
            previous: STAGE.with(|s| s.replace(name)),
            _local: PhantomData,
        }
    }
}
impl Drop for Stage {
    fn drop(&mut self) {
        STAGE.with(|s| s.set(self.previous));
    }
}
#[derive(Default)]
struct Row {
    count: u64,
    failed: u64,
    seconds: f64,
    pre_sync_seconds: f64,
    max_seconds: f64,
}
type Key = (&'static str, &'static str, Vec<usize>);
#[derive(Default)]
pub struct Profile {
    rows: Mutex<BTreeMap<Key, Row>>,
}
impl Profile {
    pub fn enabled() -> Option<Self> {
        (std::env::var("NEXTROCM_VCP_PROFILE").as_deref() == Ok("1")).then(Self::default)
    }
    pub fn measure<T>(
        &self,
        gpu: &Gpu,
        op: &'static str,
        shape: &[usize],
        run: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let pre = Instant::now();
        gpu.sync()?;
        let pre_sync_seconds = pre.elapsed().as_secs_f64();
        let start = Instant::now();
        let result = run();
        let synced = gpu.sync();
        let seconds = start.elapsed().as_secs_f64();
        // Recover telemetry after poisoning without changing numerical work.
        let mut rows = self.rows.lock().unwrap_or_else(|e| e.into_inner());
        let row = rows
            .entry((STAGE.with(Cell::get), op, shape.to_vec()))
            .or_default();
        row.count += 1;
        row.failed += u64::from(result.is_err() || synced.is_err());
        row.seconds += seconds;
        row.pre_sync_seconds += pre_sync_seconds;
        row.max_seconds = row.max_seconds.max(seconds);
        drop(rows);
        synced?;
        result
    }
}
impl Drop for Profile {
    fn drop(&mut self) {
        let rows = self.rows.get_mut().unwrap_or_else(|e| e.into_inner());
        let records:Vec<_>=rows.iter().map(|((stage,op,shape),r)|json!({"stage":stage,"operation":op,"shape":shape,"count":r.count,"failed":r.failed,"launch_plus_sync_seconds":r.seconds,"pre_sync_seconds":r.pre_sync_seconds,"max_seconds":r.max_seconds})).collect();
        // Diagnostic output failure must not panic during engine teardown.
        use std::io::Write;
        let _ = writeln!(
            std::io::stderr().lock(),
            "{}",
            json!({"schema":"voxcpm2-op-wall-v1","timing":"launch plus synchronization wall; pre-sync wait separate; not pure GPU or unprofiled throughput","operations":records})
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_stages_restore_on_unwind() {
        assert_eq!(STAGE.with(Cell::get), "load_or_unscoped");
        let outer = Stage::enter("prefill");
        let result = std::panic::catch_unwind(|| {
            let _inner = Stage::enter("encoder");
            panic!("test");
        });
        assert!(result.is_err());
        assert_eq!(STAGE.with(Cell::get), "prefill");
        drop(outer);
        assert_eq!(STAGE.with(Cell::get), "load_or_unscoped");
    }
}
