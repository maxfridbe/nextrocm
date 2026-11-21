//! Private opt-in diagnostics. Never enabled by default or used for throughput claims.
//! NEXTROCM_ISNET_PROFILE=1 produces JSONL on stderr per forward, including partial
//! results on an error. begin drains prior work OUTSIDE the timer; record waits
//! for submitted work INSIDE the timer. Thus intervals are launch + synchronized
//! host wall time, not device-event/pure kernel time. Profiling adds barriers.
use nextrocm_engine_core::{Gpu, Result};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    io::{self, Write},
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};
static NEXT_FORWARD: AtomicU64 = AtomicU64::new(1);
type Key = (&'static str, String, Vec<usize>);
#[derive(Default)]
struct Total {
    calls: u64,
    wall: Duration,
}
struct Enabled {
    id: u64,
    started: Instant,
    complete: Cell<bool>,
    totals: RefCell<BTreeMap<Key, Total>>,
}
pub(super) struct Profile(Option<Enabled>);
fn enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}
impl Profile {
    pub(super) fn new() -> Self {
        Self(
            enabled(std::env::var_os("NEXTROCM_ISNET_PROFILE").as_deref()).then(|| Enabled {
                id: NEXT_FORWARD.fetch_add(1, Ordering::Relaxed),
                started: Instant::now(),
                complete: Cell::new(false),
                totals: RefCell::new(BTreeMap::new()),
            }),
        )
    }
    pub(super) fn begin(&self, gpu: &Gpu) -> Result<Option<Instant>> {
        if self.0.is_none() {
            return Ok(None);
        }
        gpu.sync()?; // exclude pending work from the operation being measured
        Ok(Some(Instant::now()))
    }
    pub(super) fn record(
        &self,
        gpu: &Gpu,
        start: Option<Instant>,
        op: &'static str,
        layer: &str,
        shape: &[usize],
    ) -> Result<()> {
        let (Some(state), Some(start)) = (&self.0, start) else {
            return Ok(());
        };
        gpu.sync()?;
        let elapsed = start.elapsed();
        let mut totals = state.totals.borrow_mut();
        let row = totals
            .entry((op, layer.to_owned(), shape.to_vec()))
            .or_default();
        row.calls += 1;
        row.wall += elapsed;
        Ok(())
    }
    pub(super) fn complete(&self) {
        if let Some(state) = &self.0 {
            state.complete.set(true);
        }
    }
}
fn axes(op: &str) -> &'static str {
    match op {
        "conv_total" => "input_h,input_w,ci,co,kernel,stride,pad,dilation,output_h,output_w",
        "im2col" => {
            "input_h,input_w,ci,kernel,stride,pad,dilation,output_h,output_w,pixel_offset,tile_rows"
        }
        "linear" => "rows,k,co",
        "bias_relu" => "pixels,co,relu",
        "pool" => "input_h,input_w,channels,output_h,output_w",
        "up_concat" => "a_h,a_w,a_channels,b_h,b_w,b_channels",
        "resize_output" => "input_h,input_w,channels,output_h,output_w",
        "input_upload" | "residual_add" => "h,w,channels",
        _ => "elements",
    }
}
fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c < ' ' => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
fn row_json(id: u64, complete: bool, key: &Key, total: &Total) -> String {
    let (op, layer, shape) = key;
    format!(
        "{{\"schema\":\"isnet-op-profile-v1\",\"pid\":{},\"forward_id\":{},\"complete\":{},\"metric\":\"launch_sync_wall\",\"op\":{},\"layer\":{},\"shape_axes\":{},\"shape\":{:?},\"inclusive\":{},\"calls\":{},\"wall_seconds\":{:.9}}}",
        std::process::id(),
        id,
        complete,
        quote(op),
        quote(layer),
        quote(axes(op)),
        shape,
        *op == "conv_total",
        total.calls,
        total.wall.as_secs_f64()
    )
}
impl Drop for Profile {
    fn drop(&mut self) {
        let Some(state) = &mut self.0 else {
            return;
        };
        let wall = state.started.elapsed().as_secs_f64();
        let mut stderr = io::stderr().lock();
        for (key, total) in state.totals.get_mut().iter() {
            // Do not panic on a closed stderr while an inference error unwinds.
            let _ = writeln!(
                stderr,
                "{}",
                row_json(state.id, state.complete.get(), key, total)
            );
        }
        let _ = writeln!(
            stderr,
            "{{\"schema\":\"isnet-forward-profile-v1\",\"pid\":{},\"forward_id\":{},\"complete\":{},\"wall_seconds\":{:.9},\"scope\":\"instrumented forward including transfers, allocations, cleanup and profiler overhead; excludes model load and outer picture preprocessing; conv_total overlaps its im2col/linear/bias children, never sum both\"}}",
            std::process::id(),
            state.id,
            state.complete.get(),
            wall
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_literal_one_enables_profiling() {
        assert!(enabled(Some(std::ffi::OsStr::new("1"))));
        for value in [
            None,
            Some(std::ffi::OsStr::new("0")),
            Some(std::ffi::OsStr::new("true")),
        ] {
            assert!(!enabled(value));
        }
    }
    #[test]
    fn profile_units_and_hierarchy_are_explicit_json() {
        let total = Total {
            calls: 2,
            wall: Duration::from_millis(1250),
        };
        let key = ("linear", "stage1/quoted\"\n".to_owned(), vec![2, 9, 64]);
        let row: serde_json::Value =
            serde_json::from_str(&row_json(7, true, &key, &total)).unwrap();
        assert_eq!(row["layer"], key.1);
        assert_eq!(row["wall_seconds"], 1.25);
        assert_eq!(row["inclusive"], false);
        assert_eq!(row["calls"], 2);
        let key = (
            "conv_total",
            "conv_in".into(),
            vec![1024, 1024, 3, 64, 3, 2, 1, 1, 512, 512],
        );
        let row: serde_json::Value =
            serde_json::from_str(&row_json(7, false, &key, &total)).unwrap();
        assert_eq!(row["inclusive"], true);
        assert_eq!(row["complete"], false);
    }
}
