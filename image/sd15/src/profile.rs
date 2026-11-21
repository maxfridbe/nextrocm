//! Opt in with NEXTROCM_SD15_PROFILE=1 before constructing a graph.
//! One JSON line per operation/shape is written to stderr at graph drop, including
//! partial graphs on cancellation. Only successful launch + existing sync calls
//! are counted (including kernel argument/weight lookup preparation). This is
//! synchronized host wall time, not pure GPU time; allocation,
//! transfers, taps, tensor destruction and profiling bookkeeping are excluded.
use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

static NEXT_GRAPH: AtomicU64 = AtomicU64::new(1);

#[derive(Default)]
struct Total {
    count: u64,
    wall: Duration,
}

struct Enabled {
    id: u64,
    totals: Mutex<BTreeMap<(&'static str, Vec<usize>), Total>>,
}

pub(super) struct Profile(Option<Enabled>);

impl Profile {
    pub(super) fn new() -> Self {
        Self(
            (std::env::var_os("NEXTROCM_SD15_PROFILE").as_deref()
                == Some(std::ffi::OsStr::new("1")))
            .then(|| Enabled {
                id: NEXT_GRAPH.fetch_add(1, Ordering::Relaxed),
                totals: Mutex::new(BTreeMap::new()),
            }),
        )
    }

    pub(super) fn begin(&self) -> Option<Instant> {
        self.0.as_ref().map(|_| Instant::now())
    }

    pub(super) fn record(&self, start: Option<Instant>, op: &'static str, shape: &[usize]) {
        if let (Some(state), Some(start)) = (&self.0, start) {
            let elapsed = start.elapsed();
            if let Ok(mut totals) = state.totals.lock() {
                let total = totals.entry((op, shape.to_vec())).or_default();
                total.count += 1;
                total.wall += elapsed;
            }
        }
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        let Some(state) = &mut self.0 else { return };
        let Ok(totals) = state.totals.get_mut() else {
            return;
        };
        let mut stderr = io::stderr().lock();
        for ((op, shape), total) in totals {
            // Names/axes are fixed literals and shape contains only integers, so
            // these records are JSON without string escaping or a new dependency.
            let axes = match *op {
                "linear" => "rows,input,output,bias",
                "conv2d" => "batch,ci,h,w,co,kh,kw,stride,pad,oh,ow",
                "attention" => "queries,keys,channels,heads,dim,causal",
                "layer_norm" => "rows,channels",
                "geglu" => "rows,output_channels",
                "concat" => "batch,ca,cb,h,w",
                "layout_to_image" => "batch,channels,output_h,output_w",
                _ => "input_shape",
            };
            // Diagnostics must not panic during cancellation/unwinding if stderr
            // is closed. The graph id distinguishes repeated component sessions.
            let _ = writeln!(stderr,
                "{{\"schema\":\"sd15-op-profile-v1\",\"pid\":{},\"graph_id\":{},\"metric\":\"launch_sync_wall\",\"op\":\"{}\",\"shape_axes\":\"{}\",\"shape\":{:?},\"count\":{},\"wall_seconds\":{:.9}}}",
                std::process::id(), state.id, op, axes, shape, total.count, total.wall.as_secs_f64());
        }
    }
}
