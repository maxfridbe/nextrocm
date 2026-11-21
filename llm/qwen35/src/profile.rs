//! Opt-in diagnostics. NS_PROFILE=1 retains synchronized wall time; gpu uses
//! outer-layer stream events by default (including host submission/idle gaps).
//! NS_Q35_PROFILE_DETAIL=1 selects per-operation GPU intervals instead. Wall
//! mode retains per-operation timing. Reports remain cumulative: TG is the
//! final report minus report_after_pp; layer bytes are zero (not transfer counts).
//! Qwen dispatch is serialized by its scratch lock. The 4096-slot runtime event
//! ring is shared: bound our queue to 256 pairs and reject stale tickets rather
//! than report a recycled event. No API access, clock read or lock when disabled.
use nextrocm_engine_core::{api, Error, Gpu, Result};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
    time::Instant,
};

const EVENT_RING: i64 = 4096;
const MAX_PAIRS: usize = 256;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Wall,
    Gpu,
}
impl Mode {
    fn parse(value: Option<&str>) -> Self {
        match value {
            Some("1") => Self::Wall,
            Some("gpu") => Self::Gpu,
            _ => Self::Off,
        }
    }
}
trait Clock: Send + Sync {
    fn sync(&self) -> Result<()>;
    fn stamp(&self) -> Result<i64>;
    fn elapsed_ns(&self, start: i64, end: i64) -> Result<f64>;
}
struct GpuClock(Arc<Gpu>);
impl Clock for GpuClock {
    fn sync(&self) -> Result<()> {
        self.0.sync()
    }
    fn stamp(&self) -> Result<i64> {
        let a = api()?;
        let mut ticket = -1;
        // SAFETY: GPU lives through this owning Arc, ticket is a valid output.
        let rc = unsafe { (a.stamp)(self.0.raw(), &mut ticket) };
        if rc != 0 {
            return Err(Error(format!("Qwen35 profile stamp: {}", a.error())));
        }
        if ticket < 0 {
            return Err(Error("Qwen35 profile negative event ticket".into()));
        }
        Ok(ticket)
    }
    fn elapsed_ns(&self, start: i64, end: i64) -> Result<f64> {
        let a = api()?;
        let mut ns = 0.0;
        // SAFETY: tickets are checked against the ring before resolution; GPU is
        // live and ns_elapsed synchronizes the end event. ABI units are ns.
        let rc = unsafe { (a.elapsed)(self.0.raw(), start, end, &mut ns) };
        if rc != 0 {
            return Err(Error(format!("Qwen35 profile elapsed: {}", a.error())));
        }
        Ok(ns)
    }
}
#[derive(Default)]
struct Row {
    calls: u64,
    bytes: u64,
    seconds: f64,
}
struct Pair {
    name: &'static str,
    start: i64,
    end: i64,
    bytes: usize,
}
#[derive(Default)]
struct State {
    rows: BTreeMap<&'static str, Row>,
    pending: VecDeque<Pair>,
    latest: Option<i64>,
    failure: Option<String>,
}
impl State {
    fn observe(&mut self, ticket: i64) -> Result<()> {
        if ticket < 0 || self.latest.is_some_and(|last| ticket <= last) {
            return Err(Error("Qwen35 profile event tickets regressed".into()));
        }
        self.latest = Some(ticket);
        if self
            .pending
            .front()
            .is_some_and(|p| ticket - p.start >= EVENT_RING)
        {
            return Err(Error(
                "Qwen35 profile event ring overwritten; refusing stale timing".into(),
            ));
        }
        Ok(())
    }
    fn add(&mut self, name: &'static str, bytes: usize, seconds: f64) -> Result<()> {
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(Error("Qwen35 profile invalid elapsed duration".into()));
        }
        let row = self.rows.entry(name).or_default();
        row.calls = row
            .calls
            .checked_add(1)
            .ok_or_else(|| Error("profile call overflow".into()))?;
        row.bytes = row
            .bytes
            .checked_add(bytes as u64)
            .ok_or_else(|| Error("profile byte overflow".into()))?;
        row.seconds += seconds;
        Ok(())
    }
    fn resolve(&mut self, clock: &dyn Clock) -> Result<()> {
        while let Some(pair) = self.pending.front() {
            let ns = clock.elapsed_ns(pair.start, pair.end)?;
            if !ns.is_finite() || ns < 0.0 {
                return Err(Error("Qwen35 profile invalid elapsed nanoseconds".into()));
            }
            let (name, bytes) = (pair.name, pair.bytes);
            self.add(name, bytes, ns * 1e-9)?;
            self.pending.pop_front();
        }
        Ok(())
    }
}
pub(crate) enum Started {
    Wall(Instant),
    Gpu(i64),
}
pub(crate) struct Profile {
    mode: Mode,
    detail: bool,
    clock: Option<Box<dyn Clock>>,
    state: Mutex<State>,
}
impl Profile {
    pub fn new(gpu: &Arc<Gpu>) -> Self {
        let mode = Mode::parse(std::env::var("NS_PROFILE").ok().as_deref());
        Self {
            mode,
            detail: std::env::var("NS_Q35_PROFILE_DETAIL").ok().as_deref() == Some("1"),
            clock: (mode != Mode::Off).then(|| Box::new(GpuClock(gpu.clone())) as Box<dyn Clock>),
            state: Mutex::new(State::default()),
        }
    }
    fn clock(&self) -> Result<&dyn Clock> {
        self.clock
            .as_deref()
            .ok_or_else(|| Error("profile clock missing".into()))
    }
    fn with_state<T>(&self, action: impl FnOnce(&mut State, &dyn Clock) -> Result<T>) -> Result<T> {
        let mut s = self
            .state
            .lock()
            .map_err(|_| Error("Qwen35 profile lock poisoned".into()))?;
        if let Some(e) = &s.failure {
            return Err(Error(e.clone()));
        }
        let result = action(&mut s, self.clock()?);
        if let Err(e) = &result {
            s.failure = Some(e.0.clone());
        }
        result
    }
    #[inline]
    pub fn start(&self) -> Result<Option<Started>> {
        if self.mode == Mode::Gpu && !self.detail {
            return Ok(None);
        }
        self.begin()
    }
    /// Whole-layer intervals replace (never nest with) detailed GPU intervals.
    #[inline]
    pub fn start_layer(&self) -> Result<Option<Started>> {
        if self.mode != Mode::Gpu || self.detail {
            return Ok(None);
        }
        self.begin()
    }
    fn begin(&self) -> Result<Option<Started>> {
        match self.mode {
            Mode::Off => Ok(None),
            Mode::Wall => {
                self.clock()?.sync()?;
                Ok(Some(Started::Wall(Instant::now())))
            }
            Mode::Gpu => self.with_state(|s, clock| {
                let mut ticket = clock.stamp()?;
                s.observe(ticket)?;
                // External stream-copy/mark users consume the same ring. Resolve
                // early with half-ring headroom as well as the hard pair bound.
                if s.pending.len() >= MAX_PAIRS
                    || s.pending
                        .front()
                        .is_some_and(|p| ticket - p.start >= EVENT_RING / 2)
                {
                    s.resolve(clock)?;
                    // Do not charge diagnostic resolution's host wait to the
                    // next operation's interval.
                    ticket = clock.stamp()?;
                    s.observe(ticket)?;
                }
                Ok(Some(Started::Gpu(ticket)))
            }),
        }
    }
    #[inline]
    pub fn record(&self, name: &'static str, start: Option<Started>, bytes: usize) -> Result<()> {
        let Some(start) = start else {
            return Ok(());
        };
        match start {
            Started::Wall(start) => self.with_state(|s, clock| {
                clock.sync()?;
                s.add(name, bytes, start.elapsed().as_secs_f64())
            }),
            Started::Gpu(start) => self.with_state(|s, clock| {
                let end = clock.stamp()?;
                s.observe(end)?;
                if start < 0 || start >= end || end - start >= EVENT_RING {
                    return Err(Error("Qwen35 profile invalid/stale event pair".into()));
                }
                s.pending.push_back(Pair {
                    name,
                    start,
                    end,
                    bytes,
                });
                if s.pending.len() >= MAX_PAIRS
                    || s.pending
                        .front()
                        .is_some_and(|p| end - p.start >= EVENT_RING / 2)
                {
                    s.resolve(clock)?;
                }
                Ok(())
            }),
        }
    }
    pub fn lines(&self) -> Vec<String> {
        if self.mode == Mode::Off {
            return Vec::new();
        }
        let lines = self.with_state(|s, clock| {
            if !s.pending.is_empty() {
                // Check for intervening consumers before touching ring slots.
                let ticket = clock.stamp()?; s.observe(ticket)?; s.resolve(clock)?;
            }
            let timing = if self.mode == Mode::Gpu && !self.detail {
                "GPU full-layer stream elapsed incl FFN and host gaps; attention/delta names classify layer type; bytes=0; cumulative, TG=final-report_after_pp; excludes embedding/output head"
            } else if self.mode == Mode::Gpu {
                "GPU detailed stream elapsed; event ns/1e9; includes host submission/idle gaps; disjoint categories, not total inference; cache bytes=requested extent incl hits"
            } else { "synchronized wall time; diagnostic overhead" };
            Ok(s.rows.iter().map(|(name,row)|format!("qwen35 profile {name}: calls={} bytes={} seconds={:.9} ({timing})",row.calls,row.bytes,row.seconds)).collect())
        });
        lines.unwrap_or_else(|e| vec![format!("Qwen35 profile ERROR: {}", e.0)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
    #[derive(Default)]
    struct Fake {
        next: AtomicI64,
        stamps: AtomicUsize,
        syncs: AtomicUsize,
        resolves: AtomicUsize,
        invalid: AtomicUsize,
    }
    impl Clock for Arc<Fake> {
        fn sync(&self) -> Result<()> {
            self.syncs.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
        fn stamp(&self) -> Result<i64> {
            self.stamps.fetch_add(1, Ordering::Relaxed);
            Ok(self.next.fetch_add(1, Ordering::Relaxed))
        }
        fn elapsed_ns(&self, start: i64, end: i64) -> Result<f64> {
            assert!(
                self.next.load(Ordering::Relaxed) - 1 - start < EVENT_RING,
                "resolved recycled event"
            );
            self.resolves.fetch_add(1, Ordering::Relaxed);
            Ok(if self.invalid.load(Ordering::Relaxed) > 0 {
                f64::NAN
            } else {
                (end - start) as f64 * 2_000_000.0
            })
        }
    }
    fn profile(mode: Mode) -> (Profile, Arc<Fake>) {
        let f = Arc::new(Fake::default());
        (
            Profile {
                mode,
                detail: true,
                clock: Some(Box::new(f.clone())),
                state: Mutex::new(State::default()),
            },
            f,
        )
    }
    #[test]
    fn default_gpu_records_layers_without_per_op_events_or_locks() {
        let (mut p, f) = profile(Mode::Gpu);
        p.detail = false;
        {
            let _held = p.state.lock().unwrap();
            for _ in 0..1000 {
                p.record("op", p.start().unwrap(), 4).unwrap();
            }
        }
        assert_eq!(f.stamps.load(Ordering::Relaxed), 0);
        p.record("full_layer_delta", p.start_layer().unwrap(), 0)
            .unwrap();
        let pp = p.lines();
        assert!(pp[0].contains("calls=1 bytes=0 seconds=0.002000000"));
        p.record("full_layer_delta", p.start_layer().unwrap(), 0)
            .unwrap();
        let final_lines = p.lines();
        assert!(final_lines[0].contains("calls=2 bytes=0 seconds=0.004000000"));
        assert_eq!(f.resolves.load(Ordering::Relaxed), 2);
    }
    #[test]
    fn layer_and_detailed_modes_do_not_nest() {
        for mode in [Mode::Off, Mode::Wall, Mode::Gpu] {
            let (p, f) = profile(mode);
            let _held = p.state.lock().unwrap();
            assert!(p.start_layer().unwrap().is_none());
            assert_eq!(f.stamps.load(Ordering::Relaxed), 0);
            assert_eq!(f.syncs.load(Ordering::Relaxed), 0);
        }
        let (mut p, f) = profile(Mode::Wall);
        p.detail = false;
        p.record("wall", p.start().unwrap(), 0).unwrap();
        assert_eq!(f.syncs.load(Ordering::Relaxed), 2);
    }
    #[test]
    fn layer_queue_stays_bounded_across_ring_wraps() {
        let (mut p, f) = profile(Mode::Gpu);
        p.detail = false;
        for _ in 0..10_003 {
            p.record("full_layer_attention", p.start_layer().unwrap(), 0)
                .unwrap();
            assert!(p.state.lock().unwrap().pending.len() < MAX_PAIRS);
        }
        assert!(p.lines()[0].contains("calls=10003"));
        assert_eq!(f.resolves.load(Ordering::Relaxed), 10_003);
    }
    #[test]
    fn disabled_never_touches_clock_or_mutex() {
        let (p, f) = profile(Mode::Off);
        let _held = p.state.lock().unwrap();
        for _ in 0..100 {
            p.record("off", p.start().unwrap(), 7).unwrap();
        }
        assert!(p.lines().is_empty());
        assert_eq!(f.stamps.load(Ordering::Relaxed), 0);
        assert_eq!(f.syncs.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn gpu_pairs_bounded_across_many_ring_wraps() {
        let (p, f) = profile(Mode::Gpu);
        for _ in 0..10_003 {
            p.record("kernel", p.start().unwrap(), 4).unwrap();
            assert!(p.state.lock().unwrap().pending.len() < MAX_PAIRS);
        }
        assert_eq!(f.syncs.load(Ordering::Relaxed), 0);
        let lines = p.lines();
        assert!(lines[0].contains("calls=10003 bytes=40012 seconds=20.006000000"));
        assert_eq!(f.resolves.load(Ordering::Relaxed), 10_003);
        assert!(p.state.lock().unwrap().pending.is_empty());
        assert_eq!(p.lines(), lines);
    }
    #[test]
    fn reporting_flushes_partial_and_accumulates_categories() {
        let (p, f) = profile(Mode::Gpu);
        p.record("cache", p.start().unwrap(), 8).unwrap();
        let before = p.lines();
        assert!(before[0].contains("seconds=0.002000000"));
        p.record("kernel", p.start().unwrap(), 16).unwrap();
        assert_eq!(p.lines().len(), 2);
        assert_eq!(f.resolves.load(Ordering::Relaxed), 2);
    }
    #[test]
    fn external_events_trigger_early_flush() {
        let (p, f) = profile(Mode::Gpu);
        p.record("kernel", p.start().unwrap(), 0).unwrap();
        f.next.fetch_add(EVENT_RING / 2, Ordering::Relaxed);
        let t = p.start().unwrap();
        assert_eq!(f.resolves.load(Ordering::Relaxed), 1);
        p.record("next", t, 0).unwrap();
        assert_eq!(p.lines().len(), 2);
    }
    #[test]
    fn stale_ring_fails_without_reading_reused_events() {
        let (p, f) = profile(Mode::Gpu);
        p.record("kernel", p.start().unwrap(), 0).unwrap();
        f.next.fetch_add(EVENT_RING, Ordering::Relaxed);
        assert!(p.lines()[0].contains("ERROR"));
        assert_eq!(f.resolves.load(Ordering::Relaxed), 0);
        assert!(p.start().is_err());
    }
    #[test]
    fn stale_single_interval_and_nonfinite_elapsed_fail() {
        let (p, f) = profile(Mode::Gpu);
        let t = p.start().unwrap();
        f.next.fetch_add(EVENT_RING, Ordering::Relaxed);
        assert!(p.record("bad", t, 0).is_err());
        let (p, f) = profile(Mode::Gpu);
        p.record("bad", p.start().unwrap(), 0).unwrap();
        f.invalid.store(1, Ordering::Relaxed);
        assert!(p.lines()[0].contains("ERROR"));
    }
    #[test]
    fn wall_and_mode_selection_preserved() {
        assert!(Mode::parse(None) == Mode::Off);
        assert!(Mode::parse(Some("0")) == Mode::Off);
        assert!(Mode::parse(Some("1")) == Mode::Wall);
        assert!(Mode::parse(Some("gpu")) == Mode::Gpu);
        let (p, f) = profile(Mode::Wall);
        p.record("wall", p.start().unwrap(), 9).unwrap();
        assert_eq!(f.syncs.load(Ordering::Relaxed), 2);
        assert_eq!(f.stamps.load(Ordering::Relaxed), 0);
        assert!(p.lines()[0].contains("synchronized wall time"));
    }
}
