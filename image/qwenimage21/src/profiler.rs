//! Private diagnostic leaf spans. Compile only with the diagnostic feature.
//! Deferred event resolution is intrusive; stream intervals include submission
//! gaps. Host intervals are reported separately, never added to device totals.
use nextrocm_engine_core::{api, Error, Gpu, Result};
use serde_json::{json, Value};
use std::{cell::RefCell, collections::BTreeMap, ffi::c_void, sync::{Arc, OnceLock}, time::Instant};
const MAX_PENDING: usize = 128;
const MAX_SPANS: usize = 2048;
const MAX_GROUPS: usize = 256;
const RING: i64 = 4096;
type Key = (&'static str, Vec<usize>);
type Stamp = unsafe extern "C" fn(*mut c_void, *mut i64) -> i32;
type Elapsed = unsafe extern "C" fn(*mut c_void, i64, i64, *mut f64) -> i32;
struct Events { stamp: Stamp, elapsed: Elapsed }
fn events() -> Result<&'static Events> {
    static API: OnceLock<std::result::Result<Events, String>> = OnceLock::new();
    API.get_or_init(|| {
        let a = api().map_err(|e| e.0)?;
        let stamp = a.symbol("ns_stamp");
        let elapsed = a.symbol("ns_elapsed");
        if stamp.is_null() || elapsed.is_null() { return Err("missing DiT profiling ABI".into()); }
        // SAFETY: exact ns.h ABI; engine retains the runtime library.
        Ok(unsafe { Events { stamp: std::mem::transmute::<*mut c_void, Stamp>(stamp), elapsed: std::mem::transmute::<*mut c_void, Elapsed>(elapsed) } })
    }).as_ref().map_err(|e| Error(e.clone()))
}
#[derive(Default)]
struct Total { count: usize, seconds: f64 }
struct Span { key: Key, start: i64, end: i64, ordinal: usize }
#[derive(Default)]
struct Book {
    pending: Vec<Span>,
    device: BTreeMap<Key, Total>,
    host: BTreeMap<Key, Total>,
    spans: usize,
    host_spans: usize,
    last_ticket: Option<i64>,
    batches: usize,
    samples: Vec<Value>,
}
impl Book {
    fn key(label: &'static str, shape: &[usize]) -> Result<Key> {
        if label.is_empty() || label.len() > 96 || shape.len() > 8 || shape.contains(&0) {
            return Err(Error("invalid bounded profile group".into()));
        }
        Ok((label, shape.to_vec()))
    }
    fn reserve(&mut self, key: &Key) -> Result<()> {
        if self.spans >= MAX_SPANS || (!self.device.contains_key(key) && self.device.len() + self.host.len() >= MAX_GROUPS) {
            return Err(Error("DiT profile span/group cap reached".into()));
        }
        self.device.entry(key.clone()).or_default();
        self.spans += 1;
        Ok(())
    }
    fn ticket(&mut self, ticket: i64) -> Result<()> {
        if ticket < 0 || self.last_ticket.is_some_and(|last| ticket <= last)
            || self.pending.first().is_some_and(|first| ticket - first.start >= RING) {
            return Err(Error("DiT profile event ticket order/ring invalid".into()));
        }
        self.last_ticket = Some(ticket);
        Ok(())
    }
    fn add(&mut self, key: Key, start: i64, end: i64) -> Result<()> {
        if end <= start || end - start >= RING || self.pending.len() >= MAX_PENDING {
            return Err(Error("DiT profile unresolved span invalid".into()));
        }
        self.pending.push(Span { key, start, end, ordinal: self.spans });
        Ok(())
    }
    fn host(&mut self, key: Key, seconds: f64) -> Result<()> {
        if !seconds.is_finite() || seconds < 0.0 || self.host_spans >= MAX_SPANS
            || (!self.host.contains_key(&key) && self.device.len() + self.host.len() >= MAX_GROUPS) {
            return Err(Error("DiT host profile interval/cap invalid".into()));
        }
        let total = self.host.entry(key).or_default();
        total.count += 1;
        total.seconds += seconds;
        if !total.seconds.is_finite() { return Err(Error("host duration sum overflow".into())); }
        self.host_spans += 1;
        Ok(())
    }
}
struct Session { gpu: Arc<Gpu>, book: Book, wall: Instant, open: bool, failed: bool }
impl Session {
    fn stamp(&mut self) -> Result<i64> {
        let mut t = -1;
        // SAFETY: live GPU and valid out pointer; library remains loaded.
        if unsafe { (events()?.stamp)(self.gpu.raw(), &mut t) } != 0 {
            return Err(Error("DiT event stamp failed".into()));
        }
        self.book.ticket(t)?;
        Ok(t)
    }
    fn flush(&mut self) -> Result<()> {
        if self.book.pending.is_empty() { return Ok(()); }
        // Audit the current ticket before resolving: unrelated stream-copy
        // tickets must not have overwritten any pending event.
        self.stamp()?;
        let e = events()?;
        for span in self.book.pending.drain(..) {
            let mut ns = f64::NAN;
            // SAFETY: same GPU, validated bounded tickets, live output pointer.
            if unsafe { (e.elapsed)(self.gpu.raw(), span.start, span.end, &mut ns) } != 0 || !ns.is_finite() || ns < 0.0 {
                return Err(Error("DiT event elapsed invalid".into()));
            }
            let total = self.book.device.get_mut(&span.key).ok_or_else(|| Error("unreserved profile group".into()))?;
            total.count += 1;
            total.seconds += ns / 1e9;
            self.book.samples.push(json!({"ordinal":span.ordinal,"operation":span.key.0,"shape":span.key.1,"group_occurrence":total.count,"seconds":ns/1e9}));
            if !total.seconds.is_finite() { return Err(Error("device duration sum overflow".into())); }
        }
        self.book.batches += 1;
        Ok(())
    }
}
thread_local! { static STATE: RefCell<Option<Session>> = const { RefCell::new(None) }; }
/// Begin after model/prefix/input preparation and before one velocity call.
pub fn begin(gpu: &Arc<Gpu>) -> Result<()> {
    if std::env::var("NS_PROFILE").as_deref() != Ok("gpu") { return Err(Error("DiT profiler requires NS_PROFILE=gpu before GPU open".into())); }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if s.is_some() { return Err(Error("DiT profile already active; finish or abort first".into())); }
        events()?;
        gpu.sync()?;
        *s = Some(Session { gpu: gpu.clone(), book: Book::default(), wall: Instant::now(), open: false, failed: false });
        Ok(())
    })
}
struct PoisonOnDrop(bool);
impl Drop for PoisonOnDrop {
    fn drop(&mut self) {
        if self.0 { STATE.with(|s| { if let Some(s) = s.borrow_mut().as_mut() { s.failed = true; s.open = false; } }); }
    }
}
/// One leaf operation; nested GPU marks are rejected to prevent double counting.
/// Inactive calls execute the original closure without event/environment work.
pub fn mark<T>(gpu: &Arc<Gpu>, label: &'static str, shape: &[usize], f: impl FnOnce() -> Result<T>) -> Result<T> {
    let prepared = STATE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(s) = slot.as_mut() else { return Ok(None); };
        let result = (|| {
            if s.failed || s.open || !Arc::ptr_eq(&s.gpu, gpu) { return Err(Error("DiT profile failed/nested/foreign GPU".into())); }
            let key = Book::key(label, shape)?;
            if s.book.pending.len() == MAX_PENDING { s.flush()?; }
            s.book.reserve(&key)?;
            let start = s.stamp()?;
            s.open = true;
            Ok(Some((key, start)))
        })();
        if result.is_err() { s.failed = true; }
        result
    })?;
    let Some((key, start)) = prepared else { return f(); };
    let mut guard = PoisonOnDrop(true);
    let value = f()?;
    STATE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let s = slot.as_mut().ok_or_else(|| Error("profile session removed inside span".into()))?;
        if s.failed || !s.open { return Err(Error("profile session invalidated inside span".into())); }
        let end = s.stamp()?;
        s.book.add(key, start, end)?;
        s.open = false;
        Ok(())
    })?;
    guard.0 = false;
    Ok(value)
}
/// Record explicit host-only phases. Do not record a parent scope containing
/// device marks: host totals are separate and must not be used as GPU attribution.
pub fn record_host(label: &'static str, shape: &[usize], seconds: f64) -> Result<()> {
    STATE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let Some(s) = slot.as_mut() else { return Ok(()); };
        let result = if s.failed || s.open { Err(Error("invalid host profile nesting/state".into())) }
            else { Book::key(label, shape).and_then(|key| s.book.host(key, seconds)) };
        if result.is_err() { s.failed = true; }
        result
    })
}
/// Consume the session; a failed/incomplete profile can never produce a report.
pub fn finish() -> Result<Value> {
    let mut s = STATE.with(|slot| slot.borrow_mut().take()).ok_or_else(|| Error("no active DiT profile".into()))?;
    if s.failed || s.open { return Err(Error("DiT profile poisoned or incomplete".into())); }
    s.flush()?;
    s.gpu.sync()?;
    let count: usize = s.book.device.values().map(|v| v.count).sum();
    if count == 0 || count != s.book.spans { return Err(Error("incomplete DiT profile accounting".into())); }
    let rows = |groups: &BTreeMap<Key, Total>| -> Vec<Value> {
        groups.iter().map(|((label, shape), t)| json!({"operation":label,"shape":shape,"count":t.count,"seconds":t.seconds})).collect()
    };
    Ok(json!({"schema":1,"scope":"one velocity diagnostic; deferred flush intrusive, stream intervals include submission gaps","metric":"gpu_stream_interval","operations":rows(&s.book.device),"samples":s.book.samples,"host_operations":rows(&s.book.host),"operation_count":count,"host_operation_count":s.book.host_spans,"sum_stream_seconds":s.book.device.values().map(|v|v.seconds).sum::<f64>(),"synchronized_wall_seconds":s.wall.elapsed().as_secs_f64(),"flush_batches":s.book.batches,"max_pending_pairs":MAX_PENDING,"span_cap":MAX_SPANS,"group_cap":MAX_GROUPS,"event_ring":RING,"dropped_spans":0}))
}
/// Discard diagnostics after an inference error. This does not cancel GPU work;
/// the caller retains the normal synchronized cleanup/ownership obligations.
pub fn abort() { STATE.with(|slot| { slot.borrow_mut().take(); }); }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_order_wrap_and_pending_bounds() {
        let mut b = Book::default();
        let k = Book::key("test", &[256,4096]).unwrap();
        b.reserve(&k).unwrap(); b.ticket(0).unwrap(); b.ticket(1).unwrap(); b.add(k.clone(),0,1).unwrap();
        assert!(b.ticket(1).is_err()); assert!(b.ticket(-1).is_err()); assert!(b.ticket(RING).is_err());
        assert!(b.add(k.clone(), 2, 2).is_err()); assert!(b.add(k.clone(),2,2+RING).is_err());
        for i in 1..MAX_PENDING { b.add(k.clone(), (i*2) as i64,(i*2+1) as i64).unwrap(); }
        assert!(b.add(k,256,257).is_err()); assert!(MAX_PENDING*2 < RING as usize);
    }
    #[test]
    fn accounting_caps_and_invalid_host_values() {
        let mut b = Book::default(); let k=Book::key("test", &[]).unwrap();
        for _ in 0..MAX_SPANS { b.reserve(&k).unwrap(); }
        assert!(b.reserve(&k).is_err());
        let mut b=Book::default();
        for i in 1..=MAX_GROUPS { b.reserve(&Book::key("shape", &[i]).unwrap()).unwrap(); }
        assert!(b.reserve(&Book::key("shape", &[MAX_GROUPS+1]).unwrap()).is_err());
        assert!(b.host(k.clone(),1.).is_err());
        let mut b=Book::default();
        for v in [f64::NAN,f64::INFINITY,-1.] { assert!(b.host(k.clone(),v).is_err()); }
        b.host(k,0.25).unwrap(); assert_eq!(b.host_spans,1);
        assert!(Book::key("",&[]).is_err()); assert!(Book::key("x",&[0]).is_err()); assert!(Book::key("x",&[1;9]).is_err());
    }
}
