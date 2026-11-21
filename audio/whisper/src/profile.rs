//! Private opt-in encoder event attribution; never shared ops or throughput proof.
//! At most 128 pending pairs (256 events) in the native 4096-event ring. Resolving
//! a batch synchronizes its last event, so wall time includes these diagnostic
//! stalls; summed stream intervals can include host submission gaps too.
use nextrocm_audio::{Error, LoadOptions, Result};
use nextrocm_engine_core::{Gpu, api};
use serde_json::json;
use std::{collections::BTreeMap, time::Instant};
const MAX_PENDING: usize = 128;
const RING: i64 = 4096;
pub fn enabled(options: &LoadOptions) -> Result<bool> {
    match options.setting("NS_WHISPER_PROFILE").as_deref() {
        None | Some("0") => Ok(false),
        Some("1") => {
            if std::env::var("NS_PROFILE").as_deref() != Ok("gpu") {
                return Err(Error(
                    "NS_WHISPER_PROFILE=1 requires process NS_PROFILE=gpu before GPU open".into(),
                ));
            }
            Ok(true)
        }
        Some(_) => Err(Error("NS_WHISPER_PROFILE must be 0 or 1".into())),
    }
}
#[derive(Default)]
struct Total {
    count: usize,
    ns: f64,
}
struct Span {
    op: &'static str,
    shape: Vec<usize>,
    start: i64,
    end: i64,
}
pub struct Encoder<'a> {
    gpu: &'a Gpu,
    enabled: bool,
    pending: Vec<Span>,
    totals: BTreeMap<(&'static str, Vec<usize>), Total>,
    wall: Instant,
    batches: usize,
}
impl<'a> Encoder<'a> {
    pub fn new(gpu: &'a Gpu, enabled: bool) -> Result<Self> {
        if enabled {
            gpu.sync()?;
        }
        Ok(Self {
            gpu,
            enabled,
            pending: Vec::new(),
            totals: BTreeMap::new(),
            wall: Instant::now(),
            batches: 0,
        })
    }
    fn stamp(&self) -> Result<i64> {
        let a = api()?;
        let mut ticket = -1;
        // SAFETY: live GPU; synchronous host call writes a valid ticket pointer.
        if unsafe { (a.stamp)(self.gpu.raw(), &mut ticket) } != 0 {
            return Err(Error(a.error()));
        }
        if ticket < 0 {
            return Err(Error("negative profile event ticket".into()));
        }
        Ok(ticket)
    }
    pub fn run<T>(
        &mut self,
        op: &'static str,
        shape: &[usize],
        f: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        if !self.enabled {
            return f();
        }
        if self.pending.len() >= MAX_PENDING {
            self.flush()?;
        }
        let start = self.stamp()?;
        let value = f()?;
        let end = self.stamp()?;
        let earliest = self.pending.first().map_or(start, |p| p.start);
        if end < start || end - earliest >= RING {
            return Err(Error(
                "Whisper profile event ring reused before resolution".into(),
            ));
        }
        self.pending.push(Span {
            op,
            shape: shape.to_vec(),
            start,
            end,
        });
        Ok(value)
    }
    fn flush(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let a = api()?;
        for span in self.pending.drain(..) {
            let mut ns = 0.;
            // SAFETY: tickets from this GPU stream, bounded before ring reuse,
            // valid f64 output. Native elapsed synchronizes each end event.
            if unsafe { (a.elapsed)(self.gpu.raw(), span.start, span.end, &mut ns) } != 0 {
                return Err(Error(a.error()));
            }
            if !ns.is_finite() || ns < 0. {
                return Err(Error("invalid GPU event interval".into()));
            }
            let total = self.totals.entry((span.op, span.shape)).or_default();
            total.count += 1;
            total.ns += ns;
        }
        self.batches += 1;
        Ok(())
    }
    pub fn finish(mut self, tiled: bool) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        self.flush()?;
        self.gpu.sync()?;
        let mut sum = 0.;
        let mut count = 0;
        let rows: Vec<_> = self
            .totals
            .iter()
            .map(|((op, shape), t)| {
                sum += t.ns;
                count += t.count;
                json!({"op":op,"shape":shape,"count":t.count,"stream_seconds":t.ns/1e9})
            })
            .collect();
        eprintln!(
            "WHISPER_PROFILE {}",
            json!({"schema":1,"linear_tiled":tiled,"scope":"one encoder window, upload excluded; event intervals include submission gaps; bounded flush synchronization is intrusive",
            "metric":"gpu_stream_interval","operations":rows,"operation_count":count,"sum_stream_seconds":sum/1e9,"synchronized_wall_seconds":self.wall.elapsed().as_secs_f64(),"flush_batches":self.batches,"max_pending_pairs":MAX_PENDING,"event_ring":RING})
        );
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pending_bound_cannot_fill_ring() {
        assert!(MAX_PENDING * 2 < RING as usize);
    }
    #[test]
    fn explicit_zero_and_invalid_controls() {
        let mut o = LoadOptions::default();
        o.settings.insert("NS_WHISPER_PROFILE".into(), "0".into());
        assert!(!enabled(&o).unwrap());
        o.settings.insert("NS_WHISPER_PROFILE".into(), "yes".into());
        assert!(enabled(&o).is_err());
    }
}
