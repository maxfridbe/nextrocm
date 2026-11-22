//! Explicit packed-pair staging; no dequantization or inference math changes.
//! One owner, one GPU stream, synchronous uploads. Not an asynchronous prefetcher.
use crate::{
    store::{ExpertSpan, HostStore},
    Result,
};
use nextrocm_engine_core::{DevBuf, Gpu, HostBuf};
use std::sync::Arc;
pub const MAX_CACHE_BYTES: u64 = 1 << 30;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExpertMode {
    #[default]
    Stream,
    BatchPair,
    DeviceLru {
        budget_bytes: u64,
    },
    SegmentedLru {
        budget_bytes: u64,
    },
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Pair payload only: excludes dense weights and the two four-byte scale reads.
/// Misses/evictions count attempts; byte/copy counters count completed operations.
pub struct ExpertStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub read_bytes: u64,
    pub uploaded_bytes: u64,
    pub upload_calls: u64,
}
impl ExpertMode {
    /// Slab uses whole largest-pair slots. Variable ranges share the original slab; metadata has a 256KiB reserve.
    pub fn extra_bytes(self, pair: u64) -> Result<(u64, u64)> {
        if pair == 0 {
            return Err("zero expert pair".into());
        }
        match self {
            Self::Stream => Ok((0, 0)),
            Self::BatchPair => Ok((0, pair)),
            Self::SegmentedLru { budget_bytes } | Self::DeviceLru { budget_bytes } => {
                if budget_bytes > MAX_CACHE_BYTES || budget_bytes < pair {
                    return Err("expert cache budget must hold one pair and be <=1GiB".into());
                }
                let slots = budget_bytes / pair;
                if slots > 3072 {
                    return Err("expert cache exceeds physical expert count".into());
                }
                Ok((
                    slots * pair,
                    pair.checked_add(if matches!(self, Self::SegmentedLru { .. }) {
                        262144
                    } else {
                        131072
                    })
                    .ok_or("host cache size overflow")?,
                ))
            }
        }
    }
}
#[path = "range_lru.rs"]
mod range_lru;
use range_lru::{Key, RangeLru};
pub(crate) struct VariableStaging {
    slab: Option<DevBuf>,
    lru: Option<RangeLru>,
    host: HostBuf,
    pair_bytes: usize,
}
impl VariableStaging {
    pub fn new(gpu: &Arc<Gpu>, mode: ExpertMode, pair_bytes: usize) -> Result<Option<Self>> {
        let (slab_bytes, _) = mode.extra_bytes(pair_bytes as u64)?;
        if mode == ExpertMode::Stream {
            return Ok(None);
        }
        // The separately reserved host pair is page-locked, not a second device
        // slab. It stays alive until the synchronous upload has completed.
        let mut host = HostBuf::new(gpu, pair_bytes).map_err(|e| e.0)?;
        host.as_mut_slice().fill(0);
        let (slab, lru) = if slab_bytes != 0 {
            (
                Some(
                    DevBuf::new(
                        gpu,
                        usize::try_from(slab_bytes).map_err(|_| "cache size exceeds usize")?,
                    )
                    .map_err(|e| e.0)?,
                ),
                Some(RangeLru::new(
                    usize::try_from(slab_bytes).map_err(|_| "cache size exceeds usize")?,
                )?),
            )
        } else {
            (None, None)
        };
        Ok(Some(Self {
            slab,
            lru,
            host,
            pair_bytes,
        }))
    }
    /// Returned non-owning views are callback-only, never retained across a call.
    /// Same-stream write then sync orders victim overwrite after prior consumers.
    pub fn pair(
        &mut self,
        fallback: &DevBuf,
        store: &mut HostStore,
        layer: usize,
        expert: usize,
        spans: &[ExpertSpan; 2],
        stats: &mut ExpertStats,
        check: &mut impl FnMut() -> Result<()>,
    ) -> Result<(DevBuf, DevBuf)> {
        check()?;
        let key = Key { layer, expert };
        let first = usize::try_from(spans[0].bytes).map_err(|_| "pair size overflow")?;
        let second = usize::try_from(spans[1].bytes).map_err(|_| "pair size overflow")?;
        let total = first.checked_add(second).ok_or("pair size overflow")?;
        if total > self.pair_bytes {
            return Err("expert exceeds reserved pair size".into());
        }
        let hit = match &mut self.lru {
            Some(lru) => lru.hit(key, total)?,
            None => None,
        };
        let base = if let Some(offset) = hit {
            stats.hits += 1;
            offset
        } else {
            stats.misses += 1;
            let reservation = if let Some(lru) = &mut self.lru {
                let (reservation, evicted) = lru.reserve(total)?;
                stats.evictions += evicted;
                Some(reservation)
            } else {
                None
            };
            let offset = reservation.as_ref().map_or(0, |r| r.offset);
            for (span, base) in spans.iter().zip([0, first]) {
                check()?;
                let bytes = usize::try_from(span.bytes).map_err(|_| "pair size overflow")?;
                let end = base.checked_add(bytes).ok_or("pair offset overflow")?;
                let dst = self
                    .host
                    .as_mut_slice()
                    .get_mut(base..end)
                    .ok_or("pair host bounds")?;
                store.read_span_into(&span.tensor, span.offset, dst)?;
                stats.read_bytes += span.bytes;
            }
            check()?;
            let target = self.slab.as_ref().unwrap_or(fallback);
            target
                .write(offset, &self.host.as_slice()[..total])
                .map_err(|e| e.0)?;
            stats.upload_calls += 1;
            stats.uploaded_bytes += total as u64;
            check()?;
            if let (Some(lru), Some(reservation)) = (&mut self.lru, reservation) {
                lru.commit(reservation, key)?;
            }
            offset
        };
        check()?;
        let target = self.slab.as_ref().unwrap_or(fallback);
        Ok((
            target.view(base, first).map_err(|e| e.0)?,
            target.view(base + first, second).map_err(|e| e.0)?,
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn budget_is_bounded_and_whole_slots() {
        let pair = 6469632;
        assert_eq!(ExpertMode::Stream.extra_bytes(pair).unwrap(), (0, 0));
        assert_eq!(ExpertMode::BatchPair.extra_bytes(pair).unwrap(), (0, pair));
        let (slab, host) = ExpertMode::SegmentedLru {
            budget_bytes: MAX_CACHE_BYTES,
        }
        .extra_bytes(pair)
        .unwrap();
        assert_eq!(slab, pair * 165);
        assert_eq!(host, pair + 262144);
        assert_eq!(
            ExpertMode::DeviceLru {
                budget_bytes: MAX_CACHE_BYTES
            }
            .extra_bytes(pair)
            .unwrap(),
            (pair * 165, pair + 131072)
        );
        assert_eq!(ExpertMode::default(), ExpertMode::Stream);
        for budget_bytes in [0, pair - 1, MAX_CACHE_BYTES + 1] {
            assert!(ExpertMode::SegmentedLru { budget_bytes }
                .extra_bytes(pair)
                .is_err());
        }
    }
}

// The fixed control is the complete frozen implementation, without source edits.
#[path = "fixed_cache.rs"]
mod fixed_cache;
pub(crate) enum PairStaging {
    Fixed(fixed_cache::PairStaging),
    Variable(VariableStaging),
}
impl PairStaging {
    pub fn new(gpu: &Arc<Gpu>, mode: ExpertMode, pair_bytes: usize) -> Result<Option<Self>> {
        let fixed = match mode {
            ExpertMode::Stream => Some(fixed_cache::ExpertMode::Stream),
            ExpertMode::BatchPair => Some(fixed_cache::ExpertMode::BatchPair),
            ExpertMode::DeviceLru { budget_bytes } => {
                Some(fixed_cache::ExpertMode::DeviceLru { budget_bytes })
            }
            ExpertMode::SegmentedLru { .. } => None,
        };
        if let Some(fixed) = fixed {
            return fixed_cache::PairStaging::new(gpu, fixed, pair_bytes)
                .map(|x| x.map(Self::Fixed));
        }
        VariableStaging::new(gpu, mode, pair_bytes).map(|x| x.map(Self::Variable))
    }
    pub fn pair(
        &mut self,
        fallback: &DevBuf,
        store: &mut HostStore,
        layer: usize,
        expert: usize,
        spans: &[ExpertSpan; 2],
        stats: &mut ExpertStats,
        check: &mut impl FnMut() -> Result<()>,
    ) -> Result<(DevBuf, DevBuf)> {
        match self {
            Self::Variable(s) => s.pair(fallback, store, layer, expert, spans, stats, check),
            Self::Fixed(s) => {
                let mut old = fixed_cache::ExpertStats {
                    hits: stats.hits,
                    misses: stats.misses,
                    evictions: stats.evictions,
                    read_bytes: stats.read_bytes,
                    uploaded_bytes: stats.uploaded_bytes,
                    upload_calls: stats.upload_calls,
                };
                let result = s.pair(fallback, store, layer, expert, spans, &mut old, check);
                *stats = ExpertStats {
                    hits: old.hits,
                    misses: old.misses,
                    evictions: old.evictions,
                    read_bytes: old.read_bytes,
                    uploaded_bytes: old.uploaded_bytes,
                    upload_calls: old.upload_calls,
                };
                result
            }
        }
    }
}
