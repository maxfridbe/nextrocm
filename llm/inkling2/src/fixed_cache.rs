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
    /// Slab uses whole largest-pair slots. Metadata has an explicit 128KiB reserve.
    pub fn extra_bytes(self, pair: u64) -> Result<(u64, u64)> {
        if pair == 0 {
            return Err("zero expert pair".into());
        }
        match self {
            Self::Stream => Ok((0, 0)),
            Self::BatchPair => Ok((0, pair)),
            Self::DeviceLru { budget_bytes } => {
                if budget_bytes > MAX_CACHE_BYTES || budget_bytes < pair {
                    return Err("expert cache budget must hold one pair and be <=1GiB".into());
                }
                let slots = budget_bytes / pair;
                if slots > 3072 {
                    return Err("expert cache exceeds physical expert count".into());
                }
                Ok((
                    slots * pair,
                    pair.checked_add(131072).ok_or("host cache size overflow")?,
                ))
            }
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    layer: usize,
    expert: usize,
}
#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    key: Option<Key>,
    touched: u64,
}
struct Lru {
    entries: Vec<Entry>,
    clock: u64,
}
impl Lru {
    fn new(slots: usize) -> Result<Self> {
        if slots == 0 || slots > 3072 {
            return Err("invalid cache slot count".into());
        }
        Ok(Self {
            entries: vec![Entry::default(); slots],
            clock: 0,
        })
    }
    fn tick(&mut self) -> Result<u64> {
        self.clock = self.clock.checked_add(1).ok_or("cache age overflow")?;
        Ok(self.clock)
    }
    fn hit(&mut self, key: Key) -> Result<Option<usize>> {
        if let Some(i) = self.entries.iter().position(|e| e.key == Some(key)) {
            self.entries[i].touched = self.tick()?;
            Ok(Some(i))
        } else {
            Ok(None)
        }
    }
    /// Invalidate BEFORE reading/uploading. A failed or cancelled miss cannot hit.
    fn victim(&mut self) -> (usize, bool) {
        let i = self
            .entries
            .iter()
            .position(|e| e.key.is_none())
            .unwrap_or_else(|| {
                self.entries
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| e.touched)
                    .unwrap()
                    .0
            });
        let evicted = self.entries[i].key.take().is_some();
        (i, evicted)
    }
    fn commit(&mut self, i: usize, key: Key) -> Result<()> {
        let touched = self.tick()?;
        self.entries[i] = Entry {
            key: Some(key),
            touched,
        };
        Ok(())
    }
}
pub(crate) struct PairStaging {
    slab: Option<DevBuf>,
    lru: Option<Lru>,
    host: HostBuf,
    pair_bytes: usize,
}
impl PairStaging {
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
                Some(Lru::new((slab_bytes / pair_bytes as u64) as usize)?),
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
            Some(lru) => lru.hit(key)?,
            None => None,
        };
        let slot = if let Some(i) = hit {
            stats.hits += 1;
            i
        } else {
            stats.misses += 1;
            let i = if let Some(lru) = &mut self.lru {
                let (i, evicted) = lru.victim();
                stats.evictions += u64::from(evicted);
                i
            } else {
                0
            };
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
                .write(i * self.pair_bytes, &self.host.as_slice()[..total])
                .map_err(|e| e.0)?;
            stats.upload_calls += 1;
            stats.uploaded_bytes += total as u64;
            check()?;
            if let Some(lru) = &mut self.lru {
                lru.commit(i, key)?;
            }
            i
        };
        check()?;
        let target = self.slab.as_ref().unwrap_or(fallback);
        let base = slot * self.pair_bytes;
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
        let (slab, host) = ExpertMode::DeviceLru {
            budget_bytes: MAX_CACHE_BYTES,
        }
        .extra_bytes(pair)
        .unwrap();
        assert_eq!(slab, pair * 165);
        assert_eq!(host, pair + 131072);
        for budget_bytes in [0, pair - 1, MAX_CACHE_BYTES + 1] {
            assert!(ExpertMode::DeviceLru { budget_bytes }
                .extra_bytes(pair)
                .is_err());
        }
    }
    #[test]
    fn physical_keys_lru_and_failed_fill() {
        let mut p = Lru::new(2).unwrap();
        let a = Key {
            layer: 6,
            expert: 1,
        };
        let b = Key {
            layer: 6,
            expert: 2,
        };
        let c = Key {
            layer: 7,
            expert: 1,
        };
        let (i, e) = p.victim();
        assert!(!e);
        p.commit(i, a).unwrap();
        let (j, e) = p.victim();
        assert!(!e);
        p.commit(j, b).unwrap();
        assert_eq!(p.hit(a).unwrap(), Some(i));
        assert_eq!(p.hit(c).unwrap(), None);
        let (v, e) = p.victim();
        assert!(e);
        assert_eq!(v, j);
        // Simulate cancelled/failed upload: no commit, old AND new keys absent.
        assert_eq!(p.hit(b).unwrap(), None);
        assert_eq!(p.hit(c).unwrap(), None);
        let (v, e) = p.victim();
        assert!(!e);
        p.commit(v, c).unwrap();
        assert_eq!(p.hit(a).unwrap(), Some(i));
        assert_eq!(p.hit(c).unwrap(), Some(v));
    }
}
