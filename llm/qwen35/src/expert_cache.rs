//! Optional packed expert residency under the original LoadOptions budget.
//! Entries own both the allocation and its reservation; leases survive eviction.
use super::{
    ensure_memory, error,
    store::{Reservation, Store, Weight},
    DevBuf, Gpu, Result,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

type Key = (usize, u64, usize);
pub(crate) struct Cached {
    pub buffer: DevBuf,
    _reservation: Reservation,
}
struct Entry {
    tick: u64,
    data: Arc<Cached>,
}
#[derive(Default)]
struct State {
    entries: BTreeMap<Key, Entry>,
    order: BTreeMap<u64, Key>,
    tick: u64,
    bytes: usize,
    hits: u64,
    misses: u64,
    hit_bytes: u64,
    loaded_bytes: u64,
    evictions: u64,
    fallbacks: u64,
    reuses: u64,
}
pub(crate) struct ExpertCache {
    limit: usize,
    state: Mutex<State>,
}
impl State {
    /// Replace an unleased, same-sized LRU entry without freeing its allocation.
    /// The writer must order its overwrite after previous consumers and complete
    /// the host transfer before returning (Store::copy provides both guarantees).
    fn reuse(
        &mut self,
        key: Key,
        tick: u64,
        write: impl FnOnce(&DevBuf) -> Result<()>,
    ) -> Result<Option<Arc<Cached>>> {
        let victim = self.order.iter().find_map(|(&age, old)| {
            self.entries
                .get(old)
                .filter(|entry| {
                    old.2 == key.2
                        && Arc::strong_count(&entry.data) == 1
                        && Arc::weak_count(&entry.data) == 0
                })
                .map(|_| (age, *old))
        });
        let Some((age, old)) = victim else {
            return Ok(None);
        };
        let mut entry = self
            .entries
            .remove(&old)
            .ok_or_else(|| error("expert cache missing LRU entry"))?;
        // get_mut proves exclusive ownership, including absence of weak leases.
        // If ownership changed, restore the untouched entry and use fallback.
        let Some(data) = Arc::get_mut(&mut entry.data) else {
            self.entries.insert(old, entry);
            return Ok(None);
        };
        self.order.remove(&age);
        self.evictions += 1;
        // Keep bytes and the reservation charged throughout the overwrite.
        // On failure neither the stale key nor partially-written data is visible.
        if let Err(err) = write(&data.buffer) {
            self.bytes -= key.2;
            drop(entry);
            return Err(err);
        }
        entry.tick = tick;
        let lease = Arc::clone(&entry.data);
        self.entries.insert(key, entry);
        self.order.insert(tick, key);
        self.loaded_bytes += key.2 as u64;
        self.reuses += 1;
        Ok(Some(lease))
    }
    fn evict(&mut self) -> bool {
        // Do not count a still-leased allocation as freed budget.
        let victim = self.order.iter().find_map(|(&tick, key)| {
            self.entries
                .get(key)
                .filter(|e| Arc::strong_count(&e.data) == 1)
                .map(|_| (tick, *key))
        });
        let Some((tick, key)) = victim else {
            return false;
        };
        self.order.remove(&tick);
        if let Some(entry) = self.entries.remove(&key) {
            self.bytes -= key.2;
            // DevBuf drop synchronizes before freeing queued kernel inputs.
            drop(entry);
            self.evictions += 1;
        }
        true
    }
}
impl ExpertCache {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            state: Mutex::new(State::default()),
        }
    }
    pub fn get(
        &self,
        gpu: &Arc<Gpu>,
        store: &Store,
        w: &Weight,
        offset: usize,
        len: usize,
    ) -> Result<Option<Arc<Cached>>> {
        if self.limit == 0 || len == 0 || len > self.limit || !w.streamed() {
            return Ok(None);
        }
        if offset
            .checked_add(len)
            .is_none_or(|end| end as u64 > w.tensor.bytes)
        {
            return Err(error("expert cache window outside tensor"));
        }
        let absolute = w
            .tensor
            .offset
            .checked_add(offset as u64)
            .ok_or_else(|| error("expert cache offset overflow"))?;
        let key = (w.tensor.shard, absolute, len);
        let mut state = self
            .state
            .lock()
            .map_err(|_| error("expert cache lock poisoned"))?;
        state.tick = state
            .tick
            .checked_add(1)
            .ok_or_else(|| error("expert cache age overflow"))?;
        let tick = state.tick;
        if let Some(entry) = state.entries.get_mut(&key) {
            let previous = entry.tick;
            entry.tick = tick;
            let data = Arc::clone(&entry.data);
            state.order.remove(&previous);
            state.order.insert(tick, key);
            state.hits += 1;
            state.hit_bytes += len as u64;
            return Ok(Some(data));
        }
        state.misses += 1;
        while state.bytes.checked_add(len).is_none_or(|n| n > self.limit) {
            if let Some(data) =
                state.reuse(key, tick, |buffer| store.copy(w, offset, len, buffer))?
            {
                return Ok(Some(data));
            }
            if !state.evict() {
                state.fallbacks += 1;
                return Ok(None);
            }
        }
        // Cache is optional. Under pressure evict its own entries before using
        // the existing bounded staging fallback; do not evict another model.
        loop {
            let (_, free) = gpu.memory()?;
            if free.is_none_or(|n| {
                (len as u64)
                    .checked_add(super::MARGIN)
                    .is_some_and(|need| need <= n)
            }) {
                break;
            }
            if let Some(data) =
                state.reuse(key, tick, |buffer| store.copy(w, offset, len, buffer))?
            {
                return Ok(Some(data));
            }
            if !state.evict() {
                state.fallbacks += 1;
                return Ok(None);
            }
        }
        let reservation = match Reservation::device(len) {
            Ok(r) => r,
            Err(_) => {
                if let Some(data) =
                    state.reuse(key, tick, |buffer| store.copy(w, offset, len, buffer))?
                {
                    return Ok(Some(data));
                }
                state.fallbacks += 1;
                return Ok(None);
            }
        };
        ensure_memory(gpu, len as u64)?;
        let buffer = DevBuf::new(gpu, len)?;
        store.copy(w, offset, len, &buffer)?;
        let data = Arc::new(Cached {
            buffer,
            _reservation: reservation,
        });
        state.entries.insert(
            key,
            Entry {
                tick,
                data: Arc::clone(&data),
            },
        );
        state.order.insert(tick, key);
        state.bytes += len;
        state.loaded_bytes += len as u64;
        Ok(Some(data))
    }
    pub fn report(&self) -> Vec<String> {
        if self.limit == 0 {
            return Vec::new();
        }
        let Ok(s) = self.state.lock() else {
            return vec!["expert cache lock poisoned".into()];
        };
        vec![format!("qwen35 expert cache: limit={} resident={} entries={} hits={} misses={} hit_bytes={} loaded_bytes={} evictions={} fallbacks={} reuses={}", self.limit,s.bytes,s.entries.len(),s.hits,s.misses,s.hit_bytes,s.loaded_bytes,s.evictions,s.fallbacks,s.reuses)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn insert(s: &mut State, gpu: &Arc<Gpu>, tick: u64, key: Key) {
        let buffer = DevBuf::new(gpu, key.2).unwrap();
        buffer.write(0, &vec![tick as u8; key.2]).unwrap();
        s.entries.insert(
            key,
            Entry {
                tick,
                data: Arc::new(Cached {
                    buffer,
                    _reservation: Reservation::device(key.2).unwrap(),
                }),
            },
        );
        s.order.insert(tick, key);
        s.bytes += key.2;
    }
    #[test]
    fn reuse_keeps_pointer_reservation_and_budget_until_final_lease_drops() {
        let gpu = Gpu::detached("reuse ownership");
        let mut s = State::default();
        insert(&mut s, &gpu, 1, (0, 0, 4));
        let old = &s.entries[&(0, 0, 4)].data;
        let pointer = old.buffer.ptr();
        let reservation = &old._reservation as *const Reservation;
        let lease = s
            .reuse((0, 8, 4), 2, |buffer| buffer.write(0, &[9; 4]))
            .unwrap()
            .unwrap();
        assert_eq!(lease.buffer.ptr(), pointer);
        assert_eq!(&lease._reservation as *const Reservation, reservation);
        assert_eq!(s.bytes, 4);
        assert!(!s.entries.contains_key(&(0, 0, 4)));
        assert_eq!(
            s.order.iter().map(|(t, k)| (*t, *k)).collect::<Vec<_>>(),
            [(2, (0, 8, 4))]
        );
        let mut bytes = [0; 4];
        lease.buffer.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [9; 4]);
        assert!(!s.evict());
        let weak = Arc::downgrade(&lease);
        drop(s);
        assert!(weak.upgrade().is_some()); // allocation + reservation still owned by live lease
        drop(lease);
        assert!(weak.upgrade().is_none()); // Cached drops buffer, then its reservation
    }
    #[test]
    fn reuse_selects_oldest_same_size_unleased_entry() {
        let gpu = Gpu::detached("reuse LRU");
        let mut s = State::default();
        for (tick, key) in [
            (1, (0, 0, 4)),
            (2, (0, 4, 8)),
            (3, (0, 12, 4)),
            (4, (0, 16, 4)),
        ] {
            insert(&mut s, &gpu, tick, key);
        }
        let lease = Arc::clone(&s.entries[&(0, 0, 4)].data);
        let expected = s.entries[&(0, 12, 4)].data.buffer.ptr();
        let reused = s
            .reuse((0, 20, 4), 5, |buffer| buffer.write(0, &[7; 4]))
            .unwrap()
            .unwrap();
        assert_eq!(reused.buffer.ptr(), expected);
        assert_eq!(s.bytes, 20);
        assert!(s.entries.contains_key(&(0, 0, 4)));
        assert!(s.entries.contains_key(&(0, 4, 8)));
        assert!(s.entries.contains_key(&(0, 16, 4)));
        let mut bytes = [0; 4];
        lease.buffer.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [1; 4]);
        drop(reused);
        drop(lease);
        while s.evict() {}
        assert_eq!(s.bytes, 0);
        assert!(s.entries.is_empty());
        assert!(s.order.is_empty());
    }
    #[test]
    fn failed_overwrite_unpublishes_both_keys_and_releases_budget() {
        let gpu = Gpu::detached("reuse failure");
        let mut s = State::default();
        insert(&mut s, &gpu, 1, (0, 0, 4));
        assert!(s
            .reuse((0, 4, 8), 2, |_| panic!("wrong-sized buffer reused"))
            .unwrap()
            .is_none());
        let result = s.reuse((0, 8, 4), 3, |buffer| {
            buffer.write(0, &[3; 4])?;
            Err(error("copy failed"))
        });
        assert!(result.is_err());
        assert_eq!(s.bytes, 0);
        assert!(s.entries.is_empty());
        assert!(s.order.is_empty());
        assert_eq!(s.reuses, 0);
    }
    #[test]
    fn weak_lease_is_not_overwritten() {
        let gpu = Gpu::detached("reuse weak lease");
        let mut s = State::default();
        insert(&mut s, &gpu, 1, (0, 0, 4));
        let weak = Arc::downgrade(&s.entries[&(0, 0, 4)].data);
        assert!(s
            .reuse((0, 4, 4), 2, |_| panic!("weak leased buffer reused"))
            .unwrap()
            .is_none());
        assert_eq!(s.bytes, 4);
        assert!(weak.upgrade().is_some());
    }
    #[test]
    fn leased_buffers_survive_eviction_and_accounting_reclaims_only_unleased() {
        let gpu = Gpu::detached("cache ownership test");
        let mut s = State::default();
        for (tick, key) in [(1, (0, 0, 4)), (2, (0, 4, 4))] {
            let buffer = DevBuf::new(&gpu, 4).unwrap();
            buffer.write(0, &[tick as u8; 4]).unwrap();
            let data = Arc::new(Cached {
                buffer,
                _reservation: Reservation::device(4).unwrap(),
            });
            s.entries.insert(key, Entry { tick, data });
            s.order.insert(tick, key);
            s.bytes += 4;
        }
        let lease = Arc::clone(&s.entries[&(0, 0, 4)].data);
        assert!(s.evict());
        assert_eq!(s.bytes, 4);
        assert!(s.entries.contains_key(&(0, 0, 4)));
        assert!(!s.entries.contains_key(&(0, 4, 4)));
        assert!(!s.evict());
        let mut bytes = [0; 4];
        lease.buffer.read(0, &mut bytes).unwrap();
        assert_eq!(bytes, [1; 4]);
        drop(lease);
        assert!(s.evict());
        assert_eq!(s.bytes, 0);
        assert!(s.entries.is_empty());
        assert!(s.order.is_empty());
    }
}
