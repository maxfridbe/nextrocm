//! Bounded variable-size contiguous segmented LRU. No future requests, GPU calls or moves.
//! Failed fills leave evicted keys invalid; only commit makes new bytes visible.
type Result<T> = std::result::Result<T, String>;
const ALIGN: usize = 256;
const MAX_ENTRIES: usize = 3072;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Key {
    pub layer: usize,
    pub expert: usize,
}
#[derive(Clone, Copy, Debug)]
struct Entry {
    key: Key,
    offset: usize,
    reserved: usize,
    payload: usize,
    touched: u64,
    protected: bool,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Reservation {
    pub offset: usize,
    reserved: usize,
    payload: usize,
}
pub(super) struct RangeLru {
    entries: Vec<Entry>, // Sorted by address, not age. Capacity fixed at construction.
    capacity: usize,
    clock: u64,
}
impl RangeLru {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity < ALIGN {
            return Err("cache slab smaller than one aligned range".into());
        }
        Ok(Self {
            entries: Vec::with_capacity(MAX_ENTRIES),
            capacity,
            clock: 0,
        })
    }
    fn tick(&mut self) -> Result<u64> {
        self.clock = self.clock.checked_add(1).ok_or("cache age overflow")?;
        Ok(self.clock)
    }
    pub fn hit(&mut self, key: Key, payload: usize) -> Result<Option<usize>> {
        let Some(i) = self.entries.iter().position(|e| e.key == key) else {
            return Ok(None);
        };
        if self.entries[i].payload != payload {
            return Err("cached pair size changed".into());
        }
        self.entries[i].touched = self.tick()?;
        self.entries[i].protected = true;
        let offset = self.entries[i].offset;
        self.balance_protected();
        Ok(Some(offset))
    }
    // A fixed, workload-independent 3/4 byte budget protects repeated accesses.
    // New entries are probationary; no counters or future routing guide admission.
    fn balance_protected(&mut self) {
        let limit = self.capacity / 4 * 3;
        while self
            .entries
            .iter()
            .filter(|e| e.protected)
            .map(|e| e.reserved)
            .sum::<usize>()
            > limit
        {
            let i = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| e.protected)
                .min_by_key(|(_, e)| e.touched)
                .map(|(i, _)| i)
                .expect("protected bytes imply entry");
            self.entries[i].protected = false;
        }
    }
    fn gap(&self, bytes: usize) -> Option<usize> {
        let mut end = 0;
        for entry in &self.entries {
            if entry.offset.checked_sub(end)? >= bytes {
                return Some(end);
            }
            end = entry.offset.checked_add(entry.reserved)?;
        }
        (self.capacity.checked_sub(end)? >= bytes).then_some(end)
    }
    /// Caller has exclusive mutable access until commit or abandonment. There
    /// is no pending entry: a failed/cancelled upload cannot become a cache hit.
    pub fn reserve(&mut self, payload: usize) -> Result<(Reservation, u64)> {
        if payload == 0 {
            return Err("zero pair payload".into());
        }
        let reserved = payload
            .checked_add(ALIGN - 1)
            .ok_or("pair alignment overflow")?
            / ALIGN
            * ALIGN;
        if reserved > self.capacity {
            return Err("pair exceeds slab".into());
        }
        let mut evictions = 0;
        loop {
            if self.entries.len() < MAX_ENTRIES {
                if let Some(offset) = self.gap(reserved) {
                    return Ok((
                        Reservation {
                            offset,
                            reserved,
                            payload,
                        },
                        evictions,
                    ));
                }
            }
            let i = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| (e.protected, e.touched))
                .map(|(i, _)| i)
                .ok_or("cache cannot reserve empty slab")?;
            self.entries.remove(i); // Invalidate before reading or writing any replacement bytes.
            evictions += 1;
        }
    }
    pub fn commit(&mut self, reservation: Reservation, key: Key) -> Result<()> {
        let r = reservation;
        let end = r
            .offset
            .checked_add(r.reserved)
            .ok_or("cache range overflow")?;
        if end > self.capacity
            || self.entries.len() >= MAX_ENTRIES
            || self
                .entries
                .iter()
                .any(|e| e.key == key || (r.offset < e.offset + e.reserved && e.offset < end))
        {
            return Err("cache commit overlaps or duplicates existing entry".into());
        }
        let touched = self.tick()?;
        let i = self.entries.partition_point(|e| e.offset < r.offset);
        self.entries.insert(
            i,
            Entry {
                key,
                offset: r.offset,
                reserved: r.reserved,
                payload: r.payload,
                touched,
                protected: false,
            },
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn key(expert: usize) -> Key {
        Key { layer: 6, expert }
    }
    fn fill(c: &mut RangeLru, k: usize, bytes: usize) -> (usize, u64) {
        let (r, evictions) = c.reserve(bytes).unwrap();
        let offset = r.offset;
        c.commit(r, key(k)).unwrap();
        (offset, evictions)
    }
    fn invariant(c: &RangeLru) {
        let mut end = 0;
        for e in &c.entries {
            assert_eq!(e.offset % ALIGN, 0);
            assert_eq!(e.reserved % ALIGN, 0);
            assert!(e.offset >= end && e.payload > 0 && e.payload <= e.reserved);
            end = e.offset + e.reserved;
            assert!(end <= c.capacity);
        }
        assert!(c.entries.len() <= MAX_ENTRIES);
    }
    #[test]
    fn repeated_entries_survive_one_touch_scan() {
        let mut c = RangeLru::new(1024).unwrap();
        for k in 0..4 {
            fill(&mut c, k, 256);
        }
        c.hit(key(0), 256).unwrap();
        for k in 4..40 {
            fill(&mut c, k, 256);
        }
        assert!(c.hit(key(0), 256).unwrap().is_some());
        invariant(&c);
    }
    #[test]
    fn protected_budget_is_bytes_not_entry_count() {
        let mut c = RangeLru::new(1024).unwrap();
        fill(&mut c, 0, 512);
        fill(&mut c, 1, 256);
        fill(&mut c, 2, 256);
        for (k, n) in [(0, 512), (1, 256), (2, 256)] {
            c.hit(key(k), n).unwrap();
        }
        assert!(
            !c.entries
                .iter()
                .find(|e| e.key == key(0))
                .unwrap()
                .protected
        );
        assert_eq!(
            c.entries
                .iter()
                .filter(|e| e.protected)
                .map(|e| e.reserved)
                .sum::<usize>(),
            512
        );
        invariant(&c);
    }
    #[test]
    fn variable_payload_uses_space_without_fixed_slot_waste() {
        let mut c = RangeLru::new(1536).unwrap();
        fill(&mut c, 0, 768);
        fill(&mut c, 1, 256);
        fill(&mut c, 2, 512);
        for (k, size) in [(0, 768), (1, 256), (2, 512)] {
            assert!(c.hit(key(k), size).unwrap().is_some());
        }
        invariant(&c);
    }
    #[test]
    fn fragmented_gap_evicts_lru_until_contiguous_not_just_total_free() {
        let mut c = RangeLru::new(1024).unwrap();
        for k in 0..4 {
            fill(&mut c, k, 256);
        }
        c.hit(key(0), 256).unwrap();
        c.hit(key(2), 256).unwrap();
        let (r, evictions) = c.reserve(512).unwrap();
        assert_eq!(evictions, 3);
        assert_eq!(r.offset, 0);
        assert!(c.hit(key(2), 256).unwrap().is_some());
        c.commit(r, key(4)).unwrap();
        invariant(&c);
    }
    #[test]
    fn failed_fill_never_commits_and_retry_can_reuse_hole() {
        let mut c = RangeLru::new(512).unwrap();
        fill(&mut c, 0, 512);
        let (_, evictions) = c.reserve(512).unwrap();
        assert_eq!(evictions, 1);
        assert_eq!(c.hit(key(0), 512).unwrap(), None);
        assert_eq!(c.hit(key(1), 512).unwrap(), None);
        assert_eq!(fill(&mut c, 1, 512), (0, 0));
        invariant(&c);
    }
    #[test]
    fn bounds_duplicates_clock_and_changed_sizes_fail_closed() {
        let mut c = RangeLru::new(512).unwrap();
        assert!(c.reserve(0).is_err());
        assert!(c.reserve(513).is_err());
        assert!(c.reserve(usize::MAX).is_err());
        fill(&mut c, 0, 1);
        assert!(c.hit(key(0), 2).is_err());
        let (r, _) = c.reserve(1).unwrap();
        assert!(c.commit(r, key(0)).is_err());
        c.clock = u64::MAX;
        assert!(c.hit(key(0), 1).is_err());
        invariant(&c);
    }
    #[test]
    fn deterministic_mixed_sizes_never_overlap_or_move_hits() {
        let mut c = RangeLru::new(32 * 256).unwrap();
        let mut state = 7u64;
        for _ in 0..10000 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let k = (state >> 32) as usize % 37;
            let bytes = (1 + k % 7) * 121;
            if let Some(at) = c.hit(key(k), bytes).unwrap() {
                assert_eq!(c.hit(key(k), bytes).unwrap(), Some(at));
            } else {
                fill(&mut c, k, bytes);
            }
            invariant(&c);
        }
    }
    #[test]
    fn physical_layer_is_part_of_key_and_metadata_is_reserved() {
        let mut c = RangeLru::new(512).unwrap();
        fill(&mut c, 0, 256);
        assert_eq!(
            c.hit(
                Key {
                    layer: 7,
                    expert: 0
                },
                256
            )
            .unwrap(),
            None
        );
        assert!(
            std::mem::size_of::<Entry>() * MAX_ENTRIES + std::mem::size_of::<RangeLru>() <= 262144
        );
    }
}
