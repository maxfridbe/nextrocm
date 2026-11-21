//! Persistent ownership for chronological single-token execution. Physical
//! weights repeat, but every visit owns independent KV and convolution state.
use crate::{
    plan::{Attention, ExecutionPlan, Operation},
    Result,
};
#[derive(Clone, Debug)]
pub struct CacheSlot {
    pub visit: usize,
    pub physical_layer: usize,
    pub capacity: usize,
    /// F32 element offsets within one KV allocation; layout [position,8,128].
    pub key_offset: usize,
    pub value_offset: usize,
    /// F32 history [channel,oldest..newest] for K,V,attention-output,MLP-output.
    pub history_offset: usize,
    sliding: bool,
}
impl CacheSlot {
    pub fn write_row(&self, position: usize, context: usize) -> Result<usize> {
        if position >= context || (!self.sliding && position >= self.capacity) {
            return Err("KV position exceeds session context".into());
        }
        Ok(position % self.capacity)
    }
    /// Absolute positions present after this token's K/V have been written.
    pub fn visible_positions(
        &self,
        position: usize,
        context: usize,
    ) -> Result<std::ops::RangeInclusive<usize>> {
        self.write_row(position, context)?;
        Ok(position.saturating_add(1).saturating_sub(self.capacity)..=position)
    }
}
#[derive(Clone, Debug)]
pub struct StateLayout {
    pub slots: Vec<CacheSlot>,
    pub kv_bytes: usize,
    pub history_bytes: usize,
}
impl StateLayout {
    pub fn new(plan: &ExecutionPlan) -> Result<Self> {
        let mut slots = Vec::with_capacity(48);
        let mut kv = 0usize;
        let mut history = 0usize;
        for v in plan.visits() {
            let sliding = v.attention == Attention::Sliding;
            let capacity = if sliding {
                plan.context().min(1024)
            } else {
                plan.context()
            };
            let elements = capacity.checked_mul(8 * 128).ok_or("KV extent overflow")?;
            let key_offset = kv;
            kv = kv.checked_add(elements).ok_or("KV offset overflow")?;
            let value_offset = kv;
            kv = kv.checked_add(elements).ok_or("KV offset overflow")?;
            let history_offset = history;
            history = history
                .checked_add(3 * (2 * 8 * 128 + 2 * 2304))
                .ok_or("history overflow")?;
            slots.push(CacheSlot {
                visit: v.ordinal,
                physical_layer: v.physical_layer,
                capacity,
                key_offset,
                value_offset,
                history_offset,
                sliding,
            });
        }
        let kv_bytes = kv.checked_mul(4).ok_or("KV byte overflow")?;
        if kv_bytes as u64 != plan.kv_storage_bytes(4)? {
            return Err("session/plan KV mismatch".into());
        }
        Ok(Self {
            slots,
            kv_bytes,
            history_bytes: history.checked_mul(4).ok_or("history byte overflow")?,
        })
    }
}
/// An operation is advanced only after its GPU work has succeeded. Any partial
/// token failure poisons the entire session: KV/history may already be changed.
/// Reuse requires a new session with zeroed state, never just resetting a cursor.
pub struct Cursor {
    plan: ExecutionPlan,
    position: usize,
    operation: usize,
    active: bool,
    poisoned: bool,
}
impl Cursor {
    pub fn new(plan: ExecutionPlan) -> Self {
        Self {
            plan,
            position: 0,
            operation: 0,
            active: false,
            poisoned: false,
        }
    }
    pub(crate) fn settled_position(&self) -> Result<usize> {
        if self.active || self.poisoned { return Err("session is active or poisoned".into()); }
        Ok(self.position)
    }
    /// Only after all KV and convolution state has been reset/restored successfully.
    pub(crate) fn restore_position(&mut self, position: usize) -> Result<()> {
        if position > self.plan.context() { return Err("restored position exceeds context".into()); }
        self.position = position; self.operation = 0; self.active = false; self.poisoned = false;
        Ok(())
    }
    pub fn begin(&mut self) -> Result<usize> {
        if self.poisoned || self.active || self.position >= self.plan.context() {
            return Err("session unavailable/active/context exhausted".into());
        }
        self.operation = 0;
        self.active = true;
        Ok(self.position)
    }
    pub fn current(&self) -> Result<Option<&Operation>> {
        if self.poisoned || !self.active {
            return Err("no active healthy token".into());
        }
        Ok(self.plan.operations().get(self.operation))
    }
    pub fn complete_operation(&mut self) -> Result<()> {
        if self.current()?.is_none() {
            return Err("all token operations already complete".into());
        }
        self.operation += 1;
        Ok(())
    }
    pub fn commit(&mut self) -> Result<()> {
        if self.current()?.is_some() {
            return Err("cannot commit incomplete token".into());
        }
        self.active = false;
        self.position += 1;
        Ok(())
    }
    pub fn abort(&mut self) {
        self.poisoned = true;
        self.active = false;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MiniConfig;
    fn plan(context: usize) -> ExecutionPlan {
        ExecutionPlan::new(
            &MiniConfig::from_official_json(include_bytes!("../tests/mini-config.json")).unwrap(),
            context,
        )
        .unwrap()
    }
    #[test]
    fn lifecycle_position_is_settled_only_after_valid_restore() {
        let mut c = Cursor::new(plan(2));
        assert_eq!(c.settled_position().unwrap(), 0);
        c.begin().unwrap();
        assert!(c.settled_position().is_err());
        c.abort();
        assert!(c.restore_position(3).is_err());
        assert!(c.settled_position().is_err());
        c.restore_position(1).unwrap();
        assert_eq!(c.begin().unwrap(), 1);
        c.abort();
        c.restore_position(2).unwrap();
        assert_eq!(c.settled_position().unwrap(), 2);
        assert!(c.begin().is_err());
    }
    #[test]
    fn cache_ownership_and_window() {
        let p = plan(4096);
        let l = StateLayout::new(&p).unwrap();
        assert_eq!(l.slots.len(), 48);
        assert_eq!(l.history_bytes, 48 * 3 * (2 * 8 * 128 + 2 * 2304) * 4);
        for pair in l.slots.windows(2) {
            assert!(pair[0].value_offset + pair[0].capacity * 1024 <= pair[1].key_offset);
            assert!(pair[0].history_offset < pair[1].history_offset);
        }
        assert_eq!(l.slots[6].physical_layer, l.slots[18].physical_layer);
        assert_ne!(l.slots[6].key_offset, l.slots[18].key_offset);
        assert_eq!(l.slots[0].write_row(1024, 4096).unwrap(), 0);
        assert_eq!(l.slots[0].visible_positions(1024, 4096).unwrap(), 1..=1024);
        assert_eq!(l.slots[5].visible_positions(1024, 4096).unwrap(), 0..=1024);
    }
    #[test]
    fn partial_failure_requires_new_session() {
        let mut c = Cursor::new(plan(2));
        assert_eq!(c.begin().unwrap(), 0);
        assert!(c.commit().is_err());
        while c.current().unwrap().is_some() {
            c.complete_operation().unwrap();
        }
        c.commit().unwrap();
        assert_eq!(c.begin().unwrap(), 1);
        c.complete_operation().unwrap();
        c.abort();
        assert!(c.begin().is_err());
        assert!(c.commit().is_err());
    }
}
