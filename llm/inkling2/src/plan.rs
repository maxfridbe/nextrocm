//! Logical forward visits and independent cache ownership, following the pinned
//! Inkling2LoopTextModel.forward. A visit index is NOT a cache or weight index.
use crate::{config::MiniConfig, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attention {
    Sliding,
    Full,
}
impl Attention {
    /// Dense unpadded token mask; padding/packed-sequence boundaries are separate.
    pub fn allows(self, query: usize, key: usize) -> bool {
        key <= query && (self == Self::Full || query - key < 1024)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visit {
    pub ordinal: usize,
    pub physical_layer: usize,
    pub cache_slot: usize,
    /// Zero for prelude/coda; core pass numbers are one through three.
    pub core_pass: usize,
    pub attention: Attention,
    /// Save the INPUT hidden state after computing this layer, as upstream does.
    pub append_input_residual_block: bool,
    /// Four messages (Q,K,V,MLP) read from the previous pass's two block values.
    pub previous_pass_messages: bool,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    Layer(Visit),
    SavePrelude,
    /// Freeze previous core block+partial state, then restore both saved prelude
    /// hidden and block states. Never feed the previous pass output directly.
    ResetToPrelude {
        pass: usize,
        message_values: usize,
    },
    FinalResidualMixAndNorm,
}
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    operations: Vec<Operation>,
    visits: Vec<Visit>,
    context: usize,
}
impl ExecutionPlan {
    pub fn new(c: &MiniConfig, context: usize) -> Result<Self> {
        if context == 0 || context > c.context_limit() {
            return Err("context outside checkpoint bounds".into());
        }
        let mut operations = Vec::new();
        let mut visits = Vec::new();
        let add = |physical_layer,
                   cache_slot,
                   core_pass,
                   operations: &mut Vec<Operation>,
                   visits: &mut Vec<Visit>| {
            let v = Visit {
                ordinal: visits.len(),
                physical_layer,
                cache_slot,
                core_pass,
                attention: if (physical_layer + 1) % 6 == 0 {
                    Attention::Full
                } else {
                    Attention::Sliding
                },
                append_input_residual_block: physical_layer % 6 == 0,
                previous_pass_messages: core_pass > 1,
            };
            visits.push(v);
            operations.push(Operation::Layer(v));
        };
        for l in 0..6 {
            add(l, l, 0, &mut operations, &mut visits)
        }
        operations.push(Operation::SavePrelude);
        for pass in 1..=3 {
            if pass > 1 {
                operations.push(Operation::ResetToPrelude {
                    pass,
                    message_values: 2,
                })
            }
            for l in 6..18 {
                let cache = if pass == 1 {
                    l
                } else {
                    24 + (pass - 2) * 12 + l - 6
                };
                add(l, cache, pass, &mut operations, &mut visits);
            }
        }
        for l in 18..24 {
            add(l, l, 0, &mut operations, &mut visits)
        }
        operations.push(Operation::FinalResidualMixAndNorm);
        Ok(Self {
            operations,
            visits,
            context,
        })
    }
    pub fn operations(&self) -> &[Operation] {
        &self.operations
    }
    pub fn visits(&self) -> &[Visit] {
        &self.visits
    }
    pub fn context(&self) -> usize {
        self.context
    }
    /// K+V storage only, using chosen scalar width. This excludes convolution
    /// state, all weights/activations/staging and driver margin: NOT admission.
    pub fn kv_storage_bytes(&self, bytes_per_scalar: usize) -> Result<u64> {
        if ![2, 4].contains(&bytes_per_scalar) {
            return Err("KV scalar must be two or four bytes".into());
        }
        self.visits.iter().try_fold(0u64, |sum, v| {
            let tokens = if v.attention == Attention::Sliding {
                self.context.min(1024)
            } else {
                self.context
            };
            let bytes = (tokens as u64)
                .checked_mul(2 * 8 * 128)
                .and_then(|n| n.checked_mul(bytes_per_scalar as u64))
                .ok_or("KV size overflow")?;
            sum.checked_add(bytes)
                .ok_or_else(|| "KV sum overflow".into())
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> MiniConfig {
        MiniConfig::from_official_json(include_bytes!("../tests/mini-config.json")).unwrap()
    }
    #[test]
    fn schedule_separates_weight_cache_and_visit_indices() {
        let p = ExecutionPlan::new(&config(), 4096).unwrap();
        assert_eq!(p.visits.len(), 48);
        let expected: Vec<_> = (0..18).chain(6..18).chain(6..18).chain(18..24).collect();
        assert_eq!(
            p.visits
                .iter()
                .map(|v| v.physical_layer)
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(p.visits[18].cache_slot, 24);
        assert_eq!(p.visits[30].cache_slot, 36);
        assert_eq!(p.visits[42].cache_slot, 18);
        let mut slots: Vec<_> = p.visits.iter().map(|v| v.cache_slot).collect();
        slots.sort();
        assert_eq!(slots, (0..48).collect::<Vec<_>>());
        assert_eq!(
            p.visits.iter().filter(|v| v.previous_pass_messages).count(),
            24
        );
        assert_eq!(
            p.operations
                .iter()
                .filter(|o| matches!(
                    o,
                    Operation::ResetToPrelude {
                        message_values: 2,
                        ..
                    }
                ))
                .count(),
            2
        );
    }
    #[test]
    fn masks_and_independent_cache_capacities() {
        assert!(Attention::Sliding.allows(1024, 1));
        assert!(!Attention::Sliding.allows(1024, 0));
        assert!(!Attention::Full.allows(2, 3));
        assert!(Attention::Full.allows(1024, 0));
        let p = ExecutionPlan::new(&config(), 4096).unwrap();
        assert_eq!(
            p.visits
                .iter()
                .filter(|v| v.attention == Attention::Full)
                .count(),
            8
        );
        assert_eq!(
            p.kv_storage_bytes(4).unwrap(),
            (40 * 1024 + 8 * 4096) * 2 * 8 * 128 * 4
        );
        assert!(p.kv_storage_bytes(8).is_err());
        assert!(ExecutionPlan::new(&config(), 0).is_err());
        assert!(ExecutionPlan::new(&config(), 1048577).is_err());
    }
}
