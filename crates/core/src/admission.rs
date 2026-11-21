//! Conservative device admission policy. Unmeasured models remain disabled.
pub const LIMIT_BYTES: u64 = 8_000_000_000;
#[derive(Debug, Clone, Copy)]
pub struct Peak {
    pub weights: u64,
    pub cache: u64,
    pub activations: u64,
    pub scratch: u64,
    pub runtime: u64,
}
impl Peak {
    pub fn total(self) -> Option<u64> {
        [
            self.weights,
            self.cache,
            self.activations,
            self.scratch,
            self.runtime,
        ]
        .into_iter()
        .try_fold(0u64, u64::checked_add)
    }
}
pub fn admit(measured: Option<Peak>, free: u64, reserve: u64) -> Result<u64, &'static str> {
    let peak = measured
        .ok_or("model has no measured ROCm peak")?
        .total()
        .ok_or("peak overflow")?;
    if peak == 0 {
        return Err("peak must be positive");
    }
    if peak >= LIMIT_BYTES {
        return Err("model peak must be strictly below 8 GB");
    }
    if peak > free.saturating_sub(reserve) {
        return Err("insufficient free device memory after reserve");
    }
    Ok(peak)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn peak(weights: u64) -> Peak {
        Peak {
            weights,
            cache: 0,
            activations: 0,
            scratch: 0,
            runtime: 0,
        }
    }
    #[test]
    fn strict_limit_and_reserve() {
        assert!(admit(Some(peak(LIMIT_BYTES)), u64::MAX, 0).is_err());
        assert_eq!(
            admit(Some(peak(LIMIT_BYTES - 1)), LIMIT_BYTES, 1),
            Ok(LIMIT_BYTES - 1)
        );
        assert!(admit(Some(peak(10)), 12, 3).is_err());
        assert!(admit(Some(peak(1)), 0, 10).is_err());
    }
    #[test]
    fn unknown_and_overflow_are_rejected() {
        assert!(admit(None, u64::MAX, 0).is_err());
        assert!(admit(
            Some(Peak {
                cache: 1,
                ..peak(u64::MAX)
            }),
            u64::MAX,
            0
        )
        .is_err());
        assert!(admit(Some(peak(0)), u64::MAX, 0).is_err());
    }
    #[test]
    fn all_pipeline_memory_counts() {
        let p = Peak {
            weights: 4_000_000_000,
            cache: 1_000_000_000,
            activations: 1_000_000_000,
            scratch: 1_000_000_000,
            runtime: 1_000_000_000,
        };
        assert!(admit(Some(p), u64::MAX, 0).is_err());
    }
}
