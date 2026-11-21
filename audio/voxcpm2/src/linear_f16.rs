//! Mirrors the private F16 linear header's model dispatch bounds.
pub(super) fn supported(m: usize, k: usize, n: usize) -> bool {
    if !(9..=64).contains(&m) || k == 0 || n == 0 {
        return false;
    }
    let (m, k, n) = (m as u128, k as u128, n as u128);
    const LIMIT: u128 = 1_874_999_999;
    if k > LIMIT / m || n > LIMIT / m || n > LIMIT / k {
        return false;
    }
    2 * (m * k + n * k) + 4 * (m * n + n) < 7_500_000_000 && n.div_ceil(64) <= i32::MAX as u128
}

#[cfg(test)]
mod tests {
    use super::supported;
    #[test]
    fn selection_bounds_and_model_shapes() {
        for (k, n) in [(1024, 2560), (2048, 1024), (1024, 8192), (4096, 1024)] {
            assert!(supported(22, k, n));
        }
        assert!(supported(9, 1, 1));
        assert!(supported(64, 33, 65));
        for (m, k, n) in [
            (8, 1024, 1024),
            (65, 1024, 1024),
            (22, 0, 1024),
            (22, 1024, 0),
            (22, usize::MAX, 1024),
            (22, 1024, usize::MAX),
        ] {
            assert!(!supported(m, k, n));
        }
    }
}
