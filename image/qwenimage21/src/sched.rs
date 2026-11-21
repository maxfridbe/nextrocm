//! Official Turbo's stored sigma nodes, from Qwen revision
//! fc13ba1986fcb905854dc8de2f52eee395f49b5b/model_index.json.
//! No resolution-dependent shift or terminal stretch is applied.
//! This private module is not registered with the production engine yet.

pub const TURBO_SIGMAS: [f32; 8] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
];

/// Flow matching's fixed nodes plus terminal zero. This does not resample a
/// requested step count: the checkpoint's saved schedule is authoritative.
pub fn turbo_sigmas() -> [f32; 9] {
    let mut result = [0.0; 9];
    result[..8].copy_from_slice(&TURBO_SIGMAS);
    result
}

/// Preserve the reference's discrete step-tail selection. Strength-to-step
/// conversion belongs to the caller and is not invented by this helper.
pub fn tail(sigmas: &[f32], steps: usize) -> &[f32] {
    &sigmas[sigmas.len().saturating_sub(steps.saturating_add(1))..]
}

/// Preserve the reference's fixed-schedule editing path: start at the requested
/// noise level, then consume stored levels strictly below it.
pub fn below(sigmas: &[f32], strength: f32) -> Result<Vec<f32>, &'static str> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        return Err("strength must be finite and between zero and one");
    }
    Ok(std::iter::once(strength)
        .chain(sigmas.iter().copied().filter(|s| *s < strength))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_nodes_are_unshifted() {
        let s = turbo_sigmas();
        assert_eq!(s.len(), 9);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[7], 0.414568);
        assert_eq!(s[8], 0.0);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }

    #[test]
    fn reference_tail_and_fixed_strength() {
        let s = turbo_sigmas();
        assert_eq!(tail(&s, 8), &s);
        assert_eq!(tail(&s, usize::MAX), &s);
        assert_eq!(tail(&s, 0), &[0.0]);
        assert_eq!(below(&s, 0.5).unwrap(), vec![0.5, 0.414568, 0.0]);
        assert_eq!(below(&s, 0.0).unwrap(), vec![0.0]);
        for bad in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert!(below(&s, bad).is_err());
        }
    }
}
