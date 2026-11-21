//! Dispatch geometry mirrors kernels/audio/voxcpm2/conv_up_tiled.hpp.
//! Unsupported geometry keeps the original convolution; launch errors propagate.

pub(super) fn supported(b: usize, ci: usize, l: usize, co: usize, k: usize, s: usize) -> bool {
    if !matches!(s, 2 | 5 | 6 | 8) || k != 2 * s {
        return false;
    }
    const LIMIT: usize = 1_875_000_000;
    let span = |dims: &[usize]| {
        dims.iter().try_fold(1usize, |n, &d| {
            n.checked_mul(d).filter(|&v| d > 0 && v < LIMIT)
        })
    };
    let (Some(nx), Some(nw), Some(ny)) =
        (span(&[b, ci, l]), span(&[ci, co, k]), span(&[b, co, l, s]))
    else {
        return false;
    };
    nx.checked_add(nw)
        .and_then(|v| v.checked_add(ny))
        .and_then(|v| v.checked_add(co))
        .is_some_and(|v| v < LIMIT)
        && l.div_ceil(64) <= i32::MAX as usize
        && b.checked_mul(co.div_ceil(16))
            .and_then(|v| v.checked_mul(s))
            .is_some_and(|v| v <= 65535)
}

#[cfg(test)]
mod tests {
    use super::supported;

    #[test]
    fn measured_decoder_shapes_and_tails_are_supported() {
        for (ci, l, co, s) in [
            (2048, 44, 1024, 8),
            (1024, 352, 512, 6),
            (512, 2112, 256, 5),
            (256, 10560, 128, 2),
            (128, 21120, 64, 2),
            (64, 42240, 32, 2),
        ] {
            assert!(supported(1, ci, l, co, 2 * s, s));
        }
        for s in [2, 5, 6, 8] {
            for l in [1, 2, 31, 32, 33, 63, 64, 65] {
                assert!(supported(2, 3, l, 17, 2 * s, s));
            }
        }
    }

    #[test]
    fn unsupported_overflow_and_grid_limits_fall_back() {
        assert!(!supported(1, 1, 1, 1, 6, 3));
        assert!(!supported(1, 1, 1, 1, 3, 2));
        assert!(!supported(0, 1, 1, 1, 4, 2));
        assert!(!supported(1, usize::MAX, 1, 1, 4, 2));
        assert!(!supported(1, 1, usize::MAX, 1, 4, 2));
        assert!(!supported(1, 1, 1, usize::MAX, 4, 2));
        assert!(!supported(1, 1, 1, 1, 4, usize::MAX));
        assert!(supported(32767, 1, 1, 1, 4, 2));
        assert!(!supported(32768, 1, 1, 1, 4, 2));
        assert!(!supported(1, 1, 624_999_999, 1, 4, 2));
    }
}
