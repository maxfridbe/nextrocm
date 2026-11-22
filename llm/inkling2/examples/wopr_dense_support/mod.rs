use nextrocm_llm_inkling2::expert_cache::{ExpertMode, ExpertStats};
use serde_json::{json, Value};
pub fn parse(name: &str, bytes: &str) -> Result<ExpertMode, Box<dyn std::error::Error>> {
    let budget: u64 = bytes.parse()?;
    match (name, budget) {
        ("stream", 0) => Ok(ExpertMode::Stream),
        ("pair", 0) => Ok(ExpertMode::BatchPair),
        ("segmented", 939_524_096) => Ok(ExpertMode::SegmentedLru {
            budget_bytes: budget,
        }),
        ("cache", 268_435_456 | 536_870_912 | 939_524_096) => Ok(ExpertMode::DeviceLru {
            budget_bytes: budget,
        }),
        _ => Err("mode must be stream/pair bytes0, cache256/512/896MiB, or segmented896MiB".into()),
    }
}
pub fn stats(s: ExpertStats) -> Value {
    json!({"hits":s.hits,"misses":s.misses,"evictions":s.evictions,"read_bytes":s.read_bytes,"uploaded_bytes":s.uploaded_bytes,"upload_calls":s.upload_calls})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_modes() {
        assert_eq!(parse("stream", "0").unwrap(), ExpertMode::Stream);
        assert_eq!(parse("pair", "0").unwrap(), ExpertMode::BatchPair);
        assert!(matches!(
            parse("cache", "268435456").unwrap(),
            ExpertMode::DeviceLru { .. }
        ));
        for (m, b) in [
            ("stream", "1"),
            ("pair", "1"),
            ("cache", "0"),
            ("cache", "1073741825"),
            ("cache", "1073741824"),
            ("cache", "1"),
            ("other", "0"),
            ("cache", "-1"),
        ] {
            assert!(parse(m, b).is_err());
        }
    }
}

// Cumulative diagnostics; failed runs do not establish successful group identities.
pub fn batch_stats(s: nextrocm_llm_inkling2::expert_batch::BatchStats) -> serde_json::Value {
    serde_json::json!({"visits":s.visits,"logical_selections":s.logical_selections,"groups":s.groups,"group_sizes":s.group_sizes,"gu_calls":s.gu_calls,"down_calls":s.down_calls})
}

pub fn validate_batch(
    s: nextrocm_llm_inkling2::expert_batch::BatchStats,
    cache: nextrocm_llm_inkling2::expert_cache::ExpertStats,
    tokens: usize,
    chunk: usize,
    enabled: bool,
    scalar_extra: usize,
) -> Result<(), String> {
    let mut grouped_rows = 0u64;
    let mut grouped_visits = 0u64;
    if enabled && chunk > 1 {
        for start in (0..tokens).step_by(chunk) {
            let n = (tokens - start).min(chunk);
            if n > 1 {
                grouped_rows += n as u64;
                grouped_visits += 48;
            }
        }
    }
    let groups: u64 = s.group_sizes.iter().sum();
    let selections: u64 = s
        .group_sizes
        .iter()
        .enumerate()
        .map(|(i, n)| (i as u64 + 1) * n)
        .sum();
    let scalar_rows = tokens as u64 - grouped_rows + scalar_extra as u64;
    if s.visits != grouped_visits
        || s.logical_selections != grouped_rows * 384
        || selections != s.logical_selections
        || groups != s.groups
        || s.gu_calls != groups
        || s.down_calls != groups
        || s.group_sizes
            .iter()
            .enumerate()
            .any(|(i, &n)| i + 1 > chunk && n != 0)
        || cache.hits + cache.misses != groups + scalar_rows * 384
        || cache.upload_calls != cache.misses
        || cache.read_bytes != cache.uploaded_bytes
    {
        return Err("expert group/callback/selection accounting differs".into());
    }
    Ok(())
}

#[cfg(test)]
mod batch_accounting_tests {
    use super::validate_batch;
    use nextrocm_llm_inkling2::{expert_batch::BatchStats, expert_cache::ExpertStats};
    #[test]
    fn scalar_and_ragged_group_accounting() {
        let mut cache = ExpertStats::default();
        cache.hits = 384 * 50;
        assert!(validate_batch(BatchStats::default(), cache, 18, 1, false, 32).is_ok());
        // Four full chunks of four plus a two-row tail, all rows select the same eight experts.
        let batch = BatchStats {
            visits: 48 * 5,
            logical_selections: 18 * 384,
            groups: 48 * 5 * 8,
            group_sizes: [0, 48 * 8, 0, 48 * 4 * 8],
            gu_calls: 48 * 5 * 8,
            down_calls: 48 * 5 * 8,
        };
        cache.hits = batch.groups + 384 * 32;
        assert!(validate_batch(batch, cache, 18, 4, true, 32).is_ok());
        cache.hits += 1;
        assert!(validate_batch(batch, cache, 18, 4, true, 32).is_err());
    }
    #[test]
    fn singleton_tail_remains_scalar() {
        let batch = BatchStats {
            visits: 48,
            logical_selections: 4 * 384,
            groups: 48 * 8,
            group_sizes: [0, 0, 0, 48 * 8],
            gu_calls: 48 * 8,
            down_calls: 48 * 8,
        };
        let mut cache = ExpertStats::default();
        cache.hits = batch.groups + 384;
        assert!(validate_batch(batch, cache, 5, 4, true, 0).is_ok());
        let mut invalid = batch;
        invalid.down_calls -= 1;
        assert!(validate_batch(invalid, cache, 5, 4, true, 0).is_err());
    }
}

pub fn dense_stats(family_calls: [u64; 7], logical_vectors: [u64; 7]) -> serde_json::Value {
    serde_json::json!({"family_order":["Q","K","V","R","gate","O","output"],"family_calls":family_calls,"logical_vectors":logical_vectors})
}
pub fn validate_dense(
    family_calls: [u64; 7],
    logical_vectors: [u64; 7],
    tokens: usize,
    chunk: usize,
    enabled: bool,
) -> Result<(), String> {
    if ![1, 2, 4].contains(&chunk) {
        return Err("invalid dense diagnostic chunk".into());
    }
    let mut groups = 0u64;
    let mut rows = 0u64;
    if enabled {
        for start in (0..tokens).step_by(chunk) {
            let n = (tokens - start).min(chunk);
            if n > 1 {
                groups += 1;
                rows += n as u64;
            }
        }
    }
    let mut expected_calls = [groups * 48; 7];
    expected_calls[6] = groups;
    let mut expected_vectors = [rows * 48; 7];
    expected_vectors[6] = rows;
    if family_calls != expected_calls || logical_vectors != expected_vectors {
        return Err("dense family calls/logical vectors differ".into());
    }
    Ok(())
}
#[cfg(test)]
mod dense_accounting_tests {
    use super::validate_dense;
    #[test]
    fn full_and_ragged_and_disabled() {
        assert!(validate_dense(
            [240, 240, 240, 240, 240, 240, 5],
            [864, 864, 864, 864, 864, 864, 18],
            18,
            4,
            true
        )
        .is_ok());
        assert!(validate_dense(
            [48, 48, 48, 48, 48, 48, 1],
            [192, 192, 192, 192, 192, 192, 4],
            5,
            4,
            true
        )
        .is_ok());
        assert!(validate_dense([0; 7], [0; 7], 18, 4, false).is_ok());
        assert!(validate_dense(
            [240, 240, 240, 240, 240, 240, 4],
            [864, 864, 864, 864, 864, 864, 18],
            18,
            4,
            true
        )
        .is_err());
        assert!(validate_dense([0; 7], [0; 7], 18, 4, true).is_err());
    }
}
