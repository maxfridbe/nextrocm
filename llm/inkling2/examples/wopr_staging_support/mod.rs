use nextrocm_llm_inkling2::expert_cache::{ExpertMode, ExpertStats};
use serde_json::{json, Value};
pub fn parse(name: &str, bytes: &str) -> Result<ExpertMode, Box<dyn std::error::Error>> {
    let budget: u64 = bytes.parse()?;
    match (name, budget) {
        ("stream", 0) => Ok(ExpertMode::Stream),
        ("pair", 0) => Ok(ExpertMode::BatchPair),
        ("cache", 268_435_456 | 536_870_912) => Ok(ExpertMode::DeviceLru {
            budget_bytes: budget,
        }),
        _ => Err(
            "mode must be stream/pair with bytes0 or cache with exactly 256MiB or 512MiB".into(),
        ),
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
