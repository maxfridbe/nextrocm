//! Actual selected routes across both prompt processing and token generation.
use super::{error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct RouteProfile {
    counts: Option<Vec<AtomicU64>>,
    experts: usize,
    model_bytes: u64,
    model_fingerprint: u64,
}

impl RouteProfile {
    pub fn report(&self) -> Vec<String> {
        if self.counts.is_some() {
            vec![
                "expert route profile: actual selected routes, mixed PP/TG (not decode-only)"
                    .into(),
            ]
        } else {
            Vec::new()
        }
    }
    pub fn new(
        enabled: bool,
        layers: usize,
        experts: usize,
        model_bytes: u64,
        model_fingerprint: u64,
    ) -> Result<Self> {
        let counts = if enabled {
            let n = layers
                .checked_mul(experts)
                .ok_or_else(|| error("route profile size overflow"))?;
            let mut v = Vec::new();
            v.try_reserve_exact(n)
                .map_err(|e| error(format!("route profile allocation: {e}")))?;
            v.resize_with(n, || AtomicU64::new(0));
            Some(v)
        } else {
            None
        };
        Ok(Self {
            counts,
            experts,
            model_bytes,
            model_fingerprint,
        })
    }

    pub fn record(&self, layer: usize, routes: &[(usize, f32)]) -> Result<()> {
        let Some(counts) = &self.counts else {
            return Ok(());
        };
        for &(expert, _) in routes {
            let i = layer
                .checked_mul(self.experts)
                .and_then(|i| i.checked_add(expert));
            let counter = i
                .filter(|_| expert < self.experts)
                .and_then(|i| counts.get(i))
                .ok_or_else(|| error("route profile index outside geometry"))?;
            let _ = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_add(1))
            });
        }
        Ok(())
    }

    pub fn save(&self, path: &std::path::Path) -> std::io::Result<usize> {
        let Some(counts) = &self.counts else {
            return Ok(0);
        };
        let mut rows: Vec<_> = counts
            .iter()
            .enumerate()
            .filter_map(|(i, n)| {
                let n = n.load(Ordering::Relaxed);
                (n != 0).then_some((i, n))
            })
            .collect();
        rows.sort_by_key(|&(i, n)| (std::cmp::Reverse(n), i));
        let experts: Vec<_> = rows
            .iter()
            .map(|&(i, count)| {
                serde_json::json!({
                    "layer": i / self.experts, "expert": i % self.experts, "count": count
                })
            })
            .collect();
        let data = serde_json::json!({ "model_bytes": self.model_bytes,
            "model_fingerprint": self.model_fingerprint, "experts": experts });
        let mut file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(&mut file, &data)?;
        Ok(rows.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn counts_mixed_routes_saturates_and_sorts_ties() {
        let p = RouteProfile::new(true, 2, 3, 987, u64::MAX).unwrap();
        p.record(1, &[(2, 0.1), (0, 0.9)]).unwrap();
        p.record(0, &[(2, 1.0)]).unwrap();
        p.record(1, &[(2, 1.0)]).unwrap();
        p.counts.as_ref().unwrap()[0].store(u64::MAX, Ordering::Relaxed);
        p.record(0, &[(0, 1.0)]).unwrap();
        assert!(p.record(2, &[(0, 1.0)]).is_err());
        assert!(p.record(0, &[(3, 1.0)]).is_err());
        let path = std::env::temp_dir().join(format!("qwen35-routes-{}.json", std::process::id()));
        assert_eq!(p.save(&path).unwrap(), 4);
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(v["model_fingerprint"].as_u64(), Some(u64::MAX));
        assert_eq!(v["model_bytes"], 987);
        assert_eq!(
            v["experts"],
            serde_json::json!([
                {"layer":0,"expert":0,"count":u64::MAX},
                {"layer":1,"expert":2,"count":2},
                {"layer":0,"expert":2,"count":1},
                {"layer":1,"expert":0,"count":1}
            ])
        );
    }
    #[test]
    fn disabled_allocates_nothing_and_does_not_write() {
        let p = RouteProfile::new(false, usize::MAX, usize::MAX, 0, 0).unwrap();
        assert!(p.counts.is_none());
        p.record(usize::MAX, &[(usize::MAX, 1.)]).unwrap();
        assert_eq!(
            p.save(std::path::Path::new("/no/such/profile.json"))
                .unwrap(),
            0
        );
        assert!(RouteProfile::new(true, usize::MAX, 2, 0, 0).is_err());
    }
}
