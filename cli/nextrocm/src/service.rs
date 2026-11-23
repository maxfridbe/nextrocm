use nextrocm_models::Config;
use std::path::PathBuf;
pub(crate) fn need_dist(cfg: &Config, file: &str) -> Result<PathBuf, String> {
    let p = cfg.dist.join(file);
    let inside = file
        .strip_prefix("libnextrocm-")
        .and_then(|k| k.strip_suffix(".so"))
        .is_some_and(nextrocm_sys::has_embedded);
    if p.exists()
        || inside
        || cfg.dist.join("libnextrocm-runtime.so").exists()
        || nextrocm_sys::has_embedded("runtime")
    {
        Ok(p)
    } else {
        Err(format!("{}: not built yet (./build.sh)", p.display()))
    }
}

/// A one-of entry's card (`gpus: '0|1'`): the one the front door chose for it (its chat_room, handed over beside the
/// registry), else the first of its cards with the room for it now (its model card's vram_gib, else its files and a
/// tenth), else the one with the most free
pub(crate) fn one_of_gpu(cfg: &Config, m: &serde_json::Value, id: &str) -> Result<usize, String> {
    let mut cands = nextrocm_serve::home::registry::gpu_list(m["gpus"].as_str().unwrap_or(""));
    if cands.is_empty() {
        return Err(format!("{id}: one-of GPUs names no valid card"));
    }
    let reservation = nextrocm_serve::gpulock::path(cfg.get("NS_GPU_LOCK"))
        .and_then(|p| nextrocm_serve::gpulock::taken(&p).map(|t| (p, t)));
    if let Some((p, t)) = &reservation {
        cands.retain(|g| !t.takes(*g));
        if cands.is_empty() {
            return Err(nextrocm_serve::gpulock::message(p, t));
        }
    }
    if let Some(g) =
        nextrocm_serve::home::registry::chat_gpu_in(&nextrocm_models::registry::registry(cfg), id)
            .filter(|g| cands.contains(g))
    {
        return Ok(g);
    }
    let need = m["vram_gib"].as_f64().map_or(
        (nextrocm_serve::home::registry::paths(m)
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|x| x.len())
            .sum::<u64>()
            / 10
            * 11)
            >> 20,
        |g| (g * 1024.0) as u64,
    );
    let pcis = nextrocm_serve::gpustat::find_cards();
    let free = |g: usize| {
        pcis.get(g).map_or(0, |p| {
            let r = nextrocm_serve::gpustat::Card::open(p, None)
                .sample(std::time::Duration::from_secs(1))
                .0;
            r["vram_total_mb"]
                .as_u64()
                .unwrap_or(0)
                .saturating_sub(r["vram_used_mb"].as_u64().unwrap_or(u64::MAX))
        })
    };
    Ok(first_with_room(&cands, &free, need))
}

/// Of `cands` (in order of preference), the first with `need` MiB free, else the one with the most
fn first_with_room(cands: &[usize], free: &dyn Fn(usize) -> u64, need: u64) -> usize {
    let free: Vec<(usize, u64)> = cands.iter().map(|g| (*g, free(*g))).collect();
    free.iter()
        .find(|(_, f)| *f >= need)
        .or_else(|| free.iter().max_by_key(|(g, f)| (*f, std::cmp::Reverse(*g))))
        .map_or(0, |(g, _)| *g)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_preferred_fit_else_freest() {
        let f = |g: usize| [20000, 30000][g];
        assert_eq!(first_with_room(&[0, 1], &f, 18000), 0);
        assert_eq!(first_with_room(&[0, 1], &f, 25000), 1);
        assert_eq!(first_with_room(&[1, 0], &f, 18000), 1);
        assert_eq!(first_with_room(&[0, 1], &f, 40000), 1);
    }
}
