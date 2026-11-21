use nextrocm_models::Config;
use std::path::PathBuf;
pub(crate) fn need_dist(cfg: &Config, file: &str) -> Result<PathBuf, String> {
    let p = cfg.dist.join(file);
    let inside = file
        .strip_prefix("libnextrocm-")
        .and_then(|k| k.strip_suffix(".so"))
        .is_some_and(nextrocm_sys::has_embedded);
    if p.exists() || inside {
        Ok(p)
    } else {
        Err(format!("{}: not built yet (./build.sh)", p.display()))
    }
}
