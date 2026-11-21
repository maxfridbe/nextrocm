//! Bounded, optional advice for already selected expert windows; no inference work.
//!
//! Source basis: Strata 58ee3227720629ff2e41d640165ebe89c880dfef,
//! engine/src/core/expert_source.cpp:3105, expert-list role MADV_WILLNEED.
//! This adaptation advises file ranges with POSIX_FADV_WILLNEED instead of
//! mapping them. It adds a hard per-layer request budget; no guessed experts.
//! Default is disabled. Parent integration must explicitly opt in and call
//! begin_layer once before advising the selected gate/up/down windows.
use std::{fs::File, os::fd::AsRawFd};

pub const MAX_LAYER_BYTES: usize = 32 * 1024 * 1024;

/// Lifetime counters, retained across begin_layer calls. These describe advice,
/// not actual disk reads, residency or completion (all controlled by the OS).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Bytes accepted by successful advisory calls.
    pub advised_bytes: u64,
    /// Advisory syscalls attempted, including failed calls.
    pub calls: u64,
    /// Invalid requests, metadata failures and failed advisory calls.
    pub errors: u64,
}

#[derive(Debug, Default)]
pub struct Prefetch {
    max_bytes: usize,
    used_bytes: usize,
    stats: Stats,
}

impl Prefetch {
    /// Zero disables advice; larger budgets are clamped to 32 MiB per layer.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes: max_bytes.min(MAX_LAYER_BYTES),
            ..Self::default()
        }
    }

    /// Reset only the request budget. Does not wait for or cancel prior OS advice.
    pub fn begin_layer(&mut self) {
        self.used_bytes = 0;
    }

    pub fn enabled(&self) -> bool {
        self.max_bytes != 0
    }

    /// One layer's already selected windows. The caller filters RAM-mirror hits.
    /// Every call resets the byte budget, even for an empty list. Errors remain
    /// advisory counters; the return value is successfully advised bytes.
    pub fn submit(&mut self, windows: &[(&File, u64, usize)]) -> usize {
        self.begin_layer();
        windows
            .iter()
            .map(|&(file, offset, len)| self.advise(file, offset, len))
            .sum()
    }

    pub fn report(&self) -> Vec<String> {
        vec![format!(
            "qwen35 prefetch: enabled={} max_bytes_per_layer={} layer_requested_bytes={} advised_bytes={} calls={} errors={}",
            self.enabled(), self.max_bytes, self.used_bytes,
            self.stats.advised_bytes, self.stats.calls, self.stats.errors
        )]
    }

    #[cfg(test)]
    pub fn stats(&self) -> Stats {
        self.stats
    }

    /// Budget charged this layer, including failed syscalls; never exceeds cap.
    #[cfg(test)]
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Advise a prefix fitting the remaining budget, after validating the FULL
    /// requested window against EOF and off_t. Returns successfully advised bytes.
    /// Zero means disabled/exhausted/empty/invalid/failed, never an inference error.
    /// No zero-length syscall: POSIX interprets it as advice through EOF.
    /// Calls require exclusive access; caller retains immutable model files.
    pub fn advise(&mut self, file: &File, offset: u64, len: usize) -> usize {
        if self.max_bytes == 0 || self.used_bytes == self.max_bytes || len == 0 {
            return 0;
        }
        let valid = (|| {
            let len64 = u64::try_from(len).ok()?;
            let end = offset.checked_add(len64)?;
            let off = libc::off_t::try_from(offset).ok()?;
            libc::off_t::try_from(len64).ok()?;
            libc::off_t::try_from(end).ok()?;
            let metadata = file.metadata().ok()?;
            if !metadata.is_file() || end > metadata.len() {
                return None;
            }
            Some(off)
        })();
        let Some(off) = valid else {
            self.stats.errors = self.stats.errors.saturating_add(1);
            return 0;
        };
        let bytes = len.min(self.max_bytes - self.used_bytes);
        // Conversion is safe: the full length passed off_t validation above.
        let advice_len = bytes as libc::off_t;
        self.used_bytes += bytes;
        self.stats.calls = self.stats.calls.saturating_add(1);
        // SAFETY: live borrowed descriptor, checked nonnegative off_t range inside
        // the immutable regular file. Advisory only; no pointer or file mutation.
        let rc = unsafe {
            libc::posix_fadvise(file.as_raw_fd(), off, advice_len, libc::POSIX_FADV_WILLNEED)
        };
        if rc != 0 {
            // posix_fadvise returns its error code directly, not through errno.
            self.stats.errors = self.stats.errors.saturating_add(1);
            return 0;
        }
        self.stats.advised_bytes = self.stats.advised_bytes.saturating_add(bytes as u64);
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs::OpenOptions,
        sync::atomic::{AtomicU64, Ordering},
    };

    fn file(len: u64) -> File {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "nq35-prefetch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(path).unwrap();
        f.set_len(len).unwrap();
        f
    }

    #[test]
    fn disabled_and_empty_never_issue_unbounded_advice() {
        let f = file(4096);
        let mut p = Prefetch::default();
        assert_eq!(p.advise(&f, 0, 4096), 0);
        assert_eq!(p.stats(), Stats::default());
        let mut p = Prefetch::new(4096);
        assert_eq!(p.advise(&f, 0, 0), 0);
        assert_eq!(p.stats(), Stats::default());
    }

    #[test]
    fn submit_has_fresh_budget_and_nonfatal_invalid_windows() {
        let f = file(8192);
        let mut p = Prefetch::new(4096);
        assert!(p.enabled());
        assert!(!Prefetch::default().enabled());
        assert_eq!(p.submit(&[(&f, u64::MAX, 2), (&f, 0, 8192)]), 4096);
        assert_eq!(p.submit(&[(&f, 4096, 4096)]), 4096);
        assert_eq!(
            p.stats(),
            Stats {
                advised_bytes: 8192,
                calls: 2,
                errors: 1
            }
        );
        assert_eq!(p.submit(&[]), 0);
        assert_eq!(p.used_bytes(), 0);
        assert!(p.report()[0].contains("errors=1"));
    }

    #[test]
    fn cumulative_budget_prefix_and_reset() {
        let f = file(8192);
        let mut p = Prefetch::new(4096);
        assert_eq!(p.advise(&f, 0, 3072), 3072);
        assert_eq!(p.advise(&f, 3072, 3072), 1024);
        assert_eq!(p.advise(&f, 0, 4096), 0);
        assert_eq!(p.used_bytes(), 4096);
        assert_eq!(
            p.stats(),
            Stats {
                advised_bytes: 4096,
                calls: 2,
                errors: 0
            }
        );
        p.begin_layer();
        assert_eq!(p.used_bytes(), 0);
        assert_eq!(p.advise(&f, 4096, 4096), 4096);
        assert_eq!(p.stats().advised_bytes, 8192);
    }

    #[test]
    fn rejects_full_window_past_eof_even_when_prefix_fits() {
        let f = file(4096);
        let mut p = Prefetch::new(1024);
        for (off, len) in [
            (0, 4097),
            (4096, 1),
            (8192, 1),
            (u64::MAX, 2),
            (i64::MAX as u64, 1),
            (0, usize::MAX),
        ] {
            assert_eq!(p.advise(&f, off, len), 0);
        }
        assert_eq!(
            p.stats(),
            Stats {
                advised_bytes: 0,
                calls: 0,
                errors: 6
            }
        );
        assert_eq!(p.used_bytes(), 0);
        assert_eq!(p.advise(&f, 3072, 1024), 1024);
    }

    #[test]
    fn rejects_nonregular_file() {
        let f = File::open(std::env::temp_dir()).unwrap();
        let mut p = Prefetch::new(4096);
        assert_eq!(p.advise(&f, 0, 1), 0);
        assert_eq!(p.stats().errors, 1);
        assert_eq!(p.stats().calls, 0);
    }

    #[test]
    fn clamps_cap_and_never_changes_file_contents() {
        use std::os::unix::fs::FileExt;
        let f = file((MAX_LAYER_BYTES + 4096) as u64);
        f.write_all_at(b"packed weights", 0).unwrap();
        let mut p = Prefetch::new(usize::MAX);
        assert_eq!(p.advise(&f, 0, MAX_LAYER_BYTES + 4096), MAX_LAYER_BYTES);
        assert_eq!(p.advise(&f, 0, 1), 0);
        let mut bytes = [0; 14];
        f.read_exact_at(&mut bytes, 0).unwrap();
        assert_eq!(&bytes, b"packed weights");
    }
}
