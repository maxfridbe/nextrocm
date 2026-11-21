//! Packed rows from bounded mappings/staging or an explicitly budgeted RAM mirror.
use super::*;
use nextrocm_gguf::Tensor;
use std::{
    fs::File,
    os::fd::AsRawFd,
    os::unix::fs::FileExt,
    sync::atomic::{AtomicUsize, Ordering},
};
pub(crate) const TILE: usize = 16 * 1024 * 1024;
static DEVICE: AtomicUsize = AtomicUsize::new(0);
static HOST: AtomicUsize = AtomicUsize::new(0);
/// A reservation is acquired before allocation and released after its buffers.
pub(crate) struct Reservation {
    counter: &'static AtomicUsize,
    bytes: usize,
}
impl Reservation {
    fn acquire(counter: &'static AtomicUsize, bytes: usize, limit: usize) -> Result<Self> {
        counter
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                n.checked_add(bytes).filter(|v| *v < limit)
            })
            .map_err(|_| {
                error(format!(
                    "qwen35 tracked allocation limit ({limit} bytes) exceeded"
                ))
            })?;
        Ok(Self { counter, bytes })
    }
    pub fn device(n: usize) -> Result<Self> {
        Self::acquire(&DEVICE, n, 8_000_000_000)
    }
    pub fn host(n: usize) -> Result<Self> {
        // At most 2 GiB of optional expert mirror; leave room for the existing
        // staging/checkpoints. Physical UMA headroom is checked before mirroring.
        Self::acquire(&HOST, n, 3 * 1024 * 1024 * 1024)
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.counter.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
pub(crate) struct Weight {
    pub tensor: Tensor,
    pub rows: usize,
    pub cols: usize,
    pub experts: usize,
    pub row_bytes: usize,
    pub resident: Option<DevBuf>,
}
impl Weight {
    /// Only packed three-dimensional routed expert banks remain on disk.
    pub fn streamed(&self) -> bool {
        self.tensor.shape.len() == 3
    }
    pub fn resident_bytes(&self) -> Result<usize> {
        if self.streamed() {
            return Ok(0);
        }
        usize::try_from(self.tensor.bytes).map_err(|_| error("resident weight size overflow"))
    }
    pub fn new(t: &Tensor) -> Result<Self> {
        let cols = *t.shape.last().ok_or_else(|| error("empty weight"))? as usize;
        let rows = if t.shape.len() > 1 {
            t.shape[t.shape.len() - 2] as usize
        } else {
            1
        };
        let experts = if t.shape.len() == 3 {
            t.shape[0] as usize
        } else {
            1
        };
        let row_bytes = usize::try_from(
            t.ty.bytes(cols as u64)
                .ok_or_else(|| error("unaligned matrix row"))?,
        )
        .map_err(|_| error("row bytes overflow"))?;
        if row_bytes > TILE {
            return Err(error("matrix row exceeds staging capacity"));
        }
        Ok(Self {
            tensor: t.clone(),
            rows,
            cols,
            experts,
            row_bytes,
            resident: None,
        })
    }
}
pub(crate) struct Store {
    files: Vec<File>,
    cache_pages: bool,
    pinned: Option<Mutex<nextrocm_engine_core::HostBuf>>,
    mirror: Option<super::host_mirror::HostMirror>,
    prefetch: Mutex<super::prefetch::Prefetch>,
    prefetch_enabled: bool,
    _host_reservation: Option<Reservation>,
}
impl Store {
    pub fn open(f: &Gguf, gpu: &Arc<Gpu>) -> Result<Self> {
        let files = f
            .paths
            .iter()
            .map(|p| File::open(p).map_err(|e| error(format!("open {}: {e}", p.display()))))
            .collect::<Result<Vec<_>>>()?;
        for t in &f.tensors {
            let file = files
                .get(t.shard)
                .ok_or_else(|| error("invalid tensor shard"))?;
            let len = file.metadata().map_err(|e| error(e.to_string()))?.len();
            if t.offset.checked_add(t.bytes).is_none_or(|end| end > len) {
                return Err(error(format!("{} is outside file", t.name)));
            }
        }
        let cache_pages = match std::env::var("NS_Q35_PAGE_CACHE") {
            Ok(value) if value == "1" => true,
            Ok(value) if value == "0" => false,
            Err(std::env::VarError::NotPresent) => false,
            _ => return Err(error("NS_Q35_PAGE_CACHE must be 0 or 1")),
        };
        let pinned_enabled = match std::env::var("NS_Q35_PINNED_STAGE") {
            Ok(value) if value == "1" => true,
            Ok(value) if value == "0" => false,
            Err(std::env::VarError::NotPresent) => false,
            _ => return Err(error("NS_Q35_PINNED_STAGE must be 0 or 1")),
        };
        let host_reservation = pinned_enabled
            .then(|| Reservation::host(TILE))
            .transpose()?;
        let pinned = if pinned_enabled {
            Some(Mutex::new(nextrocm_engine_core::HostBuf::new(gpu, TILE)?))
        } else {
            None
        };
        let prefetch_mib = std::env::var("NS_Q35_PREFETCH_MIB")
            .map_or(Ok(0usize), |v| v.parse::<usize>())
            .map_err(|_| error("NS_Q35_PREFETCH_MIB must be an integer in 0..32"))?;
        if prefetch_mib > 32 {
            return Err(error("NS_Q35_PREFETCH_MIB exceeds 32 MiB per layer"));
        }
        Ok(Self {
            files,
            cache_pages,
            pinned,
            mirror: None,
            prefetch: Mutex::new(super::prefetch::Prefetch::new(prefetch_mib * 1024 * 1024)),
            prefetch_enabled: prefetch_mib != 0,
            _host_reservation: host_reservation,
        })
    }
    pub fn configure(
        &mut self,
        gpu: &Arc<Gpu>,
        f: &Gguf,
        weights: &BTreeMap<String, Weight>,
        model: u64,
        budget: usize,
        future_device: u64,
    ) -> Result<()> {
        if budget == 0 {
            return Ok(());
        }
        if budget > 2 * 1024 * 1024 * 1024 {
            return Err(error("host expert mirror exceeds 2 GiB experiment cap"));
        }
        let path = std::env::var_os("NS_Q35_EXPERT_PROFILE")
            .filter(|p| !p.is_empty())
            .ok_or_else(|| error("host expert mirror requires NS_Q35_EXPERT_PROFILE"))?;
        self.mirror = Some(super::host_mirror::HostMirror::load(
            gpu,
            f,
            weights,
            std::path::Path::new(&path),
            budget,
            future_device,
            model,
        )?);
        Ok(())
    }
    pub fn prefetch_enabled(&self) -> bool {
        self.prefetch_enabled
    }
    pub fn prefetch(&self, requests: &[(&Weight, usize, usize)]) {
        if !self.prefetch_enabled {
            return;
        }
        let windows: Vec<_> = requests
            .iter()
            .filter_map(|&(w, offset, len)| {
                if !w.streamed()
                    || offset
                        .checked_add(len)
                        .is_none_or(|n| n as u64 > w.tensor.bytes)
                {
                    return None;
                }
                if self
                    .mirror
                    .as_ref()
                    .is_some_and(|m| m.hit(w, offset, len).is_some())
                {
                    return None;
                }
                let file = self.files.get(w.tensor.shard)?;
                let absolute = w.tensor.offset.checked_add(offset as u64)?;
                Some((file, absolute, len))
            })
            .collect();
        if let Ok(mut prefetch) = self.prefetch.lock() {
            prefetch.submit(&windows);
        }
    }
    pub fn report(&self) -> Vec<String> {
        let mut report = self.mirror.as_ref().map_or_else(Vec::new, |m| m.report());
        if self.prefetch_enabled {
            if let Ok(prefetch) = self.prefetch.lock() {
                report.extend(prefetch.report());
            }
        }
        report
    }
    pub fn copy(&self, w: &Weight, offset: usize, len: usize, dst: &DevBuf) -> Result<()> {
        if len > TILE
            || offset
                .checked_add(len)
                .is_none_or(|v| v as u64 > w.tensor.bytes)
        {
            return Err(error("invalid mapped tensor window"));
        }
        if let Some(mirror) = &self.mirror {
            if let Some(bytes) = mirror.hit(w, offset, len) {
                dst.write(0, bytes)?;
                mirror.note_hit(len);
                return Ok(());
            }
            if w.streamed() {
                mirror.note_miss(len);
            }
        }
        let file = self
            .files
            .get(w.tensor.shard)
            .ok_or_else(|| error("missing shard"))?;
        let absolute = w
            .tensor
            .offset
            .checked_add(offset as u64)
            .ok_or_else(|| error("file offset overflow"))?;
        let copied = if let Some(pinned) = &self.pinned {
            let mut host = pinned
                .lock()
                .map_err(|_| error("pinned staging lock poisoned"))?;
            file.read_exact_at(&mut host.as_mut_slice()[..len], absolute)
                .map_err(|e| error(format!("read packed weight: {e}")))?;
            // write synchronizes before the reusable host allocation is released.
            dst.write(0, &host.as_slice()[..len])
        } else {
            let m = Mapping::new(file, absolute, len)?;
            dst.write(0, m.bytes())
        };
        // SAFETY: advisory only, valid open fd and checked window offset; no file contents changed.
        // Align down: partial boundary pages may remain cached, at most two pages per transfer.
        if !self.cache_pages {
            unsafe {
                libc::posix_fadvise(
                    file.as_raw_fd(),
                    absolute as libc::off_t,
                    len as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                );
            }
        }
        copied
    }
}
struct Mapping {
    ptr: *mut std::ffi::c_void,
    len: usize,
    skip: usize,
    bytes: usize,
}
impl Mapping {
    fn new(file: &File, offset: u64, bytes: usize) -> Result<Self> {
        // SAFETY: sysconf has no pointer arguments or process side effects.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 || bytes == 0 {
            return Err(error("invalid mapping page/length"));
        }
        let skip = (offset % page as u64) as usize;
        let len = skip
            .checked_add(bytes)
            .ok_or_else(|| error("mapping overflow"))?;
        let at = libc::off_t::try_from(offset - skip as u64)
            .map_err(|_| error("mapping offset exceeds off_t"))?;
        // SAFETY: file bounds checked by Store; read-only private mapping. Model files must remain immutable while loaded.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                at,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(error(format!("mmap: {}", std::io::Error::last_os_error())));
        }
        Ok(Self {
            ptr,
            len,
            skip,
            bytes,
        })
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: valid live read-only mapping, skip+bytes==len.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>().add(self.skip), self.bytes) }
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: this object uniquely owns this mapping; copies to GPU are synchronized before drop.
        unsafe {
            libc::madvise(self.ptr, self.len, libc::MADV_DONTNEED);
            libc::munmap(self.ptr, self.len);
        }
    }
}
/// Top-k softmax, normalized over the chosen experts (Qwen35MoE norm_topk_prob).
/// Sorting logits is equivalent to sorting their softmax; ties use lowest expert id.
pub(crate) fn route(logits: &[f32], k: usize) -> Result<Vec<(usize, f32)>> {
    if k == 0 || k > logits.len() || logits.iter().any(|v| !v.is_finite()) {
        return Err(error("invalid router logits/top-k"));
    }
    let mut ids: Vec<_> = (0..logits.len()).collect();
    ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    ids.truncate(k);
    let max = logits[ids[0]];
    let sum: f32 = ids.iter().map(|&i| (logits[i] - max).exp()).sum();
    Ok(ids
        .into_iter()
        .map(|i| (i, (logits[i] - max).exp() / sum))
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn router_stability_and_ties() {
        let v = route(&[1000., 1000., 999., -1000.], 2).unwrap();
        assert_eq!(v, vec![(0, 0.5), (1, 0.5)]);
        assert!(route(&[f32::NAN], 1).is_err());
        assert!(route(&[0.], 2).is_err());
    }
    #[test]
    fn packed_dense_residency_and_large_load_admission() {
        use nextrocm_gguf::GType;
        let tensor = Tensor {
            name: "output.weight".into(),
            shape: vec![4096, 4096],
            ty: GType::Q4K,
            shard: 0,
            offset: 0,
            bytes: 9_437_184,
        };
        let w = Weight::new(&tensor).unwrap();
        assert_eq!(w.resident_bytes().unwrap(), 9_437_184);
        assert_eq!(w.row_bytes, 2304);
        static C: AtomicUsize = AtomicUsize::new(0);
        let reserve =
            Reservation::acquire(&C, 3_000_000_000 + MARGIN as usize, 8_000_000_000).unwrap();
        assert!(Reservation::acquire(&C, 5_000_000_000, 8_000_000_000).is_err());
        drop(reserve);
        assert!(Reservation::acquire(&C, 8_000_000_000, 8_000_000_000).is_err());
        assert_eq!(C.load(Ordering::Acquire), 0);
    }
    #[test]
    fn reservations_release_and_overflow() {
        static C: AtomicUsize = AtomicUsize::new(0);
        {
            let _r = Reservation::acquire(&C, 9, 10).unwrap();
            assert!(Reservation::acquire(&C, 1, 10).is_err());
            assert!(Reservation::acquire(&C, usize::MAX, 10).is_err());
        }
        assert_eq!(C.load(Ordering::Acquire), 0);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::*;
    #[test]
    fn unaligned_mapping_reads_only_requested_slice() {
        let path = std::env::temp_dir().join(format!("q35-map-{}", std::process::id()));
        let data: Vec<_> = (0..10000).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let file = File::open(&path).unwrap();
        {
            let m = Mapping::new(&file, 4091, 33).unwrap();
            assert_eq!(m.bytes(), &data[4091..4124]);
        }
        assert!(Mapping::new(&file, 0, 0).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
