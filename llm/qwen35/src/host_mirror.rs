//! Opt-in immutable packed experts, preloaded from NVMe into bounded pinned RAM.
//! Store owns this mirror and synchronously uploads borrowed hits before returning.
use super::{
    error,
    store::{Reservation, Weight, TILE},
    Gguf, Gpu, Result,
};
use nextrocm_engine_core::HostBuf;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    os::{fd::AsRawFd, unix::fs::FileExt},
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};
const MAX_BUDGET: usize = 2 * 1024 * 1024 * 1024;
const RESERVE: u64 = 3 * 1024 * 1024 * 1024;
const ADMIN: u64 = 64 * 1024 * 1024;
const MAX_PROFILE: u64 = 4 * 1024 * 1024;
type Key = (usize, u64); // shard, absolute start of one expert's matrix
struct Chunk {
    buffer: HostBuf,
    _reservation: Reservation,
}
#[derive(Clone, Debug)]
struct Slice {
    key: Key,
    len: usize,
    chunk: usize,
    at: usize,
}
#[derive(Default)]
struct Plan {
    slices: Vec<Slice>,
    chunks: Vec<usize>,
    bytes: usize,
    experts: usize,
}
#[derive(Default)]
struct Counters {
    hits: AtomicU64,
    misses: AtomicU64,
    hit_bytes: AtomicU64,
    miss_bytes: AtomicU64,
}
/// All buffer owners precede accounting owners, so memory is freed before release.
pub(crate) struct HostMirror {
    chunks: Vec<Chunk>,
    index: BTreeMap<Key, Slice>,
    counters: Counters,
    requested: usize,
    admitted: usize,
    experts: usize,
    seconds: f64,
    advisory_failures: usize,
    advised_bytes: u64,
    mem_available_before: u64,
    _metadata_reservation: Option<Reservation>,
}
fn number(v: &Value, key: &str) -> Result<u64> {
    v.get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| error(format!("host mirror: missing/noninteger {key}")))
}
fn ranks(data: &[u8], model_bytes: u64, fingerprint: u64) -> Result<Vec<(usize, usize, u64)>> {
    if data.len() as u64 > MAX_PROFILE {
        return Err(error("host mirror: profile exceeds 4 MiB"));
    }
    let v: Value =
        serde_json::from_slice(data).map_err(|e| error(format!("host mirror profile: {e}")))?;
    if number(&v, "model_bytes")? != model_bytes || number(&v, "model_fingerprint")? != fingerprint
    {
        return Err(error("host mirror: profile model identity mismatch"));
    }
    let entries = v
        .get("experts")
        .and_then(Value::as_array)
        .ok_or_else(|| error("host mirror: missing experts array"))?;
    if entries.len() > 65536 {
        return Err(error("host mirror: too many profile entries"));
    }
    let mut seen = BTreeSet::new();
    let mut rows = Vec::with_capacity(entries.len());
    for e in entries {
        let l = usize::try_from(number(e, "layer")?)
            .map_err(|_| error("host mirror: layer overflow"))?;
        let x = usize::try_from(number(e, "expert")?)
            .map_err(|_| error("host mirror: expert overflow"))?;
        let count = number(e, "count")?;
        if !seen.insert((l, x)) {
            return Err(error("host mirror: duplicate ranked expert"));
        }
        rows.push((l, x, count));
    }
    rows.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
    Ok(rows)
}
fn plan(
    weights: &BTreeMap<String, Weight>,
    rows: &[(usize, usize, u64)],
    budget: usize,
) -> Result<Plan> {
    let mut p = Plan::default();
    let mut seen = BTreeSet::new();
    for &(layer, expert, count) in rows {
        let mut parts = Vec::with_capacity(3);
        let mut total = 0usize;
        for role in ["gate", "up", "down"] {
            let name = format!("blk.{layer}.ffn_{role}_exps.weight");
            let w = weights
                .get(&name)
                .ok_or_else(|| error(format!("host mirror: missing {name}")))?;
            if !w.streamed() || expert >= w.experts {
                return Err(error(format!(
                    "host mirror: invalid expert {layer}/{expert}"
                )));
            }
            let len = w
                .rows
                .checked_mul(w.row_bytes)
                .ok_or_else(|| error("host mirror: matrix size overflow"))?;
            let offset = expert
                .checked_mul(len)
                .ok_or_else(|| error("host mirror: expert offset overflow"))?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| error("host mirror: extent overflow"))?;
            if len == 0 || end as u64 > w.tensor.bytes {
                return Err(error("host mirror: invalid matrix extent"));
            }
            let absolute = w
                .tensor
                .offset
                .checked_add(offset as u64)
                .ok_or_else(|| error("host mirror: file offset overflow"))?;
            total = total
                .checked_add(len)
                .ok_or_else(|| error("host mirror: bundle size overflow"))?;
            parts.push(((w.tensor.shard, absolute), len));
        }
        if total > TILE {
            return Err(error(
                "host mirror: whole gate/up/down expert exceeds 16 MiB chunk",
            ));
        }
        // Validate every profile ID, even if zero-count or too cold to fit.
        if count == 0 || p.bytes.checked_add(total).is_none_or(|n| n > budget) {
            continue;
        }
        if p.chunks.last().is_none_or(|n| n + total > TILE) {
            p.chunks.push(0);
        }
        let chunk = p.chunks.len() - 1;
        for (key, len) in parts {
            if !seen.insert(key) {
                return Err(error("host mirror: aliased expert slices"));
            }
            let at = p.chunks[chunk];
            p.chunks[chunk] += len;
            p.slices.push(Slice {
                key,
                len,
                chunk,
                at,
            });
        }
        p.bytes += total;
        p.experts += 1;
    }
    Ok(p)
}
fn admission(available: u64, budget: usize, future_device: u64) -> Result<()> {
    if budget > MAX_BUDGET {
        return Err(error("host mirror: budget exceeds 2 GiB experiment cap"));
    }
    let need = RESERVE
        .checked_add(ADMIN)
        .and_then(|n| n.checked_add(TILE as u64))
        .and_then(|n| n.checked_add(future_device))
        .and_then(|n| n.checked_add(budget as u64))
        .ok_or_else(|| error("host mirror: UMA reservation overflow"))?;
    if available < need {
        return Err(error(format!(
            "host mirror: UMA headroom needs {need} bytes, available {available}"
        )));
    }
    Ok(())
}
fn available() -> Result<u64> {
    let text = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| error(format!("host mirror meminfo: {e}")))?;
    let value = text
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))
        .and_then(|s| s.split_whitespace().next())
        .and_then(|v| v.parse::<u64>().ok())
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| error("host mirror: missing MemAvailable"))?;
    Ok(value)
}
fn advise_loaded(file: &File, offset: u64, len: usize) -> Result<(bool, u64)> {
    // Only completely resident full pages are discarded; shared boundary pages remain.
    // SAFETY: sysconf takes no pointers.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return Err(error("host mirror: invalid system page size"));
    }
    let page = page as u64;
    let start = offset
        .checked_add(page - 1)
        .ok_or_else(|| error("host mirror: page alignment overflow"))?
        / page
        * page;
    let end = offset
        .checked_add(len as u64)
        .ok_or_else(|| error("host mirror: advice extent overflow"))?
        / page
        * page;
    if end <= start {
        return Ok((true, 0));
    }
    let start =
        libc::off_t::try_from(start).map_err(|_| error("host mirror: advisory offset overflow"))?;
    let n = libc::off_t::try_from(end - start as u64)
        .map_err(|_| error("host mirror: advisory length overflow"))?;
    // SAFETY: valid read-only file, checked positive aligned range. Advisory only.
    let ok =
        unsafe { libc::posix_fadvise(file.as_raw_fd(), start, n, libc::POSIX_FADV_DONTNEED) } == 0;
    Ok((ok, if ok { n as u64 } else { 0 }))
}
impl HostMirror {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        gpu: &Arc<Gpu>,
        f: &Gguf,
        weights: &BTreeMap<String, Weight>,
        profilepath: &Path,
        budget: usize,
        future_device: u64,
        expected_fingerprint: u64,
    ) -> Result<Self> {
        Self::load_with_memory(
            gpu,
            f,
            weights,
            profilepath,
            budget,
            future_device,
            expected_fingerprint,
            available,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn load_with_memory(
        gpu: &Arc<Gpu>,
        f: &Gguf,
        weights: &BTreeMap<String, Weight>,
        profilepath: &Path,
        budget: usize,
        future_device: u64,
        expected_fingerprint: u64,
        memory: impl Fn() -> Result<u64>,
    ) -> Result<Self> {
        let mut mirror = Self {
            chunks: Vec::new(),
            index: BTreeMap::new(),
            counters: Counters::default(),
            requested: budget,
            admitted: 0,
            experts: 0,
            seconds: 0.,
            advisory_failures: 0,
            advised_bytes: 0,
            mem_available_before: 0,
            _metadata_reservation: None,
        };
        if budget == 0 {
            return Ok(mirror);
        }
        if budget > MAX_BUDGET {
            return Err(error("host mirror: budget exceeds 2 GiB experiment cap"));
        }
        let start = Instant::now();
        let avail = memory()?;
        admission(avail, budget, future_device)?;
        mirror.mem_available_before = avail;
        // Conservative bounded allowance for profile parsing, indices and file metadata.
        mirror._metadata_reservation = Some(Reservation::host(ADMIN as usize)?);
        let files = f
            .paths
            .iter()
            .map(|p| File::open(p).map_err(|e| error(format!("host mirror {}: {e}", p.display()))))
            .collect::<Result<Vec<_>>>()?;
        let sizes = files
            .iter()
            .map(|f| {
                f.metadata()
                    .map(|m| m.len())
                    .map_err(|e| error(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let model_bytes = sizes
            .iter()
            .try_fold(0u64, |n, s| n.checked_add(*s))
            .ok_or_else(|| error("host mirror: model bytes overflow"))?;
        let mut bytes = Vec::new();
        File::open(profilepath)
            .map_err(|e| {
                error(format!(
                    "host mirror profile {}: {e}",
                    profilepath.display()
                ))
            })?
            .take(MAX_PROFILE + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| error(e.to_string()))?;
        let rows = ranks(&bytes, model_bytes, expected_fingerprint)?;
        drop(bytes);
        let p = plan(weights, &rows, budget)?;
        drop(rows);
        for slice in &p.slices {
            let size = sizes
                .get(slice.key.0)
                .ok_or_else(|| error("host mirror: invalid shard"))?;
            if slice
                .key
                .1
                .checked_add(slice.len as u64)
                .is_none_or(|end| end > *size)
            {
                return Err(error("host mirror: expert outside checkpoint shard"));
            }
        }
        for (index, &len) in p.chunks.iter().enumerate() {
            // Current MemAvailable already reflects earlier chunks. Reserve only the remaining allocations.
            admission(memory()?, p.bytes - mirror.admitted, future_device)?;
            let reservation = Reservation::host(len)?;
            let mut chunk = Chunk {
                buffer: HostBuf::new(gpu, len)?,
                _reservation: reservation,
            };
            for slice in p.slices.iter().filter(|s| s.chunk == index) {
                files[slice.key.0]
                    .read_exact_at(
                        &mut chunk.buffer.as_mut_slice()[slice.at..slice.at + slice.len],
                        slice.key.1,
                    )
                    .map_err(|e| error(format!("host mirror read: {e}")))?;
                let (ok, advised) = advise_loaded(&files[slice.key.0], slice.key.1, slice.len)?;
                mirror.advised_bytes += advised;
                if !ok {
                    mirror.advisory_failures += 1;
                }
            }
            mirror.admitted += len;
            mirror.chunks.push(chunk);
        }
        mirror.index = p.slices.into_iter().map(|s| (s.key, s)).collect();
        mirror.experts = p.experts;
        mirror.seconds = start.elapsed().as_secs_f64();
        Ok(mirror)
    }
    /// Pure lookup. Caller records a hit only after successful synchronized upload.
    pub fn hit(&self, w: &Weight, offset: usize, len: usize) -> Option<&[u8]> {
        if !w.streamed()
            || len == 0
            || len > TILE
            || offset.checked_add(len)? as u64 > w.tensor.bytes
        {
            return None;
        }
        let absolute = w.tensor.offset.checked_add(offset as u64)?;
        let (_, slice) = self
            .index
            .range(..=(w.tensor.shard, absolute))
            .next_back()?;
        if slice.key.0 != w.tensor.shard {
            return None;
        }
        let skip = usize::try_from(absolute.checked_sub(slice.key.1)?).ok()?;
        if skip.checked_add(len)? > slice.len {
            return None;
        }
        self.chunks
            .get(slice.chunk)?
            .buffer
            .as_slice()
            .get(slice.at + skip..slice.at + skip + len)
    }
    pub fn note_hit(&self, len: usize) {
        self.counters.hits.fetch_add(1, Ordering::Relaxed);
        self.counters
            .hit_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
    }
    pub fn note_miss(&self, len: usize) {
        self.counters.misses.fetch_add(1, Ordering::Relaxed);
        self.counters
            .miss_bytes
            .fetch_add(len as u64, Ordering::Relaxed);
    }
    pub fn report(&self) -> Vec<String> {
        vec![format!("qwen35 host mirror: requested={} resident={} experts={} chunks={} load_seconds={:.6} host_hits={} host_misses={} hit_bytes={} miss_bytes={} mem_available_before={} advisory_failures={} advised_bytes={} advisory_freed_bytes=unknown capacity_actual={} metadata_reservation={}",self.requested,self.admitted,self.experts,self.chunks.len(),self.seconds,self.counters.hits.load(Ordering::Relaxed),self.counters.misses.load(Ordering::Relaxed),self.counters.hit_bytes.load(Ordering::Relaxed),self.counters.miss_bytes.load(Ordering::Relaxed),self.mem_available_before,self.advisory_failures,self.advised_bytes,self.chunks.iter().map(|c| c.buffer.len).sum::<usize>(),if self._metadata_reservation.is_some(){ADMIN}else{0})]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextrocm_gguf::{writer::Writer, GType, Tensor};
    fn weights() -> BTreeMap<String, Weight> {
        ["gate", "up", "down"]
            .into_iter()
            .enumerate()
            .map(|(i, role)| {
                let t = Tensor {
                    name: format!("blk.0.ffn_{role}_exps.weight"),
                    shape: vec![3, 1, 256],
                    ty: GType::Q4K,
                    shard: 0,
                    offset: (i * 432) as u64,
                    bytes: 432,
                };
                (t.name.clone(), Weight::new(&t).unwrap())
            })
            .collect()
    }
    #[test]
    fn identity_duplicates_and_rank_order_are_validated() {
        let data=br#"{"model_bytes":100,"model_fingerprint":9,"experts":[{"layer":1,"expert":2,"count":3},{"layer":0,"expert":2,"count":3},{"layer":0,"expert":1,"count":8}]}"#;
        assert_eq!(
            ranks(data, 100, 9).unwrap(),
            [(0, 1, 8), (0, 2, 3), (1, 2, 3)]
        );
        assert!(ranks(data, 99, 9).is_err());
        assert!(ranks(data, 100, 8).is_err());
        assert!(ranks(br#"{"model_bytes":1,"model_fingerprint":2,"experts":[{"layer":0,"expert":0,"count":1},{"layer":0,"expert":0,"count":2}]}"#,1,2).is_err());
        assert!(ranks(br#"{"model_bytes":1,"model_fingerprint":2,"experts":[{"layer":0,"expert":0,"count":-1}]}"#,1,2).is_err());
    }
    #[test]
    fn admission_checks_future_device_and_caps_without_overflow() {
        let need = RESERVE + ADMIN + TILE as u64 + 1024 + 4096;
        assert!(admission(need, 1024, 4096).is_ok());
        assert!(admission(need - 1, 1024, 4096).is_err());
        assert!(admission(u64::MAX, MAX_BUDGET + 1, 0).is_err());
        assert!(admission(u64::MAX, 1, u64::MAX).is_err());
    }
    #[test]
    fn plan_admits_all_three_matrices_or_none_and_checks_all_ids() {
        let w = weights();
        let rows = [(0, 2, 10), (0, 0, 5), (0, 1, 0)];
        let p = plan(&w, &rows, 432).unwrap();
        assert_eq!(p.bytes, 432);
        assert_eq!(p.experts, 1);
        assert_eq!(p.slices.len(), 3);
        assert!(p.slices.iter().all(|s| s.chunk == 0));
        assert_eq!(
            p.slices.iter().map(|s| s.key.1).collect::<Vec<_>>(),
            [288, 720, 1152]
        );
        assert_eq!(plan(&w, &rows, 431).unwrap().bytes, 0);
        assert!(plan(&w, &[(0, 3, 0)], 0).is_err());
        assert!(plan(&w, &[(1, 0, 0)], 0).is_err());
        let mut huge = weights();
        for x in huge.values_mut() {
            x.rows = 40000;
            x.tensor.bytes = (40000 * 144 * 3) as u64;
        }
        assert!(plan(&huge, &[(0, 0, 1)], MAX_BUDGET).is_err());
    }
    #[test]
    fn chunks_are_bounded_without_splitting_expert_bundles() {
        let mut w = weights();
        for (i, x) in w.values_mut().enumerate() {
            x.rows = 20000;
            x.tensor.bytes = (20000 * 144 * 3) as u64;
            x.tensor.offset = i as u64 * 100_000_000;
        }
        let p = plan(&w, &[(0, 0, 9), (0, 1, 8)], MAX_BUDGET).unwrap();
        assert_eq!(p.chunks.len(), 2);
        assert!(p.chunks.iter().all(|n| *n <= TILE));
        assert_eq!(p.bytes, p.chunks.iter().sum::<usize>());
        assert!(p.slices[..3].iter().all(|s| s.chunk == 0));
        assert!(p.slices[3..].iter().all(|s| s.chunk == 1));
    }
    #[test]
    fn detached_preload_lookup_is_immutable_pure_and_survives_file_removal() {
        let dir = std::env::temp_dir().join(format!(
            "q35-mirror-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let model = dir.join("tiny.gguf");
        let profile = dir.join("rank.json");
        let specs: Vec<_> = weights()
            .values()
            .map(|w| (w.tensor.name.clone(), w.tensor.ty, w.tensor.shape.clone()))
            .collect();
        let mut writer = Writer::create(&model, &[], &specs).unwrap();
        for i in 0..3 {
            writer
                .data(
                    &(0..432)
                        .map(|n| ((n + i * 17) % 251) as u8)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        writer.finish().unwrap();
        let f = Gguf::open(&model).unwrap();
        let w: BTreeMap<_, _> = f
            .tensors
            .iter()
            .map(|t| (t.name.clone(), Weight::new(t).unwrap()))
            .collect();
        std::fs::write(&profile,format!(r#"{{"model_bytes":{},"model_fingerprint":77,"experts":[{{"layer":0,"expert":2,"count":4}}]}}"#,std::fs::metadata(&model).unwrap().len())).unwrap();
        let gpu = Gpu::detached("host mirror CPU test");
        let m = HostMirror::load_with_memory(&gpu, &f, &w, &profile, 432, 0, 77, || Ok(16 << 30))
            .unwrap();
        assert_eq!(m.admitted, 432);
        assert_eq!(m.chunks.len(), 1);
        for weight in w.values() {
            let mut expected = vec![0; 144];
            f.read_into(&weight.tensor, 288, &mut expected).unwrap();
            assert_eq!(m.hit(weight, 288, 144).unwrap(), expected);
            assert_eq!(m.hit(weight, 291, 17).unwrap(), &expected[3..20]);
            assert!(m.hit(weight, 144, 144).is_none());
            assert!(m.hit(weight, 287, 2).is_none());
            assert!(m.hit(weight, 431, 2).is_none());
        }
        assert_eq!(m.counters.hits.load(Ordering::Relaxed), 0);
        std::fs::remove_file(&model).unwrap();
        std::fs::remove_file(&profile).unwrap();
        assert!(m.hit(w.values().next().unwrap(), 288, 144).is_some());
        m.note_hit(144);
        m.note_miss(72);
        assert!(m.report()[0].contains("host_hits=1 host_misses=1 hit_bytes=144 miss_bytes=72"));
        let empty = HostMirror::load_with_memory(&gpu, &f, &w, &profile, 0, u64::MAX, 0, || {
            panic!("disabled cache queried memory")
        })
        .unwrap();
        assert!(empty.chunks.is_empty());
        drop(m);
        drop(empty);
        std::fs::remove_dir(dir).unwrap();
    }
}
