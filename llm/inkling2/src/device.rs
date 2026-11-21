//! Bounded device residency. Dense weights stay packed; only one selected expert
//! pair is streamed. Construction is not a validated inference/admission result.
use crate::{
    expert_cache::{ExpertMode, ExpertStats, PairStaging},
    memory::Estimate,
    plan::{ExecutionPlan, Visit},
    session::StateLayout,
    store::HostStore,
    weights::Weights,
    Result,
};
use nextrocm_engine_core::{DevBuf, Gpu};
use std::{collections::BTreeMap, sync::Arc};

pub struct DeviceState {
    pub layout: StateLayout,
    pub kv: DevBuf,
    pub history: DevBuf,
}
impl DeviceState {
    fn allocate(gpu: &Arc<Gpu>, plan: &ExecutionPlan) -> Result<Self> {
        let layout = StateLayout::new(plan)?;
        let kv = DevBuf::new(gpu, layout.kv_bytes).map_err(|e| e.0)?;
        let history = DevBuf::new(gpu, layout.history_bytes).map_err(|e| e.0)?;
        kv.fill(0).map_err(|e| e.0)?;
        history.fill(0).map_err(|e| e.0)?;
        gpu.sync().map_err(|e| e.0)?;
        Ok(Self {
            layout,
            kv,
            history,
        })
    }
}
pub struct DeviceWeights {
    dense: BTreeMap<String, DevBuf>,
    expert: DevBuf,
    staging: Option<PairStaging>,
    source_identity: Arc<()>,
    mode: ExpertMode,
    stats: ExpertStats,
    pub state: DeviceState,
    pub estimate: Estimate,
}
pub(crate) struct ExpertDevice {
    pub gate_up: DevBuf,
    pub down: DevBuf,
    pub gate_scale: Option<DevBuf>,
    pub down_scale: Option<DevBuf>,
}
impl DeviceWeights {
    /// Caller must resolve every required graph ABI before entering this load.
    /// Cancellation is checked before each bounded copy; errors drop all partial
    /// device allocations. No host mirror and no page-cache eviction hints.
    pub fn load(
        store: &mut HostStore,
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        host_available: u64,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        Self::load_with_expert_mode(
            store,
            gpu,
            plan,
            host_available,
            ExpertMode::Stream,
            &mut check,
        )
    }
    pub fn load_with_expert_mode(
        store: &mut HostStore,
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        host_available: u64,
        mode: ExpertMode,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<Self> {
        if gpu.raw().is_null() {
            return Err("device weights require a real GPU".into());
        }
        if plan.context() > 65536 {
            return Err("initial native attention supports context <=65536".into());
        }
        check()?;
        let free = gpu
            .memory()
            .map_err(|e| e.0)?
            .1
            .ok_or("device free memory unavailable")?;
        let weights = Weights::read(store.file())?;
        let estimate =
            Estimate::text_with_expert_mode(&weights, plan, 1, host_available, free, mode)?;
        let entries: Vec<_> = store
            .file()
            .tensors
            .iter()
            .filter(|t| !t.name.contains("_exps.weight"))
            .map(|t| (t.name.clone(), t.bytes))
            .collect();
        let mut dense = BTreeMap::new();
        let mut uploaded = 0u64;
        for (name, bytes) in entries {
            check()?;
            let b = DevBuf::new(
                gpu,
                usize::try_from(bytes).map_err(|_| "tensor bytes exceed usize")?,
            )
            .map_err(|e| e.0)?;
            store.stream(&name, 0, bytes, |offset, data| {
                check()?;
                b.write(
                    usize::try_from(offset).map_err(|_| "copy offset exceeds usize")?,
                    data,
                )
                .map_err(|e| e.0)
            })?;
            uploaded = uploaded
                .checked_add(bytes)
                .ok_or("uploaded size overflow")?;
            dense.insert(name, b);
        }
        if uploaded != estimate.resident_weights {
            return Err("resident inventory differs from plan".into());
        }
        check()?;
        let expert = DevBuf::new(
            gpu,
            usize::try_from(estimate.expert_pair).map_err(|_| "expert size exceeds usize")?,
        )
        .map_err(|e| e.0)?;
        check()?;
        let staging = PairStaging::new(gpu, mode, expert.len)?;
        check()?;
        let state = DeviceState::allocate(gpu, plan)?;
        Ok(Self {
            dense,
            expert,
            staging,
            source_identity: store.identity(),
            mode,
            stats: ExpertStats::default(),
            state,
            estimate,
        })
    }
    pub fn expert_stats(&self) -> ExpertStats {
        self.stats
    }
    pub fn expert_mode(&self) -> ExpertMode {
        self.mode
    }
    pub fn tensor(&self, name: &str) -> Result<&DevBuf> {
        self.dense
            .get(name)
            .ok_or_else(|| format!("nonresident tensor {name}"))
    }
    /// Internal graph contract: the callback must not retain views or pointers
    /// into this pair. Their values expire at the next expert upload, and their
    /// allocation dies with this owner. Outputs must use separate owned buffers.
    /// DevBuf::write calls gpu.sync() after copy_to (engine-core/src/lib.rs),
    /// so prior expert launches finish before the next upload returns.
    pub(crate) fn with_expert(
        &mut self,
        store: &mut HostStore,
        visit: &Visit,
        expert: usize,
        mut check: impl FnMut() -> Result<()>,
        f: impl FnOnce(&ExpertDevice) -> Result<()>,
    ) -> Result<()> {
        check()?;
        if !Arc::ptr_eq(&self.source_identity, &store.identity()) {
            return Err("expert source instance mismatch".into());
        }
        let spans = store.expert_spans(visit, expert)?;
        let first = usize::try_from(spans[0].bytes).map_err(|_| "expert bytes exceed usize")?;
        let second = usize::try_from(spans[1].bytes).map_err(|_| "expert bytes exceed usize")?;
        self.expert
            .bounds(0, first.checked_add(second).ok_or("expert pair overflow")?)
            .map_err(|e| e.0)?;
        let (gate_up, down) = if let Some(staging) = &mut self.staging {
            staging.pair(
                &self.expert,
                store,
                visit.physical_layer,
                expert,
                &spans,
                &mut self.stats,
                &mut check,
            )?
        } else {
            self.stats.misses += 1;
            for (s, base) in spans.iter().zip([0, first]) {
                store.stream(&s.tensor, s.offset, s.bytes, |offset, data| {
                    check()?;
                    let at = base
                        .checked_add(
                            usize::try_from(offset).map_err(|_| "expert offset exceeds usize")?,
                        )
                        .ok_or("expert offset overflow")?;
                    self.stats.read_bytes += data.len() as u64;
                    self.expert.write(at, data).map_err(|e| e.0)?;
                    self.stats.uploaded_bytes += data.len() as u64;
                    self.stats.upload_calls += 1;
                    Ok(())
                })?;
            }
            (
                self.expert.view(0, first).map_err(|e| e.0)?,
                self.expert.view(first, second).map_err(|e| e.0)?,
            )
        };
        let scale = |part: &str| -> Result<Option<DevBuf>> {
            if visit.physical_layer >= 20 {
                return Ok(None);
            }
            Ok(Some(
                self.tensor(&format!(
                    "blk.{}.ffn_{part}_exps.scale",
                    visit.physical_layer
                ))?
                .view(expert * 4, 4)
                .map_err(|e| e.0)?,
            ))
        };
        let pair = ExpertDevice {
            gate_up,
            down,
            gate_scale: scale("gate_up")?,
            down_scale: scale("down")?,
        };
        check()?;
        f(&pair)
    }
}
