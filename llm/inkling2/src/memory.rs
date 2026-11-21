//! Checked candidate residency, not measured fit or a kernel-allocation contract.
//! F32 KV/history baseline; one expert pair streamed at a time. No host mirror.
use crate::{plan::ExecutionPlan, store::STAGING_BYTES, weights::Weights, Result};
#[derive(Clone, Debug)]
pub struct Estimate {
    pub resident_weights: u64,
    pub kv: u64,
    pub convolution_history: u64,
    pub expert_pair: u64,
    pub expert_cache: u64,
    pub expert_host_pair: u64,
    pub activation_reserve: u64,
    pub tracked: u64,
    pub driver_margin: u64,
    pub device_total: u64,
    pub host_staging: u64,
    pub host_reserve: u64,
    pub prospective_shared_physical: u64,
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| "memory size overflow".into())
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| "memory size overflow".into())
}
impl Estimate {
    pub fn text(
        w: &Weights<'_>,
        p: &ExecutionPlan,
        prefill_rows: usize,
        host_available: u64,
        device_free: u64,
    ) -> Result<Self> {
        Self::text_with_expert_mode(
            w,
            p,
            prefill_rows,
            host_available,
            device_free,
            crate::expert_cache::ExpertMode::Stream,
        )
    }
    pub fn text_with_expert_mode(
        w: &Weights<'_>,
        p: &ExecutionPlan,
        prefill_rows: usize,
        host_available: u64,
        device_free: u64,
        mode: crate::expert_cache::ExpertMode,
    ) -> Result<Self> {
        if prefill_rows == 0 || prefill_rows > 32 || prefill_rows > p.context() {
            return Err("planning prefill rows must be 1..min(32,context)".into());
        }
        let resident_weights = w.non_expert_bytes();
        let kv = p.kv_storage_bytes(4)?;
        // Four independent convolution states per visit: K,V,post-attention,post-MLP;
        // three previous inputs for width-four causal convolution (merged source).
        let convolution_history = 48 * 3 * (2 * 8 * 128 + 2 * 2304) * 4;
        let expert_pair = w.largest_expert_pair_bytes();
        // Provisional explicit workspace reserve: 128 hidden-width F32 vectors per
        // token, plus full attention scores and sink, and 16 MiB kernel scratch.
        // Actual host/kernel implementation must fit this or revise/revalidate.
        let activation_reserve = add(
            add(
                mul(prefill_rows as u64, 128 * 2304 * 4)?,
                mul(
                    mul(prefill_rows as u64, 24)?,
                    mul(add(p.context() as u64, 1)?, 4)?,
                )?,
            )?,
            16 * 1024 * 1024,
        )?;
        let (expert_cache, expert_host_pair) = mode.extra_bytes(expert_pair)?;
        let tracked = [
            expert_cache,
            kv,
            convolution_history,
            expert_pair,
            activation_reserve,
        ]
        .into_iter()
        .try_fold(resident_weights, add)?;
        let driver_margin = 1024 * 1024 * 1024;
        let device_total = add(tracked, driver_margin)?;
        let host_staging = add(STAGING_BYTES as u64, expert_host_pair)?;
        let host_reserve = 4 * 1024 * 1024 * 1024;
        let prospective_shared_physical = add(add(device_total, host_staging)?, host_reserve)?;
        if tracked >= 7_500_000_000 || device_total >= 8_000_000_000 || device_total > device_free {
            return Err("candidate device budget exceeds strict cap or current free memory".into());
        }
        if prospective_shared_physical > host_available {
            return Err("candidate UMA budget exceeds host MemAvailable after reserve".into());
        }
        Ok(Self {
            resident_weights,
            kv,
            convolution_history,
            expert_pair,
            expert_cache,
            expert_host_pair,
            activation_reserve,
            tracked,
            driver_margin,
            device_total,
            host_staging,
            host_reserve,
            prospective_shared_physical,
        })
    }
}
