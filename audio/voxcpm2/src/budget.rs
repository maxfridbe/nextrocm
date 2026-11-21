//! Admission against exact native live allocations plus bounded stage work.
//! This counts tensor allocations and reserves 512 MiB for the HIP runtime;
//! parent validation must also measure driver/process memory, independently.
use nextrocm_audio::{Error, Result};
use nextrocm_engine_core::Gpu;
use std::ffi::c_void;

pub const CAP: usize = 8_000_000_000;
pub const RESERVE: usize = 512 << 20;
pub fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, v| {
        n.checked_mul(*v)
            .ok_or_else(|| Error("VoxCPM2 shape overflow".into()))
    })
}
pub fn sum(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(0usize, |n, v| {
        n.checked_add(*v)
            .ok_or_else(|| Error("VoxCPM2 size overflow".into()))
    })
}
pub fn stats(gpu: &Gpu) -> Result<(u64, u64)> {
    let api = nextrocm_engine_core::api()?;
    let symbol = api.symbol("nr_allocation_stats");
    if symbol.is_null() {
        return Err(Error(
            "VoxCPM2 requires native allocation accounting".into(),
        ));
    }
    type Stats = unsafe extern "C" fn(*mut c_void, *mut u64, *mut u64) -> i32;
    // SAFETY: exact ABI from kernels/ns/ns.hip; core owns the library lifetime.
    let call: Stats = unsafe { std::mem::transmute(symbol) };
    let (mut current, mut peak) = (0, 0);
    // SAFETY: native handle and initialized outputs; generation is serialized.
    if unsafe { call(gpu.raw(), &mut current, &mut peak) } != 0 {
        return Err(Error(api.error()));
    }
    Ok((current, peak))
}
pub fn admit(gpu: &Gpu, extra: usize, stage: &str) -> Result<()> {
    let (current, _) = stats(gpu)?;
    let current = usize::try_from(current)
        .map_err(|_| Error("VoxCPM2 allocation counter overflow".into()))?;
    let needed = sum(&[current, extra, RESERVE])?;
    let (_, free) = gpu.memory()?;
    let free = free.ok_or_else(|| Error("VoxCPM2 requires free-device-memory telemetry".into()))?;
    if needed >= CAP || sum(&[extra, RESERVE])? as u64 > free {
        return Err(Error(format!(
            "VoxCPM2 {stage}: {current} live + {extra} planned + {RESERVE} reserve = {needed}; strict cap {CAP}, free {free}"
        )));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accounting_overflow_is_error() {
        assert!(product(&[usize::MAX, 2]).is_err());
        assert!(sum(&[usize::MAX, 1]).is_err());
        assert_eq!(product(&[3, 128, 16000, 2, 4]).unwrap(), 49_152_000);
    }
}
