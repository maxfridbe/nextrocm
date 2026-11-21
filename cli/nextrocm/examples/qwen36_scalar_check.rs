//! Scalar-fast canary. Parent alone runs GPU; --cpu-only never loads a runtime.
//! qwen36_scalar_check MODEL [--gpu 0] [--reps 100]
//! Native execution requires NS_PROFILE=gpu and the parent-isolated runtime.
//! Fixed gates: bitwise baseline equality plus independent F64 CPU dot error
//! <= 1e-5 + 4e-6 * sum(abs(products)); never relax after native results.
//! Logical pack/CPU helpers copied from qwen36_batch_check (not HIP readers).
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::{GType, Gguf};
use std::{error::Error, ffi::c_void, fs::File, os::unix::fs::FileExt, path::Path, sync::Arc};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type P = *mut c_void;
type Scalar = unsafe extern "C" fn(P, P, i32, i32, i32, *mut f32, *mut f32) -> i32;
const GUARD: usize = 256;
const MARK: u8 = 0xa5;
const MAX_PAIRS: usize = 256;
const EVENT_RING: i64 = 4096;
const HELP: &str = "qwen36_scalar_check MODEL [--gpu 0] [--reps 100] | --cpu-only";

fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| "size overflow".into())
}
fn half(b: &[u8]) -> f32 {
    let h = u16::from_le_bytes([b[0], b[1]]);
    let e = (h >> 10) & 31;
    let m = h & 1023;
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-24)
    } else if e == 31 {
        if m == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1024 + m) as f32 * 2f32.powi(e as i32 - 25)
    };
    if h & 0x8000 != 0 {
        -v
    } else {
        v
    }
}
fn scale_bit(group: usize, bit: usize, minimum: bool) -> (usize, usize) {
    if group < 4 {
        (group + if minimum { 4 } else { 0 }, bit)
    } else if bit < 4 {
        (group + 4, bit + if minimum { 4 } else { 0 })
    } else {
        (if minimum { group } else { group - 4 }, bit + 2)
    }
}
fn set_bit(b: &mut [u8], byte: usize, bit: usize, value: usize) {
    b[byte] |= ((value & 1) as u8) << bit;
}
fn unpack_scale(b: &[u8], group: usize, minimum: bool) -> i32 {
    (0..6)
        .map(|k| {
            let (at, bit) = scale_bit(group, k, minimum);
            i32::from((b[at] >> bit) & 1) << k
        })
        .sum()
}

// Standalone Rust block decoder; no inclusion/call of HIP's quant helpers.
fn decode(raw: &[u8], ty: GType) -> Result<Vec<f32>> {
    let (values, bytes) = ty.block();
    if raw.len() % bytes != 0 {
        return Err("truncated block".into());
    }
    let mut out = Vec::with_capacity(raw.len() / bytes * values);
    for b in raw.chunks_exact(bytes) {
        match ty.code() {
            0 => out.push(f32::from_le_bytes(b.try_into()?)),
            1 => out.push(half(b)),
            8 => {
                for &q in &b[2..34] {
                    out.push(half(b) * (q as i8 as f32));
                }
            }
            12 | 13 => {
                let five = ty.code() == 13;
                for group in 0..8 {
                    for lane in 0..32 {
                        let lo = (b[if five { 48 } else { 16 } + group / 2 * 32 + lane]
                            >> (group % 2 * 4))
                            & 15;
                        let hi = if five {
                            ((b[16 + lane] >> group) & 1) << 4
                        } else {
                            0
                        };
                        out.push(
                            half(b)
                                * unpack_scale(&b[4..16], group, false) as f32
                                * (lo | hi) as f32
                                - half(&b[2..]) * unpack_scale(&b[4..16], group, true) as f32,
                        );
                    }
                }
            }
            14 => {
                for region in 0..2 {
                    for group in 0..4 {
                        for lane in 0..32 {
                            let lo =
                                (b[region * 64 + group % 2 * 32 + lane] >> (group / 2 * 4)) & 15;
                            let hi = (b[128 + region * 32 + lane] >> (group * 2)) & 3;
                            let scale = b[192 + region * 8 + group * 2 + lane / 16] as i8;
                            out.push(
                                half(&b[208..])
                                    * scale as f32
                                    * (i32::from(lo | hi << 4) - 32) as f32,
                            );
                        }
                    }
                }
            }
            _ => return Err("unsupported CPU weight type".into()),
        }
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite decoded weight".into());
    }
    Ok(out)
}
struct Matrix {
    name: String,
    ty: GType,
    rows: usize,
    cols: usize,
    raw: Vec<u8>,
    weights: Vec<f32>,
}
fn fixture(ty: GType) -> Result<Matrix> {
    let rows = 17;
    let cols = match ty.code() {
        0 | 1 => 259,
        8 => 96,
        _ => 768,
    };
    let (block, bytes) = ty.block();
    let mut raw = vec![0u8; rows * cols / block * bytes];
    let mut expected = Vec::with_capacity(rows * cols);
    for (index, b) in raw.chunks_exact_mut(bytes).enumerate() {
        match ty.code() {
            0 => {
                let v = ((index * 13 % 127) as f32 - 63.0) / 64.0;
                b.copy_from_slice(&v.to_le_bytes());
                expected.push(v);
            }
            1 => {
                // Exact dyadic values, signed zero and smallest half subnormal.
                let bits: [u16; 8] = [
                    0x0000, 0x8000, 0x0001, 0x3800, 0xbc00, 0x3400, 0xc000, 0x3e00,
                ];
                let values = [0.0, -0.0, 2f32.powi(-24), 0.5, -1.0, 0.25, -2.0, 1.5];
                b.copy_from_slice(&bits[index % 8].to_le_bytes());
                expected.push(values[index % 8]);
            }
            8 => {
                b[..2].copy_from_slice(&0x2800u16.to_le_bytes()); // 1/32
                for lane in 0..32 {
                    let q = ((index * 17 + lane * 11) % 256) as i32 - 128;
                    b[2 + lane] = q as i8 as u8;
                    expected.push(q as f32 / 32.0);
                }
            }
            12 | 13 => {
                b[..2].copy_from_slice(&0x2400u16.to_le_bytes()); // 1/64
                b[2..4].copy_from_slice(&0x2000u16.to_le_bytes()); // 1/128
                for group in 0..8 {
                    let scale = (index * 3 + group * 7) % 64;
                    let min = (index * 11 + group * 13) % 64;
                    for bit in 0..6 {
                        for (minimum, value) in [(false, scale), (true, min)] {
                            let (at, shift) = scale_bit(group, bit, minimum);
                            set_bit(b, 4 + at, shift, value >> bit);
                        }
                    }
                    for lane in 0..32 {
                        let q = (index + group * 3 + lane) % if ty.code() == 13 { 32 } else { 16 };
                        for bit in 0..if ty.code() == 13 { 5 } else { 4 } {
                            let at = if bit == 4 {
                                16 + lane
                            } else {
                                (if ty.code() == 13 { 48 } else { 16 }) + group / 2 * 32 + lane
                            };
                            let shift = if bit == 4 { group } else { group % 2 * 4 + bit };
                            set_bit(b, at, shift, q >> bit);
                        }
                        expected.push((scale * q) as f32 / 64.0 - min as f32 / 128.0);
                    }
                }
            }
            14 => {
                b[208..].copy_from_slice(&0x2400u16.to_le_bytes());
                for group in 0..16 {
                    b[192 + group] = (((index * 5 + group * 17) % 255) as i32 - 127) as i8 as u8;
                }
                for lane in 0..256 {
                    let q = (index * 7 + lane * 11) % 64;
                    let region = lane / 128;
                    let group = lane % 128 / 32;
                    let l = lane % 32;
                    for bit in 0..6 {
                        let (at, shift) = if bit < 4 {
                            (region * 64 + group % 2 * 32 + l, group / 2 * 4 + bit)
                        } else {
                            (128 + region * 32 + l, group * 2 + bit - 4)
                        };
                        set_bit(b, at, shift, q >> bit);
                    }
                    let s = ((index * 5 + lane / 16 * 17) % 255) as i32 - 127;
                    expected.push(s as f32 * (q as i32 - 32) as f32 / 64.0);
                }
            }
            _ => return Err("fixture unsupported type".into()),
        }
    }
    let decoded = decode(&raw, ty)?;
    if decoded.len() != expected.len()
        || decoded
            .iter()
            .zip(&expected)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(format!("independent logical pack/decode mismatch {ty:?}").into());
    }
    Ok(Matrix {
        name: format!("logical-{ty:?}"),
        ty,
        rows,
        cols,
        raw,
        weights: expected,
    })
}

struct Guarded {
    owner: DevBuf,
    view: DevBuf,
    original: Vec<u8>,
}
impl Guarded {
    fn new(gpu: &Arc<Gpu>, bytes: &[u8]) -> Result<Self> {
        let mut original = vec![MARK; GUARD + bytes.len() + GUARD];
        original[GUARD..GUARD + bytes.len()].copy_from_slice(bytes);
        let owner = DevBuf::new(gpu, original.len())?;
        owner.write(0, &original)?;
        let view = owner.view(GUARD, bytes.len())?;
        Ok(Self {
            owner,
            view,
            original,
        })
    }
    fn read(&self) -> Result<Vec<u8>> {
        let mut raw = vec![0; self.original.len()];
        self.owner.read(0, &mut raw)?;
        Ok(raw)
    }
    fn unchanged(&self) -> Result<()> {
        if self.read()? != self.original {
            return Err("input/redzone modified".into());
        }
        Ok(())
    }
    fn reset(&self) -> Result<()> {
        self.owner.write(0, &self.original)?;
        Ok(())
    }
    fn output(&self, n: usize, rows: usize, stride: usize) -> Result<Vec<f32>> {
        let raw = self.read()?;
        for (i, (&got, &old)) in raw.iter().zip(&self.original).enumerate() {
            let inside = i >= GUARD
                && i < raw.len() - GUARD
                && (i - GUARD) / 4 / stride < n
                && (i - GUARD) / 4 % stride < rows;
            if !inside && got != old {
                return Err(format!("output redzone/padding changed at byte {i}").into());
            }
        }
        let mut values = Vec::with_capacity(n * rows);
        for input in 0..n {
            for row in 0..rows {
                let at = GUARD + (input * stride + row) * 4;
                let v = f32::from_le_bytes(raw[at..at + 4].try_into()?);
                if !v.is_finite() {
                    return Err("nonfinite/unwritten output".into());
                }
                values.push(v);
            }
        }
        Ok(values)
    }
}
fn validate(scalar: &[f32], candidate: &[f32], expected: &[(f64, f64)]) -> Result<()> {
    if scalar.is_empty() || scalar.len() != candidate.len() || scalar.len() > expected.len() {
        return Err("comparison shape mismatch".into());
    }
    for (i, ((s, b), (e, tol))) in scalar.iter().zip(candidate).zip(expected).enumerate() {
        if !s.is_finite()
            || !b.is_finite()
            || !e.is_finite()
            || !tol.is_finite()
            || s.to_bits() != b.to_bits()
            || (f64::from(*s) - e).abs() > *tol
            || (f64::from(*b) - e).abs() > *tol
        {
            return Err(format!(
                "index={i} scalar={s} candidate={b} cpu={e} tolerance={tol} bits={:08x}/{:08x}",
                s.to_bits(),
                b.to_bits()
            )
            .into());
        }
    }
    Ok(())
}
fn actual(f: &Gguf) -> Result<Vec<Matrix>> {
    if f.architecture() != "qwen35moe" {
        return Err("expected qwen35moe GGUF".into());
    }
    let first = f
        .tensors
        .iter()
        .filter(|t| t.shape.len() == 3)
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")?
                .strip_suffix(".ffn_gate_exps.weight")?
                .parse::<usize>()
                .ok()
        })
        .min()
        .ok_or("no expert bank")?;
    let mut names: Vec<String> = ["gate", "up", "down"]
        .iter()
        .map(|role| format!("blk.{first}.ffn_{role}_exps.weight"))
        .collect();
    // Full resident router projection is bounded and used on every MoE layer.
    names.push(format!("blk.{first}.ffn_gate_inp.weight"));
    let mut result = Vec::new();
    for name in names {
        let t = f.tensor(&name).ok_or("missing real matrix")?;
        if ![2, 3].contains(&t.shape.len()) || ![0, 1, 8, 12, 13, 14].contains(&t.ty.code()) {
            return Err("unsupported real matrix geometry/type".into());
        }
        let cols = usize::try_from(t.shape[t.shape.len() - 1])?;
        let rows = usize::try_from(t.shape[t.shape.len() - 2])?;
        let experts = if t.shape.len() == 3 {
            usize::try_from(t.shape[0])?
        } else {
            1
        };
        let count = mul(rows, cols)?;
        if rows == 0
            || cols == 0
            || experts == 0
            || count > 16 * 1024 * 1024
            || count > i32::MAX as usize
        {
            return Err("real matrix outside canary bounds".into());
        }
        let row_bytes =
            usize::try_from(t.ty.bytes(cols as u64).ok_or("real packed row alignment")?)?;
        let bytes = mul(rows, row_bytes)?;
        if u64::try_from(mul(bytes, experts)?)? != t.bytes {
            return Err("real tensor extent mismatch".into());
        }
        let file = File::open(f.paths.get(t.shard).ok_or("missing shard")?)?;
        let file_bytes = file.metadata()?.len();
        if t.offset
            .checked_add(t.bytes)
            .is_none_or(|end| end > file_bytes)
        {
            return Err("tensor outside file".into());
        }
        let mut raw = vec![0; bytes];
        file.read_exact_at(&mut raw, t.offset)?;
        let weights = decode(&raw, t.ty)?;
        if weights.len() != count {
            return Err("CPU decoded shape mismatch".into());
        }
        println!("GGUF tensor={name} source_shape={:?} expert={} rows={rows} cols={cols} type={:?} bytes={bytes} whole_matrix=true",t.shape,if experts>1 {"0"}else{"resident"},t.ty);
        result.push(Matrix {
            name,
            ty: t.ty,
            rows,
            cols,
            raw,
            weights,
        });
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logical_quant_pack_matches_cpu_decoder() {
        for ty in [
            GType::F32,
            GType::F16,
            GType::Q8_0,
            GType::Q4K,
            GType::Q5K,
            GType::Q6K,
        ] {
            fixture(ty).expect("logical fixture");
        }
    }
    #[test]
    fn half_special_values() {
        assert_eq!(half(&1u16.to_le_bytes()), 2f32.powi(-24));
        assert_eq!(half(&0x8000u16.to_le_bytes()).to_bits(), (-0f32).to_bits());
        assert!(half(&0x7c00u16.to_le_bytes()).is_infinite());
        assert!(half(&0x7e00u16.to_le_bytes()).is_nan());
    }
    #[test]
    fn strict_gate_does_not_hide_one_ulp() {
        assert!(validate(&[1.0], &[f32::from_bits(1f32.to_bits() + 1)], &[(1.0, 1.0)]).is_err());
        assert!(validate(&[1.0], &[1.0], &[(2.0, 0.1)]).is_err());
    }
}

type Fast = unsafe extern "C" fn(P, P, i32, i32, i32, *mut f32, *mut f32, i32) -> i32;
enum Call {
    Baseline(Scalar),
    Fast(Fast, i32),
}
struct Variant {
    name: &'static str,
    call: Call,
}
fn status(rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(api()?.error().into())
    }
}
fn invoke(gpu: &Gpu, v: &Variant, m: &Matrix, w: &DevBuf, x: &DevBuf, y: &DevBuf) -> Result<()> {
    w.bounds(0, m.raw.len())?;
    x.bounds(0, mul(m.cols, 4)?)?;
    y.bounds(0, mul(m.rows, 4)?)?;
    if m.rows == 0 || m.cols == 0 || mul(m.rows, m.cols)? > i32::MAX as usize {
        return Err("scalar index extent".into());
    }
    // SAFETY: scalar_fast and nq35_mv share this ABI; extents checked above,
    // all pointers GPU-backed, aligned, nonaliasing and retained until sync.
    status(unsafe {
        match v.call {
            Call::Baseline(f) => f(
                gpu.raw(),
                w.ptr(),
                m.ty.code() as i32,
                m.rows as i32,
                m.cols as i32,
                x.fp(),
                y.fp(),
            ),
            Call::Fast(f, id) => f(
                gpu.raw(),
                w.ptr(),
                m.ty.code() as i32,
                m.rows as i32,
                m.cols as i32,
                x.fp(),
                y.fp(),
                id,
            ),
        }
    })
}
fn oracle(m: &Matrix, x: &[f32]) -> Vec<(f64, f64)> {
    m.weights
        .chunks_exact(m.cols)
        .map(|row| {
            let mut sum = 0.0;
            let mut abs = 0.0;
            for (&w, &x) in row.iter().zip(x) {
                let v = f64::from(w) * f64::from(x);
                sum += v;
                abs += v.abs();
            }
            (sum, 1e-5 + 4e-6 * abs)
        })
        .collect()
}
fn input(cols: usize, seed: usize) -> Vec<f32> {
    (0..cols)
        .map(|i| match seed {
            0 => ((i * 17 % 251) as f32 - 125.0) / 128.0,
            1 => {
                if i % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            }
            2 => {
                if i % 257 == 0 {
                    0.5
                } else {
                    0.0
                }
            }
            _ => {
                if i % 2 == 0 {
                    0.0
                } else {
                    -0.0
                }
            }
        })
        .collect()
}
fn stamp(gpu: &Gpu) -> Result<i64> {
    let a = api()?;
    let mut ticket = -1;
    // SAFETY: live GPU, valid out pointer, runtime-owned ring event.
    status(unsafe { (a.stamp)(gpu.raw(), &mut ticket) })?;
    if ticket < 0 {
        return Err("negative event ticket".into());
    }
    Ok(ticket)
}
fn duration_ms(ns: f64) -> Result<f64> {
    if !ns.is_finite() || ns < 0.0 {
        return Err("nonfinite/negative event duration".into());
    }
    Ok(ns / 1e6) // ns_elapsed returns ns, not ms.
}
fn live_pair(start: i64, end: i64, newest: i64) -> Result<()> {
    if start < 0 || end <= start || newest < end || newest - start >= EVENT_RING {
        return Err("stale/invalid event pair".into());
    }
    Ok(())
}
struct EventTimes {
    pending: Vec<(usize, i64, i64)>,
    samples: Vec<Vec<f64>>,
}
impl EventTimes {
    fn new(variants: usize) -> Self {
        Self {
            pending: Vec::with_capacity(MAX_PAIRS),
            samples: vec![Vec::new(); variants],
        }
    }
    fn record(&mut self, gpu: &Gpu, index: usize, start: i64, end: i64) -> Result<()> {
        live_pair(start, end, end)?;
        self.pending.push((index, start, end));
        if self.pending.len() >= MAX_PAIRS {
            self.flush(gpu)?;
        }
        Ok(())
    }
    fn flush(&mut self, gpu: &Gpu) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let newest = stamp(gpu)?;
        let a = api()?;
        for &(index, start, end) in &self.pending {
            live_pair(start, end, newest)?;
            let mut ns = 0.0;
            // SAFETY: checked live ring tickets; ns_elapsed waits for end, and
            // writes ns to the valid local. No allocations/copies in interval.
            status(unsafe { (a.elapsed)(gpu.raw(), start, end, &mut ns) })?;
            self.samples[index].push(duration_ms(ns)?);
        }
        self.pending.clear();
        Ok(())
    }
    fn report(&self, case: &str, variants: &[Variant]) -> Result<()> {
        let base = &self.samples[0];
        if base.is_empty() {
            return Err("no event samples".into());
        }
        let baseline = base.iter().sum::<f64>() / base.len() as f64;
        for (v, values) in variants.iter().zip(&self.samples) {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            if sorted.len() != base.len() {
                return Err("event sample count mismatch".into());
            }
            let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
            let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0;
            let ratio = if mean > 0.0 && baseline > 0.0 {
                format!("{:.6}", baseline / mean)
            } else {
                "unresolved_zero_duration".into()
            };
            println!("event_timing case={case} variant={} reps={} mean_ms={mean:.9} median_ms={median:.9} min_ms={:.9} max_ms={:.9} baseline_over_variant={ratio} warm=true ns_elapsed_units=ns stream_elapsed_includes_submission_gaps=true",v.name,sorted.len(),sorted[0],sorted[sorted.len()-1]);
        }
        Ok(())
    }
}
fn check_matrix(gpu: &Arc<Gpu>, variants: &[Variant], m: &Matrix, reps: usize) -> Result<()> {
    println!(
        "case={} type={:?} rows={} cols={} bytes={} cpu_gate=1e-5+4e-6sumabs bitwise_gate=true",
        m.name,
        m.ty,
        m.rows,
        m.cols,
        m.raw.len()
    );
    let w = Guarded::new(gpu, &m.raw)?;
    let output: Vec<Guarded> = (0..variants.len())
        .map(|_| Guarded::new(gpu, &vec![MARK; mul(m.rows + 3, 4)?]))
        .collect::<Result<_>>()?;
    // Dense, cancellation, sparse/block-boundary, signed-zero inputs.
    for seed in 0..4 {
        let values = input(m.cols, seed);
        let expected = oracle(m, &values);
        let raw: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let x = Guarded::new(gpu, &raw)?;
        for (v, y) in variants.iter().zip(&output) {
            y.reset()?;
            invoke(gpu, v, m, &w.view, &x.view, &y.view)?;
        }
        gpu.sync()?;
        let baseline = output[0].output(1, m.rows, m.rows + 3)?;
        for (v, y) in variants.iter().zip(&output) {
            let candidate = y.output(1, m.rows, m.rows + 3)?;
            validate(&baseline, &candidate, &expected)
                .map_err(|e| format!("case={} seed={seed} variant={}: {e}", m.name, v.name))?;
            println!("PASS case={} seed={seed} variant={} finite=true cpu=true baseline_bitwise=true redzones=true",m.name,v.name);
        }
        x.unchanged()?;
        w.unchanged()?;
    }
    let values = input(m.cols, 0);
    let expected = oracle(m, &values);
    let raw: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
    let x = Guarded::new(gpu, &raw)?;
    for (v, y) in variants.iter().zip(&output) {
        invoke(gpu, v, m, &w.view, &x.view, &y.view)?;
    }
    gpu.sync()?;
    let mut events = EventTimes::new(variants.len());
    for rep in 0..reps {
        for at in 0..variants.len() {
            let index = (at + rep) % variants.len();
            let start = stamp(gpu)?;
            invoke(
                gpu,
                &variants[index],
                m,
                &w.view,
                &x.view,
                &output[index].view,
            )?;
            let end = stamp(gpu)?;
            events.record(gpu, index, start, end)?;
        }
    }
    events.flush(gpu)?;
    gpu.sync()?;
    let baseline = output[0].output(1, m.rows, m.rows + 3)?;
    for y in &output {
        validate(&baseline, &y.output(1, m.rows, m.rows + 3)?, &expected)?;
    }
    x.unchanged()?;
    w.unchanged()?;
    events.report(&m.name, variants)?;
    Ok(())
}

fn kernels() -> Result<(Vec<Variant>, Fast)> {
    let a = api()?;
    let scalar = a.symbol("nq35_mv");
    let fast = a.symbol("nq35_mv_fast");
    if scalar.is_null() || fast.is_null() {
        return Err("isolated runtime must export nq35_mv and nq35_mv_fast".into());
    }
    // SAFETY: exact qwen35.h/scalar_fast.h signatures; process-lived API handle.
    let (scalar, fast) = unsafe {
        (
            std::mem::transmute::<P, Scalar>(scalar),
            std::mem::transmute::<P, Fast>(fast),
        )
    };
    Ok((
        vec![
            Variant {
                name: "baseline",
                call: Call::Baseline(scalar),
            },
            Variant {
                name: "specialization",
                call: Call::Fast(fast, 1),
            },
            Variant {
                name: "shuffle",
                call: Call::Fast(fast, 2),
            },
            Variant {
                name: "combined",
                call: Call::Fast(fast, 3),
            },
        ],
        fast,
    ))
}
fn invalid(gpu: &Arc<Gpu>, fast: Fast) -> Result<()> {
    let w = Guarded::new(gpu, &vec![MARK; 4096])?;
    let x = Guarded::new(gpu, &vec![MARK; 4096])?;
    let y = Guarded::new(gpu, &vec![MARK; 4096])?;
    for variant in 1..=3 {
        for case in 0..19 {
            let (mut g, mut wp, mut xp, mut yp) =
                (gpu.raw(), w.view.ptr(), x.view.fp(), y.view.fp());
            let (mut rows, mut cols, mut ty, mut id) = (2, 256, 0, variant);
            let label = match case {
                0 => {
                    g = std::ptr::null_mut();
                    "null_gpu"
                }
                1 => {
                    wp = std::ptr::null_mut();
                    "null_weights"
                }
                2 => {
                    xp = std::ptr::null_mut();
                    "null_input"
                }
                3 => {
                    yp = std::ptr::null_mut();
                    "null_output"
                }
                4 => {
                    rows = 0;
                    "zero_rows"
                }
                5 => {
                    rows = -1;
                    "negative_rows"
                }
                6 => {
                    cols = 0;
                    "zero_cols"
                }
                7 => {
                    cols = -1;
                    "negative_cols"
                }
                8 => {
                    ty = 2;
                    "unsupported_type"
                }
                9 => {
                    id = 0;
                    "zero_variant"
                }
                10 => {
                    id = 4;
                    "variant_above3"
                }
                11 => {
                    wp = wp.cast::<u8>().wrapping_add(1).cast();
                    "misaligned_weights"
                }
                12 => {
                    xp = xp.cast::<u8>().wrapping_add(1).cast();
                    "misaligned_input"
                }
                13 => {
                    yp = yp.cast::<u8>().wrapping_add(1).cast();
                    "misaligned_output"
                }
                14 => {
                    rows = 65536;
                    cols = 65536;
                    "rows_cols_overflow"
                }
                15 => {
                    rows = 1;
                    cols = i32::MAX;
                    "cols_overflow"
                }
                16 => {
                    ty = 8;
                    cols = 255;
                    "q8_alignment"
                }
                17 => {
                    ty = 12;
                    cols = 255;
                    "k_alignment"
                }
                _ => {
                    ty = 8;
                    rows = 1;
                    cols = 2147483392;
                    "q8_packed_overflow"
                }
            };
            // SAFETY: guards explicitly documented in scalar_fast.h/hip; all
            // backing buffers stay allocated. Invalid calls must not launch.
            let rc = unsafe { fast(g, wp, ty, rows, cols, xp, yp, id) };
            gpu.sync()?;
            if rc == 0 || api()?.error().is_empty() {
                return Err(
                    format!("invalid parameter accepted variant={variant} case={label}").into(),
                );
            }
            w.unchanged()?;
            x.unchanged()?;
            y.unchanged()?;
            println!("PASS invalid={label} variant={variant} ns_fail=true memory_unchanged=true");
        }
    }
    Ok(())
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or(HELP)?;
    if model == "--help" {
        println!("{HELP}");
        return Ok(());
    }
    let mut gpu_index = 0;
    let mut reps = 100;
    while let Some(key) = args.next() {
        let value = args
            .next()
            .ok_or("option value missing")?
            .parse::<usize>()?;
        match key.as_str() {
            "--gpu" => gpu_index = value,
            "--reps" => reps = value,
            _ => return Err("unknown option".into()),
        }
    }
    if !(1..=10000).contains(&reps) {
        return Err("reps must be 1..10000".into());
    }
    let mut matrices = Vec::new();
    for ty in [
        GType::F32,
        GType::F16,
        GType::Q8_0,
        GType::Q4K,
        GType::Q5K,
        GType::Q6K,
    ] {
        let m = fixture(ty)?;
        println!(
            "PASS CPU independent pack/decode type={ty:?} values={}",
            m.weights.len()
        );
        matrices.push(m);
    }
    if model == "--cpu-only" {
        println!("PASS CPU-only no runtime loaded");
        return Ok(());
    }
    if std::env::var("NS_PROFILE").as_deref() != Ok("gpu") {
        return Err("set NS_PROFILE=gpu before opening GPU; event timing is mandatory".into());
    }
    matrices.extend(actual(&Gguf::open(Path::new(&model))?)?);
    let (variants, fast) = kernels()?;
    let gpu = Gpu::open(gpu_index)?;
    if gpu.raw().is_null() {
        return Err("native GPU required".into());
    }
    println!("runtime={} gpu={gpu_index} timing=GPU_events max_pending_pairs={MAX_PAIRS} ring={EVENT_RING} host_wall_timing=false",api()?.path().display());
    invalid(&gpu, fast)?;
    for m in &matrices {
        check_matrix(&gpu, &variants, m, reps)?;
    }
    println!(
        "PASS matrices={} variants=3_plus_baseline input_patterns=4 invalid_calls=57",
        matrices.len()
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod event_tests {
    use super::*;
    #[test]
    fn units_are_ns_and_invalid_durations_rejected() {
        assert_eq!(duration_ms(2_500_000.0).unwrap(), 2.5);
        assert!(duration_ms(f64::NAN).is_err());
        assert!(duration_ms(f64::INFINITY).is_err());
        assert!(duration_ms(-1.0).is_err());
    }
    #[test]
    fn ring_checks_reject_stale_and_reversed_tickets() {
        assert!(live_pair(100, 101, 100 + EVENT_RING - 1).is_ok());
        assert!(live_pair(100, 101, 100 + EVENT_RING).is_err());
        assert!(live_pair(-1, 1, 2).is_err());
        assert!(live_pair(5, 4, 7).is_err());
        assert!(live_pair(5, 7, 6).is_err());
    }
    #[test]
    fn cpu_dot_detects_wrong_arithmetic() {
        let m = Matrix {
            name: "cpu".into(),
            ty: GType::F32,
            rows: 2,
            cols: 2,
            raw: Vec::new(),
            weights: vec![1.0, -1.0, 0.5, 2.0],
        };
        let expected = oracle(&m, &[3.0, 4.0]);
        assert_eq!(expected[0].0, -1.0);
        assert_eq!(expected[1].0, 9.5);
        validate(&[-1.0, 9.5], &[-1.0, 9.5], &expected).unwrap();
        assert!(validate(&[-1.0, 9.5], &[-1.0, 9.0], &expected).is_err());
    }
}
