//! Parent-serialized canary for batch.h. No GPU execution in --cpu-only/tests.
//! qwen36_batch_check MODEL [--gpu N] [--reps 10]
//! qwen36_batch_check --cpu-only
//! Fixed gates BEFORE native results: scalar/batch bitwise equality, and both
//! against an F64 CPU dot with abs error <= 1e-5 + 4e-6 * sum(abs(products)).
//! Logical quant pack fixtures produce expected weights independently of decode.
use nextrocm_engine_core::{api, DevBuf, Gpu};
use nextrocm_gguf::{GType, Gguf};
use std::{
    error::Error, ffi::c_void, fs::File, os::unix::fs::FileExt, path::Path, sync::Arc,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;
type P = *mut c_void;
type Scalar = unsafe extern "C" fn(P, P, i32, i32, i32, *mut f32, *mut f32) -> i32;
type Batch =
    unsafe extern "C" fn(P, P, i32, i32, i32, u64, *mut f32, i32, u64, *mut f32, u64) -> i32;
const NS: [usize; 10] = [1, 2, 3, 4, 5, 7, 8, 16, 63, 64];
const GUARD: usize = 256;
const MARK: u8 = 0xa5;
const HELP: &str = "qwen36_batch_check MODEL [--gpu N] [--reps 10] | --cpu-only";

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
struct Kernels {
    scalar: Scalar,
    batch: Batch,
}
impl Kernels {
    fn load() -> Result<Self> {
        let a = api()?;
        let s = a.symbol("nq35_mv");
        let b = a.symbol("nq35_mv_batch");
        if s.is_null() || b.is_null() {
            return Err("isolated runtime must export nq35_mv and nq35_mv_batch".into());
        }
        // SAFETY: exact qwen35.h/batch.h ABI; Api remains live for process lifetime.
        Ok(unsafe {
            Self {
                scalar: std::mem::transmute::<P, Scalar>(s),
                batch: std::mem::transmute::<P, Batch>(b),
            }
        })
    }
    fn check(rc: i32) -> Result<()> {
        if rc != 0 {
            Err(api()?.error().into())
        } else {
            Ok(())
        }
    }
}
fn oracle(m: &Matrix, x: &[f32], stride: usize) -> Vec<(f64, f64)> {
    let mut out = Vec::with_capacity(64 * m.rows);
    for input in 0..64 {
        for row in 0..m.rows {
            let mut sum = 0.0;
            let mut abs = 0.0;
            for col in 0..m.cols {
                let v =
                    f64::from(m.weights[row * m.cols + col]) * f64::from(x[input * stride + col]);
                sum += v;
                abs += v.abs();
            }
            out.push((sum, 1e-5 + 4e-6 * abs));
        }
    }
    out
}
fn validate(scalar: &[f32], batch: &[f32], expected: &[(f64, f64)]) -> Result<()> {
    if scalar.is_empty() || scalar.len() != batch.len() || scalar.len() > expected.len() {
        return Err("comparison shape mismatch".into());
    }
    for (i, ((s, b), (e, tol))) in scalar.iter().zip(batch).zip(expected).enumerate() {
        if !s.is_finite()
            || !b.is_finite()
            || !e.is_finite()
            || !tol.is_finite()
            || s.to_bits() != b.to_bits()
            || (f64::from(*s) - e).abs() > *tol
            || (f64::from(*b) - e).abs() > *tol
        {
            return Err(format!(
                "index={i} scalar={s} batch={b} cpu={e} tolerance={tol} bits={:08x}/{:08x}",
                s.to_bits(),
                b.to_bits()
            )
            .into());
        }
    }
    Ok(())
}
fn check_matrix(gpu: &Arc<Gpu>, k: &Kernels, m: &Matrix, reps: usize) -> Result<()> {
    let row_bytes = usize::try_from(m.ty.bytes(m.cols as u64).ok_or("unaligned row")?)?;
    let ws = row_bytes + 8;
    let xs = m.cols + 3;
    let ys = m.rows + 5;
    let mut packed = vec![MARK; m.rows * ws];
    for row in 0..m.rows {
        packed[row * ws..row * ws + row_bytes]
            .copy_from_slice(&m.raw[row * row_bytes..(row + 1) * row_bytes]);
    }
    let w = Guarded::new(gpu, &packed)?;
    let sw = Guarded::new(gpu, &m.raw)?;
    let mut x = vec![f32::from_bits(0x7fc12345); 64 * xs];
    for input in 0..64 {
        for col in 0..m.cols {
            x[input * xs + col] = (((input * 31 + col * 17) % 251) as f32 - 125.0) / 128.0;
        }
    }
    let xraw: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xb = Guarded::new(gpu, &xraw)?;
    let by = Guarded::new(gpu, &vec![MARK; 64 * ys * 4])?;
    let sy = Guarded::new(gpu, &vec![MARK; 64 * ys * 4])?;
    println!("case={} type={:?} rows={} cols={} w_stride_bytes={ws} x_stride_floats={xs} y_stride_floats={ys} oracle=f64_full",m.name,m.ty,m.rows,m.cols);
    let expected = oracle(m, &x, xs);
    let scalar = |n: usize| -> Result<()> {
        for input in 0..n {
            // SAFETY: guarded buffers allocated for 64 rows; matrix dimensions and
            // signed scalar row*col bounds validated; offsets are F32 aligned.
            Kernels::check(unsafe {
                (k.scalar)(
                    gpu.raw(),
                    sw.view.ptr(),
                    m.ty.code() as i32,
                    m.rows as i32,
                    m.cols as i32,
                    xb.view.fp().add(input * xs),
                    sy.view.fp().add(input * ys),
                )
            })?;
        }
        Ok(())
    };
    let batch = |n: usize| -> Result<()> {
        // SAFETY: exact batch.h ABI; extents include all stride padding, buffers
        // are distinct/aligned and live until the sync following each dispatch.
        Kernels::check(unsafe {
            (k.batch)(
                gpu.raw(),
                w.view.ptr(),
                m.ty.code() as i32,
                m.rows as i32,
                m.cols as i32,
                ws as u64,
                xb.view.fp(),
                n as i32,
                xs as u64,
                by.view.fp(),
                ys as u64,
            )
        })
    };
    for n in NS {
        by.reset()?;
        sy.reset()?;
        scalar(n)?;
        batch(n)?;
        gpu.sync()?;
        validate(
            &sy.output(n, m.rows, ys)?,
            &by.output(n, m.rows, ys)?,
            &expected,
        )?;
        println!(
            "PASS case={} n={n} scalar_bitwise=true cpu=true padding=true",
            m.name
        );
        if [1, 4, 16, 64].contains(&n) {
            let mut st = 0.0;
            let mut bt = 0.0;
            for rep in 0..reps {
                for mode in [rep % 2, 1 - rep % 2] {
                    gpu.sync()?;
                    let start = Instant::now();
                    if mode == 0 {
                        scalar(n)?;
                    } else {
                        batch(n)?;
                    }
                    gpu.sync()?;
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    if mode == 0 {
                        st += ms;
                    } else {
                        bt += ms;
                    }
                }
            }
            validate(
                &sy.output(n, m.rows, ys)?,
                &by.output(n, m.rows, ys)?,
                &expected,
            )?;
            println!("timing case={} n={n} reps={reps} scalar_ms={:.6} batch_ms={:.6} speedup={:.4} warm=true upload_readback_excluded=true",m.name,st/reps as f64,bt/reps as f64,st/bt);
        }
    }
    w.unchanged()?;
    sw.unchanged()?;
    xb.unchanged()?;
    // Also contiguous batch strides, guarding total extents independently.
    let compact: Vec<u8> = (0..4)
        .flat_map(|i| {
            x[i * xs..i * xs + m.cols]
                .iter()
                .flat_map(|v| v.to_le_bytes())
        })
        .collect();
    let cx = Guarded::new(gpu, &compact)?;
    let cy = Guarded::new(gpu, &vec![MARK; 4 * m.rows * 4])?;
    // SAFETY: contiguous weight/input/output dimensions and nonaliasing checked above.
    Kernels::check(unsafe {
        (k.batch)(
            gpu.raw(),
            sw.view.ptr(),
            m.ty.code() as i32,
            m.rows as i32,
            m.cols as i32,
            row_bytes as u64,
            cx.view.fp(),
            4,
            m.cols as u64,
            cy.view.fp(),
            m.rows as u64,
        )
    })?;
    gpu.sync()?;
    let out = cy.output(4, m.rows, m.rows)?;
    scalar(4)?;
    gpu.sync()?;
    // Scalar buffer still has n=64 data outside these rows; select active values directly.
    let all = sy.output(64, m.rows, ys)?;
    validate(&all[..4 * m.rows], &out, &expected)?;
    cx.unchanged()?;
    sw.unchanged()?;
    println!(
        "PASS case={} contiguous_n4=true inputs_unchanged=true",
        m.name
    );
    Ok(())
}

fn invalid(gpu: &Arc<Gpu>, k: &Kernels) -> Result<()> {
    let w = Guarded::new(gpu, &vec![MARK; 4096])?;
    let x = Guarded::new(gpu, &vec![MARK; 4096])?;
    let y = Guarded::new(gpu, &vec![MARK; 4096])?;
    for case in 0..23 {
        let (mut g, mut wp, mut xp, mut yp) = (gpu.raw(), w.view.ptr(), x.view.fp(), y.view.fp());
        let (mut ty, mut rows, mut cols, mut n) = (0, 2, 256, 2);
        let (mut ws, mut xs, mut ys) = (1024u64, 256u64, 2u64);
        let label = match case {
            0 => {
                g = std::ptr::null_mut();
                "null_gpu"
            }
            1 => {
                wp = std::ptr::null_mut();
                "null_w"
            }
            2 => {
                xp = std::ptr::null_mut();
                "null_x"
            }
            3 => {
                yp = std::ptr::null_mut();
                "null_y"
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
                n = 0;
                "zero_n"
            }
            9 => {
                n = 65;
                "n_over64"
            }
            10 => {
                ty = 2;
                "unsupported_type"
            }
            11 => {
                ws = 1020;
                "short_w_stride"
            }
            12 => {
                xs = 255;
                "short_x_stride"
            }
            13 => {
                ys = 1;
                "short_y_stride"
            }
            14 => {
                ws = 1025;
                "unaligned_w_stride"
            }
            15 => {
                wp = wp.cast::<u8>().wrapping_add(1).cast();
                "unaligned_w"
            }
            16 => {
                xp = xp.cast::<u8>().wrapping_add(1).cast();
                "unaligned_x"
            }
            17 => {
                yp = yp.cast::<u8>().wrapping_add(1).cast();
                "unaligned_y"
            }
            18 => {
                ws = u64::MAX - 3;
                "overflow_w_span"
            }
            19 => {
                xs = u64::MAX;
                "overflow_x_span"
            }
            20 => {
                ys = u64::MAX;
                "overflow_y_span"
            }
            21 => {
                cols = i32::MAX;
                "signed_index_overflow"
            }
            _ => {
                n = -1;
                "negative_n"
            }
        };
        // SAFETY: deliberately invalid values exercise documented prelaunch
        // guards; valid backing allocations remain live, no aliasing introduced.
        let rc = unsafe { (k.batch)(g, wp, ty, rows, cols, ws, xp, n, xs, yp, ys) };
        gpu.sync()?;
        if rc == 0 || api()?.error().is_empty() {
            return Err(format!("invalid case accepted/no ns_fail text: {label}").into());
        }
        w.unchanged()?;
        x.unchanged()?;
        y.unchanged()?;
        println!("PASS invalid={label} rejected=true memory_unchanged=true");
    }
    for ty in [8, 12, 13, 14] {
        // SAFETY: valid buffers but deliberately misaligned packed column count.
        let rc = unsafe {
            (k.batch)(
                gpu.raw(),
                w.view.ptr(),
                ty,
                2,
                255,
                1024,
                x.view.fp(),
                2,
                256,
                y.view.fp(),
                2,
            )
        };
        gpu.sync()?;
        if rc == 0 || api()?.error().is_empty() {
            return Err("quant column alignment accepted".into());
        }
        w.unchanged()?;
        x.unchanged()?;
        y.unchanged()?;
        println!("PASS invalid=quant_columns type={ty}");
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
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or(HELP)?;
    if model == "--help" {
        println!("{HELP}");
        return Ok(());
    }
    let mut gpu_index = 0;
    let mut reps = 10;
    while let Some(key) = args.next() {
        let v = args
            .next()
            .ok_or("option value required")?
            .parse::<usize>()?;
        match key.as_str() {
            "--gpu" => gpu_index = v,
            "--reps" => reps = v,
            _ => return Err("unknown option".into()),
        }
    }
    if !(1..=100).contains(&reps) {
        return Err("reps must be 1..100".into());
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
            "PASS CPU logical pack/decode {ty:?} values={}",
            m.weights.len()
        );
        matrices.push(m);
    }
    if model == "--cpu-only" {
        println!("PASS CPU-only; no runtime opened");
        return Ok(());
    }
    matrices.extend(actual(&Gguf::open(Path::new(&model))?)?);
    let k = Kernels::load()?;
    let gpu = Gpu::open(gpu_index)?;
    if gpu.raw().is_null() {
        return Err("native GPU required".into());
    }
    println!("runtime={} device={gpu_index} scalar_batch_gate=bitwise cpu_gate=1e-5+4e-6*sumabs timing=host_wall_with_sync",api()?.path().display());
    invalid(&gpu, &k)?;
    for matrix in &matrices {
        check_matrix(&gpu, &k, matrix, reps)?;
    }
    println!(
        "PASS matrices={} batch_sizes=10 contiguous_n4=true invalid_cases=27",
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
