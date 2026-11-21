//! Parent-serialized GPU microbench; never runs as part of a build/test.
//! qwen36_transfer MODEL [--gpu 0] [--reps 10] [--chunk-mib 16] [--expert 0] [--zero-copy]
//! Reads the first layer's gate/up/down expert chunks. Reuses allocations and
//! alternates pageable/pinned order. Warm page-cache measurements, not cold I/O.
use nextrocm_engine_core::{api, DevBuf, Gpu, HostBuf};
use nextrocm_gguf::{Gguf, Tensor};
use std::{
    error::Error, ffi::c_void, fs::File, os::unix::fs::FileExt, path::Path, sync::Arc,
    time::Instant,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type Mv = unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, i32, *mut f32, *mut f32) -> i32;
const HELP: &str =
    "qwen36_transfer MODEL [--gpu 0] [--reps 10] [--chunk-mib 16] [--expert 0] [--zero-copy]";

struct Options {
    model: String,
    gpu: usize,
    reps: usize,
    chunk: usize,
    expert: usize,
    zero: bool,
}
fn options() -> Result<Option<Options>> {
    let mut args = std::env::args().skip(1);
    let model = args.next().ok_or(HELP)?;
    if model == "--help" {
        println!("{HELP}");
        return Ok(None);
    }
    let mut o = Options {
        model,
        gpu: 0,
        reps: 10,
        chunk: 16 * 1024 * 1024,
        expert: 0,
        zero: false,
    };
    while let Some(key) = args.next() {
        if key == "--zero-copy" {
            o.zero = true;
            continue;
        }
        let value = args
            .next()
            .ok_or("option requires a value")?
            .parse::<usize>()?;
        match key.as_str() {
            "--gpu" => o.gpu = value,
            "--reps" => o.reps = value,
            "--chunk-mib" => o.chunk = value.checked_mul(1024 * 1024).ok_or("chunk overflow")?,
            "--expert" => o.expert = value,
            _ => return Err(format!("unknown option {key}").into()),
        }
    }
    if !(1..=100).contains(&o.reps) || !(1..=16 * 1024 * 1024).contains(&o.chunk) {
        return Err("reps must be 1..100 and chunk-mib 1..16".into());
    }
    Ok(Some(o))
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| "size overflow".into())
}

struct Chunk<'a> {
    tensor: &'a Tensor,
    file: File,
    rows: usize,
    cols: usize,
    bytes: usize,
    offset: u64,
}
impl<'a> Chunk<'a> {
    fn new(f: &Gguf, t: &'a Tensor, o: &Options) -> Result<Self> {
        let [experts, total_rows, cols] = t.shape.as_slice() else {
            return Err("expert tensor must be 3D".into());
        };
        let (experts, total_rows, cols) = (
            usize::try_from(*experts)?,
            usize::try_from(*total_rows)?,
            usize::try_from(*cols)?,
        );
        if experts == 0 || total_rows == 0 || cols == 0 || o.expert >= experts {
            return Err("invalid expert dimensions/index".into());
        }
        if !matches!(t.ty.code(), 0 | 1 | 8 | 12 | 13 | 14) {
            return Err(format!("unsupported nq35 weight type {:?}", t.ty).into());
        }
        let row = usize::try_from(t.ty.bytes(cols as u64).ok_or("unaligned packed row")?)?;
        if row == 0 || row > o.chunk {
            return Err("packed row exceeds chunk".into());
        }
        let full_bytes = product(product(experts, total_rows)?, row)?;
        if u64::try_from(full_bytes)? != t.bytes {
            return Err("tensor byte extent does not match packed dimensions".into());
        }
        let rows = total_rows.min(o.chunk / row).min(i32::MAX as usize / cols);
        if rows == 0 {
            return Err("dimensions exceed kernel index range".into());
        }
        i32::try_from(cols)?;
        let bytes = product(rows, row)?;
        let offset = t
            .offset
            .checked_add(u64::try_from(product(
                product(o.expert, total_rows)?,
                row,
            )?)?)
            .ok_or("file offset overflow")?;
        let file = File::open(f.paths.get(t.shard).ok_or("missing GGUF shard")?)?;
        let tensor_end = t.offset.checked_add(t.bytes).ok_or("tensor end overflow")?;
        if tensor_end > file.metadata()?.len()
            || offset
                .checked_add(bytes as u64)
                .is_none_or(|end| end > tensor_end)
        {
            return Err("chunk outside tensor/file".into());
        }
        Ok(Self {
            tensor: t,
            file,
            rows,
            cols,
            bytes,
            offset,
        })
    }
}

fn compute(mv: Mv, gpu: &Gpu, c: &Chunk<'_>, w: &DevBuf, x: &DevBuf, y: &DevBuf) -> Result<f64> {
    w.bounds(0, c.bytes)?;
    x.bounds(0, product(c.cols, 4)?)?;
    y.bounds(0, product(c.rows, 4)?)?;
    gpu.sync()?;
    let start = Instant::now();
    // SAFETY: exact nq35_mv ABI; validated packed extents and i32 index bounds;
    // all buffers live on this GPU, including any caller-owned pinned view.
    let rc = unsafe {
        mv(
            gpu.raw(),
            w.ptr(),
            c.tensor.ty.code() as i32,
            c.rows as i32,
            c.cols as i32,
            x.fp(),
            y.fp(),
        )
    };
    // Synchronize even on dispatch error before a pinned view's owner may drop.
    let sync = gpu.sync();
    if rc != 0 {
        return Err(format!("nq35_mv: {}", api()?.error()).into());
    }
    sync?;
    Ok(start.elapsed().as_secs_f64() * 1000.0)
}

fn equal(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.len() != expected.len() || actual.is_empty() {
        return Err("output length mismatch/empty".into());
    }
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        if !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits() {
            return Err(format!(
                "output mismatch at {i}: {a:?} / {b:?} ({:08x}/{:08x})",
                a.to_bits(),
                b.to_bits()
            )
            .into());
        }
    }
    Ok(())
}

#[derive(Default)]
struct Times {
    read: Vec<f64>,
    copy: Vec<f64>,
    compute: Vec<f64>,
}
fn summary(name: &str, values: &[f64], bytes: Option<usize>) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0;
    let gib_s = bytes.map_or(String::new(), |n| {
        format!(
            " mean_GiB_s={:.6}",
            n as f64 / (1024.0_f64.powi(3) * mean / 1000.0)
        )
    });
    println!("summary metric={name} n={} mean_ms={mean:.6} median_ms={median:.6} min_ms={:.6} max_ms={:.6}{gib_s}", sorted.len(), sorted[0], sorted[sorted.len()-1]);
}

fn bench(gpu: &Arc<Gpu>, mv: Mv, c: &Chunk<'_>, o: &Options) -> Result<()> {
    println!("tensor={} shape={:?} type={:?} type_code={} expert={} first_row=0 chunk_rows={} cols={} bytes={} offset={} reps={}", c.tensor.name, c.tensor.shape, c.tensor.ty, c.tensor.ty.code(), o.expert, c.rows, c.cols, c.bytes, c.offset, o.reps);
    let mut pageable = vec![0u8; c.bytes];
    let mut pinned = HostBuf::new(gpu, c.bytes)?;
    let weights = DevBuf::new(gpu, c.bytes)?;
    let x = DevBuf::new(gpu, product(c.cols, 4)?)?;
    let y = DevBuf::new(gpu, product(c.rows, 4)?)?;
    let input: Vec<u8> = (0..c.cols)
        .flat_map(|i| (((i % 257) as f32 - 128.0) / 128.0).to_le_bytes())
        .collect();
    x.write(0, &input)?;
    // Pre-touch/reuse both allocations and prime file cache outside measured reps.
    c.file.read_exact_at(&mut pageable, c.offset)?;
    c.file.read_exact_at(pinned.as_mut_slice(), c.offset)?;
    if pageable != pinned.as_slice() {
        return Err("initial read bytes differ".into());
    }
    weights.write(0, &pageable)?;
    compute(mv, gpu, c, &weights, &x, &y)?;
    let expected = y.to_f32()?;
    equal(&expected, &expected)?;
    weights.write(0, pinned.as_slice())?;
    compute(mv, gpu, c, &weights, &x, &y)?;
    equal(&y.to_f32()?, &expected)?;
    if o.zero {
        let view = pinned.device_view(0, c.bytes)?;
        compute(mv, gpu, c, &view, &x, &y)?;
        equal(&y.to_f32()?, &expected)?;
    }
    let mut times = [Times::default(), Times::default()];
    let mut zero = Vec::new();
    for rep in 0..o.reps {
        for mode in [rep % 2, 1 - rep % 2] {
            let name = if mode == 0 { "pageable" } else { "pinned" };
            let buffer = if mode == 0 {
                pageable.as_mut_slice()
            } else {
                pinned.as_mut_slice()
            };
            let start = Instant::now();
            c.file.read_exact_at(buffer, c.offset)?;
            let read = start.elapsed().as_secs_f64() * 1000.0;
            gpu.sync()?;
            let start = Instant::now();
            weights.write(0, buffer)?; // Original synchronous API includes its own stream sync.
            let copy = start.elapsed().as_secs_f64() * 1000.0;
            let math = compute(mv, gpu, c, &weights, &x, &y)?;
            equal(&y.to_f32()?, &expected)?; // D2H/validation excluded from all timings.
            times[mode].read.push(read);
            times[mode].copy.push(copy);
            times[mode].compute.push(math);
            println!("sample rep={rep} mode={name} read_ms={read:.6} h2d_write_ms={copy:.6} mv_sync_ms={math:.6} exact=true");
        }
        if pageable != pinned.as_slice() {
            return Err("repeated read bytes differ".into());
        }
        if o.zero {
            // View is dropped before pinned is mutated/reused/freed; compute synchronizes.
            let view = pinned.device_view(0, c.bytes)?;
            let math = compute(mv, gpu, c, &view, &x, &y)?;
            equal(&y.to_f32()?, &expected)?;
            zero.push(math);
            println!("sample rep={rep} mode=zero_copy mv_sync_ms={math:.6} exact=true");
        }
    }
    for (mode, t) in ["pageable", "pinned"].iter().zip(times) {
        summary(&format!("{mode}.read"), &t.read, Some(c.bytes));
        summary(&format!("{mode}.h2d_write"), &t.copy, Some(c.bytes));
        summary(&format!("{mode}.mv_sync"), &t.compute, None);
    }
    if o.zero {
        summary("zero_copy.mv_sync", &zero, None);
    }
    println!(
        "PASS tensor={} finite=true bitwise_equal=true",
        c.tensor.name
    );
    Ok(())
}

fn run() -> Result<()> {
    let Some(o) = options()? else {
        return Ok(());
    };
    let file = Gguf::open(Path::new(&o.model))?;
    if file.architecture() != "qwen35moe" {
        return Err("expected qwen35moe GGUF".into());
    }
    let first = file
        .tensors
        .iter()
        .filter_map(|t| {
            if t.shape.len() != 3 {
                return None;
            }
            t.name
                .strip_prefix("blk.")?
                .strip_suffix(".ffn_gate_exps.weight")?
                .parse::<usize>()
                .ok()
        })
        .min()
        .ok_or("no 3D expert gate tensor")?;
    let mut chunks = Vec::new();
    for role in ["gate", "up", "down"] {
        let name = format!("blk.{first}.ffn_{role}_exps.weight");
        chunks.push(Chunk::new(
            &file,
            file.tensor(&name).ok_or("missing paired expert tensor")?,
            &o,
        )?);
    }
    let a = api()?; // No host fallback: native runtime and symbol are mandatory.
    let symbol = a.symbol("nq35_mv");
    if symbol.is_null() {
        return Err("runtime missing nq35_mv".into());
    }
    // SAFETY: nq35_mv signature is kernels/llm/qwen35/qwen35.h; Api is process-lived.
    let mv: Mv = unsafe { std::mem::transmute(symbol) };
    let gpu = Gpu::open(o.gpu)?;
    if gpu.raw().is_null() {
        return Err("native GPU required".into());
    }
    println!("runtime={} gpu={} warm_cache=true allocation_excluded=true readback_excluded=true timing=host_wall_sync reps={} zero_copy={}", a.path().display(), o.gpu, o.reps, o.zero);
    for chunk in chunks {
        bench(&gpu, mv, &chunk, &o)?;
    }
    println!("PASS all_three_expert_chunks");
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL {e}");
        std::process::exit(1);
    }
}
