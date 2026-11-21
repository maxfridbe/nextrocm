//! Independent scalar F64 controls for private operators; run only on the native GPU.
use crate::ffi;
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use std::sync::Arc;
fn same(a: &[f32], b: &[f32], tol: f64) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error("operator shape mismatch".into()));
    }
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        if !x.is_finite()
            || !y.is_finite()
            || (x as f64 - y as f64).abs() > tol * (1. + (y as f64).abs())
        {
            return Err(Error(format!("operator value {i}: {x} != {y}")));
        }
    }
    Ok(())
}
fn wave(i: usize) -> f32 {
    ((i * 37 % 101) as f32 - 50.) / 31.
}
/// Conventional dense softmax in F64, independent of GPU chunking/reduction.
fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    kv: usize,
    d: usize,
    stride: usize,
) -> Vec<f32> {
    let mut out = vec![0.; heads * d];
    for h in 0..heads {
        let group = h / (heads / kv);
        let scores: Vec<f64> = (0..n)
            .map(|j| {
                (0..d)
                    .map(|c| q[h * d + c] as f64 * k[j * stride + group * d + c] as f64)
                    .sum::<f64>()
                    / (d as f64).sqrt()
            })
            .collect();
        let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
        let den: f64 = p.iter().sum();
        for c in 0..d {
            out[h * d + c] = (p
                .iter()
                .enumerate()
                .map(|(j, p)| p * v[j * stride + group * d + c] as f64)
                .sum::<f64>()
                / den) as f32;
        }
    }
    out
}
pub fn run(g: &Arc<Gpu>, log: &mut dyn FnMut(String)) -> Result<usize> {
    if g.raw().is_null() {
        return Err(Error("operators require native GPU".into()));
    }
    let api = ffi::api()?;
    let mut cases = 0;
    // Long reduction lengths used by the decoder, plus partial tiles. Inputs
    // are exactly representable in F16; the oracle is a separate scalar F64 dot.
    for width in [17, 1024, 2048, 3072, 4097] {
        for (rows, outputs) in [(1, 19), (17, 5)] {
            for cancel in [false, true] {
                let value = |i: usize| {
                    let mantissa = ((i * 37 % 1021) as i32 - 510) as f32;
                    mantissa * 2f32.powi(-12 - (i % 5) as i32)
                };
                let x: Vec<f32> = (0..rows * width)
                    .map(|i| {
                        if cancel {
                            value((i / width) * width + (i % width) / 2)
                        } else {
                            value(i)
                        }
                    })
                    .collect();
                let weights: Vec<f32> = (0..outputs * width)
                    .map(|i| {
                        if cancel {
                            let col = i % width;
                            // Opposing adjacent terms, with a small unequal tail.
                            let sign = if col % 2 == 0 { 1. } else { -1. };
                            sign * 2f32.powi(-2 - ((col / 2) % 10) as i32)
                        } else {
                            value(i + 143)
                        }
                    })
                    .collect();
                let input = DevBuf::from_f32(g, &x)?;
                let weight = DevBuf::from_f32(g, &weights)?;
                let half_x = DevBuf::new(g, x.len() * 2)?;
                let half_w = DevBuf::new(g, weights.len() * 2)?;
                let output = DevBuf::new(g, rows * outputs * 4)?;
                // SAFETY: all dense buffers cover the independently checked sizes.
                unsafe {
                    ffi::check(
                        (api.to_half)(g.raw(), input.fp(), half_x.ptr(), x.len() as i64),
                        "Linear input half",
                    )?;
                    ffi::check(
                        (api.to_half)(g.raw(), weight.fp(), half_w.ptr(), weights.len() as i64),
                        "Linear weight half",
                    )?;
                    ffi::check(
                        (api.linear)(
                            g.raw(),
                            half_x.ptr(),
                            rows as i64,
                            width as i64,
                            half_w.ptr(),
                            outputs as i64,
                            output.fp(),
                        ),
                        "Linear8",
                    )?;
                }
                let mut expected = Vec::with_capacity(rows * outputs);
                for r in 0..rows {
                    for c in 0..outputs {
                        expected.push(
                            (0..width)
                                .map(|j| x[r * width + j] as f64 * weights[c * width + j] as f64)
                                .sum::<f64>() as f32,
                        );
                    }
                }
                same(&output.to_f32()?, &expected, 2e-5)?;
                cases += 1;
            }
        }
    }
    // RNE ties, signs, exact powers, normal/subnormal boundary and signed zero.
    let x = [
        0.,
        -0.,
        1.,
        -2.,
        65504.,
        2f32.powi(-14),
        2f32.powi(-24),
        2f32.powi(-25),
        1. + 2f32.powi(-11),
        1. + 3. * 2f32.powi(-11),
    ];
    let expected = [
        0u16, 0x8000, 0x3c00, 0xc000, 0x7bff, 0x0400, 1, 0, 0x3c00, 0x3c02,
    ];
    let input = DevBuf::from_f32(g, &x)?;
    let out = DevBuf::new(g, expected.len() * 2)?;
    // SAFETY: both buffers cover the checked spans and remain live until readback.
    ffi::check(
        unsafe { (api.to_half)(g.raw(), input.fp(), out.ptr(), x.len() as i64) },
        "conversion",
    )?;
    let mut bytes = vec![0u8; expected.len() * 2];
    out.read(0, &mut bytes)?;
    let got: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    if got != expected {
        return Err(Error(format!("half bits {got:?}")));
    }
    cases += 1;
    for (rows, cols, ss, ds) in [(1, 1, 3, 4), (5, 7, 11, 13), (257, 3, 5, 9)] {
        let src: Vec<f32> = (0..rows * ss).map(wave).collect();
        let mut expected = vec![-999.; rows * ds];
        for r in 0..rows {
            expected[r * ds..r * ds + cols].copy_from_slice(&src[r * ss..r * ss + cols]);
        }
        let input = DevBuf::from_f32(g, &src)?;
        let out = DevBuf::from_f32(g, &vec![-999.; rows * ds])?;
        // SAFETY: rows and strides are within the allocated buffers, including untouched guards.
        ffi::check(
            unsafe {
                (api.copy_rows)(
                    g.raw(),
                    input.fp(),
                    ss as i64,
                    out.fp(),
                    ds as i64,
                    rows as i64,
                    cols as i64,
                )
            },
            "copy",
        )?;
        same(&out.to_f32()?, &expected, 0.)?;
        cases += 1;
    }
    for (h, w, c) in [(2, 2, 1), (4, 6, 3), (6, 4, 17)] {
        let src: Vec<f32> = (0..h * w * c).map(|n| n as f32).collect();
        let mut expected = vec![0.; src.len()];
        // Independent scatter from each source pixel to its block quadrant.
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    expected
                        [((y / 2) * (w / 2) + x / 2) * 4 * c + ((y % 2) * 2 + x % 2) * c + ch] =
                        src[(y * w + x) * c + ch];
                }
            }
        }
        let input = DevBuf::from_f32(g, &src)?;
        let out = DevBuf::f32(g, src.len())?;
        // SAFETY: even spatial dimensions and exactly H*W*C values on both sides.
        ffi::check(
            unsafe { (api.merge2)(g.raw(), input.fp(), h as i64, w as i64, c as i64, out.fp()) },
            "merge",
        )?;
        same(&out.to_f32()?, &expected, 0.)?;
        cases += 1;
    }
    for d in (16..=256).step_by(16) {
        for n in [1, 2, 255, 256, 257, 513] {
            for rep in [1, 2, 8, 16] {
                let kv = 2;
                let heads = kv * rep;
                let stride = kv * d + 3;
                let q: Vec<f32> = (0..heads * d).map(wave).collect();
                let k: Vec<f32> = (0..n * stride).map(|i| wave(i + 11)).collect();
                let v: Vec<f32> = (0..n * stride).map(|i| wave(i + 29)).collect();
                let expected = attention(&q, &k, &v, n, heads, kv, d, stride);
                let q = DevBuf::from_f32(g, &q)?;
                let k = DevBuf::from_f32(g, &k)?;
                let v = DevBuf::from_f32(g, &v)?;
                let out = DevBuf::f32(g, heads * d)?;
                // SAFETY: scratch query has no device access; dimensions are small positive controls.
                let size = unsafe { (api.attn1_scratch)(n as i64, heads as i64, d as i64) };
                let want = heads * n + heads * n.div_ceil(256) * (d + 2);
                if size != want as i64 {
                    return Err(Error("scratch size differs".into()));
                }
                let scratch = DevBuf::f32(g, want)?;
                // SAFETY: all spans, grouped heads and scratch sizes were checked above.
                ffi::check(
                    unsafe {
                        (api.attn1)(
                            g.raw(),
                            q.fp(),
                            k.fp(),
                            v.fp(),
                            n as i64,
                            stride as i64,
                            heads as i64,
                            kv as i64,
                            d as i64,
                            scratch.fp(),
                            out.fp(),
                        )
                    },
                    "attention",
                )?;
                same(&out.to_f32()?, &expected, 2e-5)?;
                cases += 1;
            }
        }
    }
    // Large positive/negative scores exercise stable softmax without overflow.
    for sign in [-1f32, 1.] {
        let (n, d, heads, kv, stride) = (257usize, 128usize, 2usize, 1usize, 131usize);
        let q = vec![sign * 100.; heads * d];
        let mut k = vec![0.; n * stride];
        let v: Vec<f32> = (0..n * stride).map(wave).collect();
        for j in 0..n {
            for c in 0..d {
                k[j * stride + c] = if j == 256 { 100. } else { -100. };
            }
        }
        let expected = attention(&q, &k, &v, n, heads, kv, d, stride);
        let q = DevBuf::from_f32(g, &q)?;
        let k = DevBuf::from_f32(g, &k)?;
        let v = DevBuf::from_f32(g, &v)?;
        let out = DevBuf::f32(g, heads * d)?;
        let scratch = DevBuf::f32(g, heads * n + heads * n.div_ceil(256) * (d + 2))?;
        // SAFETY: allocated spans exactly cover the declared grouped attention geometry.
        ffi::check(
            unsafe {
                (api.attn1)(
                    g.raw(),
                    q.fp(),
                    k.fp(),
                    v.fp(),
                    n as i64,
                    stride as i64,
                    heads as i64,
                    kv as i64,
                    d as i64,
                    scratch.fp(),
                    out.fp(),
                )
            },
            "extreme attention",
        )?;
        same(&out.to_f32()?, &expected, 2e-5)?;
        cases += 1;
    }
    // BF16 decoding is independent bit expansion, with exact F16 rounding controls.
    let words = [0u16, 0x8000, 0x3f80, 0xbf80, 0x3f81, 0x0080];
    let input = DevBuf::new(g, words.len() * 2)?;
    input.write(
        0,
        &words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let out = DevBuf::f32(g, words.len())?;
    // SAFETY: buffers hold exactly six BF16 inputs and F32 outputs.
    ffi::check(
        unsafe { (api.dequant)(g.raw(), input.ptr(), 30, words.len() as i64, out.ptr(), 0) },
        "BF16",
    )?;
    let got = out.to_f32()?;
    if got
        .iter()
        .zip(words)
        .any(|(a, b)| a.to_bits() != (b as u32) << 16)
    {
        return Err(Error("BF16 bits differ".into()));
    }
    cases += 1;
    let half = DevBuf::new(g, words.len() * 2)?;
    // SAFETY: six BF16 inputs and six F16 destinations.
    ffi::check(
        unsafe { (api.dequant)(g.raw(), input.ptr(), 30, words.len() as i64, half.ptr(), 1) },
        "BF16 half",
    )?;
    let mut bytes = vec![0; words.len() * 2];
    half.read(0, &mut bytes)?;
    if bytes
        != [0u16, 0x8000, 0x3c00, 0xbc00, 0x3c08, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    {
        return Err(Error("BF16 to half differs".into()));
    }
    cases += 1;
    for d in [72usize, 128] {
        let (rows, heads) = (3, 2);
        let stride = heads * d + 7;
        let x: Vec<f32> = (0..rows * stride).map(wave).collect();
        let cs: Vec<f32> = (0..rows * d / 2)
            .flat_map(|i| {
                let a = i as f64 / 17.;
                [a.cos() as f32, a.sin() as f32]
            })
            .collect();
        let mut expected = x.clone();
        for r in 0..rows {
            for h in 0..heads {
                for j in 0..d / 2 {
                    let at = r * stride + h * d + j;
                    let (c, s) = (cs[r * d + 2 * j] as f64, cs[r * d + 2 * j + 1] as f64);
                    let (a, b) = (x[at] as f64, x[at + d / 2] as f64);
                    expected[at] = (a * c - b * s) as f32;
                    expected[at + d / 2] = (b * c + a * s) as f32;
                }
            }
        }
        let x = DevBuf::from_f32(g, &x)?;
        let cs = DevBuf::from_f32(g, &cs)?;
        // SAFETY: strided rows plus untouched padding and a complete cosine table.
        ffi::check(
            unsafe {
                (api.rope)(
                    g.raw(),
                    x.fp(),
                    rows as i64,
                    heads as i64,
                    d as i64,
                    stride as i64,
                    cs.fp(),
                )
            },
            "RoPE",
        )?;
        same(&x.to_f32()?, &expected, 2e-5)?;
        cases += 1;
    }
    // Dense bidirectional/causal attention against an independent CPU reference.
    for (rows, heads, kv, d, causal) in [
        (13usize, 2usize, 2usize, 72usize, false),
        (17, 4, 2, 128, true),
        (257, 1, 1, 16, false),
    ] {
        let (qs, ks) = (heads * d + 3, kv * d + 5);
        let q: Vec<f32> = (0..rows * qs).map(wave).collect();
        let k: Vec<f32> = (0..rows * ks).map(|i| wave(i + 7)).collect();
        let v: Vec<f32> = (0..rows * ks).map(|i| wave(i + 13)).collect();
        let mut expected = Vec::new();
        for r in 0..rows {
            expected.extend(attention(
                &q[r * qs..r * qs + heads * d],
                &k,
                &v,
                if causal { r + 1 } else { rows },
                heads,
                kv,
                d,
                ks,
            ));
        }
        let q = DevBuf::from_f32(g, &q)?;
        let k = DevBuf::from_f32(g, &k)?;
        let v = DevBuf::from_f32(g, &v)?;
        let out = DevBuf::f32(g, rows * heads * d)?;
        for normalize_first in [0, 2] {
            // SAFETY: all rows/heads/strides correspond to allocated inputs and output.
            ffi::check(
                unsafe {
                    (api.attention)(
                        g.raw(),
                        q.fp(),
                        k.fp(),
                        v.fp(),
                        rows as i64,
                        heads as i64,
                        kv as i64,
                        d as i64,
                        qs as i64,
                        ks as i64,
                        out.fp(),
                        i32::from(causal) | normalize_first,
                    )
                },
                "prefill",
            )?;
            same(&out.to_f32()?, &expected, 2e-5)?;
            cases += 1;
        }
    }
    // Original maximum page range above the shared speech kernel's 4096-key limit.
    // Uniform scores have the analytic mean V, avoiding a second quadratic oracle.
    for rows in [4097usize, 8192] {
        let d = 16;
        let q = DevBuf::from_f32(g, &vec![0.; rows * d])?;
        let k = DevBuf::from_f32(g, &vec![0.; rows * d])?;
        let values: Vec<f32> = (0..rows * d).map(wave).collect();
        let expected: Vec<f32> = (0..d)
            .map(|c| {
                ((0..rows).map(|r| values[r * d + c] as f64).sum::<f64>() / rows as f64) as f32
            })
            .collect();
        let v = DevBuf::from_f32(g, &values)?;
        let out = DevBuf::f32(g, rows * d)?;
        // SAFETY: contiguous rows, one head, maximum supported original page size.
        ffi::check(
            unsafe {
                (api.attention)(
                    g.raw(),
                    q.fp(),
                    k.fp(),
                    v.fp(),
                    rows as i64,
                    1,
                    1,
                    d as i64,
                    d as i64,
                    d as i64,
                    out.fp(),
                    0,
                )
            },
            "large page",
        )?;
        for row in out.to_f32()?.chunks_exact(d) {
            same(row, &expected, 2e-5)?;
        }
        cases += 1;
    }
    // Reject malformed geometry before any launch, including division-by-zero and overflow.
    let b = DevBuf::f32(g, 16)?;
    // SAFETY: these calls must reject geometry on the host without accessing buffer data.
    unsafe {
        for (n, h, d) in [(i64::MAX, 16, 128), (1, 0, 128), (1, 1, 0)] {
            if (api.attn1_scratch)(n, h, d) >= 0 {
                return Err(Error("invalid scratch accepted".into()));
            }
            cases += 1;
        }
        if (api.attn1)(
            g.raw(),
            b.fp(),
            b.fp(),
            b.fp(),
            1,
            1,
            1,
            0,
            16,
            b.fp(),
            b.fp(),
        ) == 0
        {
            return Err(Error("zero KV heads accepted".into()));
        }
        if (api.merge2)(g.raw(), b.fp(), 3, 2, 1, b.fp()) == 0 {
            return Err(Error("odd merge accepted".into()));
        }
        if (api.copy_rows)(g.raw(), b.fp(), 1, b.fp(), 1, 2, 3) == 0 {
            return Err(Error("short stride accepted".into()));
        }
    }
    cases += 3;
    log(format!("private operator controls PASS: {cases}"));
    Ok(cases)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_attention_zero_logits_is_mean() {
        let out = attention(
            &[0.; 32],
            &[0.; 48],
            &(0..48).map(|i| i as f32).collect::<Vec<_>>(),
            3,
            2,
            1,
            16,
            16,
        );
        for h in 0..2 {
            for c in 0..16 {
                assert_eq!(out[h * 16 + c], 16. + c as f32);
            }
        }
    }
}
