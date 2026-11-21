//! Isolated joint text/image DiT block for the first numerical gate. This does
//! not expose an image engine or claim the full reference resolution envelope.
//! Graph order follows nextsycl 98124afb image/qwenimage21/src/dit.rs.
use std::ffi::c_void;
use std::sync::Arc;

use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_gguf::Gguf;

use crate::ffi::{self, OutputType, PackedKernel};
use crate::plan::{DitPlan, FFN, HEADS, HEAD_DIM, LAYERS, WIDTH};
use crate::weights::PackedMatrix;

pub struct Block {
    pub(crate) gpu: Arc<Gpu>,
    pub(crate) qkv: PackedMatrix,
    pub(crate) out: PackedMatrix,
    pub(crate) gate_up: PackedMatrix,
    pub(crate) down: PackedMatrix,
    pub(crate) norm_q: DevBuf,
    pub(crate) norm_k: DevBuf,
    pub packed_bytes: usize,
    pub(crate) kernel: PackedKernel,
}

impl Block {
    pub fn load(file: &Gguf, gpu: &Arc<Gpu>, layer: usize, allowance: usize) -> Result<Self> {
        Self::load_with_kernel(file, gpu, layer, allowance, PackedKernel::Baseline)
    }

    pub fn load_with_kernel(
        file: &Gguf,
        gpu: &Arc<Gpu>,
        layer: usize,
        allowance: usize,
        kernel: PackedKernel,
    ) -> Result<Self> {
        ffi::api()?.require_kernel(kernel)?;
        DitPlan::validate(file)?;
        if layer >= LAYERS {
            return Err(Error("invalid DiT block index".into()));
        }
        // Reserve both norm vectors before loading packed matrices.
        let mut remaining = allowance
            .checked_sub(2 * HEAD_DIM * 4)
            .ok_or_else(|| Error("block allowance does not cover norms".into()))?;
        let p = format!("transformer_blocks.{layer}.");
        let mut load = |parts: &[(&str, bool)]| -> Result<PackedMatrix> {
            let names: Vec<_> = parts.iter().map(|(n, b)| (format!("{p}{n}"), *b)).collect();
            let refs: Vec<_> = names.iter().map(|(n, b)| (n.as_str(), *b)).collect();
            let matrix = PackedMatrix::load(file, gpu, &refs, remaining)?;
            remaining -= matrix.bytes;
            Ok(matrix)
        };
        let qkv = load(&[
            ("attn.to_q.weight", true),
            ("attn.to_k.weight", true),
            ("attn.to_v.weight", false),
        ])?;
        let out = load(&[("attn.to_out.0.weight", false)])?;
        let gate_up = load(&[
            ("img_mlp.gate_layer.weight", false),
            ("img_mlp.proj.weight", false),
        ])?;
        let down = load(&[("img_mlp.out.weight", false)])?;
        let norm = |name: &str| -> Result<DevBuf> {
            let name = format!("{p}{name}");
            let t = file
                .tensor(&name)
                .ok_or_else(|| Error(format!("missing {name}")))?;
            let raw = file.read(t).map_err(|e| Error(e.0))?;
            let v: Vec<_> = raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            if v.len() != HEAD_DIM || v.iter().any(|x| !x.is_finite()) {
                return Err(Error("invalid head norm".into()));
            }
            let permuted: Vec<_> = (0..HEAD_DIM).map(|i| v[(i % 64) * 2 + i / 64]).collect();
            DevBuf::from_f32(gpu, &permuted)
        };
        Ok(Self {
            gpu: gpu.clone(),
            qkv,
            out,
            gate_up,
            down,
            norm_q: norm("attn.norm_q.weight")?,
            norm_k: norm("attn.norm_k.weight")?,
            packed_bytes: allowance - remaining,
            kernel,
        })
    }

    /// Inputs are the independently captured first-block joint rows and exact
    /// rotary table. Text rows precede one target image. No editing segments,
    /// padding, side LoRAs or prefix-cache admission is implied by this slice.
    pub fn joint(
        &self,
        input: &[f32],
        text_rows: usize,
        modulation: &[f32],
        rope: &[f32],
    ) -> Result<DevBuf> {
        if input.len() % WIDTH != 0 || input.is_empty() {
            return Err(Error("block input must contain complete joint rows".into()));
        }
        let m = input.len() / WIDTH;
        if m > 4096
            || text_rows == 0
            || text_rows >= m
            || modulation.len() != 2 * 4 * WIDTH
            || rope.len() != m * HEAD_DIM
            || input
                .iter()
                .chain(modulation)
                .chain(rope)
                .any(|x| !x.is_finite())
        {
            return Err(Error("invalid bounded block fixture".into()));
        }
        let api = ffi::api()?;
        let g = &self.gpu;
        let x = DevBuf::from_f32(g, input)?;
        let cs = DevBuf::from_f32(g, rope)?;
        // Modulation row0 = sampled timestep, row1 = zero timestep for text.
        let mut md = Vec::with_capacity(8);
        for row in 0..2 {
            for part in 0..4 {
                let values: Vec<_> = modulation
                    [(row * 4 + part) * WIDTH..(row * 4 + part + 1) * WIDTH]
                    .iter()
                    .map(|&v| if part % 2 == 1 { v.tanh() } else { v })
                    .collect();
                md.push(DevBuf::from_f32(g, &values)?);
            }
        }
        let h = DevBuf::new(g, m * WIDTH * 2)?;
        let qkv = DevBuf::new(g, m * 3 * WIDTH * 2)?;
        let att = DevBuf::new(g, m * WIDTH * 2)?;
        let o = DevBuf::new(g, m * WIDTH * 2)?;
        let gu = DevBuf::new(g, m * 2 * FFN * 2)?;
        let act = DevBuf::new(g, m * FFN * 2)?;
        let segments = [(0, text_rows, 1), (text_rows, m - text_rows, 0)];
        let half_at = |b: &DevBuf, at: usize| -> *mut c_void {
            b.ptr().cast::<u16>().wrapping_add(at).cast()
        };
        let norm = |part: usize| -> Result<()> {
            for &(start, rows, mdrow) in &segments {
                // SAFETY: prevalidated complete rows and separately owned buffers;
                // segment ranges partition [0,m), scales hold WIDTH floats.
                ffi::check(
                    unsafe {
                        (api.ln_mod)(
                            g.raw(),
                            x.fp().wrapping_add(start * WIDTH),
                            rows as i64,
                            WIDTH as i64,
                            1e-6,
                            md[mdrow * 4 + part].fp(),
                            half_at(&h, start * WIDTH),
                        )
                    },
                    "block layer norm",
                )?;
            }
            Ok(())
        };
        let gate = |part: usize| -> Result<()> {
            for &(start, rows, mdrow) in &segments {
                // SAFETY: same partition as normalization; half residual and
                // F32 destination are disjoint owned allocations.
                ffi::check(
                    unsafe {
                        (api.gate_add)(
                            g.raw(),
                            x.fp().wrapping_add(start * WIDTH),
                            half_at(&o, start * WIDTH),
                            rows as i64,
                            WIDTH as i64,
                            md[mdrow * 4 + part].fp(),
                        )
                    },
                    "block residual",
                )?;
            }
            Ok(())
        };
        norm(0)?;
        api.packed_matrix_using(self.kernel, g, &self.qkv, &h, m, &qkv, OutputType::Half)?;
        for (offset, w) in [(0, &self.norm_q), (WIDTH, &self.norm_k)] {
            // SAFETY: each strided head lies inside qkv; rotary table holds one
            // interleaved pair per half-head channel, per joint row.
            ffi::check(
                unsafe {
                    (api.rms_rope)(
                        g.raw(),
                        half_at(&qkv, offset),
                        m as i64,
                        HEADS as i64,
                        HEAD_DIM as i64,
                        (3 * WIDTH) as i64,
                        w.fp(),
                        1e-6,
                        cs.fp(),
                        HEAD_DIM as i32,
                    )
                },
                "block RMS rotary",
            )?;
        }
        // SAFETY: first segment is causal text; the remaining image queries
        // attend all text and image keys. Shared baseline caps total keys4096.
        ffi::check(
            unsafe {
                (api.attention_causal_rows)(
                    g.raw(),
                    qkv.ptr(),
                    half_at(&qkv, WIDTH),
                    half_at(&qkv, 2 * WIDTH),
                    1,
                    0,
                    text_rows as i64,
                    HEADS as i64,
                    HEADS as i64,
                    HEAD_DIM as i64,
                    (3 * WIDTH) as i64,
                    (3 * WIDTH) as i64,
                    att.ptr(),
                )
            },
            "block causal text attention",
        )?;
        ffi::check(
            unsafe {
                (api.attention_qk)(
                    g.raw(),
                    half_at(&qkv, text_rows * 3 * WIDTH),
                    (m - text_rows) as i64,
                    (3 * WIDTH) as i64,
                    half_at(&qkv, WIDTH),
                    half_at(&qkv, 2 * WIDTH),
                    m as i64,
                    (3 * WIDTH) as i64,
                    1,
                    HEADS as i64,
                    HEAD_DIM as i64,
                    half_at(&att, text_rows * WIDTH),
                    1,
                )
            },
            "block image attention",
        )?;
        api.packed_matrix_using(self.kernel, g, &self.out, &att, m, &o, OutputType::Half)?;
        gate(1)?;
        norm(2)?;
        api.packed_matrix_using(self.kernel, g, &self.gate_up, &h, m, &gu, OutputType::Half)?;
        // SAFETY: gu holds m rows of gate|up, act m rows of FFN outputs.
        ffi::check(
            unsafe { (api.swiglu)(g.raw(), gu.ptr(), 1, m as i64, FFN as i64, act.ptr(), 1) },
            "block SwiGLU",
        )?;
        api.packed_matrix_using(self.kernel, g, &self.down, &act, m, &o, OutputType::Half)?;
        gate(3)?;
        g.sync()?;
        Ok(x)
    }
}
