//! AudioVAE V2 (voxcpm's modules/audiovae/audio_vae_v2.py): 64-wide latents at 25 a second, decoded to 48 kHz and
//! encoded from 16 kHz. Causal convolutions throughout, weight-normalized (the weights made at load: g * v / |v|).
//!
//! ```text
//!   decode: depthwise k7 (64), 1x1 to 2,048; 6 blocks (x8, x6, x5, x2, x2, x2), each after a sample-rate scale and
//!           bias (48 kHz's bucket): Snake, a transposed conv halving the channels, 3 residual units (Snake, depthwise
//!           k7 at dilation 1 / 3 / 9, Snake, 1x1); Snake, conv k7 to 1, tanh
//!   encode: conv k7 to 128; 4 blocks (x2, x5, x8, x8: 3 residual units, Snake, a strided conv doubling the
//!           channels); conv k3 to 64 (the mean)
//! ```
//!
//! Long latents are decoded in chunks, each behind some of the latents before it (the convolutions are causal: their
//! outputs are cut).

use std::collections::HashMap;

use nextrocm_audio::{Error, Result};
use nextrocm_engine_core::DevBuf;

use crate::ops::Ops;
use crate::pth::Tensor;

/// A causal convolution: weight [co, ci, k] (depthwise: [c, 1, k]; transposed: [ci, co, k]), bias
struct Conv {
    w: DevBuf,
    b: Option<DevBuf>,
    ci: usize,
    co: usize,
    k: usize,
    depthwise: bool,
}

struct Unit {
    a1: DevBuf,
    dw: Conv,
    a2: DevBuf,
    pw: Conv,
    dil: usize,
}

struct DecBlock {
    sr_scale: DevBuf,
    sr_bias: DevBuf,
    act: DevBuf,
    up: Conv,
    stride: usize,
    units: Vec<Unit>,
}

struct EncBlock {
    units: Vec<Unit>,
    act: DevBuf,
    down: Conv,
    stride: usize,
}

pub struct Vae {
    d_in: Conv,
    d_pw: Conv,
    blocks: Vec<DecBlock>,
    d_act: DevBuf,
    d_out: Conv,
    e_in: Conv,
    e_blocks: Vec<EncBlock>,
    e_mu: Conv,
    pub latent: usize,
    /// output samples a latent (48 kHz), input samples a latent (16 kHz)
    pub hop_out: usize,
    pub hop_in: usize,
}

/// Largest live activation in the encoder. Downsampling lengths are exact because
/// the caller pads the input to the complete encoder hop. Every residual unit
/// preserves its channel count and length; no convolution uses these buffers as
/// hidden scratch space.
fn encoder_workspace(
    samples: usize,
    input_channels: usize,
    blocks: impl IntoIterator<Item = (usize, usize)>,
    latent_channels: usize,
) -> Result<usize> {
    let mut length = samples;
    let mut widest = samples.max(crate::budget::product(&[input_channels, samples])?);
    for (channels, stride) in blocks {
        if stride == 0 || length == 0 || !length.is_multiple_of(stride) {
            return Err(Error(
                "AudioVAE encoder input must align to every stride".into(),
            ));
        }
        length /= stride;
        widest = widest.max(crate::budget::product(&[channels, length])?);
    }
    Ok(widest.max(crate::budget::product(&[latent_channels, length])?))
}

/// Input-sample reach of an output at index zero, relative to its stride grid.
/// Strided encoder convolutions have a right reach as well as a causal left pad.
#[derive(Default)]
struct EncoderReach {
    past: usize,
    future: usize,
    jump: usize,
}

impl EncoderReach {
    fn new() -> Self {
        Self {
            jump: 1,
            ..Self::default()
        }
    }

    fn causal(&mut self, kernel: usize, dilation: usize) -> Result<()> {
        let width = kernel
            .checked_sub(1)
            .ok_or_else(|| Error("zero encoder kernel".into()))?;
        if dilation == 0 {
            return Err(Error("zero encoder dilation".into()));
        }
        self.past = crate::budget::sum(&[
            self.past,
            crate::budget::product(&[width, dilation, self.jump])?,
        ])?;
        Ok(())
    }

    fn down(&mut self, kernel: usize, stride: usize) -> Result<()> {
        if stride == 0 {
            return Err(Error("zero encoder stride".into()));
        }
        let left = kernel
            .checked_sub(stride)
            .ok_or_else(|| Error("encoder kernel smaller than stride".into()))?;
        self.past = crate::budget::sum(&[self.past, crate::budget::product(&[left, self.jump])?])?;
        self.future = crate::budget::sum(&[
            self.future,
            crate::budget::product(&[stride - 1, self.jump])?,
        ])?;
        self.jump = crate::budget::product(&[self.jump, stride])?;
        Ok(())
    }

    fn context(&self, hop: usize) -> Result<usize> {
        if hop == 0 || self.jump != hop || self.future != hop - 1 {
            return Err(Error("inconsistent encoder hop or right context".into()));
        }
        crate::budget::product(&[self.past.div_ceil(hop), hop])
    }
}

/// The weight of a weight-normalized layer: g * v / |v| (the norm over all but the first dimension)
fn wn(t: &HashMap<String, Tensor>, p: &str) -> Result<(Vec<f32>, Vec<usize>)> {
    let miss = |n: &str| Error(format!("the AudioVAE has no {p}.{n}"));
    let v = t
        .get(&format!("{p}.weight_v"))
        .ok_or_else(|| miss("weight_v"))?;
    let g = t
        .get(&format!("{p}.weight_g"))
        .ok_or_else(|| miss("weight_g"))?;
    let per = v.data.len() / v.shape[0];
    let mut w = v.data.clone();
    for (o, row) in w.chunks_mut(per).enumerate() {
        let n = row
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt() as f32;
        let s = g.data[o] / n.max(1e-12);
        row.iter_mut().for_each(|x| *x *= s);
    }
    Ok((w, v.shape.clone()))
}

fn dev(ops: &Ops, v: &[f32]) -> Result<DevBuf> {
    DevBuf::from_f32(&ops.gpu, v)
}

impl Vae {
    pub fn load(ops: &Ops, t: &HashMap<String, Tensor>, conf: &serde_json::Value) -> Result<Vae> {
        let get = |n: &str| {
            t.get(n)
                .ok_or_else(|| Error(format!("the AudioVAE has no {n}")))
        };
        // depthwise: one input channel a group, as many groups as channels (the weight alone cannot say: [co, 1, k]
        // is also a convolution of a single channel)
        let conv_g = |p: &str, transposed: bool, depthwise: bool| -> Result<Conv> {
            let (w, s) = wn(t, p)?;
            let b = t
                .get(&format!("{p}.bias"))
                .map(|b| dev(ops, &b.data))
                .transpose()?;
            let (ci, co) = if transposed {
                (s[0], s[1])
            } else if depthwise {
                (s[0], s[0])
            } else {
                (s[1], s[0])
            };
            Ok(Conv {
                depthwise,
                w: dev(ops, &w)?,
                b,
                ci,
                co,
                k: s[2],
            })
        };
        let conv = |p: &str, transposed: bool| conv_g(p, transposed, false);
        let alpha = |p: &str| -> Result<DevBuf> { dev(ops, &get(&format!("{p}.alpha"))?.data) };
        let unit = |p: &str, dil: usize| -> Result<Unit> {
            Ok(Unit {
                a1: alpha(&format!("{p}.block.0"))?,
                dw: conv_g(&format!("{p}.block.1"), false, true)?,
                a2: alpha(&format!("{p}.block.2"))?,
                pw: conv(&format!("{p}.block.3"), false)?,
                dil,
            })
        };
        let vc = &conf["audio_vae_config"];
        let rates: Vec<usize> = vc["decoder_rates"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let erates: Vec<usize> = vc["encoder_rates"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
        let out_rate = vc["out_sample_rate"].as_u64().unwrap_or(48000) as i64;
        let bounds: Vec<i64> = vc["sr_bin_boundaries"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_i64())
            .collect();
        // torch.bucketize: the first boundary at or above the rate
        let bucket = bounds
            .iter()
            .position(|b| out_rate <= *b)
            .unwrap_or(bounds.len());
        let mut blocks = Vec::new();
        for (i, &s) in rates.iter().enumerate() {
            let p = format!("decoder.model.{}", i + 2);
            let sc = get(&format!(
                "decoder.sr_cond_model.{}.scale_embed.weight",
                i + 2
            ))?;
            let bi = get(&format!(
                "decoder.sr_cond_model.{}.bias_embed.weight",
                i + 2
            ))?;
            let c = sc.shape[1];
            blocks.push(DecBlock {
                sr_scale: dev(ops, &sc.data[bucket * c..(bucket + 1) * c])?,
                sr_bias: dev(ops, &bi.data[bucket * c..(bucket + 1) * c])?,
                act: alpha(&format!("{p}.block.0"))?,
                up: conv(&format!("{p}.block.1"), true)?,
                stride: s,
                units: [1, 3, 9]
                    .iter()
                    .enumerate()
                    .map(|(j, d)| unit(&format!("{p}.block.{}", j + 2), *d))
                    .collect::<Result<_>>()?,
            });
        }
        let n = rates.len() + 2;
        let mut e_blocks = Vec::new();
        for (i, &s) in erates.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            e_blocks.push(EncBlock {
                units: [1, 3, 9]
                    .iter()
                    .enumerate()
                    .map(|(j, d)| unit(&format!("{p}.block.{j}"), *d))
                    .collect::<Result<_>>()?,
                act: alpha(&format!("{p}.block.3"))?,
                down: conv(&format!("{p}.block.4"), false)?,
                stride: s,
            });
        }
        let v = Vae {
            d_in: conv_g("decoder.model.0", false, true)?,
            d_pw: conv("decoder.model.1", false)?,
            blocks,
            d_act: alpha(&format!("decoder.model.{n}"))?,
            d_out: conv(&format!("decoder.model.{}", n + 1), false)?,
            e_in: conv("encoder.block.0", false)?,
            e_blocks,
            e_mu: conv("encoder.fc_mu", false)?,
            latent: vc["latent_dim"].as_u64().unwrap_or(64) as usize,
            hop_out: rates.iter().product(),
            hop_in: erates.iter().product(),
        };
        ops.gpu.sync()?;
        Ok(v)
    }

    fn conv(
        &self,
        ops: &Ops,
        c: &Conv,
        x: *const f32,
        l: usize,
        dil: usize,
        out: *mut f32,
    ) -> Result<()> {
        let bias =
            c.b.as_ref()
                .map_or(std::ptr::null(), |b| b.fp() as *const f32);
        if c.depthwise {
            ops.dwconv(x, c.ci, l, &c.w, bias, c.k, dil, out)
        } else {
            ops.conv_causal(x, 1, c.ci, l, &c.w, c.co, c.k, bias, dil, out)
        }
    }

    /// x [c, l] (in a) += the units, through b and c
    #[allow(clippy::too_many_arguments)]
    fn units(
        &self,
        ops: &Ops,
        units: &[Unit],
        ch: usize,
        l: usize,
        a: &DevBuf,
        b: &DevBuf,
        c: &DevBuf,
    ) -> Result<()> {
        for u in units {
            ops.snake(a.fp(), ch, l, &u.a1, b.fp())?;
            self.conv(ops, &u.dw, b.fp(), l, u.dil, c.fp())?;
            ops.snake(c.fp(), ch, l, &u.a2, b.fp())?;
            self.conv(ops, &u.pw, b.fp(), l, 1, c.fp())?;
            ops.add(a.fp(), c.fp(), ch * l)?;
        }
        Ok(())
    }

    /// Latents [64, t] (channels first) to samples (48 kHz)
    fn decode_once(&self, ops: &Ops, z: &[f32], t: usize) -> Result<Vec<f32>> {
        let _profile_stage = ops.stage("vae_decode");
        let g = &ops.gpu;
        let mut widest = crate::budget::product(&[self.d_pw.co, t])?;
        let mut l = t;
        for b in &self.blocks {
            l = crate::budget::product(&[l, b.stride])?;
            widest = widest.max(crate::budget::product(&[b.up.co, l])?);
        }
        crate::budget::admit(
            g,
            crate::budget::product(&[3, widest, 4])?,
            "AudioVAE workspace",
        )?;
        let (a, b, c) = (
            DevBuf::f32(g, widest)?,
            DevBuf::f32(g, widest)?,
            DevBuf::f32(g, widest)?,
        );
        b.write(0, &crate::talker_bytes(z))?;
        self.conv(ops, &self.d_in, b.fp(), t, 1, c.fp())?;
        self.conv(ops, &self.d_pw, c.fp(), t, 1, a.fp())?;
        let (mut l, mut ch) = (t, self.d_pw.co);
        for blk in &self.blocks {
            ops.affine(a.fp(), ch, l, blk.sr_scale.fp(), blk.sr_bias.fp())?;
            ops.snake(a.fp(), ch, l, &blk.act, b.fp())?;
            ops.conv_up(
                b.fp(),
                1,
                ch,
                l,
                &blk.up.w,
                blk.up.co,
                blk.up.k,
                blk.up.b.as_ref().map_or(std::ptr::null(), |x| x.fp()),
                blk.stride,
                a.fp(),
            )?;
            l *= blk.stride;
            ch = blk.up.co;
            self.units(ops, &blk.units, ch, l, &a, &b, &c)?;
        }
        ops.snake(a.fp(), ch, l, &self.d_act, b.fp())?;
        self.conv(ops, &self.d_out, b.fp(), l, 1, c.fp())?;
        ops.tanh(c.fp(), l)?;
        g.sync()?;
        c.view(0, l * 4)?.to_f32()
    }

    /// Latents [64, t] to samples, in chunks of `chunk` latents behind `ctx` of the ones before
    pub fn decode(&self, ops: &Ops, z: &[f32], chunk: usize, ctx: usize) -> Result<Vec<f32>> {
        let d = self.latent;
        if d == 0
            || z.is_empty()
            || !z.len().is_multiple_of(d)
            || chunk == 0
            || z.iter().any(|x| !x.is_finite())
        {
            return Err(Error("invalid AudioVAE decode input".into()));
        }
        let t = z.len() / d;
        if t <= chunk + ctx {
            return self.decode_once(ops, z, t);
        }
        let mut out = Vec::with_capacity(t * self.hop_out);
        let mut s = 0;
        while s < t {
            let e = (s + chunk).min(t);
            let c0 = s.saturating_sub(ctx);
            let part: Vec<f32> = (0..d)
                .flat_map(|ch| z[ch * t + c0..ch * t + e].iter().copied())
                .collect();
            let wav = self.decode_once(ops, &part, e - c0)?;
            out.extend_from_slice(&wav[(s - c0) * self.hop_out..]);
            s = e;
        }
        Ok(out)
    }

    /// Exact left context, rounded up to the encoder's sample grid. The right
    /// reach is hop-1, so a chunk ending on that grid includes every needed sample.
    pub fn encoder_context_samples(&self) -> Result<usize> {
        let mut reach = EncoderReach::new();
        reach.causal(self.e_in.k, 1)?;
        for block in &self.e_blocks {
            for unit in &block.units {
                reach.causal(unit.dw.k, unit.dil)?;
                reach.causal(unit.pw.k, 1)?;
            }
            reach.down(block.down.k, block.stride)?;
        }
        reach.causal(self.e_mu.k, 1)?;
        reach.context(self.hop_in)
    }

    /// Encode the entire recording with bounded device workspace: 125 output
    /// latents (5 seconds at the canonical 25 Hz), plus exact preceding context.
    pub fn encode(&self, ops: &Ops, wav: &[f32]) -> Result<Vec<f32>> {
        self.encode_chunked(ops, wav, 125)
    }

    /// Same encoder operations on overlapping, hop-aligned segments. Only context
    /// outputs are discarded; every requested latent is kept in channel-major order.
    pub fn encode_chunked(&self, ops: &Ops, wav: &[f32], chunk: usize) -> Result<Vec<f32>> {
        self.validate_encoder_input(wav)?;
        if chunk == 0 {
            return Err(Error("zero encoder chunk".into()));
        }
        let context = self.encoder_context_samples()? / self.hop_in;
        let frames = wav.len() / self.hop_in;
        if frames <= chunk {
            return self.encode_unchunked(ops, wav);
        }
        let size = crate::budget::product(&[self.latent, frames])?;
        let mut out = Vec::new();
        out.try_reserve_exact(size)
            .map_err(|e| Error(format!("encoder output allocation: {e}")))?;
        out.resize(size, 0.0);
        // One admission and three allocations for the entire recording. Holding
        // the workspace avoids repeated HIP allocation/free-pool churn and never
        // counts an already-live workspace as another planned allocation.
        let max_frames = frames.min(crate::budget::sum(&[chunk, context])?);
        let workspace = self
            .allocate_encoder_workspace(ops, crate::budget::product(&[max_frames, self.hop_in])?)?;
        let mut start = 0;
        while start < frames {
            let end = start + chunk.min(frames - start);
            let from = start.saturating_sub(context);
            let part = self.encode_in_workspace(
                ops,
                &wav[from * self.hop_in..end * self.hop_in],
                &workspace,
            )?;
            let part_frames = end - from;
            let skip = start - from;
            for channel in 0..self.latent {
                out[channel * frames + start..channel * frames + end].copy_from_slice(
                    &part[channel * part_frames + skip..channel * part_frames + part_frames],
                );
            }
            start = end;
        }
        Ok(out)
    }

    fn validate_encoder_input(&self, wav: &[f32]) -> Result<()> {
        let n = wav.len();
        if self.hop_in == 0
            || n == 0
            || !n.is_multiple_of(self.hop_in)
            || wav.iter().any(|v| !v.is_finite())
        {
            return Err(Error(format!("{n} samples: a multiple of {}", self.hop_in)));
        }
        // The residual additions and the final readback require these shapes;
        // reject incompatible checkpoints before allocating or enqueueing work.
        let mut channels = self.e_in.co;
        if self.e_in.ci != 1 || channels == 0 {
            return Err(Error("invalid AudioVAE encoder input channels".into()));
        }
        for block in &self.e_blocks {
            if block.stride == 0
                || block.down.k < block.stride
                || block.down.ci != channels
                || block.down.co == 0
                || block.units.iter().any(|u| {
                    u.dw.ci != channels
                        || u.dw.co != channels
                        || u.pw.ci != channels
                        || u.pw.co != channels
                })
            {
                return Err(Error("invalid AudioVAE encoder block geometry".into()));
            }
            channels = block.down.co;
        }
        if self.e_mu.ci != channels || self.e_mu.co != self.latent || self.latent == 0 {
            return Err(Error("invalid AudioVAE encoder latent geometry".into()));
        }
        Ok(())
    }

    /// Diagnostic full-recording control; production uses `encode`'s bounded chunks.
    /// Retains the original native operations and the same memory admission check.
    pub fn encode_unchunked(&self, ops: &Ops, wav: &[f32]) -> Result<Vec<f32>> {
        self.validate_encoder_input(wav)?;
        let workspace = self.allocate_encoder_workspace(ops, wav.len())?;
        self.encode_in_workspace(ops, wav, &workspace)
    }

    fn allocate_encoder_workspace(&self, ops: &Ops, n: usize) -> Result<[DevBuf; 3]> {
        let g = &ops.gpu;
        let widest = encoder_workspace(
            n,
            self.e_in.co,
            self.e_blocks.iter().map(|b| (b.down.co, b.stride)),
            self.e_mu.co,
        )?;
        crate::budget::admit(
            g,
            crate::budget::product(&[3, widest, 4])?,
            "AudioVAE workspace",
        )?;
        Ok([
            DevBuf::f32(g, widest)?,
            DevBuf::f32(g, widest)?,
            DevBuf::f32(g, widest)?,
        ])
    }

    /// Each operation writes its entire active extent. Tail bytes from a longer
    /// segment are never read; the final sync/readback completes before reuse.
    fn encode_in_workspace(
        &self,
        ops: &Ops,
        wav: &[f32],
        workspace: &[DevBuf; 3],
    ) -> Result<Vec<f32>> {
        let _profile_stage = ops.stage("vae_encode");
        let g = &ops.gpu;
        let n = wav.len();
        let needed = crate::budget::product(&[
            4,
            encoder_workspace(
                n,
                self.e_in.co,
                self.e_blocks.iter().map(|b| (b.down.co, b.stride)),
                self.e_mu.co,
            )?,
        ])?;
        if workspace.iter().any(|buffer| buffer.len < needed) {
            return Err(Error("encoder workspace is too small".into()));
        }
        let [a, b, c] = workspace;
        b.write(0, &crate::talker_bytes(wav))?;
        self.conv(ops, &self.e_in, b.fp(), n, 1, a.fp())?;
        let (mut l, mut ch) = (n, self.e_in.co);
        for blk in &self.e_blocks {
            self.units(ops, &blk.units, ch, l, a, b, c)?;
            ops.snake(a.fp(), ch, l, &blk.act, b.fp())?;
            l = ops.conv_strided(
                b.fp(),
                1,
                ch,
                l,
                &blk.down.w,
                blk.down.co,
                blk.down.k,
                blk.down.b.as_ref().map_or(std::ptr::null(), |x| x.fp()),
                blk.stride,
                a.fp(),
            )?;
            ch = blk.down.co;
        }
        self.conv(ops, &self.e_mu, a.fp(), l, 1, b.fp())?;
        g.sync()?;
        b.view(0, self.latent * l * 4)?.to_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoder_context_covers_left_and_right_reach() {
        let mut reach = EncoderReach::new();
        reach.causal(7, 1).unwrap();
        for stride in [2, 5, 8, 8] {
            for dilation in [1, 3, 9] {
                reach.causal(7, dilation).unwrap();
                reach.causal(1, 1).unwrap();
            }
            reach.down(stride * 2, stride).unwrap();
        }
        reach.causal(3, 1).unwrap();
        assert_eq!((reach.past, reach.future, reach.jump), (9272, 639, 640));
        assert_eq!(reach.context(640).unwrap(), 9600);
        assert!(reach.context(320).is_err());
        assert!(EncoderReach::new().causal(0, 1).is_err());
        assert!(EncoderReach::new().causal(7, 0).is_err());
        assert!(EncoderReach::new().down(7, 0).is_err());
        assert!(EncoderReach::new().down(3, 4).is_err());
        let mut huge = EncoderReach {
            past: usize::MAX,
            future: 0,
            jump: 1,
        };
        assert!(huge.causal(2, 1).is_err());
        let huge = EncoderReach {
            past: usize::MAX,
            future: 1,
            jump: 2,
        };
        assert!(huge.context(2).is_err());
    }

    #[test]
    fn encoder_workspace_covers_each_stage_without_doubling() {
        // 16 seconds at 16 kHz: the first stride doubles channels but halves
        // length. Subsequent stages shrink, so the original 2x reserve was unused.
        let n = 256_000;
        let blocks = [(256, 2), (512, 5), (1024, 8), (2048, 8)];
        assert_eq!(encoder_workspace(n, 128, blocks, 64).unwrap(), 128 * n);
        assert_eq!(encoder_workspace(640, 128, blocks, 64).unwrap(), 128 * 640);
        let reuse = encoder_workspace((125 + 15) * 640, 128, blocks, 64).unwrap();
        assert_eq!(crate::budget::product(&[3, reuse, 4]).unwrap(), 137_625_600);
        assert_eq!(encoder_workspace(8, 1, [(32, 2)], 64).unwrap(), 256);
        assert!(encoder_workspace(641, 128, blocks, 64).is_err());
        assert!(encoder_workspace(640, 128, [(256, 0)], 64).is_err());
        assert!(encoder_workspace(usize::MAX, 128, [], 64).is_err());
    }

    #[test]
    fn weight_norm_uses_all_nonoutput_axes() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "conv.weight_v".into(),
            Tensor {
                shape: vec![2, 1, 2],
                data: vec![3., 4., 0., 2.],
            },
        );
        tensors.insert(
            "conv.weight_g".into(),
            Tensor {
                shape: vec![2, 1, 1],
                data: vec![10., 3.],
            },
        );
        let (weights, shape) = wn(&tensors, "conv").unwrap();
        assert_eq!(shape, vec![2, 1, 2]);
        assert_eq!(weights, vec![6., 8., 0., 3.]);
    }
}
