//! Original SD1.5 AutoencoderKL decoder, including post_quant_conv and global norms.
use crate::{
    graph::{spatial, Graph, Tap, Tensor},
    kernels::Activation,
    unet::validate_latent,
};
use nextrocm_image::{Error, Picture, Result};
const P: &str = "first_stage_model";
fn res(g: &Graph<'_>, x: &Tensor, p: &str) -> Result<Tensor> {
    let mut a = g.act(g.gn(x, &format!("{p}.norm1"))?, Activation::Silu)?;
    a = g.conv(&a, &format!("{p}.conv1"), 1, 1)?;
    a = g.act(g.gn(&a, &format!("{p}.norm2"))?, Activation::Silu)?;
    a = g.conv(&a, &format!("{p}.conv2"), 1, 1)?;
    if a.shape[1] != x.shape[1] {
        let skip = g.conv(x, &format!("{p}.nin_shortcut"), 1, 0)?;
        g.add(a, &skip)
    } else {
        g.add(a, x)
    }
}
fn attention(g: &Graph<'_>, x: &Tensor, p: &str) -> Result<Tensor> {
    let [_, _, h, w] = spatial(&x.shape)?;
    let a = g.gn(x, &format!("{p}.norm"))?;
    let q = g.tokens(&g.conv(&a, &format!("{p}.q"), 1, 0)?)?;
    let k = g.tokens(&g.conv(&a, &format!("{p}.k"), 1, 0)?)?;
    let v = g.tokens(&g.conv(&a, &format!("{p}.v"), 1, 0)?)?;
    let a = g.attention(&q, &k, &v, 1, false)?;
    let a = g.image(&a, h, w)?;
    let a = g.conv(&a, &format!("{p}.proj_out"), 1, 0)?;
    g.add(a, x)
}
pub(crate) fn decode(
    g: &Graph<'_>,
    latent: &[f32],
    h: usize,
    w: usize,
    tap: &mut Option<&mut Tap<'_>>,
) -> Result<Picture> {
    validate_latent(h, w)?;
    let scaled: Vec<_> = latent.iter().map(|v| v / 0.18215).collect();
    let mut x = g.upload(&[1, 4, h, w], &scaled)?;
    x = g.conv(&x, &format!("{P}.post_quant_conv"), 1, 0)?;
    g.tap("vae.post_quant_conv", &x, tap)?;
    x = g.conv(&x, &format!("{P}.decoder.conv_in"), 1, 1)?;
    g.tap("vae.decoder.conv_in", &x, tap)?;
    x = res(g, &x, &format!("{P}.decoder.mid.block_1"))?;
    g.tap("vae.decoder.mid.block_1", &x, tap)?;
    x = attention(g, &x, &format!("{P}.decoder.mid.attn_1"))?;
    g.tap("vae.decoder.mid.attn_1", &x, tap)?;
    x = res(g, &x, &format!("{P}.decoder.mid.block_2"))?;
    g.tap("vae.decoder.mid.block_2", &x, tap)?;
    for level in (0..4).rev() {
        for block in 0..3 {
            let p = format!("decoder.up.{level}.block.{block}");
            x = res(g, &x, &format!("{P}.{p}"))?;
            g.tap(&format!("vae.{p}"), &x, tap)?;
        }
        if level != 0 {
            let up = g.up(&x)?;
            drop(x);
            x = g.conv(&up, &format!("{P}.decoder.up.{level}.upsample.conv"), 1, 1)?;
            g.tap(&format!("vae.decoder.up.{level}.upsample"), &x, tap)?;
        }
    }
    x = g.act(
        g.gn(&x, &format!("{P}.decoder.norm_out"))?,
        Activation::Silu,
    )?;
    x = g.conv(&x, &format!("{P}.decoder.conv_out"), 1, 1)?;
    g.tap("vae.decoder.conv_out", &x, tap)?;
    rgb(&x.host()?, w * 8, h * 8)
}
/// NCHW float decoder output to RGB: clamp (x+1)/2, scale 255, round halves up.
pub fn rgb(values: &[f32], width: usize, height: usize) -> Result<Picture> {
    let pixels = width
        .checked_mul(height)
        .filter(|n| *n > 0 && *n <= 512 * 512)
        .ok_or_else(|| Error("invalid RGB geometry".into()))?;
    if values.len() != pixels * 3 || values.iter().any(|x| !x.is_finite()) {
        return Err(Error("invalid VAE RGB values".into()));
    }
    let mut data = vec![0; pixels * 3];
    for i in 0..pixels {
        for c in 0..3 {
            data[i * 3 + c] =
                (((values[c * pixels + i] + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
    Ok(Picture {
        width: width as u32,
        height: height as u32,
        channels: 3,
        data,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgb_layout_and_rounding() {
        assert_eq!(
            rgb(&[-1.0, 1.0, 0.0, -2.0, 2.0, 0.0], 2, 1).unwrap().data,
            [0, 128, 255, 255, 0, 128]
        );
        assert!(rgb(&[f32::NAN, 0.0, 0.0], 1, 1).is_err());
    }
}
