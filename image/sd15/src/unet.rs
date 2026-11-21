//! SD1.5 epsilon UNet: twelve down/skip stages, middle, twelve up stages.
use crate::{
    graph::{spatial, Graph, Tap, Tensor},
    kernels::Activation,
    schedule,
};
use nextrocm_image::{Error, Result};
const P: &str = "model.diffusion_model";

fn res(g: &Graph<'_>, x: &Tensor, e: &Tensor, p: &str) -> Result<Tensor> {
    let a = g.act(g.gn(x, &format!("{p}.in_layers.0"))?, Activation::Silu)?;
    let a = g.conv(&a, &format!("{p}.in_layers.2"), 1, 1)?;
    let channels = a.shape[1];
    // e is SiLU(time_embed), shared by all residual blocks.
    let t = g.linear(e, &format!("{p}.emb_layers.1"), channels, true)?;
    let a = g.time(a, &t)?;
    let a = g.act(g.gn(&a, &format!("{p}.out_layers.0"))?, Activation::Silu)?;
    let a = g.conv(&a, &format!("{p}.out_layers.3"), 1, 1)?;
    if x.shape[1] != channels {
        let skip = g.conv(x, &format!("{p}.skip_connection"), 1, 0)?;
        g.add(a, &skip)
    } else {
        g.add(a, x)
    }
}
fn cross(g: &Graph<'_>, x: &Tensor, ctx: &Tensor, p: &str) -> Result<Tensor> {
    let c = x.shape[1];
    let q = g.linear(x, &format!("{p}.to_q"), c, false)?;
    let k = g.linear(ctx, &format!("{p}.to_k"), c, false)?;
    let v = g.linear(ctx, &format!("{p}.to_v"), c, false)?;
    let a = g.attention(&q, &k, &v, 8, false)?;
    g.linear(&a, &format!("{p}.to_out.0"), c, true)
}
fn transformer(g: &Graph<'_>, x: &Tensor, context: &Tensor, p: &str) -> Result<Tensor> {
    let [_, c, h, w] = spatial(&x.shape)?;
    let a = g.gn(x, &format!("{p}.norm"))?;
    let a = g.conv(&a, &format!("{p}.proj_in"), 1, 0)?;
    let mut a = g.tokens(&a)?;
    let t = format!("{p}.transformer_blocks.0");
    let z = g.ln(&a, &format!("{t}.norm1"))?;
    let z = cross(g, &z, &z, &format!("{t}.attn1"))?;
    a = g.add(a, &z)?;
    let z = g.ln(&a, &format!("{t}.norm2"))?;
    let z = cross(g, &z, context, &format!("{t}.attn2"))?;
    a = g.add(a, &z)?;
    let z = g.ln(&a, &format!("{t}.norm3"))?;
    let z = g.linear(&z, &format!("{t}.ff.net.0.proj"), 8 * c, true)?;
    let z = g.geglu(&z)?;
    let z = g.linear(&z, &format!("{t}.ff.net.2"), c, true)?;
    a = g.add(a, &z)?;
    let a = g.image(&a, h, w)?;
    let a = g.conv(&a, &format!("{p}.proj_out"), 1, 0)?;
    g.add(a, x)
}
/// Shapes of the twelve saved input stages, including downsample stages.
pub fn skip_shapes(height: usize, width: usize) -> Result<Vec<[usize; 4]>> {
    validate_latent(height, width)?;
    let mut out = vec![[1, 320, height, width]];
    let (mut h, mut w) = (height, width);
    for (level, c) in [320, 640, 1280, 1280].into_iter().enumerate() {
        out.push([1, c, h, w]);
        out.push([1, c, h, w]);
        if level != 3 {
            h /= 2;
            w /= 2;
            out.push([1, c, h, w]);
        }
    }
    Ok(out)
}
pub(crate) fn validate_latent(h: usize, w: usize) -> Result<()> {
    if h == 0 || w == 0 || h % 8 != 0 || w % 8 != 0 || h > 64 || w > 64 {
        return Err(Error(
            "SD1.5 latent dimensions must be multiples of 8 in 8..=64".into(),
        ));
    }
    Ok(())
}
pub(crate) fn forward(
    g: &Graph<'_>,
    latent: &[f32],
    h: usize,
    w: usize,
    t: f32,
    context: &[f32],
    tap: &mut Option<&mut Tap<'_>>,
) -> Result<Vec<f32>> {
    validate_latent(h, w)?;
    let input = g.upload(&[1, 4, h, w], latent)?;
    let ctx = g.upload(&[77, 768], context)?;
    let e = g.upload(&[1, 320], &schedule::embedding(t)?)?;
    let e = g.act(
        g.linear(&e, &format!("{P}.time_embed.0"), 1280, true)?,
        Activation::Silu,
    )?;
    let e = g.linear(&e, &format!("{P}.time_embed.2"), 1280, true)?;
    if let Some(f) = tap.as_mut() {
        f("unet.time_embed", &[1280], &e.host()?)?;
    }
    let e = g.act(e, Activation::Silu)?;
    let mut skips = Vec::with_capacity(12);
    let x = g.conv(&input, &format!("{P}.input_blocks.0.0"), 1, 1)?;
    drop(input);
    g.tap("unet.input_blocks.0", &x, tap)?;
    skips.push(x);
    for i in 1..12 {
        let prev = skips
            .last()
            .ok_or_else(|| Error("empty UNet skip stack".into()))?;
        let p = format!("{P}.input_blocks.{i}");
        let mut x = if i % 3 == 0 {
            g.conv(prev, &format!("{p}.0.op"), 2, 1)?
        } else {
            res(g, prev, &e, &format!("{p}.0"))?
        };
        if i % 3 != 0 && i < 9 {
            x = transformer(g, &x, &ctx, &format!("{p}.1"))?;
        }
        g.tap(&format!("unet.input_blocks.{i}"), &x, tap)?;
        skips.push(x);
    }
    let last = skips
        .last()
        .ok_or_else(|| Error("missing final skip".into()))?;
    let mut x = res(g, last, &e, &format!("{P}.middle_block.0"))?;
    g.tap("unet.middle_block.0", &x, tap)?;
    x = transformer(g, &x, &ctx, &format!("{P}.middle_block.1"))?;
    g.tap("unet.middle_block.1", &x, tap)?;
    x = res(g, &x, &e, &format!("{P}.middle_block.2"))?;
    g.tap("unet.middle_block.2", &x, tap)?;
    for i in 0..12 {
        let skip = skips
            .pop()
            .ok_or_else(|| Error("UNet skip underflow".into()))?;
        let combined = g.concat(&x, &skip)?;
        drop(skip);
        drop(x);
        let p = format!("{P}.output_blocks.{i}");
        x = res(g, &combined, &e, &format!("{p}.0"))?;
        drop(combined);
        if i >= 3 {
            x = transformer(g, &x, &ctx, &format!("{p}.1"))?;
        }
        if i % 3 == 2 && i < 11 {
            let up = if i == 2 { 1 } else { 2 };
            x = g.conv(&g.up(&x)?, &format!("{p}.{up}.conv"), 1, 1)?;
        }
        g.tap(&format!("unet.output_blocks.{i}"), &x, tap)?;
    }
    let x = g.act(g.gn(&x, &format!("{P}.out.0"))?, Activation::Silu)?;
    let x = g.conv(&x, &format!("{P}.out.2"), 1, 1)?;
    g.tap("unet.out", &x, tap)?;
    x.host()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn skip_geometry() {
        let x = skip_shapes(64, 64).unwrap();
        assert_eq!(x.len(), 12);
        assert_eq!(x[0], [1, 320, 64, 64]);
        assert_eq!(x[3], [1, 320, 32, 32]);
        assert_eq!(x[9], [1, 1280, 8, 8]);
        assert_eq!(x[11], [1, 1280, 8, 8]);
        assert_eq!(skip_shapes(16, 16).unwrap()[11], [1, 1280, 2, 2]);
        assert!(skip_shapes(17, 16).is_err());
        assert!(skip_shapes(0, 0).is_err());
    }
}
