//! The complete 12-layer CLIP ViT-L/14 text tower, causal 77-token context.
use crate::{
    graph::{Graph, Tap, Tensor},
    kernels::Activation,
    weights::Checkpoint,
};
use nextrocm_image::Result;
const PREFIX: &str = "cond_stage_model.transformer.text_model";

pub(crate) fn encode(
    g: &Graph<'_>,
    file: &Checkpoint,
    ids: &[u32; 77],
    tap: &mut Option<&mut Tap<'_>>,
) -> Result<Tensor> {
    let mut x = g.upload(&[77, 768], &file.embeddings(ids)?)?;
    g.tap("clip.embedding", &x, tap)?;
    for i in 0..12 {
        let p = format!("{PREFIX}.encoder.layers.{i}");
        let a = g.ln(&x, &format!("{p}.layer_norm1"))?;
        let q = g.linear(&a, &format!("{p}.self_attn.q_proj"), 768, true)?;
        let k = g.linear(&a, &format!("{p}.self_attn.k_proj"), 768, true)?;
        let v = g.linear(&a, &format!("{p}.self_attn.v_proj"), 768, true)?;
        let a = g.attention(&q, &k, &v, 12, true)?;
        let a = g.linear(&a, &format!("{p}.self_attn.out_proj"), 768, true)?;
        x = g.add(x, &a)?;
        let a = g.ln(&x, &format!("{p}.layer_norm2"))?;
        let a = g.linear(&a, &format!("{p}.mlp.fc1"), 3072, true)?;
        let a = g.act(a, Activation::QuickGelu)?;
        let a = g.linear(&a, &format!("{p}.mlp.fc2"), 768, true)?;
        x = g.add(x, &a)?;
        g.tap(&format!("clip.encoder.layers.{i}"), &x, tap)?;
    }
    let x = g.ln(&x, &format!("{PREFIX}.final_layer_norm"))?;
    g.tap("clip.final", &x, tap)?;
    Ok(x)
}
