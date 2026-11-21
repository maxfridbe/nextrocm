//! Validate the original architecture's tensor geometry before device allocation.
use nextrocm_engine_core::{Error, Result};
use nextrocm_gguf::{GType, Gguf};
fn dim(g: &Gguf, key: &str) -> Result<usize> {
    let n = g.meta(key).and_then(|v| v.as_u64()).unwrap_or(0);
    if n == 0 || n > 1_000_000 {
        return Err(Error(format!("invalid geometry {key}: {n}")));
    }
    Ok(n as usize)
}
fn tensor(g: &Gguf, name: &str, shape: &[usize]) -> Result<()> {
    let t = g
        .tensor(name)
        .ok_or_else(|| Error(format!("missing tensor {name}")))?;
    let want: Vec<u64> = shape.iter().map(|&n| n as u64).collect();
    if t.shape != want || !matches!(t.ty, GType::BF16 | GType::F32) {
        return Err(Error(format!(
            "{name}: {:?} {}, expected {want:?} BF16/F32",
            t.shape,
            t.ty.name()
        )));
    }
    Ok(())
}
pub fn validate(m: &Gguf, v: &Gguf) -> Result<()> {
    let e = dim(m, "paddleocr.embedding_length")?;
    let heads = dim(m, "paddleocr.attention.head_count")?;
    let kv = dim(m, "paddleocr.attention.head_count_kv")?;
    let hd = m
        .arch_meta("attention.key_length")
        .and_then(|v| v.as_u64())
        .unwrap_or((e / heads) as u64) as usize;
    let ff = dim(m, "paddleocr.feed_forward_length")?;
    let layers = dim(m, "paddleocr.block_count")?;
    if e != 1024
        || heads > 256
        || kv > heads
        || heads % kv != 0
        || heads / kv > 16
        || hd == 0
        || hd > 256
        || hd % 16 != 0
        || ff > 65536
        || layers > 128
    {
        return Err(Error("unsupported PaddleOCR decoder geometry".into()));
    }
    let emb = m
        .tensor("token_embd.weight")
        .ok_or_else(|| Error("no embeddings".into()))?;
    let vocab = *emb
        .shape
        .first()
        .ok_or_else(|| Error("empty embedding shape".into()))? as usize;
    if vocab <= crate::IMAGE_TOKEN as usize || vocab > 1_000_000 || emb.ty != GType::BF16 {
        return Err(Error("invalid BF16 vocabulary".into()));
    }
    tensor(m, "token_embd.weight", &[vocab, e])?;
    tensor(m, "output_norm.weight", &[e])?;
    if m.tensor("output.weight").is_some() {
        tensor(m, "output.weight", &[vocab, e])?;
    }
    for l in 0..layers {
        for (name, shape) in [
            ("attn_norm", vec![e]),
            ("ffn_norm", vec![e]),
            ("attn_q", vec![heads * hd, e]),
            ("attn_k", vec![kv * hd, e]),
            ("attn_v", vec![kv * hd, e]),
            ("attn_output", vec![e, heads * hd]),
            ("ffn_gate", vec![ff, e]),
            ("ffn_up", vec![ff, e]),
            ("ffn_down", vec![e, ff]),
        ] {
            tensor(m, &format!("blk.{l}.{name}.weight"), &shape)?;
        }
    }
    let c = dim(v, "clip.vision.embedding_length")?;
    let h = dim(v, "clip.vision.attention.head_count")?;
    let f = dim(v, "clip.vision.feed_forward_length")?;
    let n = dim(v, "clip.vision.block_count")?;
    if c > 8192 || h > 256 || c % h != 0 || (c / h) % 4 != 0 || f > 65536 || n > 128 {
        return Err(Error("unsupported vision geometry".into()));
    }
    // GGUF exporters may retain the original patch-convolution dimensions.
    let patch = v
        .tensor("v.patch_embd.weight")
        .ok_or_else(|| Error("missing patch weights".into()))?;
    if patch.shape != vec![c as u64, 3, 14, 14] && patch.shape != vec![c as u64, 588] {
        return Err(Error(format!("patch shape {:?}", patch.shape)));
    }
    let pos = v
        .tensor("v.position_embd.weight")
        .ok_or_else(|| Error("missing positions".into()))?;
    if pos.shape.len() != 2 || pos.shape[1] != c as u64 {
        return Err(Error("invalid position grid".into()));
    }
    let side = (pos.shape[0] as f64).sqrt() as u64;
    if side == 0 || side * side != pos.shape[0] {
        return Err(Error("position grid is not square".into()));
    }
    for name in [
        "v.patch_embd.bias",
        "v.post_ln.weight",
        "v.post_ln.bias",
        "mm.input_norm.weight",
        "mm.input_norm.bias",
    ] {
        tensor(v, name, &[c])?;
    }
    for (name, shape) in [
        ("mm.1.weight", vec![4 * c, 4 * c]),
        ("mm.1.bias", vec![4 * c]),
        ("mm.2.weight", vec![1024, 4 * c]),
        ("mm.2.bias", vec![1024]),
    ] {
        tensor(v, name, &shape)?;
    }
    for l in 0..n {
        for norm in ["ln1", "ln2"] {
            for suffix in ["weight", "bias"] {
                tensor(v, &format!("v.blk.{l}.{norm}.{suffix}"), &[c])?;
            }
        }
        for (name, rows, cols) in [
            ("attn_q", c, c),
            ("attn_k", c, c),
            ("attn_v", c, c),
            ("attn_out", c, c),
            ("ffn_up", f, c),
            ("ffn_down", c, f),
        ] {
            tensor(v, &format!("v.blk.{l}.{name}.weight"), &[rows, cols])?;
            tensor(v, &format!("v.blk.{l}.{name}.bias"), &[rows])?;
        }
    }
    Ok(())
}

/// Conservative resident + one upload scratch bound, not a measured admission peak.
pub fn load_bytes(m: &Gguf, v: &Gguf, decoder_f32: bool) -> Result<u64> {
    let elements = |t: &nextrocm_gguf::Tensor| {
        t.shape
            .iter()
            .try_fold(1u64, |a, &b| a.checked_mul(b))
            .ok_or_else(|| Error("tensor element overflow".into()))
    };
    let mut total = 0u64;
    let mut staging = 0;
    for (g, vision) in [(m, false), (v, true)] {
        for t in &g.tensors {
            let bytes = elements(t)?
                .checked_mul(
                    if vision
                        || t.name.contains("norm")
                        || (decoder_f32 && t.name != "token_embd.weight")
                    {
                        4
                    } else {
                        2
                    },
                )
                .ok_or_else(|| Error("tensor byte overflow".into()))?;
            total = total
                .checked_add(bytes)
                .ok_or_else(|| Error("load bytes overflow".into()))?;
            staging = staging.max(t.bytes);
        }
    }
    if m.tensor("output.weight").is_none() {
        let t = m
            .tensor("token_embd.weight")
            .ok_or_else(|| Error("no embedding".into()))?;
        total = total
            .checked_add(elements(t)? * if decoder_f32 { 4 } else { 2 })
            .ok_or_else(|| Error("tied head overflow".into()))?;
    }
    total
        .checked_add(staging)
        .ok_or_else(|| Error("load plan overflow".into()))
}
