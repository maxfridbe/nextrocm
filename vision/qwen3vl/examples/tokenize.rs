//! CPU only: TOKENIZER_JSON ORIGINAL_TOKENS_JSON [PROMPT]. No runtime/GPU calls.
use nextrocm_tok::Tokenizer;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if !(3..=4).contains(&a.len()) {
        return Err("usage: tokenize TOKENIZER ORIGINAL_TOKENS [PROMPT]".into());
    }
    let prompt = a
        .get(3)
        .map(String::as_str)
        .unwrap_or("A red ceramic teapot on a wooden table.");
    let tok = Tokenizer::from_hf_json(std::path::Path::new(&a[1]))?;
    let (ids, drop) = nextrocm_vision_qwen3vl::prompt_ids(&tok, prompt)?;
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(&a[2])?)?;
    if serde_json::json!(ids) != reference["ids"] || serde_json::json!(drop) != reference["drop"] {
        return Err("original token IDs/drop mismatch".into());
    }
    if reference["image_pad"]
        .as_array()
        .is_none_or(|v| v.len() != ids.len() - drop || v.iter().any(|x| x.as_i64() != Some(0)))
    {
        return Err("image-pad mask mismatch".into());
    }
    println!(
        "CPU tokenization PASS ids={} drop={} cropped={}",
        ids.len(),
        drop,
        ids.len() - drop
    );
    Ok(())
}
