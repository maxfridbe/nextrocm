//! CPU-only check against tooling/stablelm-reference.py; no GPU is opened.
use nextrocm_gguf::Gguf;
use nextrocm_tok::{Effort, Message, Tokenizer};
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("tokenizer_parity FIXTURE_DIR")?,
    );
    let mut count = 0;
    for name in [
        "stablelm-zephyr-3b",
        "stable-code-3b",
        "stable-code-instruct-3b",
    ] {
        let fixture: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.json")))?)?;
        let f = Gguf::open(&dir.join(fixture["model"].as_str().ok_or("missing model")?))?;
        println!("{}", nextrocm_llm_stablelm::info(&f)?);
        let tok = Tokenizer::from_gguf(&f)?;
        for (i, c) in fixture["cases"]
            .as_array()
            .ok_or("missing cases")?
            .iter()
            .enumerate()
        {
            let text = c["text"].as_str().ok_or("missing text")?;
            let expected: Vec<u32> = serde_json::from_value(c["ids"].clone())?;
            let actual = tok.encode(text);
            if actual != expected {
                return Err(format!(
                    "{name} case {i} {text:?}: actual {actual:?}, expected {expected:?}"
                )
                .into());
            }
            count += 1;
        }
        let chat = nextrocm_llm_stablelm::chat_for(&f)?;
        for c in fixture["chats"].as_array().ok_or("missing chats")? {
            let messages: Vec<_> = c["messages"]
                .as_array()
                .ok_or("messages")?
                .iter()
                .map(|m| Message {
                    role: m["role"].as_str().unwrap_or(""),
                    content: m["content"].as_str().unwrap_or(""),
                    ..Default::default()
                })
                .collect();
            let prompt = chat(&messages, Effort::Default, &[]);
            if Some(prompt.as_str()) != c["prompt"].as_str() {
                return Err(
                    format!("{name}: template mismatch {prompt:?} vs {}", c["prompt"]).into(),
                );
            }
            let expected: Vec<u32> = serde_json::from_value(c["ids"].clone())?;
            if tok.encode(&prompt) != expected {
                return Err(format!("{name}: template token mismatch").into());
            }
            count += 1;
        }
    }
    println!("PASS: {count} CPU tokenizer/template cases against pinned HF tokenizer and actual GGUF headers");
    Ok(())
}
