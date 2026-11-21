//! CPU-only official-tokenizer comparison against the actual GGUF.
use nextrocm_gguf::Gguf;
use nextrocm_tok::{Effort, Message, Tokenizer, ToolCall};
use serde_json::{json, Value};
use std::path::Path;
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("tokenizer_parity MODEL FIXTURES.json".into());
    }
    let g = Gguf::open(Path::new(&args[1]))?;
    let tok = Tokenizer::from_gguf(&g)?;
    let f: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let mut failures = Vec::new();
    let mut count = 0;
    for (i, c) in f["raw"].as_array().ok_or("missing raw")?.iter().enumerate() {
        let text = c["text"].as_str().ok_or("missing text")?;
        let ids: Vec<u32> = serde_json::from_value(c["input_ids"].clone())?;
        let actual = tok.encode(text);
        if actual != ids {
            failures.push(
                json!({"kind":"raw_ids","index":i,"text":text,"expected":ids,"actual":actual}),
            );
        }
        if tok.decode(&ids) != c["decoded"].as_str().ok_or("missing decoded")? {
            failures.push(json!({"kind":"decode","index":i}));
        }
        count += 1;
    }
    for c in f["chat"].as_array().ok_or("missing chat")? {
        let case = &c["case"];
        let ms = case["messages"].as_array().ok_or("messages")?;
        let calls: Vec<Vec<ToolCall>> = ms
            .iter()
            .map(|m| {
                m["tool_calls"]
                    .as_array()
                    .map(|cs| {
                        cs.iter()
                            .map(|c| {
                                let c = &c["function"];
                                ToolCall {
                                    name: c["name"].as_str().unwrap_or("").into(),
                                    args: c["arguments"]
                                        .as_object()
                                        .map(|a| {
                                            a.iter()
                                                .map(|(k, v)| {
                                                    (
                                                        k.clone(),
                                                        v.as_str()
                                                            .map(str::to_owned)
                                                            .unwrap_or_else(|| v.to_string()),
                                                    )
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default(),
                                }
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();
        let messages: Vec<_> = ms
            .iter()
            .zip(&calls)
            .map(|(m, calls)| Message {
                role: m["role"].as_str().unwrap_or(""),
                content: m["content"].as_str().unwrap_or(""),
                reasoning: None,
                calls,
            })
            .collect();
        let tools: Vec<_> = case["tools"]
            .as_array()
            .ok_or("tools")?
            .iter()
            .map(Value::to_string)
            .collect();
        let rendered = match nextrocm_llm_qwen3moe::kind().chat {
            nextrocm_llm::ChatTemplate::Legacy(format) => format(&messages, Effort::Low, &tools),
            nextrocm_llm::ChatTemplate::Checked(format) => format(&json!({
                "messages": case["messages"],
                "tools": case["tools"],
                "reasoning_effort": "low"
            }))?,
        };
        let expected: Vec<u32> = serde_json::from_value(c["input_ids"].clone())?;
        if Some(rendered.as_str()) != c["rendered_prompt"].as_str() {
            failures.push(json!({"kind":"render","case":case["name"],"actual":rendered}));
        }
        if tok.encode(&rendered) != expected {
            failures.push(json!({"kind":"rendered_ids","case":case["name"],"actual":tok.encode(&rendered),"expected":expected}));
        }
        count += 1;
    }
    println!(
        "{}",
        json!({"cases":count,"raw":f["raw"].as_array().map(Vec::len),"chat":f["chat"].as_array().map(Vec::len),"failures":failures,"gpu_opened":false})
    );
    if !failures.is_empty() {
        return Err("tokenizer parity failed".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1)
    }
}
