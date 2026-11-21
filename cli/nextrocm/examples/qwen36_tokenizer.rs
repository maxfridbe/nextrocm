//! Check the actual GGUF tokenizer and official text-template fixtures without a GPU.
use nextrocm_gguf::Gguf;
use nextrocm_tok::{qwen35_chat_for, Effort, Message, Tokenizer, ToolCall};
use serde_json::Value;
use std::{error::Error, path::Path};

fn py_json(v: &Value) -> String {
    match v {
        Value::Array(a) => format!("[{}]", a.iter().map(py_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), py_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => v.to_string(),
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("qwen36_tokenizer MODEL FIXTURE_JSON [gguf|hf|both]".into());
    }
    let f = Gguf::open(Path::new(&args[1]))?;
    let tok = Tokenizer::from_gguf(&f)?;
    let chat = qwen35_chat_for(&f)?;
    let source = args.get(3).map(String::as_str).unwrap_or("both");
    if !matches!(source, "gguf" | "hf" | "both") {
        return Err("invalid fixture source".into());
    }
    let fixture: Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let mut checked = 0;
    for row in fixture["raw"].as_array().ok_or("raw fixtures")? {
        let ids: Vec<u32> = serde_json::from_value(row["input_ids"].clone())?;
        let text = row["text"].as_str().ok_or("text")?;
        if tok.encode(text) != ids {
            return Err(format!("raw tokenizer mismatch {text:?}").into());
        }
        if Some(tok.decode(&ids).as_str()) != row["decoded"].as_str() {
            return Err("decoded mismatch".into());
        }
        checked += 1;
    }
    let mut unsupported = 0;
    for row in fixture["chat"].as_array().ok_or("chat fixtures")? {
        if source != "both" && row["source"] != source {
            continue;
        }
        let case = &row["case"];
        // ChatFn renders a generation prompt and has no error-return contract.
        // Source rejection cases and completed-conversation rendering aren't this API.
        if row.get("error").is_some() || case["add_generation_prompt"] == false {
            unsupported += 1;
            continue;
        }
        let raw = case["messages"].as_array().ok_or("messages")?;
        let mut calls = Vec::new();
        for m in raw {
            let mut list = Vec::new();
            for call in m["tool_calls"].as_array().into_iter().flatten() {
                let c = call.get("function").unwrap_or(call);
                let mut args = Vec::new();
                for (k, v) in c["arguments"].as_object().ok_or("tool arguments")? {
                    args.push((
                        k.clone(),
                        v.as_str().map(str::to_owned).unwrap_or_else(|| py_json(v)),
                    ));
                }
                list.push(ToolCall {
                    name: c["name"].as_str().ok_or("tool name")?.to_owned(),
                    args,
                });
            }
            calls.push(list);
        }
        let messages: Vec<_> = raw
            .iter()
            .zip(&calls)
            .map(|(m, c)| Message {
                role: m["role"].as_str().unwrap_or(""),
                content: m["content"].as_str().unwrap_or(""),
                reasoning: m["reasoning_content"].as_str(),
                calls: c,
            })
            .collect();
        let tools: Vec<_> = case["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .map(py_json)
            .collect();
        let effort = if case["enable_thinking"] == false {
            Effort::Off
        } else if case["enable_thinking"] == true {
            Effort::High
        } else {
            Effort::Default
        };
        let prompt = chat(&messages, effort, &tools);
        if Some(prompt.as_str()) != row["rendered_prompt"].as_str() {
            return Err(format!(
                "template mismatch {} {}: actual {prompt:?}, expected {}",
                row["source"], case["name"], row["rendered_prompt"]
            )
            .into());
        }
        let ids: Vec<u32> = serde_json::from_value(row["input_ids"].clone())?;
        if tok.encode(&prompt) != ids {
            return Err(format!("template IDs mismatch {}", case["name"]).into());
        }
        checked += 1;
    }
    println!("PASS: {checked} tokenizer/template comparisons; {unsupported} source-only rejection/completed-conversation cases outside ChatFn");
    Ok(())
}
