use super::*;
#[test]
fn actual_geometry_and_memory() {
    let g = Geometry {
        arch: "qwen3moe",
        layers: 48,
        width: 2048,
        ff: 768,
        heads: 32,
        kv_heads: 4,
        head: 128,
        rotary: 128,
        vocab: 151936,
        context: 262144,
        experts: 128,
        used: 8,
        eps: 1e-6,
        theta: 1e7,
    };
    g.validate().unwrap();
    assert_eq!(g.cache_bytes(4096).unwrap(), 402653184);
    assert_eq!(g.specs().len(), 579);
    let mut bad = g.clone();
    bad.kv_heads = 3;
    assert!(bad.validate().is_err());
    bad = g.clone();
    bad.used = 129;
    assert!(bad.validate().is_err());
}
#[test]
fn coder_template_preserves_text_and_has_no_think() {
    let messages = [nextrocm_tok::Message {
        role: "user",
        content: " hi ",
        reasoning: None,
        calls: &[],
    }];
    assert_eq!(
        chat::chat(&messages, nextrocm_tok::Effort::Low, &[]),
        "<|im_start|>user\n hi <|im_end|>\n<|im_start|>assistant\n"
    );
}
#[test]
fn actual_header() {
    let Some(path) = std::env::var_os("QWEN3MOE_TEST_GGUF") else {
        return;
    };
    let f = Gguf::open(std::path::Path::new(&path)).unwrap();
    let g = Geometry::read(&f).unwrap();
    let (_, bytes) = weight_plan(&f, &g).unwrap();
    assert!(bytes + g.cache_bytes(4096).unwrap() + (MARGIN as usize) < 8_000_000_000);
    assert_eq!(g.layers, 48);
    assert_eq!(g.used, 8);
    eprintln!(
        "load bytes {bytes}; 4K KV bytes {}",
        g.cache_bytes(4096).unwrap()
    );
}

#[test]
fn checkpoint_jinja_template_parity() {
    use nextrocm_tok::{Effort, Message, ToolCall};
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("../tests/template-fixtures.json")).unwrap();
    for case in fixtures.as_array().unwrap() {
        let ms = case["messages"].as_array().unwrap();
        let calls: Vec<Vec<ToolCall>> = ms
            .iter()
            .map(|m| {
                m["tool_calls"]
                    .as_array()
                    .map(|calls| {
                        calls
                            .iter()
                            .map(|c| {
                                let c = &c["function"];
                                ToolCall {
                                    name: c["name"].as_str().unwrap().into(),
                                    args: c["arguments"]
                                        .as_object()
                                        .unwrap()
                                        .iter()
                                        .map(|(k, v)| {
                                            (
                                                k.clone(),
                                                v.as_str()
                                                    .map(str::to_owned)
                                                    .unwrap_or_else(|| v.to_string()),
                                            )
                                        })
                                        .collect(),
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
                role: m["role"].as_str().unwrap(),
                content: m["content"].as_str().unwrap(),
                reasoning: None,
                calls,
            })
            .collect();
        let tools: Vec<_> = case["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.to_string())
            .collect();
        assert_eq!(
            chat::chat(&messages, Effort::Low, &tools),
            case["expected"].as_str().unwrap(),
            "{}",
            case["name"]
        );
    }
}
