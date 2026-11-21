//! Coder checkpoint's Unsloth template, including its XML tool-call syntax.
use nextrocm_tok::{Effort, Message};
use serde_json::Value;

fn string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        _ => v.to_string(),
    }
}
// HF's Jinja tojson uses Python json.dumps separators (comma-space, colon-space).
fn json_spaced(v: &Value) -> String {
    match v {
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!(
                    "{}: {}",
                    serde_json::Value::String(k.clone()),
                    json_spaced(v)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(json_spaced).collect::<Vec<_>>().join(", ")
        ),
        _ => v.to_string(),
    }
}
fn schema_value(v: &Value) -> String {
    if v.is_object() {
        json_spaced(v)
    } else {
        string(v)
    }
}
fn tag(s: &mut String, key: &str, value: &str) {
    s.push_str(&format!("\n<{key}>{value}</{key}>"));
}
fn items(s: &mut String, key: &str, value: &Value) {
    if let Some(a) = value.as_array().filter(|a| !a.is_empty()) {
        let values: Vec<_> = a
            .iter()
            .map(|v| {
                if let Some(v) = v.as_str() {
                    format!("`{v}`")
                } else {
                    string(v)
                }
            })
            .collect();
        tag(s, key, &format!("[{}]", values.join(", ")));
    }
}
fn tool(s: &mut String, v: &Value) {
    let v = v.get("function").unwrap_or(v);
    s.push_str("\n<function>");
    tag(s, "name", v["name"].as_str().unwrap_or(""));
    tag(
        s,
        "description",
        v["description"].as_str().unwrap_or("").trim(),
    );
    s.push_str("\n<parameters>");
    if let Some(params) = v["parameters"]["properties"].as_object() {
        for (name, fields) in params {
            s.push_str("\n<parameter>");
            tag(s, "name", name);
            if let Some(v) = fields.get("type") {
                tag(s, "type", &string(v));
            }
            if let Some(v) = fields.get("description") {
                tag(s, "description", string(v).trim());
            }
            items(s, "enum", &fields["enum"]);
            if let Some(fields) = fields.as_object() {
                for (k, v) in fields {
                    if !["type", "description", "enum", "required"].contains(&k.as_str()) {
                        tag(s, k, &schema_value(v));
                    }
                }
            }
            items(s, "required", &fields["required"]);
            s.push_str("\n</parameter>");
        }
    }
    items(s, "required", &v["parameters"]["required"]);
    s.push_str("\n</parameters>");
    if let Some(v) = v.get("return") {
        tag(s, "return", &schema_value(v));
    }
    s.push_str("\n</function>");
}
/// Text and structured tools use the actual checkpoint template; effort adds no thinking prefix.
pub fn chat(messages: &[Message], _effort: Effort, tools: &[String]) -> String {
    let mut s = String::new();
    let has_system = messages.first().is_some_and(|m| m.role == "system");
    if has_system {
        s.push_str("<|im_start|>system\n");
        s.push_str(messages[0].content);
    } else if !tools.is_empty() {
        s.push_str("<|im_start|>system\nYou are Qwen, a helpful AI assistant that can interact with a computer to solve tasks.");
    }
    if !tools.is_empty() {
        s.push_str("\n\nYou have access to the following functions:\n\n<tools>");
        for t in tools {
            // ChatFn is infallible. Invalid tool schemas are retained verbatim so
            // callers can diagnose their input rather than silently losing it.
            if let Ok(v) = serde_json::from_str(t) {
                tool(&mut s, &v);
            } else {
                s.push('\n');
                s.push_str(t);
            }
        }
        s.push_str("\n</tools>");
        s.push_str(include_str!("tool_format.txt"));
    }
    if has_system || !tools.is_empty() {
        s.push_str("<|im_end|>\n");
    }
    let ms = &messages[usize::from(has_system)..];
    for (i, m) in ms.iter().enumerate() {
        if m.role == "assistant" && !m.calls.is_empty() {
            s.push_str("<|im_start|>assistant");
            if !m.content.trim().is_empty() {
                s.push('\n');
                s.push_str(m.content.trim());
                s.push('\n');
            }
            for call in m.calls {
                s.push_str("\n<tool_call>\n<function=");
                s.push_str(&call.name);
                s.push_str(">\n");
                for (k, v) in &call.args {
                    s.push_str("<parameter=");
                    s.push_str(k);
                    s.push_str(">\n");
                    s.push_str(v);
                    s.push_str("\n</parameter>\n");
                }
                s.push_str("</function>\n</tool_call>");
            }
            s.push_str("<|im_end|>\n");
        } else if m.role == "tool" {
            if i > 0 && ms[i - 1].role != "tool" {
                s.push_str("<|im_start|>user\n");
            }
            s.push_str("<tool_response>\n");
            s.push_str(m.content);
            s.push_str("\n</tool_response>\n");
            if i + 1 == ms.len() || ms[i + 1].role != "tool" {
                s.push_str("<|im_end|>\n");
            }
        } else {
            s.push_str("<|im_start|>");
            s.push_str(m.role);
            s.push('\n');
            s.push_str(m.content);
            s.push_str("<|im_end|>\n");
        }
    }
    s.push_str("<|im_start|>assistant\n");
    s
}
