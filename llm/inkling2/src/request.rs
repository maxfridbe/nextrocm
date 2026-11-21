//! Checked raw request boundary: preserve source template fields before Ask normalization.
use crate::Result;
use serde_json::{Map, Value};
const MAX_REQUEST_BYTES: usize = 1024 * 1024;
fn unsupported(s: &str) -> String {
    format!("Inkling text-only request: {s}")
}
fn text(s: &str) -> Result<()> {
    if s.contains("<|") {
        return Err(unsupported(
            "literal model controls are unsupported in user-provided text",
        ));
    }
    Ok(())
}
fn message(mut m: Map<String, Value>) -> Result<Value> {
    if m.keys().any(|k| {
        !matches!(
            k.as_str(),
            "role" | "content" | "thinking" | "reasoning_content"
        )
    }) {
        return Err(unsupported("message fields, tools or media unsupported"));
    }
    let role = m
        .get("role")
        .and_then(Value::as_str)
        .ok_or_else(|| unsupported("message role required"))?;
    if !matches!(role, "user" | "assistant" | "system") {
        return Err(unsupported("message role unsupported"));
    }
    if m.contains_key("thinking") && m.contains_key("reasoning_content") {
        return Err(unsupported("use one reasoning field"));
    }
    if let Some(v) = m.get("thinking").or_else(|| m.get("reasoning_content")) {
        if role != "assistant" {
            return Err(unsupported("reasoning belongs only to assistant messages"));
        }
        text(
            v.as_str()
                .ok_or_else(|| unsupported("reasoning must be text"))?,
        )?;
    }
    match m.get("content") {
        Some(Value::String(s)) => text(s)?,
        Some(Value::Array(parts)) if !parts.is_empty() => {
            for part in parts {
                let p = part
                    .as_object()
                    .ok_or_else(|| unsupported("content parts must be text objects"))?;
                if p.len() != 2 || p.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(unsupported(
                        "only type:text content parts supported; images unsupported",
                    ));
                }
                text(
                    p.get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| unsupported("text part missing text"))?,
                )?;
            }
        }
        _ => return Err(unsupported("content must be text or text-only parts")),
    }
    if let Some(v) = m.remove("thinking") {
        m.insert("reasoning_content".into(), v);
    }
    Ok(Value::Object(m))
}
/// Full raw OpenAI/Lines request JSON, validated before generic parsing can drop fields.
pub fn checked_request(v: &Value) -> Result<String> {
    if v.to_string().len() > MAX_REQUEST_BYTES {
        return Err(unsupported("request exceeds1MiB"));
    }
    let r = v
        .as_object()
        .ok_or_else(|| unsupported("request must be object"))?;
    for k in r.keys() {
        if !matches!(
            k.as_str(),
            "messages"
                | "model"
                | "stream"
                | "max_tokens"
                | "max_completion_tokens"
                | "temperature"
                | "top_p"
                | "seed"
                | "reasoning_effort"
                | "effort"
                | "chat_template_kwargs"
                | "enable_thinking"
                | "tools"
                | "tool_choice"
                | "parallel_tool_calls"
                | "n"
                | "request_id"
                | "logprobs"
                | "top_logprobs"
                | "top_k"
                | "min_p"
                | "logprob_chain"
                | "LogProbChain"
        ) {
            return Err(unsupported(&format!("unsupported request field {k}")));
        }
    }
    for key in ["model", "request_id"] {
        if let Some(x) = r.get(key) {
            if !x.is_string() {
                return Err(unsupported("model/request_id must be string"));
            }
        }
    }
    for key in ["stream", "logprobs"] {
        if let Some(x) = r.get(key) {
            if !x.is_boolean() {
                return Err(unsupported("stream/logprobs must be boolean"));
            }
        }
    }
    // Parser buffering can span several BPE tokens; no partial/misassigned
    // token logprob metadata is exposed until that association is implemented.
    if r.get("logprobs").is_some_and(|v| v != &Value::Bool(false))
        || r.get("top_logprobs").is_some_and(|v| v.as_u64() != Some(0))
    {
        return Err(unsupported(
            "Inkling token logprob reporting is not supported yet",
        ));
    }
    for key in ["parallel_tool_calls", "logprob_chain", "LogProbChain"] {
        if let Some(x) = r.get(key) {
            if x != &Value::Bool(false) {
                return Err(unsupported("tools/attention-chain options unsupported"));
            }
        }
    }
    if let Some(x) = r.get("tools") {
        if !x.is_null() && x.as_array().is_none_or(|a| !a.is_empty()) {
            return Err(unsupported("tools unsupported even with tool_choice:none"));
        }
    }
    if let Some(x) = r.get("tool_choice") {
        if !x.is_null() && x.as_str() != Some("none") {
            return Err(unsupported("tool choice unsupported"));
        }
    }
    if let Some(x) = r.get("n") {
        if x.as_u64() != Some(1) {
            return Err(unsupported("only n=1 supported"));
        }
    }
    if let Some(x) = r.get("seed") {
        if x.as_u64().is_none() {
            return Err(unsupported("seed must be u64"));
        }
    }
    for key in ["max_tokens", "max_completion_tokens"] {
        if let Some(x) = r.get(key) {
            if x.as_u64().is_none_or(|n| n == 0 || n > 4096) {
                return Err(unsupported("max tokens must be1..4096"));
            }
        }
    }
    if r.contains_key("max_tokens")
        && r.contains_key("max_completion_tokens")
        && r["max_tokens"] != r["max_completion_tokens"]
    {
        return Err(unsupported("conflicting token limits"));
    }
    if let Some(x) = r.get("top_logprobs") {
        if x.as_u64().is_none_or(|n| n > 20) {
            return Err(unsupported("top_logprobs must be0..20"));
        }
    }
    if let Some(x) = r.get("top_k") {
        if x.as_u64() != Some(0) {
            return Err(unsupported("full-vocabulary sampling requires top_k=0"));
        }
    }
    if let Some(x) = r.get("min_p") {
        if x.as_f64() != Some(0.0) {
            return Err(unsupported("min_p filtering unsupported"));
        }
    }
    for (key, top_p) in [("temperature", false), ("top_p", true)] {
        if let Some(x) = r.get(key) {
            let d = x
                .as_f64()
                .ok_or_else(|| unsupported("sampling parameters must be numeric"))?;
            let f = d as f32;
            if !f.is_finite()
                || d < 0.0
                || (d > 0.0 && f == 0.0)
                || (top_p && (d <= 0.0 || d > 1.0))
            {
                return Err(unsupported("invalid F32 sampling parameters"));
            }
        }
    }
    let kwargs = match r.get("chat_template_kwargs") {
        None => None,
        Some(x) => Some(
            x.as_object()
                .ok_or_else(|| unsupported("template kwargs must be object"))?,
        ),
    };
    if kwargs.is_some_and(|k| {
        k.keys()
            .any(|n| !matches!(n.as_str(), "reasoning_effort" | "enable_thinking"))
    }) {
        return Err(unsupported("unsupported template kwargs"));
    }
    let efforts = [
        r.get("reasoning_effort"),
        r.get("effort"),
        kwargs.and_then(|k| k.get("reasoning_effort")),
    ];
    let mut effort: Option<&Value> = None;
    for v in efforts.into_iter().flatten() {
        if effort.is_some_and(|e| e != v) {
            return Err(unsupported("conflicting reasoning effort fields"));
        }
        effort = Some(v);
    }
    let switches = [
        r.get("enable_thinking"),
        kwargs.and_then(|k| k.get("enable_thinking")),
    ];
    let mut switch = None;
    for v in switches.into_iter().flatten() {
        let b = v
            .as_bool()
            .ok_or_else(|| unsupported("enable_thinking must be boolean"))?;
        if switch.is_some_and(|old| old != b) {
            return Err(unsupported("conflicting thinking switches"));
        }
        switch = Some(b);
    }
    let none = Value::String("none".into());
    if switch == Some(false) {
        if effort.is_some_and(|e| e != &none && e.as_f64() != Some(0.0) && e != &Value::Bool(false))
        {
            return Err(unsupported("disabled thinking conflicts with effort"));
        }
        effort = Some(&none);
    }
    let input = r
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| unsupported("messages required"))?;
    if input.is_empty() || input.len() > 128 {
        return Err(unsupported("message count must be1..128"));
    }
    let messages = input
        .iter()
        .map(|m| {
            message(
                m.as_object()
                    .ok_or_else(|| unsupported("message must be object"))?
                    .clone(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    nextrocm_tok::inkling2_chat(&messages, effort, &[])
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn base() -> Value {
        json!({"messages":[{"role":"user","content":"Hello"}]})
    }
    #[test]
    fn raw_effort_and_lines_reasoning_preserved() {
        let v = json!({"effort":0.2,"messages":[{"role":"assistant","content":"a","thinking":"r"},{"role":"user","content":[{"type":"text","text":"x"},{"type":"text","text":"y"}]}]});
        let got = checked_request(&v).unwrap();
        assert!(got.contains("Thinking effort level: 0.2"));
        assert!(got.contains("<|content_thinking|>r<|end_message|>"));
        assert!(got.contains("<|content_text|>x<|end_message|>"));
    }
    #[test]
    fn rejects_raw_features_before_normalization() {
        for (k, v) in [
            ("tools", json!([{}])),
            ("response_format", json!({"type":"json_object"})),
            ("logit_bias", json!({})),
            ("top_k", json!(256)),
            ("top_p", json!(0)),
            ("n", json!(2)),
            ("seed", json!(-1)),
            ("max_tokens", json!(4097)),
            ("temperature", json!(1e100)),
            ("parallel_tool_calls", json!(true)),
            ("logprobs", json!(true)),
            ("top_logprobs", json!(1)),
        ] {
            let mut b = base();
            b[k] = v;
            assert!(checked_request(&b).is_err(), "{k}");
        }
        for m in [
            json!({"role":"user","content":"x","images":["url"]}),
            json!({"role":"user","content":[{"type":"image_url","image_url":"url"}]}),
            json!({"role":"assistant","content":"x","tool_calls":[]}),
        ] {
            let b = json!({"messages":[m]});
            assert!(checked_request(&b).is_err());
        }
    }
    #[test]
    fn logprob_metadata_is_explicitly_unsupported() {
        for (key, value) in [
            ("logprobs", json!(true)),
            ("logprobs", json!(1)),
            ("top_logprobs", json!(1)),
            ("top_logprobs", json!(-1)),
            ("top_logprobs", json!(0.5)),
            ("logprob_chain", json!(true)),
            ("LogProbChain", json!(true)),
        ] {
            let mut request = base();
            request[key] = value;
            assert!(checked_request(&request).is_err(), "{key}");
        }
        let mut request = base();
        request["logprobs"] = json!(false);
        request["top_logprobs"] = json!(0);
        request["logprob_chain"] = json!(false);
        request["LogProbChain"] = json!(false);
        assert_eq!(checked_request(&request), checked_request(&base()));
    }
    #[test]
    fn rejects_conflicts_and_preserves_official_default() {
        assert!(checked_request(&base())
            .unwrap()
            .contains("Thinking effort level: 0.9"));
        let mut v = base();
        v["effort"] = json!("high");
        v["reasoning_effort"] = json!("low");
        assert!(checked_request(&v).is_err());
        let mut v = base();
        v["enable_thinking"] = json!(false);
        assert!(checked_request(&v)
            .unwrap()
            .contains("Thinking effort level: 0"));
    }
}
