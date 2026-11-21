//! Inkling2 text-only output grammar from merged883286 common/parsers/inkling.cpp.
//! Reasoning extraction matches AUTO. Tool/media output is explicitly unsupported.
const HEADER: &str = "<|message_model|>";
const TEXT: &str = "<|content_text|>";
const THINK: &str = "<|content_thinking|>";
const END: &str = "<|end_message|>";
const EOS: &str = "<|content_model_end_sampling|>";
// Exact60 special spellings from pinned official tokenizer ed40e7da.
const SPECIAL: &[&str] = &[
    "<|unused|>",
    "<|endoftext|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|content_model_end_sampling|>",
    "<|unused_200007|>",
    "<|content_thinking|>",
    "<|unused_200009|>",
    "<|end_message|>",
    "<|unused_200011|>",
    "<|unused_200012|>",
    "<|unused_200013|>",
    "<|unused_200014|>",
    "<|unused_200015|>",
    "<|unused_200016|>",
    "<|unused_200017|>",
    "<|unused_200018|>",
    "<|unused_200019|>",
    "<|unused_200020|>",
    "<|unused_200021|>",
    "<|content_tool_error|>",
    "<|unused_200023|>",
    "<|content_xml|>",
    "<|unused_200025|>",
    "<|unused_200026|>",
    "<|unused_200027|>",
    "<|begin_of_text|>",
    "<|unused_200029|>",
    "<|unused_200030|>",
    "<|unused_200031|>",
    "<|unused_200032|>",
    "<|unused_200033|>",
    "<|unused_200034|>",
    "<|unused_200035|>",
    "<|unused_200036|>",
    "<|unused_200037|>",
    "<|unused_200038|>",
    "<|unused_200039|>",
    "<|unused_200040|>",
    "<|unused_200041|>",
    "<|unused_200042|>",
    "<|unused_200043|>",
    "<|unused_200044|>",
    "<|unused_200045|>",
    "<|unused_200046|>",
    "<|unused_200047|>",
    "<|unused_200048|>",
    "<|content_invoke_tool_json|>",
    "<|unused_200050|>",
    "<|unused_200051|>",
    "<|unused_200052|>",
    "<|unused_200053|>",
    "<|unused_200054|>",
    "<|unused_200055|>",
    "<|unused_200056|>",
    "<|content_invoke_tool_text|>",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Channel { Text, Thinking }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State { Header, Kind, Body(Channel), Ended, Closed, Failed }

/// Incremental additions, mapped by the engine to existing content/reasoning fields.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Inkling2Delta {
    pub content: String,
    pub reasoning: String,
    /// True only for validated EOS, never for an end-of-block or length cutoff.
    pub done: bool,
    /// Length termination left an unfinished block/header/control-token prefix.
    pub incomplete: bool,
}

/// Consumes newly decoded, stable UTF-8 chunks, not cumulative text. A caller
/// decoding individual byte-BPE tokens must preserve split UTF-8 before push;
/// do not pass lossy per-token replacement characters. Starts before the first
/// assistant header (the original Inkling2 template adds no assistant prefix).
/// Parser errors poison the instance. No tool/image fallback into ordinary text.
#[derive(Debug)]
pub struct Inkling2Parser { state: State, pending: String }
impl Default for Inkling2Parser {
    fn default() -> Self { Self { state: State::Header, pending: String::new() } }
}
impl Inkling2Parser {
    pub fn new() -> Self { Self::default() }
    fn error<T>(&mut self, message: &str) -> Result<T, String> {
        self.state = State::Failed;
        self.pending.clear();
        Err(format!("Inkling output: {message}"))
    }
    /// Emit only complete plain-text prefixes; partial special markers stay held.
    pub fn push(&mut self, chunk: &str) -> Result<Inkling2Delta, String> {
        if matches!(self.state, State::Closed | State::Failed) {
            return self.error("parser is closed or failed");
        }
        let mut delta = Inkling2Delta::default();
        for ch in chunk.chars() {
            if self.state == State::Ended { return self.error("data after EOS"); }
            self.pending.push(ch);
            match self.state {
                State::Header => {
                    if self.pending == HEADER { self.pending.clear(); self.state = State::Kind; }
                    else if self.pending == EOS { self.pending.clear(); self.state = State::Ended; }
                    else if !HEADER.starts_with(&self.pending) && !EOS.starts_with(&self.pending) {
                        return self.error("expected assistant block; role/tool/media output unsupported");
                    }
                }
                State::Kind => {
                    if self.pending == TEXT { self.pending.clear(); self.state = State::Body(Channel::Text); }
                    else if self.pending == THINK { self.pending.clear(); self.state = State::Body(Channel::Thinking); }
                    else if !TEXT.starts_with(&self.pending) && !THINK.starts_with(&self.pending) {
                        return self.error("tool names and non-text content types are unsupported");
                    }
                }
                State::Body(channel) => {
                    loop {
                        if self.pending == END {
                            self.pending.clear(); self.state = State::Header; break;
                        }
                        if SPECIAL.contains(&self.pending.as_str()) {
                            return self.error("unexpected special token inside content block");
                        }
                        if self.pending.is_empty() || SPECIAL.iter().any(|s| s.starts_with(&self.pending)) { break; }
                        let first = self.pending.chars().next().expect("nonempty pending");
                        self.pending.drain(..first.len_utf8());
                        match channel { Channel::Text => delta.content.push(first), Channel::Thinking => delta.reasoning.push(first) }
                    }
                }
                State::Ended | State::Closed | State::Failed => unreachable!(),
            }
        }
        delta.done = self.state == State::Ended;
        Ok(delta)
    }
    /// End generation: eos=true means the engine consumed token200006 without
    /// passing its spelling. EOS requires all blocks closed. A length cutoff may
    /// leave a partial body/header; it never leaks held control-token prefixes.
    pub fn finish(&mut self, eos: bool) -> Result<Inkling2Delta, String> {
        if matches!(self.state, State::Failed | State::Closed) { return self.error("parser is closed or failed"); }
        if eos && self.state != State::Ended && !(self.state == State::Header && self.pending.is_empty()) {
            return self.error("EOS before block closure");
        }
        let delta = Inkling2Delta {
            done: eos || self.state == State::Ended,
            incomplete: !matches!(self.state, State::Header | State::Ended) || !self.pending.is_empty(),
            ..Inkling2Delta::default()
        };
        self.pending.clear(); self.state = State::Closed;
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Original test-chat.cpp5550ff, version2: P=HEADER, reasoning_format=AUTO.
    fn parsed(chunks: &[&str], eos: bool) -> Result<(String,String,bool),String> {
        let mut p=Inkling2Parser::new();let(mut text,mut reasoning)=(String::new(),String::new());
        for chunk in chunks { let d=p.push(chunk)?;text.push_str(&d.content);reasoning.push_str(&d.reasoning); }
        let d=p.finish(eos)?;Ok((text,reasoning,d.done))
    }
    #[test]
    fn original_examples_at_every_chunk_boundary() {
        let cases=[
            (format!("{HEADER}{TEXT}Hello, world!\nWhat's up?{END}"),"Hello, world!\nWhat's up?",""),
            (format!("{HEADER}{THINK}I'm thinking{END}{HEADER}{TEXT}Hello, world!\nWhat's up?{END}"),"Hello, world!\nWhat's up?","I'm thinking"),
            (format!("{HEADER}{THINK}I was still thinking"),"","I was still thinking"),
            (format!("{HEADER}{THINK}I'm thinking{END}{HEADER}{TEXT}Hello, wor"),"Hello, wor","I'm thinking"),
        ];
        for (raw,text,reasoning) in cases {
            for split in 0..=raw.len() {
                let got=parsed(&[&raw[..split],&raw[split..]],false).unwrap();
                assert_eq!(got,(text.into(),reasoning.into(),false));
            }
            let chars:Vec<_>=raw.char_indices().map(|(i,c)|&raw[i..i+c.len_utf8()]).collect();
            assert_eq!(parsed(&chars,false).unwrap(),(text.into(),reasoning.into(),false));
        }
    }
    #[test]
    fn eos_only_stops_turn_and_no_markers_leak() {
        let mut p=Inkling2Parser::new();
        let d=p.push(&format!("{HEADER}{THINK}r{END}{HEADER}{TEXT}a{END}")).unwrap();
        assert_eq!(d.reasoning,"r");assert_eq!(d.content,"a");assert!(!d.done);
        assert!(p.push(EOS).unwrap().done);assert!(p.finish(true).unwrap().done);
        assert!(p.push("tail").is_err());
        assert!(parsed(&[&format!("{HEADER}{TEXT}a")],true).is_err());
    }
    #[test]
    fn utf8_and_noncontrol_angle_text_preserved() {
        let raw=format!("{HEADER}{TEXT}雪🦀 <b> x <|ordinary|> y{END}{EOS}");
        let chunks:Vec<_>=raw.char_indices().map(|(i,c)|&raw[i..i+c.len_utf8()]).collect();
        assert_eq!(parsed(&chunks,true).unwrap(),("雪🦀 <b> x <|ordinary|> y".into(),"".into(),true));
    }
    #[test]
    fn tools_roles_media_and_unknown_specials_fail_closed() {
        for raw in [format!("{HEADER}special_function<|content_invoke_tool_json|>{{}}{END}"),
                    format!("{HEADER}<|content_image|><|unused_200054|>{END}"),
                    format!("<|message_user|>{TEXT}bad{END}"),
                    format!("{HEADER}{TEXT}bad<|message_system|>"),
                    format!("{HEADER}{TEXT}bad<|endoftext|>")] {
            let mut p=Inkling2Parser::new();assert!(p.push(&raw).is_err());assert!(p.push("more").is_err());
        }
    }
    #[test]
    fn partial_controls_are_never_flushed_at_length_stop() {
        for token in SPECIAL {
            for n in 1..token.len() {
                let raw=format!("{HEADER}{TEXT}safe{}",&token[..n]);
                let mut p=Inkling2Parser::new();let d=p.push(&raw).unwrap();
                assert_eq!(d.content,"safe");let d=p.finish(false).unwrap();assert!(d.incomplete);assert!(!d.done);assert!(d.content.is_empty());
            }
        }
    }
    #[test]
    fn actual_bounded_native_prompt_stays_in_reasoning() {
        // Parent native "Say hello." capture:8 IDs, ends by length not EOS.
        let raw = "<|message_model|><|content_thinking|>The user asks to say hello";
        let mut p = Inkling2Parser::new();
        let d = p.push(raw).unwrap();
        assert!(d.content.is_empty());
        assert_eq!(d.reasoning, "The user asks to say hello");
        assert!(!d.done);
        let d = p.finish(false).unwrap();
        assert!(!d.done);assert!(d.incomplete);
    }
    #[test]
    fn repeated_blocks_concatenate_without_invented_whitespace() {
        let raw=format!("{HEADER}{TEXT}a{END}{HEADER}{TEXT}b{END}{HEADER}{THINK}c{END}{HEADER}{THINK}d{END}");
        assert_eq!(parsed(&[&raw],true).unwrap(),("ab".into(),"cd".into(),true));
    }
}
