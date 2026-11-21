//! Inkling token-to-text bridge. Never decode BPE bytes lossily.
use crate::{Inkling2Delta, Inkling2Parser, Tokenizer};
type Result<T> = std::result::Result<T, String>;
#[derive(Default)]
pub struct Inkling2TextOutput {
    parser: Inkling2Parser,
    pending_utf8: Vec<u8>,
    closed: bool,
}
impl Inkling2TextOutput {
    pub fn new() -> Self {
        Self::default()
    }
    fn fail<T>(&mut self, message: impl Into<String>) -> Result<T> {
        self.closed = true;
        self.pending_utf8.clear();
        Err(message.into())
    }
    pub fn push_token(
        &mut self,
        tokenizer: &Tokenizer,
        token: u32,
    ) -> Result<Inkling2Delta> {
        if self.closed {
            return Err("Inkling output is closed/failed".into());
        }
        if token >= 200058 || token == 200054 {
            return self.fail("invalid/media output token");
        }
        if token == 200006 {
            if !self.pending_utf8.is_empty() {
                return self.fail("EOS inside incomplete UTF-8");
            }
            self.closed = true;
            return self.parser.finish(true);
        }
        self.push_bytes(&tokenizer.decode_bytes(&[token]))
    }
    fn push_bytes(&mut self, bytes: &[u8]) -> Result<Inkling2Delta> {
        if self.closed {
            return Err("Inkling output is closed/failed".into());
        }
        self.pending_utf8.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending_utf8) {
            Ok(_) => self.pending_utf8.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => return self.fail("invalid UTF-8 in Inkling output"),
        };
        let text = std::str::from_utf8(&self.pending_utf8[..valid]).map_err(|e| e.to_string())?;
        let result = self.parser.push(text);
        self.pending_utf8.drain(..valid);
        match result {
            Ok(delta) => {
                if delta.done {
                    return self.fail("EOS spelling without token200006");
                }
                Ok(delta)
            }
            Err(e) => self.fail(e),
        }
    }
    pub fn finish_length(&mut self) -> Result<Inkling2Delta> {
        if self.closed {
            return Err("Inkling output is closed/failed".into());
        }
        self.closed = true;
        let mut delta = self.parser.finish(false)?;
        delta.incomplete |= !self.pending_utf8.is_empty();
        self.pending_utf8.clear();
        Ok(delta)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn split_utf8_never_emits_replacement_or_markers() {
        let mut p = Inkling2TextOutput::new();
        assert!(p
            .push_bytes(b"<|message_model|><|content_text|>")
            .unwrap()
            .content
            .is_empty());
        let bytes = "雪".as_bytes();
        assert!(p.push_bytes(&bytes[..1]).unwrap().content.is_empty());
        assert!(p.push_bytes(&bytes[1..2]).unwrap().content.is_empty());
        assert_eq!(p.push_bytes(&bytes[2..]).unwrap().content, "雪");
        assert!(p.push_bytes(b"<|end_message|>").unwrap().content.is_empty());
        assert!(!p.finish_length().unwrap().done);
    }
    #[test]
    fn invalid_utf8_poison_and_incomplete_length() {
        let mut p = Inkling2TextOutput::new();
        p.push_bytes(b"<|message_model|><|content_thinking|>")
            .unwrap();
        p.push_bytes(&[0xe9]).unwrap();
        let d = p.finish_length().unwrap();
        assert!(d.incomplete && !d.done && d.reasoning.is_empty());
        assert!(p.push_bytes(b"x").is_err());
        let mut p = Inkling2TextOutput::new();
        assert!(p.push_bytes(&[0xff]).is_err());
        assert!(p.push_bytes(b"x").is_err());
    }
}
