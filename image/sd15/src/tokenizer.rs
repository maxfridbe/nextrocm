//! CLIP byte BPE with pinned sd.cpp normalization (ASCII case/whitespace only).
//! No FTFY, HTML entity decoding, prompt weighting, truncation or extra chunks.
use nextrocm_image::{Error, Result};
use regex::Regex;
use std::{collections::HashMap, path::Path};

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    bytes: [char; 256],
    pattern: Regex,
}
impl Tokenizer {
    /// Load original CLIP vocab.json and merges.txt, checking the complete ID space.
    pub fn load(vocab: &Path, merges: &Path) -> Result<Self> {
        let v = std::fs::read(vocab).map_err(|e| Error(e.to_string()))?;
        if v.len() > 4 << 20 {
            return Err(Error("oversized CLIP vocabulary".into()));
        }
        let vocab: HashMap<String, u32> =
            serde_json::from_slice(&v).map_err(|e| Error(e.to_string()))?;
        if vocab.len() != 49408
            || vocab.get("<|startoftext|>") != Some(&49406)
            || vocab.get("<|endoftext|>") != Some(&49407)
        {
            return Err(Error("not the SD1.5 CLIP vocabulary".into()));
        }
        let mut ids: Vec<_> = vocab.values().copied().collect();
        ids.sort_unstable();
        if ids.iter().enumerate().any(|(i, &id)| i != id as usize) {
            return Err(Error(
                "CLIP vocabulary IDs must be unique and contiguous".into(),
            ));
        }
        let m = std::fs::read_to_string(merges).map_err(|e| Error(e.to_string()))?;
        if m.len() > 4 << 20 {
            return Err(Error("oversized CLIP merges".into()));
        }
        let mut ranks = HashMap::new();
        for line in m.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.len() != 2 {
                return Err(Error("invalid CLIP merge".into()));
            }
            let pair = (fields[0].to_owned(), fields[1].to_owned());
            if !vocab.contains_key(&format!("{}{}", pair.0, pair.1)) {
                return Err(Error("merge is absent from CLIP vocab".into()));
            }
            let rank = ranks.len();
            if ranks.insert(pair, rank).is_some() {
                return Err(Error("duplicate CLIP merge".into()));
            }
        }
        if ranks.len() != 48894 {
            return Err(Error(format!(
                "CLIP needs 48894 merges, got {}",
                ranks.len()
            )));
        }
        let mut bytes = ['\0'; 256];
        let mut extra = 256u32;
        for (i, ch) in bytes.iter_mut().enumerate() {
            let code = if (33..=126).contains(&i)
                || (161..=172).contains(&i)
                || (174..=255).contains(&i)
            {
                i as u32
            } else {
                let n = extra;
                extra += 1;
                n
            };
            *ch = char::from_u32(code).ok_or_else(|| Error("invalid byte alphabet".into()))?;
        }
        let pattern=Regex::new(r"<\|startoftext\|>|<\|endoftext\|>|(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+").map_err(|e|Error(e.to_string()))?;
        Ok(Self {
            vocab,
            ranks,
            bytes,
            pattern,
        })
    }
    /// Exactly one context: BOS, at most 75 BPE pieces, EOS, then EOS padding.
    pub fn encode(&self, text: &str) -> Result<[u32; 77]> {
        if text.len() > 16_384 {
            return Err(Error("SD1.5 prompt exceeds bounded input length".into()));
        }
        if text.contains(['(', ')', '[', ']']) || text.split_whitespace().any(|s| s == "BREAK") {
            return Err(Error(
                "SD1.5 weighted prompts and BREAK/chunk syntax are not implemented".into(),
            ));
        }
        let text = normalize(text);
        let mut out = [49407; 77];
        out[0] = 49406;
        let mut at = 1;
        for matched in self.pattern.find_iter(&text) {
            let word = matched.as_str();
            if matches!(word, "<|startoftext|>" | "<|endoftext|>") {
                if at >= 76 {
                    return Err(Error("SD1.5 prompt exceeds 75 tokens".into()));
                }
                out[at] = self.vocab[word];
                at += 1;
                continue;
            }
            let mut pieces: Vec<String> = word
                .as_bytes()
                .iter()
                .map(|b| self.bytes[*b as usize].to_string())
                .collect();
            if let Some(last) = pieces.last_mut() {
                last.push_str("</w>");
            }
            while pieces.len() > 1 {
                let best = pieces
                    .windows(2)
                    .filter_map(|p| {
                        self.ranks
                            .get(&(p[0].clone(), p[1].clone()))
                            .map(|&r| (r, p[0].clone(), p[1].clone()))
                    })
                    .min_by_key(|x| x.0);
                let Some((_, a, b)) = best else { break };
                let mut merged = Vec::with_capacity(pieces.len());
                let mut i = 0;
                while i < pieces.len() {
                    if i + 1 < pieces.len() && pieces[i] == a && pieces[i + 1] == b {
                        merged.push(format!("{a}{b}"));
                        i += 2;
                    } else {
                        merged.push(pieces[i].clone());
                        i += 1;
                    }
                }
                pieces = merged;
            }
            for p in pieces {
                if at >= 76 {
                    return Err(Error(
                        "SD1.5 prompt exceeds 75 tokens; long/chunked prompts are not implemented"
                            .into(),
                    ));
                }
                out[at] = *self
                    .vocab
                    .get(&p)
                    .ok_or_else(|| Error(format!("CLIP missing BPE piece {p:?}")))?;
                at += 1;
            }
        }
        Ok(out)
    }
}
/// sd.cpp's C-locale bytewise tolower and ASCII whitespace collapse.
pub fn normalize(text: &str) -> String {
    text.split(|c: char| c.is_ascii_whitespace())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sdcpp_normalization_is_not_ftfy() {
        assert_eq!(normalize("  A\tCAT\nÉ &amp;  "), "a cat É &amp;");
    }
    #[test]
    fn pinned_tokenizer_basic_and_boundary() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference/sd15-cpu/tokenizer");
        if !root.join("vocab.json").exists() {
            return;
        }
        let t = Tokenizer::load(&root.join("vocab.json"), &root.join("merges.txt")).unwrap();
        let empty = t.encode("").unwrap();
        assert_eq!(empty[0], 49406);
        assert!(empty[1..].iter().all(|&x| x == 49407));
        assert_eq!(
            &t.encode("a photo of a cat").unwrap()[..7],
            &[49406, 320, 1125, 539, 320, 2368, 49407]
        );
        assert!(t.encode(&vec!["a"; 75].join(" ")).is_ok());
        assert!(t.encode(&vec!["a"; 76].join(" ")).is_err());
        assert!(t.encode("(cat:1.2)").is_err());
    }
}
