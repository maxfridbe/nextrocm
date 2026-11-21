//! Names and aliases ported from audio/whisper/src/text.rs (nextcuda/nextsycl).
use nextrocm_audio::{Error, Result};
/// openai-whisper's language names by code (tokenizer.LANGUAGES), in the vocabulary's order
pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

/// openai-whisper's other names for some (TO_LANGUAGE_CODE's aliases)
const ALIASES: &[(&str, &str)] = &[
    ("burmese", "my"),
    ("valencian", "ca"),
    ("flemish", "nl"),
    ("haitian", "ht"),
    ("letzeburgesch", "lb"),
    ("pushto", "ps"),
    ("panjabi", "pa"),
    ("moldavian", "ro"),
    ("moldovan", "ro"),
    ("sinhalese", "si"),
    ("castilian", "es"),
    ("mandarin", "zh"),
];

pub fn normalize(asked: Option<&str>, supported: &[String]) -> Result<String> {
    let a = asked.unwrap_or("auto").trim().to_lowercase();
    if a.is_empty() || a == "auto" || a == "unknown" {
        return Ok("auto".into());
    }
    let code = LANGUAGES
        .iter()
        .find(|(c, n)| *c == a || *n == a)
        .map(|(c, _)| *c)
        .or_else(|| ALIASES.iter().find(|(n, _)| *n == a).map(|(_, c)| *c))
        .ok_or_else(|| {
            Error(format!(
                "unknown language {a:?}: expected a language code or name"
            ))
        })?;
    if !supported.iter().any(|s| s == code) {
        return Err(Error(format!(
            "language {code} is not supported by this checkpoint"
        )));
    }
    Ok(code.into())
}
