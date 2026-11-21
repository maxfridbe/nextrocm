//! Resolve card settings once per engine; process environment is fallback only.
use nextrocm_image::{Error, LoadOptions, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RuntimeSettings {
    pub decoder_f32: bool,
    pub linear8: bool,
    pub attention_probability_first: bool,
    pub vision_attention_tiled: bool,
}

impl RuntimeSettings {
    pub fn from_options(options: &LoadOptions) -> Result<Self> {
        let flag = |name| match options.setting(name).as_deref() {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            Some(_) => Err(Error(format!("{name} must be 0 or 1"))),
        };
        let value = Self {
            decoder_f32: flag("NS_POCR_DECODER_F32")?,
            linear8: flag("NS_POCR_LINEAR8")?,
            attention_probability_first: flag("NS_POCR_ATTN_PROB")?,
            vision_attention_tiled: flag("NS_POCR_VISION_TILED")?,
        };
        if value.decoder_f32 && value.linear8 {
            return Err(Error(
                "F32 decoder and F16 Linear8 are separate precision controls".into(),
            ));
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn options(f32: &str, linear8: &str, prob: &str) -> LoadOptions {
        LoadOptions {
            settings: [
                ("NS_POCR_DECODER_F32".into(), f32.into()),
                ("NS_POCR_LINEAR8".into(), linear8.into()),
                ("NS_POCR_ATTN_PROB".into(), prob.into()),
                ("NS_POCR_VISION_TILED".into(), "0".into()),
            ]
            .into(),
            ..Default::default()
        }
    }
    #[test]
    fn explicit_card_controls_are_independent_per_engine() {
        let f32 = RuntimeSettings::from_options(&options("1", "0", "1")).unwrap();
        let half = RuntimeSettings::from_options(&options("0", "1", "0")).unwrap();
        assert!(f32.decoder_f32 && f32.attention_probability_first && !f32.linear8);
        assert!(!half.decoder_f32 && !half.attention_probability_first && half.linear8);
        assert_eq!(
            RuntimeSettings::from_options(&options("1", "0", "1")).unwrap(),
            f32
        );
        assert_eq!(
            RuntimeSettings::from_options(&options("0", "0", "0")).unwrap(),
            RuntimeSettings::default()
        );
    }
    #[test]
    fn vision_tiling_is_explicit_and_per_engine() {
        let mut on = options("1", "0", "1");
        on.settings.insert("NS_POCR_VISION_TILED".into(), "1".into());
        let mut off = on.clone();
        off.settings.insert("NS_POCR_VISION_TILED".into(), "0".into());
        assert!(RuntimeSettings::from_options(&on).unwrap().vision_attention_tiled);
        assert!(!RuntimeSettings::from_options(&off).unwrap().vision_attention_tiled);
        on.settings.insert("NS_POCR_VISION_TILED".into(), "invalid".into());
        assert!(RuntimeSettings::from_options(&on).is_err());
    }
    #[test]
    fn invalid_or_conflicting_precision_is_rejected() {
        assert!(RuntimeSettings::from_options(&options("1", "1", "0")).is_err());
        assert!(RuntimeSettings::from_options(&options("yes", "0", "0")).is_err());
        assert!(RuntimeSettings::from_options(&options("0", "0", "invalid")).is_err());
    }
}
