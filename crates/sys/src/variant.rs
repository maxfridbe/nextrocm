//! Silo variants: a silo's ops tuned for other hardware, as a delta off the default silo (CONTRIBUTING rule 5;
//! nextsycl's, in ROCm terms).
//!
//! `kernels/<kind>/<arch>/silo/variants/<name>/` holds a manifest (`variant.yml`) and only the ops that differ; it
//! builds to `dist/silo/libnextrocm-<arch>@<name>.so` with its manifest beside it (`libnextrocm-<arch>@<name>.yml`).
//! An op is then taken, per GPU, from the variant that GPU matches, else from the default silo (else the engine's
//! shared kernel). A variant may also carry `settings`: engine options (the names `--opt-NAME` takes) that become the
//! engine's defaults when the variant applies to every GPU it loads on (`nextrocm_core::variant_settings`). This
//! module is the pure part: the manifest, the device a rule reads, the choice, and whether two rules can overlap.

use std::path::PathBuf;

/// What a variant's match rule reads of one GPU (`ns_gpu_name`, `ns_gpu_units`, `ns_gpu_props`)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    /// compute units (CUs) and their clock, MHz
    pub units: u32,
    pub mhz: u32,
    /// its memory in all, MiB (None: a kernel library built before ns_gpu_props)
    pub vram_mb: Option<u64>,
    /// Exact HIP architecture string, including feature suffixes.
    pub gfx: Option<String>,
    pub wave_size: Option<u32>,
    pub integrated: Option<bool>,
}

impl DeviceInfo {
    /// The card's id in the merge gate's baselines (tooling/gate-baselines.json `cards`): its name without the
    /// vendor's marks, its compute units and its memory in GiB - `radeon-rx-550-8u-4g`. Two boxes with the same card share it.
    pub fn card_id(&self) -> String {
        let mut words = Vec::new();
        for w in self
            .name
            .to_lowercase()
            .replace("(r)", " ")
            .replace("(tm)", " ")
            .split(|c: char| !c.is_ascii_alphanumeric())
        {
            if !w.is_empty()
                && !matches!(
                    w,
                    "amd" | "advanced" | "micro" | "devices" | "intel" | "graphics"
                )
            {
                words.push(w.to_string());
            }
        }
        words.push(format!("{}u", self.units));
        if let Some(m) = self.vram_mb {
            words.push(format!("{}g", (m + 512) / 1024));
        }
        words.join("-")
    }

    /// Device-reported properties for portable gate records; absent properties remain null.
    pub fn to_json(&self) -> String {
        fn quoted(s: &str) -> String {
            let mut out = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
                    c => out.push(c),
                }
            }
            out.push('"');
            out
        }
        format!("{{\"name\":{},\"units\":{},\"mhz\":{},\"vram_mb\":{},\"gfx\":{},\"wave_size\":{},\"integrated\":{}}}",
            quoted(&self.name), self.units,self.mhz,
            self.vram_mb.map_or("null".into(),|v|v.to_string()),
            self.gfx.as_deref().map_or("null".into(),quoted),
            self.wave_size.map_or("null".into(),|v|v.to_string()),
            self.integrated.map_or("null".into(),|v|v.to_string()))
    }
}

/// A variant's match rule: every bound given must hold (an unknown device value fails a bound on it)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Match {
    pub vram_mb_min: Option<u64>,
    pub vram_mb_max: Option<u64>,
    pub units_min: Option<u32>,
    pub units_max: Option<u32>,
    pub mhz_min: Option<u32>,
    pub mhz_max: Option<u32>,
    pub gfx: Option<String>,
    pub wave_size: Option<u32>,
    pub integrated: Option<bool>,
    /// a substring of the device's name: the last resort, when the numbers do not tell the cards apart
    pub name: Option<String>,
}

impl Match {
    pub fn matches(&self, d: &DeviceInfo) -> bool {
        let within = |v: Option<u64>, lo: Option<u64>, hi: Option<u64>| match v {
            Some(v) => lo.is_none_or(|l| v >= l) && hi.is_none_or(|h| v <= h),
            None => lo.is_none() && hi.is_none(),
        };
        let w = |v: Option<u32>| v.map(u64::from);
        within(d.vram_mb, self.vram_mb_min, self.vram_mb_max)
            && within(Some(d.units as u64), w(self.units_min), w(self.units_max))
            && within(Some(d.mhz as u64), w(self.mhz_min), w(self.mhz_max))
            && self.gfx.as_ref().is_none_or(|g| d.gfx.as_ref() == Some(g))
            && self.wave_size.is_none_or(|v| d.wave_size == Some(v))
            && self.integrated.is_none_or(|v| d.integrated == Some(v))
            && self.name.as_deref().is_none_or(|n| d.name.contains(n))
    }

    fn is_empty(&self) -> bool {
        *self == Match::default()
    }

    /// Whether one device could match both rules. Numeric ranges and exact hardware capabilities separate two rules (their ranges do not meet on
    /// some dimension, an absent bound being open); a name never does (one name can hold two substrings) -
    /// conservative: a pair this calls overlapping may in fact never meet, and its fix is a tighter numeric bound.
    pub fn overlaps(&self, other: &Match) -> bool {
        let meet = |lo1: Option<u64>, hi1: Option<u64>, lo2: Option<u64>, hi2: Option<u64>| {
            let lo = lo1.unwrap_or(0).max(lo2.unwrap_or(0));
            let hi = hi1.unwrap_or(u64::MAX).min(hi2.unwrap_or(u64::MAX));
            lo <= hi
        };
        let w = |v: Option<u32>| v.map(u64::from);
        meet(
            self.vram_mb_min,
            self.vram_mb_max,
            other.vram_mb_min,
            other.vram_mb_max,
        ) && meet(
            w(self.units_min),
            w(self.units_max),
            w(other.units_min),
            w(other.units_max),
        ) && meet(
            w(self.mhz_min),
            w(self.mhz_max),
            w(other.mhz_min),
            w(other.mhz_max),
        ) && (self.gfx.is_none() || other.gfx.is_none() || self.gfx == other.gfx)
            && (self.wave_size.is_none()
                || other.wave_size.is_none()
                || self.wave_size == other.wave_size)
            && (self.integrated.is_none()
                || other.integrated.is_none()
                || self.integrated == other.integrated)
    }
}

/// A variant's manifest (`variant.yml`)
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Manifest {
    /// one line: what it changes and why
    pub description: String,
    /// who measured it, on what (the before / after numbers are in its pull request)
    pub measured: String,
    pub rule: Match,
    /// engine options (option name, value) that become the engine's defaults where the variant applies: below the
    /// model card's settings, `--opt` and the environment, above the engine's own default (`settings:`)
    pub settings: Vec<(String, String)>,
}

/// The keys a `match:` block takes
pub const MATCH_KEYS: &[&str] = &[
    "vram_mb_min",
    "vram_mb_max",
    "units_min",
    "units_max",
    "mhz_min",
    "mhz_max",
    "gfx",
    "wave_size",
    "integrated",
    "name",
];

/// Reads a manifest: `description:`, `measured:`, a `match:` block of the bounds (`vram_mb_min`, `vram_mb_max`,
/// `units_min`, `units_max` - CUs -, `mhz_min`, `mhz_max`, `gfx`, `wave_size`, `integrated` - actual HIP capabilities -,
/// `name`) and an optional `settings:` block (engine option name: value), a `key: value` a line, `#` comments. A flat
/// subset of YAML, read here without a YAML crate (the foundation has no dependencies); an unknown key, a missing
/// field, an empty rule, a setting given twice or without a value is an error. Whether a setting names one of its
/// engine's options, with a value that option takes, is checked against the engine
/// (`nextrocm_core::options::variant_settings`, and the architecture test).
pub fn parse_manifest(text: &str) -> Result<Manifest, String> {
    let mut m = Manifest::default();
    let mut in_match = false;
    let mut in_settings = false;
    let mut seen_match = false;
    let mut seen = std::collections::BTreeSet::new();
    for (n, raw) in text.lines().enumerate() {
        let line = match raw.find(" #") {
            Some(i) => &raw[..i],
            None if raw.trim_start().starts_with('#') => "",
            None => raw,
        };
        if line.trim().is_empty() {
            continue;
        }
        let indented = line.starts_with([' ', '\t']);
        let (k, v) = line
            .trim()
            .split_once(':')
            .ok_or_else(|| format!("line {}: `key: value` expected", n + 1))?;
        let (k, v) = (k.trim(), v.trim().trim_matches(|c| c == '"' || c == '\''));
        let num = |what: &str| {
            v.parse::<u64>()
                .map_err(|_| format!("line {}: {what}: a whole number, not {v:?}", n + 1))
        };
        if !indented {
            in_match = false;
            in_settings = false;
        }
        if indented && in_settings {
            if k.is_empty()
                || !k
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
            {
                return Err(format!(
                    "line {}: settings: an engine option's name ([a-z0-9-_]), not {k:?}",
                    n + 1
                ));
            }
            if v.is_empty() {
                return Err(format!("line {}: settings: {k} has no value", n + 1));
            }
            if m.settings.iter().any(|(x, _)| x == k) {
                return Err(format!("line {}: settings: {k} given twice", n + 1));
            }
            m.settings.push((k.to_string(), v.to_string()));
            continue;
        }
        if !seen.insert((indented, k.to_string())) {
            return Err(format!("line {}: {k} given twice", n + 1));
        }
        let num32 =
            |what: &str| u32::try_from(num(what)?).map_err(|_| format!("{what}: out of u32 range"));
        match (indented, in_match, k) {
            (false, _, "description") => m.description = v.to_string(),
            (false, _, "measured") => m.measured = v.to_string(),
            (false, _, "match") if v.is_empty() => {
                in_match = true;
                seen_match = true;
            }
            (false, _, "settings") if v.is_empty() => in_settings = true,
            (true, true, "vram_mb_min") => m.rule.vram_mb_min = Some(num(k)?),
            (true, true, "vram_mb_max") => m.rule.vram_mb_max = Some(num(k)?),
            (true, true, "units_min") => m.rule.units_min = Some(num32(k)?),
            (true, true, "units_max") => m.rule.units_max = Some(num32(k)?),
            (true, true, "mhz_min") => m.rule.mhz_min = Some(num32(k)?),
            (true, true, "mhz_max") => m.rule.mhz_max = Some(num32(k)?),
            (true, true, "gfx") if !v.is_empty() => m.rule.gfx = Some(v.to_string()),
            (true, true, "wave_size") => {
                let n = num32(k)?;
                if n == 0 {
                    return Err("wave_size must be positive".into());
                }
                m.rule.wave_size = Some(n);
            }
            (true, true, "integrated") => {
                m.rule.integrated = Some(match v {
                    "true" => true,
                    "false" => false,
                    _ => return Err("integrated must be true or false".into()),
                })
            }
            (true, true, "name") if !v.is_empty() => m.rule.name = Some(v.to_string()),
            _ => return Err(format!("line {}: unknown key {k:?}", n + 1)),
        }
    }
    if m.description.is_empty() || m.measured.is_empty() {
        return Err("description and measured are required".into());
    }
    if !seen_match || m.rule.is_empty() {
        return Err(format!(
            "a match rule is required ({})",
            MATCH_KEYS.join(", ")
        ));
    }
    for (lo, hi) in [
        (m.rule.vram_mb_min, m.rule.vram_mb_max),
        (
            m.rule.units_min.map(u64::from),
            m.rule.units_max.map(u64::from),
        ),
        (m.rule.mhz_min.map(u64::from), m.rule.mhz_max.map(u64::from)),
    ] {
        if lo.zip(hi).is_some_and(|(a, b)| a > b) {
            return Err("inverted match bounds".into());
        }
    }
    Ok(m)
}

/// A built variant of a silo: dist/silo/libnextrocm-<silo>@<name>.so and its manifest
#[derive(Clone, Debug)]
pub struct VariantFile {
    pub name: String,
    pub path: PathBuf,
    pub manifest: Manifest,
}

/// The variants every one of these devices matches, by name
pub fn matching<'a>(variants: &'a [VariantFile], devices: &[DeviceInfo]) -> Vec<&'a VariantFile> {
    if devices.is_empty() {
        return Vec::new();
    }
    let mut v: Vec<&VariantFile> = variants
        .iter()
        .filter(|v| devices.iter().all(|d| v.manifest.rule.matches(d)))
        .collect();
    v.sort_by(|a, b| a.name.cmp(&b.name));
    v
}

/// A warning when more than one variant matches these devices and none is forced: two variants of one silo whose
/// rules meet (the architecture test refuses such a pair; a manifest written or built outside the tree can still
/// make one). `choose` still picks the first by name.
pub fn ambiguity(
    variants: &[VariantFile],
    devices: &[DeviceInfo],
    force: Option<&str>,
) -> Option<String> {
    if force.is_some_and(|f| !f.trim().is_empty()) {
        return None;
    }
    let m = matching(variants, devices);
    (m.len() > 1).then(|| {
        let names: Vec<&str> = m.iter().map(|v| v.name.as_str()).collect();
        let on: Vec<String> = devices.iter().map(|d| format!("GPU {}", d.index)).collect();
        format!("warning: silo variants {} all match {}: taking {} (the first by name); tighten their match rules",
                names.join(", "), on.join(", "), names[0])
    })
}

/// The variant for these devices: `force` (NS_SILO_VARIANT, a model card's `silo.variant`) names one - `none` none -,
/// else the one every device matches (by name order when several do: `ambiguity` says so). Err: a forced variant
/// that is not built.
pub fn choose<'a>(
    variants: &'a [VariantFile],
    devices: &[DeviceInfo],
    force: Option<&str>,
) -> Result<Option<&'a VariantFile>, String> {
    match force.map(str::trim).filter(|f| !f.is_empty()) {
        Some("none") => Ok(None),
        Some(f) => variants
            .iter()
            .find(|v| v.name == f)
            .map(Some)
            .ok_or_else(|| {
                let built: Vec<&str> = variants.iter().map(|v| v.name.as_str()).collect();
                format!(
                    "silo variant {f:?} is not built (built: {})",
                    if built.is_empty() {
                        "none".into()
                    } else {
                        built.join(", ")
                    }
                )
            }),
        None => Ok(matching(variants, devices).first().copied()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rule(s: &str) -> Result<Manifest, String> {
        parse_manifest(&format!(
            "description: test\nmeasured: fixture\nmatch:\n{s}"
        ))
    }
    #[test]
    fn strict_properties() {
        for s in [
            "  cc_min: 1\n",
            "  units_min: 4294967296\n",
            "  units_min: 3\n  units_max: 2\n",
            "  wave_size: 0\n",
            "  integrated: 1\n",
            "  units_min: 1\n  units_min: 2\n",
        ] {
            assert!(rule(s).is_err(), "{s}");
        }
        let m = rule("  gfx: gfx803\n  wave_size: 64\n  integrated: false\n").unwrap();
        assert!(!m.rule.matches(&DeviceInfo::default()));
        let d = DeviceInfo {
            gfx: Some("gfx803".into()),
            wave_size: Some(64),
            integrated: Some(false),
            ..Default::default()
        };
        assert!(m.rule.matches(&d));
        assert!(!m.rule.matches(&DeviceInfo {
            gfx: Some("gfx803:xnack+".into()),
            ..d
        }));
    }
    #[test]
    fn conservative_overlap() {
        assert!(rule("  name: a\n")
            .unwrap()
            .rule
            .overlaps(&rule("  name: b\n").unwrap().rule));
        for (a, b) in [
            ("wave_size: 32", "wave_size: 64"),
            ("gfx: gfx803", "gfx: gfx1100"),
            ("units_max: 4", "units_min: 5"),
            ("integrated: true", "integrated: false"),
        ] {
            assert!(!rule(&format!("  {a}\n"))
                .unwrap()
                .rule
                .overlaps(&rule(&format!("  {b}\n")).unwrap().rule));
        }
    }
    #[test]
    fn deterministic_choice_and_forcing() {
        let v = |name: &str, w: &str| VariantFile {
            name: name.into(),
            path: PathBuf::new(),
            manifest: rule(w).unwrap(),
        };
        let variants = vec![v("z", "  units_max: 20\n"), v("a", "  units_max: 20\n")];
        let devices = [DeviceInfo {
            units: 10,
            ..Default::default()
        }];
        assert_eq!(
            choose(&variants, &devices, None).unwrap().unwrap().name,
            "a"
        );
        assert!(ambiguity(&variants, &devices, None).is_some());
        assert!(choose(&variants, &devices, Some("none")).unwrap().is_none());
        assert!(choose(&variants, &devices, Some("missing")).is_err());
        assert!(choose(&variants, &[], None).unwrap().is_none());
        assert!(choose(
            &variants,
            &[
                devices[0].clone(),
                DeviceInfo {
                    units: 30,
                    ..Default::default()
                }
            ],
            None
        )
        .unwrap()
        .is_none());
    }
    #[test]
    fn settings_and_unknowns() {
        assert!(rule("  units_min: 1\nsettings:\n  mode: half\n").is_ok());
        assert!(rule("  units_min: 1\nsettings:\n  mode: half\n  mode: full\n").is_err());
        assert!(rule("  mystery: 1\n").is_err());
        assert!(DeviceInfo::default().to_json().contains("\"gfx\":null"));
    }
}
