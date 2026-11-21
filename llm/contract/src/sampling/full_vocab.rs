//! Explicit temperature -> full-vocabulary nucleus -> draw policy.
//! No implicit top-k. Greedy ties select the lowest token ID.
//! Uses the caller's existing RNG, not Python or llama.cpp seed-compatible draws.
use super::Rng;

fn validate(logits: &[f32], temperature: f32, top_p: f32) -> Result<(), &'static str> {
    if logits.is_empty() || logits.len() > u32::MAX as usize {
        return Err("invalid vocabulary length");
    }
    if logits.iter().any(|x| !x.is_finite()) {
        return Err("full-vocabulary sampling requires finite logits");
    }
    if !temperature.is_finite()
        || temperature < 0.0
        || !top_p.is_finite()
        || top_p <= 0.0
        || top_p > 1.0
    {
        return Err("temperature must be finite/nonnegative and top_p in (0,1]");
    }
    Ok(())
}

/// None means greedy. At top_p=1 every vocabulary entry is retained, including
/// underflowed zero probabilities. At smaller top_p include the crossing token.
/// Probability accumulation initially follows vocabulary order (original CPU
/// softmax on an unsorted row); nucleus selection sorts by logit, then token ID.
pub fn dist(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
) -> Result<Option<Vec<(u32, f32)>>, &'static str> {
    validate(logits, temperature, top_p)?;
    if temperature == 0.0 {
        return Ok(None);
    }
    let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
    if scaled.iter().any(|x| !x.is_finite()) {
        return Err("temperature scaling overflow");
    }
    let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<(u32, f32)> = scaled
        .iter()
        .enumerate()
        .map(|(i, x)| (i as u32, (x - max).exp()))
        .collect();
    let sum: f32 = p.iter().map(|x| x.1).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("invalid probability mass");
    }
    for x in &mut p {
        x.1 /= sum;
    }
    if top_p < 1.0 {
        // Numeric equality (including signed zero) breaks ties by original ID.
        p.sort_by(|a, b| {
            scaled[b.0 as usize]
                .partial_cmp(&scaled[a.0 as usize])
                .expect("validated finite logits")
                .then(a.0.cmp(&b.0))
        });
        let mut mass = 0.0;
        let mut keep = p.len();
        for (i, x) in p.iter().enumerate() {
            mass += x.1;
            if mass >= top_p {
                keep = i + 1;
                break;
            }
        }
        p.truncate(keep);
        // The original distribution stage computes softmax again after nucleus
        // truncation. Preserve that operation instead of renormalizing rounded p.
    }
    // Original CPU draw normalizes expf weights using a double accumulator.
    // Recompute also at top_p=1: the earlier F32 softmax is only for selection.
    let max = p
        .iter()
        .map(|x| scaled[x.0 as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    for x in &mut p {
        x.1 = (scaled[x.0 as usize] - max).exp();
    }
    let mass: f64 = p.iter().map(|x| f64::from(x.1)).sum();
    for x in &mut p {
        x.1 = (f64::from(x.1) / mass) as f32;
    }
    Ok(Some(p))
}

/// Validation completes before RNG consumption. Greedy consumes no randomness.
pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_p: f32,
    rng: &mut Rng,
) -> Result<u32, &'static str> {
    let Some(p) = dist(logits, temperature, top_p)? else {
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        return Ok(best as u32);
    };
    let r = f64::from(rng.next_f32());
    let mut cumulative = 0.0_f64;
    let mut last_positive = 0;
    for (id, mass) in p {
        if mass > 0.0 {
            last_positive = id;
            cumulative += f64::from(mass);
            if r < cumulative {
                return Ok(id);
            }
        }
    }
    Ok(last_positive)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_mass_and_nucleus_have_no_hidden_256_cap() {
        let logits = vec![0.0; 1024];
        let p = dist(&logits, 1.0, 1.0).unwrap().unwrap();
        assert_eq!(p.len(), 1024);
        assert!(
            p.iter()
                .enumerate()
                .all(|(i, x)| *x == (i as u32, 1.0 / 1024.0))
        );
        let p = dist(&logits, 1.0, 0.75).unwrap().unwrap();
        assert_eq!(p.len(), 768);
        assert_eq!(p.last().unwrap().0, 767);
        assert_eq!(super::super::dist(&logits, 1.0, 1.0).unwrap().len(), 256);
    }
    #[test]
    fn greedy_first_tie_and_legacy_last_tie_remain_distinct() {
        let mut rng = Rng(42);
        assert_eq!(sample(&[1.0, 1.0], 0.0, 1.0, &mut rng).unwrap(), 0);
        assert_eq!(sample(&[-0.0, 0.0], 0.0, 1.0, &mut rng).unwrap(), 0);
        assert_eq!(rng.0, 42);
        assert_eq!(super::super::sample(&[1.0, 1.0], 0.0, 1.0, &mut rng), 1);
    }
    #[test]
    fn known_softmax_crossing_and_temperature() {
        let p = dist(&[0.0, 2.0_f32.ln(), 4.0_f32.ln()], 1.0, 0.6)
            .unwrap()
            .unwrap();
        assert_eq!(p.iter().map(|x| x.0).collect::<Vec<_>>(), [2, 1]);
        assert!((p[0].1 - 2.0 / 3.0).abs() < 1e-6);
        let p = dist(&[0.0, 4.0_f32.ln()], 2.0, 1.0).unwrap().unwrap();
        assert!((p[1].1 - 2.0 / 3.0).abs() < 1e-6);
    }
    #[test]
    fn invalid_input_never_consumes_rng() {
        for (l, t, p) in [
            (vec![], 1.0, 1.0),
            (vec![f32::NAN], 1.0, 1.0),
            (vec![f32::INFINITY], 0.0, 1.0),
            (vec![0.0], -1.0, 1.0),
            (vec![0.0], 1.0, 0.0),
            (vec![0.0], 1.0, 1.1),
            (vec![0.0], f32::NAN, 1.0),
            (vec![0.0], 1.0, f32::NAN),
            (vec![f32::MAX], f32::MIN_POSITIVE, 1.0),
        ] {
            let mut r = Rng(42);
            assert!(sample(&l, t, p, &mut r).is_err());
            assert_eq!(r.0, 42);
        }
    }
    #[test]
    fn zero_draw_never_selects_underflowed_probability() {
        assert_eq!(sample(&[-1000.0, 0.0], 1.0, 1.0, &mut Rng(0)).unwrap(), 1);
    }
}
