//! SD1 epsilon parameterization; the shared flow schedule is deliberately not used.
use nextrocm_image::{Error, Result};

/// Training sigmas from the SD1 scaled-linear beta schedule, ascending.
pub fn train_sigmas() -> Vec<f32> {
    let mut alpha = 1.0f32;
    (0..1000)
        .map(|i| {
            let root =
                0.00085f32.sqrt() + (0.012f32.sqrt() - 0.00085f32.sqrt()) * (i as f32 / 999.0);
            alpha *= 1.0 - root * root;
            ((1.0 - alpha) / alpha).sqrt() as f32
        })
        .collect()
}

/// Karras rho=7, full training range, followed by terminal zero.
pub fn karras(steps: u32) -> Result<Vec<f32>> {
    if !(1..=200).contains(&steps) {
        return Err(Error("SD1.5 steps must be 1..=200".into()));
    }
    let train = train_sigmas();
    let hi = train[999].powf(1.0 / 7.0);
    let lo = train[0].powf(1.0 / 7.0);
    let mut out: Vec<_> = (0..steps)
        .map(|i| {
            let r = if steps == 1 {
                0.0
            } else {
                i as f32 / (steps - 1) as f32
            };
            (hi + r * (lo - hi)).powf(7.0)
        })
        .collect();
    out.push(0.0);
    Ok(out)
}

/// Fractional diffusion timestep by linear interpolation in log-sigma.
pub fn timestep(sigma: f32) -> Result<f32> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(Error("SD1.5 sigma must be finite and positive".into()));
    }
    let s = train_sigmas();
    let i = s.partition_point(|v| *v < sigma);
    if i == 0 {
        return Ok(0.0);
    }
    if i >= s.len() {
        return Ok(999.0);
    }
    Ok((i - 1) as f32 + (sigma.ln() - s[i - 1].ln()) / (s[i].ln() - s[i - 1].ln()))
}

/// CompVis/sd.cpp sinusoidal embedding: cosine half, then sine half.
pub fn embedding(t: f32) -> Result<Vec<f32>> {
    if !t.is_finite() {
        return Err(Error("nonfinite timestep".into()));
    }
    let mut out = vec![0.0; 320];
    for i in 0..160 {
        let phase = t * (-10000.0f32.ln() * i as f32 / 160.0).exp();
        out[i] = phase.cos();
        out[i + 160] = phase.sin();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn training_grid_and_inverse() {
        let s = train_sigmas();
        assert!((s[0] - 0.0291675).abs() < 1e-6);
        assert!((s[999] - 14.61465).abs() < 1e-3);
        for (i, &x) in s.iter().enumerate() {
            assert!((timestep(x).unwrap() - i as f32).abs() < 1e-3);
        }
        let k = karras(25).unwrap();
        assert_eq!(k.len(), 26);
        assert_eq!(k[25], 0.0);
        assert!(k.windows(2).all(|p| p[0] > p[1]));
        assert!(karras(0).is_err());
        assert!(timestep(f32::NAN).is_err());
    }
    #[test]
    fn zero_embedding() {
        let x = embedding(0.0).unwrap();
        assert!(x[..160].iter().all(|x| *x == 1.0));
        assert!(x[160..].iter().all(|x| *x == 0.0));
    }
}
