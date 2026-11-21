//! development-host's pinned sd.cpp CUDA_RNG default: Philox4x32-10 emulated on the CPU.
//! Adapted from stable-diffusion.cpp 3f8527a46c54ecf4cb4ed6003da8e8982283c73c,
//! src/core/rng_philox.hpp (MIT; accompanying NOTICE). Floating transcendentals
//! are checked numerically against frozen bytes from that original C++ header.
#[derive(Clone, Debug)]
pub struct Philox {
    seed: u64,
    offset: u32,
}
impl Philox {
    pub fn new(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        let offset = self.offset;
        self.offset = self.offset.wrapping_add(1);
        (0..n)
            .map(|i| {
                let mut c = [offset, 0, i as u32, 0];
                let mut key = [self.seed as u32, (self.seed >> 32) as u32];
                for round in 0..10 {
                    let a = c[0] as u64 * 0xd2511f53;
                    let b = c[2] as u64 * 0xcd9e8d57;
                    c = [
                        (b >> 32) as u32 ^ c[1] ^ key[0],
                        b as u32,
                        (a >> 32) as u32 ^ c[3] ^ key[1],
                        a as u32,
                    ];
                    if round != 9 {
                        key[0] = key[0].wrapping_add(0x9e3779b9);
                        key[1] = key[1].wrapping_add(0xbb67ae85);
                    }
                }
                let inv = 2.328_306_4e-10f32;
                let angular = inv * 6.283_185_5f32;
                let u = c[0] as f32 * inv + inv / 2.0;
                let v = c[1] as f32 * angular + angular / 2.0;
                (-2.0 * u.ln()).sqrt() * v.sin()
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_cpp_noise_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../reference/sd15-cpu/rng/noise-128.f32");
        if !path.exists() {
            return;
        }
        let bytes = std::fs::read(path).unwrap();
        let expected: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let actual = Philox::new(42).randn(expected.len());
        let max = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(max < 2e-6, "Philox original-header max error {max}");
    }
}
