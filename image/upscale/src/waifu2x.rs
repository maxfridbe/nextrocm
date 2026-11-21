//! Original waifu2x request mapping and CPU graph/weight preflight.
//! Kernels and graph execution are shared with the existing native upscaler.
use crate::{
    ncnn::{Bin, Graph},
    Plan,
};
use nextrocm_image::{Error, Result, UpscaleRequest, UpscaleVariant};

pub const VARIANTS: &[UpscaleVariant] = &[
    UpscaleVariant {
        id: "models-cunet",
        about: "illustration (cunet); denoise -1 to 3",
        scales: &[1, 2, 4, 8, 16, 32],
        noise: &[-1, 0, 1, 2, 3],
    },
    UpscaleVariant {
        id: "models-upconv_7_anime_style_art_rgb",
        about: "anime-style art (upconv_7)",
        scales: &[2, 4, 8, 16, 32],
        noise: &[-1, 0, 1, 2, 3],
    },
    UpscaleVariant {
        id: "models-upconv_7_photo",
        about: "photographs (upconv_7)",
        scales: &[2, 4, 8, 16, 32],
        noise: &[-1, 0, 1, 2, 3],
    },
];
pub(crate) fn plan(req: &UpscaleRequest) -> Result<Plan> {
    let variant = req
        .variant
        .as_deref()
        .filter(|s| !s.is_empty())
        .unwrap_or("models-cunet");
    let v = VARIANTS
        .iter()
        .find(|v| v.id == variant)
        .ok_or_else(|| Error(format!("unknown waifu2x variant {variant}")))?;
    let scale = req.scale.unwrap_or(2);
    let noise = req.noise.unwrap_or(0);
    if !v.scales.contains(&scale) || !v.noise.contains(&noise) || (scale == 1 && noise < 0) {
        return Err(Error(
            "unsupported waifu2x scale/denoise combination".into(),
        ));
    }
    let cunet = variant == "models-cunet";
    let (file, prepad) = if noise == -1 {
        ("scale2.0x_model".to_owned(), if cunet { 18 } else { 7 })
    } else if scale == 1 {
        (format!("noise{noise}_model"), 28)
    } else {
        (
            format!("noise{noise}_scale2.0x_model"),
            if cunet { 18 } else { 7 },
        )
    };
    Ok(Plan {
        stem: format!("{variant}/{file}"),
        scale: if scale == 1 { 1 } else { 2 },
        prepad,
        border: 1,
        passes: if scale == 1 {
            1
        } else {
            scale.trailing_zeros() as usize
        },
        variant: variant.into(),
        noise: Some(noise),
        tile: req.tile.unwrap_or(400).max(32) as usize,
    })
}

pub(crate) fn validate_request(req: &UpscaleRequest, p: &Plan) -> Result<()> {
    let scale = p
        .scale
        .checked_pow(p.passes as u32)
        .ok_or_else(|| Error("scale overflow".into()))?;
    let w = (req.picture.width as usize)
        .checked_mul(scale)
        .filter(|&n| n <= u32::MAX as usize);
    let h = (req.picture.height as usize)
        .checked_mul(scale)
        .filter(|&n| n <= u32::MAX as usize);
    let bytes = w
        .zip(h)
        .and_then(|(w, h)| w.checked_mul(h))
        .and_then(|n| n.checked_mul(req.picture.channels as usize));
    // Both whole pictures plus a 1 GiB workspace allowance must fit below 8 GB.
    // Actual graph workspace is checked separately before every tile allocation.
    if bytes
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(1 << 30))
        .is_none_or(|n| n >= 8_000_000_000)
        || p.tile > 4096
    {
        return Err(Error(
            "waifu2x picture/tile exceeds bounded device budget".into(),
        ));
    }
    Ok(())
}

/// A graph activation; spatial storage is HWC half, SE vectors are F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Extent {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub vector: bool,
}
impl Extent {
    fn bytes(self) -> std::result::Result<usize, String> {
        self.h
            .checked_mul(self.w)
            .and_then(|n| n.checked_mul(self.c))
            .and_then(|n| n.checked_mul(if self.vector { 4 } else { 2 }))
            .ok_or_else(|| "activation size overflow".into())
    }
}
/// Conservative allocation bound (all distinct outputs and convolution temporaries,
/// without reuse), plus exact resident weights. This is not measured VRAM.
#[derive(Debug)]
pub struct Inspection {
    pub output: Extent,
    pub workspace_bound: usize,
    pub weight_bytes: usize,
    pub layers: usize,
}
fn checked_sum(a: usize, b: usize) -> std::result::Result<usize, String> {
    a.checked_add(b).ok_or_else(|| "size overflow".into())
}

/// Validate supported topology, tensor dimensions and optionally every weight before GPU use.
/// A `None` binary performs shape preflight only; `Some` requires exact consumption.
pub fn inspect(
    g: &Graph,
    bin: Option<&[u8]>,
    h: usize,
    w: usize,
) -> std::result::Result<Inspection, String> {
    if ![8, 59].contains(&g.layers.len())
        || g.blobs.len() > 128
        || h == 0
        || w == 0
        || h > 8192
        || w > 8192
    {
        return Err("unsupported waifu2x graph/input dimensions".into());
    }
    if bin.is_some_and(|b| b.len() > 64 * 1024 * 1024) {
        return Err("oversized waifu2x weights".into());
    }
    let mut reader = bin.map(Bin::new);
    let mut shapes: Vec<Option<Extent>> = vec![None; g.blobs.len()];
    let mut workspace = 0usize;
    let mut weight_bytes = 0usize;
    let mut inputs = 0;
    for l in &g.layers {
        let p = &l.params;
        let err = || format!("{}: unsupported waifu2x layer/shape/parameters", l.name);
        let ins = l
            .inputs
            .iter()
            .map(|&i| shapes.get(i).copied().flatten().ok_or_else(err))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let required = match l.kind.as_str() {
            "Input" => 0,
            "Crop" | "Scale" | "Eltwise" => 2,
            _ => 1,
        };
        if ins.len() != required
            || l.outputs.is_empty()
            || (l.kind != "Split" && l.outputs.len() != 1)
        {
            return Err(err());
        }
        let allowed: &[i32] = match l.kind.as_str() {
            "Input" => &[0, 1, 2],
            "Convolution" | "Deconvolution" => &[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19,
            ],
            "Split" => &[],
            "Crop" => &[0, 1],
            "Eltwise" => &[0, 1],
            "Pooling" => &[0, 4],
            "InnerProduct" => &[0, 1, 2, 9, 10],
            "Scale" => &[0, 1],
            _ => return Err(err()),
        };
        if p.0.keys().any(|id| !allowed.contains(id)) {
            return Err(err());
        }
        let sh = match l.kind.as_str() {
            "Input" => {
                inputs += 1;
                if inputs != 1 || !g.layers.first().is_some_and(|x| x.name == l.name) {
                    return Err(err());
                }
                Extent {
                    h,
                    w,
                    c: 3,
                    vector: false,
                }
            }
            "Convolution" | "Deconvolution" => {
                let x = ins[0];
                let co = p.int(0, 0);
                let k = p.int(1, 0);
                let stride = p.int(3, 1);
                let pl = p.int(4, 0);
                let pt = p.int(14, pl);
                let pr = p.int(15, pl);
                let pb = p.int(16, pt);
                let activation = p.int(9, 0);
                if x.vector
                    || !(1..=256).contains(&co)
                    || !(1..=4).contains(&k)
                    || !(1..=2).contains(&stride)
                    || p.int(11, k) != k
                    || p.int(2, 1) != 1
                    || p.int(12, 1) != 1
                    || p.int(13, stride) != stride
                    || p.int(7, 1) != 1
                    || p.int(8, 0) != 0
                    || p.int(5, 0) != 1
                    || [pl, pt, pr, pb].iter().any(|&v| v < 0 || v >= k)
                    || p.int(18, 0) != 0
                    || p.int(19, 0) != 0
                    || ![0, 2].contains(&activation)
                    || (activation == 2 && p.array(10) != [0.1])
                {
                    return Err(err());
                }
                let (co, k, stride) = (co as usize, k as usize, stride as usize);
                let n = co * x.c * k * k;
                if p.int(6, 0) as usize != n {
                    return Err(err());
                }
                weight_bytes = checked_sum(weight_bytes, n * 2 + co * 4)?;
                if let Some(r) = reader.as_mut() {
                    let a = r.weight(n)?.to_f32();
                    let b = r.floats(co)?;
                    if a.iter().chain(&b).any(|v| !v.is_finite()) {
                        return Err(err());
                    }
                }
                let (ph, pw, oh, ow) = if l.kind == "Deconvolution" {
                    let dh = (x.h - 1) * stride + 1;
                    let dw = (x.w - 1) * stride + 1;
                    workspace = checked_sum(workspace, Extent { h: dh, w: dw, ..x }.bytes()?)?;
                    let ph = dh + 2 * (k - 1) - pt as usize - pb as usize;
                    let pw = dw + 2 * (k - 1) - pl as usize - pr as usize;
                    if ph < k || pw < k {
                        return Err(err());
                    }
                    (ph, pw, ph - k + 1, pw - k + 1)
                } else {
                    let ph = x.h + pt as usize + pb as usize;
                    let pw = x.w + pl as usize + pr as usize;
                    if ph < k || pw < k {
                        return Err(err());
                    }
                    (ph, pw, (ph - k) / stride + 1, (pw - k) / stride + 1)
                };
                workspace = checked_sum(workspace, Extent { h: ph, w: pw, ..x }.bytes()?)?;
                Extent {
                    h: oh,
                    w: ow,
                    c: co,
                    vector: false,
                }
            }
            "Split" => ins[0],
            "Crop" => {
                let x = ins[0];
                let r = ins[1];
                let ox = p.int(0, 0);
                let oy = p.int(1, 0);
                if x.vector
                    || r.vector
                    || ox < 0
                    || oy < 0
                    || ox as usize + r.w > x.w
                    || oy as usize + r.h > x.h
                {
                    return Err(err());
                }
                Extent {
                    h: r.h,
                    w: r.w,
                    ..x
                }
            }
            "Eltwise" => {
                if ins[0] != ins[1] || ins[0].vector || p.int(0, 1) != 1 || !p.array(1).is_empty() {
                    return Err(err());
                }
                ins[0]
            }
            "Pooling" => {
                if ins[0].vector || p.int(0, 0) != 1 || p.int(4, 0) != 1 {
                    return Err(err());
                }
                Extent {
                    h: 1,
                    w: 1,
                    c: ins[0].c,
                    vector: true,
                }
            }
            "InnerProduct" => {
                let x = ins[0];
                let co = p.int(0, 0);
                if !x.vector
                    || !(1..=256).contains(&co)
                    || p.int(1, 0) != 1
                    || p.int(2, 0) != co * x.c as i32
                    || ![1, 4].contains(&p.int(9, 0))
                    || !p.array(10).is_empty()
                {
                    return Err(err());
                }
                let co = co as usize;
                let n = co * x.c;
                weight_bytes = checked_sum(weight_bytes, (n + co) * 4)?;
                if let Some(r) = reader.as_mut() {
                    let a = r.weight(n)?.to_f32();
                    let b = r.floats(co)?;
                    if a.iter().chain(&b).any(|v| !v.is_finite()) {
                        return Err(err());
                    }
                }
                Extent {
                    h: 1,
                    w: 1,
                    c: co,
                    vector: true,
                }
            }
            "Scale" => {
                if ins[0].vector
                    || !ins[1].vector
                    || ins[0].c != ins[1].c
                    || p.int(0, 0) != -233
                    || p.int(1, 0) != 0
                {
                    return Err(err());
                }
                ins[0]
            }
            _ => return Err(err()),
        };
        if l.kind != "Split" {
            workspace = checked_sum(workspace, sh.bytes()?)?;
        }
        for &out in &l.outputs {
            let dst = shapes.get_mut(out).ok_or_else(err)?;
            if dst.is_some() {
                return Err(err());
            }
            *dst = Some(sh);
        }
    }
    if reader.as_ref().is_some_and(|r| r.rest() != 0) {
        return Err("trailing waifu2x weights".into());
    }
    let output = g
        .layers
        .last()
        .and_then(|l| l.outputs.first())
        .and_then(|&i| shapes.get(i))
        .copied()
        .flatten()
        .ok_or("missing output")?;
    if inputs != 1 || output.c != 3 || output.vector {
        return Err("invalid waifu2x input/output".into());
    }
    if workspace >= 7_000_000_000 {
        return Err("waifu2x tile workspace exceeds device budget".into());
    }
    Ok(Inspection {
        output,
        workspace_bound: workspace,
        weight_bytes,
        layers: g.layers.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ncnn::{parse_param, Val};
    use nextrocm_image::Picture;
    fn request() -> UpscaleRequest {
        UpscaleRequest {
            picture: Picture {
                width: 1,
                height: 1,
                channels: 3,
                data: vec![0; 3],
            },
            variant: None,
            scale: None,
            noise: None,
            tile: None,
        }
    }
    #[test]
    fn reference_request_matrix() {
        let mut r = request();
        let p = plan(&r).unwrap();
        assert_eq!(p.stem, "models-cunet/noise0_scale2.0x_model");
        assert_eq!((p.prepad, p.tile, p.passes), (18, 400, 1));
        let mut valid = 0;
        for v in VARIANTS {
            for &scale in v.scales {
                for &noise in v.noise {
                    r.variant = Some(v.id.into());
                    r.scale = Some(scale);
                    r.noise = Some(noise);
                    if scale == 1 && noise == -1 {
                        assert!(plan(&r).is_err());
                        continue;
                    }
                    let p = plan(&r).unwrap();
                    assert_eq!(p.scale.pow(p.passes as u32), scale as usize);
                    assert_eq!(p.border, 1);
                    valid += 1;
                }
            }
        }
        assert_eq!(valid, 79);
        r.scale = Some(3);
        assert!(plan(&r).is_err());
        r.scale = Some(2);
        r.noise = Some(4);
        assert!(plan(&r).is_err());
        r.noise = Some(0);
        r.variant = Some("../bad".into());
        assert!(plan(&r).is_err());
    }
    #[test]
    fn actual_topologies_and_deconvolution_extents() {
        for (text, h, w, oh, ow) in [
            (
                include_str!("../tests/waifu2x/cunet1.param"),
                88,
                92,
                32,
                36,
            ),
            (
                include_str!("../tests/waifu2x/cunet2.param"),
                68,
                70,
                64,
                68,
            ),
            (
                include_str!("../tests/waifu2x/upconv2.param"),
                46,
                48,
                64,
                68,
            ),
        ] {
            let g = parse_param(text).unwrap();
            let a = inspect(&g, None, h, w).unwrap();
            assert_eq!(
                a.output,
                Extent {
                    h: oh,
                    w: ow,
                    c: 3,
                    vector: false
                }
            );
            assert!(a.weight_bytes < 3_000_000);
            assert!(inspect(&g, Some(&[0; 4]), h, w).is_err());
            let mut bad = g.clone();
            bad.layers[1].params.0.insert(3, Val::Int(0));
            assert!(inspect(&bad, None, h, w).is_err());
            bad = g.clone();
            bad.layers[1].inputs.clear();
            assert!(inspect(&bad, None, h, w).is_err());
            bad = g.clone();
            bad.layers[1].outputs[0] = g.layers[0].outputs[0];
            assert!(inspect(&bad, None, h, w).is_err());
        }
    }
    #[test]
    fn oversized_requests_rejected_before_allocation() {
        let mut r = request();
        r.scale = Some(32);
        r.picture.width = u32::MAX;
        let p = plan(&r).unwrap();
        assert!(validate_request(&r, &p).is_err());
        r.picture.width = 1;
        r.tile = Some(u32::MAX);
        let p = plan(&r).unwrap();
        assert!(validate_request(&r, &p).is_err());
    }
}
