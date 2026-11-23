//! Real-ESRGAN x4plus and waifu2x: reference ncnn graphs with native HIP kernels.
pub mod ffi;
pub mod ncnn;
pub mod net;
pub mod waifu2x;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_image::{
    Defaults, Error, ImageEngine, ImageKind, ImageRequest, LoadOptions, ModelFiles, Picture,
    Result, Sampler, Schedule, Step, UpscaleInfo, UpscaleRequest, UpscaleVariant,
};

use crate::net::{Net, Pool, Shape};

pub const ARCH_ESRGAN: &str = "realesrgan";
pub const ARCH_WAIFU2X: &str = "waifu2x";

pub fn kind() -> ImageKind {
    ImageKind {
        archs: &[ARCH_ESRGAN, ARCH_WAIFU2X],
        name: "Real-ESRGAN x4plus and waifu2x (Rust + HIP)",
        roles: &[],
        load,
        options: &[],
    }
}

fn load(
    files: &ModelFiles,
    gpus: &[Arc<Gpu>],
    _o: &LoadOptions,
    log: &mut dyn FnMut(String),
) -> Result<Box<dyn ImageEngine>> {
    Ok(Box::new(Upscaler::load(files, gpus, log)?))
}

const ESRGAN: &[UpscaleVariant] = &[UpscaleVariant {
    id: "realesrgan-x4plus",
    about: "photographs and general pictures (RRDB, 4x)",
    scales: &[4],
    noise: &[],
}];

/// One pass of a model: its graph's files (the role names the card gives them), the factor, the prepadding, the border
struct Plan {
    /// the role of the .param (the .bin's is the same with .bin)
    stem: String,
    scale: usize,
    prepad: usize,
    /// 0: mirrored borders (Real-ESRGAN), 1: repeated (waifu2x)
    border: i32,
    /// how many times the model runs (waifu2x 4x: twice its 2x model)
    passes: usize,
    variant: String,
    noise: Option<i32>,
    tile: usize,
}

struct State {
    nets: HashMap<String, Arc<Net>>,
    pool: Pool,
}

pub struct Upscaler {
    arch: &'static str,
    gpu: Arc<Gpu>,
    files: ModelFiles,
    state: Mutex<State>,
    load_secs: f64,
}

impl Upscaler {
    pub fn load(
        files: &ModelFiles,
        gpus: &[Arc<Gpu>],
        log: &mut dyn FnMut(String),
    ) -> Result<Upscaler> {
        let t0 = Instant::now();
        let gpu = gpus
            .first()
            .cloned()
            .ok_or_else(|| Error("an upscaler needs a GPU".into()))?;
        let arch = if files.keys().any(|k| k.starts_with("models-")) {
            ARCH_WAIFU2X
        } else if files.keys().any(|k| k.contains("realesr")) {
            ARCH_ESRGAN
        } else {
            return Err(Error("no supported upscaler graph files".into()));
        };
        if gpu.raw().is_null() {
            return Err(Error("upscaler needs a native HIP GPU".into()));
        }
        ffi::api(&gpu)?;
        let up = Upscaler {
            arch,
            gpu: gpu.clone(),
            files: files.clone(),
            state: Mutex::new(State {
                nets: HashMap::new(),
                pool: Pool::new(&gpu),
            }),
            load_secs: 0.0,
        };
        // the default variant's graph now (the others load when first asked: they are small)
        let default = if arch == ARCH_ESRGAN {
            "realesrgan-x4plus"
        } else {
            "models-cunet/noise1_scale2.0x_model"
        };
        if up.files.contains_key(&format!("{default}.param")) {
            up.net(default)?;
        }
        log(format!(
            "{arch}: {} graph files on {}",
            up.files.len() / 2,
            gpu.name
        ));
        Ok(Upscaler {
            load_secs: t0.elapsed().as_secs_f64(),
            ..up
        })
    }

    fn path(&self, role: &str) -> Result<PathBuf> {
        self.files.get(role).cloned().ok_or_else(|| {
            Error(format!(
                "{role}: not among this model's files (the card lists them)"
            ))
        })
    }

    fn net(&self, stem: &str) -> Result<Arc<Net>> {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(n) = st.nets.get(stem) {
            return Ok(n.clone());
        }
        let param = std::fs::read_to_string(self.path(&format!("{stem}.param"))?)
            .map_err(|e| Error(format!("{stem}.param: {e}")))?;
        let bin = std::fs::read(self.path(&format!("{stem}.bin"))?)
            .map_err(|e| Error(format!("{stem}.bin: {e}")))?;
        let g = ncnn::parse_param(&param).map_err(|e| Error(format!("{stem}.param: {e}")))?;
        if self.arch == ARCH_ESRGAN {
            ncnn::validate_x4plus(&g).map_err(Error)?;
        } else {
            waifu2x::inspect(&g, Some(&bin), 456, 456).map_err(Error)?;
        }
        let n = Arc::new(Net::load(&self.gpu, &g, &bin, stem)?);
        st.nets.insert(stem.to_string(), n.clone());
        Ok(n)
    }

    /// What a request runs, as the programs choose it
    fn plan(&self, req: &UpscaleRequest) -> Result<Plan> {
        if self.arch == ARCH_WAIFU2X {
            return waifu2x::plan(req);
        }
        let scale = req.scale.unwrap_or(4);
        let variant = req
            .variant
            .as_deref()
            .filter(|v| !v.is_empty())
            .unwrap_or("realesrgan-x4plus");
        if scale != 4 || variant != "realesrgan-x4plus" {
            return Err(Error(
                "this port supports realesrgan-x4plus at 4x only".into(),
            ));
        }
        Ok(Plan {
            stem: variant.into(),
            scale: 4,
            prepad: 10,
            border: 0,
            passes: 1,
            variant: variant.into(),
            noise: None,
            tile: req.tile.unwrap_or(200).max(32) as usize,
        })
    }

    /// One pass over the picture on the card (u8 [h, w, ch]): the picture `scale` times bigger, on the card
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        net: &Net,
        p: &Plan,
        src: &DevBuf,
        w: usize,
        h: usize,
        ch: usize,
        done: &mut u32,
        of: u32,
        t0: Instant,
        progress: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<DevBuf> {
        let k = ffi::api(&self.gpu)?;
        let g = ffi::g(&self.gpu);
        let s = p.scale;
        let (ow, oh) = (w * s, h * s);
        let dst = DevBuf::new(&self.gpu, ow * oh * ch)?;
        let t = p.tile;
        let (xt, yt) = (w.div_ceil(t), h.div_ceil(t));
        let mut st = self.state.lock().unwrap_or_else(|q| q.into_inner());
        for yi in 0..yt {
            for xi in 0..xt {
                let tw = ((xi + 1) * t).min(w) - xi * t;
                let th = ((yi + 1) * t).min(h) - yi * t;
                // waifu2x's far side rounded up: its cunet halves the tile twice (scale 1) or once (scale 2)
                let (mut pr, mut pb) = (p.prepad, p.prepad);
                if p.border == 1 {
                    let m = if s == 1 { 4 } else { 2 };
                    pr += tw.div_ceil(m) * m - tw;
                    pb += th.div_ceil(m) * m - th;
                }
                let (iw, ih) = (tw + p.prepad + pr, th + p.prepad + pb);
                let x0 = (xi * t) as i64 - p.prepad as i64;
                let y0 = (yi * t) as i64 - p.prepad as i64;
                if let Some(workspace) = net.waifu_workspace(ih, iw)? {
                    let weights = st
                        .nets
                        .values()
                        .try_fold(0usize, |n, net| n.checked_add(net.weight_bytes))
                        .ok_or_else(|| Error("weight byte count overflow".into()))?;
                    let budget = workspace
                        .checked_add(src.len)
                        .and_then(|n| n.checked_add(dst.len))
                        .and_then(|n| n.checked_add(weights))
                        .and_then(|n| n.checked_add(256 * 1024 * 1024))
                        .ok_or_else(|| Error("waifu2x memory estimate overflow".into()))?;
                    if budget >= 8_000_000_000 {
                        return Err(Error(
                            "waifu2x request exceeds bounded device budget".into(),
                        ));
                    }
                    // Clear only this family's previous tile buffers: the all-output
                    // bound applies to this tile, with no retained earlier shapes.
                    st.pool.clear();
                }
                let input = st.pool.get(iw * ih * 3 * 2)?;
                // SAFETY: src holds h x w x ch bytes, input iw x ih x 3 halves.
                ffi::check(
                    unsafe {
                        (k.pre)(
                            g,
                            src.ptr() as *const u8,
                            h as i64,
                            w as i64,
                            ch as i64,
                            x0,
                            y0,
                            iw as i64,
                            ih as i64,
                            p.border,
                            input.ptr(),
                        )
                    },
                    "the tile",
                )?;
                let (out, sh) = net.run(
                    &self.gpu,
                    &mut st.pool,
                    input,
                    Shape {
                        h: ih,
                        w: iw,
                        c: 3,
                        vec: false,
                    },
                )?;
                // where the tile's own pixels are in the output: Real-ESRGAN's past its prepadding, waifu2x's at the corner
                let (cx, cy) = if p.border == 0 {
                    (p.prepad * s, p.prepad * s)
                } else {
                    (0, 0)
                };
                if sh.c != 3 || sh.w < cx + tw * s || sh.h < cy + th * s {
                    return Err(Error(format!("{}: a {}x{}x{} output for a {iw}x{ih} tile (expected {}x{} from ({cx}, {cy}))", net.name, sh.w, sh.h, sh.c,
                                             tw * s, th * s)));
                }
                let alpha = if ch == 4 {
                    let a = st.pool.get(tw * th * 2)?;
                    // SAFETY: as above, a tw x th halves.
                    ffi::check(
                        unsafe {
                            (k.pre_alpha)(
                                g,
                                src.ptr() as *const u8,
                                h as i64,
                                w as i64,
                                (xi * t) as i64,
                                (yi * t) as i64,
                                tw as i64,
                                th as i64,
                                a.ptr(),
                            )
                        },
                        "the tile's alpha",
                    )?;
                    if s == 1 {
                        Some(a)
                    } else {
                        let big = st.pool.get(tw * s * th * s * 2)?;
                        // SAFETY: as above.
                        ffi::check(
                            unsafe {
                                (k.resize)(
                                    g,
                                    a.ptr(),
                                    th as i64,
                                    tw as i64,
                                    1,
                                    (th * s) as i64,
                                    (tw * s) as i64,
                                    3,
                                    big.ptr(),
                                )
                            },
                            "the alpha's bicubic",
                        )?;
                        st.pool.put(a);
                        Some(big)
                    }
                } else {
                    None
                };
                let ap = alpha
                    .as_ref()
                    .map(|a| a.ptr() as *const _)
                    .unwrap_or(std::ptr::null());
                // SAFETY: out holds sh's halves, dst ow x oh x ch bytes.
                ffi::check(
                    unsafe {
                        (k.post)(
                            g,
                            out.ptr(),
                            sh.w as i64,
                            cx as i64,
                            cy as i64,
                            ap,
                            dst.ptr() as *mut u8,
                            ow as i64,
                            ch as i64,
                            (xi * t * s) as i64,
                            (yi * t * s) as i64,
                            (tw * s) as i64,
                            (th * s) as i64,
                        )
                    },
                    "the tile's pixels",
                )?;
                st.pool.put(out);
                if let Some(a) = alpha {
                    st.pool.put(a);
                }
                *done += 1;
                drop(st);
                self.gpu.sync()?;
                progress(Step {
                    picture: 0,
                    at: *done,
                    of,
                    seconds: t0.elapsed().as_secs_f64(),
                })?;
                st = self.state.lock().unwrap_or_else(|q| q.into_inner());
            }
        }
        Ok(dst)
    }
}

impl ImageEngine for Upscaler {
    fn arch(&self) -> &'static str {
        self.arch
    }
    fn defaults(&self) -> Defaults {
        Defaults {
            width: 0,
            height: 0,
            steps: 0,
            cfg: 1.0,
            sampler: Sampler::Euler,
            schedule: Schedule::Shift,
            shift: 0.0,
        }
    }
    fn guidance(&self) -> bool {
        false
    }
    fn takes_loras(&self) -> bool {
        false
    }
    fn load_seconds(&self) -> f64 {
        self.load_secs
    }
    fn generate(
        &self,
        _req: &ImageRequest,
        _progress: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<Vec<Picture>> {
        Err(Error(format!(
            "{} upscales pictures (image.upscale); it does not make them",
            self.arch
        )))
    }
    fn report(&self) -> Vec<String> {
        let st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        vec![format!(
            "{} on {}: {} graph(s) loaded, {:.0} MB of buffers kept",
            self.arch,
            self.gpu.name,
            st.nets.len(),
            st.pool.held() as f64 / 1e6
        )]
    }
    fn upscale_variants(&self) -> Vec<UpscaleVariant> {
        if self.arch == ARCH_WAIFU2X {
            waifu2x::VARIANTS.to_vec()
        } else {
            ESRGAN.to_vec()
        }
    }
    fn upscale(
        &self,
        req: &UpscaleRequest,
        progress: &mut dyn FnMut(Step) -> Result<()>,
    ) -> Result<(Picture, UpscaleInfo)> {
        let t0 = Instant::now();
        let p = self.plan(req)?;
        let net = self.net(&p.stem)?;
        let pic = &req.picture;
        let ch = pic.channels as usize;
        if !(ch == 3 || ch == 4) {
            return Err(Error(format!("a picture of {ch} channels: RGB or RGBA")));
        }
        let (mut w, mut h) = (pic.width as usize, pic.height as usize);
        if w == 0 || h == 0 {
            return Err(Error("an empty picture".into()));
        }
        validate_picture(pic)?;
        // Waifu2x supports repeated 2x passes, so validate the requested total
        // output and tile-memory bound before allocating either picture.
        if self.arch == ARCH_WAIFU2X {
            waifu2x::validate_request(req, &p)?;
        }
        let mut src = DevBuf::new(&self.gpu, pic.data.len())?;
        src.write(0, &pic.data)?;
        // the tiles of every pass, for the progress
        let mut of = 0u32;
        {
            let (mut a, mut b) = (w, h);
            for _ in 0..p.passes {
                of += (a.div_ceil(p.tile) * b.div_ceil(p.tile)) as u32;
                a *= p.scale;
                b *= p.scale;
            }
        }
        let mut done = 0u32;
        for _ in 0..p.passes {
            let out = self.pass(&net, &p, &src, w, h, ch, &mut done, of, t0, progress)?;
            src = out;
            w *= p.scale;
            h *= p.scale;
        }
        let mut data = vec![0u8; w * h * ch];
        src.read(0, &mut data)?;
        // the buffers stay for the next picture of this size; a big one should not keep a card's memory for good
        {
            let mut st = self.state.lock().unwrap_or_else(|q| q.into_inner());
            if st.pool.held() > (2usize << 30) {
                st.pool.clear();
            }
        }
        let total = p.scale.pow(p.passes as u32) as u32;
        Ok((
            Picture {
                width: w as u32,
                height: h as u32,
                channels: ch as u8,
                data,
            },
            UpscaleInfo {
                variant: p.variant,
                scale: total,
                noise: p.noise,
                tile: p.tile as u32,
                tiles: done,
                seconds: t0.elapsed().as_secs_f64(),
            },
        ))
    }
}

fn validate_picture(pic: &Picture) -> Result<()> {
    let bytes = (pic.width as usize)
        .checked_mul(pic.height as usize)
        .and_then(|n| n.checked_mul(pic.channels as usize))
        .ok_or_else(|| Error("picture dimensions overflow".into()))?;
    if bytes != pic.data.len() {
        return Err(Error(
            "picture byte count does not match its dimensions".into(),
        ));
    }
    if pic.width.checked_mul(4).is_none()
        || pic.height.checked_mul(4).is_none()
        || bytes
            .checked_mul(16)
            .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(Error("4x picture dimensions overflow".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn picture_storage_and_output_dimensions_are_checked() {
        let mut pic = Picture {
            width: 1,
            height: 1,
            channels: 3,
            data: vec![0; 3],
        };
        validate_picture(&pic).unwrap();
        pic.data.pop();
        assert!(validate_picture(&pic).is_err());
        pic.width = u32::MAX;
        assert!(validate_picture(&pic).is_err());
    }
    #[test]
    fn only_the_requested_model_is_advertised() {
        assert_eq!(kind().archs, ["realesrgan", "waifu2x"]);
        assert_eq!(ESRGAN.len(), 1);
        assert_eq!(ESRGAN[0].scales, [4]);
    }
}
