//! Small graph executor: checked geometry, bounded live allocations, explicit layouts.
#[path = "profile.rs"]
mod profile;
use crate::{
    kernels::{Activation, Attention, Conv2d, Kernels, Linear},
    weights::Weights,
};
use nextrocm_engine_core::{DevBuf, Gpu};
use nextrocm_image::{Error, Result};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

pub const ACTIVATION_CAP: usize = 768 << 20;
/// Named graph output observer. Spatial tensors are NCHW; dense tensors are row-major.
pub type Tap<'a> = dyn FnMut(&str, &[usize], &[f32]) -> Result<()> + 'a;

#[derive(Default)]
struct Accounting {
    live: AtomicUsize,
    peak: AtomicUsize,
}
/// A single owned activation. Dropping it releases memory and its budget charge.
pub(crate) struct Tensor {
    pub buf: DevBuf,
    pub shape: Vec<usize>,
    account: Arc<Accounting>,
    bytes: usize,
}
impl Drop for Tensor {
    fn drop(&mut self) {
        self.account.live.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}
impl Tensor {
    pub fn n(&self) -> usize {
        self.bytes / 4
    }
    pub fn host(&self) -> Result<Vec<f32>> {
        self.buf.to_f32()
    }
}
/// One component execution session; weights are owned by its caller.
pub(crate) struct Graph<'a> {
    pub gpu: Arc<Gpu>,
    pub k: Kernels,
    pub w: &'a Weights,
    account: Arc<Accounting>,
    profile: profile::Profile,
}
impl<'a> Graph<'a> {
    pub fn new(gpu: Arc<Gpu>, w: &'a Weights) -> Result<Self> {
        Ok(Self {
            k: Kernels::new(gpu.clone())?,
            gpu,
            w,
            account: Arc::new(Accounting::default()),
            profile: profile::Profile::new(),
        })
    }
    pub fn peak(&self) -> usize {
        self.account.peak.load(Ordering::Relaxed)
    }
    pub fn alloc(&self, shape: &[usize]) -> Result<Tensor> {
        let bytes = shape
            .iter()
            .try_fold(4usize, |n, &d| if d == 0 { None } else { n.checked_mul(d) })
            .ok_or_else(|| Error("invalid activation shape".into()))?;
        let old = self.account.live.fetch_add(bytes, Ordering::Relaxed);
        let total = old.checked_add(bytes);
        if total.is_none_or(|v| v > ACTIVATION_CAP) {
            self.account.live.fetch_sub(bytes, Ordering::Relaxed);
            return Err(Error("SD1.5 live activation budget exceeded".into()));
        }
        let buf = match DevBuf::new(&self.gpu, bytes) {
            Ok(b) => b,
            Err(e) => {
                self.account.live.fetch_sub(bytes, Ordering::Relaxed);
                return Err(e);
            }
        };
        self.account.peak.fetch_max(old + bytes, Ordering::Relaxed);
        Ok(Tensor {
            buf,
            shape: shape.to_vec(),
            account: self.account.clone(),
            bytes,
        })
    }
    pub fn upload(&self, shape: &[usize], v: &[f32]) -> Result<Tensor> {
        let t = self.alloc(shape)?;
        if t.n() != v.len() || v.iter().any(|x| !x.is_finite()) {
            return Err(Error("invalid input shape or nonfinite input".into()));
        }
        // Bounded input tensors; preserve host endianness explicitly at the API boundary.
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        t.buf.write(0, &b)?;
        Ok(t)
    }
    pub fn tap(&self, name: &str, t: &Tensor, tap: &mut Option<&mut Tap<'_>>) -> Result<()> {
        if let Some(f) = tap.as_mut() {
            self.gpu.sync()?;
            f(name, &t.shape, &t.host()?)?;
        }
        Ok(())
    }
    pub fn linear(&self, x: &Tensor, name: &str, out: usize, bias: bool) -> Result<Tensor> {
        if x.shape.len() != 2 {
            return Err(Error("linear requires matrix".into()));
        }
        let (r, c) = (x.shape[0], x.shape[1]);
        let y = self.alloc(&[r, out])?;
        let w = self.w.get(&format!("{name}.weight"), &[out, c])?;
        let b = if bias {
            Some(self.w.get(&format!("{name}.bias"), &[out])?)
        } else {
            None
        };
        let started = self.profile.begin();
        self.k.linear(
            &x.buf,
            w,
            b,
            &y.buf,
            Linear {
                rows: r as i64,
                input: c as i64,
                output: out as i64,
                x_stride: c as i64,
                out_stride: out as i64,
            },
        )?;
        self.gpu.sync()?;
        self.profile
            .record(started, "linear", &[r, c, out, usize::from(bias)]);
        Ok(y)
    }
    pub fn conv(&self, x: &Tensor, name: &str, stride: usize, pad: usize) -> Result<Tensor> {
        let [n, c, h, w] = spatial(&x.shape)?;
        let s = self.w.shape(&format!("{name}.weight"))?;
        if s.len() != 4 || s[1] != c || s[2] != s[3] || stride == 0 {
            return Err(Error(format!("{name}: invalid convolution geometry")));
        }
        let (co, kh, kw) = (s[0], s[2], s[3]);
        let oh = (h + 2 * pad)
            .checked_sub(kh)
            .ok_or_else(|| Error("conv kernel exceeds input".into()))?
            / stride
            + 1;
        let ow = (w + 2 * pad)
            .checked_sub(kw)
            .ok_or_else(|| Error("conv kernel exceeds input".into()))?
            / stride
            + 1;
        let y = self.alloc(&[n, co, oh, ow])?;
        let started = self.profile.begin();
        self.k.conv2d(
            &x.buf,
            self.w.get(&format!("{name}.weight"), &[co, c, kh, kw])?,
            Some(self.w.get(&format!("{name}.bias"), &[co])?),
            &y.buf,
            Conv2d {
                batch: n as i64,
                ci: c as i64,
                height: h as i64,
                width: w as i64,
                co: co as i64,
                kh: kh as i64,
                kw: kw as i64,
                sh: stride as i64,
                sw: stride as i64,
                ph: pad as i64,
                pw: pad as i64,
                oh: oh as i64,
                ow: ow as i64,
            },
        )?;
        self.gpu.sync()?;
        self.profile.record(
            started,
            "conv2d",
            &[n, c, h, w, co, kh, kw, stride, pad, oh, ow],
        );
        Ok(y)
    }
    pub fn gn(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        let [n, c, h, w] = spatial(&x.shape)?;
        let y = self.alloc(&x.shape)?;
        let started = self.profile.begin();
        self.k.group_norm(
            &x.buf,
            Some(self.w.get(&format!("{name}.weight"), &[c])?),
            Some(self.w.get(&format!("{name}.bias"), &[c])?),
            &y.buf,
            n as i64,
            c as i64,
            h as i64,
            w as i64,
            32,
            1e-6,
        )?;
        self.gpu.sync()?;
        self.profile.record(started, "group_norm", &x.shape);
        Ok(y)
    }
    pub fn ln(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        if x.shape.len() != 2 {
            return Err(Error("LN requires matrix".into()));
        }
        let (r, c) = (x.shape[0], x.shape[1]);
        let y = self.alloc(&x.shape)?;
        let started = self.profile.begin();
        self.k.layer_norm(
            &x.buf,
            Some(self.w.get(&format!("{name}.weight"), &[c])?),
            Some(self.w.get(&format!("{name}.bias"), &[c])?),
            &y.buf,
            r as i64,
            c as i64,
            1e-5,
        )?;
        self.gpu.sync()?;
        self.profile.record(started, "layer_norm", &[r, c]);
        Ok(y)
    }
    pub fn act(&self, x: Tensor, kind: Activation) -> Result<Tensor> {
        let started = self.profile.begin();
        self.k.activation(&x.buf, &x.buf, x.n() as i64, kind)?;
        self.gpu.sync()?;
        self.profile.record(
            started,
            match kind {
                Activation::Silu => "silu",
                Activation::Gelu => "gelu",
                Activation::QuickGelu => "quick_gelu",
                Activation::GeluTanh => "gelu_tanh",
            },
            &x.shape,
        );
        Ok(x)
    }
    pub fn add(&self, x: Tensor, y: &Tensor) -> Result<Tensor> {
        if x.shape != y.shape {
            return Err(Error("residual shape mismatch".into()));
        }
        let started = self.profile.begin();
        self.k
            .add(&x.buf, Some(&y.buf), &x.buf, x.n() as i64, 1.0, 1.0)?;
        self.gpu.sync()?;
        self.profile.record(started, "add", &x.shape);
        Ok(x)
    }
    pub fn time(&self, x: Tensor, e: &Tensor) -> Result<Tensor> {
        let [n, c, h, w] = spatial(&x.shape)?;
        if e.shape != [n, c] {
            return Err(Error("timestep projection shape mismatch".into()));
        }
        let started = self.profile.begin();
        self.k.time_emb(
            &x.buf, &e.buf, &x.buf, n as i64, c as i64, h as i64, w as i64,
        )?;
        self.gpu.sync()?;
        self.profile.record(started, "time_emb", &x.shape);
        Ok(x)
    }
    pub fn tokens(&self, x: &Tensor) -> Result<Tensor> {
        let [n, c, h, w] = spatial(&x.shape)?;
        if n != 1 {
            return Err(Error("SD1.5 supports batch one".into()));
        }
        let y = self.alloc(&[h * w, c])?;
        let started = self.profile.begin();
        self.k
            .layout(&x.buf, &y.buf, 1, c as i64, h as i64, w as i64, true)?;
        self.gpu.sync()?;
        self.profile.record(started, "layout_to_tokens", &x.shape);
        Ok(y)
    }
    pub fn image(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        if x.shape.len() != 2 || x.shape[0] != h * w {
            return Err(Error("token/image geometry mismatch".into()));
        }
        let c = x.shape[1];
        let y = self.alloc(&[1, c, h, w])?;
        let started = self.profile.begin();
        self.k
            .layout(&x.buf, &y.buf, 1, c as i64, h as i64, w as i64, false)?;
        self.gpu.sync()?;
        self.profile
            .record(started, "layout_to_image", &[1, c, h, w]);
        Ok(y)
    }
    pub fn up(&self, x: &Tensor) -> Result<Tensor> {
        let [n, c, h, w] = spatial(&x.shape)?;
        let y = self.alloc(&[n, c, h * 2, w * 2])?;
        let started = self.profile.begin();
        self.k
            .up2(&x.buf, &y.buf, n as i64, c as i64, h as i64, w as i64)?;
        self.gpu.sync()?;
        self.profile.record(started, "up2", &x.shape);
        Ok(y)
    }
    pub fn concat(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let [n, c, h, w] = spatial(&x.shape)?;
        let [m, d, j, k] = spatial(&y.shape)?;
        if [n, h, w] != [m, j, k] {
            return Err(Error("UNet skip geometry mismatch".into()));
        }
        let z = self.alloc(&[n, c + d, h, w])?;
        let started = self.profile.begin();
        self.k.concat(
            &x.buf, &y.buf, &z.buf, n as i64, c as i64, d as i64, h as i64, w as i64,
        )?;
        self.gpu.sync()?;
        self.profile.record(started, "concat", &[n, c, d, h, w]);
        Ok(z)
    }
    pub fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        heads: usize,
        causal: bool,
    ) -> Result<Tensor> {
        if q.shape.len() != 2
            || k.shape.len() != 2
            || v.shape != k.shape
            || q.shape[1] != k.shape[1]
            || heads == 0
            || q.shape[1] % heads != 0
        {
            return Err(Error("attention shape mismatch".into()));
        }
        let (n, m, c) = (q.shape[0] as i64, k.shape[0] as i64, q.shape[1] as i64);
        let d = c / heads as i64;
        let y = self.alloc(&q.shape)?;
        let started = self.profile.begin();
        self.k.attention(
            &q.buf,
            &k.buf,
            &v.buf,
            &y.buf,
            Attention {
                batch: 1,
                queries: n,
                keys: m,
                heads: heads as i64,
                dim: d,
                qb: n * c,
                qt: c,
                qh: d,
                kb: m * c,
                kt: c,
                kh: d,
                vb: m * c,
                vt: c,
                vh: d,
                ob: n * c,
                ot: c,
                oh: d,
                scale: 1.0 / (d as f32).sqrt(),
                causal: i32::from(causal),
            },
        )?;
        self.gpu.sync()?;
        self.profile.record(
            started,
            "attention",
            &[
                n as usize,
                m as usize,
                c as usize,
                heads,
                d as usize,
                usize::from(causal),
            ],
        );
        Ok(y)
    }
    pub fn geglu(&self, x: &Tensor) -> Result<Tensor> {
        if x.shape.len() != 2 || x.shape[1] % 2 != 0 {
            return Err(Error("GEGLU shape mismatch".into()));
        }
        let (r, c) = (x.shape[0], x.shape[1] / 2);
        let y = self.alloc(&[r, c])?;
        let started = self.profile.begin();
        self.k.geglu(&x.buf, &y.buf, r as i64, c as i64)?;
        self.gpu.sync()?;
        self.profile.record(started, "geglu", &[r, c]);
        Ok(y)
    }
}
pub(crate) fn spatial(shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| Error("expected NCHW tensor".into()))
}
