//! Reference image CLI and native image server.
use nextrocm_image::{ImageRequest, LoraUse, ModelFiles, Sampler, Schedule};
use nextrocm_models::{config::Config, registry as models};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[cfg(not(feature = "starvector-experimental"))]
const USAGE: &str = "nextrocm image gen \"<prompt>\" [--model ID] [--size WxH | --aspect W:H] [--steps N] [--seed N] [--n N] [--sampler S] [--schedule S] [--out FILE|DIR] [--gpu N]\nnextrocm image upscale PICTURE [--model real-esrgan|waifu2x | --dir NCNN_DIR] [--variant V] [--scale N] [--noise N] [--tile N] [--runs N] [--out FILE] [--gpu N]
nextrocm image upscale-check PICTURE REFERENCE.png [upscale's options] (requires >=40 dB per channel and worst error <=16/255)
nextrocm image serve [MODEL] [--port 8086] [--host 127.0.0.1] [--gpu N] [--set NAME=VALUE] [--wfe] [--out DIR] [--cors ORIGIN]
nextrocm image start [serve's options] [--slot N] | ps | logs [-f] | stop [--slot N]
nextrocm image engines";
#[cfg(feature = "starvector-experimental")]
const USAGE: &str = "nextrocm image gen \"<prompt>\" [--model ID] [--size WxH | --aspect W:H] [--steps N] [--seed N] [--n N] [--sampler S] [--schedule S] [--out FILE|DIR] [--gpu N]\nnextrocm image upscale PICTURE [--model real-esrgan|waifu2x | --dir NCNN_DIR] [--variant V] [--scale N] [--noise N] [--tile N] [--runs N] [--out FILE] [--gpu N]
nextrocm image upscale-check PICTURE REFERENCE.png [upscale's options] (requires >=40 dB per channel and worst error <=16/255)
nextrocm image serve [MODEL] [--port 8086] [--host 127.0.0.1] [--gpu N] [--set NAME=VALUE] [--wfe] [--out DIR] [--cors ORIGIN]
nextrocm image start [serve's options] [--slot N] | ps | logs [-f] | stop [--slot N]
nextrocm image engines\nnextrocm image vectorize PICTURE [--model starvector-1b-im2svg | --dir CHECKPOINT_DIR] [--max-length N] [--beams N] [--greedy] [--top-p P] [--seed S] [--out FILE.svg] [--gpu N]\nnextrocm image vectorize PICTURE --check REFERENCE_DIR [--stages pixels,encoder,prefill,logits,greedy] [--dir CHECKPOINT_DIR]";

pub fn engines() -> Vec<nextrocm_image::ImageKind> {
    vec![
        nextrocm_image_upscale::kind(),
        nextrocm_image_sd15::kind(),
        nextrocm_image_qwenimage21::kind(),
        nextrocm_image_rembg::kind(),
        #[cfg(feature = "starvector-experimental")]
        nextrocm_image_starvector::kind(),
    ]
}

/// `--gpu 0` or `--gpu 0,1` (an engine that splits over two cards takes the second too: olmOCR 2's vision tower)
fn gpus(args: &[String]) -> Result<Vec<std::sync::Arc<nextrocm_engine_core::Gpu>>, String> {
    let list = opt(args, "--gpu").unwrap_or("0");
    let mut v = Vec::new();
    for i in list.split(',').filter(|x| !x.trim().is_empty()) {
        let i: usize = i
            .trim()
            .parse()
            .map_err(|_| format!("--gpu {list}: card numbers, comma separated"))?;
        v.push(nextrocm_engine_core::Gpu::open(i).map_err(|e| e.0)?);
    }
    if v.is_empty() {
        v.push(nextrocm_engine_core::Gpu::open(0).map_err(|e| e.0)?);
    }
    Ok(v)
}

pub(crate) fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn model(cfg: &Config, id: Option<&str>) -> Result<(Value, ModelFiles), String> {
    let all = models::all(cfg)?;
    let m = match id {
        Some(id) => all.into_iter().find(|m| m["id"] == id)
            .ok_or_else(|| format!("no model {id} (nextrocm models search --kind image)"))?,
        None => all.into_iter().find(|m| models::kind_of(m) == "image")
            .ok_or("no image model (nextrocm models search --kind image)")?,
    };
    if models::kind_of(&m) != "image" {
        return Err(format!("{} is not an image model", m["id"].as_str().unwrap_or("?")));
    }
    nextrocm_image::kind_for(&engines(), m["arch"].as_str().unwrap_or(""))?;
    crate::drives::not_moving(cfg, m["id"].as_str().unwrap_or(""))?;
    let mut files: ModelFiles = m["files"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(r, p)| Some((r.clone(), PathBuf::from(p.as_str()?))))
        .collect();
    local_upscale_weights(&m, &mut files)?;
    Ok((m, files))
}

/// The catalog resolves local conversion paths into entry.env rather than entry.files.
/// Keep this model-specific bridge here; explicit registry file roles take precedence.
fn local_upscale_weights(m: &Value, files: &mut ModelFiles) -> Result<(), String> {
    const ROLE: &str = "realesrgan-x4plus.bin";
    const ENV: &str = "NS_ESRGAN_X4PLUS_BIN";
    if m["arch"].as_str() != Some("realesrgan") || files.contains_key(ROLE) {
        return Ok(());
    }
    let path = m["env"][ENV]
        .as_str()
        .filter(|p| !p.is_empty())
        .ok_or_else(|| {
            format!(
                "Real-ESRGAN needs {ROLE} in files or a resolved env.{ENV} local conversion path"
            )
        })?;
    let path = PathBuf::from(path);
    if !path.is_file() {
        return Err(format!(
            "env.{ENV}: {} is not a file (the converted Real-ESRGAN bin)",
            path.display()
        ));
    }
    files.insert(ROLE.into(), path);
    Ok(())
}

fn options(
    cfg: &Config,
    m: &Value,
    args: &[String],
) -> Result<nextrocm_image::LoadOptions, String> {
    let merge_loras = loras(cfg, args)?;
    if m["arch"] == "realesrgan" && !merge_loras.is_empty() {
        return Err("Real-ESRGAN does not take LoRAs".into());
    }
    let mut settings: std::collections::BTreeMap<String, String> =
        cfg.with_prefix("NC_").into_iter().collect();
    for (k, v) in m["env"].as_object().into_iter().flatten() {
        if let Some(v) = v.as_str() {
            settings.insert(k.clone(), v.into());
        }
    }
    for (i, a) in args.iter().enumerate() {
        if a == "--set" {
            let (k, v) = args
                .get(i + 1)
                .and_then(|v| v.split_once('='))
                .ok_or("--set NAME=VALUE")?;
            settings.insert(k.into(), v.into());
        }
    }
    let ks = engines();
    let k = nextrocm_image::kind_for(&ks, m["arch"].as_str().unwrap_or(""))?;
    settings.extend(super::opt_env(
        k.options,
        nextrocm_engine_core::At::Load,
        k.name,
    )?);
    Ok(nextrocm_image::LoadOptions {
        merge_loras,
        settings,
    })
}

/// Every `--lora NAME[:SCALE]`: a registered LoRA's id or a file
fn loras(cfg: &Config, args: &[String]) -> Result<Vec<LoraUse>, String> {
    let all = models::all(cfg)?;
    let find = |n: &str| -> Option<PathBuf> {
        all.iter()
            .find(|m| m["id"] == n && models::kind_of(m) == "lora")
            .and_then(|m| m["file"].as_str().map(PathBuf::from))
            .or_else(|| Path::new(n).is_file().then(|| PathBuf::from(n)))
    };
    args.iter()
        .enumerate()
        .filter(|(_, a)| *a == "--lora")
        .map(|(i, _)| {
            let v = args.get(i + 1).ok_or("--lora NAME[:SCALE]")?;
            LoraUse::parse(v, &find)
        })
        .collect()
}

pub(crate) fn load(
    cfg: &Config,
    args: &[String],
) -> Result<(Value, Box<dyn nextrocm_image::ImageEngine>), String> {
    nextrocm_engine_core::use_kind("image");
    let (m, files) = model(cfg, opt(args, "--model"))?;
    let arch = m["arch"].as_str().unwrap_or("");
    let ks = engines();
    let k = nextrocm_image::kind_for(&ks, arch)?;
    let gs = gpus(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let o = options(cfg, &m, args)?;
    let e = (k.load)(&files, &gs, &o, &mut log).map_err(|e| e.0)?;
    Ok((m, e))
}

/// `--size WxH`, or `--aspect W:H` at about the default's area (sides multiples of 16)
fn size(args: &[String], d: (u32, u32)) -> Result<(u32, u32), String> {
    if let Some(s) = opt(args, "--size") {
        let (w, h) = s.split_once('x').ok_or("--size WxH")?;
        return Ok((
            w.parse().map_err(|_| "--size WxH")?,
            h.parse().map_err(|_| "--size WxH")?,
        ));
    }
    if let Some(a) = opt(args, "--aspect") {
        let (w, h) = a.split_once(':').ok_or("--aspect W:H")?;
        let (w, h): (f64, f64) = (
            w.parse().map_err(|_| "--aspect W:H")?,
            h.parse().map_err(|_| "--aspect W:H")?,
        );
        let area = d.0 as f64 * d.1 as f64;
        let hh = (area * h / w).sqrt();
        let r = |x: f64| ((x / 16.0).round() as u32 * 16).max(16);
        return Ok((r(hh * w / h), r(hh)));
    }
    Ok(d)
}

fn gen(cfg: &Config, args: &[String]) -> Result<(), String> {
    gen_or_edit(cfg, args, false)
}

fn gen_or_edit(cfg: &Config, args: &[String], editing: bool) -> Result<(), String> {
    let prompt = args
        .first()
        .filter(|p| !p.starts_with("--"))
        .ok_or(USAGE)?
        .clone();
    let pics: Vec<nextrocm_image::Picture> = args
        .iter()
        .enumerate()
        .filter(|(_, a)| *a == "--image")
        .map(|(i, _)| {
            args.get(i + 1)
                .ok_or("--image PICTURE".to_string())
                .and_then(|p| nextrocm_image::Picture::read_png(Path::new(p)))
        })
        .collect::<Result<_, _>>()?;
    if editing && pics.is_empty() {
        return Err("image edit: --image PICTURE (a PNG)".into());
    }
    // the server path comes with `image serve`; until then every generation runs here
    let (m, e) = load(cfg, args)?;
    let d = e.defaults();
    // an edit's size: its picture's aspect unless one is given
    let given = opt(args, "--size").is_some() || opt(args, "--aspect").is_some();
    let (width, height) = size(args, (d.width, d.height))?;
    let seed: u64 = opt(args, "--seed")
        .map(|s| s.parse().map_err(|_| "--seed N"))
        .transpose()?
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|t| t.as_nanos() as u64 % 100_000)
                .unwrap_or(0)
        });
    let req = ImageRequest {
        prompt: prompt.clone(),
        width: (!editing || given).then_some(width),
        height: (!editing || given).then_some(height),
        steps: opt(args, "--steps")
            .map(|s| s.parse().map_err(|_| "--steps N"))
            .transpose()?,
        seed,
        n: opt(args, "--n")
            .map(|s| s.parse().map_err(|_| "--n N"))
            .transpose()?
            .unwrap_or(1),
        sampler: opt(args, "--sampler")
            .map(|s| Sampler::parse(s).ok_or(format!("--sampler: {s}?")))
            .transpose()?,
        schedule: opt(args, "--schedule")
            .map(|s| Schedule::parse(s).ok_or(format!("--schedule: {s}?")))
            .transpose()?,
        rgba: args.iter().any(|a| a == "--rgba"),
        negative: opt(args, "--negative").map(str::to_string),
        cfg: opt(args, "--cfg")
            .map(|s| s.parse().map_err(|_| "--cfg X"))
            .transpose()?,
        edit: editing.then(|| nextrocm_image::Edit {
            image: pics[0].clone(),
            refs: pics[1..].to_vec(),
            mask: None,
            strength: 1.0,
            condition: true,
        }),
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let mut last = 0f64;
    let pics = e
        .generate(&req, &mut |s| {
            if s.at > 0 {
                let per = (s.seconds - last).max(0.0);
                last = s.seconds;
                eprint!(
                    "\rpicture {}  step {}/{}  {:.2} s/step  {:.0} s   ",
                    s.picture + 1,
                    s.at,
                    s.of,
                    per,
                    s.seconds
                );
            } else {
                last = s.seconds;
                eprintln!("prompt encoded in {:.1} s", s.seconds);
            }
            Ok(())
        })
        .map_err(|e| e.0)?;
    eprintln!();
    let id = m["id"].as_str().unwrap_or("image");
    let out = opt(args, "--out").map(PathBuf::from);
    let mut saved = Vec::new();
    for (i, p) in pics.iter().enumerate() {
        let name = format!("{id}-{seed}-{}.png", i + 1);
        let path = match &out {
            Some(o) if o.is_dir() => o.join(name),
            Some(o) if pics.len() == 1 => o.clone(),
            Some(o) => o.with_file_name(format!(
                "{}-{}.png",
                o.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                i + 1
            )),
            None => PathBuf::from(name),
        };
        let lora_list: Vec<String> = loras(cfg, args)?
            .iter()
            .map(|l| format!("{}:{}", l.name, l.scale))
            .collect();
        let meta = [
            ("prompt", prompt.clone()),
            ("model", id.to_string()),
            ("seed", (seed + i as u64).to_string()),
            ("size", format!("{}x{}", p.width, p.height)),
            ("steps", req.steps.unwrap_or(d.steps).to_string()),
            (
                "edit",
                if editing {
                    format!("{} picture(s)", pics.len())
                } else {
                    String::new()
                },
            ),
            ("loras", lora_list.join(",")),
        ];
        p.write_png(&path, &meta)?;
        saved.push(path.display().to_string());
    }
    let (pw, ph) = pics_size(&saved);
    println!(
        "saved {}  ({pw}x{ph}, {:.1} s, settings in the PNG)",
        saved.join(", "),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The first saved picture's size
fn pics_size(saved: &[String]) -> (u32, u32) {
    saved
        .first()
        .and_then(|p| nextrocm_image::Picture::read_png(Path::new(p)).ok())
        .map_or((0, 0), |p| (p.width, p.height))
}

fn ncnn_files(dir: &Path) -> Result<ModelFiles, String> {
    let mut files = ModelFiles::new();
    let mut add = |sub: Option<&str>, d: &Path| -> Result<(), String> {
        for e in std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()))?.flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if name.ends_with(".param") || name.ends_with(".bin") {
                files.insert(sub.map(|s| format!("{s}/{name}")).unwrap_or(name), p);
            }
        }
        Ok(())
    };
    add(None, dir)?;
    for e in std::fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if e.path().is_dir() && (n.starts_with("models-") || n == "models") {
            if n == "models" {
                add(None, &e.path())?;
            } else {
                add(Some(&n), &e.path())?;
            }
        }
    }
    if files.is_empty() {
        return Err(format!("{}: no .param / .bin files (a realesrgan models/ directory, or waifu2x's with models-*/)", dir.display()));
    }
    Ok(files)
}

/// `image upscale` and `image upscale-check`: the engine in this process
fn upscale_cmd(cfg: &Config, args: &[String], check: bool) -> Result<(), String> {
    let pos: Vec<&String> = {
        let mut v = Vec::new();
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a.starts_with("--") {
                it.next();
            } else {
                v.push(a);
            }
        }
        v
    };
    let input = pos.first().ok_or(USAGE)?;
    if check && pos.len() != 2 {
        return Err("upscale-check PICTURE REFERENCE.png".into());
    }
    if !check && pos.len() != 1 {
        return Err(USAGE.into());
    }
    for key in ["--dir", "--model", "--variant", "--out", "--gpu"] {
        if args.iter().any(|v| v == key) && opt(args, key).is_none_or(|v| v.starts_with("--")) {
            return Err(format!("{key} needs a value"));
        }
    }
    let scale = number::<u32>(args, "--scale")?;
    let noise = number::<i32>(args, "--noise")?;
    let tile = number::<u32>(args, "--tile")?;
    let runs = number::<usize>(args, "--runs")?.unwrap_or(1).max(1);
    nextrocm_engine_core::use_kind("image");
    let files = match opt(args, "--dir") {
        Some(d) => ncnn_files(Path::new(d))?,
        None => model(cfg, Some(opt(args, "--model").unwrap_or("real-esrgan")))?.1,
    };
    let gs = gpus(args)?;
    let baseline = crate::speech::allocation_stats(&gs[0])?.0;
    let mut log = |s: String| eprintln!("{s}");
    let t0 = std::time::Instant::now();
    let e = nextrocm_image_upscale::Upscaler::load(&files, &gs, &mut log).map_err(|e| e.0)?;
    eprintln!("loaded in {:.2} s", t0.elapsed().as_secs_f64());
    // Keep cleanup observable even when inference, decoding, or telemetry fails.
    let result = (|| -> Result<nextrocm_image::Picture, String> {
        let pic = nextrocm_image::Picture::decode(
            &std::fs::read(input).map_err(|e| format!("{input}: {e}"))?,
        )?;
        let req = nextrocm_image::UpscaleRequest {
            picture: pic,
            variant: opt(args, "--variant").map(str::to_string),
            scale,
            noise,
            tile,
        };
        use nextrocm_image::ImageEngine;
        let mut out = None;
        for r in 0..runs {
            let (p, info) = e.upscale(&req, &mut |_| Ok(())).map_err(|e| e.0)?;
            eprintln!(
                "run {}: {} {}x{} -> {}x{} ({}x{}, {} tiles of {}) in {:.3} s on {}",
                r + 1,
                info.variant,
                req.picture.width,
                req.picture.height,
                p.width,
                p.height,
                info.scale,
                info.noise
                    .map(|n| format!(" denoise {n}"))
                    .unwrap_or_default(),
                info.tiles,
                info.tile,
                info.seconds,
                gs[0].name
            );
            let (current, peak) = crate::speech::allocation_stats(&gs[0])?;
            eprintln!("run {}: tensor bytes current={current}, peak={peak}; excludes HIP/driver/desktop memory", r + 1);
            out = Some(p);
        }
        out.ok_or_else(|| "upscale produced no output".to_string())
    })();
    let before_drop = crate::speech::allocation_stats(&gs[0]);
    drop(e);
    let cleanup = (|| -> Result<(), String> {
        gs[0].sync().map_err(|e| e.0)?;
        let (current, peak) = crate::speech::allocation_stats(&gs[0])?;
        eprintln!(
            "after drop(engine): tensor bytes current={current}, peak={peak}, baseline={baseline}"
        );
        if current != baseline {
            return Err(format!(
                "upscale allocation cleanup failed: {current} bytes remain, baseline {baseline}"
            ));
        }
        let (loaded, peak) = before_drop?;
        eprintln!("before drop(engine): tensor bytes current={loaded}, peak={peak}; cleanup PASS");
        Ok(())
    })();
    let p = match (result, cleanup) {
        (Ok(p), Ok(())) => p,
        (Err(error), Ok(())) | (Ok(_), Err(error)) => return Err(error),
        (Err(error), Err(cleanup)) => return Err(format!("{error}; {cleanup}")),
    };
    if let Some(o) = opt(args, "--out") {
        p.write_png(Path::new(o), &[])
            .map_err(|e| format!("{o}: {e}"))?;
        eprintln!("saved {o}");
    }
    if check {
        let r = pos.get(1).ok_or("upscale-check PICTURE REFERENCE.png")?;
        let want =
            nextrocm_image::Picture::decode(&std::fs::read(r).map_err(|e| format!("{r}: {e}"))?)?;
        check_picture(&p, &want, r)?;
    }
    Ok(())
}

// Validation gate for the verified CPU/ncnn references, not a promise of bit identity.
const MIN_PSNR_DB: f64 = 40.0;
const MAX_PIXEL_ERROR: i32 = 16;

fn check_picture(
    p: &nextrocm_image::Picture,
    want: &nextrocm_image::Picture,
    name: &str,
) -> Result<(), String> {
    if (p.width, p.height, p.channels) != (want.width, want.height, want.channels) {
        return Err(format!(
            "reference shape/channels {:?}, ours {:?}",
            (want.width, want.height, want.channels),
            (p.width, p.height, p.channels)
        ));
    }
    let c = p.channels as usize;
    let n = (p.width as usize)
        .checked_mul(p.height as usize)
        .and_then(|n| n.checked_mul(c))
        .ok_or("comparison size overflow")?;
    if n == 0 || !(c == 3 || c == 4) || p.data.len() != n || want.data.len() != n {
        return Err("invalid comparison picture storage".into());
    }
    let mut channel_se = vec![0f64; c];
    let (mut worst, mut same, mut near) = (0i32, 0usize, 0usize);
    for (i, (&a, &b)) in p.data.iter().zip(&want.data).enumerate() {
        let d = a as i32 - b as i32;
        channel_se[i % c] += (d * d) as f64;
        worst = worst.max(d.abs());
        same += usize::from(d == 0);
        near += usize::from(d.abs() <= 2);
    }
    let psnr = |mse: f64| {
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64.powi(2) / mse).log10()
        }
    };
    let overall = psnr(channel_se.iter().sum::<f64>() / n as f64);
    let channels: Vec<f64> = channel_se
        .iter()
        .map(|se| psnr(se / (n / c) as f64))
        .collect();
    println!("against {name}: PSNR {overall:.2} dB, worst {worst}/255, {:.2}% values equal, {:.2}% within 2; per-channel PSNR {channels:?}",
        100.0 * same as f64 / n as f64, 100.0 * near as f64 / n as f64);
    if worst > MAX_PIXEL_ERROR || channels.iter().any(|v| v.is_nan() || *v < MIN_PSNR_DB) {
        return Err(format!("upscale-check FAILED: requires >= {MIN_PSNR_DB} dB on every channel and worst <= {MAX_PIXEL_ERROR}/255"));
    }
    println!(
        "upscale-check PASS (>= {MIN_PSNR_DB} dB per channel, worst <= {MAX_PIXEL_ERROR}/255)"
    );
    Ok(())
}

fn number<T: std::str::FromStr>(args: &[String], key: &str) -> Result<Option<T>, String> {
    if !args.iter().any(|v| v == key) {
        return Ok(None);
    }
    opt(args, key)
        .ok_or_else(|| format!("{key} needs a number"))?
        .parse()
        .map(Some)
        .map_err(|_| format!("{key} needs a number"))
}

/// Where pictures go: --out, NS_IMAGE_OUT, ~/.local/share/nextrocm/images
fn out_dir(cfg: &Config, args: &[String]) -> PathBuf {
    opt(args, "--out")
        .map(PathBuf::from)
        .or_else(|| cfg.get("NS_IMAGE_OUT").map(PathBuf::from))
        .unwrap_or_else(|| {
            PathBuf::from(format!(
                "{}/.local/share/nextrocm/images",
                std::env::var("HOME").unwrap_or_else(|_| "/tmp".into())
            ))
        })
}

/// `nextrocm image serve`: from the host, the build image as container `nextrocm-image` (the GPU, the program and its
/// libraries, the registry, the model's and its LoRAs' files read-only, the output directory) running this command
/// inside with `--here`; there, the engine loaded and served (nextrocm_serve::image)
/// The image server's container (`image serve` in the foreground, `image start` in the background)
const SERVED: crate::served::Served = crate::served::Served {
    name: "nextrocm-image",
    kind: "image",
    port: "8086",
};

fn serve(cfg: &Config, args: &[String]) -> Result<(), String> {
    run(cfg, args, false)
}

fn run(cfg: &Config, args: &[String], detach: bool) -> Result<(), String> {
    let id = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .map(String::as_str)
        .or_else(|| opt(args, "--model"));
    let (m, files) = model(cfg, id)?;
    let model_id = m["id"].as_str().unwrap_or("image").to_string();
    let out = out_dir(cfg, args);
    if args.iter().any(|a| a == "--here") {
        return serve_here(cfg, &m, files, args, out);
    }
    // no container program asked for: the server is this program, as a process
    if crate::native::native(cfg) {
        options(cfg, &m, args)?;
        std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
        let name = SERVED.name_for(args)?;
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let mut a = vec!["image".to_string(), "serve".into(), model_id.clone()];
        let mut it = args.iter().skip(
            if id.is_some() && !args.first().is_some_and(|x| x.starts_with("--")) {
                1
            } else {
                0
            },
        );
        while let Some(x) = it.next() {
            if x == "--out" || x == "--model" || x == "--slot" {
                it.next();
                continue;
            }
            a.push(x.clone());
        }
        a.extend([
            "--out".into(),
            out.to_string_lossy().into_owned(),
            "--here".into(),
        ]);
        let port = opt(args, "--port").unwrap_or(SERVED.port).to_string();
        return crate::native::launch(cfg, &name, &port, &exe, &a, !detach);
    }
    use crate::container::{mount, Ce};
    // the engine options checked here, before a container starts (they travel inside with the other arguments)
    options(cfg, &m, args)?;
    let ce = Ce::new(cfg)?;
    ce.need_image()?;
    for f in ["nextrocm", "libnextrocm-runtime.so"] {
        crate::service::need_dist(cfg, f)?;
    }
    if args.iter().any(|a| a == "--wfe")
        && !nextrocm_serve::wfe::exists(&cfg.dist.join("wfe"), "image/index.html")
    {
        return Err(format!(
            "{}: the web front end is not built yet (./build.sh wfe)",
            cfg.dist.join("wfe").display()
        ));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let name = SERVED.name_for(args)?;
    if ce.running(&name) {
        return Err(format!(
            "{name} is running already ({} stop {name})",
            ce.bin
        ));
    }
    ce.remove(&name);
    // --init: the server is not PID 1, so a stop's SIGTERM ends it (PID 1 ignores signals it does not handle)
    let mut a = SERVED.run_args(detach, args)?;
    a.extend(ce.user_args());
    a.extend(ce.gpu_args());
    a.extend(mount(&cfg.dist, "/app", true));
    let reg = models::registry(cfg);
    let mut dirs = std::collections::BTreeSet::new();
    let mut paths = models::paths(&m);
    paths.push(reg.clone());
    for p in paths {
        let d = if p.is_dir() {
            p
        } else {
            p.parent().map(PathBuf::from).unwrap_or_default()
        };
        if !d.as_os_str().is_empty() && d.exists() && dirs.insert(d.clone()) {
            a.extend(mount(&d, &d.to_string_lossy(), true));
        }
    }
    a.extend(mount(&out, &out.to_string_lossy(), false));
    a.extend(["-e".into(), format!("NS_REGISTRY={}", reg.display())]);
    // the engines' own settings from the configuration (NS_QI_INT8=1 ...)
    for (k, v) in cfg
        .with_prefix("NS_QI_")
        .into_iter()
        .chain(cfg.with_prefix("NS_HI_"))
        .chain(cfg.with_prefix("NSD_"))
    {
        a.extend(["-e".into(), format!("{k}={v}")]);
    }
    a.extend([
        ce.image.clone(),
        "bash".into(),
        "-c".into(),
        "exec /app/nextrocm image serve \"$@\"".into(),
        "nextrocm".into(),
    ]);
    a.push(model_id);
    let mut it = args.iter().skip(
        if id.is_some() && !args.first().is_some_and(|x| x.starts_with("--")) {
            1
        } else {
            0
        },
    );
    while let Some(x) = it.next() {
        if x == "--out" || x == "--model" || x == "--slot" {
            it.next();
            continue;
        }
        a.push(x.clone());
    }
    a.extend([
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--here".into(),
    ]);
    SERVED.launch(&ce, &a, detach)
}

fn serve_here(
    cfg: &Config,
    m: &Value,
    files: ModelFiles,
    args: &[String],
    out: PathBuf,
) -> Result<(), String> {
    use nextrocm_serve::image::ImageServer;
    nextrocm_engine_core::use_kind("image");
    let arch = m["arch"].as_str().unwrap_or("").to_string();
    let ks = engines();
    let load_fn = nextrocm_image::kind_for(&ks, &arch)?.load;
    let gs = gpus(args)?;
    let gpu = gs[0].clone();
    let first = options(cfg, m, args)?;
    let loader: nextrocm_serve::image::Loader =
        Box::new(move |use_: &[LoraUse], log: &mut dyn FnMut(String)| {
            if !use_.is_empty() {
                return Err("Real-ESRGAN does not take LoRAs".into());
            }
            load_fn(&files, &gs, &first, log).map_err(|e| e.0)
        });
    let wfe = args
        .iter()
        .any(|a| a == "--wfe")
        .then(|| cfg.dist.join("wfe"));
    let cors: Vec<String> = opt(args, "--cors")
        .or(cfg.get("NS_CORS").as_deref())
        .map(|c| c.split(',').map(|x| x.trim().to_string()).collect())
        .unwrap_or_default();
    let addr = format!(
        "{}:{}",
        opt(args, "--host").unwrap_or("127.0.0.1"),
        opt(args, "--port").unwrap_or("8086")
    );
    let mut srv = ImageServer::new(
        m["id"].as_str().unwrap_or("image").to_string(),
        gpu,
        loader,
        vec![],
        vec![],
        Some(out),
        wfe,
        cors,
    )?;
    srv.idle = crate::served::idle_exit(args)?;
    srv.license = nextrocm_serve::page::licence_of(m);
    std::sync::Arc::new(srv).run(&addr)
}

/// `image vectorize`: StarVector in this process (`--check DIR`: its stages against the reference's dumps)
#[cfg(feature = "starvector-experimental")]
fn vectorize_cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let mut pos = Vec::new();
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--greedy" {
            continue;
        }
        if a.starts_with("--") {
            it.next();
        } else {
            pos.push(a.clone());
        }
    }
    let input = pos.first().ok_or(USAGE)?;
    nextrocm_engine_core::use_kind("image");
    let files = match opt(args, "--dir") {
        Some(d) => nextrocm_image_starvector::files_in(Path::new(d)).map_err(|e| e.0)?,
        None => model(cfg, Some(opt(args, "--model").unwrap_or("starvector-1b-im2svg")))?.1,
    };
    let gs = gpus(args)?;
    let mut log = |s: String| eprintln!("{s}");
    let e = nextrocm_image_starvector::StarVector::load(&files, &gs[0], &mut log).map_err(|e| e.0)?;
    if let Some(d) = opt(args, "--check") {
        let stages: Vec<&str> = opt(args, "--stages").unwrap_or("pixels,encoder,prefill,logits,greedy").split(',').collect();
        let worst = nextrocm_image_starvector::check::run(&e, Path::new(d), Path::new(input), &stages, &mut log).map_err(|e| e.0)?;
        println!("worst relative error {worst:.2e}");
        if !worst.is_finite() || worst > 0.001 {
            return Err(format!("StarVector relative L2 gate failed: {worst}"));
        }
        return Ok(());
    }
    let pic = nextrocm_image::Picture::decode(&std::fs::read(input).map_err(|e| format!("{input}: {e}"))?)?;
    let greedy = args.iter().any(|a| a == "--greedy");
    let req = nextrocm_image::VectorizeRequest {
        picture: pic,
        max_length: opt(args, "--max-length").and_then(|v| v.parse().ok()).unwrap_or(0),
        beams: if greedy { Some(1) } else { opt(args, "--beams").and_then(|v| v.parse().ok()) },
        sample: if greedy { Some(false) } else { None },
        top_p: opt(args, "--top-p").and_then(|v| v.parse().ok()),
        seed: opt(args, "--seed").and_then(|v| v.parse().ok()).unwrap_or(0),
        ..Default::default()
    };
    use nextrocm_image::ImageEngine;
    let out = e.vectorize(&req, &mut |_| Ok(())).map_err(|e| e.0)?;
    eprintln!("{} tokens in {:.2} s after {:.2} s for the picture and prompt ({:.1} tokens/s){} on {}", out.tokens, out.seconds, out.prompt_seconds,
              out.tokens as f64 / out.seconds.max(1e-9), if out.closed { ", ended by </svg>" } else { ", stopped at the length limit" }, gs[0].name);
    match opt(args, "--out") {
        Some(o) => {
            std::fs::write(o, &out.svg).map_err(|e| format!("{o}: {e}"))?;
            eprintln!("saved {o}");
        }
        None => println!("{}", out.svg),
    }
    Ok(())
}

/// The supported reference image commands.
pub fn cmd(cfg: &Config, args: &[String]) -> Result<(), String> {
    let rest = args.get(1..).unwrap_or(&[]);
    match args.first().map(String::as_str) {
        Some("engines") => {
            super::list_engines(
                engines()
                    .iter()
                    .map(|k| (k.archs.join(", "), k.name, k.options.to_vec())),
            );
            Ok(())
        }
        Some("gen") => gen(cfg, rest),
        #[cfg(feature = "starvector-experimental")]
        Some("vectorize") => vectorize_cmd(cfg, rest),
        Some("upscale") => upscale_cmd(cfg, rest, false),
        Some("upscale-check") => upscale_cmd(cfg, rest, true),
        Some("serve") => serve(cfg, rest),
        Some("start") => SERVED.start(cfg, rest, |d| run(cfg, rest, d)),
        Some("ps") => SERVED.ps(cfg, rest),
        Some("stop") => SERVED.stop(cfg, rest),
        Some("logs") => SERVED.logs(cfg, rest),
        None | Some("--help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}

#[cfg(test)]
mod comparison_tests {
    use super::*;
    fn picture(data: Vec<u8>, channels: u8) -> nextrocm_image::Picture {
        nextrocm_image::Picture {
            width: 8,
            height: 8,
            channels,
            data,
        }
    }
    #[test]
    fn exact_and_small_rounding_differences_pass() {
        let want = picture(vec![100; 192], 3);
        assert!(check_picture(&want, &want, "exact").is_ok());
        assert!(check_picture(&picture(vec![102; 192], 3), &want, "rounding").is_ok());
    }
    #[test]
    fn black_output_and_isolated_outlier_fail() {
        let want = picture(vec![100; 192], 3);
        assert!(check_picture(&picture(vec![0; 192], 3), &want, "black").is_err());
        let mut data = want.data.clone();
        data[0] += 17;
        assert!(check_picture(&picture(data, 3), &want, "outlier").is_err());
    }
    #[test]
    fn alpha_cannot_hide_behind_rgb_and_cannot_be_dropped() {
        let want = picture(vec![100; 256], 4);
        let mut data = want.data.clone();
        for pixel in data.chunks_exact_mut(4) {
            pixel[3] += 3;
        }
        assert!(check_picture(&picture(data, 4), &want, "alpha").is_err());
        assert!(check_picture(&picture(vec![100; 192], 3), &want, "missing alpha").is_err());
    }
}

#[cfg(test)]
mod local_weight_tests {
    use super::*;
    #[test]
    fn resolved_local_bin_is_added_without_changing_param_role() {
        let path = std::env::temp_dir().join(format!(
            "nextrocm-esrgan-local-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, [0u8]).unwrap();
        let entry = serde_json::json!({"arch":"realesrgan", "env":{"NS_ESRGAN_X4PLUS_BIN":path}});
        let mut files = ModelFiles::from([(
            "realesrgan-x4plus.param".into(),
            PathBuf::from("pinned.param"),
        )]);
        let result = local_upscale_weights(&entry, &mut files);
        std::fs::remove_file(&path).unwrap();
        result.unwrap();
        assert_eq!(files["realesrgan-x4plus.bin"], path);
        assert_eq!(
            files["realesrgan-x4plus.param"],
            PathBuf::from("pinned.param")
        );
    }
    #[test]
    fn explicit_role_wins_and_missing_or_directory_fallback_fails() {
        let mut files = ModelFiles::from([(
            "realesrgan-x4plus.bin".into(),
            PathBuf::from("explicit.bin"),
        )]);
        let empty = serde_json::json!({"arch":"realesrgan"});
        local_upscale_weights(&empty, &mut files).unwrap();
        assert_eq!(
            files["realesrgan-x4plus.bin"],
            PathBuf::from("explicit.bin")
        );
        files.clear();
        assert!(local_upscale_weights(&empty, &mut files)
            .unwrap_err()
            .contains("NS_ESRGAN_X4PLUS_BIN"));
        let directory = serde_json::json!({"arch":"realesrgan", "env":{"NS_ESRGAN_X4PLUS_BIN":std::env::temp_dir()}});
        assert!(local_upscale_weights(&directory, &mut files)
            .unwrap_err()
            .contains("not a file"));
        assert!(files.is_empty());
    }
}
