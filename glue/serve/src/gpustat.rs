//! AMD telemetry with the reference front-door schema. HIP reports allocatable capacity and free memory;
//! sysfs supplies available sensors, and DRM fdinfo accounts for each process's VRAM and GTT.
//! The APU's allocatable HIP memory differs from its small firmware-reserved VRAM segment.

use std::path::PathBuf;

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub struct Options {
    pub out: PathBuf,
    pub interval: Duration,
    /// Retained reference sampler option; AMD hwmon supplies the available power reading.
    pub power_window: Duration,
    /// the card (PCI address); none: the first AMD GPU
    pub pci: Option<String>,
    /// VRAM in MiB; none: what the driver says
    pub vram_mb: Option<u64>,
    /// one sample, printed instead of written
    pub once: bool,
}

/// One card's row of the query
#[derive(Clone, Debug, Default)]
struct Row {
    pci: String,
    name: String,
    used_mb: u64,
    total_mb: u64,
    busy: Option<f64>,
    temp: Option<f64>,
    temp_mem: Option<f64>,
    power: Option<f64>,
    cap: Option<f64>,
    fan_pct: Option<f64>,
    sm_mhz: Option<f64>,
    gen: Option<u32>,
    width: Option<u32>,
    gen_max: Option<u32>,
    width_max: Option<u32>,
}

fn rows() -> Vec<Row> {
    let Ok(devices) = nextrocm_engine_core::gpus() else {
        return Vec::new();
    };
    devices
        .into_iter()
        .filter_map(|(index, name)| {
            let gpu = nextrocm_engine_core::Gpu::open(index).ok()?;
            let pci = gpu.pci.clone()?;
            let (total, free) = gpu.memory().ok()?;
            let dev = std::path::PathBuf::from(format!("/sys/bus/pci/devices/{pci}"));
            let num =
                |p: std::path::PathBuf| std::fs::read_to_string(p).ok()?.trim().parse::<f64>().ok();
            let hw = std::fs::read_dir(dev.join("hwmon"))
                .ok()
                .and_then(|mut d| d.next())
                .and_then(Result::ok)
                .map(|e| e.path());
            Some(Row {
                pci,
                name,
                total_mb: total >> 20,
                used_mb: total.saturating_sub(free?) >> 20,
                busy: num(dev.join("gpu_busy_percent")),
                temp: hw
                    .as_ref()
                    .and_then(|h| num(h.join("temp1_input")))
                    .map(|v| v / 1000.0),
                power: hw
                    .as_ref()
                    .and_then(|h| num(h.join("power1_average")))
                    .map(|v| v / 1e6),
                cap: hw
                    .as_ref()
                    .and_then(|h| num(h.join("power1_cap")))
                    .map(|v| v / 1e6),
                ..Default::default()
            })
        })
        .collect()
}

// The reference DRM fdinfo approach, using AMD's VRAM + GTT fields (APUs allocate in GTT).
fn processes(pci: &str) -> Vec<(u32, u64)> {
    let mut clients = std::collections::BTreeSet::new();
    let mut procs = std::collections::BTreeMap::<u32, u64>::new();
    for p in std::fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Ok(pid) = p.file_name().to_string_lossy().parse::<u32>() else {
            continue;
        };
        for fd in std::fs::read_dir(p.path().join("fdinfo"))
            .into_iter()
            .flatten()
            .flatten()
        {
            let Ok(info) = std::fs::read_to_string(fd.path()) else {
                continue;
            };
            let fields: std::collections::BTreeMap<_, _> = info
                .lines()
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k, v.trim()))
                .collect();
            if fields.get("drm-pdev").copied() != Some(pci) {
                continue;
            }
            let Some(cid) = fields.get("drm-client-id") else {
                continue;
            };
            if !clients.insert(cid.to_string()) {
                continue;
            }
            let n = |key| {
                fields
                    .get(key)
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(0)
            };
            *procs.entry(pid).or_default() += (n("drm-memory-vram") + n("drm-memory-gtt")) / 1024;
        }
    }
    procs.into_iter().collect()
}

/// The first AMD GPU, as a PCI address
pub fn find_card() -> Option<String> {
    find_cards().into_iter().next()
}

/// Every AMD GPU by PCI address in HIP device-index order.
pub fn find_cards() -> Vec<String> {
    rows().into_iter().map(|r| r.pci).collect()
}

fn round1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// One card's sampler
pub struct Card {
    pub pci: String,
    pub name: String,
    total_mb: u64,
    vram_override: Option<u64>,
    /// joules so far (power x time between samples)
    energy_j: f64,
    last: Option<Instant>,
}

impl Card {
    pub fn open(pci: &str, vram_mb: Option<u64>) -> Card {
        let r = rows().into_iter().find(|r| r.pci == pci);
        Card {
            pci: pci.to_string(),
            name: r
                .as_ref()
                .map_or_else(|| format!("AMD GPU {pci}"), |r| r.name.clone()),
            total_mb: vram_mb
                .or_else(|| r.as_ref().map(|r| r.total_mb))
                .unwrap_or(0),
            vram_override: vram_mb,
            energy_j: 0.0,
            last: None,
        }
    }

    /// A reading (the gpustat file's fields) and the VRAM each process holds on the card, in MiB
    pub fn sample(&mut self, _power_window: Duration) -> (Value, Vec<(u32, u64)>) {
        let now = Instant::now();
        let r = rows()
            .into_iter()
            .find(|r| r.pci == self.pci)
            .unwrap_or_default();
        if let (Some(last), Some(p)) = (self.last, r.power) {
            self.energy_j += p * (now - last).as_secs_f64();
        }
        self.last = Some(now);
        if self.vram_override.is_none() && r.total_mb > 0 {
            self.total_mb = r.total_mb;
        }
        let load1 = std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|l| l.split_whitespace().next()?.parse::<f64>().ok())
            .unwrap_or(0.0);
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |d| d.as_secs_f64());
        let link = |g: Option<u32>, w: Option<u32>| json!({"gen": g, "width": w.unwrap_or(0)});
        let out = json!({
            "name": self.name, "pci": self.pci, "ts": ts,
            "vram_used_mb": r.used_mb, "vram_total_mb": self.total_mb,
            "busy_pct": r.busy.map(round1).unwrap_or(0.0),
            "temp_pkg": r.temp, "temp_vram": r.temp_mem,
            "power_w": r.power.map(round1), "pkg_power_w": Value::Null, "power_cap_w": r.cap.map(|c| c.round() as i64),
            "temp_vram_max": r.temp_mem, "temp_pcie": Value::Null, "temp_mctrl": Value::Null,
            "pcie": json!({"cur": link(r.gen, r.width), "card_max": link(r.gen_max, r.width_max), "slot_max": link(r.gen_max, r.width_max)}),
            "host_load1": load1, "host_cpus": std::thread::available_parallelism().map_or(1, |n| n.get()),
            "energy_j": round1(self.energy_j),
            "fan_pct": r.fan_pct, "fan_rpm": Value::Null,
            "freq_mhz": r.sm_mhz.map(|f| f.round() as i64),
        });
        let procs = processes(&self.pci);
        (out, procs)
    }
}

/// Samples until the process ends (or once)
pub fn run(o: Options) -> Result<(), String> {
    let pci = o
        .pci
        .clone()
        .or_else(find_card)
        .ok_or("no HIP GPU (--pci ADDR)")?;
    let mut card = Card::open(&pci, o.vram_mb);
    loop {
        let (out, _) = card.sample(o.power_window);
        if o.once {
            println!("{out}");
            return Ok(());
        }
        let tmp = o.out.with_extension("json.tmp");
        std::fs::write(&tmp, out.to_string())
            .and_then(|_| {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o644))?;
                std::fs::rename(&tmp, &o.out)
            })
            .map_err(|e| format!("{}: {e}", o.out.display()))?;
        std::thread::sleep(o.interval);
    }
}
