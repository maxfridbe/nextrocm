use nextrocm_models::{registry as models, Config};
use nextrocm_serve::http::Target;
use nextrocm_serve::switch::{Model, Switch};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;
fn opt<'a>(args: &'a [String], k: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}
pub(crate) struct Store(pub(crate) Config);

impl nextrocm_serve::home::registry::Store for Store {
    fn models(&self) -> Result<Vec<serde_json::Value>, String> {
        models::all(&self.0)
    }
    fn update(
        &self,
        id: &str,
        f: &mut dyn FnMut(&mut serde_json::Value),
    ) -> Result<serde_json::Value, String> {
        models::update(&self.0, id, f)
    }
    fn registry_file(&self) -> PathBuf {
        models::registry(&self.0)
    }
    fn get(&self, name: &str) -> Option<String> {
        self.0.get(name)
    }
    fn in_env(&self, name: &str) -> bool {
        self.0.from_env(name)
    }
    fn file(&self) -> PathBuf {
        self.0.file()
    }
    fn guide(&self, entry: &serde_json::Value) -> Option<String> {
        nextrocm_models::catalog::Catalog::load(&self.0)
            .ok()?
            .guide(entry)
    }
    fn cards(&self) -> Vec<serde_json::Value> {
        nextrocm_models::catalog::Catalog::load(&self.0)
            .map(|c| c.models)
            .unwrap_or_default()
    }
    fn card_files(&self, id: &str) -> Vec<(String, u64)> {
        nextrocm_models::catalog::Catalog::load(&self.0)
            .ok()
            .and_then(|c| c.model(id).map(|m| c.file_list(m)))
            .unwrap_or_default()
    }
}

pub fn home(cfg: &Config, args: &[String]) -> Result<(), String> {
    use nextrocm_serve::home::{Home, Options, Ports};
    let port = |k: &str, d: u16| cfg.get(k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let wfe = cfg.dist.join("wfe");
    if !nextrocm_serve::wfe::exists(&wfe, "home/index.html") {
        return Err(format!(
            "{}: the page is not built yet (./build.sh wfe)",
            wfe.display()
        ));
    }
    let idle = opt(args, "--idle-minutes")
        .map(str::to_string)
        .or_else(|| cfg.get("NS_IDLE_MINUTES"))
        .map_or(Ok(10), |v| v.parse::<u64>().map_err(|_| "--idle-minutes N"))?;
    let o = Options {
        wfe,
        exe: std::env::current_exe().map_err(|e| e.to_string())?,
        ports: Ports {
            llm: port("NS_PORT", 8085),
            switch: port("NS_SWITCH_PORT", 8001),
            image: port("NS_IMAGE_PORT", 8086),
            audio: port("NS_AUDIO_PORT", 8087),
            model3d: port("NS_MODEL3D_PORT", 8088),
            image2: port("NS_IMAGE_PORT2", 8096),
            audio2: port("NS_AUDIO_PORT2", 8097),
            model3d2: port("NS_MODEL3D_PORT2", 8098),
            video: port("NS_VIDEO_PORT", 8090),
            chat_ui: port("NS_CHAT_UI_PORT", 3000),
        },
        serve_host: opt(args, "--serve-host").unwrap_or("0.0.0.0").to_string(),
        cfg: Box::new(Store(Config::load())),
        idle_minutes: idle,
    };
    let addr = format!(
        "{}:{}",
        opt(args, "--host").unwrap_or("0.0.0.0"),
        opt(args, "--port")
            .map(str::to_string)
            .or_else(|| cfg.get("NS_SERVE_PORT"))
            .unwrap_or("8000".into())
    );
    let home = Home::new(o)?;
    let host = opt(args, "--host").unwrap_or("0.0.0.0");
    let tls_port = opt(args, "--tls-port")
        .map(str::to_string)
        .or_else(|| cfg.get("NS_TLS_PORT"))
        .unwrap_or("8443".into());
    match tls_port.trim().parse::<u16>() {
        Ok(0) => {}
        Ok(p) => {
            let (h, a) = (home.clone(), format!("{host}:{p}"));
            std::thread::spawn(move || {
                if let Err(e) = h.run_tls(&a) {
                    eprintln!("nextrocm serve: no https on {a}: {e}");
                }
            });
        }
        Err(_) => return Err(format!("invalid TLS port: {tls_port}")),
    }
    home.run(&addr)
}

fn secs(args: &[String], key: &str, default: f64) -> Result<Duration, String> {
    let value = opt(args, key)
        .map(str::parse::<f64>)
        .transpose()
        .map_err(|_| format!("{key} SECONDS"))?
        .unwrap_or(default);
    Duration::try_from_secs_f64(value)
        .map_err(|_| format!("{key} must be finite nonnegative seconds"))
}

/// `nextrocm switch [--host 0.0.0.0] [--port 8001] [--upstream 127.0.0.1:8085] [--studio URL] [--wait 420]
/// [--in-use 90] [--alias OLD=NEW]... [--images 127.0.0.1:8086] [--idle-minutes 10]`
pub fn switch(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help") {
        println!("nextrocm switch [--host 0.0.0.0] [--port 8001] [--upstream 127.0.0.1:8085] [--studio URL] [--wait 420] [--in-use 90] [--alias OLD=NEW] [--images HOST:PORT] [--idle-minutes N]");
        return Ok(());
    }
    for (i, arg) in args.iter().enumerate() {
        if i % 2 == 0
            && (!matches!(
                arg.as_str(),
                "--host"
                    | "--port"
                    | "--upstream"
                    | "--studio"
                    | "--wait"
                    | "--in-use"
                    | "--alias"
                    | "--images"
                    | "--idle-minutes"
            ) || args.get(i + 1).is_none())
        {
            return Err(format!("unknown switch option or missing value: {arg}"));
        }
    }
    // the same settings, the closure's own (the registry is read on every request)
    let cfg = Config::load();
    let list: nextrocm_serve::switch::Models = Box::new(move || {
        let all = models::all(&cfg).unwrap_or_else(|e| {
            eprintln!("the registry: {e}");
            Vec::new()
        });
        // the chat models (entries of other kinds - image, video, audio, lora - are not chat models)
        all.iter()
            .filter(|m| m["enabled"] != false && models::kind_of(m) == "llm")
            .filter_map(|m| {
                let id = m["id"].as_str()?.to_string();
                // its cards as the front door counts them (life.rs gpus_of): none named, the first; "all": every card
                let gpus = match m["gpus"].as_str().map(str::trim) {
                    None | Some("") => vec![0],
                    Some("all") => Vec::new(),
                    Some(g) => nextrocm_serve::home::registry::gpu_list(g),
                };
                Some(Model {
                    title: m["title"].as_str().unwrap_or(&id).to_string(),
                    tools: m["tools"] != false,
                    tasks: m["tasks"] != false,
                    pinned: m["pinned"] == true,
                    gpus,
                    one_of: nextrocm_serve::home::registry::one_of(m),
                    id,
                })
            })
            .collect()
    });
    let mut aliases = HashMap::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--alias" {
            let (o, n) = args
                .get(i + 1)
                .and_then(|x| x.split_once('='))
                .ok_or("--alias OLD=NEW")?;
            aliases.insert(o.to_string(), n.to_string());
        }
    }
    let upstream = Target::Tcp(
        opt(args, "--upstream")
            .unwrap_or("127.0.0.1:8085")
            .to_string(),
    );
    let studio = opt(args, "--studio").unwrap_or("http://127.0.0.1:8090/rpc/llm.mode");
    let mut sw = Switch::new(
        list,
        aliases,
        upstream,
        studio,
        secs(args, "--wait", 420.0)?,
        secs(args, "--in-use", 90.0)?,
    )?;
    sw.images = opt(args, "--images").map(|a| Target::Tcp(a.to_string()));
    // the front door's maintenance notice, beside the registry
    sw.maintenance = Some(nextrocm_serve::maintenance::file_beside(&models::registry(
        &Config::load(),
    )));
    // the GPU run lock (NS_GPU_LOCK): no chat model swapped in while a run holds it
    sw.gpu_lock = nextrocm_serve::gpulock::path(Config::load().get("NS_GPU_LOCK"));
    // the same idle time as the front door's (NS_IDLE_MINUTES), unless given
    let idle = opt(args, "--idle-minutes")
        .map(str::to_string)
        .or_else(|| Config::load().get("NS_IDLE_MINUTES"));
    sw.idle = idle
        .map(|v| v.parse::<f64>().map_err(|_| "--idle-minutes N"))
        .transpose()?
        .map(|m| {
            if m.is_finite() && m >= 0.0 {
                Ok(m)
            } else {
                Err("--idle-minutes must be finite nonnegative minutes")
            }
        })
        .transpose()?
        .filter(|m| *m > 0.0)
        .map(|m| {
            Duration::try_from_secs_f64(m * 60.0)
                .map_err(|_| "--idle-minutes must be finite and representable")
        })
        .transpose()?;
    let addr = format!(
        "{}:{}",
        opt(args, "--host").unwrap_or("0.0.0.0"),
        opt(args, "--port").unwrap_or("8001")
    );
    std::sync::Arc::new(sw).run(&addr)
}

#[cfg(test)]
mod switch_tests {
    use super::*;
    #[test]
    fn switch_durations_reject_invalid_values_without_starting_services() {
        for bad in ["NaN", "inf", "-1", "1e300", "invalid"] {
            assert!(secs(&["--wait".into(), bad.into()], "--wait", 420.0).is_err());
        }
        assert_eq!(
            secs(&[], "--wait", 420.0).unwrap(),
            Duration::from_secs(420)
        );
        assert_eq!(
            secs(&["--wait".into(), "0.5".into()], "--wait", 420.0).unwrap(),
            Duration::from_millis(500)
        );
    }
    #[test]
    fn switch_rejects_unknown_and_valueless_options_before_loading_registry() {
        assert!(switch(&["--unknown".into(), "1".into()]).is_err());
        assert!(switch(&["--port".into()]).is_err());
    }
}
