use nextrocm_models::{registry as models, Config};
use std::path::PathBuf;
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
    Home::new(o)?.run(&addr)
}
