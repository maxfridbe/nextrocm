//! Raw bindings to `libnextrocm.so`: one function pointer per shared declaration of `kernels/ns/ns.h` (an engine
//! binds its own kernels by name: `Api::symbol`).
//!
//! The library is opened at run time (`dlopen`), not linked: the Rust side builds anywhere without the HIP
//! toolchain, and the kernels rebuild without relinking the runtime. Nothing here is safe to call directly;
//! `nextrocm-core` wraps it. A binary built with the libraries beside it (build.rs) carries them and opens them from
//! anonymous memory files; NS_LIB_FROM_DISK=1 or NEXTROCM_LIB opens the files instead.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
pub mod variant;

include!(concat!(env!("OUT_DIR"), "/libs_embedded.rs"));

extern "C" {
    fn memfd_create(name: *const c_char, flags: u32) -> c_int;
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: c_int = 2;
const RTLD_LOCAL: c_int = 0;
const RTLD_GLOBAL: c_int = 0x100;

/// An opened GPU (`ns_gpu*`).
pub type Gpu = *mut c_void;

#[allow(clippy::type_complexity)]
pub struct Api {
    /// the library's handle: an engine binds its own ABI from it (`symbol`)
    handle: *mut c_void,
    /// where it was opened from (a name, for a library out of the binary)
    path: PathBuf,
    /// the directory the silos are in (`silo/`): beside the library, or beside the executable for one out of the binary
    dir: PathBuf,
    pub last_error: unsafe extern "C" fn() -> *const c_char,
    pub version: unsafe extern "C" fn() -> *const c_char,
    pub gpu_count: unsafe extern "C" fn() -> c_int,
    pub gpu_name: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    pub gpu_pci: unsafe extern "C" fn(c_int, *mut c_char, usize) -> c_int,
    pub gpu_units: unsafe extern "C" fn(c_int, *mut c_int, *mut c_int) -> c_int,
    pub gpu_props: Option<
        unsafe extern "C" fn(c_int, *mut u64, *mut c_int, *mut c_int, *mut c_char, usize) -> c_int,
    >,
    pub gpu_cc: unsafe extern "C" fn(c_int, *mut c_int) -> c_int,
    pub gpu_open: unsafe extern "C" fn(c_int, *mut Gpu) -> c_int,
    pub gpu_close: unsafe extern "C" fn(Gpu),
    pub gpu_queue: unsafe extern "C" fn(Gpu) -> *mut c_void,
    pub gpu_memory: unsafe extern "C" fn(Gpu, *mut u64, *mut i64) -> c_int,
    pub alloc: unsafe extern "C" fn(Gpu, usize, *mut *mut c_void) -> c_int,
    pub free: unsafe extern "C" fn(Gpu, *mut c_void) -> c_int,
    pub alloc_host: unsafe extern "C" fn(Gpu, usize, *mut *mut c_void) -> c_int,
    pub free_host: unsafe extern "C" fn(Gpu, *mut c_void) -> c_int,
    pub copy_to: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_from: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_dev: unsafe extern "C" fn(Gpu, *mut c_void, *const c_void, usize) -> c_int,
    pub copy_2d:
        unsafe extern "C" fn(Gpu, *mut c_void, usize, *const c_void, usize, usize, usize) -> c_int,
    pub mark: unsafe extern "C" fn(Gpu, *mut i64) -> c_int,
    pub stream_copy: unsafe extern "C" fn(
        Gpu,
        *mut c_void,
        *const c_void,
        usize,
        *const i64,
        c_int,
        c_int,
        *mut i64,
    ) -> c_int,
    pub await_: unsafe extern "C" fn(Gpu, i64) -> c_int,
    pub ticket_wait: unsafe extern "C" fn(Gpu, i64) -> c_int,
    pub capture_begin: unsafe extern "C" fn(Gpu) -> c_int,
    pub capture_launch: unsafe extern "C" fn(Gpu) -> c_int,
    pub capture_abort: unsafe extern "C" fn(Gpu) -> c_int,
    pub stamp: unsafe extern "C" fn(Gpu, *mut i64) -> c_int,
    pub elapsed: unsafe extern "C" fn(Gpu, i64, i64, *mut f64) -> c_int,
    pub fill: unsafe extern "C" fn(Gpu, *mut c_void, u8, usize) -> c_int,
    pub sync: unsafe extern "C" fn(Gpu) -> c_int,
    pub copy_peer:
        unsafe extern "C" fn(Gpu, *mut c_void, Gpu, *const c_void, usize, *mut c_void) -> c_int,
    pub copy_peer_async:
        unsafe extern "C" fn(Gpu, *mut c_void, Gpu, *const c_void, usize, *mut c_void) -> c_int,
}

// SAFETY: plain function pointers into a library that stays loaded for the process's life.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// NS_LIB_FROM_DISK=1: the kernel libraries from files, not from the binary (rebuild a kernel without relinking)
fn from_disk() -> bool {
    std::env::var("NS_LIB_FROM_DISK").is_ok_and(|v| v == "1")
}

/// A library of the binary by file name
fn embedded(name: &str) -> Option<&'static [u8]> {
    LIBS.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
}

/// The libraries a kind's library needs by name, opened before it: the image library links stable-diffusion.cpp's
/// (found through its RUNPATH $ORIGIN beside a file; out of the binary, by its soname once it is open)
fn deps(_name: &str) -> &'static [&'static str] {
    &[]
}

/// An embedded library as a file the loader can open: an anonymous memory file (the descriptor stays open), its
/// /proc/self/fd path. Once a name.
fn materialize(name: &str) -> Result<PathBuf, String> {
    use std::io::Write;
    use std::os::fd::FromRawFd;
    static MADE: std::sync::Mutex<Vec<(String, PathBuf)>> = std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, p)) = made.iter().find(|(n, _)| n == name) {
        return Ok(p.clone());
    }
    let bytes = embedded(name).ok_or_else(|| format!("{name}: not in this binary"))?;
    let c = CString::new(name).map_err(|e| e.to_string())?;
    // SAFETY: memfd_create with a valid C string; the descriptor is checked.
    let fd = unsafe { memfd_create(c.as_ptr(), 0) };
    if fd < 0 {
        return Err(format!(
            "{name}: memfd_create: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: a descriptor just made and owned here.
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    f.write_all(bytes).map_err(|e| format!("{name}: {e}"))?;
    std::mem::forget(f);
    let p = PathBuf::from(format!("/proc/self/fd/{fd}"));
    made.push((name.to_string(), p.clone()));
    Ok(p)
}

/// Whether this binary carries the kind's library (and NS_LIB_FROM_DISK=1 does not turn it off): a check for the file
/// beside the executable is then moot
pub fn has_embedded(kind: &str) -> bool {
    !from_disk() && embedded(&format!("libnextrocm-{kind}.so")).is_some()
}

/// The kind's library out of the binary as a path (`/proc/self/fd/N`, which a child process that inherits the
/// descriptor opens too: `ldd` can check it), None when this binary has none or NS_LIB_FROM_DISK=1
pub fn embedded_path(kind: &str) -> Option<PathBuf> {
    if !has_embedded(kind) {
        return None;
    }
    materialize(&format!("libnextrocm-{kind}.so")).ok()
}

/// dlopen a path globally (its symbols and its soname are then found by the libraries opened after it)
fn open(path: &Path) -> Result<(), String> {
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
    // SAFETY: dlopen with a valid C string; the handle is checked.
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
    if h.is_null() {
        return Err(format!("{}: {}", path.display(), dl_error()));
    }
    Ok(())
}

fn dl_error() -> String {
    // SAFETY: dlerror returns null or a C string valid until the next dl* call on this thread.
    let p = unsafe { dlerror() };
    if p.is_null() {
        "unknown dlopen error".into()
    } else {
        // SAFETY: non-null, NUL-terminated (above).
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

impl Api {
    /// A kind's kernel library (`libnextrocm-<kind>.so`: "llm", "image", "video"): `NEXTROCM_LIB` when set, else the
    /// copy inside the binary (unless NS_LIB_FROM_DISK=1), else the library beside the executable, else in `dist/` of
    /// the working directory.
    pub fn load(kind: &str) -> Result<Api, String> {
        if let Some(p) = std::env::var_os("NEXTROCM_LIB") {
            return Api::load_from(&PathBuf::from(p));
        }
        for candidate in [kind, "runtime"] {
            let name = format!("libnextrocm-{candidate}.so");
            if !from_disk() {
                if let Some(api) = Api::load_embedded(&name)? {
                    return Ok(api);
                }
            }
            if let Some(beside) = std::env::current_exe()
                .ok()
                .and_then(|e| e.parent().map(|d| d.join(&name)))
            {
                if beside.exists() {
                    return Api::load_from(&beside);
                }
            }
            let p = PathBuf::from("dist").join(&name);
            if p.exists() {
                return Api::load_from(&p);
            }
            if candidate == "runtime" {
                break;
            }
        }
        Err(format!(
            "no libnextrocm-{kind}.so or legacy libnextrocm-runtime.so; set NEXTROCM_LIB"
        ))
    }

    /// The kind's library out of the binary (None when this binary has none): the libraries it needs by name first
    /// (`deps`), each as an anonymous memory file opened by its /proc/self/fd path
    fn load_embedded(name: &str) -> Result<Option<Api>, String> {
        if embedded(name).is_none() {
            return Ok(None);
        }
        for d in deps(name) {
            if embedded(d).is_some() {
                open(&materialize(d)?)?;
            }
        }
        let p = materialize(name)?;
        let dir = std::env::var_os("NS_SILO_DIR")
            .map(PathBuf::from)
            .and_then(|d| d.parent().map(Path::to_path_buf))
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|e| e.parent().map(Path::to_path_buf))
            })
            .unwrap_or_else(|| PathBuf::from("dist"));
        Api::load_at(&p, PathBuf::from(format!("{name} (in the binary)")), dir).map(Some)
    }

    pub fn load_from(path: &Path) -> Result<Api, String> {
        let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
        Api::load_at(path, path.to_path_buf(), dir)
    }

    #[allow(clippy::missing_transmute_annotations)] // the target type is each field's, from ns.h
    fn load_at(path: &Path, shown: PathBuf, dir: PathBuf) -> Result<Api, String> {
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(format!("{}: {}", path.display(), dl_error()));
        }
        macro_rules! sym {
            ($n:literal) => {{
                let n = CString::new($n).unwrap();
                // SAFETY: a symbol lookup in the handle above; the pointer is checked and has ns.h's signature.
                let p = unsafe { dlsym(h, n.as_ptr()) };
                if p.is_null() {
                    return Err(format!("{}: no symbol {}", path.display(), $n));
                }
                // SAFETY: the declaration in ns.h is the field's type.
                unsafe { std::mem::transmute::<*mut c_void, _>(p) }
            }};
        }
        Ok(Api {
            handle: h,
            path: shown,
            dir,
            last_error: sym!("ns_last_error"),
            version: sym!("ns_version"),
            gpu_count: sym!("ns_gpu_count"),
            gpu_name: sym!("ns_gpu_name"),
            gpu_pci: sym!("ns_gpu_pci"),
            gpu_units: sym!("ns_gpu_units"),
            gpu_props: {
                let p = unsafe { dlsym(h, c"ns_gpu_props".as_ptr()) };
                if p.is_null() {
                    None
                } else {
                    Some(unsafe { std::mem::transmute(p) })
                }
            },
            gpu_cc: sym!("ns_gpu_cc"),
            gpu_open: sym!("ns_gpu_open"),
            gpu_close: sym!("ns_gpu_close"),
            gpu_queue: sym!("ns_gpu_queue"),
            gpu_memory: sym!("ns_gpu_memory"),
            alloc: sym!("ns_alloc"),
            free: sym!("ns_free"),
            alloc_host: sym!("ns_alloc_host"),
            free_host: sym!("ns_free_host"),
            copy_to: sym!("ns_copy_to"),
            copy_from: sym!("ns_copy_from"),
            copy_dev: sym!("ns_copy_dev"),
            copy_2d: sym!("ns_copy_2d"),
            mark: sym!("ns_mark"),
            stream_copy: sym!("ns_stream_copy"),
            await_: sym!("ns_await"),
            ticket_wait: sym!("ns_ticket_wait"),
            capture_begin: sym!("ns_capture_begin"),
            capture_launch: sym!("ns_capture_launch"),
            capture_abort: sym!("ns_capture_abort"),
            stamp: sym!("ns_stamp"),
            elapsed: sym!("ns_elapsed"),
            fill: sym!("ns_fill"),
            sync: sym!("ns_sync"),
            copy_peer: sym!("ns_copy_peer"),
            copy_peer_async: sym!("ns_copy_peer_async"),
        })
    }

    /// Where the library was opened from (a name and "in the binary" for an embedded one)
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// A symbol of the library by name (null when it has none): an engine's own ABI (`kernels/<kind>/<arch>`) is
    /// bound by its crate, not here.
    pub fn symbol(&self, name: &str) -> *mut c_void {
        let Ok(n) = CString::new(name) else {
            return std::ptr::null_mut();
        };
        // SAFETY: a lookup in the handle the library was opened with; it stays open for the process.
        unsafe { dlsym(self.handle, n.as_ptr()) }
    }

    /// A model's silo (`<the library's directory>/silo/libnextrocm-<arch>.so` - beside the executable for a library
    /// out of the binary -, or `NS_SILO_DIR`): kernels tuned for that one model, built from
    /// `kernels/<kind>/<arch>/silo/`, opened on demand. None when there is none or `NS_SILO=0` (the shared kernels
    /// then); an error when the file is there but does not open.
    pub fn silo(&self, arch: &str) -> Result<Option<Silo>, String> {
        if std::env::var("NS_SILO").is_ok_and(|v| v == "0") {
            return Ok(None);
        }
        // a model card's silo: NS_SILO_NAME another silo of this engine's ABI (default: the architecture's own),
        // NS_SILO_REQUIRED=1 no load without it (not the shared kernels)
        let name = self.silo_name(arch);
        let required = std::env::var("NS_SILO_REQUIRED").is_ok_and(|v| v == "1");
        let dir = self.silo_dir();
        let path = dir.join(format!("libnextrocm-{name}.so"));
        if !path.exists() {
            if required {
                return Err(format!("{}: the model's silo is not built (NS_SILO_REQUIRED=1; kernels/build.sh builds the silos)", path.display()));
            }
            return Ok(None);
        }
        let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked. Its undefined symbols (the shared part's
        // ns_fail ...) resolve against the kind's library, opened RTLD_GLOBAL before it.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if h.is_null() {
            return Err(format!("{}: {}", path.display(), dl_error()));
        }
        Ok(Some(Silo {
            handle: h,
            path,
            variant: None,
        }))
    }

    /// The silo an architecture runs on: NS_SILO_NAME, else its own
    pub fn silo_name(&self, arch: &str) -> String {
        std::env::var("NS_SILO_NAME")
            .ok()
            .filter(|n| !n.is_empty())
            .unwrap_or_else(|| arch.to_string())
    }

    /// Where the silos and their variants are: NS_SILO_DIR, else silo/ beside the kind's library (or the executable)
    pub fn silo_dir(&self) -> PathBuf {
        std::env::var_os("NS_SILO_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.dir.join("silo"))
    }

    /// The variants built of silo `name` (`<silo dir>/libnextrocm-<name>@<variant>.so`, each with its manifest
    /// `libnextrocm-<name>@<variant>.yml` beside it, `variant.rs`), by name. A library without its manifest, or with
    /// one that does not read, is an error: a variant never applies on a guess.
    pub fn silo_variants(&self, name: &str) -> Result<Vec<variant::VariantFile>, String> {
        let prefix = format!("libnextrocm-{name}@");
        let mut out = Vec::new();
        let entries = match std::fs::read_dir(self.silo_dir()) {
            Ok(v) => v,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(e.to_string()),
        };
        for entry in entries {
            let e = entry.map_err(|e| e.to_string())?;
            let f = e.file_name().to_string_lossy().into_owned();
            let Some(v) = f.strip_prefix(&prefix).and_then(|r| r.strip_suffix(".so")) else {
                continue;
            };
            if v.is_empty()
                || !v
                    .bytes()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
            {
                return Err(format!("invalid silo variant name {v:?}"));
            }
            let yml = e.path().with_extension("yml");
            let text = std::fs::read_to_string(&yml).map_err(|x| {
                format!(
                    "{}: {x} (a variant's manifest: kernels/build.sh copies it there)",
                    yml.display()
                )
            })?;
            let manifest =
                variant::parse_manifest(&text).map_err(|x| format!("{}: {x}", yml.display()))?;
            out.push(variant::VariantFile {
                name: v.to_string(),
                path: e.path(),
                manifest,
            });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// `base` with a variant's library over it (`Silo::symbol`: the variant's op when it has one, else the base's).
    /// Opened locally: its ops have the default silo's names, and must not stand in for the default's in the
    /// process's global scope (nor the default's for its own: it is linked -Bsymbolic).
    pub fn with_variant(&self, base: &Silo, v: &variant::VariantFile) -> Result<Silo, String> {
        let c = CString::new(v.path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        // SAFETY: dlopen with a valid C string; the handle is checked. Its undefined symbols resolve against the
        // kind's library (global).
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        if h.is_null() {
            return Err(format!("{}: {}", v.path.display(), dl_error()));
        }
        Ok(Silo {
            handle: base.handle,
            path: base.path.clone(),
            variant: Some(VariantLib {
                name: v.name.clone(),
                handle: h,
                path: v.path.clone(),
            }),
        })
    }

    /// Read actual HIP properties without opening a GPU context. Older libraries leave optional fields unknown.
    pub fn device_info(&self, index: usize) -> Result<variant::DeviceInfo, String> {
        let i = c_int::try_from(index).map_err(|_| "GPU index exceeds ABI range")?;
        let mut name = [0 as c_char; 256];
        if unsafe { (self.gpu_name)(i, name.as_mut_ptr(), name.len()) } != 0 {
            return Err(self.error());
        }
        let (mut units, mut mhz) = (0, 0);
        if unsafe { (self.gpu_units)(i, &mut units, &mut mhz) } != 0 {
            return Err(self.error());
        }
        if units < 0 || mhz < 0 {
            return Err("negative GPU properties".into());
        }
        let mut d = variant::DeviceInfo {
            index,
            name: unsafe { CStr::from_ptr(name.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
            units: units as u32,
            mhz: mhz as u32,
            ..Default::default()
        };
        if let Some(f) = self.gpu_props {
            let (mut total, mut wave, mut integrated) = (0u64, 0, 0);
            let mut gfx = [0 as c_char; 256];
            if unsafe {
                f(
                    i,
                    &mut total,
                    &mut wave,
                    &mut integrated,
                    gfx.as_mut_ptr(),
                    gfx.len(),
                )
            } != 0
            {
                return Err(self.error());
            }
            if total == 0 || wave <= 0 || !matches!(integrated, 0 | 1) {
                return Err("invalid GPU properties".into());
            }
            d.vram_mb = Some(total >> 20);
            d.wave_size = Some(wave as u32);
            d.integrated = Some(integrated == 1);
            d.gfx = Some(
                unsafe { CStr::from_ptr(gfx.as_ptr()) }
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        Ok(d)
    }

    /// The reason of the last failed call on this thread.
    pub fn error(&self) -> String {
        // SAFETY: ns_last_error returns a C string owned by the library, valid until the next call on this thread.
        unsafe { CStr::from_ptr((self.last_error)()) }
            .to_string_lossy()
            .into_owned()
    }
}

/// A model's silo library (`Api::silo`): its tuned kernels, bound by the engine by name; with a variant
/// (`Api::with_variant`) the variant's ops over the default silo's
pub struct Silo {
    handle: *mut c_void,
    pub path: PathBuf,
    variant: Option<VariantLib>,
}

/// A silo variant's library (dist/silo/libnextrocm-<arch>@<name>.so)
struct VariantLib {
    name: String,
    handle: *mut c_void,
    path: PathBuf,
}

// SAFETY: a handle to a library that stays loaded for the process's life.
unsafe impl Send for Silo {}
unsafe impl Sync for Silo {}

impl Silo {
    /// A symbol of the silo by name (null when it has none): the variant's when it has one, else the default silo's
    pub fn symbol(&self, name: &str) -> *mut c_void {
        let Ok(n) = CString::new(name) else {
            return std::ptr::null_mut();
        };
        if let Some(v) = &self.variant {
            // SAFETY: a lookup in the variant's own handle (opened locally: its symbols, not the default silo's).
            let p = unsafe { dlsym(v.handle, n.as_ptr()) };
            if !p.is_null() {
                return p;
            }
        }
        // SAFETY: a lookup in the silo's handle; it stays open for the process.
        unsafe { dlsym(self.handle, n.as_ptr()) }
    }

    /// The variant over the default silo (its name, its library); None for the default silo alone
    pub fn variant(&self) -> Option<(&str, &Path)> {
        self.variant
            .as_ref()
            .map(|v| (v.name.as_str(), v.path.as_path()))
    }
}

#[cfg(test)]
mod loader_tests {
    use super::*;
    use std::process::Command;
    fn compile(dir: &Path, name: &str, code: &str) -> PathBuf {
        let source = dir.join(format!("{name}.c"));
        let library = dir.join(format!("{name}.so"));
        std::fs::write(&source, code).unwrap();
        let result = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source)
            .arg("-o")
            .arg(&library)
            .output()
            .expect("CPU C compiler required for loader contract test");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        library
    }
    #[test]
    #[ignore = "subprocess-only loader isolation"]
    fn loader_child() {
        let expected = std::env::var("NEXTROCM_TEST_EXPECT").expect("child contract");
        let result = Api::load("image");
        if expected == "error" {
            assert!(result.is_err());
            return;
        }
        let api = result.unwrap();
        let name = api.path.file_name().unwrap().to_string_lossy();
        assert!(match expected.as_str() {
            "runtime" => name == "libnextrocm-runtime.so",
            "image" => name == "libnextrocm-image.so",
            "override" => name == "new.so",
            _ => false,
        });
    }
    #[test]
    fn optional_properties_and_local_variant_precedence() {
        let dir = std::env::temp_dir().join(format!("nextrocm-loader-test-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let mut code=String::from("#include <stdint.h>\n#include <stddef.h>\n#include <string.h>\nconst char* ns_last_error(void){return \"fixture error\";}\nint ns_gpu_name(int i,char*p,size_t n){(void)i;if(n<9)return 1;memcpy(p,\"AMD test\",9);return 0;}\nint ns_gpu_units(int i,int*u,int*m){(void)i;*u=12;*m=1500;return 0;}\nint test_op(void){return 11;}\nint base_only(void){return 13;}\n");
        // Only properties and the test operations are called; remaining ABI exports are presence stubs.
        for name in include_str!("lib.rs")
            .split("sym!(\"")
            .skip(1)
            .filter_map(|s| s.split('"').next())
        {
            if !["ns_last_error", "ns_gpu_name", "ns_gpu_units"].contains(&name) {
                code.push_str(&format!("void {name}(void){{}}\n"));
            }
        }
        let old_path = compile(&dir, "old", &code);
        let old = Api::load_from(&old_path).unwrap();
        let d = old.device_info(0).unwrap();
        assert_eq!(d.units, 12);
        assert_eq!(d.mhz, 1500);
        assert_eq!(d.gfx, None);
        assert_eq!(d.vram_mb, None);
        assert!(old.device_info(usize::MAX).is_err());
        code.push_str("int ns_gpu_props(int i,uint64_t*t,int*w,int*g,char*p,size_t n){(void)i;if(n<7)return 1;*t=4294967296ULL;*w=64;*g=0;memcpy(p,\"gfx803\",7);return 0;}\n");
        let new_path = compile(&dir, "new", &code);
        let new = Api::load_from(&new_path).unwrap();
        let d = new.device_info(0).unwrap();
        assert_eq!(d.gfx.as_deref(), Some("gfx803"));
        assert_eq!(d.wave_size, Some(64));
        assert_eq!(d.integrated, Some(false));
        assert_eq!(d.vram_mb, Some(4096));
        let variant_path = compile(&dir, "delta", "int test_op(void){return 22;}\n");
        let base = Silo {
            handle: old.handle,
            path: old_path,
            variant: None,
        };
        let delta = new
            .with_variant(
                &base,
                &variant::VariantFile {
                    name: "delta".into(),
                    path: variant_path,
                    manifest: Default::default(),
                },
            )
            .unwrap();
        type Op = unsafe extern "C" fn() -> i32;
        let call = |p: *mut c_void| {
            assert!(!p.is_null());
            unsafe { std::mem::transmute::<*mut c_void, Op>(p)() }
        };
        assert_eq!(call(base.symbol("test_op")), 11);
        assert_eq!(call(delta.symbol("test_op")), 22);
        assert_eq!(call(delta.symbol("base_only")), 13);
        assert!(delta.symbol("absent_operation").is_null());
        assert_eq!(delta.variant().unwrap().0, "delta");
        // Child processes isolate environment/current-directory loader precedence from parallel tests.
        let dist = dir.join("dist");
        std::fs::create_dir(&dist).unwrap();
        std::fs::copy(&new_path, dist.join("libnextrocm-runtime.so")).unwrap();
        let run = |override_path: Option<&Path>, expected: &str| {
            let mut c = Command::new(std::env::current_exe().unwrap());
            c.args(["--exact", "loader_tests::loader_child", "--ignored"])
                .current_dir(&dir)
                .env("NS_LIB_FROM_DISK", "1")
                .env("NEXTROCM_TEST_EXPECT", expected)
                .env_remove("NEXTROCM_LIB");
            if let Some(p) = override_path {
                c.env("NEXTROCM_LIB", p);
            }
            let o = c.output().unwrap();
            assert!(
                o.status.success(),
                "{} {}",
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr)
            );
        };
        run(None, "runtime");
        let kind = dist.join("libnextrocm-image.so");
        std::fs::copy(&new_path, &kind).unwrap();
        run(None, "image");
        std::fs::write(&kind, b"broken").unwrap();
        run(None, "error");
        run(Some(&new_path), "override");
        run(Some(&dir.join("missing.so")), "error");
        std::fs::remove_dir_all(dir).unwrap();
    }
}
