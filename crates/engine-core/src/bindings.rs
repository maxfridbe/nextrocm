//! Immutable per-device engine symbol tables. Libraries remain loaded for the process.
use crate::{api, silo_for, Api, Gpu, Result};
use std::{ffi::c_void, sync::Mutex};

pub struct Library {
    shared: &'static Api,
    silo: Option<&'static nextrocm_sys::Silo>,
}
impl Library {
    pub fn new(arch: &str, gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            shared: api()?,
            silo: silo_for(arch, gpu.index)?,
        })
    }
    /// Engine-local variant, default architecture library, then shared runtime.
    pub fn symbol(&self, name: &str) -> *mut c_void {
        let ptr = self.silo.map_or(std::ptr::null_mut(), |s| s.symbol(name));
        if ptr.is_null() {
            self.shared.symbol(name)
        } else {
            ptr
        }
    }
}

/// Each FFI module has its own cache; physical GPU index selects its immutable table.
/// Failed bindings are cached too. Function tables and libraries have process lifetime.
/// Initializers must not recursively initialize this same cache.
pub struct PerGpu<T: 'static> {
    entries: Mutex<Vec<(usize, &'static std::result::Result<T, String>)>>,
}
impl<T: Send + Sync + 'static> PerGpu<T> {
    pub const fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }
    pub fn get_or_init(
        &self,
        gpu: usize,
        init: impl FnOnce() -> std::result::Result<T, String>,
    ) -> &'static std::result::Result<T, String> {
        let mut entries = self.entries.lock().unwrap();
        if let Some((_, value)) = entries.iter().find(|(index, _)| *index == gpu) {
            return value;
        }
        let value = Box::leak(Box::new(init()));
        entries.push((gpu, value));
        value
    }
}
impl<T: Send + Sync + 'static> Default for PerGpu<T> {
    fn default() -> Self {
        Self::new()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn device_bindings_and_errors_are_independent_and_immutable() {
        let tables = PerGpu::new();
        assert_eq!(tables.get_or_init(0, || Ok(11)), &Ok(11));
        assert_eq!(tables.get_or_init(1, || Ok(22)), &Ok(22));
        assert_eq!(tables.get_or_init(0, || panic!("must not rebind")), &Ok(11));
        assert_eq!(
            tables.get_or_init(2, || Err("missing".into())),
            &Err("missing".into())
        );
        assert_eq!(
            tables.get_or_init(2, || panic!("must not retry")),
            &Err("missing".into())
        );
    }
}
