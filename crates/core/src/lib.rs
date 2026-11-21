//! Single-threaded, owning HIP handles. No implicit CPU fallback.
use nextrocm_sys::Api;
use std::{
    ffi::{c_char, c_void, CStr},
    rc::Rc,
};

pub type Result<T> = std::result::Result<T, String>;
fn check(api: &Api, rc: i32) -> Result<()> {
    if rc == 0 {
        Ok(())
    } else {
        Err(api.error())
    }
}

/// One HIP device and its streams. Rc deliberately prevents concurrent ticket-ring access.
pub struct Gpu {
    api: Api,
    raw: nextrocm_sys::Gpu,
}
impl Gpu {
    pub fn open(index: i32) -> Result<Rc<Self>> {
        let api = Api::load("runtime")?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: valid out pointer; the returned handle is owned until Drop.
        check(&api, unsafe { (api.gpu_open)(index, &mut raw) })?;
        Ok(Rc::new(Self { api, raw }))
    }
    pub fn devices() -> Result<Vec<String>> {
        let api = Api::load("runtime")?;
        // SAFETY: no arguments; library stays loaded.
        let count = unsafe { (api.gpu_count)() };
        if count < 0 {
            return Err(api.error());
        }
        (0..count)
            .map(|i| {
                let mut name = [0 as c_char; 256];
                // SAFETY: writable array and matching capacity.
                check(&api, unsafe {
                    (api.gpu_name)(i, name.as_mut_ptr(), name.len())
                })?;
                // SAFETY: ABI guarantees a terminating NUL.
                Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
                    .to_string_lossy()
                    .into_owned())
            })
            .collect()
    }
    pub fn memory(&self) -> Result<(u64, i64)> {
        let (mut total, mut free) = (0, 0);
        // SAFETY: live handle and valid output pointers.
        check(&self.api, unsafe {
            (self.api.gpu_memory)(self.raw, &mut total, &mut free)
        })?;
        Ok((total, free))
    }
    pub fn sync(&self) -> Result<()> {
        // SAFETY: live device handle.
        check(&self.api, unsafe { (self.api.sync)(self.raw) })
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        // SAFETY: all buffers retain this device; handle closed exactly once.
        unsafe { (self.api.gpu_close)(self.raw) };
    }
}

/// Owning allocation. Host transfers synchronize before borrowed slices can expire.
pub struct Buffer {
    gpu: Rc<Gpu>,
    ptr: *mut c_void,
    len: usize,
}
impl Buffer {
    pub fn new(gpu: &Rc<Gpu>, len: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: live device and valid output pointer.
        check(&gpu.api, unsafe {
            (gpu.api.alloc)(gpu.raw, len.max(1), &mut ptr)
        })?;
        Ok(Self {
            gpu: Rc::clone(gpu),
            ptr,
            len,
        })
    }
    fn bounds(&self, len: usize) -> Result<()> {
        if len <= self.len {
            Ok(())
        } else {
            Err("transfer exceeds allocation".into())
        }
    }
    pub fn write(&mut self, bytes: &[u8]) -> Result<()> {
        self.bounds(bytes.len())?;
        // SAFETY: both regions have bytes.len() bytes and sync retains the host borrow.
        check(&self.gpu.api, unsafe {
            (self.gpu.api.copy_to)(self.gpu.raw, self.ptr, bytes.as_ptr().cast(), bytes.len())
        })?;
        self.gpu.sync()
    }
    pub fn read(&self, bytes: &mut [u8]) -> Result<()> {
        self.bounds(bytes.len())?;
        // SAFETY: both regions have bytes.len() bytes and sync retains the host borrow.
        check(&self.gpu.api, unsafe {
            (self.gpu.api.copy_from)(
                self.gpu.raw,
                bytes.as_mut_ptr().cast(),
                self.ptr,
                bytes.len(),
            )
        })?;
        self.gpu.sync()
    }
    pub fn fill(&mut self, value: u8) -> Result<()> {
        // SAFETY: live allocation of self.len bytes.
        check(&self.gpu.api, unsafe {
            (self.gpu.api.fill)(self.gpu.raw, self.ptr, value, self.len)
        })?;
        self.gpu.sync()
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: unique owning allocation, device retained by Rc; free synchronizes.
        unsafe { (self.gpu.api.free)(self.gpu.raw, self.ptr) };
    }
}

/// Compare a HIP kernel with exact host arithmetic, including a partial final workgroup.
pub fn selftest(index: i32) -> Result<()> {
    let gpu = Gpu::open(index)?;
    type Add = unsafe extern "C" fn(
        nextrocm_sys::Gpu,
        *mut c_void,
        *const c_void,
        *const c_void,
        usize,
    ) -> i32;
    let symbol = gpu.api.symbol("nr_test_add");
    if symbol.is_null() {
        return Err("missing nr_test_add kernel".into());
    }
    // SAFETY: nr_test_add has this exact signature in kernels/selftest.hip.
    let add: Add = unsafe { std::mem::transmute(symbol) };
    for n in [0usize, 1, 255, 256, 257, 4099] {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
        let b: Vec<f32> = (0..n).map(|i| -(i as f32) * 0.5).collect();
        let encode = |v: &[f32]| v.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>();
        let mut da = Buffer::new(&gpu, n * 4)?;
        let mut db = Buffer::new(&gpu, n * 4)?;
        let mut out = Buffer::new(&gpu, n * 4)?;
        da.write(&encode(&a))?;
        db.write(&encode(&b))?;
        out.fill(0)?;
        // SAFETY: all three allocations belong to gpu and hold n f32 values.
        check(&gpu.api, unsafe {
            add(gpu.raw, out.ptr, da.ptr, db.ptr, n)
        })?;
        let mut result = vec![0u8; n * 4];
        out.read(&mut result)?;
        if result != encode(&a.iter().zip(&b).map(|(a, b)| a + b).collect::<Vec<_>>()) {
            return Err(format!("HIP arithmetic mismatch at length {n}"));
        }
        if out.write(&vec![0u8; n * 4 + 1]).is_ok() {
            return Err("bounds check failed".into());
        }
    }
    Ok(())
}

pub mod admission;
