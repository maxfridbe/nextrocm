//! F32 auxiliary linears retain the source precision contract. Q4_K remains
//! packed; the three BF16 auxiliary matrices are expanded once to F32, matching
//! the reference's `single` loader (135,266,304 resident bytes in total).
use crate::{
    ffi,
    plan::{tensor_contract, DitPlan, WeightArithmetic},
};
use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};
use nextrocm_gguf::Gguf;
use std::ffi::{c_int, c_void};
use std::sync::{Arc};

type AuxFn = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    i64,
    i64,
    *const c_void,
    c_int,
    i64,
    *mut f32,
    i64,
) -> c_int;
fn packed_api(gpu: &Gpu) -> Result<AuxFn> {
    static API: nextrocm_engine_core::bindings::PerGpu<AuxFn> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("qwenimage21", gpu).map_err(|e| e.0)?;
        let ptr = lib.symbol("nrqi_aux_f32_linear");
        if ptr.is_null() {
            return Err("required nrqi_aux_f32_linear is absent".into());
        }
        // SAFETY: signature matches private tests/aux_runtime.hip; runtime stays loaded.
        Ok(unsafe { std::mem::transmute::<*mut c_void, AuxFn>(ptr) })
    })
    .as_ref()
    .copied()
    .map_err(|e| Error(e.clone()))
}

pub struct Auxiliary {
    weights: DevBuf,
    packed: bool,
    pub rows: usize,
    pub columns: usize,
    pub bytes: usize,
}

impl Auxiliary {
    pub fn load(file: &Gguf, gpu: &Arc<Gpu>, name: &str, allowance: usize) -> Result<Self> {
        DitPlan::validate(file)?;
        let spec = tensor_contract()
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| Error("unknown auxiliary matrix".into()))?;
        if spec.arithmetic != WeightArithmetic::Float || spec.shape.len() != 2 {
            return Err(Error("not an F32 auxiliary matrix".into()));
        }
        let t = file
            .tensor(name)
            .ok_or_else(|| Error(format!("missing {name}")))?;
        let rows = usize::try_from(spec.shape[0]).map_err(|_| Error("row overflow".into()))?;
        let columns =
            usize::try_from(spec.shape[1]).map_err(|_| Error("column overflow".into()))?;
        let packed = t.ty.code() == 12;
        if !packed && t.ty.code() != 30 {
            return Err(Error("unsupported auxiliary weight type".into()));
        }
        let bytes = if packed {
            usize::try_from(t.bytes).map_err(|_| Error("weight size overflow".into()))?
        } else {
            rows.checked_mul(columns)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| Error("auxiliary size overflow".into()))?
        };
        if bytes > allowance || bytes > 64 * 1024 * 1024 {
            return Err(Error("auxiliary weight allowance exceeded".into()));
        }
        let raw = file.read(t).map_err(|e| Error(e.0))?;
        let weights = if packed {
            let weights = DevBuf::new(gpu, bytes)?;
            weights.write(0, &raw)?;
            weights
        } else {
            let values: Vec<_> = raw
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect();
            if values.len() != rows * columns || values.iter().any(|x| !x.is_finite()) {
                return Err(Error("invalid BF16 auxiliary matrix".into()));
            }
            DevBuf::from_f32(gpu, &values)?
        };
        Ok(Self {
            weights,
            packed,
            rows,
            columns,
            bytes,
        })
    }

    pub fn apply(&self, gpu: &Gpu, x: &DevBuf, m: usize, out: &DevBuf) -> Result<()> {
        if m == 0 || m > 65536 {
            return Err(Error("invalid auxiliary row count".into()));
        }
        let nx = m
            .checked_mul(self.columns)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error("input span overflow".into()))?;
        let ny = m
            .checked_mul(self.rows)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| Error("output span overflow".into()))?;
        if nx > x.len || ny > out.len {
            return Err(Error("auxiliary buffer too short".into()));
        }
        let xo = x.ptr() as usize;
        let yo = out.ptr() as usize;
        let wo = self.weights.ptr() as usize;
        let xe = xo
            .checked_add(nx)
            .ok_or_else(|| Error("input pointer wrap".into()))?;
        let ye = yo
            .checked_add(ny)
            .ok_or_else(|| Error("output pointer wrap".into()))?;
        let we = wo
            .checked_add(self.weights.len)
            .ok_or_else(|| Error("weight pointer wrap".into()))?;
        if (xo < ye && yo < xe) || (wo < ye && yo < we) {
            return Err(Error("auxiliary output overlaps input/weights".into()));
        }
        // SAFETY: dimensions/storage were validated at load, input/output spans
        // above are bounded and disjoint, and both variants preserve F32 math.
        if self.packed {
            ffi::check(
                unsafe {
                    packed_api(gpu)?(
                        gpu.raw(),
                        x.fp(),
                        m as i64,
                        self.columns as i64,
                        self.weights.ptr(),
                        12,
                        self.rows as i64,
                        out.fp(),
                        self.rows as i64,
                    )
                },
                "packed F32 auxiliary",
            )
        } else {
            ffi::check(
                unsafe {
                    (ffi::api(gpu)?.linear)(
                        gpu.raw(),
                        x.ptr(),
                        0,
                        m as i64,
                        self.columns as i64,
                        self.weights.ptr(),
                        self.rows as i64,
                        std::ptr::null(),
                        out.ptr(),
                        0,
                    )
                },
                "F32 auxiliary",
            )
        }
    }
}
