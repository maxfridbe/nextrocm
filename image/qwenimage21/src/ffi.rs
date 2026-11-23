//! Private subset binder; deliberately does not instantiate the shared Nsd
//! binder, which requires operations this runtime does not implement yet.
use std::ffi::{c_int, c_void};

use nextrocm_engine_core::{DevBuf, Error, Gpu, Result};

use crate::weights::PackedMatrix;

type M = *mut c_void;
type P = *const c_void;
type I = i64;
type ExpandedFn = unsafe extern "C" fn(M,P,I,I,P,c_int,I,M,c_int,I,M,I)->c_int;
type PackedFn = unsafe extern "C" fn(M, P, I, I, P, c_int, I, M, c_int, I) -> c_int;

type BatchedAttentionFn = unsafe extern "C" fn(M, P, P, P, I, I, I, I, I, I, M, M, I) -> c_int;

type ImageAttentionFn = unsafe extern "C" fn(M, P, P, P, I, I, I, I, I, I, M) -> c_int;

/// Private image-only control, captured at load; never applies to causal text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKernel {
    Baseline,
    Tiled,
    /// Separate private 1024px candidate; never selected by the existing 0/1 setting.
    Tiled1024,
    /// Private caller-workspace candidate; excluded from the old0/1 parser.
    Batched1024,
}
impl AttentionKernel {
    pub fn from_setting(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("0") => Ok(Self::Baseline),
            Some("1") => Ok(Self::Tiled),
            _ => Err(Error(
                "NS_QI_IMAGE_ATTENTION_TILED must be exactly 0 or 1".into(),
            )),
        }
    }
    pub fn setting(self) -> &'static str {
        match self {
            Self::Baseline => "0",
            Self::Tiled => "1",
            Self::Tiled1024 => "1024",
            Self::Batched1024 => "batched1024",
        }
    }
    pub fn uses_tiled(
        self,
        sq: usize,
        sk: usize,
        h: usize,
        d: usize,
        qs: usize,
        ks: usize,
    ) -> bool {
        self == Self::Tiled && (sq, sk, h, d, qs, ks) == (256, 274, 32, 128, 12288, 12288)
    }
    pub fn uses_tiled_1024(
        self,
        sq: usize,
        sk: usize,
        h: usize,
        d: usize,
        qs: usize,
        ks: usize,
    ) -> bool {
        self == Self::Tiled1024
            && sq == 4096
            && (4097..=4608).contains(&sk)
            && (h, d, qs, ks) == (32, 128, 12288, 12288)
    }
    pub fn uses_batched_1024(
        self,
        sq: usize,
        sk: usize,
        h: usize,
        d: usize,
        qs: usize,
        ks: usize,
    ) -> bool {
        self == Self::Batched1024
            && sq == 4096
            && (4097..=4608).contains(&sk)
            && (h, d, qs, ks) == (32, 128, 12288, 12288)
    }
    /// Scores[H,min(SQ,256),SK] followed by denominators[H,min(SQ,256)].
    /// Prefix execution has no image suffix and must never allocate this scratch.
    pub fn scratch_bytes(self, image_rows: usize, text_rows: usize) -> Result<u64> {
        if self != Self::Batched1024 {
            return Ok(0);
        }
        let keys = image_rows
            .checked_add(text_rows)
            .ok_or_else(|| Error("batched attention keys overflow".into()))?;
        if !self.uses_batched_1024(image_rows, keys, 32, 128, 12288, 12288) {
            return Err(Error("batched1024 image geometry unsupported".into()));
        }
        (keys as u64)
            .checked_add(1)
            .and_then(|n| n.checked_mul(4 * 32 * 256))
            .ok_or_else(|| Error("batched attention scratch overflow".into()))
    }
    /// Fail before loading weights when the shared fallback cannot support the keys.
    pub fn validate_image_geometry(self, image_rows: usize, text_rows: usize) -> Result<()> {
        let keys = image_rows
            .checked_add(text_rows)
            .ok_or_else(|| Error("attention key count overflow".into()))?;
        if keys > 4096
            && !self.uses_tiled_1024(image_rows, keys, 32, 128, 12288, 12288)
            && !self.uses_batched_1024(image_rows, keys, 32, 128, 12288, 12288)
        {
            return Err(Error("image attention exceeds baseline capacity; explicit private 1024 selection required".into()));
        }
        Ok(())
    }
}

/// Captured once at engine/block load. Production integration must pass the
/// original LoadOptions setting; dispatch never rereads process environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedKernel {
    Baseline,
    Register64,
    Register128Image4096,
    ExpandedImage4096,
}
impl PackedKernel {
    pub fn expanded_scratch(self,image_rows:usize)->Result<u64>{
        if self!=Self::ExpandedImage4096{return Ok(0)}
        if image_rows!=4096{return Err(Error("expanded profile requires4096image rows".into()))}
        Ok(100_663_296)
    }
    fn uses_expanded(self,m:usize,k:usize,n:usize,qt:u32)->bool{self==Self::ExpandedImage4096 && Self::Register128Image4096.uses_register128(m,k,n,qt)}
    fn uses_register128(self,m:usize,k:usize,n:usize,qt:u32)->bool {
        self==Self::Register128Image4096 && m==4096 && matches!((qt,k,n),(12,4096,4096)|(12,4096,12288)|(12,12288,4096)|(14,4096,4096))
    }

    pub fn from_setting(value: Option<&str>) -> Result<Self> {
        match value {
            None | Some("0") => Ok(Self::Baseline),
            Some("1") => Ok(Self::Register64),
            _ => Err(Error("NS_QI_PACKED64 must be exactly 0 or 1".into())),
        }
    }
    pub fn setting(self) -> &'static str {
        match self {
            Self::Baseline => "0",
            Self::Register64 => "1",
            Self::Register128Image4096 => "register128-image4096",
            Self::ExpandedImage4096 => "expanded-image4096",
        }
    }
}

#[allow(clippy::type_complexity)]
pub struct Api {
    // kernels/image/qwenimage21/tests/packed_runtime.hip
    packed: PackedFn,
    packed64: Option<PackedFn>,
    packed128: Option<PackedFn>,
    expanded:Option<ExpandedFn>,
    pub(crate) image_attention_tiled: Option<ImageAttentionFn>,
    pub(crate) image_attention_1024: Option<ImageAttentionFn>,
    pub(crate) image_attention_batched: Option<BatchedAttentionFn>,
    // kernels/image/qwenimage21/dit_ops.h is the authoritative private ABI.
    pub ln_mod: unsafe extern "C" fn(M, *const f32, I, I, f32, *const f32, M) -> c_int,
    pub rms_rope:
        unsafe extern "C" fn(M, M, I, I, I, I, *const f32, f32, *const f32, c_int) -> c_int,
    pub text_rms: unsafe extern "C" fn(M, *const f32, I, I, *const f32, f32, *mut f32) -> c_int,
    pub gate_add: unsafe extern "C" fn(M, *mut f32, P, I, I, *const f32) -> c_int,
    // Existing shared operators, resolved individually rather than through Nsd.
    pub linear: unsafe extern "C" fn(M, P, c_int, I, I, P, I, *const f32, M, c_int) -> c_int,
    pub swiglu: unsafe extern "C" fn(M, P, c_int, I, I, M, c_int) -> c_int,
    pub attention_qk: unsafe extern "C" fn(M, P, I, I, P, P, I, I, c_int, I, I, M, c_int) -> c_int,
    pub attention_causal_rows:
        unsafe extern "C" fn(M, P, P, P, c_int, I, I, I, I, I, I, I, M) -> c_int,
}

pub fn api(gpu: &nextrocm_engine_core::Gpu) -> Result<&'static Api> {
    static API: nextrocm_engine_core::bindings::PerGpu<Api> = nextrocm_engine_core::bindings::PerGpu::new();
    API.get_or_init(gpu.index, || {
        let lib = nextrocm_engine_core::bindings::Library::new("qwenimage21", gpu).map_err(|e| e.0)?;
        macro_rules! symbol {
            ($name:literal) => {{
                let p = lib.symbol($name);
                if p.is_null() {
                    return Err(format!(
                        "required private QwenImage21 symbol {} is absent",
                        $name
                    ));
                }
                // SAFETY: signature matches the named private header or the
                // existing speech.hip export. The runtime stays loaded.
                #[allow(clippy::missing_transmute_annotations)]
                unsafe {
                    std::mem::transmute::<*mut c_void, _>(p)
                }
            }};
        }
        let ep=lib.symbol("nrqi_packed_linear_expanded");
        // SAFETY: exact reviewed private expanded ABI plus caller workspace.
        let expanded=if ep.is_null(){None}else{Some(unsafe{std::mem::transmute::<*mut c_void,ExpandedFn>(ep)})};
        let optional128=lib.symbol("nrqi_packed_linear128");
        // SAFETY: independently validated private packed128 export has the same PackedFn ABI.
        let packed128=if optional128.is_null(){None}else{Some(unsafe{std::mem::transmute::<*mut c_void,PackedFn>(optional128)})};
        let optional64 = lib.symbol("nrqi_packed_linear64");
        let packed64 = if optional64.is_null() {
            None
        } else {
            // SAFETY: optional symbol has exactly the baseline packed ABI.
            Some(unsafe { std::mem::transmute::<*mut c_void, PackedFn>(optional64) })
        };
        let attention_symbol = lib.symbol("nrqi_image_attention_tiled");
        let image_attention_tiled = if attention_symbol.is_null() {
            None
        } else {
            // SAFETY: exact frozen private attention_runtime.hip ABI.
            Some(unsafe { std::mem::transmute::<*mut c_void, ImageAttentionFn>(attention_symbol) })
        };
        let attention_1024_symbol = lib.symbol("nrqi_image_attention_1024");
        let image_attention_1024 = if attention_1024_symbol.is_null() {
            None
        } else {
            // SAFETY: same private half Q/K/V attention ABI; explicit selection only.
            Some(unsafe {
                std::mem::transmute::<*mut c_void, ImageAttentionFn>(attention_1024_symbol)
            })
        };
        let batched_symbol = lib.symbol("nrqi_image_attention_batched");
        let image_attention_batched = if batched_symbol.is_null() {
            None
        } else {
            // SAFETY: reviewed private Q/K/V attention ABI plus caller scratch/byte capacity.
            Some(unsafe { std::mem::transmute::<*mut c_void, BatchedAttentionFn>(batched_symbol) })
        };
        Ok(Api {
            image_attention_batched,
            image_attention_tiled,
            image_attention_1024,
            packed: symbol!("nrqi_packed_linear"),
            packed64,
            packed128,
            expanded,
            ln_mod: symbol!("nrqi_ln_mod"),
            rms_rope: symbol!("nrqi_rms_rope"),
            text_rms: symbol!("nrqi_text_rms_f32"),
            gate_add: symbol!("nrqi_gate_add_f32_f16"),
            linear: symbol!("nsd_linear"),
            swiglu: symbol!("nsd_swiglu"),
            attention_qk: symbol!("nsd_attention_qk"),
            attention_causal_rows: symbol!("nsd_attention_causal_rows"),
        })
    })
    .as_ref()
    .map_err(|e| Error(e.clone()))
}

pub fn check(code: c_int, operation: &str) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        let detail = nextrocm_engine_core::api()
            .map(|a| a.error())
            .unwrap_or_default();
        Err(Error(format!("{operation}: {detail} (code {code})")))
    }
}

#[derive(Clone, Copy)]
#[repr(i32)]
pub enum OutputType {
    Float = 0,
    Half = 1,
}

impl OutputType {
    fn bytes(self) -> usize {
        match self {
            Self::Float => 4,
            Self::Half => 2,
        }
    }
}

impl Api {
    pub fn require_attention(&self, kernel: AttentionKernel) -> Result<()> {
        if kernel == AttentionKernel::Batched1024 && self.image_attention_batched.is_none() {
            return Err(Error("private batched1024 attention requested but nrqi_image_attention_batched is absent".into()));
        }
        if kernel == AttentionKernel::Tiled && self.image_attention_tiled.is_none() {
            return Err(Error(
                "NS_QI_IMAGE_ATTENTION_TILED=1 requested but private ABI missing".into(),
            ));
        }
        if kernel == AttentionKernel::Tiled1024 && self.image_attention_1024.is_none() {
            return Err(Error(
                "private 1024 attention requested but nrqi_image_attention_1024 is absent".into(),
            ));
        }
        Ok(())
    }

    fn selected(&self, kernel: PackedKernel) -> Result<PackedFn> {
        match kernel {
            PackedKernel::ExpandedImage4096 => {
                self.expanded.ok_or_else(||Error("explicit expanded ABI absent".into()))?;
                self.selected(PackedKernel::Register128Image4096)
            },
            PackedKernel::Register128Image4096 => {
                self.packed64.ok_or_else(||Error("register128 fallback requires packed64 ABI".into()))?;
                self.packed128.ok_or_else(||Error("explicit register128 requested but ABI absent".into()))
            },
            PackedKernel::Baseline => Ok(self.packed),
            PackedKernel::Register64 => self.packed64.ok_or_else(|| {
                Error("NS_QI_PACKED64=1 requested but nrqi_packed_linear64 is absent".into())
            }),
        }
    }
    pub fn require_kernel(&self, kernel: PackedKernel) -> Result<()> {
        self.selected(kernel).map(|_| ())
    }
    /// Apply each differently quantized part into one fused row-major output.
    /// Part offsets are columns, not rows; `stride` is the full fused width.
    pub fn packed_matrix(
        &self,
        gpu: &Gpu,
        matrix: &PackedMatrix,
        x: &DevBuf,
        m: usize,
        out: &DevBuf,
        dtype: OutputType,
    ) -> Result<()> {
        self.packed_matrix_using(PackedKernel::Baseline, gpu, matrix, x, m, out, dtype)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn packed_matrix_using(
        &self,
        kernel: PackedKernel,
        gpu: &Gpu,
        matrix: &PackedMatrix,
        x: &DevBuf,
        m: usize,
        out: &DevBuf,
        dtype: OutputType,
    ) -> Result<()> {
        self.packed_matrix_using_workspace(kernel,gpu,matrix,x,m,out,dtype,None)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn packed_matrix_using_workspace(&self,kernel:PackedKernel,gpu:&Gpu,matrix:&PackedMatrix,x:&DevBuf,m:usize,out:&DevBuf,dtype:OutputType,scratch:Option<&DevBuf>)->Result<()> {
        let function = self.selected(kernel)?;
        if m == 0
            || m > 65536
            || matrix.columns == 0
            || matrix.columns > 16384
            || matrix.columns % 256 != 0
            || matrix.rows == 0
            || matrix.rows > 65536
        {
            return Err(Error("invalid packed application geometry".into()));
        }
        let x_bytes = m.checked_mul(matrix.columns).and_then(|n| n.checked_mul(2));
        let y_bytes = m
            .checked_mul(matrix.rows)
            .and_then(|n| n.checked_mul(dtype.bytes()));
        if x_bytes.is_none_or(|n| n > x.len) || y_bytes.is_none_or(|n| n > out.len) {
            return Err(Error("packed application buffer is too short".into()));
        }
        let mut offset = 0usize;
        // Validate all part ranges before any launch; no partial execution for
        // a malformed public descriptor.
        for part in &matrix.parts {
            if part.columns != matrix.columns || part.rows == 0 || part.rows > 16384 {
                return Err(Error("invalid packed matrix part".into()));
            }
            let block_bytes = match part.quant_type {
                12 => 144,
                14 => 210,
                _ => return Err(Error("unsupported packed part type".into())),
            };
            let stored = (part.columns / 256)
                .checked_mul(block_bytes)
                .and_then(|n| n.checked_mul(part.rows));
            if stored != Some(part.weights.len) {
                return Err(Error("packed matrix part storage differs".into()));
            }
            offset = offset
                .checked_add(part.rows)
                .ok_or_else(|| Error("part row overflow".into()))?;
        }
        if offset != matrix.rows {
            return Err(Error("packed matrix parts do not cover the output".into()));
        }
        // Validate workspace for every selected part before any launch.
        for part in &matrix.parts {if kernel.uses_expanded(m,matrix.columns,part.rows,part.quant_type){
            let needed=matrix.columns.checked_mul(part.rows).and_then(|v|v.checked_mul(2)).ok_or_else(||Error("expanded workspace overflow".into()))?;
            let buf=scratch.ok_or_else(||Error("expanded workspace missing".into()))?;
            if needed>100_663_296||buf.len<needed||buf.len>100_663_296{return Err(Error("expanded workspace capacity".into()))}
        }}
        offset = 0;
        for part in &matrix.parts {
            let fallback_kind=if kernel==PackedKernel::ExpandedImage4096{PackedKernel::Register128Image4096}else{kernel};
            let function=if fallback_kind==PackedKernel::Register128Image4096 && !fallback_kind.uses_register128(m,matrix.columns,part.rows,part.quant_type){self.selected(PackedKernel::Register64)?}else{function};
            // SAFETY: matrix loader validated packed storage, checked input /
            // output lengths above cover every launch including its row stride.
            // Pointer arithmetic is within out's allocation. HIP checks aliases.
            let target = out
                .ptr()
                .cast::<u8>()
                .wrapping_add(offset * dtype.bytes())
                .cast();
            if kernel.uses_expanded(m,matrix.columns,part.rows,part.quant_type){
                let buf=scratch.ok_or_else(||Error("expanded workspace missing".into()))?;
                let expanded=self.expanded.ok_or_else(||Error("expanded ABI missing".into()))?;
                // SAFETY: validated matrix/output views and owned disjoint reusable workspace; launcher rechecks spans.
                check(unsafe{expanded(gpu.raw(),x.ptr(),m as I,matrix.columns as I,part.weights.ptr(),part.quant_type as c_int,part.rows as I,target,dtype as c_int,matrix.rows as I,buf.ptr(),buf.len as I)},"expanded packed matrix")?;
            }else{check(
                unsafe {
                    function(
                        gpu.raw(),
                        x.ptr(),
                        m as I,
                        matrix.columns as I,
                        part.weights.ptr(),
                        part.quant_type as c_int,
                        part.rows as I,
                        target,
                        dtype as c_int,
                        matrix.rows as I,
                    )
                },
                "QwenImage21 packed matrix",
            )?;}
            offset += part.rows;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expanded_is_bounded_explicit(){let k=PackedKernel::ExpandedImage4096;assert_eq!(k.expanded_scratch(4096).unwrap(),100_663_296);assert!(k.expanded_scratch(256).is_err());assert_eq!(PackedKernel::Register128Image4096.expanded_scratch(4096).unwrap(),0);assert!(k.uses_expanded(4096,4096,12288,12));assert!(!k.uses_expanded(18,4096,4096,12));assert!(PackedKernel::from_setting(Some("expanded-image4096")).is_err());}
    #[test]
    fn register128_is_explicit_and_shape_limited(){
        let k=PackedKernel::Register128Image4096;
        for (qt,cols,rows) in [(12,4096,4096),(12,4096,12288),(12,12288,4096),(14,4096,4096)] {assert!(k.uses_register128(4096,cols,rows,qt));assert!(!k.uses_register128(18,cols,rows,qt));}
        assert!(!k.uses_register128(4096,4096,12288,14));
        assert!(!k.uses_register128(4096,4096,1024,12));
        assert!(!PackedKernel::Register64.uses_register128(4096,4096,4096,12));
        assert!(PackedKernel::from_setting(Some("register128-image4096")).is_err());
    }
    #[test]
    fn batched_workspace_is_explicit_and_geometry_bounded() {
        let k = AttentionKernel::Batched1024;
        assert_eq!(k.scratch_bytes(4096, 18).unwrap(), 134_840_320);
        assert_eq!(k.scratch_bytes(4096, 512).unwrap(), 151_027_712);
        assert!(k.scratch_bytes(4096, 0).is_err());
        assert!(k.scratch_bytes(4096, 513).is_err());
        assert!(k.scratch_bytes(usize::MAX, 1).is_err());
        assert!(k.validate_image_geometry(4096, 18).is_ok());
        assert!(!k.uses_batched_1024(4096, 4114, 32, 128, 4096, 12288));
        assert!(AttentionKernel::from_setting(Some("batched1024")).is_err());
        assert_eq!(
            AttentionKernel::Tiled1024.scratch_bytes(4096, 18).unwrap(),
            0
        );
    }
    #[test]
    fn private_1024_requires_explicit_geometry_before_loading() {
        let t = AttentionKernel::Tiled1024;
        for prefix in [1, 18, 512] {
            assert!(t.validate_image_geometry(4096, prefix).is_ok());
            assert!(t.uses_tiled_1024(4096, 4096 + prefix, 32, 128, 12288, 12288));
            assert!(AttentionKernel::Baseline
                .validate_image_geometry(4096, prefix)
                .is_err());
            assert!(AttentionKernel::Tiled
                .validate_image_geometry(4096, prefix)
                .is_err());
        }
        assert!(t.validate_image_geometry(4096, 513).is_err());
        assert!(t.validate_image_geometry(usize::MAX, 1).is_err());
        assert!(!t.uses_tiled_1024(4096, 4114, 32, 128, 4096, 12288));
        assert!(!t.uses_tiled_1024(256, 274, 32, 128, 12288, 12288));
        assert!(!t.uses_tiled(256, 274, 32, 128, 12288, 12288));
        assert!(AttentionKernel::from_setting(Some("1024")).is_err());
    }
    #[test]
    fn image_attention_optin_is_strict_and_exact_geometry_only() {
        assert_eq!(
            AttentionKernel::from_setting(None).unwrap(),
            AttentionKernel::Baseline
        );
        assert_eq!(
            AttentionKernel::from_setting(Some("0")).unwrap(),
            AttentionKernel::Baseline
        );
        let t = AttentionKernel::from_setting(Some("1")).unwrap();
        assert!(t.uses_tiled(256, 274, 32, 128, 12288, 12288));
        assert!(!AttentionKernel::Baseline.uses_tiled(256, 274, 32, 128, 12288, 12288));
        for g in [
            (1024, 1042, 32, 128, 12288, 12288),
            (256, 275, 32, 128, 12288, 12288),
            (255, 274, 32, 128, 12288, 12288),
            (256, 274, 31, 128, 12288, 12288),
            (256, 274, 32, 64, 12288, 12288),
            (256, 274, 32, 128, 4096, 12288),
            (256, 274, 32, 128, 12288, 4096),
        ] {
            assert!(!t.uses_tiled(g.0, g.1, g.2, g.3, g.4, g.5));
        }
        for bad in ["", "true", "2", " 1", "1 "] {
            assert!(AttentionKernel::from_setting(Some(bad)).is_err());
        }
    }
    #[test]
    fn optin_setting_is_strict_and_defaults_off() {
        assert_eq!(
            PackedKernel::from_setting(None).unwrap(),
            PackedKernel::Baseline
        );
        assert_eq!(
            PackedKernel::from_setting(Some("0")).unwrap(),
            PackedKernel::Baseline
        );
        assert_eq!(
            PackedKernel::from_setting(Some("1")).unwrap(),
            PackedKernel::Register64
        );
        for bad in ["", "true", "2", " 1", "1 "] {
            assert!(PackedKernel::from_setting(Some(bad)).is_err());
        }
    }
}
