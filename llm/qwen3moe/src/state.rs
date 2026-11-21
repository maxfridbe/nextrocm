use super::*;
use std::io::{self, Read, Write};
pub(crate) struct Cache {
    pub a: DevBuf,
    pub b: DevBuf,
}
pub(crate) struct Sess {
    pub owner: Arc<()>,
    pub pos: usize,
    pub max_ctx: usize,
    pub cache: Vec<Cache>,
    pub last: Option<usize>,
    pub poisoned: bool,
    _reservation: store::Reservation,
}
impl Sess {
    pub fn new(gpu: &Arc<Gpu>, g: &Geometry, max_ctx: usize, owner: Arc<()>) -> Result<Self> {
        let snapshot = g.cache_bytes(0)?;
        let bytes = g
            .cache_bytes(max_ctx)?
            .checked_add(snapshot)
            .ok_or_else(|| error("session size overflow"))?;
        let reservation = store::Reservation::device(bytes)?;
        ensure_memory(gpu, bytes as u64)?;
        let mut cache = Vec::new();
        for l in 0..g.layers {
            let (a, b) = g.layer_bytes(l, max_ctx)?;
            let ca = DevBuf::new(gpu, a)?;
            let cb = DevBuf::new(gpu, b)?;
            ca.fill(0)?;
            cb.fill(0)?;
            cache.push(Cache { a: ca, b: cb });
        }
        gpu.sync()?;
        Ok(Self {
            owner,
            pos: 0,
            max_ctx,
            cache,
            last: None,
            poisoned: false,
            _reservation: reservation,
        })
    }
}
impl SessionState for Sess {
    fn pos(&self) -> usize {
        self.pos
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
pub(crate) struct Dec {
    pub owner: Arc<()>,
    pub logits: Option<Vec<f32>>,
    pub next: Option<u32>,
}
impl DecoderState for Dec {
    fn drafts(&self) -> (u64, u64) {
        (0, 0)
    }
    fn pending(&mut self) -> Option<u32> {
        self.next.take()
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
pub(crate) struct Ck {
    pub owner: Arc<()>,
    pub model: u64,
    pub pos: usize,
    pub data: Vec<u8>,
    _reservation: store::Reservation,
}
const MAGIC: &[u8; 8] = b"NQ3MKV01";
impl Ck {
    pub fn new(owner: Arc<()>, model: u64, pos: usize, len: usize) -> Result<Self> {
        let reservation = store::Reservation::host(len)?;
        let mut data = Vec::new();
        data.try_reserve_exact(len)
            .map_err(|e| error(format!("checkpoint allocation: {e}")))?;
        data.resize(len, 0);
        Ok(Self {
            owner,
            model,
            pos,
            data,
            _reservation: reservation,
        })
    }
    pub fn read(r: &mut dyn Read, g: &Geometry, owner: Arc<()>, model: u64) -> io::Result<Self> {
        let invalid = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid Qwen3Moe checkpoint model/geometry/length",
            )
        };
        let mut h = [0u8; 32];
        r.read_exact(&mut h)?;
        let word = |i: usize| {
            let mut b = [0; 8];
            b.copy_from_slice(&h[i..i + 8]);
            u64::from_le_bytes(b)
        };
        let pos = usize::try_from(word(16)).map_err(|_| invalid())?;
        let len = usize::try_from(word(24)).map_err(|_| invalid())?;
        if &h[..8] != MAGIC
            || word(8) != model
            || pos > g.context
            || g.cache_bytes(pos).map_err(|_| invalid())? != len
        {
            return Err(invalid());
        }
        let mut ck =
            Self::new(owner, model, pos, len).map_err(|e| io::Error::other(e.to_string()))?;
        r.read_exact(&mut ck.data)?;
        Ok(ck)
    }
}
impl CheckpointState for Ck {
    fn pos(&self) -> usize {
        self.pos
    }
    fn bytes(&self) -> usize {
        self.data.len()
    }
    fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&self.model.to_le_bytes())?;
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        w.write_all(&(self.data.len() as u64).to_le_bytes())?;
        w.write_all(&self.data)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
