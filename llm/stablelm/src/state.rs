use super::*;
use std::io::{self, Read, Write};
pub(super) struct Sess {
    pub owner: Arc<()>,
    pub pos: usize,
    pub max_ctx: usize,
    pub kv: Vec<(DevBuf, DevBuf)>,
    pub last: Option<(usize, usize)>,
    pub poisoned: bool,
}
impl Sess {
    pub fn new(gpu: &Arc<Gpu>, g: &Geometry, max_ctx: usize, owner: Arc<()>) -> Result<Self> {
        let bytes = max_ctx * g.width * 2;
        let kv = (0..g.layers)
            .map(|_| Ok((DevBuf::new(gpu, bytes)?, DevBuf::new(gpu, bytes)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            owner,
            pos: 0,
            max_ctx,
            kv,
            last: None,
            poisoned: false,
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
pub(super) struct Dec {
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
pub(super) struct Ck {
    pub owner: Arc<()>,
    pub pos: usize,
    pub data: Vec<u8>,
}
const MAGIC: &[u8; 8] = b"NSLMKV01";
impl CheckpointState for Ck {
    fn pos(&self) -> usize {
        self.pos
    }
    fn bytes(&self) -> usize {
        self.data.len()
    }
    fn write_to(&self, w: &mut dyn Write) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&(self.pos as u64).to_le_bytes())?;
        w.write_all(&(self.data.len() as u64).to_le_bytes())?;
        w.write_all(&self.data)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}
impl Ck {
    pub fn read(r: &mut dyn Read, g: &Geometry, owner: Arc<()>) -> io::Result<Self> {
        fn invalid() -> io::Error {
            io::Error::new(io::ErrorKind::InvalidData, "invalid StableLM checkpoint")
        }
        let mut header = [0; 24];
        r.read_exact(&mut header)?;
        if &header[..8] != MAGIC {
            return Err(invalid());
        }
        let mut p = [0; 8];
        p.copy_from_slice(&header[8..16]);
        let pos = usize::try_from(u64::from_le_bytes(p)).map_err(|_| invalid())?;
        p.copy_from_slice(&header[16..24]);
        let len = usize::try_from(u64::from_le_bytes(p)).map_err(|_| invalid())?;
        if pos > g.context || g.cache_bytes(pos).map_err(|_| invalid())? != len {
            return Err(invalid());
        }
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|_| invalid())?;
        data.resize(len, 0);
        r.read_exact(&mut data)?;
        Ok(Self { owner, pos, data })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn geometry() -> Geometry {
        Geometry {
            layers: 1,
            width: 2,
            ff: 4,
            heads: 1,
            head: 2,
            rotary: 2,
            vocab: 3,
            context: 4,
            theta: 10000.0,
            eps: 1e-5,
        }
    }
    #[test]
    fn checkpoint_roundtrip_and_reject_bad_lengths() {
        let owner = Arc::new(());
        let ck = Ck {
            owner: owner.clone(),
            pos: 2,
            data: vec![42; 16],
        };
        let mut bytes = Vec::new();
        ck.write_to(&mut bytes).unwrap();
        let c = Ck::read(&mut bytes.as_slice(), &geometry(), owner.clone()).unwrap();
        assert_eq!(c.pos, 2);
        assert_eq!(c.data, ck.data);
        bytes[16] = 17;
        assert!(Ck::read(&mut bytes.as_slice(), &geometry(), owner.clone()).is_err());
        bytes[16] = 16;
        bytes[8] = 255;
        assert!(Ck::read(&mut bytes.as_slice(), &geometry(), owner).is_err());
    }
    #[test]
    fn pending_token_handoff_is_once() {
        let mut d = Dec {
            owner: Arc::new(()),
            logits: None,
            next: Some(9),
        };
        assert_eq!(d.pending(), Some(9));
        assert_eq!(d.pending(), None);
        assert_eq!(d.drafts(), (0, 0));
    }
}
