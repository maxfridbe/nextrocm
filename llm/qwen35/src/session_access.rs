use super::*;
impl Qwen35 {
    pub(super) fn sess<'a>(&self, s: &'a Session) -> Result<&'a Sess> {
        s.get::<Sess>()
            .filter(|s| Arc::ptr_eq(&s.owner, &self.owner))
            .ok_or_else(|| error("session belongs to another model instance"))
    }
    pub(super) fn sess_mut<'a>(&self, s: &'a mut Session) -> Result<&'a mut Sess> {
        s.get_mut::<Sess>()
            .filter(|s| Arc::ptr_eq(&s.owner, &self.owner))
            .ok_or_else(|| error("session belongs to another model instance"))
    }
    pub(super) fn validate_pass(&self, s: &Session, tokens: &[u32]) -> Result<()> {
        let s = self.sess(s)?;
        if s.poisoned {
            return Err(error("session had a failed GPU pass; reset or restore it"));
        }
        if tokens.is_empty() {
            return Err(error("empty Qwen35 pass"));
        }
        if s.pos
            .checked_add(tokens.len())
            .is_none_or(|n| n > s.max_ctx)
        {
            return Err(error("Qwen35 context exhausted"));
        }
        if tokens.iter().any(|t| *t as usize >= self.geom.vocab) {
            return Err(error("token outside vocabulary"));
        }
        Ok(())
    }
}
