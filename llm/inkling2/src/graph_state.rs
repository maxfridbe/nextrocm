//! Lifecycle only. Never restore a position without its complete ring/history bytes.
use super::*;
impl TextGraph {
    pub(crate) fn reset_state(&mut self) -> Result<()> {
        self.cursor.abort();
        self.gpu.sync().map_err(|e| e.0)?;
        self.weights.state.kv.fill(0).map_err(|e| e.0)?;
        self.weights.state.history.fill(0).map_err(|e| e.0)?;
        self.gpu.sync().map_err(|e| e.0)?;
        self.cursor.restore_position(0)
    }
    pub(crate) fn state_bytes(&self) -> usize {
        self.weights.state.kv.len + self.weights.state.history.len
    }
    pub(crate) fn snapshot(&self) -> Result<(usize, Vec<u8>)> {
        let pos = self.cursor.settled_position()?;
        self.gpu.sync().map_err(|e| e.0)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(self.state_bytes())
            .map_err(|e| e.to_string())?;
        bytes.resize(self.state_bytes(), 0);
        let n = self.weights.state.kv.len;
        self.weights
            .state
            .kv
            .read(0, &mut bytes[..n])
            .map_err(|e| e.0)?;
        self.weights
            .state
            .history
            .read(0, &mut bytes[n..])
            .map_err(|e| e.0)?;
        Ok((pos, bytes))
    }
    pub(crate) fn restore_state(&mut self, position: usize, bytes: &[u8]) -> Result<()> {
        if position > self.plan.context() || bytes.len() != self.state_bytes() {
            return Err("invalid complete Inkling state snapshot".into());
        }
        self.cursor.abort();
        let n = self.weights.state.kv.len;
        self.weights
            .state
            .kv
            .write(0, &bytes[..n])
            .map_err(|e| e.0)?;
        self.weights
            .state
            .history
            .write(0, &bytes[n..])
            .map_err(|e| e.0)?;
        self.gpu.sync().map_err(|e| e.0)?;
        self.cursor.restore_position(position)
    }
}

#[cfg(feature = "diagnostic-state")]
impl TextGraph {
    pub fn diagnostic_reset(&mut self) -> Result<()> {
        self.reset_state()
    }
    pub fn diagnostic_snapshot(&self) -> Result<(usize, Vec<u8>)> {
        self.snapshot()
    }
    pub fn diagnostic_restore(&mut self, pos: usize, bytes: &[u8]) -> Result<()> {
        self.restore_state(pos, bytes)
    }
    pub fn diagnostic_state_bytes(&self) -> usize {
        self.state_bytes()
    }
}
