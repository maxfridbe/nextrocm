use super::*;
impl Engine for Qwen35 {
    fn save_profile(&self, path: &std::path::Path) -> std::io::Result<usize> {
        self.route_profile.save(path)
    }
    fn report(&self, _tokens: usize) -> Vec<String> {
        let mut lines = self.profile.lines();
        lines.extend(self.expert_cache.report());
        lines.extend(self.store.report());
        lines.extend(self.route_profile.report());
        lines
    }
    fn arch(&self) -> &'static str {
        self.geom.arch
    }
    fn load_seconds(&self) -> f64 {
        self.seconds
    }
    fn load_bytes(&self) -> u64 {
        self.bytes
    }
    fn has_draft(&self) -> bool {
        false
    }
    fn prefill_chunk(&self) -> usize {
        self.chunk
    }
    fn max_verify(&self) -> usize {
        CHUNK
    }
    fn cache_fingerprint(&self) -> String {
        format!("qwen35-hip-f16kv-f32delta-v1-{:016x}", self.model)
    }
    fn session(&self, max_ctx: usize) -> Result<Session> {
        if max_ctx == 0 || max_ctx > self.geom.context {
            return Err(error("session context exceeds trained context or is zero"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        ensure_memory(&self.gpu, self.geom.cache_bytes(max_ctx)? as u64)?;
        Ok(Session::new(Sess::new(
            &self.gpu,
            &self.geom,
            max_ctx,
            Arc::clone(&self.owner),
        )?))
    }
    fn reset_session(&self, s: &mut Session) -> Result<()> {
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        let s = self.sess_mut(s)?;
        self.gpu.sync()?;
        s.poisoned = true;
        for c in &s.cache {
            c.a.fill(0)?;
            c.b.fill(0)?;
        }
        self.gpu.sync()?;
        s.pos = 0;
        s.last = None;
        s.poisoned = false;
        Ok(())
    }
    fn copy_session(&self, dst: &mut Session, src: &Session) -> Result<()> {
        let src = self.sess(src)?;
        let dst = self.sess_mut(dst)?;
        if src.poisoned || src.pos > dst.max_ctx {
            return Err(error("cannot copy poisoned or oversized session"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        dst.poisoned = true;
        dst.last = None;
        for (l, (from, to)) in src.cache.iter().zip(&dst.cache).enumerate() {
            let (a, b) = self.geom.layer_bytes(l, src.pos)?;
            to.a.copy_within(0, &from.a, 0, a)?;
            to.b.copy_within(0, &from.b, 0, b)?;
        }
        self.gpu.sync()?;
        dst.pos = src.pos;
        dst.poisoned = false;
        Ok(())
    }
    fn feed(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        Ok(self.feed_until(s, tokens, &|_| false, tap)?.1)
    }
    fn feed_until(
        &self,
        s: &mut Session,
        tokens: &[u32],
        stop: &(dyn Fn(usize) -> bool + Sync),
        tap: Tap,
    ) -> Result<(usize, Vec<f32>)> {
        self.validate_pass(s, tokens)?;
        feed_chunks(tokens, self.chunk, stop, |chunk, n_out| {
            self.forward_rows(s, chunk, n_out, tap)
        })
    }
    fn forward(&self, s: &mut Session, tokens: &[u32], tap: Tap) -> Result<Vec<f32>> {
        self.forward_rows(s, tokens, 1, tap)?
            .pop()
            .ok_or_else(|| error("missing final logits"))
    }
    fn forward_rows(
        &self,
        s: &mut Session,
        tokens: &[u32],
        n_out: usize,
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        self.validate_pass(s, tokens)?;
        if tokens.len() > self.chunk || n_out > tokens.len() || n_out > self.max_verify() {
            return Err(error("invalid Qwen35 pass/row count"));
        }
        if tokens.len() > 1 {
            return self.batch_pass(s, tokens, n_out, tap);
        }
        let a = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        let s = self.sess_mut(s)?;
        let start = s.pos;
        s.last = None;
        s.poisoned = true;
        s.snapshot()?;
        let mut out = Vec::with_capacity(n_out);
        for (i, t) in tokens.iter().enumerate() {
            match self.token(s, *t, &a, tap, i >= tokens.len() - n_out) {
                Ok(row) => {
                    if i >= tokens.len() - n_out {
                        out.push(row);
                    }
                }
                Err(e) => {
                    let _ = self.gpu.sync();
                    s.pos = start;
                    s.poisoned = true;
                    return Err(e);
                }
            }
        }
        s.poisoned = false;
        s.last = Some(start);
        Ok(out)
    }
    fn rollback(&self, s: &mut Session, keep: usize) -> Result<()> {
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        let s = self.sess_mut(s)?;
        if s.poisoned {
            return Err(error("cannot roll back a failed pass"));
        }
        let start = s
            .last
            .ok_or_else(|| error("no pass available for rollback"))?;
        if keep > 1 {
            return Err(error("rollback exceeds last one-token pass"));
        }
        s.poisoned = true;
        if keep == 0 {
            s.rewind()?;
        }
        self.gpu.sync()?;
        s.pos = start + keep;
        s.last = None;
        s.poisoned = false;
        Ok(())
    }
    fn forward_batch(
        &self,
        sessions: &mut [&mut Session],
        tokens: &[u32],
        tap: Tap,
    ) -> Result<Vec<Vec<f32>>> {
        if sessions.len() != tokens.len() {
            return Err(error("batch session/token count mismatch"));
        }
        for (s, t) in sessions.iter().zip(tokens) {
            self.validate_pass(s, &[*t])?;
        }
        sessions
            .iter_mut()
            .zip(tokens)
            .map(|(s, t)| self.forward(s, &[*t], tap))
            .collect()
    }
    fn decoder(&self, logits: Vec<f32>, _: bool) -> Decoder {
        Decoder::new(Dec {
            owner: Arc::clone(&self.owner),
            logits: Some(logits),
            next: None,
        })
    }
    fn decoder_after(&self, next: u32, _: bool) -> Decoder {
        Decoder::new(Dec {
            owner: Arc::clone(&self.owner),
            logits: None,
            next: Some(next),
        })
    }
    fn step(
        &self,
        s: &mut Session,
        d: &mut Decoder,
        smp: &mut dyn Sampler,
        tap: Tap,
    ) -> Result<Vec<u32>> {
        if self.sess(s)?.poisoned {
            return Err(error("cannot decode from failed session"));
        }
        let d = d
            .get_mut::<Dec>()
            .filter(|d| Arc::ptr_eq(&d.owner, &self.owner))
            .ok_or_else(|| error("decoder belongs to another model"))?;
        if let Some(t) = d.next {
            let logits = self.forward(s, &[t], tap)?;
            d.next = None;
            d.logits = Some(logits);
        }
        let row = d
            .logits
            .as_ref()
            .ok_or_else(|| error("decoder pending token was already handed off"))?;
        if row.len() != self.geom.vocab || row.iter().any(|x| !x.is_finite()) {
            return Err(error("invalid decoder logits"));
        }
        let t = smp.sample(row);
        if t as usize >= self.geom.vocab {
            return Err(error("sampler returned out-of-vocabulary token"));
        }
        d.logits = None;
        d.next = Some(t);
        Ok(vec![t])
    }
    fn save(&self, s: &Session) -> Result<Checkpoint> {
        let s = self.sess(s)?;
        if s.poisoned {
            return Err(error("cannot save failed session"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        self.gpu.sync()?;
        let mut ck = Ck::new(
            Arc::clone(&self.owner),
            self.model,
            s.pos,
            self.geom.cache_bytes(s.pos)?,
        )?;
        let mut at = 0;
        for (l, c) in s.cache.iter().enumerate() {
            let (a, b) = self.geom.layer_bytes(l, s.pos)?;
            for (buf, n) in [(&c.a, a), (&c.b, b)] {
                buf.read(0, &mut ck.data[at..at + n])?;
                at += n;
            }
        }
        Ok(Checkpoint::new(ck))
    }
    fn restore(&self, s: &mut Session, ck: &Checkpoint) -> Result<()> {
        let ck = ck
            .get::<Ck>()
            .filter(|c| Arc::ptr_eq(&c.owner, &self.owner))
            .ok_or_else(|| error("checkpoint belongs to another model instance"))?;
        let s = self.sess_mut(s)?;
        if ck.pos > s.max_ctx || ck.data.len() != self.geom.cache_bytes(ck.pos)? {
            return Err(error("checkpoint exceeds session or has invalid length"));
        }
        let _lock = self
            .scratch
            .lock()
            .map_err(|_| error("Qwen35 scratch lock poisoned"))?;
        s.poisoned = true;
        s.last = None;
        let mut at = 0;
        for (l, c) in s.cache.iter().enumerate() {
            let (a, b) = self.geom.layer_bytes(l, ck.pos)?;
            for (buf, n) in [(&c.a, a), (&c.b, b)] {
                buf.write(0, &ck.data[at..at + n])?;
                at += n;
            }
        }
        self.gpu.sync()?;
        s.pos = ck.pos;
        s.poisoned = false;
        Ok(())
    }
    fn read_checkpoint(&self, r: &mut dyn std::io::Read) -> std::io::Result<Checkpoint> {
        Ok(Checkpoint::new(Ck::read(
            r,
            &self.geom,
            Arc::clone(&self.owner),
            self.model,
        )?))
    }
    fn gpu_info(&self) -> Vec<GpuInfo> {
        let (total, free) = self.gpu.memory().unwrap_or((0, None));
        vec![GpuInfo {
            index: self.gpu.index,
            name: self.gpu.name.clone(),
            pci: self.gpu.pci.clone(),
            total,
            free,
            layers: (0, self.geom.layers as u64),
            expert_slots: usize::from(self.geom.experts > 0) as _,
            host_slots: 0,
        }]
    }
}

/// The reference prefill queues intermediate chunks without an output head.
/// Cancellation before the final chunk returns empty logits. Unlike the original
/// nextcuda loop, retain nextrocm's stop(0) check before any session mutation.
fn feed_chunks(
    tokens: &[u32],
    chunk_size: usize,
    stop: &(dyn Fn(usize) -> bool + Sync),
    mut pass: impl FnMut(&[u32], usize) -> Result<Vec<Vec<f32>>>,
) -> Result<(usize, Vec<f32>)> {
    let mut read = 0;
    let mut logits = Vec::new();
    for chunk in tokens.chunks(chunk_size) {
        if stop(read) {
            break;
        }
        let last = read + chunk.len() == tokens.len();
        logits = pass(chunk, usize::from(last))?.pop().unwrap_or_default();
        read += chunk.len();
    }
    Ok((read, logits))
}

#[cfg(test)]
mod prefill_tests {
    use super::*;

    #[test]
    fn prefill_computes_only_final_logits_including_short_tail() {
        let mut passes = Vec::new();
        let (read, logits) = feed_chunks(&[1, 2, 3, 4, 5], 2, &|_| false, |chunk, n| {
            passes.push((chunk.to_vec(), n));
            Ok(if n == 0 { vec![] } else { vec![vec![42.0]] })
        })
        .expect("valid passes");
        assert_eq!(passes, vec![(vec![1, 2], 0), (vec![3, 4], 0), (vec![5], 1)]);
        assert_eq!(read, 5);
        assert_eq!(logits, vec![42.0]);
    }

    #[test]
    fn stop_zero_never_invokes_pass_and_partial_has_no_logits() {
        let (read, logits) = feed_chunks(&[1, 2, 3], 1, &|_| true, |_, _| {
            panic!("initial cancellation must not mutate a session")
        })
        .expect("stopped");
        assert_eq!(read, 0);
        assert!(logits.is_empty());
        let mut calls = 0;
        let (read, logits) = feed_chunks(&[1, 2, 3], 1, &|n| n == 2, |_, n| {
            calls += 1;
            assert_eq!(n, 0);
            Ok(vec![])
        })
        .expect("partially stopped");
        assert_eq!((read, calls), (2, 2));
        assert!(logits.is_empty());
    }

    #[test]
    fn pass_failure_propagates_without_running_later_chunks() {
        let mut calls = 0;
        let result = feed_chunks(&[1, 2, 3], 1, &|_| false, |_, _| {
            calls += 1;
            Err(error("pass failed"))
        });
        assert!(result.is_err());
        assert_eq!(calls, 1);
    }
}
