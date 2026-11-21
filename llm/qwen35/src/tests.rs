use super::*;
use nextrocm_gguf::{writer::Writer, GType, Tensor, Value};
fn geometry() -> Geometry {
    Geometry {
        arch: "qwen35moe",
        layers: 4,
        width: 8,
        ff: 4,
        heads: 2,
        kv_heads: 1,
        head: 4,
        rotary: 2,
        k_heads: 1,
        v_heads: 2,
        state: 4,
        conv: 4,
        interval: 4,
        vocab: 16,
        context: 32,
        experts: 4,
        used: 2,
        shared: 4,
        eps: 1e-6,
        theta: 10000.,
    }
}
fn tmp() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "nq35-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}
#[test]
fn geometry_rejects_invalid_divisors_and_shapes() {
    let g = geometry();
    g.validate().unwrap();
    assert_eq!(
        g.cache_bytes(0).unwrap(),
        3 * (2 * 4 * 4 * 4 + 3 * (2 * 1 * 4 + 2 * 4) * 4)
    );
    assert_eq!(
        g.cache_bytes(3).unwrap() - g.cache_bytes(0).unwrap(),
        3 * 1 * 4 * 4
    );
    for bad in 0..7 {
        let mut b = g.clone();
        match bad {
            0 => b.k_heads = 0,
            1 => b.kv_heads = 0,
            2 => b.interval = 0,
            3 => b.rotary = 3,
            4 => b.v_heads = 3,
            5 => b.used = 5,
            _ => b.eps = f32::NAN,
        };
        if bad == 4 {
            b.k_heads = 2;
        }
        assert!(b.validate().is_err());
    }
    let t = Tensor {
        name: "w".into(),
        shape: vec![2, 256],
        ty: GType::Q4K,
        shard: 0,
        offset: 0,
        bytes: 288,
    };
    geometry::validate_tensor(&t, &[2, 256]).unwrap();
    assert!(geometry::validate_tensor(&t, &[256, 2]).is_err());
    let mut b = t.clone();
    b.bytes = 287;
    assert!(geometry::validate_tensor(&b, &[2, 256]).is_err());
    let mut b = t;
    b.shape = vec![256, 2];
    assert!(geometry::validate_tensor(&b, &[256, 2]).is_err());
}
#[test]
fn metadata_and_tensor_directory_follow_dense_and_moe() {
    for (moe, tied) in [(false, false), (false, true), (true, false), (true, true)] {
        let mut g = geometry();
        if !moe {
            g.arch = "qwen35";
            g.experts = 0;
            g.used = 0;
            g.shared = 0;
        }
        let mut meta: Vec<(String, Value)> =
            vec![("general.architecture".into(), Value::Str(g.arch.into()))];
        for (k, v) in [
            ("block_count", g.layers),
            ("embedding_length", g.width),
            ("expert_feed_forward_length", g.ff),
            ("feed_forward_length", g.ff),
            ("attention.head_count", g.heads),
            ("attention.head_count_kv", g.kv_heads),
            ("attention.key_length", g.head),
            ("rope.dimension_count", g.rotary),
            ("ssm.group_count", g.k_heads),
            ("ssm.time_step_rank", g.v_heads),
            ("ssm.state_size", g.state),
            ("ssm.inner_size", g.inner()),
            ("ssm.conv_kernel", g.conv),
            ("full_attention_interval", g.interval),
            ("context_length", g.context),
            ("expert_count", g.experts),
            ("expert_used_count", g.used),
            ("expert_shared_feed_forward_length", g.shared),
        ] {
            meta.push((format!("{}.{k}", g.arch), Value::U(v as u64)));
        }
        for (k, v) in [
            ("attention.layer_norm_rms_epsilon", g.eps),
            ("rope.freq_base", g.theta),
        ] {
            meta.push((format!("{}.{k}", g.arch), Value::F(v as f64)));
        }
        let meta: Vec<_> = meta.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        let specs: Vec<_> = g
            .specs()
            .into_iter()
            .filter(|(n, _)| !tied || n != "output.weight")
            .map(|(n, s)| (n, GType::F32, s))
            .collect();
        let path = tmp();
        let mut writer = Writer::create(&path, &meta, &specs).unwrap();
        for (_, _, shape) in &specs {
            writer
                .data(&vec![0; shape.iter().product::<u64>() as usize * 4])
                .unwrap();
        }
        writer.finish().unwrap();
        let mut file = Gguf::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        let got = Geometry::read(&file).unwrap();
        assert_eq!(got.arch, g.arch);
        assert_eq!(got.experts, g.experts);
        let (plan, bytes) = weight_plan(&file, &got).unwrap();
        let resident: usize = file
            .tensors
            .iter()
            .filter(|t| t.shape.len() != 3)
            .map(|t| t.bytes as usize)
            .sum();
        assert_eq!(bytes, resident + Scratch::bytes(&got));
        assert_eq!(plan.contains_key("output.weight"), !tied);
        assert!(plan["token_embd.weight"].resident_bytes().unwrap() > 0);
        if moe {
            assert_eq!(
                plan["blk.0.ffn_gate_exps.weight"].resident_bytes().unwrap(),
                0
            );
            assert!(
                plan["blk.0.ffn_gate_shexp.weight"]
                    .resident_bytes()
                    .unwrap()
                    > 0
            );
            assert!(plan["blk.0.ffn_gate_inp.weight"].resident_bytes().unwrap() > 0);
        }

        file.metadata
            .insert(format!("{}.nextn_predict_layers", g.arch), Value::U(100));
        assert!(Geometry::read(&file).is_err());
        file.metadata
            .remove(&format!("{}.nextn_predict_layers", g.arch));
        file.metadata.insert(
            format!("{}.rope.scaling.type", g.arch),
            Value::Str("linear".into()),
        );
        assert!(Geometry::read(&file).is_err());
    }
}
#[test]
fn checkpoints_reject_foreign_model_oversize_and_truncation() {
    let g = geometry();
    let owner = Arc::new(());
    let len = g.cache_bytes(3).unwrap();
    let mut c = Ck::new(owner.clone(), 42, 3, len).unwrap();
    for (i, v) in c.data.iter_mut().enumerate() {
        *v = (i % 251) as u8;
    }
    let mut bytes = Vec::new();
    c.write_to(&mut bytes).unwrap();
    let d = Ck::read(&mut &bytes[..], &g, owner.clone(), 42).unwrap();
    assert_eq!(c.data, d.data);
    assert_eq!(d.pos, 3);
    assert!(Ck::read(&mut &bytes[..], &g, owner.clone(), 43).is_err());
    assert!(Ck::read(&mut &bytes[..bytes.len() - 1], &g, owner.clone(), 42).is_err());
    bytes[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Ck::read(&mut &bytes[..], &g, owner.clone(), 42).is_err());
    bytes[24..32].copy_from_slice(&(len as u64).to_le_bytes());
    bytes[16..24].copy_from_slice(&33u64.to_le_bytes());
    assert!(Ck::read(&mut &bytes[..], &g, owner, 42).is_err());
}
#[test]
fn decoder_pending_handoff_is_once() {
    let mut d = Dec {
        owner: Arc::new(()),
        logits: None,
        next: Some(7),
    };
    assert_eq!(d.pending(), Some(7));
    assert_eq!(d.pending(), None);
    assert_eq!(d.drafts(), (0, 0));
}

#[test]
fn q5k_and_q8_rows_are_admitted_with_exact_packed_lengths() {
    for (ty, cols, bytes) in [(GType::Q5K, 256, 352), (GType::Q8_0, 32, 68)] {
        let mut t = Tensor {
            name: "matrix.weight".into(),
            shape: vec![2, cols],
            ty,
            shard: 0,
            offset: 0,
            bytes,
        };
        geometry::validate_tensor(&t, &[2, cols]).unwrap();
        t.bytes -= 1;
        assert!(geometry::validate_tensor(&t, &[2, cols]).is_err());
    }
    let t = Tensor {
        name: "matrix.weight".into(),
        shape: vec![32, 2],
        ty: GType::Q8_0,
        shard: 0,
        offset: 0,
        bytes: 68,
    };
    assert!(geometry::validate_tensor(&t, &[32, 2]).is_err());
}
