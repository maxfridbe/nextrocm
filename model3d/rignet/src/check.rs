//! Stage by stage against the reference (reference/rignet/ref.py: RigNet's own networks and numpy code on the CPU, with
//! pure-torch stand-ins for torch_geometric). The geometry comes from here - `inputs` writes it - so each network and
//! each numpy step is compared on the same input, its own given the reference's:
//!
//! ```text
//!   inputs   the mesh (--mesh)         -> pos, faces, tpl, geo, geodesic, vox, vox_meta (.npy, written for ref.py)
//!   joints   the inputs                -> disp.npy, attn.npy (JointNet); then joints.npy from the reference's outputs
//!   root     joints.npy                -> root_logits.npy (RootNet)
//!   bones    joints.npy                -> bone_logits.npy (BoneNet); then parent.npy from the reference's logits
//!   skin     joints + parent           -> geo_dist.npy (volumetric geodesics, trimesh's rays there), skin_logits.npy
//!                                         (SkinNet on skin_input.npy), skin_full.npy (the filtered weights)
//! ```

use std::path::Path;

use nextrocm_model3d::{Error, Result};

use crate::npy::{compare, read, write};
use crate::{geom, skel, Prepared, RigNet};

/// Predeclared full-stage relative-RMS gate; see reference/tolerance.json.
/// Original tests establish component scales, not a historical full-model threshold.
pub const FULL_STAGE_RELATIVE_TOLERANCE: f64 = 1e-4;

fn line(what: &str, got: &[f32], want: &[f32], log: &mut dyn FnMut(String)) -> f64 {
    if got.len() != want.len() {
        log(format!(
            "{what:<14} {} values here, {} in the reference",
            got.len(),
            want.len()
        ));
        return f64::INFINITY;
    }
    let (rel, mx) = compare(got, want);
    log(format!(
        "{what:<14} rel rms {rel:.2e}  max abs {mx:.3e}  ({} values)",
        want.len()
    ));
    rel
}

fn pairs_of(v: &[f32]) -> Vec<(u32, u32)> {
    v.chunks_exact(2)
        .map(|c| (c[0] as u32, c[1] as u32))
        .collect()
}

/// The prepared mesh the reference read (the dumps of `inputs`)
fn prepared(dir: &Path) -> Result<Prepared> {
    let (ps, pos) = read(&dir.join("pos.npy"))?;
    if ps.len() != 2 || ps[1] != 3 || ps[0] > crate::limits::MAX_VERTICES {
        return Err(Error("invalid reference positions".into()));
    }
    let n = ps[0];
    let exact = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
        let (s, v) = read(&dir.join(format!("{name}.npy")))?;
        if s != shape {
            return Err(Error(format!(
                "{name}: reference shape {s:?}, expected {shape:?}"
            )));
        }
        Ok(v)
    };
    let (fs, faces) = read(&dir.join("faces.npy"))?;
    if fs.len() != 2 || fs[1] != 3 || fs[0] > crate::limits::MAX_FACES {
        return Err(Error("invalid reference faces".into()));
    }
    let edges = |name: &str| -> Result<Vec<(u32, u32)>> {
        let (s, v) = read(&dir.join(format!("{name}.npy")))?;
        if s.len() != 2
            || s[1] != 2
            || v.iter()
                .any(|x| !x.is_finite() || *x < 0. || x.fract() != 0. || *x >= n as f32)
        {
            return Err(Error("invalid reference edges".into()));
        }
        let e = pairs_of(&v);
        crate::limits::graph(n, &e)?;
        Ok(e)
    };
    if faces
        .iter()
        .any(|x| !x.is_finite() || *x < 0. || x.fract() != 0. || *x >= n as f32)
    {
        return Err(Error("invalid reference face indices".into()));
    }
    let positions: Vec<[f32; 3]> = pos.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    let triangles: Vec<[u32; 3]> = faces
        .chunks_exact(3)
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    crate::limits::mesh(&positions, &triangles, false)?;
    let g = exact("geodesic", &[n, n])?;
    let (vs, vox) = read(&dir.join("vox.npy"))?;
    if vs.len() != 3 || vs[0] == 0 || vs[0] > 128 || vs[1] != vs[0] || vs[2] != vs[0] {
        return Err(Error("invalid reference voxel dimensions".into()));
    }
    let meta = exact("vox_meta", &[4])?;
    if meta.iter().any(|v| !v.is_finite()) || meta[3] <= 0. {
        return Err(Error("invalid voxel transform".into()));
    }
    Ok(Prepared {
        pos: positions,
        faces: triangles,
        pivot: [0.; 3],
        scale: 1.,
        tpl: edges("tpl")?,
        geo: edges("geo")?,
        geodesic: geom::Geodesic {
            s: n,
            dist: g,
            nn: (0..n as u32).collect(),
        },
        vox: geom::Vox {
            dims: vs[0],
            translate: [meta[0], meta[1], meta[2]],
            scale: meta[3],
            data: vox.iter().map(|x| *x > 0.5).collect(),
        },
    })
}

fn joints_of(v: &[f32]) -> Vec<[f64; 3]> {
    v.chunks_exact(3)
        .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
        .collect()
}

/// Every stage of `stages` against the dumps in `dir` (`mesh`: the input mesh, for `inputs`); the worst relative error
pub fn run(
    m: &RigNet,
    dir: &Path,
    stages: &[&str],
    mesh: Option<&Path>,
    log: &mut dyn FnMut(String),
) -> Result<f64> {
    if stages.is_empty()
        || stages
            .iter()
            .any(|s| !["inputs", "joints", "root", "bones", "skin"].contains(s))
    {
        return Err(Error("unknown or empty validation stages".into()));
    }
    let mut worst = 0f64;
    let mut note = |r: f64| worst = worst.max(if r.is_nan() { f64::INFINITY } else { r });
    if stages.contains(&"inputs") {
        let path =
            mesh.ok_or_else(|| Error("the inputs stage reads the mesh: --mesh FILE".into()))?;
        let bytes = std::fs::read(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let mesh = nextrocm_model3d::read_mesh(&bytes).map_err(Error)?;
        let t0 = std::time::Instant::now();
        let p = m.prepare(&mesh, 4000, 0)?;
        let n = p.pos.len();
        write_inputs(&p, dir)?;
        let d = p.vox.dims;
        log(format!("inputs         {n} vertices, {} faces, {} mesh edges, {} geodesic edges, {} of {} voxels filled ({:.1} s)", p.faces.len(),
                    p.tpl.len(), p.geo.len(), p.vox.data.iter().filter(|x| **x).count(), d * d * d, t0.elapsed().as_secs_f64()));
    }
    let need = |s: &str| stages.contains(&s);
    if !(need("joints") || need("root") || need("bones") || need("skin")) {
        return Ok(worst);
    }
    let p = prepared(dir)?;
    let mi = m.mesh_in(&p)?;
    if need("joints") {
        let (disp, attn) = m.joint.run(&m.ops, &mi)?;
        note(line(
            "joint disp",
            &disp,
            &read(&dir.join("disp.npy"))?.1,
            log,
        ));
        note(line(
            "joint attn",
            &attn,
            &read(&dir.join("attn.npy"))?.1,
            log,
        ));
        let (_, rd) = read(&dir.join("disp.npy"))?;
        let (_, ra) = read(&dir.join("attn.npy"))?;
        let (_, bw) = read(&dir.join("bandwidth.npy"))?;
        if bw.len() != 1
            || !bw[0].is_finite()
            || bw[0] <= 0.
            || rd.len() != p.pos.len() * 3
            || ra.len() != p.pos.len()
        {
            return Err(Error("invalid reference JointNet outputs".into()));
        }
        note(line("bandwidth", &[m.joint.bandwidth], &bw, log));
        log(format!(
            "bandwidth      {:.6} here, {:.6} there",
            m.joint.bandwidth, bw[0]
        ));
        let j = skel::joints_from(&p.pos, &rd, &ra, &p.vox, bw[0] as f64, 1e-5);
        let (_, want) = read(&dir.join("joints.npy"))?;
        let got: Vec<f32> = j.iter().flatten().map(|x| *x as f32).collect();
        log(format!(
            "joints         {} here, {} there (from the reference's outputs)",
            j.len(),
            want.len() / 3
        ));
        note(line("joints", &got, &want, log));
    }
    let (_, jv) = read(&dir.join("joints.npy"))?;
    if jv.len() % 3 != 0 {
        return Err(Error("invalid joint rows".into()));
    }
    let joints = joints_of(&jv);
    crate::limits::joints(&joints)?;
    let jf: Vec<[f32; 3]> = joints.iter().map(|j| j.map(|x| x as f32)).collect();
    if need("root") {
        let shape = m.root.shape(&m.ops, &mi)?;
        let got = m.root.run(&m.ops, &shape, &jf)?;
        note(line(
            "root logits",
            &got,
            &read(&dir.join("root_logits.npy"))?.1,
            log,
        ));
    }
    if need("bones") {
        let (pairs, rows) = skel::pairs(&joints, &p.vox);
        let shape = m.bone.shape(&m.ops, &mi)?;
        let got = m.bone.run(&m.ops, &shape, &jf, &rows)?;
        note(line(
            "pair inputs",
            &rows,
            &read(&dir.join("pair_input.npy"))?.1,
            log,
        ));
        note(line(
            "bone logits",
            &got,
            &read(&dir.join("bone_logits.npy"))?.1,
            log,
        ));
        let (_, rl) = read(&dir.join("bone_logits.npy"))?;
        let (_, rr) = read(&dir.join("root_id.npy"))?;
        if rl.len() != pairs.len()
            || rr.len() != 1
            || !rr[0].is_finite()
            || rr[0] < 0.
            || rr[0].fract() != 0.
            || rr[0] >= joints.len() as f32
        {
            return Err(Error("invalid reference bone/root output".into()));
        }
        let prob: Vec<f32> = rl.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let cost = skel::costs(&joints, &pairs, &prob, &p.vox);
        let (parent, _) = skel::prim_symmetric(&cost, rr[0] as usize, &joints);
        let (_, want) = read(&dir.join("parent.npy"))?;
        let same = parent
            .iter()
            .zip(&want)
            .filter(|(a, b)| **a == **b as i64)
            .count();
        log(format!("parents        {same} of {} the same", want.len()));
        if same != want.len() || parent.len() != want.len() {
            note(f64::INFINITY);
        }
    }
    if need("skin") {
        let (_, pv) = read(&dir.join("parent.npy"))?;
        let parent: Vec<i64> = pv.iter().map(|x| *x as i64).collect();
        if parent.len() != joints.len()
            || pv.iter().any(|v| !v.is_finite() || v.fract() != 0.)
            || parent.iter().filter(|&&v| v == -1).count() != 1
        {
            return Err(Error("invalid reference parent array".into()));
        }
        let root = parent
            .iter()
            .position(|x| *x == -1)
            .ok_or_else(|| Error("missing reference root".into()))?;
        for start in 0..parent.len() {
            let mut i = start;
            let mut steps = 0;
            while i != root {
                let p = parent[i];
                if p < 0 || p as usize >= parent.len() || steps >= parent.len() {
                    return Err(Error("invalid or cyclic reference skeleton".into()));
                }
                i = p as usize;
                steps += 1;
            }
        }
        let (order, parents) = skel::tree_order(&parent, root);
        let tj: Vec<[f64; 3]> = order.iter().map(|&o| joints[o]).collect();
        let (bones, leaf, _) = skel::bones(&tj, &parents);
        let (_, rb) = read(&dir.join("bones.npy"))?;
        note(line(
            "bones",
            &bones.iter().flatten().copied().collect::<Vec<_>>(),
            &rb,
            log,
        ));
        let t0 = std::time::Instant::now();
        let geo = geom::volumetric_geodesic(&p.pos, &p.faces, &bones, &p.geodesic);
        if let Ok((_, rv)) = read(&dir.join("vis_raw.npy")) {
            if rv.len() != p.pos.len() * bones.len() {
                return Err(Error("invalid reference visibility".into()));
            }
            let (org, _) = geom::pts2line(&p.pos, &bones);
            let ends: Vec<[f32; 3]> = (0..bones.len())
                .flat_map(|_| p.pos.iter().copied())
                .collect();
            let vis = geom::visible(&p.pos, &p.faces, &org, &ends);
            let n = p.pos.len();
            let (mut a, mut b) = (0, 0);
            for (k, &x) in vis.iter().enumerate() {
                let (bo, v) = (k / n, k % n);
                let r = rv[v * bones.len() + bo] > 0.5;
                if x && !r {
                    a += 1;
                }
                if !x && r {
                    b += 1;
                }
            }
            log(format!("visibility     {a} rays seen here only, {b} there only, of {} (trimesh misses some rays' own end vertex)", vis.len()));
            let (_, dist) = geom::pts2line(&p.pos, &bones);
            let rvis: Vec<bool> = (0..vis.len())
                .map(|k| rv[(k % n) * bones.len() + k / n] > 0.5)
                .collect();
            let gv = geom::geodesic_by_visibility(n, bones.len(), &dist, rvis, &p.geodesic);
            let (_, want) = read(&dir.join("geo_dist.npy"))?;
            note(line("geo (ref vis)", &gv, &want, log));
        }
        log(format!(
            "volumetric     {} vertices x {} bones in {:.2} s",
            p.pos.len(),
            bones.len(),
            t0.elapsed().as_secs_f64()
        ));
        let (_, want) = read(&dir.join("geo_dist.npy"))?;
        note(line("geo dist", &geo, &want, log));
        let off = geo
            .iter()
            .zip(&want)
            .filter(|(a, b)| (**a - **b).abs() > 1e-3)
            .count();
        log(format!("geo dist       {off} of {} (vertex, bone) distances differ by more than 1e-3 (ray hits at edges and corners)", want.len()));
        if want.len() != p.pos.len() * bones.len() {
            return Err(Error("invalid reference geodesics".into()));
        }
        let (input, ids, _) = skel::skin_input(&p.pos, &want, &bones, &leaf);
        let (_, ri) = read(&dir.join("skin_input.npy"))?;
        let (_, rn) = read(&dir.join("skin_nn.npy"))?;
        if ri.len() != p.pos.len() * 43
            || rn.len() != p.pos.len() * 5
            || rn
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || v.fract() != 0. || *v >= bones.len() as f32)
        {
            return Err(Error("invalid reference skin inputs".into()));
        }
        // numpy's argsort is not stable: bones at the same distance (a leaf bone and the bone ending at its joint,
        // seen from a vertex nearest that joint) may come in either order
        let differ = (0..p.pos.len())
            .filter(|&v| (0..5).any(|k| ids[v * 5 + k] != rn[v * 5 + k] as usize))
            .count();
        let ties = (0..p.pos.len())
            .filter(|&v| {
                let mut d: Vec<f32> = (0..5)
                    .map(|k| want[v * bones.len() + rn[v * 5 + k] as usize])
                    .collect();
                d.sort_by(f32::total_cmp);
                d.windows(2).any(|w| w[0] == w[1])
            })
            .count();
        log(format!("skin bones     {differ} of {} vertices order their 5 nearest bones otherwise ({ties} have equal distances)", p.pos.len()));
        let same: Vec<usize> = (0..p.pos.len())
            .filter(|&v| (0..5).all(|k| ids[v * 5 + k] == rn[v * 5 + k] as usize))
            .collect();
        let pick = |x: &[f32]| {
            same.iter()
                .flat_map(|&v| x[v * 43..(v + 1) * 43].to_vec())
                .collect::<Vec<f32>>()
        };
        note(line("skin input", &pick(&input), &pick(&ri), log));
        let logits = m.skin.run(&m.ops, &mi, &ri)?;
        note(line(
            "skin logits",
            &logits,
            &read(&dir.join("skin_logits.npy"))?.1,
            log,
        ));
        let (_, rlog) = read(&dir.join("skin_logits.npy"))?;
        let (_, rm) = read(&dir.join("skin_mask.npy"))?;
        if rlog.len() != p.pos.len() * 5 || rm.len() != p.pos.len() * 5 {
            return Err(Error("invalid reference skin outputs".into()));
        }
        let rids: Vec<usize> = rn.iter().map(|x| *x as usize).collect();
        let w = skel::skin_weights(
            &rlog,
            &rids,
            &rm,
            bones.len(),
            &crate::one_ring(p.pos.len(), &p.tpl),
        );
        note(line(
            "skin weights",
            &w,
            &read(&dir.join("skin_full.npy"))?.1,
            log,
        ));
    }
    Ok(worst)
}

/// Independent small numerical checks for all five private kernel entrypoints.
/// Requires a real GPU and the linked private HIP translation unit.
pub fn kernels(gpu: &std::sync::Arc<nextrocm_engine_core::Gpu>) -> Result<usize> {
    let o = crate::net::Ops::new(gpu)?;
    let verify = |name: &str, got: &[f32], want: &[f32], tol: f32| -> Result<()> {
        if got.len() != want.len()
            || got
                .iter()
                .zip(want)
                .any(|(a, b)| !a.is_finite() || (a - b).abs() > tol)
        {
            return Err(Error(format!(
                "RigNet kernel {name} mismatch: {got:?} vs {want:?}"
            )));
        }
        Ok(())
    };
    let x = o.up(&[-2., 1., 3., -4., 0., 2.])?;
    let scale = o.up(&[2., 3.])?;
    let shift = o.up(&[1., -1.])?;
    o.k.relu_affine(&x, 3, 2, Some((&scale, &shift)))?;
    verify(
        "relu_affine",
        &x.to_f32()?,
        &[1., 2., 7., -1., 1., 5.],
        1e-6,
    )?;
    for kind in [crate::ffi::TANH, crate::ffi::SIGMOID] {
        let values = [-4.0f32, -0.5, 0., 0.5, 4.];
        let x = o.up(&values)?;
        o.k.act(&x, values.len(), kind)?;
        let expected: Vec<f32> = values
            .iter()
            .map(|v| {
                if kind == 0 {
                    v.tanh()
                } else {
                    1. / (1. + (-v).exp())
                }
            })
            .collect();
        verify("activation", &x.to_f32()?, &expected, 2e-6)?;
    }
    let x = o.up(&[1., 2., 3., 4., 5., 6.])?;
    let src = o.up_i32(&[0, 2, 1])?;
    let dst = o.up_i32(&[1, 0, 2])?;
    let out = o.buf(12)?;
    o.k.edge_feat(&x, 2, &src, &dst, 3, &out)?;
    verify(
        "edge_feat",
        &out.to_f32()?,
        &[3., 4., -2., -2., 1., 2., 4., 4., 5., 6., -2., -2.],
        0.,
    )?;
    let rowptr = o.up_i32(&[0, 2, 2, 3])?;
    let out = o.up(&[-99.; 12])?;
    o.k.seg_max(&x, 2, &rowptr, 3, &out, 4, 1)?;
    verify(
        "seg_max",
        &out.to_f32()?,
        &[-99., 3., 4., -99., -99., 0., 0., -99., -99., 5., 6., -99.],
        0.,
    )?;
    let out = o.up(&[-99.; 12])?;
    o.k.copy_cols(&x, 2, 3, 2, &out, 4, 1)?;
    verify(
        "copy_cols",
        &out.to_f32()?,
        &[-99., 1., 2., -99., -99., 3., 4., -99., -99., 5., 6., -99.],
        0.,
    )?;
    o.k.copy_cols(&x, 0, 3, 2, &out, 4, 1)?;
    verify(
        "copy broadcast",
        &out.to_f32()?,
        &[-99., 1., 2., -99., -99., 1., 2., -99., -99., 1., 2., -99.],
        0.,
    )?;
    gpu.sync()?;
    Ok(7)
}

/// Write CPU geometry for the independent RigNet reference script. No GPU required.
pub fn write_inputs(p: &Prepared, dir: &Path) -> Result<()> {
    let n = p.pos.len();
    std::fs::create_dir_all(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
    write(
        &dir.join("pos.npy"),
        &[n, 3],
        &p.pos.iter().flatten().copied().collect::<Vec<_>>(),
    )?;
    write(
        &dir.join("faces.npy"),
        &[p.faces.len(), 3],
        &p.faces
            .iter()
            .flatten()
            .map(|x| *x as f32)
            .collect::<Vec<_>>(),
    )?;
    let ed = |e: &[(u32, u32)]| {
        e.iter()
            .flat_map(|(a, b)| [*a as f32, *b as f32])
            .collect::<Vec<_>>()
    };
    write(&dir.join("tpl.npy"), &[p.tpl.len(), 2], &ed(&p.tpl))?;
    write(&dir.join("geo.npy"), &[p.geo.len(), 2], &ed(&p.geo))?;
    write(&dir.join("geodesic.npy"), &[n, n], &p.geodesic.matrix())?;
    let d = p.vox.dims;
    write(
        &dir.join("vox.npy"),
        &[d, d, d],
        &p.vox
            .data
            .iter()
            .map(|x| if *x { 1.0 } else { 0.0 })
            .collect::<Vec<_>>(),
    )?;
    let t = p.vox.translate;
    write(
        &dir.join("vox_meta.npy"),
        &[4],
        &[t[0], t[1], t[2], p.vox.scale],
    )?;
    Ok(())
}
