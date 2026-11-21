//! CPU-only diagnostic. Does not modify production visibility or reference fixtures.
use nextrocm_model3d_rignet::{geom, npy};
use serde_json::json;
use std::path::Path;
type V = [f64; 3];
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn dot(a: V, b: V) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V) -> f64 {
    dot(a, a).sqrt()
}
// Independent f64 Moller-Trumbore: accepts triangle boundaries, tests the full ray.
fn hit(o: V, d: V, a: V, b: V, c: V) -> Option<(f64, f64)> {
    let e = sub(b, a);
    let f = sub(c, a);
    let p = cross(d, f);
    let det = dot(e, p);
    if det.abs() < 1e-18 {
        return None;
    }
    let t = sub(o, a);
    let u = dot(t, p) / det;
    let q = cross(t, e);
    let v = dot(d, q) / det;
    if u < -1e-10 || v < -1e-10 || u + v > 1.0 + 1e-10 {
        return None;
    }
    let h = dot(f, q) / det;
    if h < 0.0 {
        return None;
    }
    Some((h, dot(d, cross(e, f)).abs() / norm(cross(e, f))))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args()
        .nth(1)
        .ok_or("usage: rignet_ray_audit REFERENCE_DIR")?;
    let dir = Path::new(&a);
    let read = |n: &str, width: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (shape, values) = npy::read(&dir.join(n)).map_err(|e| e.0)?;
        if shape.len() != 2
            || shape[0] == 0
            || shape[1] != width
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(format!("invalid {n} geometry or values").into());
        }
        Ok(values)
    };
    let p = read("pos.npy", 3)?;
    let f = read("faces.npy", 3)?;
    let b = read("bones.npy", 6)?;
    if f.iter()
        .any(|v| *v < 0. || v.fract() != 0. || *v as usize >= p.len() / 3)
    {
        return Err("invalid face index".into());
    }
    let rv = read("vis_raw.npy", b.len() / 6)?;
    if rv.len() != p.len() / 3 * (b.len() / 6) || rv.iter().any(|v| *v != 0. && *v != 1.) {
        return Err("invalid visibility matrix".into());
    }
    let pos: Vec<[f32; 3]> = p.chunks_exact(3).map(|x| x.try_into().unwrap()).collect();
    let faces: Vec<[u32; 3]> = f
        .chunks_exact(3)
        .map(|x| [x[0] as u32, x[1] as u32, x[2] as u32])
        .collect();
    let bones: Vec<[f32; 6]> = b.chunks_exact(6).map(|x| x.try_into().unwrap()).collect();
    let (org, _) = geom::pts2line(&pos, &bones);
    let ends: Vec<_> = (0..bones.len()).flat_map(|_| pos.iter().copied()).collect();
    let native = geom::visible(&pos, &faces, &org, &ends);
    let mut cases = Vec::new();
    let mut independent_disagrees = 0;
    let mut same_origin_disagrees = 0;
    let mut plane_cutoff_explains = 0;
    for k in 0..native.len() {
        let vertex = k % pos.len();
        let bone = k / pos.len();
        let reference = rv[vertex * bones.len() + bone] > 0.5;
        if native[k] == reference {
            continue;
        }
        let bo = bones[bone].map(f64::from);
        let start = [bo[0], bo[1], bo[2]];
        let delta = sub([bo[3], bo[4], bo[5]], start);
        let end = pos[vertex].map(f64::from);
        let t = if dot(delta, delta) < 1e-8 {
            0.0
        } else {
            (dot(sub(end, start), delta) / dot(delta, delta)).clamp(0.0, 1.0)
        };
        let o = std::array::from_fn(|i| start[i] + delta[i] * t);
        let d = sub(end, o);
        let len = norm(d);
        let mut nearest = (f64::INFINITY, 0, 0.0);
        let mut cut = f64::INFINITY;
        let native_o = org[k].map(f64::from);
        let native_d = sub(end, native_o);
        let mut same_nearest = f64::INFINITY;
        for (i, tri) in faces.iter().enumerate() {
            if let Some((h, _)) = hit(
                native_o,
                native_d,
                pos[tri[0] as usize].map(f64::from),
                pos[tri[1] as usize].map(f64::from),
                pos[tri[2] as usize].map(f64::from),
            ) {
                same_nearest = same_nearest.min(h);
            }
            if let Some((h, denom)) = hit(
                o,
                d,
                pos[tri[0] as usize].map(f64::from),
                pos[tri[1] as usize].map(f64::from),
                pos[tri[2] as usize].map(f64::from),
            ) {
                if h < nearest.0 {
                    nearest = (h, i, denom);
                }
                if denom > 1e-5 {
                    cut = cut.min(h);
                }
            }
        }
        let same_visible =
            !same_nearest.is_finite() || ((same_nearest - 1.0) * norm(native_d)).abs() < 1e-4;
        same_origin_disagrees += usize::from(same_visible != native[k]);
        let visible = !nearest.0.is_finite() || ((nearest.0 - 1.0) * len).abs() < 1e-4;
        let cutoff_visible = !cut.is_finite() || ((cut - 1.0) * len).abs() < 1e-4;
        independent_disagrees += usize::from(visible != native[k]);
        plane_cutoff_explains += usize::from(cutoff_visible == reference);
        cases.push(json!({"vertex":vertex,"bone":bone,"native":native[k],"reference":reference,"independent_f64":visible,"independent_native_origin":same_visible,"native_origin":native_o,"plane_cutoff_visible":cutoff_visible,"origin":o,"end":end,"nearest_t":nearest.0,"nearest_face":nearest.1,"plane_projection":nearest.2,"cutoff_nearest_t":if cut.is_finite(){Some(cut)}else{None}}));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"scope":"independent diagnostic using saved F32 bones promoted to F64; not full trimesh emulation","mismatches":cases.len(),"independent_disagrees_with_native":independent_disagrees,"same_origin_disagrees_with_native":same_origin_disagrees,"plane_cutoff_explains":plane_cutoff_explains,"cases":cases})
        )?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_parallel_endpoint_and_farther_surface_are_not_occlusion() {
        // Exact analytic intersections: endpoint t=1, farther plane t=2.
        let mut v = vec![
            [0., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [-1., 0.5, -1.],
            [1., 0.5, -1.],
            [0., 0.5, 1.],
        ];
        let mut faces = vec![[0, 1, 2], [3, 4, 5]];
        let o = [-1e-6, -0.5, 0.];
        let end = [0., 0., 0.];
        assert_eq!(geom::visible(&v, &faces, &[o], &[end]), vec![true]);
        let origin = o.map(f64::from);
        let d = sub(end.map(f64::from), origin);
        let (t, _) = hit(
            origin,
            d,
            v[0].map(f64::from),
            v[1].map(f64::from),
            v[2].map(f64::from),
        )
        .unwrap();
        assert!((t - 1.).abs() < 1e-12);
        // A real blocker at t=.5 must still hide the endpoint.
        v.extend([[-1., -0.25, -1.], [1., -0.25, -1.], [0., -0.25, 1.]]);
        faces.push([6, 7, 8]);
        assert_eq!(geom::visible(&v, &faces, &[o], &[end]), vec![false]);
        let (t, _) = hit(
            origin,
            d,
            v[6].map(f64::from),
            v[7].map(f64::from),
            v[8].map(f64::from),
        )
        .unwrap();
        assert!((t - 0.5).abs() < 1e-12);
    }
}
