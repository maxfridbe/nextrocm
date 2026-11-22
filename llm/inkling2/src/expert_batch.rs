//! Stable first-encounter routing groups; no reordered weighted accumulation.
use crate::Result;
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Group {
    pub expert: usize,
    pub members: Vec<(usize, usize)>,
}
pub(crate) fn groups(ids: &[[i32; 8]]) -> Result<Vec<Group>> {
    if ids.is_empty() || ids.len() > 4 {
        return Err("batch routing rows outside1..4".into());
    }
    for row in ids {
        let mut seen = [false; 128];
        for &id in row {
            if !(0..128).contains(&id) || seen[id as usize] {
                return Err("invalid batch routing IDs".into());
            }
            seen[id as usize] = true;
        }
    }
    let mut groups: Vec<Group> = Vec::with_capacity(ids.len() * 8);
    for (token, row) in ids.iter().enumerate() {
        for (rank, &expert) in row.iter().enumerate() {
            if let Some(group) = groups.iter_mut().find(|g| g.expert == expert as usize) {
                group.members.push((token, rank));
            } else {
                groups.push(Group {
                    expert: expert as usize,
                    members: vec![(token, rank)],
                });
            }
        }
    }
    Ok(groups)
}
pub(crate) fn scratch_bytes(rows: usize) -> Result<usize> {
    if !(1..=4).contains(&rows) {
        return Err("expert batch rows outside1..4".into());
    }
    rows.checked_mul((2304 + 2304 + 1152 + 2304 + 8 * 2304) * 4)
        .ok_or_else(|| "batch scratch overflow".into())
}
#[derive(Clone, Copy, Debug, Default)]
pub struct BatchStats {
    pub visits: u64,
    pub logical_selections: u64,
    pub groups: u64,
    pub group_sizes: [u64; 4],
    pub gu_calls: u64,
    pub down_calls: u64,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rank_membership_and_stable_groups() {
        let ids = [
            [7, 2, 9, 1, 4, 6, 0, 3],
            [3, 9, 7, 2, 1, 4, 6, 0],
            [2, 3, 9, 7, 1, 4, 0, 6],
        ];
        let g = groups(&ids).unwrap();
        assert_eq!(g.len(), 8);
        assert_eq!(g[0].expert, 7);
        assert_eq!(g[0].members, vec![(0, 0), (1, 2), (2, 3)]);
        let mut recovered = vec![[0i32; 8]; 3];
        for x in &g {
            for &(t, r) in &x.members {
                recovered[t][r] = x.expert as i32;
            }
        }
        assert_eq!(recovered, ids);
        assert_eq!(g.iter().map(|x| x.members.len()).sum::<usize>(), 24);
    }
    #[test]
    fn invalid_routes_and_exact_scratch() {
        assert!(groups(&[]).is_err());
        assert!(groups(&[[0; 8]]).is_err());
        assert!(groups(&[[0, 1, 2, 3, 4, 5, 6, 128]]).is_err());
        assert!(groups(&[[0, 1, 2, 3, 4, 5, 6, -1]]).is_err());
        assert!(groups(&[[0, 1, 2, 3, 4, 5, 6, 7]; 5]).is_err());
        assert_eq!(scratch_bytes(2).unwrap(), 211968);
        assert_eq!(scratch_bytes(4).unwrap(), 423936);
        assert!(scratch_bytes(usize::MAX).is_err());
    }
}
