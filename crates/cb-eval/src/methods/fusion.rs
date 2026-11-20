//! The arithmetic every method is built from: reciprocal rank fusion, the
//! half-life decay, min–max scaling, the spread of timestamps, and the one
//! ordering rule.

use crate::context::Listed;
use crate::methods::Ranked;
use crate::records::{RecordIdx, RecordTable};
use cb_types::UnixSeconds;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

/// Higher score first. Scores are finite, so this is a total order; `0.0` and `-0.0` tie.
pub fn cmp_score_desc(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

/// Order by score, highest first; equal scores by newer `written_at` first, then by record id.
pub fn order_by_score_then_newer_then_id(rows: &mut [Ranked], records: &RecordTable) {
    rows.sort_by(|a, b| {
        cmp_score_desc(a.score, b.score).then_with(|| records.cmp_newer_then_id(a.record, b.record))
    });
}

/// `2^(-age / h)`: 1 for a record written this second, one half after one half-life.
pub fn half_life_factor(age_s: i64, half_life_s: u64) -> f64 {
    (-(age_s as f64) / half_life_s as f64).exp2()
}

/// `decay_h(r) = max(2^(-age(r) / h), eps)`.
pub fn decay_factor(age_s: i64, half_life_s: u64, floor: f64) -> f64 {
    half_life_factor(age_s, half_life_s).max(floor)
}

/// `rrf_k(lists, weights)(r) = sum over lists L containing r of w_L / (k + rank_L(r))`,
/// with `rank_L` counted from 1.
///
/// Each list is given with its weight. Records come back in the order they
/// first appear, list after list; each list must hold a record at most once.
pub fn weighted_reciprocal_rank_fusion(lists: &[(&[RecordIdx], f64)], k: u32) -> Vec<(RecordIdx, f64)> {
    let mut row_of: HashMap<RecordIdx, usize> = HashMap::new();
    let mut fused: Vec<(RecordIdx, f64)> = Vec::new();
    for (list, weight) in lists {
        for (position, record) in list.iter().enumerate() {
            let term = weight / (f64::from(k) + (position + 1) as f64);
            match row_of.entry(*record) {
                Entry::Occupied(row) => fused[*row.get()].1 += term,
                Entry::Vacant(row) => {
                    row.insert(fused.len());
                    fused.push((*record, term));
                }
            }
        }
    }
    fused
}

/// Qdrant's built-in merge, as the product runs it: a record's score is the
/// sum of `1 / (position + 2)` over the lists that hold it, `position` counted
/// from 0, in `f32` as Qdrant computes it. Best first; equal scores by newer
/// `written_at` first, then by record id.
pub fn qdrant_rank_fusion(lists: &[&[RecordIdx]], records: &RecordTable) -> Vec<(RecordIdx, f32)> {
    const QDRANT_RRF_K: f32 = 2.0;
    let mut row_of: HashMap<RecordIdx, usize> = HashMap::new();
    let mut fused: Vec<(RecordIdx, f32)> = Vec::new();
    for list in lists {
        for (position, record) in list.iter().enumerate() {
            let term = 1.0 / (position as f32 + QDRANT_RRF_K);
            match row_of.entry(*record) {
                Entry::Occupied(row) => fused[*row.get()].1 += term,
                Entry::Vacant(row) => {
                    row.insert(fused.len());
                    fused.push((*record, term));
                }
            }
        }
    }
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| records.cmp_newer_then_id(a.0, b.0))
    });
    fused
}

/// `mm(x) = (x - min) / (max - min)` over the scores of one list, in list
/// order. When every score is the same (a list of one included) each record
/// gets 1, so being listed never counts the same as being absent.
pub fn min_max_scaled(list: &[Listed]) -> Vec<f64> {
    let min = list.iter().map(|l| l.score).fold(f64::INFINITY, f64::min);
    let max = list.iter().map(|l| l.score).fold(f64::NEG_INFINITY, f64::max);
    list.iter()
        .map(|l| {
            if max > min {
                (l.score - min) / (max - min)
            } else {
                1.0
            }
        })
        .collect()
}

/// The standard deviation of a set of timestamps, dividing by their number
/// (not by one less): `sqrt(sum (t - mean)^2 / n)`. Zero for an empty set.
///
/// Computed as `sqrt((n * sum d^2 - (sum d)^2) / n^2)` over whole-second
/// offsets `d` from the earliest timestamp, in integers, so only the final
/// division and root are rounded.
pub fn population_std_of_timestamps(times: &[UnixSeconds]) -> f64 {
    let Some(earliest) = times.iter().copied().min() else {
        return 0.0;
    };
    let n = times.len() as u128;
    let exact = || -> Option<f64> {
        let mut sum: u128 = 0;
        let mut sum_of_squares: u128 = 0;
        for t in times {
            let offset = u128::try_from(i128::from(*t) - i128::from(earliest)).ok()?;
            sum = sum.checked_add(offset)?;
            sum_of_squares = sum_of_squares.checked_add(offset.checked_mul(offset)?)?;
        }
        let numerator = n
            .checked_mul(sum_of_squares)?
            .checked_sub(sum.checked_mul(sum)?)?;
        Some((numerator as f64 / (n * n) as f64).sqrt())
    };
    exact().unwrap_or_else(|| {
        // Timestamps too far apart for 128-bit integers: the plain floating-point form.
        let offsets: Vec<f64> = times.iter().map(|t| (*t as f64) - (earliest as f64)).collect();
        let mean = offsets.iter().sum::<f64>() / offsets.len() as f64;
        (offsets.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / offsets.len() as f64).sqrt()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::record;

    fn table(times: &[(&str, i64)]) -> RecordTable {
        let records: Vec<_> = times.iter().map(|(id, t)| record(id, *t)).collect();
        RecordTable::from_records(&records).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-15
    }

    #[test]
    fn rrf_at_k_60_gives_the_known_scores() {
        let t = table(&[("a", 1), ("b", 1), ("c", 1)]);
        let at = |id: &str| t.index_of(id).unwrap();
        let dense = [at("a"), at("b"), at("c")];
        let sparse = [at("c"), at("a")];
        let fused = weighted_reciprocal_rank_fusion(&[(&dense, 0.5), (&sparse, 0.5)], 60);
        let ids: Vec<&str> = fused.iter().map(|(r, _)| t.id(*r)).collect();
        assert_eq!(ids, ["a", "b", "c"]);
        // a: rank 1 in D, rank 2 in S: 0.5/61 + 0.5/62 = 0.008196721… + 0.008064516… = 0.016261237…
        assert_eq!(fused[0].1, 0.5 / 61.0 + 0.5 / 62.0);
        assert!(close(fused[0].1, 0.016261237440507668));
        // b: rank 2 in D only: 0.5/62 = 0.008064516…
        assert_eq!(fused[1].1, 0.5 / 62.0);
        assert!(close(fused[1].1, 0.008064516129032258));
        // c: rank 3 in D, rank 1 in S: 0.5/63 + 0.5/61 = 0.007936507… + 0.008196721… = 0.016133229…
        assert_eq!(fused[2].1, 0.5 / 63.0 + 0.5 / 61.0);
        assert!(close(fused[2].1, 0.016133229247983346));
    }

    #[test]
    fn rrf_uses_each_lists_own_weight_and_the_given_k() {
        let t = table(&[("a", 1), ("b", 1)]);
        let at = |id: &str| t.index_of(id).unwrap();
        let (first, second, third) = ([at("a")], [at("b"), at("a")], [at("b")]);
        let third_each = 1.0 / 3.0;
        let fused = weighted_reciprocal_rank_fusion(
            &[(&first, third_each), (&second, third_each), (&third, third_each)],
            10,
        );
        // a: rank 1 in the first list, rank 2 in the second: (1/3)/11 + (1/3)/12 = 0.030303… + 0.027777… = 0.058080…
        assert_eq!(fused[0].1, third_each / 11.0 + third_each / 12.0);
        assert!(close(fused[0].1, 0.05808080808080808));
        // b: rank 1 in the second and in the third list: (1/3)/11 + (1/3)/11 = 0.060606…
        assert_eq!(fused[1].1, third_each / 11.0 + third_each / 11.0);
        assert!(close(fused[1].1, 0.06060606060606061));
        // k = 0 is the plain reciprocal rank: 1/1 and 1/2.
        let plain = weighted_reciprocal_rank_fusion(&[(&second, 1.0)], 0);
        assert_eq!((plain[0].1, plain[1].1), (1.0, 0.5));
    }

    #[test]
    fn the_decay_halves_every_half_life_and_stops_at_the_floor() {
        assert_eq!(half_life_factor(0, 1800), 1.0);
        assert_eq!(half_life_factor(1800, 1800), 0.5);
        assert_eq!(half_life_factor(3600, 1800), 0.25);
        // 2^(-900/1800) = 2^(-0.5) = 0.70710678…
        assert!(close(
            half_life_factor(900, 1800),
            std::f64::consts::FRAC_1_SQRT_2
        ));
        // h = 60 s. After 19 half-lives 2^-19 = 1.9073486328125e-6, still above the floor 1e-6.
        assert_eq!(decay_factor(19 * 60, 60, 1e-6), 1.9073486328125e-6);
        // After 20 half-lives 2^-20 = 9.5367431640625e-7 is below it: the floor holds.
        assert_eq!(half_life_factor(20 * 60, 60), 9.5367431640625e-7);
        assert_eq!(decay_factor(20 * 60, 60, 1e-6), 1e-6);
        assert_eq!(decay_factor(1_000_000_000, 60, 1e-6), 1e-6);
        // Without a floor a very old record underflows to zero.
        assert_eq!(decay_factor(1_000_000_000, 60, 0.0), 0.0);
    }

    #[test]
    fn min_max_scaling_spreads_a_list_over_zero_to_one() {
        let t = table(&[("a", 1), ("b", 1), ("c", 1)]);
        let listed = |pairs: &[(&str, f64)]| -> Vec<Listed> {
            pairs
                .iter()
                .enumerate()
                .map(|(served, (id, score))| Listed {
                    record: t.index_of(id).unwrap(),
                    score: *score,
                    served,
                })
                .collect()
        };
        // (10-4)/(10-4) = 1, (7-4)/6 = 0.5, (4-4)/6 = 0
        assert_eq!(
            min_max_scaled(&listed(&[("a", 10.0), ("b", 7.0), ("c", 4.0)])),
            [1.0, 0.5, 0.0]
        );
        assert_eq!(min_max_scaled(&listed(&[("a", 3.0)])), [1.0]);
        assert_eq!(min_max_scaled(&listed(&[("a", 3.0), ("b", 3.0)])), [1.0, 1.0]);
        assert_eq!(min_max_scaled(&[]), Vec::<f64>::new());
    }

    #[test]
    fn the_qdrant_merge_counts_positions_from_zero_with_constant_two() {
        let t = table(&[("a", 1), ("b", 1), ("c", 1), ("d", 1)]);
        let at = |id: &str| t.index_of(id).unwrap();
        let dense = [at("a"), at("b"), at("c")];
        let sparse = [at("b"), at("d")];
        let merged = qdrant_rank_fusion(&[&dense, &sparse], &t);
        let ids: Vec<&str> = merged.iter().map(|(r, _)| t.id(*r)).collect();
        // b: position 1 in D and 0 in S: 1/3 + 1/2 = 0.8333…; a: position 0 in D: 1/2;
        // d: position 1 in S: 1/3; c: position 2 in D: 1/4.
        assert_eq!(ids, ["b", "a", "d", "c"]);
        assert_eq!(merged[0].1, 1.0f32 / 3.0 + 1.0 / 2.0);
        assert_eq!(merged[1].1, 0.5);
        assert_eq!(merged[2].1, 1.0f32 / 3.0);
        assert_eq!(merged[3].1, 0.25);
    }

    #[test]
    fn the_qdrant_merge_breaks_ties_by_newer_then_id() {
        // Each record leads one list: both score 1/2.
        let t = table(&[("old", 100), ("new", 200), ("b", 50), ("a", 50)]);
        let at = |id: &str| t.index_of(id).unwrap();
        let merged = qdrant_rank_fusion(&[&[at("old"), at("b")], &[at("new"), at("a")]], &t);
        let ids: Vec<&str> = merged.iter().map(|(r, _)| t.id(*r)).collect();
        assert_eq!(ids, ["new", "old", "a", "b"]);
    }

    #[test]
    fn ordering_is_by_score_then_newer_then_id() {
        let t = table(&[("x", 100), ("y", 200), ("b", 300), ("a", 300), ("top", 1)]);
        let row = |id: &str, score: f64| Ranked {
            record: t.index_of(id).unwrap(),
            score,
        };
        let mut rows = [
            row("x", 0.5),
            row("b", 0.5),
            row("top", 0.9),
            row("y", 0.5),
            row("a", 0.5),
        ];
        order_by_score_then_newer_then_id(&mut rows, &t);
        let ids: Vec<&str> = rows.iter().map(|r| t.id(r.record)).collect();
        // 0.9 first; of the four at 0.5 the two written at 300 come first, a before b; then y (200), x (100).
        assert_eq!(ids, ["top", "a", "b", "y", "x"]);
        assert_eq!(cmp_score_desc(0.0, -0.0), Ordering::Equal);
    }

    #[test]
    fn the_spread_of_timestamps_divides_by_their_number() {
        // Two values 100 apart: mean 50, deviations ±50, sqrt((2500+2500)/2) = 50.
        assert_eq!(population_std_of_timestamps(&[0, 100]), 50.0);
        assert_eq!(population_std_of_timestamps(&[0, 120]), 60.0);
        // 1,2,3,4: mean 2.5, squares 2.25+0.25+0.25+2.25 = 5, sqrt(5/4) = 1.1180339887…
        assert!(close(
            population_std_of_timestamps(&[1, 2, 3, 4]),
            1.118033988749895
        ));
        assert_eq!(population_std_of_timestamps(&[10, 10, 10]), 0.0);
        assert_eq!(population_std_of_timestamps(&[7]), 0.0);
        assert_eq!(population_std_of_timestamps(&[]), 0.0);
        // Real timestamps are large; the offsets keep the arithmetic exact.
        assert_eq!(
            population_std_of_timestamps(&[1_780_000_120, 1_780_000_000]),
            60.0
        );
        // Out of reach of 128-bit integers: the floating-point form still answers.
        let far = population_std_of_timestamps(&[i64::MIN, i64::MAX, 0, 1, 2]);
        assert!(far.is_finite() && far > 1e18);
    }
}
