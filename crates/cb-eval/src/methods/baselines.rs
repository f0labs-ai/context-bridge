//! The paper's baselines (DESIGN.md section 4.1).

use crate::context::{QuestionContext, ViewLists};
use crate::error::EvalError;
use crate::methods::Ranked;
use crate::methods::fusion::{
    decay_factor, min_max_scaled, order_by_score_then_newer_then_id, weighted_reciprocal_rank_fusion,
};
use crate::records::RecordIdx;
use std::collections::HashMap;

/// Below this many records in its window the hard cut-off falls back to the whole index.
pub const CUTOFF_MIN_RECORDS: usize = 5;

/// `dense`: the dense list exactly as the service returned it, with its cosine scores.
pub fn dense_as_given(lists: &ViewLists) -> Vec<Ranked> {
    lists
        .dense
        .iter()
        .map(|listed| Ranked {
            record: listed.record,
            score: listed.score,
        })
        .collect()
}

/// `rrf(k)`: `rrf_k([D, S], [0.5, 0.5])` over the whole lists.
pub fn rrf_of_dense_and_keywords(ctx: &QuestionContext<'_>, k: u32) -> Result<Vec<Ranked>, EvalError> {
    let now = ctx.now()?;
    let dense: Vec<RecordIdx> = now.dense.iter().map(|l| l.record).collect();
    let sparse: Vec<RecordIdx> = now.sparse.iter().map(|l| l.record).collect();
    let mut rows: Vec<Ranked> = weighted_reciprocal_rank_fusion(&[(&dense, 0.5), (&sparse, 0.5)], k)
        .into_iter()
        .map(|(record, score)| Ranked { record, score })
        .collect();
    order_by_score_then_newer_then_id(&mut rows, ctx.records());
    Ok(rows)
}

/// `decay_only(h)`: the records of `D`, scored `cos * decay_h`.
pub fn dense_times_decay(
    ctx: &QuestionContext<'_>,
    half_life_s: u64,
    floor: f64,
) -> Result<Vec<Ranked>, EvalError> {
    let mut rows: Vec<Ranked> = ctx
        .now()?
        .dense
        .iter()
        .map(|listed| Ranked {
            record: listed.record,
            score: listed.score * decay_factor(ctx.age_of(listed.record), half_life_s, floor),
        })
        .collect();
    order_by_score_then_newer_then_id(&mut rows, ctx.records());
    Ok(rows)
}

/// `cutoff(W)`: `dense` on the window's view; when that list has fewer than 5 records, `dense` on `now`.
pub fn dense_within_window_or_all(window: &ViewLists, now: &ViewLists) -> Vec<Ranked> {
    if window.dense.len() < CUTOFF_MIN_RECORDS {
        dense_as_given(now)
    } else {
        dense_as_given(window)
    }
}

/// `cc(alpha)`: `alpha * mm(cos) + (1 - alpha) * mm(bm25)` over the records of
/// either list, each score min–max scaled within its own list and a record
/// missing from a list counting 0 there.
pub fn convex_combination(ctx: &QuestionContext<'_>, alpha: f64) -> Result<Vec<Ranked>, EvalError> {
    let now = ctx.now()?;
    let mut row_of: HashMap<RecordIdx, usize> = HashMap::new();
    let mut rows: Vec<Ranked> = Vec::new();
    for (list, weight) in [(&now.dense, alpha), (&now.sparse, 1.0 - alpha)] {
        for (listed, scaled) in list.iter().zip(min_max_scaled(list)) {
            let row = *row_of.entry(listed.record).or_insert_with(|| {
                rows.push(Ranked {
                    record: listed.record,
                    score: 0.0,
                });
                rows.len() - 1
            });
            rows[row].score += weight * scaled;
        }
    }
    order_by_score_then_newer_then_id(&mut rows, ctx.records());
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use crate::testkit::{Scene, record};

    fn records() -> Vec<cb_types::Record> {
        vec![
            record("a", 1_000),
            record("b", 2_000),
            record("c", 3_000),
            record("d", 3_000),
            record("e", 3_590),
            record("f", 3_700),
        ]
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-15
    }

    #[test]
    fn dense_keeps_the_service_order_and_puts_tied_scores_newer_first() {
        // Different scores keep the service's order, even against recency: a (written 1 000) before
        // c (3 000). The service returned e and b tied at 0.5 with b first; e is newer and goes first.
        // c and d tie and share a second: the smaller id goes first.
        let scene = Scene::new(4_000, records()).now(
            &[("a", 0.9), ("b", 0.5), ("e", 0.5), ("d", 0.4), ("c", 0.4)],
            &[("a", 3.0)],
        );
        assert_eq!(scene.order("dense"), ["a", "e", "b", "c", "d"]);
    }

    #[test]
    fn a_tie_in_a_list_decides_the_ranks_fusion_sees() {
        // D lists b and e tied at 0.5 (b first as returned); S lists only b. Ranked as returned,
        // b would get 0.5/61 + 0.5/61. With the tie put newer first, e is rank 1 and b rank 2 in D:
        // b gets 0.5/62 + 0.5/61 = 0.016261237…, e gets 0.5/61 = 0.008196721….
        let scene = Scene::new(4_000, records()).now(&[("b", 0.5), ("e", 0.5)], &[("b", 2.0)]);
        let ranked = scene.rank("rrf(k=60)").unwrap();
        assert_eq!(ranked[0], ("b".to_string(), 0.5 / 62.0 + 0.5 / 61.0));
        assert_eq!(ranked[1], ("e".to_string(), 0.5 / 61.0));
    }

    #[test]
    fn rrf_fuses_both_lists_with_half_weights() {
        let scene = Scene::new(4_000, records()).now(
            &[("a", 0.9), ("b", 0.8), ("c", 0.7)],
            &[("c", 12.0), ("a", 9.0), ("d", 1.0)],
        );
        let ranked = scene.rank("rrf(k=60)").unwrap();
        let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
        // a: 0.5/61 + 0.5/62 = 0.016261237…   (rank 1 in D, rank 2 in S)
        // c: 0.5/63 + 0.5/61 = 0.016133229…   (rank 3 in D, rank 1 in S)
        // b: 0.5/62          = 0.008064516…   (rank 2 in D)
        // d: 0.5/63          = 0.007936507…   (rank 3 in S)
        assert_eq!(ids, ["a", "c", "b", "d"]);
        assert_eq!(ranked[0].1, 0.5 / 61.0 + 0.5 / 62.0);
        assert_eq!(ranked[1].1, 0.5 / 63.0 + 0.5 / 61.0);
        assert_eq!(ranked[2].1, 0.5 / 62.0);
        assert_eq!(ranked[3].1, 0.5 / 63.0);
        assert!(close(ranked[0].1, 0.016261237440507668));
        // With k = 10 the same ranks give a: 0.5/11 + 0.5/12 = 0.045454… + 0.041666… = 0.087121…
        assert!(close(scene.rank("rrf(k=10)").unwrap()[0].1, 0.08712121212121213));
    }

    #[test]
    fn rrf_breaks_a_tie_by_the_newer_record_then_by_id() {
        // Each record is first in one list only: all score 0.5/61.
        let scene = Scene::new(4_000, records()).now(&[("a", 0.9)], &[("b", 5.0)]);
        assert_eq!(scene.order("rrf(k=60)"), ["b", "a"]);
        // c and d were written in the same second: the smaller id goes first.
        let same_second = Scene::new(4_000, records()).now(&[("d", 0.9)], &[("c", 5.0)]);
        assert_eq!(same_second.order("rrf(k=60)"), ["c", "d"]);
    }

    #[test]
    fn decay_only_multiplies_cosine_by_the_decay() {
        // Asked at 4 000 with h = 1 000 s. a is 3 000 s old: 2^-3 = 0.125. b is 2 000 s old: 2^-2 = 0.25.
        // c is 1 000 s old: 2^-1 = 0.5.
        let scene = Scene::new(4_000, records()).now(&[("a", 0.9), ("b", 0.8), ("c", 0.3)], &[]);
        let ranked = scene.rank("decay_only(h=1000)").unwrap();
        // a: 0.9 * 0.125 = 0.1125; b: 0.8 * 0.25 = 0.2; c: 0.3 * 0.5 = 0.15
        assert_eq!(
            ranked,
            [
                ("b".to_string(), 0.8 * 0.25),
                ("c".to_string(), 0.3 * 0.5),
                ("a".to_string(), 0.9 * 0.125)
            ]
        );
    }

    #[test]
    fn decay_only_holds_the_floor_for_very_old_records() {
        // h = 60 s. a is 3 000 s = 50 half-lives old, far past the floor of 1e-6: 0.9 * 1e-6.
        // c is 1 000 s old: 2^(-1000/60) = 2^-16 * 2^(-2/3) = 1.52587890625e-5 * 0.6299605249…
        // = 9.612434768e-6, still above the floor: 0.3 * 9.612434768e-6 = 2.8837304304e-6.
        let scene = Scene::new(4_000, records()).now(&[("a", 0.9), ("c", 0.3)], &[]);
        let ranked = scene.rank("decay_only(h=60)").unwrap();
        assert_eq!(ranked[1], ("a".to_string(), 0.9 * 1e-6));
        assert_eq!(ranked[0].0, "c");
        assert!((ranked[0].1 - 2.8837304304e-6).abs() < 1e-14, "{}", ranked[0].1);
    }

    #[test]
    fn batch_is_dense_when_nothing_was_written_after_the_boundary() {
        // Asked at 3 500 with a 300 s period: the index last refreshed at floor(3500/300)*300 = 3 300.
        // The newest records written by 3 500 are c and d (3 000), so nothing was written after the
        // boundary and the batch index holds exactly what the live one holds.
        let lists: [(&str, f64); 4] = [("c", 0.9), ("a", 0.8), ("d", 0.7), ("b", 0.6)];
        let scene =
            Scene::new(3_500, records())
                .now(&lists, &[("a", 2.0)])
                .view("batch_300", &lists, &[("a", 2.0)]);
        assert_eq!(
            scene.rank("batch(period=300)").unwrap(),
            scene.rank("dense").unwrap()
        );
        assert_eq!(scene.order("batch(period=300)"), ["c", "a", "d", "b"]);
    }

    #[test]
    fn batch_misses_what_was_written_after_the_boundary() {
        // Asked at 3 595: e (3 590) is in the live index. The 300 s index last refreshed at
        // floor(3595/300)*300 = 3 300, before e was written, so its list starts at c.
        let scene = Scene::new(3_595, records())
            .now(&[("e", 0.95), ("c", 0.9), ("a", 0.8)], &[])
            .view("batch_300", &[("c", 0.9), ("a", 0.8)], &[]);
        assert_eq!(scene.order("dense"), ["e", "c", "a"]);
        assert_eq!(scene.order("batch(period=300)"), ["c", "a"]);
    }

    #[test]
    fn cutoff_uses_the_window_when_it_holds_five_records() {
        let now = [
            ("a", 0.9),
            ("b", 0.8),
            ("c", 0.7),
            ("d", 0.6),
            ("e", 0.5),
            ("f", 0.4),
        ];
        // Asked at 4 000 with a 3 600 s window: records written after 400, here all six; five listed.
        let scene = Scene::new(4_000, records()).now(&now, &[]).view(
            "window_3600",
            &[("b", 0.8), ("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4)],
            &[],
        );
        assert_eq!(scene.order("cutoff(window=3600)"), ["b", "c", "d", "e", "f"]);
    }

    #[test]
    fn cutoff_falls_back_to_dense_below_five_records() {
        let now = [
            ("a", 0.9),
            ("b", 0.8),
            ("c", 0.7),
            ("d", 0.6),
            ("e", 0.5),
            ("f", 0.4),
        ];
        // A 1 800 s window at 4 000 holds records written after 2 200: c, d, e, f. Four is below five.
        let scene = Scene::new(4_000, records()).now(&now, &[]).view(
            "window_1800",
            &[("c", 0.7), ("d", 0.6), ("e", 0.5), ("f", 0.4)],
            &[],
        );
        assert_eq!(
            scene.rank("cutoff(window=1800)").unwrap(),
            scene.rank("dense").unwrap()
        );
        // An empty window falls back too.
        let empty = Scene::new(4_000, records())
            .now(&now, &[])
            .view("window_60", &[], &[]);
        assert_eq!(empty.order("cutoff(window=60)"), ["a", "b", "c", "d", "e", "f"]);
    }

    #[test]
    fn cc_scales_each_list_and_counts_a_missing_score_as_zero() {
        // D: a 0.9, b 0.7, c 0.5 → mm 1, 0.5, 0.   S: c 10, d 6, a 2 → mm 1, 0.5, 0.
        let scene = Scene::new(4_000, records()).now(
            &[("a", 0.9), ("b", 0.7), ("c", 0.5)],
            &[("c", 10.0), ("d", 6.0), ("a", 2.0)],
        );
        let ranked = scene.rank("cc(alpha=0.6)").unwrap();
        // a: 0.6*1   + 0.4*0   = 0.6
        // c: 0.6*0   + 0.4*1   = 0.4
        // b: 0.6*0.5 + (absent) = 0.3
        // d: (absent) + 0.4*0.5 = 0.2
        let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["a", "c", "b", "d"]);
        for ((_, score), expected) in ranked.iter().zip([0.6, 0.4, 0.3, 0.2]) {
            assert!(close(*score, expected), "{score} vs {expected}");
        }
        // alpha = 1 is the dense scores alone; d has none, and c's scaled score is 0: they tie at 0
        // and the newer-then-id rule orders them (c and d share a second).
        assert_eq!(scene.order("cc(alpha=1)"), ["a", "b", "c", "d"]);
        // alpha = 0 is the keyword scores alone: c 1, d 0.5, then a (scaled 0) and b (absent) tie at 0;
        // b is newer.
        assert_eq!(scene.order("cc(alpha=0)"), ["c", "d", "b", "a"]);
    }
}
