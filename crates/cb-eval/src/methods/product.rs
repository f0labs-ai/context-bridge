//! The product's own path and the corrected rules (DESIGN.md section 4.4).
//!
//! The product merges the first 64 records of `D` and of `S` with Qdrant's
//! built-in rank fusion (a record's score is the sum of `1 / (position + 2)`,
//! positions from 0), sends the first `depth` merged records to its relevance
//! model, drops those the model scores under 0.12 and orders the rest by that
//! score `rho`. Every method here starts from that. With `recent` above 0, the
//! first `recent` records of the newest list join the records sent to the
//! relevance model.

use crate::context::{Listed, QuestionContext};
use crate::error::EvalError;
use crate::methods::Ranked;
use crate::methods::fusion::{
    decay_factor, half_life_factor, order_by_score_then_newer_then_id, qdrant_rank_fusion,
};
use crate::records::RecordIdx;

/// How many records of each list the product merges.
pub const PRODUCT_PREFETCH: usize = 64;
/// The relevance score under which the voice agent's search drops a record.
pub const PRODUCT_CUTOFF: f64 = 0.12;

/// Which records reach the product's relevance model: the first `depth` of the
/// merged lists, and the first `recent` of the newest list (0: none).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool {
    pub depth: usize,
    pub recent: usize,
}

impl Pool {
    pub fn new(depth: u32, recent: u32) -> Self {
        Self {
            depth: depth as usize,
            recent: recent as usize,
        }
    }
}

/// The product's candidates with their relevance scores: the first `depth`
/// records of the merge of `D[..64]` and `S[..64]`, then the first `recent`
/// records of the newest list that are not among them, after removing from
/// each list every record `keep` rejects. Removed records are not replaced by
/// deeper ones: the run holds relevance scores for the first ones only.
///
/// "The first 64" are the records the service listed in its first 64 places
/// ([`Listed::served`]), merged in the list's own order (equal scores newer
/// first): a tie across place 64 never lets in a record the relevance model did
/// not score. A record of several passages may sit in the first 64 records
/// while its first passage came after the 64th; the model never scored it from
/// that list, so unless the other list brought it in, it is left out too.
///
/// With `recent` 0 the newest list is never read and a record's score is the
/// candidate lists' own; with `recent` above 0, a record's score is the higher
/// of the two passes' ([`QuestionContext::relevance_with_newest`]).
pub fn product_candidates(
    ctx: &QuestionContext<'_>,
    pool: Pool,
    keep: impl Fn(RecordIdx) -> bool,
) -> Result<Vec<(RecordIdx, f64)>, EvalError> {
    let now = ctx.now()?;
    let first = |list: &[Listed], n: usize| -> Vec<RecordIdx> {
        list.iter()
            .filter(|l| l.served < n && ctx.has_relevance(l.record))
            .map(|l| l.record)
            .filter(|r| keep(*r))
            .collect()
    };
    let (dense, sparse) = (
        first(&now.dense, PRODUCT_PREFETCH),
        first(&now.sparse, PRODUCT_PREFETCH),
    );
    let mut chosen: Vec<RecordIdx> = qdrant_rank_fusion(&[&dense, &sparse], ctx.records())
        .into_iter()
        .take(pool.depth)
        .map(|(record, _)| record)
        .collect();
    if pool.recent == 0 {
        return chosen
            .into_iter()
            .map(|record| Ok((record, ctx.relevance_of(record)?)))
            .collect();
    }
    // The newest list's records were all scored by its own pass.
    let newest = ctx
        .newest()?
        .iter()
        .filter(|l| l.served < pool.recent)
        .map(|l| l.record)
        .filter(|r| keep(*r));
    for record in newest.collect::<Vec<_>>() {
        if !chosen.contains(&record) {
            chosen.push(record);
        }
    }
    chosen
        .into_iter()
        .map(|record| Ok((record, ctx.relevance_with_newest(record)?)))
        .collect()
}

/// The product path with a score of the caller's choosing: candidates with
/// `rho >= 0.12` (the cut-off always reads `rho` itself), ordered by
/// `score(rho, age)`.
fn product_ordered_by(
    ctx: &QuestionContext<'_>,
    pool: Pool,
    keep: impl Fn(RecordIdx) -> bool,
    score: impl Fn(f64, i64) -> f64,
) -> Result<Vec<Ranked>, EvalError> {
    let mut rows: Vec<Ranked> = product_candidates(ctx, pool, keep)?
        .into_iter()
        .filter(|(_, rho)| *rho >= PRODUCT_CUTOFF)
        .map(|(record, rho)| Ranked {
            record,
            score: score(rho, ctx.age_of(record)),
        })
        .collect();
    order_by_score_then_newer_then_id(&mut rows, ctx.records());
    Ok(rows)
}

/// `product`: product candidates with `rho >= 0.12`, ordered by `rho`.
pub fn product(ctx: &QuestionContext<'_>, pool: Pool) -> Result<Vec<Ranked>, EvalError> {
    product_ordered_by(ctx, pool, |_| true, |rho, _| rho)
}

/// `product_decay(h)`: as `product`, ordered by `rho * decay_h`.
pub fn product_times_decay(
    ctx: &QuestionContext<'_>,
    pool: Pool,
    half_life_s: u64,
    floor: f64,
) -> Result<Vec<Ranked>, EvalError> {
    product_ordered_by(
        ctx,
        pool,
        |_| true,
        |rho, age| rho * decay_factor(age, half_life_s, floor),
    )
}

/// `product_bonus(beta, h)`: as `product`, ordered by `rho * (1 + beta * 2^(-age / h))`.
pub fn product_with_recency_bonus(
    ctx: &QuestionContext<'_>,
    pool: Pool,
    beta: f64,
    half_life_s: u64,
) -> Result<Vec<Ranked>, EvalError> {
    product_ordered_by(
        ctx,
        pool,
        |_| true,
        |rho, age| rho * (1.0 + beta * half_life_factor(age, half_life_s)),
    )
}

/// `product_ends`: as `product`, after dropping records with `valid_until <= t`.
pub fn product_without_ended(ctx: &QuestionContext<'_>, pool: Pool) -> Result<Vec<Ranked>, EvalError> {
    let records = ctx.records();
    let asked_at = ctx.asked_at();
    product_ordered_by(
        ctx,
        pool,
        |record| records.valid_until(record).is_none_or(|end| end > asked_at),
        |rho, _| rho,
    )
}

/// `keyed_latest` (**keyed**): only the newest visible record of each thread
/// is a candidate (a record without a thread always is), then `product`.
pub fn keyed_latest(ctx: &QuestionContext<'_>, pool: Pool) -> Result<Vec<Ranked>, EvalError> {
    let records = ctx.records();
    let asked_at = ctx.asked_at();
    product_ordered_by(
        ctx,
        pool,
        |record| {
            records
                .keyed_newest_visible_in_thread_of(record, asked_at)
                .is_none_or(|newest| newest == record)
        },
        |rho, _| rho,
    )
}

/// `product_newer_wins(tau)`: as `product`; then, walking down the list, a
/// record is dropped when a kept record says the same thing
/// (`cos(passage, passage) >= tau`) and is newer; when the newer one comes
/// later in the list it takes the older one's place.
///
/// A record that says the same as several kept older records takes the place
/// of the first and the others are dropped. Two records written in the same
/// second are both kept. Each record keeps its own `rho` as its score, so the
/// scores of this method's answer need not fall from top to bottom.
pub fn product_newer_wins(ctx: &QuestionContext<'_>, pool: Pool, tau: f64) -> Result<Vec<Ranked>, EvalError> {
    let records = ctx.records();
    let mut kept: Vec<Ranked> = Vec::new();
    for row in product(ctx, pool)? {
        let written_at = records.written_at(row.record);
        let mut a_newer_one_is_kept = false;
        let mut older_kept: Vec<usize> = Vec::new();
        for (place, other) in kept.iter().enumerate() {
            if ctx.passage_cosine(row.record, other.record, "product_newer_wins")? >= tau {
                let other_written_at = records.written_at(other.record);
                if other_written_at > written_at {
                    a_newer_one_is_kept = true;
                } else if other_written_at < written_at {
                    older_kept.push(place);
                }
            }
        }
        if a_newer_one_is_kept {
            continue;
        }
        match older_kept.split_first() {
            Some((first, rest)) => {
                kept[*first] = row;
                for place in rest.iter().rev() {
                    kept.remove(*place);
                }
            }
            None => kept.push(row),
        }
    }
    Ok(kept)
}

/// `oracle`: the graded answers, highest grade first (equal grades by newer
/// `written_at`, then id). A record's score is its grade.
pub fn gold_order(ctx: &QuestionContext<'_>) -> Vec<Ranked> {
    let mut rows: Vec<Ranked> = ctx
        .gold_for_oracle()
        .iter()
        .filter(|graded| graded.grade > 0)
        .map(|graded| Ranked {
            record: graded.record,
            score: f64::from(graded.grade),
        })
        .collect();
    order_by_score_then_newer_then_id(&mut rows, ctx.records());
    rows
}

#[cfg(test)]
mod tests {
    use crate::error::EvalError;
    use crate::testkit::{Scene, ending, record, standing, threaded};
    use cb_types::Record;

    fn records() -> Vec<Record> {
        vec![
            record("a", 1_000),
            record("b", 2_000),
            record("c", 3_000),
            record("d", 3_500),
            record("e", 3_900),
        ]
    }

    /// Asked at 4 000. Merge of D = a, b, c and S = b, d:
    /// b: 1/3 + 1/2 = 0.8333; a: 1/2; d: 1/3; c: 1/4.
    fn scene(rho: &[(&str, f64)]) -> Scene {
        Scene::new(4_000, records())
            .now(&[("a", 0.9), ("b", 0.8), ("c", 0.7)], &[("b", 5.0), ("d", 4.0)])
            .relevance(rho)
    }

    fn pairs(expected: &[(&str, f64)]) -> Vec<(String, f64)> {
        expected
            .iter()
            .map(|(id, score)| (id.to_string(), *score))
            .collect()
    }

    #[test]
    fn product_cuts_at_0_12_and_orders_by_the_relevance_score() {
        let scene = scene(&[("a", 0.30), ("b", 0.50), ("c", 0.11), ("d", 0.90)]);
        // c scores 0.11 < 0.12 and is dropped; the rest by rho.
        assert_eq!(
            scene.rank("product").unwrap(),
            pairs(&[("d", 0.90), ("b", 0.50), ("a", 0.30)])
        );
        // Exactly 0.12 is kept.
        let at_the_line = self::scene(&[("a", 0.30), ("b", 0.50), ("c", 0.12), ("d", 0.90)]);
        assert_eq!(at_the_line.order("product"), ["d", "b", "a", "c"]);
    }

    #[test]
    fn only_the_first_depth_merged_records_reach_the_relevance_model() {
        let scene = scene(&[("a", 0.30), ("b", 0.50), ("c", 0.11), ("d", 0.90)]);
        // depth 2: the merge's first two are b (0.8333) and a (0.5). d is never scored, whatever its rho.
        assert_eq!(
            scene.rank("product(depth=2)").unwrap(),
            pairs(&[("b", 0.50), ("a", 0.30)])
        );
        // depth 3 lets d (1/3) in, and it goes first.
        assert_eq!(scene.order("product(depth=3)"), ["d", "b", "a"]);
        assert_eq!(scene.order("product(depth=1)"), ["b"]);
    }

    #[test]
    fn equal_relevance_scores_go_to_the_newer_record_then_to_the_id() {
        let scene = scene(&[("a", 0.5), ("b", 0.5), ("c", 0.5), ("d", 0.5)]);
        assert_eq!(scene.order("product"), ["d", "c", "b", "a"]);
    }

    /// The scene of [`scene`] with a newest list: e (3 900, in no list) scored 0.70,
    /// d (3 500, also merged) scored 0.95 by the newest pass, c (3 000) scored 0.05.
    fn with_newest() -> Scene {
        scene(&[("a", 0.30), ("b", 0.50), ("c", 0.11), ("d", 0.90)])
            .newest(&["e", "d", "c"], &[("e", 0.70), ("d", 0.95), ("c", 0.05)])
    }

    #[test]
    fn recent_brings_the_newest_records_in_scored_and_cut_like_the_rest() {
        let scene = with_newest();
        // recent 0: the newest list is never read, and d keeps the candidate lists' score.
        assert_eq!(
            scene.rank("product").unwrap(),
            pairs(&[("d", 0.90), ("b", 0.50), ("a", 0.30)])
        );
        // recent 1: e joins. d, in both passes, takes the higher of its two scores.
        assert_eq!(
            scene.rank("product(recent=1)").unwrap(),
            pairs(&[("d", 0.95), ("e", 0.70), ("b", 0.50), ("a", 0.30)])
        );
        // recent 3: c joins too, and both passes put it under the cut-off of 0.12.
        assert_eq!(scene.order("product(recent=3)"), ["d", "e", "b", "a"]);
        // A merged record found in the newest list is not listed twice.
        assert_eq!(scene.order("product(depth=4,recent=3)"), ["d", "e", "b", "a"]);
    }

    #[test]
    fn recent_without_a_newest_list_is_an_error_and_never_a_silent_zero() {
        let scene = scene(&[("a", 0.30), ("b", 0.50), ("c", 0.11), ("d", 0.90)]);
        assert!(matches!(
            scene.rank("product_bonus(beta=0.5,h=1800,recent=8)"),
            Err(EvalError::MissingNewest { .. })
        ));
        // Every product method takes recent.
        let scene = with_newest();
        for method in [
            "product_decay(h=1800,recent=1)",
            "product_bonus(beta=0.5,h=1800,recent=1)",
            "product_ends(recent=1)",
            "keyed_latest(recent=1)",
        ] {
            assert!(scene.order(method).contains(&"e".to_string()), "{method}");
        }
    }

    #[test]
    fn recent_takes_the_first_of_the_newest_list_and_keep_still_rules() {
        // product_ends drops an ended record from the newest list too, and a deeper one
        // does not take its place.
        let records = vec![
            record("a", 1_000),
            ending("gone", 3_800, 3_900),
            record("late", 3_700),
        ];
        let scene = Scene::new(4_000, records)
            .now(&[("a", 0.9)], &[])
            .relevance(&[("a", 0.5)])
            .newest(&["gone", "late"], &[("gone", 0.9), ("late", 0.8)]);
        assert_eq!(scene.order("product_ends(recent=1)"), ["a"]);
        assert_eq!(scene.order("product_ends(recent=2)"), ["late", "a"]);
        assert_eq!(scene.order("product(recent=1)"), ["gone", "a"]);
    }

    #[test]
    fn a_tie_across_place_64_never_lets_in_a_record_the_model_did_not_score() {
        // 65 records with one keyword score: the service listed old-01..old-64 (scored),
        // then "new" in place 64 (not scored). Equal scores go newer first, so "new" moves
        // to the top of the list; it is still not one of the service's first 64.
        let mut records: Vec<Record> = (1..=64).map(|i| record(&format!("old-{i:02}"), 1_000)).collect();
        records.push(record("new", 2_000));
        let sparse: Vec<(String, f64)> = records.iter().map(|r| (r.id.clone(), 3.0)).collect();
        let sparse: Vec<(&str, f64)> = sparse.iter().map(|(id, s)| (id.as_str(), *s)).collect();
        let rho: Vec<(&str, f64)> = sparse[..64].iter().map(|(id, _)| (*id, 0.5)).collect();
        let scene = Scene::new(4_000, records).now(&[], &sparse).relevance(&rho);
        let order = scene.order("product(depth=200)");
        assert_eq!(order.len(), 64);
        assert!(!order.contains(&"new".to_string()));
    }

    #[test]
    fn only_the_first_64_of_each_list_are_merged() {
        let mut records = Vec::new();
        let mut dense = Vec::new();
        for i in 0..70 {
            records.push(record(&format!("r{i:02}"), 1_000 + i));
            dense.push((format!("r{i:02}"), 0.9 - 0.01 * i as f64));
        }
        let dense: Vec<(&str, f64)> = dense.iter().map(|(id, score)| (id.as_str(), *score)).collect();
        let rho: Vec<(&str, f64)> = dense.iter().map(|(id, _)| (*id, 0.5)).collect();
        let scene = Scene::new(4_000, records).now(&dense, &[]).relevance(&rho);
        let order = scene.order("product(depth=200)");
        assert_eq!(order.len(), 64);
        assert!(order.contains(&"r63".to_string()));
        assert!(!order.contains(&"r64".to_string()));
    }

    #[test]
    fn a_record_deep_in_both_lists_can_be_a_candidate_ahead_of_shallower_ones() {
        // D holds d0..d8, S holds s0..s8, and "both" sits at position 9 of each (counted from 0).
        // Merged: d0, s0 1/2; d1, s1 1/3; d2, s2 1/4; d3, s3 1/5; both 1/11 + 1/11 = 0.1818; then
        // d4, s4 at 1/6 = 0.1667. With depth 9, "both" is a candidate although it lies outside the
        // first 9 of either list: the relevance scores must reach deeper than `depth`.
        let mut records = vec![record("both", 500)];
        let mut dense = Vec::new();
        let mut sparse = Vec::new();
        for i in 0..9 {
            records.push(record(&format!("d{i}"), 1_000 + i));
            records.push(record(&format!("s{i}"), 2_000 + i));
            dense.push((format!("d{i}"), 0.9 - 0.01 * i as f64));
            sparse.push((format!("s{i}"), 9.0 - i as f64));
        }
        dense.push(("both".to_string(), 0.5));
        sparse.push(("both".to_string(), 0.5));
        let dense: Vec<(&str, f64)> = dense.iter().map(|(id, s)| (id.as_str(), *s)).collect();
        let sparse: Vec<(&str, f64)> = sparse.iter().map(|(id, s)| (id.as_str(), *s)).collect();
        let rho: Vec<(&str, f64)> = records.iter().map(|r| (r.id.as_str(), 0.5)).collect();
        let scene = Scene::new(4_000, records.clone())
            .now(&dense, &sparse)
            .relevance(&rho);
        let candidates = scene.order("product(depth=9)");
        assert_eq!(candidates.len(), 9);
        assert!(candidates.contains(&"both".to_string()));
        assert!(!candidates.contains(&"d4".to_string()) && !candidates.contains(&"s4".to_string()));
    }

    #[test]
    fn a_record_the_relevance_model_did_not_score_is_left_out_never_scored_zero() {
        // c and d were never scored: they are not candidates, and take no one's place.
        let scene = scene(&[("a", 0.30), ("b", 0.50)]);
        assert_eq!(scene.rank("product").unwrap(), pairs(&[("b", 0.50), ("a", 0.30)]));
        assert_eq!(scene.order("product(depth=2)"), ["b", "a"]);
        // With the newest list, a record it holds without a score of either pass is an error.
        let missing = scene.newest(&["e"], &[]);
        assert!(matches!(
            missing.rank("product(recent=1)"),
            Err(EvalError::MissingRelevance { record, .. }) if record == "e"
        ));
    }

    #[test]
    fn product_decay_multiplies_rho_by_the_decay_and_cuts_on_rho_alone() {
        // Asked at 4 000, h = 1 000 s. Ages: a 3 000 s (2^-3 = 0.125), b 2 000 s (0.25),
        // c 1 000 s (0.5), d 500 s (2^-0.5 = 0.70710678).
        let scene = scene(&[("a", 0.90), ("b", 0.50), ("c", 0.11), ("d", 0.20)]);
        assert_eq!(scene.order("product"), ["a", "b", "d"]);
        let ranked = scene.rank("product_decay(h=1000)").unwrap();
        // d: 0.20 * 0.70710678 = 0.14142136; b: 0.50 * 0.25 = 0.125; a: 0.90 * 0.125 = 0.1125.
        // c stays out: its rho is under 0.12. a stays in although 0.1125 < 0.12: the cut reads rho.
        assert_eq!(ranked[0].0, "d");
        assert!((ranked[0].1 - 0.1414213562373095).abs() < 1e-15);
        assert_eq!(ranked[1..], pairs(&[("b", 0.50 * 0.25), ("a", 0.90 * 0.125)]));
    }

    #[test]
    fn product_decay_with_a_huge_half_life_is_product() {
        for rho in [
            [("a", 0.90), ("b", 0.50), ("c", 0.11), ("d", 0.20)],
            [("a", 0.13), ("b", 0.14), ("c", 0.99), ("d", 0.14)],
        ] {
            let scene = scene(&rho);
            let plain = scene.rank("product").unwrap();
            // A half-life of 10^15 s: the oldest record (3 000 s) keeps 2^(-3e-12) of its score.
            let slow = scene.rank("product_decay(h=1000000000000000)").unwrap();
            assert_eq!(slow.len(), plain.len());
            for ((id, score), (plain_id, plain_score)) in slow.iter().zip(&plain) {
                assert_eq!(id, plain_id);
                assert!((score - plain_score).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn product_bonus_with_beta_zero_is_product() {
        for rho in [
            [("a", 0.90), ("b", 0.50), ("c", 0.11), ("d", 0.20)],
            [("a", 0.13), ("b", 0.14), ("c", 0.99), ("d", 0.14)],
        ] {
            let scene = scene(&rho);
            assert_eq!(
                scene.rank("product_bonus(beta=0,h=1000)").unwrap(),
                scene.rank("product").unwrap()
            );
        }
    }

    #[test]
    fn product_bonus_lifts_fresh_records_and_cuts_on_rho_alone() {
        let scene = scene(&[("a", 0.90), ("b", 0.50), ("c", 0.11), ("d", 0.20)]);
        // beta = 1: a: 0.90 * (1 + 0.125) = 1.0125; b: 0.50 * 1.25 = 0.625;
        // d: 0.20 * (1 + 0.70710678) = 0.34142136. c (rho 0.11) is not rescued by its bonus.
        let ranked = scene.rank("product_bonus(beta=1,h=1000)").unwrap();
        assert_eq!(ranked[..2], pairs(&[("a", 0.90 * 1.125), ("b", 0.50 * 1.25)]));
        assert_eq!(ranked[2].0, "d");
        assert!((ranked[2].1 - 0.3414213562373095).abs() < 1e-15);
        assert_eq!(ranked.len(), 3);
        // beta = 20: a: 0.90 * 3.5 = 3.15; d: 0.20 * (1 + 14.1421356) = 3.02842712; b: 0.50 * 6 = 3.0.
        assert_eq!(scene.order("product_bonus(beta=20,h=1000)"), ["a", "d", "b"]);
    }

    #[test]
    fn product_ends_drops_records_whose_stated_end_has_passed_before_the_merge() {
        let mut records = records();
        records.push(ending("x", 500, 3_800)); // ended 200 s before the question
        records.push(ending("y", 600, 4_000)); // ends at the very moment asked: dropped
        records.push(ending("z", 700, 4_001)); // still running
        let scene = Scene::new(4_000, records)
            .now(
                &[("x", 0.9), ("a", 0.8), ("b", 0.7), ("y", 0.6)],
                &[("x", 5.0), ("d", 4.0), ("z", 3.0)],
            )
            .relevance(&[
                ("x", 0.9),
                ("a", 0.3),
                ("b", 0.5),
                ("d", 0.4),
                ("y", 0.8),
                ("z", 0.2),
            ]);
        assert_eq!(scene.order("product"), ["x", "y", "b", "d", "a", "z"]);
        assert_eq!(scene.order("product_ends"), ["b", "d", "a", "z"]);
        // The ended records are removed before the merge. With depth 2:
        //   product: x leads both lists (1/2 + 1/2); a and d tie at 1/3 and d is newer → x, d.
        //   product_ends: lists are a, b and d, z: a and d tie at 1/2 (d newer) → d, a.
        assert_eq!(scene.order("product(depth=2)"), ["x", "d"]);
        assert_eq!(scene.order("product_ends(depth=2)"), ["d", "a"]);
    }

    fn same_thing_scene(times: [i64; 5], rho: [f64; 5]) -> Scene {
        let ids = ["p1", "p2", "p3", "p4", "p5"];
        let records: Vec<Record> = ids.iter().zip(times).map(|(id, t)| record(id, t)).collect();
        let dense: Vec<(&str, f64)> = ids
            .iter()
            .zip([0.9, 0.8, 0.7, 0.6, 0.5])
            .map(|(id, s)| (*id, s))
            .collect();
        let rho: Vec<(&str, f64)> = ids.iter().zip(rho).map(|(id, r)| (*id, r)).collect();
        // p1, p3, p4 and p5 say the same thing (cosines 0.98 and up); p2 is about something else.
        Scene::new(4_000, records)
            .now(&dense, &[])
            .relevance(&rho)
            .vectors(
                2,
                &[
                    ("p1", &[1.0, 0.0]),
                    ("p2", &[0.0, 1.0]),
                    ("p3", &[1.0, 0.1]),
                    ("p4", &[1.0, 0.2]),
                    ("p5", &[1.0, 0.05]),
                ],
            )
    }

    #[test]
    fn a_newer_record_that_says_the_same_takes_the_older_ones_place() {
        // By rho: p1 (written 1 000), p2, p3 (3 000), p4 (3 500), p5 (500).
        let scene = same_thing_scene([1_000, 2_000, 3_000, 3_500, 500], [0.9, 0.8, 0.7, 0.6, 0.5]);
        assert_eq!(scene.order("product"), ["p1", "p2", "p3", "p4", "p5"]);
        // p1 kept. p2 kept (another matter). p3 says the same as p1 and is newer: it takes first
        // place. p4 says the same as p3 and is newer: it takes first place. p5 says the same as p4,
        // which is kept and newer: p5 is dropped.
        assert_eq!(
            scene.rank("product_newer_wins(tau=0.9)").unwrap(),
            pairs(&[("p4", 0.6), ("p2", 0.8)])
        );
    }

    #[test]
    fn an_older_record_that_says_the_same_as_a_kept_newer_one_is_dropped() {
        // By rho: p1 is now the newest of its group and already first.
        let scene = same_thing_scene([3_900, 2_000, 3_000, 3_500, 500], [0.9, 0.8, 0.7, 0.6, 0.5]);
        assert_eq!(scene.order("product_newer_wins(tau=0.9)"), ["p1", "p2"]);
    }

    #[test]
    fn the_bar_decides_what_says_the_same_thing() {
        let scene = same_thing_scene([1_000, 2_000, 3_000, 3_500, 500], [0.9, 0.8, 0.7, 0.6, 0.5]);
        // cos(p1, p4) = 1 / sqrt(1.04) = 0.98058; cos(p3, p4) = 1.02 / sqrt(1.01 * 1.04) = 0.99523;
        // cos(p1, p3) = 1 / sqrt(1.01) = 0.99504; cos(p4, p5) = 1.01 / sqrt(1.04 * 1.0025) = 0.98915;
        // cos(p3, p5) = 1.005 / sqrt(1.01 * 1.0025) = 0.99876; cos(p1, p5) = 0.99875.
        // At 0.999 nothing says the same thing: the product's order stands.
        assert_eq!(
            scene.order("product_newer_wins(tau=0.999)"),
            scene.order("product")
        );
        // At 0.995: p3 replaces p1 (0.99504). p4 against p3: 0.99523, replaces it. p5 against p4:
        // 0.98915, under the bar, so p5 is kept.
        assert_eq!(scene.order("product_newer_wins(tau=0.995)"), ["p4", "p2", "p5"]);
        // At -1 everything says the same thing: the newest record of all is left.
        assert_eq!(scene.order("product_newer_wins(tau=-1)"), ["p4"]);
    }

    #[test]
    fn records_written_in_the_same_second_are_both_kept() {
        let scene = same_thing_scene([3_000, 2_000, 3_000, 3_000, 3_000], [0.9, 0.8, 0.7, 0.6, 0.5]);
        assert_eq!(scene.order("product_newer_wins(tau=0.9)"), scene.order("product"));
    }

    #[test]
    fn a_record_newer_than_several_kept_ones_takes_the_first_place_and_the_others_go() {
        // q1 and q2 do not say the same thing (cos 0), r says the same as both at a bar of 0.7
        // (cos 0.7071) and is the newest; "other" is unrelated to all three.
        let records = vec![
            record("q1", 1_000),
            record("other", 1_500),
            record("q2", 2_000),
            record("r", 3_000),
        ];
        let scene = Scene::new(4_000, records)
            .now(&[("q1", 0.9), ("other", 0.8), ("q2", 0.7), ("r", 0.6)], &[])
            .relevance(&[("q1", 0.9), ("other", 0.8), ("q2", 0.7), ("r", 0.6)])
            .vectors(
                3,
                &[
                    ("q1", &[1.0, 0.0, 0.0]),
                    ("q2", &[0.0, 1.0, 0.0]),
                    ("r", &[1.0, 1.0, 0.0]),
                    ("other", &[0.0, 0.0, 1.0]),
                ],
            );
        assert_eq!(scene.order("product_newer_wins(tau=0.7)"), ["r", "other"]);
        // A record that matches a newer kept one and an older kept one is dropped; the older stays.
        let records = vec![record("q1", 1_000), record("q2", 3_500), record("r", 3_000)];
        let scene = Scene::new(4_000, records)
            .now(&[("q1", 0.9), ("q2", 0.7), ("r", 0.6)], &[])
            .relevance(&[("q1", 0.9), ("q2", 0.7), ("r", 0.6)])
            .vectors(2, &[("q1", &[1.0, 0.0]), ("q2", &[0.0, 1.0]), ("r", &[1.0, 1.0])]);
        assert_eq!(scene.order("product_newer_wins(tau=0.7)"), ["q1", "q2"]);
    }

    #[test]
    fn newer_wins_needs_a_vector_for_every_record_it_compares() {
        let no_vectors = scene(&[("a", 0.30), ("b", 0.50), ("c", 0.11), ("d", 0.90)]);
        assert!(matches!(
            no_vectors.rank("product_newer_wins(tau=0.9)"),
            Err(EvalError::VectorsNotLoaded { .. })
        ));
        let one_missing = scene(&[("a", 0.30), ("b", 0.50), ("c", 0.11), ("d", 0.90)])
            .vectors(1, &[("a", &[1.0]), ("d", &[1.0])]);
        assert!(matches!(
            one_missing.rank("product_newer_wins(tau=0.9)"),
            Err(EvalError::MissingVector { record, .. }) if record == "b"
        ));
        // A single record is compared with nothing and needs no vector.
        assert_eq!(no_vectors.order("product_newer_wins(tau=0.9,depth=1)"), ["b"]);
    }

    fn incident() -> Vec<Record> {
        vec![
            threaded("i1.s1", 1_000, "i1"),
            threaded("i1.s2", 2_000, "i1"),
            threaded("i1.s3", 3_000, "i1"),
            threaded("i2.s1", 1_500, "i2"),
            standing("doc", 100),
        ]
    }

    #[test]
    fn keyed_latest_keeps_only_the_newest_visible_record_of_each_thread() {
        let rho = [
            ("i1.s1", 0.9),
            ("i1.s2", 0.8),
            ("i1.s3", 0.5),
            ("i2.s1", 0.6),
            ("doc", 0.3),
        ];
        let scene = Scene::new(4_000, incident())
            .now(
                &[("i1.s1", 0.9), ("i1.s2", 0.8), ("doc", 0.7), ("i1.s3", 0.6)],
                &[("i1.s2", 5.0), ("i2.s1", 4.0)],
            )
            .relevance(&rho);
        assert_eq!(
            scene.order("product"),
            ["i1.s1", "i1.s2", "i2.s1", "i1.s3", "doc"]
        );
        // i1's newest record is s3; i2 has one record; doc has no thread and always stays.
        assert_eq!(
            scene.rank("keyed_latest").unwrap(),
            pairs(&[("i2.s1", 0.6), ("i1.s3", 0.5), ("doc", 0.3)])
        );
        // Asked at 2 500, before s3 was written, the newest record of i1 is s2.
        let earlier = Scene::new(2_500, incident())
            .now(&[("i1.s1", 0.9), ("i1.s2", 0.8), ("doc", 0.7)], &[("i1.s2", 5.0)])
            .relevance(&[("i1.s1", 0.9), ("i1.s2", 0.8), ("doc", 0.3)]);
        assert_eq!(earlier.order("keyed_latest"), ["i1.s2", "doc"]);
    }

    #[test]
    fn keyed_latest_drops_a_thread_whose_newest_record_was_not_retrieved() {
        // s3 exists and is visible but neither ranker returned it: s1 is not the newest and goes.
        let scene = Scene::new(4_000, incident())
            .now(&[("i1.s1", 0.9), ("doc", 0.7)], &[])
            .relevance(&[("i1.s1", 0.9), ("doc", 0.3)]);
        assert_eq!(scene.order("keyed_latest"), ["doc"]);
        // The removal happens before the merge: with depth 1 the product keeps s1, keyed keeps doc.
        assert_eq!(scene.order("product(depth=1)"), ["i1.s1"]);
        assert_eq!(scene.order("keyed_latest(depth=1)"), ["doc"]);
    }

    #[test]
    fn the_oracle_is_the_gold_order() {
        let scene = Scene::new(4_000, incident()).now(&[("doc", 0.9)], &[]).gold(&[
            ("i1.s1", 1),
            ("i1.s3", 3),
            ("i1.s2", 1),
            ("doc", 0),
        ]);
        // Grade 3 first; the two grade-1 records by newer first; a grade of 0 is not an answer.
        assert_eq!(
            scene.rank("oracle").unwrap(),
            pairs(&[("i1.s3", 3.0), ("i1.s2", 1.0), ("i1.s1", 1.0)])
        );
    }
}
