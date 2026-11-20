//! Published alternatives to the paper's formula (DESIGN.md section 4.3).

use crate::context::QuestionContext;
use crate::error::EvalError;
use crate::methods::Ranked;
use crate::methods::fusion::{
    half_life_factor, min_max_scaled, order_by_score_then_newer_then_id, weighted_reciprocal_rank_fusion,
};
use crate::records::RecordIdx;
use std::collections::HashMap;

/// `rrf_add(beta, h)`: `rrf_k([D, S]) + beta * (1 / (k + 1)) * 2^(-age / h)`,
/// the additive pattern of Qdrant's documentation.
///
/// The fusion is the plain one (each list weighs 1), so `1 / (k + 1)` is what
/// first place in one list is worth and `beta` is the size of the recency
/// bonus in those units. There is no floor: an old record's bonus goes to 0.
pub fn rrf_plus_recency_bonus(
    ctx: &QuestionContext<'_>,
    beta: f64,
    half_life_s: u64,
    k: u32,
) -> Result<Vec<Ranked>, EvalError> {
    let now = ctx.now()?;
    let dense: Vec<RecordIdx> = now.dense.iter().map(|l| l.record).collect();
    let sparse: Vec<RecordIdx> = now.sparse.iter().map(|l| l.record).collect();
    let first_place = 1.0 / (f64::from(k) + 1.0);
    let mut rows: Vec<Ranked> = weighted_reciprocal_rank_fusion(&[(&dense, 1.0), (&sparse, 1.0)], k)
        .into_iter()
        .map(|(record, fused)| Ranked {
            record,
            score: fused + beta * first_place * half_life_factor(ctx.age_of(record), half_life_s),
        })
        .collect();
    order_by_score_then_newer_then_id(&mut rows, ctx.records());
    Ok(rows)
}

/// `tw_cs_rrf(lambda)`: the sum over the lists holding `r` of
/// `(lambda * mm(score) + (1 - lambda) * recency(r)) / (k + rank)`
/// (Rollings, Rankel and Oard, TREC 2022).
///
/// `mm` scales a list's scores to 0–1 within that list. `recency` scales
/// `written_at` linearly to 0–1 over the records of either list: 1 for the
/// newest, 0 for the oldest, 1 for all when they share one moment.
pub fn time_and_score_weighted_rrf(
    ctx: &QuestionContext<'_>,
    lambda: f64,
    k: u32,
) -> Result<Vec<Ranked>, EvalError> {
    let records = ctx.records();
    let now = ctx.now()?;
    let times = || {
        now.dense
            .iter()
            .chain(&now.sparse)
            .map(|l| records.written_at(l.record))
    };
    let (Some(oldest), Some(newest)) = (times().min(), times().max()) else {
        return Ok(Vec::new());
    };
    let recency = |record: RecordIdx| {
        if newest > oldest {
            (records.written_at(record) - oldest) as f64 / (newest - oldest) as f64
        } else {
            1.0
        }
    };
    let mut row_of: HashMap<RecordIdx, usize> = HashMap::new();
    let mut rows: Vec<Ranked> = Vec::new();
    for list in [&now.dense, &now.sparse] {
        for ((position, listed), scaled) in list.iter().enumerate().zip(min_max_scaled(list)) {
            let term = (lambda * scaled + (1.0 - lambda) * recency(listed.record))
                / (f64::from(k) + (position + 1) as f64);
            let row = *row_of.entry(listed.record).or_insert_with(|| {
                rows.push(Ranked {
                    record: listed.record,
                    score: 0.0,
                });
                rows.len() - 1
            });
            rows[row].score += term;
        }
    }
    order_by_score_then_newer_then_id(&mut rows, records);
    Ok(rows)
}

/// `newest_first(n)`: the first `n` records of `D`, newest first (Li and
/// Croft, 2003). A record's score is its `written_at`, the key it is ordered by.
pub fn first_dense_by_recency(ctx: &QuestionContext<'_>, n: usize) -> Result<Vec<Ranked>, EvalError> {
    let records = ctx.records();
    let mut rows: Vec<Ranked> = ctx
        .now()?
        .dense
        .iter()
        .take(n)
        .map(|listed| Ranked {
            record: listed.record,
            score: records.written_at(listed.record) as f64,
        })
        .collect();
    order_by_score_then_newer_then_id(&mut rows, records);
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
            record("e", 4_000),
        ]
    }

    #[test]
    fn rrf_add_adds_a_bonus_worth_beta_first_places_to_a_fresh_record() {
        // Asked at 4 000, h = 1 000 s, k = 60, beta = 0.5.
        let scene = Scene::new(4_000, records())
            .now(&[("a", 0.9), ("b", 0.8), ("e", 0.7)], &[("b", 5.0), ("a", 4.0)]);
        let ranked = scene.rank("rrf_add(beta=0.5,h=1000)").unwrap();
        let bonus = |half_lives: i32| 0.5 * (1.0 / 61.0) * 2f64.powi(-half_lives);
        let expected = [
            // b: 1/62 + 1/61 = 0.032522475…, 2 half-lives old: + 0.5 * (1/61) * 0.25 = 0.002049180… → 0.034571655…
            ("b", 1.0 / 62.0 + 1.0 / 61.0 + bonus(2)),
            // a: 1/61 + 1/62 = 0.032522475…, 3 half-lives old: + 0.5 * (1/61) * 0.125 = 0.001024590… → 0.033547065…
            ("a", 1.0 / 61.0 + 1.0 / 62.0 + bonus(3)),
            // e: 1/63 = 0.015873016…, written this second: + 0.5 * (1/61) * 1 = 0.008196721… → 0.024069737…
            ("e", 1.0 / 63.0 + bonus(0)),
        ];
        for ((id, score), (expected_id, expected_score)) in ranked.iter().zip(expected) {
            assert_eq!(id, expected_id);
            assert!(
                (score - expected_score).abs() < 1e-15,
                "{id}: {score} vs {expected_score}"
            );
        }
        assert!((ranked[0].1 - 0.03457165520888419).abs() < 1e-12);
        assert!((ranked[2].1 - 0.02406973718449128).abs() < 1e-12);
    }

    #[test]
    fn rrf_add_with_no_bonus_is_plain_fusion_and_a_large_bonus_puts_the_fresh_record_first() {
        let scene = Scene::new(4_000, records())
            .now(&[("a", 0.9), ("b", 0.8), ("e", 0.7)], &[("b", 5.0), ("a", 4.0)]);
        // beta = 0: the order of rrf (the weights differ by a constant factor only). a and b tie at
        // 1/61 + 1/62 and b is newer.
        assert_eq!(scene.order("rrf_add(beta=0,h=1000)"), scene.order("rrf(k=60)"));
        assert_eq!(scene.order("rrf_add(beta=0,h=1000)"), ["b", "a", "e"]);
        // beta = 3: e gets 1/63 + 3/61 = 0.065053, b gets 0.032522 + 3 * (1/61) * 0.25 = 0.044818.
        assert_eq!(scene.order("rrf_add(beta=3,h=1000)"), ["e", "b", "a"]);
        // The fusion constant is a parameter: with k = 10, e gets 1/13 + 3/11 = 0.349650.
        let k10 = scene.rank("rrf_add(beta=3,h=1000,k=10)").unwrap();
        assert_eq!(k10[0].0, "e");
        assert!((k10[0].1 - (1.0 / 13.0 + 3.0 / 11.0)).abs() < 1e-15);
    }

    #[test]
    fn tw_cs_rrf_mixes_scaled_score_and_linear_recency_inside_each_term() {
        // Asked at 4 000. Candidates a (1 000), b (2 000), e (4 000): recency = (t - 1000) / 3000
        // = 0, 1/3, 1.  D: a 0.9, b 0.7, e 0.5 → mm 1, 0.5, 0.  S: e 8, a 2 → mm 1, 0.
        let scene = Scene::new(4_000, records())
            .now(&[("a", 0.9), ("b", 0.7), ("e", 0.5)], &[("e", 8.0), ("a", 2.0)]);
        let ranked = scene.rank("tw_cs_rrf(lambda=0.9)").unwrap();
        // e: D rank 3: (0.9*0 + 0.1*1)/63 = 0.001587302; S rank 1: (0.9*1 + 0.1*1)/61 = 0.016393443 → 0.017980744…
        // a: D rank 1: (0.9*1 + 0.1*0)/61 = 0.014754098; S rank 2: (0.9*0 + 0.1*0)/62 = 0 → 0.014754098…
        // b: D rank 2: (0.9*0.5 + 0.1*(1/3))/62 = 0.483333333/62 = 0.007795699…
        let expected = [
            ("e", 0.1 / 63.0 + 1.0 / 61.0),
            ("a", 0.9 / 61.0),
            ("b", (0.45 + 0.1 / 3.0) / 62.0),
        ];
        for ((id, score), (expected_id, expected_score)) in ranked.iter().zip(expected) {
            assert_eq!(id, expected_id);
            assert!(
                (score - expected_score).abs() < 1e-15,
                "{id}: {score} vs {expected_score}"
            );
        }
        assert!((ranked[0].1 - 0.017980744210252407).abs() < 1e-12);
        assert!((ranked[2].1 - 0.007795698924731183).abs() < 1e-12);
    }

    #[test]
    fn tw_cs_rrf_at_the_ends_of_lambda() {
        let scene = Scene::new(4_000, records())
            .now(&[("a", 0.9), ("b", 0.7), ("e", 0.5)], &[("e", 8.0), ("a", 2.0)]);
        // lambda = 1: the scaled score alone. a: 1/61 = 0.016393 (+ 0/62); e: 0/63 + 1/61 = 0.016393;
        // b: 0.5/62 = 0.008065. a and e tie and e is newer.
        assert_eq!(scene.order("tw_cs_rrf(lambda=1)"), ["e", "a", "b"]);
        // lambda = 0: recency alone. e: 1/63 + 1/61 = 0.032266; b: (1/3)/62 = 0.005376; a: 0.
        assert_eq!(scene.order("tw_cs_rrf(lambda=0)"), ["e", "b", "a"]);
        // Records written in one second all have recency 1.
        let same = Scene::new(4_000, records()).now(&[("c", 0.9), ("d", 0.5)], &[]);
        let ranked = same.rank("tw_cs_rrf(lambda=0.9)").unwrap();
        // c: (0.9*1 + 0.1*1)/61; d: (0.9*0 + 0.1*1)/62
        assert!((ranked[0].1 - 1.0 / 61.0).abs() < 1e-15);
        assert!((ranked[1].1 - 0.1 / 62.0).abs() < 1e-15);
        assert!(
            Scene::new(4_000, records())
                .now(&[], &[])
                .order("tw_cs_rrf(lambda=0.9)")
                .is_empty()
        );
    }

    #[test]
    fn newest_first_reorders_the_first_n_dense_records_by_recency() {
        let scene = Scene::new(4_000, records()).now(
            &[("a", 0.9), ("d", 0.8), ("b", 0.7), ("c", 0.6), ("e", 0.5)],
            &[("e", 9.0)],
        );
        // The first four of D are a, d, b, c. Newest first: c and d share 3 000 (c by id), then b, then a.
        // e is fifth in D and is left out, however new it is.
        assert_eq!(
            scene.rank("newest_first(n=4)").unwrap(),
            [
                ("c".to_string(), 3_000.0),
                ("d".to_string(), 3_000.0),
                ("b".to_string(), 2_000.0),
                ("a".to_string(), 1_000.0)
            ]
        );
        assert_eq!(scene.order("newest_first(n=1)"), ["a"]);
        assert_eq!(scene.order("newest_first(n=50)"), ["e", "c", "d", "b", "a"]);
    }
}
