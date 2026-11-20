//! The paper's formula (DESIGN.md section 4.2): Algorithm 1 of the first
//! paper, as printed, and its two variants.
//!
//! 1. `T` = the `list` newest records visible at `t`, taken from the whole
//!    data set (not from the candidate lists), newest first.
//! 2. `C = D[..list] ∪ S[..list] ∪ T`.
//! 3. `s(r) = rrf_k([D[..list], S[..list], T], weights)(r)`.
//! 4. `sigma` = the standard deviation of `written_at` over `C`. When
//!    `sigma > theta`, `score(r) = s(r) * decay_h(r)`; otherwise `score(r) = s(r)`.
//! 5. Order by `score` descending, then `written_at` descending, then `s`
//!    descending (then by record id, so the order is total).

use crate::context::QuestionContext;
use crate::error::EvalError;
use crate::methods::Ranked;
use crate::methods::fusion::{
    cmp_score_desc, decay_factor, population_std_of_timestamps, weighted_reciprocal_rank_fusion,
};
use crate::records::{RecordIdx, RecordTable};
use crate::spec::RrfTdParams;
use cb_types::UnixSeconds;
use std::cmp::Ordering;
use std::collections::HashSet;

/// Where the third ranker's list comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeRanker {
    /// The newest records of the whole data set: Algorithm 1 as printed.
    /// Records no relevance ranker returned enter the candidates this way.
    NewestOfDataSet,
    /// `D[..list] ∪ S[..list]` ordered by recency: time may reorder the
    /// relevance candidates but adds none (`rrf_td_reorder`).
    CandidatesByRecency,
}

/// One candidate with the three keys of step 5.
#[derive(Clone, Copy, Debug)]
struct PaperRow {
    record: RecordIdx,
    score: f64,
    written_at: UnixSeconds,
    fused: f64,
}

/// Step 5: `score` descending, then `written_at` descending, then `s` descending, then record id.
fn cmp_paper_order(a: &PaperRow, b: &PaperRow, records: &RecordTable) -> Ordering {
    cmp_score_desc(a.score, b.score)
        .then_with(|| b.written_at.cmp(&a.written_at))
        .then_with(|| cmp_score_desc(a.fused, b.fused))
        .then_with(|| records.id(a.record).cmp(records.id(b.record)))
}

/// `rrf_td` (`with_decay`), `rrf_td_reorder` (`CandidatesByRecency`) and
/// `rrf3` (`with_decay = false`: step 4 skipped).
pub fn rrf_td(
    ctx: &QuestionContext<'_>,
    params: &RrfTdParams,
    time_ranker: TimeRanker,
    with_decay: bool,
) -> Result<Vec<Ranked>, EvalError> {
    let records = ctx.records();
    let now = ctx.now()?;
    let list = params.list as usize;
    let dense: Vec<RecordIdx> = now.dense.iter().take(list).map(|l| l.record).collect();
    let sparse: Vec<RecordIdx> = now.sparse.iter().take(list).map(|l| l.record).collect();
    let by_time: Vec<RecordIdx> = match time_ranker {
        TimeRanker::NewestOfDataSet => records.newest_visible(ctx.asked_at(), list).to_vec(),
        TimeRanker::CandidatesByRecency => {
            let in_dense: HashSet<RecordIdx> = dense.iter().copied().collect();
            let mut union = dense.clone();
            union.extend(sparse.iter().copied().filter(|r| !in_dense.contains(r)));
            union.sort_by(|a, b| records.cmp_newer_then_id(*a, *b));
            union
        }
    };
    let [w_dense, w_sparse, w_time] = params.weights;
    let fused = weighted_reciprocal_rank_fusion(
        &[(&dense, w_dense), (&sparse, w_sparse), (&by_time, w_time)],
        params.k,
    );
    let times: Vec<UnixSeconds> = fused.iter().map(|(r, _)| records.written_at(*r)).collect();
    let decays = with_decay && population_std_of_timestamps(&times) > params.theta;
    let mut rows: Vec<PaperRow> = fused
        .into_iter()
        .map(|(record, fused)| {
            let gamma = if decays {
                decay_factor(ctx.age_of(record), params.h, params.eps)
            } else {
                1.0
            };
            PaperRow {
                record,
                score: fused * gamma,
                written_at: records.written_at(record),
                fused,
            }
        })
        .collect();
    rows.sort_by(|a, b| cmp_paper_order(a, b, records));
    Ok(rows
        .into_iter()
        .map(|row| Ranked {
            record: row.record,
            score: row.score,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{Scene, record};
    use cb_types::Record;

    const W: f64 = 1.0 / 3.0;

    fn position(order: &[String], id: &str) -> usize {
        order.iter().position(|x| x == id).unwrap()
    }

    #[test]
    fn the_three_rankers_are_fused_with_one_third_each() {
        // Written within 30 s of each other: the spread is sqrt((225+25+25+225)/4) = 11.2 s,
        // under theta = 60, so no decay is applied and score = s.
        let records = vec![
            record("a", 1_000),
            record("b", 1_010),
            record("c", 1_020),
            record("d", 1_030),
        ];
        let scene = Scene::new(2_000, records).now(&[("a", 0.9), ("b", 0.8)], &[("b", 5.0), ("c", 4.0)]);
        // T, newest first: d, c, b, a.
        let ranked = scene.rank("rrf_td(k=60,h=1800)").unwrap();
        let expected = [
            // b: D rank 2, S rank 1, T rank 3: (1/3)/62 + (1/3)/61 + (1/3)/63 = 0.016131830…
            ("b", W / 62.0 + W / 61.0 + W / 63.0),
            // c: S rank 2, T rank 2: (1/3)/62 + (1/3)/62 = 0.010752688…
            ("c", W / 62.0 + W / 62.0),
            // a: D rank 1, T rank 4: (1/3)/61 + (1/3)/64 = 0.010672814…
            ("a", W / 61.0 + W / 64.0),
            // d: T rank 1 only: (1/3)/61 = 0.005464480…
            ("d", W / 61.0),
        ];
        for ((id, score), (expected_id, expected_score)) in ranked.iter().zip(expected) {
            assert_eq!((id.as_str(), *score), (expected_id, expected_score));
        }
        assert!((ranked[0].1 - 0.016131830251343737).abs() < 1e-15);
        assert!((ranked[2].1 - 0.010672814207650273).abs() < 1e-15);
        assert!((ranked[3].1 - 0.00546448087431694).abs() < 1e-15);
    }

    #[test]
    fn the_decay_multiplies_the_fused_score() {
        // Asked at 3 600, h = 1 800 s. Spread of {0, 1800, 3600} = sqrt(2 * 1800^2 / 3) = 1 469.7 s > 60.
        let records = vec![record("a", 0), record("b", 1_800), record("c", 3_600)];
        let scene = Scene::new(3_600, records).now(&[("a", 0.9), ("b", 0.8), ("c", 0.7)], &[]);
        // T, newest first: c, b, a.
        let ranked = scene.rank("rrf_td(k=60,h=1800)").unwrap();
        let expected = [
            // c: D rank 3, T rank 1, age 0: ((1/3)/63 + (1/3)/61) * 1 = 0.010755486…
            ("c", W / 63.0 + W / 61.0),
            // b: D rank 2, T rank 2, one half-life old: ((1/3)/62 + (1/3)/62) * 0.5 = 0.005376344…
            ("b", (W / 62.0 + W / 62.0) * 0.5),
            // a: D rank 1, T rank 3, two half-lives old: ((1/3)/61 + (1/3)/63) * 0.25 = 0.002688871…
            ("a", (W / 61.0 + W / 63.0) * 0.25),
        ];
        for ((id, score), (expected_id, expected_score)) in ranked.iter().zip(expected) {
            assert_eq!((id.as_str(), *score), (expected_id, expected_score));
        }
        assert!((ranked[2].1 - 0.002688871541330558).abs() < 1e-15);
    }

    /// The supplement's own example (section 1.4): one ranker with unit weight, an
    /// older record at rank 1 and a newer one at rank 5.
    fn rank_1_old_against_rank_5_new(gap_s: i64, half_life_s: u64) -> Vec<String> {
        let records = vec![
            record("old", 99_000 - gap_s),
            record("f2", 50_000),
            record("f3", 50_000),
            record("f4", 50_000),
            record("new", 99_000),
        ];
        Scene::new(100_000, records)
            .now(
                &[("old", 0.9), ("f2", 0.8), ("f3", 0.7), ("f4", 0.6), ("new", 0.5)],
                &[],
            )
            .order(&format!("rrf_td(k=60,h={half_life_s},w=1:0:0)"))
    }

    #[test]
    fn a_newer_record_at_rank_5_overtakes_rank_1_when_the_gap_exceeds_h_log2_65_over_61() {
        // old: (1/61) * 2^(-(1000 + gap)/h)    new: (1/65) * 2^(-1000/h)
        // new > old  <=>  2^(gap/h) > 65/61  <=>  gap > h * log2(65/61).
        // log2(65/61) = 0.0916305, so for h = 1 800 s the threshold is 164.93 s.
        let threshold = 1_800.0 * (65.0f64 / 61.0).log2();
        assert!(164.0 < threshold && threshold < 165.0, "{threshold}");
        // 164 s: (65/61) * 2^(-164/1800) = 1.065574 * 0.938800 = 1.00036 > 1, rank 1 stays ahead.
        assert_eq!(
            rank_1_old_against_rank_5_new(164, 1_800),
            ["old", "new", "f2", "f3", "f4"]
        );
        // 165 s: (65/61) * 2^(-165/1800) = 1.065574 * 0.938438 = 0.99997 < 1, the newer record wins.
        assert_eq!(
            rank_1_old_against_rank_5_new(165, 1_800),
            ["new", "old", "f2", "f3", "f4"]
        );
        // The threshold scales with the half-life: for h = 86 400 s it is 7 916.9 s.
        let threshold = 86_400.0 * (65.0f64 / 61.0).log2();
        assert!(7_916.0 < threshold && threshold < 7_917.0, "{threshold}");
        assert_eq!(rank_1_old_against_rank_5_new(7_916, 86_400)[..2], ["old", "new"]);
        assert_eq!(rank_1_old_against_rank_5_new(7_917, 86_400)[..2], ["new", "old"]);
    }

    #[test]
    fn the_same_threshold_holds_with_the_papers_weights_once_twenty_newer_records_fill_the_time_ranker() {
        // Twenty records written in the last 20 seconds fill T, so neither "old" nor "new" is in it:
        // s(old) = (1/3)/61 and s(new) = (1/3)/65, the same ratio 65/61 as with one ranker.
        let scene = |gap_s: i64| {
            let mut records = vec![
                record("old", 99_000 - gap_s),
                record("f2", 50_000),
                record("f3", 50_000),
                record("f4", 50_000),
                record("new", 99_000),
            ];
            records.extend((1..=20).map(|i| record(&format!("fresh-{i:02}"), 99_980 + i)));
            Scene::new(100_000, records)
                .now(
                    &[("old", 0.9), ("f2", 0.8), ("f3", 0.7), ("f4", 0.6), ("new", 0.5)],
                    &[],
                )
                .order("rrf_td(k=60,h=1800)")
        };
        let below = scene(164);
        assert!(position(&below, "old") < position(&below, "new"));
        let above = scene(165);
        assert!(position(&above, "new") < position(&above, "old"));
        // Algorithm 1 as printed takes T from the whole data set: the twenty newest records,
        // which no relevance ranker returned, fill the first twenty places.
        assert!(above[..20].iter().all(|id| id.starts_with("fresh-")), "{above:?}");
        assert_eq!(above.len(), 25);
    }

    /// Two records `gap_s` apart, the older at rank 1 of the only weighted ranker and 200 s old.
    fn two_records(gap_s: i64, method: &str) -> Vec<String> {
        let records = vec![record("old", 10_000), record("new", 10_000 + gap_s)];
        Scene::new(10_200, records)
            .now(&[("old", 0.9), ("new", 0.8)], &[])
            .order(method)
    }

    #[test]
    fn no_decay_is_applied_while_the_spread_is_at_or_under_theta() {
        // With h = 60 s any gap over 60 * log2(62/61) = 1.4 s would let the decay put "new" first.
        // Two records 120 s apart have a spread of exactly 60 s, which is not above theta = 60:
        // cold start, score = s, rank 1 stays first (1/61 > 1/62).
        assert_eq!(two_records(120, "rrf_td(k=60,h=60,w=1:0:0)"), ["old", "new"]);
        // 121 s apart the spread is 60.5 s: the decay applies and "new" wins,
        // (1/62) * 2^(-79/60) = 0.00648 against (1/61) * 2^(-200/60) = 0.00163.
        assert_eq!(two_records(121, "rrf_td(k=60,h=60,w=1:0:0)"), ["new", "old"]);
        // theta is a parameter: at 0 the decay applies to any spread above zero…
        assert_eq!(
            two_records(120, "rrf_td(k=60,h=60,w=1:0:0,theta=0)"),
            ["new", "old"]
        );
        // …and at 1 000 s it stays off.
        assert_eq!(
            two_records(121, "rrf_td(k=60,h=60,w=1:0:0,theta=1000)"),
            ["old", "new"]
        );
    }

    fn mixed_scene(asked_at: i64, records: Vec<Record>) -> Scene {
        Scene::new(asked_at, records).now(
            &[("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)],
            &[("d", 9.0), ("e", 8.0), ("a", 7.0)],
        )
    }

    #[test]
    fn under_cold_start_the_order_is_that_of_three_rankers_without_decay() {
        // All five written within 40 s: spread 14.1 s, under theta.
        let records: Vec<Record> = ["a", "b", "c", "d", "e"]
            .iter()
            .zip(0..)
            .map(|(id, i)| record(id, 5_000 + 10 * i))
            .collect();
        let scene = mixed_scene(9_000, records);
        assert_eq!(
            scene.rank("rrf_td(k=60,h=60)").unwrap(),
            scene.rank("rrf3(k=60)").unwrap()
        );
        assert_eq!(
            scene.rank("rrf_td(k=10,h=300)").unwrap(),
            scene.rank("rrf3(k=10)").unwrap()
        );
    }

    #[test]
    fn with_every_record_past_the_floor_the_order_is_that_of_three_rankers_without_decay() {
        // h = 60 s and every record at least 100 000 s (1 666 half-lives) old: 2^(-age/h) is far below
        // eps = 1e-6 for all of them, so score = 1e-6 * s. The spread (hours) is far above theta.
        let records: Vec<Record> = ["a", "b", "c", "d", "e"]
            .iter()
            .zip(0..)
            .map(|(id, i)| record(id, 100_000 + 50_000 * i))
            .collect();
        let scene = mixed_scene(1_000_000, records);
        let decayed = scene.rank("rrf_td(k=60,h=60)").unwrap();
        let plain = scene.rank("rrf3(k=60)").unwrap();
        assert_eq!(decayed.len(), 5);
        for ((id, score), (plain_id, plain_score)) in decayed.iter().zip(&plain) {
            assert_eq!(id, plain_id);
            assert_eq!(*score, plain_score * 1e-6);
        }
        // The same holds for the whole k grid.
        for k in crate::spec::PAPER_K_GRID {
            assert_eq!(
                scene.order(&format!("rrf_td(k={k},h=60)")),
                scene.order(&format!("rrf3(k={k})"))
            );
        }
        // A floor of zero is no floor: every score underflows to 0 and only the tie-break is left
        // (newest first), which is not the order of the fused scores.
        let no_floor = scene.rank("rrf_td(k=60,h=60,eps=0)").unwrap();
        assert!(no_floor.iter().all(|(_, score)| *score == 0.0));
        assert_eq!(scene.order("rrf_td(k=60,h=60,eps=0)"), ["e", "d", "c", "b", "a"]);
    }

    #[test]
    fn the_floor_is_reached_between_19_and_20_half_lives() {
        // h = 60 s, one ranker. "young" is 19 half-lives old: 2^-19 = 1.907e-6 > 1e-6, no floor.
        // "aged" is 20 half-lives old: 2^-20 = 9.54e-7 < 1e-6, held at 1e-6.
        let records = vec![
            record("young", 100_000 - 19 * 60),
            record("aged", 100_000 - 20 * 60),
        ];
        let scene = Scene::new(100_000, records).now(&[("aged", 0.9), ("young", 0.8)], &[]);
        let ranked = scene.rank("rrf_td(k=60,h=60,w=1:0:0,theta=0)").unwrap();
        assert_eq!(
            ranked[0],
            ("young".to_string(), (1.0 / 62.0) * 1.9073486328125e-6)
        );
        assert_eq!(ranked[1], ("aged".to_string(), (1.0 / 61.0) * 1e-6));
    }

    #[test]
    fn equal_scores_go_to_the_newer_record_then_to_the_higher_fused_score_then_to_the_id() {
        // Dense and keyword rankers weigh the same and the time ranker nothing: x and y both get 1/61.
        let records = vec![
            record("x", 5_000),
            record("y", 5_030),
            record("b", 5_010),
            record("a", 5_010),
        ];
        let scene = Scene::new(9_000, records).now(&[("x", 0.9), ("b", 0.8)], &[("y", 3.0), ("a", 2.0)]);
        // y is newer than x; a and b share a second and go by id.
        assert_eq!(scene.order("rrf_td(k=60,h=1800,w=1:1:0)"), ["y", "x", "a", "b"]);

        let table = RecordTable::from_records(&[record("m", 1), record("n", 1)]).unwrap();
        let row = |id: &str, score: f64, written_at: i64, fused: f64| PaperRow {
            record: table.index_of(id).unwrap(),
            score,
            written_at,
            fused,
        };
        let cmp = |a: &PaperRow, b: &PaperRow| cmp_paper_order(a, b, &table);
        // score first
        assert_eq!(
            cmp(&row("n", 0.2, 1, 0.1), &row("m", 0.1, 9, 0.9)),
            Ordering::Less
        );
        // then written_at, newer first
        assert_eq!(
            cmp(&row("n", 0.2, 9, 0.1), &row("m", 0.2, 1, 0.9)),
            Ordering::Less
        );
        // then the fused score before decay, higher first
        assert_eq!(
            cmp(&row("n", 0.2, 9, 0.9), &row("m", 0.2, 9, 0.1)),
            Ordering::Less
        );
        // then the id
        assert_eq!(
            cmp(&row("m", 0.2, 9, 0.9), &row("n", 0.2, 9, 0.9)),
            Ordering::Less
        );
        assert_eq!(
            cmp(&row("m", 0.2, 9, 0.9), &row("m", 0.2, 9, 0.9)),
            Ordering::Equal
        );
    }

    #[test]
    fn the_time_ranker_reads_the_whole_data_set_but_the_reorder_variant_adds_nothing() {
        // "fresh" is in neither relevance list; "future" is written after the question.
        let records = vec![
            record("a", 0),
            record("b", 1_800),
            record("c", 3_600),
            record("fresh", 3_599),
            record("future", 5_000),
        ];
        let scene = Scene::new(3_600, records).now(&[("a", 0.9), ("b", 0.8)], &[("c", 4.0)]);

        // Algorithm 1: T = c, fresh, b, a (newest first; "future" is not visible).
        let printed = scene.rank("rrf_td(k=60,h=1800)").unwrap();
        let expected = [
            // c: S rank 1, T rank 1, age 0: (1/3)/61 + (1/3)/61 = 0.010928961…
            ("c", W / 61.0 + W / 61.0),
            // fresh: T rank 2 only, 1 s old: (1/3)/62 * 2^(-1/1800) = 0.005376344 * 0.999615 = 0.005374274…
            ("fresh", W / 62.0 * (-1.0f64 / 1_800.0).exp2()),
            // b: D rank 2, T rank 3, one half-life: ((1/3)/62 + (1/3)/63) * 0.5 = 0.005333674…
            ("b", (W / 62.0 + W / 63.0) * 0.5),
            // a: D rank 1, T rank 4, two half-lives: ((1/3)/61 + (1/3)/64) * 0.25 = 0.002668203…
            ("a", (W / 61.0 + W / 64.0) * 0.25),
        ];
        assert_eq!(printed.len(), 4);
        for ((id, score), (expected_id, expected_score)) in printed.iter().zip(expected) {
            assert_eq!((id.as_str(), *score), (expected_id, expected_score));
        }
        assert!((printed[1].1 - 0.005374274).abs() < 1e-9);

        // Reorder variant: T = the candidates by recency = c, b, a. Nothing is added.
        let reorder = scene.rank("rrf_td_reorder(k=60,h=1800)").unwrap();
        let expected = [
            ("c", W / 61.0 + W / 61.0),
            // b: D rank 2, T rank 2: ((1/3)/62 + (1/3)/62) * 0.5
            ("b", (W / 62.0 + W / 62.0) * 0.5),
            // a: D rank 1, T rank 3: ((1/3)/61 + (1/3)/63) * 0.25
            ("a", (W / 61.0 + W / 63.0) * 0.25),
        ];
        assert_eq!(reorder.len(), 3);
        for ((id, score), (expected_id, expected_score)) in reorder.iter().zip(expected) {
            assert_eq!((id.as_str(), *score), (expected_id, expected_score));
        }
    }

    #[test]
    fn each_ranker_contributes_its_first_list_records_only() {
        let records = vec![
            record("a", 100),
            record("b", 200),
            record("c", 300),
            record("d", 400),
            record("e", 500),
        ];
        let scene =
            Scene::new(1_000, records).now(&[("a", 0.9), ("b", 0.8), ("c", 0.7)], &[("b", 3.0), ("a", 2.0)]);
        // list = 1: D[..1] = a, S[..1] = b, T = the newest record, e. c and d are not candidates.
        let mut small = scene.order("rrf3(k=60,list=1)");
        small.sort();
        assert_eq!(small, ["a", "b", "e"]);
        // list = 2: D[..2] = a, b; S[..2] = b, a; T = e, d.
        let mut two = scene.order("rrf3(k=60,list=2)");
        two.sort();
        assert_eq!(two, ["a", "b", "d", "e"]);
        // The paper's list of 20 takes everything here.
        assert_eq!(scene.order("rrf3(k=60)").len(), 5);
    }
}
