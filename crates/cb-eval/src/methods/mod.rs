//! The ranking methods under test (DESIGN.md section 4), each a pure function
//! of one [`QuestionContext`].
//!
//! Notation, as in DESIGN.md: a question asked at `t`; `D` and `S` are the
//! dense and keyword lists of the `now` view, best first; `rho(r)` is the
//! relevance model's score; `age(r) = t - written_at(r)`;
//! `decay_h(r) = max(2^(-age(r)/h), eps)`; `rank_L(r)` counts from 1.
//!
//! Wherever a method orders by a score, equal scores go by newer `written_at`
//! first, then by record id. A list a method returns "as given" keeps the
//! service's order.

pub mod alternatives;
pub mod baselines;
pub mod fusion;
pub mod paper;
pub mod product;

use crate::context::{QuestionContext, View};
use crate::error::EvalError;
use crate::methods::product::Pool;
use crate::records::RecordIdx;
use crate::spec::{MethodSpec, PAPER_EPS};
use std::collections::HashSet;

/// A record in a method's answer with the score the method gave it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ranked {
    pub record: RecordIdx,
    pub score: f64,
}

/// Run one method on one question. Best first.
pub fn rank(spec: &MethodSpec, ctx: &QuestionContext<'_>) -> Result<Vec<Ranked>, EvalError> {
    spec.validate()?;
    let ranked = match *spec {
        MethodSpec::Dense => baselines::dense_as_given(ctx.now()?),
        MethodSpec::Rrf { k } => baselines::rrf_of_dense_and_keywords(ctx, k)?,
        MethodSpec::DecayOnly { h } => baselines::dense_times_decay(ctx, h, PAPER_EPS)?,
        MethodSpec::Batch { period } => baselines::dense_as_given(ctx.view(View::Batch(period))?),
        MethodSpec::Cutoff { window } => {
            baselines::dense_within_window_or_all(ctx.view(View::Window(window))?, ctx.now()?)
        }
        MethodSpec::Cc { alpha } => baselines::convex_combination(ctx, alpha)?,
        MethodSpec::RrfTd(params) => paper::rrf_td(ctx, &params, paper::TimeRanker::NewestOfDataSet, true)?,
        MethodSpec::RrfTdReorder(params) => {
            paper::rrf_td(ctx, &params, paper::TimeRanker::CandidatesByRecency, true)?
        }
        MethodSpec::Rrf3 { k, weights, list } => {
            // Step 4 is skipped, so the half-life, the floor and the threshold are never read.
            let params = crate::spec::RrfTdParams {
                k,
                h: 1,
                weights,
                list,
                eps: PAPER_EPS,
                theta: 0.0,
            };
            paper::rrf_td(ctx, &params, paper::TimeRanker::NewestOfDataSet, false)?
        }
        MethodSpec::RrfAdd { beta, h, k } => alternatives::rrf_plus_recency_bonus(ctx, beta, h, k)?,
        MethodSpec::TwCsRrf { lambda, k } => alternatives::time_and_score_weighted_rrf(ctx, lambda, k)?,
        MethodSpec::NewestFirst { n } => alternatives::first_dense_by_recency(ctx, n as usize)?,
        MethodSpec::Product { depth, recent } => product::product(ctx, Pool::new(depth, recent))?,
        MethodSpec::ProductDecay { h, depth, recent } => {
            product::product_times_decay(ctx, Pool::new(depth, recent), h, PAPER_EPS)?
        }
        MethodSpec::ProductBonus {
            beta,
            h,
            depth,
            recent,
        } => product::product_with_recency_bonus(ctx, Pool::new(depth, recent), beta, h)?,
        MethodSpec::ProductEnds { depth, recent } => {
            product::product_without_ended(ctx, Pool::new(depth, recent))?
        }
        MethodSpec::ProductNewerWins { tau, depth, recent } => {
            product::product_newer_wins(ctx, Pool::new(depth, recent), tau)?
        }
        MethodSpec::KeyedLatest { depth, recent } => product::keyed_latest(ctx, Pool::new(depth, recent))?,
        MethodSpec::Oracle => product::gold_order(ctx),
    };
    check_answer(spec, ctx, &ranked)?;
    Ok(ranked)
}

/// A method's answer must hold a record at most once and only finite scores.
fn check_answer(spec: &MethodSpec, ctx: &QuestionContext<'_>, ranked: &[Ranked]) -> Result<(), EvalError> {
    let mut seen = HashSet::with_capacity(ranked.len());
    for row in ranked {
        let name = |record: RecordIdx| ctx.records().id(record).to_string();
        if !row.score.is_finite() {
            return Err(EvalError::ScoreNotFinite {
                query: ctx.query().to_string(),
                method: spec.name(),
                record: name(row.record),
            });
        }
        if !seen.insert(row.record) {
            return Err(EvalError::ListedTwice {
                query: ctx.query().to_string(),
                method: spec.name(),
                record: name(row.record),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spec::{KINDS, MethodSpec};
    use crate::testkit::{Scene, ending, record, standing, threaded};
    use cb_types::Record;

    /// One method of every kind, in text form.
    const EVERY_KIND: [&str; 19] = [
        "dense",
        "rrf(k=60)",
        "decay_only(h=1800)",
        "batch(period=300)",
        "cutoff(window=3600)",
        "cc(alpha=0.4)",
        "rrf_td(k=60,h=1800)",
        "rrf_td_reorder(k=60,h=1800)",
        "rrf3(k=60)",
        "rrf_add(beta=0.5,h=1800)",
        "tw_cs_rrf(lambda=0.9)",
        "newest_first(n=3)",
        "product",
        "product_decay(h=1800)",
        "product_bonus(beta=0.5,h=1800)",
        "product_ends",
        "product_newer_wins(tau=0.9)",
        "keyed_latest",
        "oracle",
    ];

    /// A scene every method can rank: two incidents, a help page, every view, scores and vectors.
    fn busy_scene(records: Vec<Record>) -> Scene {
        let dense = [
            ("i1.s1", 0.9),
            ("i1.s2", 0.85),
            ("doc", 0.8),
            ("i2.s1", 0.7),
            ("i1.s3", 0.6),
        ];
        let sparse = [("i1.s2", 9.0), ("i2.s1", 6.0), ("doc", 5.0)];
        Scene::new(10_000, records)
            .now(&dense, &sparse)
            .relevance(&[
                ("i1.s1", 0.7),
                ("i1.s2", 0.6),
                ("doc", 0.4),
                ("i2.s1", 0.3),
                ("i1.s3", 0.65),
            ])
            .view("batch_300", &dense, &sparse)
            .view("window_3600", &dense[..2], &sparse[..1])
            .gold(&[("i1.s3", 3), ("i1.s2", 1), ("i1.s1", 1)])
            .vectors(
                2,
                &[
                    ("i1.s1", &[1.0, 0.0]),
                    ("i1.s2", &[1.0, 0.1]),
                    ("i1.s3", &[0.0, 1.0]),
                    ("i2.s1", &[0.5, 0.5]),
                    ("doc", &[-1.0, 0.3]),
                ],
            )
    }

    fn busy_records(thread_of: impl Fn(&str) -> Option<&'static str>) -> Vec<Record> {
        let with_thread = |id: &str, written_at: i64| match thread_of(id) {
            Some(thread) => threaded(id, written_at, thread),
            None => record(id, written_at),
        };
        vec![
            with_thread("i1.s1", 7_000),
            with_thread("i1.s2", 8_000),
            with_thread("i1.s3", 9_000),
            with_thread("i2.s1", 9_500),
            standing("doc", 100),
            ending("old-news", 50, 60),
        ]
    }

    fn by_incident(id: &str) -> Option<&'static str> {
        match id.split('.').next() {
            Some("i1") => Some("i1"),
            Some("i2") => Some("i2"),
            _ => None,
        }
    }

    #[test]
    fn the_list_of_kinds_is_covered() {
        let kinds: Vec<&str> = EVERY_KIND
            .iter()
            .map(|text| text.parse::<MethodSpec>().unwrap().kind())
            .collect();
        assert_eq!(kinds, KINDS);
    }

    #[test]
    fn every_method_ranks_the_same_way_twice() {
        let scene = busy_scene(busy_records(by_incident));
        for text in EVERY_KIND {
            let first = scene.rank(text).unwrap();
            assert!(!first.is_empty(), "{text}");
            assert_eq!(first, scene.rank(text).unwrap(), "{text}");
        }
    }

    #[test]
    fn only_the_keyed_method_reads_threads() {
        let grouped = busy_scene(busy_records(by_incident));
        // The same records with every thread label removed, and with the labels swapped around.
        let unlabelled = busy_scene(busy_records(|_| None));
        let relabelled = busy_scene(busy_records(|id| match id {
            "i1.s1" | "i2.s1" => Some("x"),
            "i1.s2" => Some("y"),
            _ => None,
        }));
        for text in EVERY_KIND {
            let spec: MethodSpec = text.parse().unwrap();
            let same = grouped.rank(text).unwrap() == unlabelled.rank(text).unwrap()
                && grouped.rank(text).unwrap() == relabelled.rank(text).unwrap();
            assert_eq!(same, !spec.is_keyed(), "{text}");
        }
    }

    #[test]
    fn only_the_oracle_reads_the_answers() {
        let scene = busy_scene(busy_records(by_incident));
        let other_answers = busy_scene(busy_records(by_incident)).gold(&[("doc", 3)]);
        for text in EVERY_KIND {
            let spec: MethodSpec = text.parse().unwrap();
            let same = scene.rank(text).unwrap() == other_answers.rank(text).unwrap();
            assert_eq!(same, !spec.reads_gold(), "{text}");
        }
    }

    #[test]
    fn a_view_the_run_did_not_collect_is_an_error_not_an_empty_answer() {
        let scene = Scene::new(10_000, busy_records(by_incident)).now(&[("doc", 0.5)], &[]);
        for text in ["batch(period=300)", "cutoff(window=3600)"] {
            assert!(
                matches!(scene.rank(text), Err(crate::error::EvalError::MissingView { .. })),
                "{text}"
            );
        }
    }
}
