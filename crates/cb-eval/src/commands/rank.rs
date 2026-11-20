//! `cb-eval rank`: run ranking methods over every question of a run.
//!
//! Reads `records.jsonl` and `queries.jsonl` of a data set and
//! `candidates.jsonl` (and, for `product_newer_wins`, `vectors.f32` with
//! `vectors.json`; for a method with `recent` above 0, `newest.jsonl`) of a
//! run. Writes, per method, `rankings/<name>.jsonl` (`cb_types::Ranking`, one
//! line per question) and `measures/<name>.json` ([`MethodScores`]: the
//! method, the hashes of the inputs, the measures of every question and their
//! means per split and group).

use crate::aggregate::{GroupMean, QuestionScore, means_by_split_and_group};
use crate::context::{PassageVectors, QuestionContext, View};
use crate::error::EvalError;
use crate::files::{
    CANDIDATES, CandidateFile, LineFile, MEASURES_DIR, NEWEST, QUERIES, RANKINGS_DIR, RECORDS, VECTORS,
    made_by, read_jsonl_file, read_vectors, sha256_of_file, write_json_file,
};
use crate::measures::{DEEPEST_CUT, Measures, QuestionKey, ScoringKey, score_answer};
use crate::methods;
use crate::methods::product::PRODUCT_PREFETCH;
use crate::records::{RecordIdx, RecordTable};
use crate::spec::MethodSpec;
use cb_types::{Candidates, Newest, Query, Ranking, Record, Scored};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// How many records of each answer are written to `rankings/` unless asked
/// otherwise: as deep as the deepest measure looks.
pub const DEFAULT_KEEP: usize = DEEPEST_CUT;

#[derive(Clone, Debug)]
pub struct RankArgs {
    /// The data set directory.
    pub data: PathBuf,
    /// The run directory.
    pub run: PathBuf,
    pub methods: Vec<MethodSpec>,
    /// How many records of each answer to write; 0 writes all. Measures are
    /// always computed on the whole answer.
    pub keep: usize,
}

/// One method's measures on every question of a run: `measures/<name>.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethodScores {
    /// The method's file-safe name.
    pub method: String,
    /// The method's command-line form.
    pub text: String,
    pub spec: MethodSpec,
    pub made_by: String,
    /// File name → sha256 of every input the scores were computed from.
    pub inputs: BTreeMap<String, String>,
    /// Means per split and group, for a quick look; reports recompute them.
    pub means: Vec<GroupMean>,
    /// One entry per question, in the order of `queries.jsonl`.
    pub questions: Vec<QuestionScore>,
}

/// The path of a method's scores inside a run directory.
pub fn scores_path(run: &Path, method_name: &str) -> PathBuf {
    run.join(MEASURES_DIR).join(format!("{method_name}.json"))
}

/// What `rank` reports back for each method.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedMethod {
    pub name: String,
    pub text: String,
    pub questions: usize,
    pub means: Vec<GroupMean>,
}

/// What `rank` reports back: each method, and the records the product's methods left
/// out because the relevance pass never scored them ([`QuestionContext::unscored_in_first`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RankReport {
    pub methods: Vec<RankedMethod>,
    /// Over all questions, records in the first 64 places of a `now` list without a score.
    pub unscored_records: usize,
    /// The questions that had any.
    pub questions_with_unscored: usize,
}

/// Rank every question with every method and write the rankings and scores.
pub fn run(args: &RankArgs) -> Result<RankReport, EvalError> {
    let mut methods: Vec<(MethodSpec, String)> = Vec::new();
    for spec in &args.methods {
        spec.validate()?;
        let name = spec.name();
        if !methods.iter().any(|(_, known)| *known == name) {
            methods.push((*spec, name));
        }
    }

    let records_path = args.data.join(RECORDS);
    let queries_path = args.data.join(QUERIES);
    let candidates_path = args.run.join(CANDIDATES);
    let records: Vec<Record> = read_jsonl_file(&records_path)?;
    let queries: Vec<Query> = read_jsonl_file(&queries_path)?;
    let table = RecordTable::from_records(&records)?;
    let scoring = ScoringKey::new(&records, &table)?;
    drop(records);

    let mut known = HashSet::with_capacity(queries.len());
    for query in &queries {
        if !known.insert(query.id.as_str()) {
            return Err(EvalError::invalid(format!(
                "{}: question id {:?} appears twice",
                queries_path.display(),
                query.id
            )));
        }
    }
    let mut candidates = CandidateFile::open(&candidates_path)?;
    if let Some(stranger) = candidates.questions().find(|q| !known.contains(q)) {
        return Err(EvalError::invalid(format!(
            "{}: question {stranger:?} is not in {}",
            candidates_path.display(),
            queries_path.display()
        )));
    }

    let mut inputs = BTreeMap::new();
    for (name, path) in [
        (RECORDS, &records_path),
        (QUERIES, &queries_path),
        (CANDIDATES, &candidates_path),
    ] {
        inputs.insert(name.to_string(), sha256_of_file(path)?);
    }
    let vectors = if methods.iter().any(|(spec, _)| spec.needs_vectors()) {
        let stem = args.run.join(VECTORS);
        for extension in ["f32", "json"] {
            let path = stem.with_extension(extension);
            inputs.insert(format!("{VECTORS}.{extension}"), sha256_of_file(&path)?);
        }
        Some(PassageVectors::new(read_vectors(&stem)?, &table)?)
    } else {
        None
    };
    // The newest lists, read only when a method reads them, so a method that does not
    // can never be moved by them; their hash goes into those methods' inputs only.
    let newest_path = args.run.join(NEWEST);
    let newest = if methods.iter().any(|(spec, _)| spec.needs_newest()) {
        let mut by_question: HashMap<String, Newest> = HashMap::new();
        for line in read_jsonl_file::<Newest>(&newest_path)? {
            if !known.contains(line.query.as_str()) {
                return Err(EvalError::invalid(format!(
                    "{}: question {:?} is not in {}",
                    newest_path.display(),
                    line.query,
                    queries_path.display()
                )));
            }
            let question = line.query.clone();
            if by_question.insert(question.clone(), line).is_some() {
                return Err(EvalError::invalid(format!(
                    "{}: question {question:?} appears twice",
                    newest_path.display()
                )));
            }
        }
        Some((by_question, sha256_of_file(&newest_path)?))
    } else {
        None
    };

    let mut ranking_files: Vec<LineFile> = methods
        .iter()
        .map(|(_, name)| LineFile::new(args.run.join(RANKINGS_DIR).join(format!("{name}.jsonl"))))
        .collect();
    let mut scores: Vec<Vec<Measures>> = vec![Vec::with_capacity(queries.len()); methods.len()];
    let (mut unscored_records, mut questions_with_unscored) = (0, 0);
    for query in &queries {
        let key = QuestionKey::new(query, &table)?;
        let lines = candidates.lines_of(&query.id)?;
        if lines.is_empty() {
            return Err(EvalError::invalid(format!(
                "{}: no candidate lists for question {:?}",
                candidates_path.display(),
                query.id
            )));
        }
        let lines: Vec<&Candidates> = lines.iter().collect();
        let newest_line = newest.as_ref().and_then(|(lines, _)| lines.get(&query.id));
        let ctx = QuestionContext::new(query, &lines, newest_line, &table, vectors.as_ref(), key.gold())?;
        if ctx.view(View::Now).is_ok() {
            let unscored = ctx.unscored_in_first(PRODUCT_PREFETCH)?;
            unscored_records += unscored;
            questions_with_unscored += usize::from(unscored > 0);
        }
        for (((spec, name), file), scores) in methods.iter().zip(&mut ranking_files).zip(&mut scores) {
            let answer = methods::rank(spec, &ctx)?;
            let order: Vec<RecordIdx> = answer.iter().map(|r| r.record).collect();
            scores.push(score_answer(&order, &key, &scoring));
            let kept = if args.keep == 0 {
                answer.len()
            } else {
                answer.len().min(args.keep)
            };
            file.push(&Ranking {
                query: query.id.clone(),
                method: name.clone(),
                ranked: answer[..kept]
                    .iter()
                    .map(|r| Scored {
                        record: table.id(r.record).to_string(),
                        score: r.score,
                    })
                    .collect(),
            })?;
        }
    }

    let mut done = Vec::with_capacity(methods.len());
    for (((spec, name), file), scores) in methods.into_iter().zip(ranking_files).zip(scores) {
        file.finish()?;
        let questions: Vec<QuestionScore> = queries
            .iter()
            .zip(scores)
            .map(|(query, measures)| QuestionScore::new(query, measures))
            .collect();
        let mut inputs = inputs.clone();
        if spec.needs_newest()
            && let Some((_, sha256)) = &newest
        {
            inputs.insert(NEWEST.to_string(), sha256.clone());
        }
        let method_scores = MethodScores {
            method: name.clone(),
            text: spec.to_string(),
            spec,
            made_by: made_by(),
            inputs,
            means: means_by_split_and_group(&questions),
            questions,
        };
        write_json_file(&scores_path(&args.run, &name), &method_scores)?;
        done.push(RankedMethod {
            name,
            text: method_scores.text,
            questions: method_scores.questions.len(),
            means: method_scores.means,
        });
    }
    Ok(RankReport {
        methods: done,
        unscored_records,
        questions_with_unscored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aggregate::Group;
    use crate::files::{read_json_file, test_dir, write_jsonl_file, write_vectors};
    use crate::testkit::{candidates, question, scored, threaded};
    use crate::vectors::VectorIndex;
    use cb_types::{QueryClass, Split};
    use std::fs;

    /// One incident (thread inc-0001, test split) with two states and one question after each.
    fn write_tiny(dir: &Path) -> (PathBuf, PathBuf) {
        let (data, run) = (dir.join("data"), dir.join("run"));
        fs::create_dir_all(&data).unwrap();
        fs::create_dir_all(&run).unwrap();
        let records = vec![threaded("s1", 100, "inc-0001"), threaded("s2", 200, "inc-0001")];
        let queries = vec![
            question("q1", 150, QueryClass::Changing, &[("s1", 3)], Some("inc-0001")),
            question(
                "q2",
                250,
                QueryClass::Changing,
                &[("s2", 3), ("s1", 1)],
                Some("inc-0001"),
            ),
        ];
        let lines = vec![
            candidates(
                "q1",
                "now",
                150,
                &scored(&[("s1", 0.9)]),
                &scored(&[("s1", 3.0)]),
                &scored(&[("s1", 0.8)]),
            ),
            candidates(
                "q2",
                "now",
                250,
                &scored(&[("s1", 0.9), ("s2", 0.8)]),
                &scored(&[("s1", 3.0), ("s2", 2.0)]),
                &scored(&[("s1", 0.8), ("s2", 0.7)]),
            ),
        ];
        write_jsonl_file(&data.join(RECORDS), &records).unwrap();
        write_jsonl_file(&data.join(QUERIES), &queries).unwrap();
        write_jsonl_file(&run.join(CANDIDATES), &lines).unwrap();
        (data, run)
    }

    fn specs(texts: &[&str]) -> Vec<MethodSpec> {
        texts.iter().map(|t| t.parse().unwrap()).collect()
    }

    #[test]
    fn rank_writes_rankings_and_scores_for_each_method() {
        let dir = test_dir("rank");
        let (data, run_dir) = write_tiny(&dir);
        let done = run(&RankArgs {
            data,
            run: run_dir.clone(),
            methods: specs(&["dense", "product", "oracle", "dense"]),
            keep: 1,
        })
        .unwrap();
        let names: Vec<&str> = done.methods.iter().map(|m| m.name.as_str()).collect();
        // Every record of the first 64 places was scored.
        assert_eq!((done.unscored_records, done.questions_with_unscored), (0, 0));
        assert_eq!(names, ["dense", "product", "oracle"]);

        // q1: dense answers s1, the answer. q2: dense answers s1 then s2; the answer s2 is second.
        let dense: MethodScores = read_json_file(&scores_path(&run_dir, "dense")).unwrap();
        assert_eq!(dense.text, "dense");
        assert_eq!(dense.questions.len(), 2);
        assert_eq!(dense.questions[0].measures.current_1, 1.0);
        assert_eq!(dense.questions[1].measures.current_1, 0.0);
        assert_eq!(dense.questions[1].measures.mrr, 0.5);
        assert_eq!(dense.questions[1].measures.stale_4, 0.25);
        // Both questions belong to thread inc-0001, which is in the test split.
        assert_eq!(dense.means.len(), 1);
        assert_eq!(
            (dense.means[0].split, dense.means[0].group),
            (Split::Test, Group::Changing)
        );
        assert_eq!(
            (dense.means[0].questions, dense.means[0].means.current_1),
            (2, 0.5)
        );
        assert_eq!(dense.inputs.len(), 3);
        assert_eq!(
            dense.inputs[CANDIDATES],
            sha256_of_file(&run_dir.join(CANDIDATES)).unwrap()
        );

        // The oracle is first on both.
        let oracle: MethodScores = read_json_file(&scores_path(&run_dir, "oracle")).unwrap();
        assert_eq!(oracle.means[0].means.current_1, 1.0);

        // keep = 1 wrote one record per answer; the measures above saw the whole answer.
        let rankings: Vec<Ranking> =
            read_jsonl_file(&run_dir.join(RANKINGS_DIR).join("dense.jsonl")).unwrap();
        assert_eq!(rankings.len(), 2);
        assert_eq!(rankings[1].method, "dense");
        assert_eq!(rankings[1].ranked, scored(&[("s1", 0.9)]));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_newest_lists_are_read_only_for_a_method_that_reads_them() {
        let dir = test_dir("rank-newest");
        let (data, run_dir) = write_tiny(&dir);
        let with_recent = || RankArgs {
            data: data.clone(),
            run: run_dir.clone(),
            methods: specs(&["product", "product(recent=1)"]),
            keep: 0,
        };
        // No newest.jsonl: a method with recent fails and names the file it needs.
        let error = run(&with_recent()).unwrap_err().to_string();
        assert!(error.contains(NEWEST), "{error}");
        // Without such a method the file is never asked for.
        assert!(
            run(&RankArgs {
                methods: specs(&["product"]),
                ..with_recent()
            })
            .is_ok()
        );
        // q1 (asked at 150) could see s1 only; q2 sees s2, newest, which no candidate list
        // scored as high as the newest pass did.
        let newest = vec![
            Newest {
                query: "q1".into(),
                as_of: 150,
                newest: scored(&[("s1", 100.0)]),
                rerank: scored(&[("s1", 0.8)]),
                timings_ms: Default::default(),
            },
            Newest {
                query: "q2".into(),
                as_of: 250,
                newest: scored(&[("s2", 200.0), ("s1", 100.0)]),
                rerank: scored(&[("s2", 0.9), ("s1", 0.8)]),
                timings_ms: Default::default(),
            },
        ];
        write_jsonl_file(&run_dir.join(NEWEST), &newest).unwrap();
        run(&with_recent()).unwrap();
        let plain: MethodScores = read_json_file(&scores_path(&run_dir, "product")).unwrap();
        let recent: MethodScores = read_json_file(&scores_path(&run_dir, "product-recent1")).unwrap();
        // product: s1 (0.8) above s2 (0.7) on q2; with the newest pass s2 scores 0.9 and is first.
        assert_eq!(plain.questions[1].measures.current_1, 0.0);
        assert_eq!(recent.questions[1].measures.current_1, 1.0);
        // The file's hash is an input of the method that read it, and of no other.
        assert!(!plain.inputs.contains_key(NEWEST));
        assert_eq!(
            recent.inputs[NEWEST],
            sha256_of_file(&run_dir.join(NEWEST)).unwrap()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rank_is_byte_for_byte_repeatable() {
        let dir = test_dir("rank-repeat");
        let (data, run_dir) = write_tiny(&dir);
        let args = RankArgs {
            data,
            run: run_dir.clone(),
            methods: specs(&["rrf_td(k=60,h=1800)", "product_decay(h=60)"]),
            keep: 0,
        };
        let read_all = || {
            [
                "rankings/rrf_td-k60-h1800.jsonl",
                "measures/rrf_td-k60-h1800.json",
                "measures/product_decay-h60.json",
            ]
            .map(|file| fs::read(run_dir.join(file)).unwrap())
        };
        run(&args).unwrap();
        let first = read_all();
        run(&args).unwrap();
        assert_eq!(first, read_all());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rank_reads_vectors_only_for_the_method_that_needs_them() {
        let dir = test_dir("rank-vectors");
        let (data, run_dir) = write_tiny(&dir);
        let args = |methods: &[&str]| RankArgs {
            data: data.clone(),
            run: run_dir.clone(),
            methods: specs(methods),
            keep: DEFAULT_KEEP,
        };
        assert!(run(&args(&["product"])).is_ok());
        assert!(matches!(
            run(&args(&["product_newer_wins(tau=0.9)"])),
            Err(EvalError::Io { .. })
        ));
        let index = VectorIndex {
            dim: 2,
            ids: vec!["s1".into(), "s2".into()],
        };
        write_vectors(&run_dir.join(VECTORS), &index, &[1.0, 0.0, 1.0, 0.01]).unwrap();
        run(&args(&["product_newer_wins(tau=0.9)"])).unwrap();
        let scores: MethodScores =
            read_json_file(&scores_path(&run_dir, "product_newer_wins-tau0.9")).unwrap();
        // q2: the product answers s1 (0.8) then s2 (0.7); s2 says the same and is newer, so it
        // takes first place and s1 goes.
        assert_eq!(scores.questions[1].measures.current_1, 1.0);
        assert_eq!(scores.questions[1].measures.stale_4, 0.0);
        assert_eq!(scores.inputs.len(), 5);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rank_refuses_a_run_that_does_not_match_the_data_set() {
        let dir = test_dir("rank-refuse");
        let (data, run_dir) = write_tiny(&dir);
        let args = RankArgs {
            data: data.clone(),
            run: run_dir.clone(),
            methods: specs(&["dense"]),
            keep: DEFAULT_KEEP,
        };
        let message = |args: &RankArgs| match run(args) {
            Err(EvalError::Invalid(message)) => message,
            other => panic!("expected a refusal, got {other:?}"),
        };
        // A question without candidate lists.
        let only_q1 = vec![candidates("q1", "now", 150, &scored(&[("s1", 0.9)]), &[], &[])];
        write_jsonl_file(&run_dir.join(CANDIDATES), &only_q1).unwrap();
        assert!(message(&args).contains("no candidate lists for question \"q2\""));
        // Candidate lists of a question the data set does not have.
        let stranger = vec![candidates("q9", "now", 150, &[], &[], &[])];
        write_jsonl_file(&run_dir.join(CANDIDATES), &stranger).unwrap();
        assert!(message(&args).contains("question \"q9\" is not in"));
        // A list holding a record that was not written yet when the question was asked.
        let future = vec![
            candidates("q1", "now", 150, &scored(&[("s2", 0.9)]), &[], &[]),
            candidates("q2", "now", 250, &[], &[], &[]),
        ];
        write_jsonl_file(&run_dir.join(CANDIDATES), &future).unwrap();
        assert!(message(&args).contains("could not see"));
        // A question id used twice.
        let twice = vec![
            question("q1", 150, QueryClass::Changing, &[("s1", 3)], None),
            question("q1", 150, QueryClass::Changing, &[("s1", 3)], None),
        ];
        write_jsonl_file(&data.join(QUERIES), &twice).unwrap();
        assert!(message(&args).contains("appears twice"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
