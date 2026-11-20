//! The scores `cb-eval rank` wrote, loaded and lined up for statistics.
//!
//! One [`Run`] holds, for one split, every question of a run directory and
//! every method's measures on those questions. Several runs (the five seeds
//! of the generated corpus) are kept apart: they are the strata of
//! [`crate::stats`].

use crate::aggregate::{Cluster, Group};
use crate::commands::rank::{MethodScores, scores_path};
use crate::error::EvalError;
use crate::files::{CANDIDATES, MEASURES_DIR, QUERIES, RECORDS, read_json_file};
use crate::measures::{Measure, Measures};
use crate::spec::MethodSpec;
use crate::stats::Stratum;
use cb_types::{QueryClass, Split};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// A question of the loaded split.
#[derive(Clone, Debug, PartialEq)]
pub struct Asked {
    pub query: String,
    pub class: QueryClass,
    pub cluster: Cluster,
}

/// One method's measures on the questions of a [`Run`], in the run's question order.
#[derive(Clone, Debug)]
pub struct LoadedMethod {
    pub text: String,
    pub spec: MethodSpec,
    pub measures: Vec<Measures>,
}

/// One run directory's scores for one split.
#[derive(Debug)]
pub struct Run {
    /// The directory's name: how the run is called in outputs.
    pub label: String,
    pub dir: PathBuf,
    pub split: Split,
    /// sha256 of the data set and candidate files every method was ranked from.
    pub inputs: BTreeMap<String, String>,
    pub questions: Vec<Asked>,
    /// By method name.
    pub methods: BTreeMap<String, LoadedMethod>,
}

/// The names of the methods a run directory holds scores for, sorted.
pub fn method_names_in(dir: &Path) -> Result<Vec<String>, EvalError> {
    let measures = dir.join(MEASURES_DIR);
    let entries = std::fs::read_dir(&measures).map_err(|source| EvalError::Io {
        path: measures.clone(),
        source,
    })?;
    let mut names = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|source| EvalError::Io {
                path: measures.clone(),
                source,
            })?
            .path();
        if path.extension().is_some_and(|e| e == "json")
            && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
        {
            names.push(stem.to_string());
        }
    }
    names.sort();
    Ok(names)
}

/// The label of a run directory: its last path component.
pub fn label_of(dir: &Path) -> String {
    dir.canonicalize()
        .ok()
        .as_deref()
        .unwrap_or(dir)
        .file_name()
        .map_or_else(
            || dir.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        )
}

/// Load the scores of `method_names` for the questions of `split`.
///
/// Refused: a method without scores, a scores file whose name is not its
/// method's name, and methods that were not ranked from the same input files
/// or do not list the same questions (they must be ranked again together).
pub fn load_run(dir: &Path, method_names: &[String], split: Split) -> Result<Run, EvalError> {
    let shared = [RECORDS, QUERIES, CANDIDATES];
    let mut run = Run {
        label: label_of(dir),
        dir: dir.to_path_buf(),
        split,
        inputs: BTreeMap::new(),
        questions: Vec::new(),
        methods: BTreeMap::new(),
    };
    let mut first: Option<String> = None;
    for name in method_names {
        let path = scores_path(dir, name);
        let scores: MethodScores = read_json_file(&path)?;
        let wrong = |what: &str| EvalError::invalid(format!("{}: {what}", path.display()));
        if scores.method != *name || scores.spec.name() != *name {
            return Err(wrong("the file's name is not the name of the method inside it"));
        }
        let inputs: BTreeMap<String, String> = scores
            .inputs
            .iter()
            .filter(|(file, _)| shared.contains(&file.as_str()))
            .map(|(file, hash)| (file.clone(), hash.clone()))
            .collect();
        let questions: Vec<Asked> = scores
            .questions
            .iter()
            .filter(|q| q.split == split)
            .map(|q| Asked {
                query: q.query.clone(),
                class: q.class,
                cluster: q.cluster.clone(),
            })
            .collect();
        match &first {
            None => {
                run.inputs = inputs;
                run.questions = questions;
                first = Some(name.clone());
            }
            Some(first) => {
                if inputs != run.inputs {
                    return Err(wrong(&format!(
                        "ranked from other input files than {first}; rank the methods again together"
                    )));
                }
                if questions != run.questions {
                    return Err(wrong(&format!(
                        "does not hold the same questions as {first}; rank the methods again together"
                    )));
                }
            }
        }
        run.methods.insert(
            name.clone(),
            LoadedMethod {
                text: scores.text,
                spec: scores.spec,
                measures: scores
                    .questions
                    .iter()
                    .filter(|q| q.split == split)
                    .map(|q| q.measures)
                    .collect(),
            },
        );
    }
    Ok(run)
}

/// Load the same methods from several run directories.
pub fn load_runs(dirs: &[PathBuf], method_names: &[String], split: Split) -> Result<Vec<Run>, EvalError> {
    let runs: Vec<Run> = dirs
        .iter()
        .map(|dir| load_run(dir, method_names, split))
        .collect::<Result<_, _>>()?;
    for (i, run) in runs.iter().enumerate() {
        if runs[..i].iter().any(|earlier| earlier.label == run.label) {
            return Err(EvalError::invalid(format!(
                "two run directories are both called {:?}; give them different names",
                run.label
            )));
        }
    }
    Ok(runs)
}

/// The method names every run directory holds scores for; refused when the directories differ.
pub fn methods_common_to(dirs: &[PathBuf]) -> Result<Vec<String>, EvalError> {
    let mut common: Option<(Vec<String>, &PathBuf)> = None;
    for dir in dirs {
        let names = method_names_in(dir)?;
        match &common {
            None => common = Some((names, dir)),
            Some((expected, first)) if *expected != names => {
                let only = |a: &[String], b: &[String]| -> Vec<String> {
                    a.iter().filter(|n| !b.contains(n)).cloned().collect()
                };
                return Err(EvalError::invalid(format!(
                    "{} and {} do not hold the same methods (only in the first: {:?}; only in the second: {:?})",
                    first.display(),
                    dir.display(),
                    only(expected, &names),
                    only(&names, expected)
                )));
            }
            Some(_) => {}
        }
    }
    Ok(common.map(|(names, _)| names).unwrap_or_default())
}

/// The questions of one group, by cluster, in every run.
#[derive(Debug)]
pub struct GroupLayout {
    pub group: Group,
    /// The runs' labels.
    labels: Vec<String>,
    /// For each run, for each cluster (sorted by its key), the positions of its questions.
    clusters: Vec<Vec<Vec<usize>>>,
}

impl GroupLayout {
    /// `None` when no run has a question of the group. Refused when only some runs do.
    pub fn of(runs: &[Run], group: Group) -> Result<Option<Self>, EvalError> {
        let labels = runs.iter().map(|run| run.label.clone()).collect();
        let mut clusters = Vec::with_capacity(runs.len());
        for run in runs {
            let mut by_cluster: BTreeMap<&Cluster, Vec<usize>> = BTreeMap::new();
            for (position, asked) in run.questions.iter().enumerate() {
                if group.holds(asked.class) {
                    by_cluster.entry(&asked.cluster).or_default().push(position);
                }
            }
            clusters.push(by_cluster.into_values().collect::<Vec<_>>());
        }
        let with_questions = clusters.iter().filter(|c| !c.is_empty()).count();
        if with_questions == 0 {
            return Ok(None);
        }
        if with_questions != runs.len() {
            return Err(EvalError::invalid(format!(
                "group {group}: only {with_questions} of {} runs have {} questions of it",
                runs.len(),
                runs.first().map_or("", |r| match r.split {
                    Split::Validation => "validation",
                    Split::Test => "test",
                })
            )));
        }
        Ok(Some(Self {
            group,
            labels,
            clusters,
        }))
    }

    /// For each run, the number of questions in each cluster.
    pub fn questions_per_cluster(&self) -> Vec<Vec<u32>> {
        self.clusters
            .iter()
            .map(|run| run.iter().map(|cluster| cluster.len() as u32).collect())
            .collect()
    }

    /// Each run as a stratum for [`crate::stats`]: its label and its cluster sizes.
    pub fn strata(&self) -> Vec<Stratum> {
        self.labels
            .iter()
            .zip(self.questions_per_cluster())
            .map(|(label, questions_per_cluster)| Stratum {
                label: label.clone(),
                questions_per_cluster,
            })
            .collect()
    }

    /// For each run, how many questions and clusters the group has.
    pub fn sizes(&self) -> Vec<(usize, usize)> {
        self.clusters
            .iter()
            .map(|run| (run.iter().map(Vec::len).sum(), run.len()))
            .collect()
    }

    /// For each run and cluster, one measure of one method summed over the cluster's questions.
    pub fn totals(&self, runs: &[Run], method: &str, measure: Measure) -> Result<Vec<Vec<f64>>, EvalError> {
        self.clusters
            .iter()
            .zip(runs)
            .map(|(clusters, run)| {
                let loaded = run.methods.get(method).ok_or_else(|| {
                    EvalError::invalid(format!("run {}: no scores for method {method}", run.label))
                })?;
                Ok(clusters
                    .iter()
                    .map(|cluster| cluster.iter().map(|q| loaded.measures[*q].get(measure)).sum())
                    .collect())
            })
            .collect()
    }
}

/// Each run's mean over questions.
pub fn means_per_run(totals: &[Vec<f64>], questions_per_cluster: &[Vec<u32>]) -> Vec<f64> {
    totals
        .iter()
        .zip(questions_per_cluster)
        .map(|(totals, sizes)| totals.iter().sum::<f64>() / sizes.iter().map(|s| f64::from(*s)).sum::<f64>())
        .collect()
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use crate::aggregate::{QuestionScore, means_by_split_and_group};
    use crate::files::{made_by, write_json_file};
    use crate::testkit::question;

    /// A question for a scores file: (id, class, thread, one value used for every measure).
    pub(crate) type Row<'a> = (&'a str, QueryClass, Option<&'a str>, f64);

    /// Write `measures/<name>.json` for a method whose every measure of a question is one value.
    pub(crate) fn write_scores(dir: &Path, method: &str, rows: &[Row<'_>], inputs_tag: &str) {
        let spec: MethodSpec = method.parse().unwrap();
        let questions: Vec<QuestionScore> = rows
            .iter()
            .map(|(id, class, thread, value)| {
                QuestionScore::new(
                    &question(id, 0, *class, &[], *thread),
                    Measures::from_fn(|_| *value),
                )
            })
            .collect();
        let scores = MethodScores {
            method: spec.name(),
            text: spec.to_string(),
            spec,
            made_by: made_by(),
            inputs: [RECORDS, QUERIES, CANDIDATES]
                .iter()
                .map(|file| (file.to_string(), format!("{inputs_tag}-{file}")))
                .collect(),
            means: means_by_split_and_group(&questions),
            questions,
        };
        write_json_file(&scores_path(dir, &spec.name()), &scores).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{Row, write_scores};
    use super::*;
    use crate::files::test_dir;
    use std::fs;

    // Threads inc-0001 and inc-0003 are in the test split, inc-0006 in validation (see aggregate.rs).
    // Question ids q-ord-1 (test) and q-ord-6 (validation) hash for themselves.
    const ROWS: [Row<'static>; 6] = [
        ("q1", QueryClass::Changing, Some("inc-0003"), 1.0),
        ("q2", QueryClass::Changing, Some("inc-0001"), 0.0),
        ("q3", QueryClass::Changing, Some("inc-0001"), 1.0),
        ("q4", QueryClass::Changing, Some("inc-0006"), 1.0),
        ("q-ord-1", QueryClass::Ordinary, None, 0.5),
        ("q-ord-6", QueryClass::Ordinary, None, 0.25),
    ];

    #[test]
    fn a_run_loads_one_split_and_lays_its_questions_out_by_thread() {
        let dir = test_dir("scores-load").join("seed-42");
        write_scores(&dir, "dense", &ROWS, "a");
        write_scores(&dir, "rrf(k=60)", &ROWS, "a");
        assert_eq!(method_names_in(&dir).unwrap(), ["dense", "rrf-k60"]);
        let names = vec!["dense".to_string(), "rrf-k60".to_string()];
        let run = load_run(&dir, &names, Split::Test).unwrap();
        assert_eq!(run.label, "seed-42");
        let asked: Vec<&str> = run.questions.iter().map(|q| q.query.as_str()).collect();
        assert_eq!(asked, ["q1", "q2", "q3", "q-ord-1"]);
        assert_eq!(run.methods["rrf-k60"].text, "rrf(k=60)");
        assert_eq!(run.methods["dense"].measures.len(), 4);

        let runs = [run];
        let layout = GroupLayout::of(&runs, Group::Changing).unwrap().unwrap();
        // Clusters in key order: thread inc-0001 (q2, q3), then thread inc-0003 (q1).
        assert_eq!(layout.questions_per_cluster(), [[2, 1]]);
        let strata = layout.strata();
        assert_eq!(
            (
                strata[0].label.as_str(),
                strata[0].questions_per_cluster.as_slice()
            ),
            ("seed-42", [2, 1].as_slice())
        );
        assert_eq!(layout.sizes(), [(3, 2)]);
        let totals = layout.totals(&runs, "dense", Measure::Hit4).unwrap();
        assert_eq!(totals, [[0.0 + 1.0, 1.0]]);
        assert_eq!(
            means_per_run(&totals, &layout.questions_per_cluster()),
            [2.0 / 3.0]
        );
        assert!(layout.totals(&runs, "oracle", Measure::Hit4).is_err());
        // A question without a thread is its own cluster.
        let ordinary = GroupLayout::of(&runs, Group::OrdinaryAll).unwrap().unwrap();
        assert_eq!(ordinary.questions_per_cluster(), [[1]]);
        assert!(GroupLayout::of(&runs, Group::Ambiguous).unwrap().is_none());

        let validation = load_run(&dir, &names, Split::Validation).unwrap();
        let asked: Vec<&str> = validation.questions.iter().map(|q| q.query.as_str()).collect();
        assert_eq!(asked, ["q4", "q-ord-6"]);
        fs::remove_dir_all(dir.parent().unwrap()).unwrap();
    }

    #[test]
    fn methods_ranked_from_different_files_or_questions_are_refused() {
        let dir = test_dir("scores-refuse");
        write_scores(&dir, "dense", &ROWS, "a");
        write_scores(&dir, "rrf(k=60)", &ROWS, "b");
        write_scores(&dir, "oracle", &ROWS[..5], "a");
        let load = |names: &[&str]| {
            let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            load_run(&dir, &names, Split::Test)
                .map(|_| ())
                .map_err(|e| e.to_string())
        };
        assert!(load(&["dense"]).is_ok());
        assert!(
            load(&["dense", "rrf-k60"])
                .unwrap_err()
                .contains("other input files")
        );
        // The validation question q-ord-6 is missing from oracle's file; the test split still matches.
        assert!(load(&["dense", "oracle"]).is_ok());
        write_scores(&dir, "oracle", &ROWS[1..], "a");
        assert!(load(&["dense", "oracle"]).unwrap_err().contains("same questions"));
        assert!(load(&["dense", "product"]).unwrap_err().contains("product.json"));
        // A file renamed by hand no longer matches the method inside it.
        fs::copy(scores_path(&dir, "dense"), scores_path(&dir, "product")).unwrap();
        assert!(
            load(&["product"])
                .unwrap_err()
                .contains("not the name of the method")
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn several_runs_must_hold_the_same_methods_and_groups_and_have_different_names() {
        let root = test_dir("scores-runs");
        let (a, b) = (root.join("seed-1"), root.join("seed-2"));
        write_scores(&a, "dense", &ROWS, "a");
        write_scores(&b, "dense", &ROWS[..4], "b");
        let dirs = vec![a.clone(), b.clone()];
        assert_eq!(methods_common_to(&dirs).unwrap(), ["dense"]);
        let runs = load_runs(&dirs, &["dense".to_string()], Split::Test).unwrap();
        let layout = GroupLayout::of(&runs, Group::Changing).unwrap().unwrap();
        assert_eq!(layout.questions_per_cluster(), [[2, 1], [2, 1]]);
        // Run seed-2 has no ordinary question: a group must be in every run or in none.
        assert!(
            GroupLayout::of(&runs, Group::Ordinary)
                .unwrap_err()
                .to_string()
                .contains("only 1 of 2 runs")
        );
        write_scores(&b, "oracle", &ROWS, "b");
        assert!(
            methods_common_to(&dirs)
                .unwrap_err()
                .to_string()
                .contains("oracle")
        );
        let same_name = vec![a.clone(), a];
        assert!(load_runs(&same_name, &["dense".to_string()], Split::Test).is_err());
        fs::remove_dir_all(&root).unwrap();
    }
}
