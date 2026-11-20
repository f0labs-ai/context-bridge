//! `cb-eval tune`: choose a method's parameters on the validation split.
//!
//! The candidates are scored by one named measure on one named group of
//! validation questions (the mean over questions; with several runs, the mean
//! of the runs' means). The best candidate is written to `tuned.json` under a
//! label; `cb-eval report` then reports it on the test split and leaves the
//! other candidates out of its main tables.
//!
//! A choice may carry a guard: a second measure on a second group that must
//! not fall more than a margin below a baseline's (for claim 2: at most 1 point below `product` on
//! `hit@4` of ordinary questions). Only
//! candidates within the guard can win; when none is, the one closest to it does.

use crate::aggregate::Group;
use crate::error::EvalError;
use crate::files::{TUNED, read_json_file, write_json_file};
use crate::latex::is_plain_name;
use crate::measures::Measure;
use crate::scores::{GroupLayout, Run, load_runs};
use crate::spec::MethodSpec;
use crate::stats::mean_over_strata;
use cb_types::Split;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct GuardArgs {
    pub measure: Measure,
    pub group: Group,
    pub baseline: MethodSpec,
    /// How far below the baseline the guard measure may fall, as a fraction (0.01 = 1 point).
    pub margin: f64,
}

#[derive(Clone, Debug)]
pub struct TuneArgs {
    /// One or more run directories.
    pub runs: Vec<PathBuf>,
    /// Where `tuned.json` is kept.
    pub out: PathBuf,
    /// The name of this choice, e.g. `product_decay`.
    pub label: String,
    pub candidates: Vec<MethodSpec>,
    pub measure: Measure,
    pub group: Group,
    /// `None` takes the measure's own direction.
    pub higher_is_better: Option<bool>,
    pub guard: Option<GuardArgs>,
}

/// Every choice made so far: `tuned.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tuned {
    pub choices: Vec<Choice>,
}

impl Tuned {
    /// Read `tuned.json` from `dir`; no file means no choices.
    pub fn read(dir: &Path) -> Result<Self, EvalError> {
        let path = dir.join(TUNED);
        if path.exists() {
            read_json_file(&path)
        } else {
            Ok(Self::default())
        }
    }
}

/// One parameter choice made on validation questions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    pub label: String,
    /// The labels of the runs the choice was made on.
    pub runs: Vec<String>,
    pub split: Split,
    pub measure: Measure,
    pub group: Group,
    pub higher_is_better: bool,
    pub questions: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<Guard>,
    /// The name of the chosen method.
    pub chosen: String,
    /// Every candidate with its validation value, in the order given.
    pub candidates: Vec<Candidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Guard {
    pub measure: Measure,
    pub group: Group,
    /// The name of the baseline method.
    pub baseline: String,
    pub margin: f64,
    /// The baseline's validation value of the guard measure.
    pub baseline_value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    /// The method's file-safe name.
    pub method: String,
    pub text: String,
    /// The validation value of the measure the choice is made by.
    pub value: f64,
    /// The validation value of the guard measure, when there is a guard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard_value: Option<f64>,
    /// Whether the candidate is within the guard (true when there is none).
    pub within_guard: bool,
}

fn validation_value(
    runs: &[Run],
    method: &str,
    measure: Measure,
    group: Group,
) -> Result<(f64, usize), EvalError> {
    let layout = GroupLayout::of(runs, group)?
        .ok_or_else(|| EvalError::invalid(format!("no validation question belongs to group {group}")))?;
    let sizes = layout.questions_per_cluster();
    let totals = layout.totals(runs, method, measure)?;
    let questions = layout.sizes().iter().map(|(questions, _)| questions).sum();
    Ok((mean_over_strata(&totals, &sizes), questions))
}

/// How far `value` is on the wrong side of `bar`; 0 when it is on the right side.
fn shortfall(value: f64, bar: f64, higher_is_better: bool) -> f64 {
    if higher_is_better {
        (bar - value).max(0.0)
    } else {
        (value - bar).max(0.0)
    }
}

/// Make the choice without writing it.
pub fn choose(args: &TuneArgs) -> Result<Choice, EvalError> {
    if args.candidates.is_empty() {
        return Err(EvalError::invalid("tune: no candidate was named"));
    }
    if !is_plain_name(&args.label) {
        return Err(EvalError::invalid(format!(
            "tune: label {:?} may hold letters, digits, '-', '_' and '.' only (it becomes part of number keys)",
            args.label
        )));
    }
    let mut names: Vec<String> = Vec::new();
    for spec in &args.candidates {
        if !names.contains(&spec.name()) {
            names.push(spec.name());
        }
    }
    let candidate_count = names.len();
    if let Some(guard) = &args.guard
        && !names.contains(&guard.baseline.name())
    {
        names.push(guard.baseline.name());
    }
    let runs = load_runs(&args.runs, &names, Split::Validation)?;
    let higher_is_better = args.higher_is_better.unwrap_or(args.measure.higher_is_better());

    let guard = match &args.guard {
        Some(guard) => {
            let (baseline_value, _) =
                validation_value(&runs, &guard.baseline.name(), guard.measure, guard.group)?;
            Some(Guard {
                measure: guard.measure,
                group: guard.group,
                baseline: guard.baseline.name(),
                margin: guard.margin,
                baseline_value,
            })
        }
        None => None,
    };
    // The guard's bar and its direction: a measure that counts something wanted may not fall
    // below baseline - margin; one that counts something unwanted may not rise above baseline + margin.
    let guard_bar = guard.as_ref().map(|g| {
        let up = g.measure.higher_is_better();
        let bar = if up {
            g.baseline_value - g.margin
        } else {
            g.baseline_value + g.margin
        };
        (g.measure, g.group, bar, up)
    });

    let mut questions = 0;
    let mut candidates = Vec::with_capacity(candidate_count);
    let mut shortfalls = Vec::with_capacity(candidate_count);
    for name in &names[..candidate_count] {
        let (value, asked) = validation_value(&runs, name, args.measure, args.group)?;
        questions = asked;
        let (guard_value, missed_by) = match guard_bar {
            Some((measure, group, bar, up)) => {
                let (guard_value, _) = validation_value(&runs, name, measure, group)?;
                (Some(guard_value), shortfall(guard_value, bar, up))
            }
            None => (None, 0.0),
        };
        shortfalls.push(missed_by);
        candidates.push(Candidate {
            method: name.clone(),
            text: runs
                .first()
                .and_then(|r| r.methods.get(name))
                .map_or_else(String::new, |m| m.text.clone()),
            value,
            guard_value,
            within_guard: missed_by == 0.0,
        });
    }

    // Smallest shortfall first (0 for everyone within the guard), then the best value, then the
    // candidate named first.
    let mut best = 0;
    for i in 1..candidates.len() {
        let better_value = if higher_is_better {
            candidates[i].value > candidates[best].value
        } else {
            candidates[i].value < candidates[best].value
        };
        if shortfalls[i] < shortfalls[best] || (shortfalls[i] == shortfalls[best] && better_value) {
            best = i;
        }
    }
    Ok(Choice {
        label: args.label.clone(),
        runs: runs.iter().map(|r| r.label.clone()).collect(),
        split: Split::Validation,
        measure: args.measure,
        group: args.group,
        higher_is_better,
        questions,
        guard,
        chosen: candidates[best].method.clone(),
        candidates,
    })
}

/// Make the choice and record it in `tuned.json`, replacing an earlier choice with the same label.
pub fn run(args: &TuneArgs) -> Result<Choice, EvalError> {
    let choice = choose(args)?;
    let mut tuned = Tuned::read(&args.out)?;
    tuned.choices.retain(|earlier| earlier.label != choice.label);
    tuned.choices.push(choice.clone());
    tuned.choices.sort_by(|a, b| a.label.cmp(&b.label));
    write_json_file(&args.out.join(TUNED), &tuned)?;
    Ok(choice)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::test_dir;
    use crate::scores::testing::{Row, write_scores};
    use cb_types::QueryClass;
    use std::fs;

    // inc-0006 and inc-a are validation threads; inc-0001 is a test thread; q-ord-6 hashes to
    // validation and q-ord-1 to test.
    fn rows(validation: [f64; 3], test: f64, ordinary_validation: f64) -> Vec<Row<'static>> {
        vec![
            ("v1", QueryClass::Changing, Some("inc-0006"), validation[0]),
            ("v2", QueryClass::Changing, Some("inc-0006"), validation[1]),
            ("v3", QueryClass::Changing, Some("inc-a"), validation[2]),
            ("t1", QueryClass::Changing, Some("inc-0001"), test),
            ("q-ord-6", QueryClass::Ordinary, None, ordinary_validation),
            ("q-ord-1", QueryClass::Ordinary, None, 0.0),
        ]
    }

    fn args(dir: &Path, candidates: &str) -> TuneArgs {
        TuneArgs {
            runs: vec![dir.to_path_buf()],
            out: dir.to_path_buf(),
            label: "product_decay".to_string(),
            candidates: MethodSpec::parse_grid(candidates).unwrap(),
            measure: Measure::Current1,
            group: Group::Changing,
            higher_is_better: None,
            guard: None,
        }
    }

    /// Three half-lives: 60 s is best on validation, 300 s is best on test.
    fn write_three(dir: &Path) {
        // validation means: (1 + 1 + 0)/3 = 0.667, (1 + 0 + 0)/3 = 0.333, (0 + 0 + 0)/3 = 0
        write_scores(dir, "product_decay(h=60)", &rows([1.0, 1.0, 0.0], 0.0, 0.2), "a");
        write_scores(dir, "product_decay(h=300)", &rows([1.0, 0.0, 0.0], 1.0, 0.9), "a");
        write_scores(
            dir,
            "product_decay(h=1800)",
            &rows([0.0, 0.0, 0.0], 1.0, 1.0),
            "a",
        );
        write_scores(dir, "product", &rows([0.0, 0.0, 0.0], 0.0, 1.0), "a");
    }

    #[test]
    fn the_best_candidate_on_validation_is_chosen_whatever_the_test_split_says() {
        let dir = test_dir("tune");
        write_three(&dir);
        let choice = run(&args(&dir, "product_decay(h=60|300|1800)")).unwrap();
        assert_eq!(choice.chosen, "product_decay-h60");
        assert_eq!(choice.split, Split::Validation);
        assert_eq!(choice.questions, 3);
        let values: Vec<f64> = choice.candidates.iter().map(|c| c.value).collect();
        assert_eq!(values, [2.0 / 3.0, 1.0 / 3.0, 0.0]);
        assert!(
            choice
                .candidates
                .iter()
                .all(|c| c.within_guard && c.guard_value.is_none())
        );
        assert_eq!(Tuned::read(&dir).unwrap().choices, [choice]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_measure_of_unwanted_records_is_minimised_and_the_direction_can_be_set() {
        let dir = test_dir("tune-direction");
        write_three(&dir);
        let mut lowest = args(&dir, "product_decay(h=60|300|1800)");
        lowest.measure = Measure::Stale4;
        assert_eq!(choose(&lowest).unwrap().chosen, "product_decay-h1800");
        assert!(!choose(&lowest).unwrap().higher_is_better);
        lowest.higher_is_better = Some(true);
        assert_eq!(choose(&lowest).unwrap().chosen, "product_decay-h60");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ties_go_to_the_candidate_named_first() {
        let dir = test_dir("tune-ties");
        write_scores(&dir, "cc(alpha=0.2)", &rows([1.0, 0.0, 0.0], 0.0, 0.0), "a");
        write_scores(&dir, "cc(alpha=0.5)", &rows([0.0, 0.0, 1.0], 0.0, 0.0), "a");
        assert_eq!(
            choose(&args(&dir, "cc(alpha=0.2|0.5)")).unwrap().chosen,
            "cc-alpha0.2"
        );
        assert_eq!(
            choose(&args(&dir, "cc(alpha=0.5|0.2)")).unwrap().chosen,
            "cc-alpha0.5"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_guard_keeps_out_candidates_that_cost_ordinary_questions_too_much() {
        let dir = test_dir("tune-guard");
        write_three(&dir);
        let mut guarded = args(&dir, "product_decay(h=60|300|1800)");
        // hit@4 on ordinary validation questions: product 1.0; h=60 0.2, h=300 0.9, h=1800 1.0.
        guarded.guard = Some(GuardArgs {
            measure: Measure::Hit4,
            group: Group::OrdinaryAll,
            baseline: "product".parse().unwrap(),
            margin: 0.1,
        });
        // Within 0.1 of 1.0: h=300 (0.9) and h=1800 (1.0). Of those h=300 has the better current@1.
        let choice = choose(&guarded).unwrap();
        assert_eq!(choice.chosen, "product_decay-h300");
        let within: Vec<bool> = choice.candidates.iter().map(|c| c.within_guard).collect();
        assert_eq!(within, [false, true, true]);
        assert_eq!(choice.candidates[0].guard_value, Some(0.2));
        let guard = choice.guard.unwrap();
        assert_eq!((guard.baseline.as_str(), guard.baseline_value), ("product", 1.0));
        // A margin of 0.05 leaves only h=1800.
        guarded.guard.as_mut().unwrap().margin = 0.05;
        assert_eq!(choose(&guarded).unwrap().chosen, "product_decay-h1800");
        // When no candidate is within the guard, the one that misses it by least is chosen.
        let mut none_within = args(&dir, "product_decay(h=60|300)");
        none_within.guard = guarded.guard.clone();
        let choice = choose(&none_within).unwrap();
        assert_eq!(choice.chosen, "product_decay-h300");
        assert!(choice.candidates.iter().all(|c| !c.within_guard));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_later_choice_with_the_same_label_replaces_the_earlier_one() {
        let dir = test_dir("tune-replace");
        write_three(&dir);
        run(&args(&dir, "product_decay(h=60|300|1800)")).unwrap();
        let mut other = args(&dir, "product_decay(h=300|1800)");
        other.label = "a_first".to_string();
        run(&other).unwrap();
        let again = run(&args(&dir, "product_decay(h=300|1800)")).unwrap();
        let tuned = Tuned::read(&dir).unwrap();
        let labels: Vec<&str> = tuned.choices.iter().map(|c| c.label.as_str()).collect();
        assert_eq!(labels, ["a_first", "product_decay"]);
        assert_eq!(tuned.choices[1], again);
        assert_eq!(again.chosen, "product_decay-h300");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn with_several_runs_the_mean_of_the_runs_means_decides() {
        let root = test_dir("tune-runs");
        let (a, b) = (root.join("seed-1"), root.join("seed-2"));
        // h=60: run means 1/3 and 1/3 → 0.333. h=300: run means 1 and 0 → 0.5.
        write_scores(&a, "product_decay(h=60)", &rows([1.0, 0.0, 0.0], 0.0, 0.0), "a");
        write_scores(&b, "product_decay(h=60)", &rows([0.0, 1.0, 0.0], 0.0, 0.0), "b");
        write_scores(&a, "product_decay(h=300)", &rows([1.0, 1.0, 1.0], 0.0, 0.0), "a");
        write_scores(&b, "product_decay(h=300)", &rows([0.0, 0.0, 0.0], 0.0, 0.0), "b");
        let mut both = args(&root, "product_decay(h=60|300)");
        both.runs = vec![a, b];
        let choice = run(&both).unwrap();
        assert_eq!(choice.chosen, "product_decay-h300");
        assert_eq!(choice.runs, ["seed-1", "seed-2"]);
        assert_eq!(choice.questions, 6);
        assert_eq!(choice.candidates[1].value, 0.5);
        assert!(root.join(TUNED).exists());
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn a_candidate_without_scores_or_a_group_without_questions_is_an_error() {
        let dir = test_dir("tune-missing");
        write_three(&dir);
        assert!(choose(&args(&dir, "product_decay(h=60|7200)")).is_err());
        let mut no_group = args(&dir, "product_decay(h=60)");
        no_group.group = Group::Ambiguous;
        assert!(
            choose(&no_group)
                .unwrap_err()
                .to_string()
                .contains("no validation question")
        );
        let mut nothing = args(&dir, "product_decay(h=60)");
        nothing.candidates.clear();
        assert!(choose(&nothing).is_err());
        let mut slash = args(&dir, "product_decay(h=60)");
        slash.label = "decay/fast".to_string();
        assert!(choose(&slash).unwrap_err().to_string().contains("label"));
        fs::remove_dir_all(&dir).unwrap();
    }
}
