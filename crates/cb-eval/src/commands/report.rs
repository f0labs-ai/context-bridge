//! `cb-eval report`: every ranking number of the paper for a set of runs, on
//! the test split only.
//!
//! For every group of questions, every method and every measure: the mean
//! (over the questions of each run, then over runs), a 95% percentile
//! bootstrap interval from resamples of threads, and against each named
//! baseline the difference, its 95% interval (from the same resamples, so it
//! is paired) and a two-sided paired randomization p-value. The seed and the
//! number of draws are written into every output.
//!
//! Outputs, under the output directory:
//! - `summary.json`: everything above, with the inputs' hashes and the
//!   parameter choices of `tuned.json`;
//! - `figures/*.csv`: the same numbers as tables of rows for figures;
//! - `tables/*.tex`: booktabs tables of the featured methods, and one table
//!   per family of methods that differ in one or two parameters;
//! - `numbers.tex`: every number as a macro, read in the paper with `\cbnum{key}`.

use crate::aggregate::Group;
use crate::commands::tune::{Choice, Tuned};
use crate::error::EvalError;
use crate::files::{
    FIGURES_DIR, NUMBERS, SUMMARY, TABLES_DIR, made_by, read_json_file, write_json_file, write_text_file,
};
use crate::latex::{
    Numbers, Table, csv_line, csv_number, escape, format_difference, format_p, format_value, is_plain_name,
};
use crate::measures::{Measure, Unit};
use crate::parallel::{map_in_parallel, threads_to_use};
use crate::scores::{GroupLayout, Run, load_runs, means_per_run, methods_common_to};
use crate::spec::{KINDS, MethodSpec};
use crate::stats::{
    Interval, LabelSwaps, Resamples, mean_over_strata, percentile_interval, two_sided_p_of_difference,
};
use cb_types::Split;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The measures shown in the tables unless others are named.
pub const DEFAULT_COLUMNS: [Measure; 7] = [
    Measure::Ndcg10,
    Measure::Mrr,
    Measure::Current1,
    Measure::Hit4,
    Measure::Stale4,
    Measure::Harmful4,
    Measure::Changing4,
];

#[derive(Clone, Debug)]
pub struct ReportArgs {
    /// One or more run directories; several are the strata of one analysis (the seeds of a corpus).
    pub runs: Vec<PathBuf>,
    pub out: PathBuf,
    /// The first part of every key in `numbers.tex`, e.g. `synth` or `github`.
    pub name: String,
    /// The methods to report; `None` reports every method the runs hold scores for.
    pub methods: Option<Vec<MethodSpec>>,
    pub baselines: Vec<MethodSpec>,
    /// A `tuned.json` whose choices are applied.
    pub tuned: Option<PathBuf>,
    /// Methods to show in the main tables besides the ones shown anyway.
    pub feature: Vec<MethodSpec>,
    /// The measures the tables show.
    pub columns: Vec<Measure>,
    pub seed: u64,
    pub draws: usize,
    /// Worker threads; 0 uses every core.
    pub threads: usize,
}

/// `summary.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub name: String,
    pub made_by: String,
    pub split: Split,
    pub seed: u64,
    pub draws: usize,
    /// How the intervals and p-values were computed, in words.
    pub statistics: String,
    /// The names of the baselines.
    pub baselines: Vec<String>,
    /// In the order every per-run list below follows.
    pub runs: Vec<RunSummary>,
    /// The parameter choices applied, as `tune` made them on the validation split.
    pub tuned: Vec<Choice>,
    pub groups: Vec<GroupSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunSummary {
    pub label: String,
    /// File name → sha256 of the data set and candidate files the scores were computed from.
    pub inputs: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupSummary {
    pub group: Group,
    /// Per run.
    pub questions: Vec<usize>,
    /// Per run: threads, plus questions without a thread.
    pub clusters: Vec<usize>,
    /// In listing order.
    pub methods: Vec<MethodSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MethodSummary {
    pub method: String,
    pub text: String,
    pub spec: MethodSpec,
    /// Whether the method is a row of the main tables.
    pub featured: bool,
    /// The labels of the `tuned.json` choices that chose this method.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chosen_by: Vec<String>,
    /// One entry per measure, in the order of [`Measure::ALL`].
    pub measures: Vec<MeasureSummary>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasureSummary {
    pub measure: Measure,
    pub mean: f64,
    /// Each run's mean over its questions.
    pub per_run: Vec<f64>,
    pub ci: Interval,
    /// Against each baseline other than the method itself.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub versus: Vec<Comparison>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Comparison {
    pub baseline: String,
    /// The method's mean minus the baseline's.
    pub difference: f64,
    pub ci: Interval,
    /// Two-sided paired randomization test.
    pub p: f64,
}

impl MethodSummary {
    pub fn measure(&self, measure: Measure) -> Option<&MeasureSummary> {
        self.measures.iter().find(|m| m.measure == measure)
    }
}

impl MeasureSummary {
    pub fn versus(&self, baseline: &str) -> Option<&Comparison> {
        self.versus.iter().find(|c| c.baseline == baseline)
    }
}

impl GroupSummary {
    pub fn method(&self, name: &str) -> Option<&MethodSummary> {
        self.methods.iter().find(|m| m.method == name)
    }
}

impl Summary {
    pub fn group(&self, group: Group) -> Option<&GroupSummary> {
        self.groups.iter().find(|g| g.group == group)
    }
}

/// What `report` did.
#[derive(Debug)]
pub struct ReportOutcome {
    pub summary: Summary,
    /// Every file written, in the order written.
    pub files: Vec<PathBuf>,
    pub numbers: usize,
    /// Things the reader of the report should know (a group left out, a choice made on other runs).
    pub notes: Vec<String>,
}

/// A method of the report, before its numbers are computed.
#[derive(Clone, Debug)]
struct Listed {
    name: String,
    text: String,
    spec: MethodSpec,
    featured: bool,
    chosen_by: Vec<String>,
}

/// Compute the report and write every output.
pub fn run(args: &ReportArgs) -> Result<ReportOutcome, EvalError> {
    let (summary, notes) = summarise(args)?;
    let mut files = Vec::new();
    let summary_path = args.out.join(SUMMARY);
    write_json_file(&summary_path, &summary)?;
    files.push(summary_path);
    files.extend(write_figures(&args.out, &summary)?);
    files.extend(write_tables(&args.out, &summary, &args.columns)?);
    let numbers = numbers_of(&summary)?;
    let numbers_path = args.out.join(NUMBERS);
    write_text_file(&numbers_path, &numbers.render())?;
    files.push(numbers_path);
    Ok(ReportOutcome {
        summary,
        files,
        numbers: numbers.len(),
        notes,
    })
}

fn names_of(specs: &[MethodSpec]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for spec in specs {
        if !names.contains(&spec.name()) {
            names.push(spec.name());
        }
    }
    names
}

/// Compute every number of the report without writing anything.
pub fn summarise(args: &ReportArgs) -> Result<(Summary, Vec<String>), EvalError> {
    if !is_plain_name(&args.name) {
        return Err(EvalError::invalid(format!(
            "report name {:?} may hold letters, digits, '-', '_' and '.' only",
            args.name
        )));
    }
    if args.draws == 0 {
        return Err(EvalError::invalid(
            "report: the number of draws must be at least 1",
        ));
    }
    if args.runs.is_empty() {
        return Err(EvalError::invalid("report: no run directory was given"));
    }
    let mut notes = Vec::new();
    let tuned = match &args.tuned {
        Some(path) => read_json_file::<Tuned>(path)?,
        None => Tuned::default(),
    };
    let mut names = match &args.methods {
        Some(specs) => names_of(specs),
        None => methods_common_to(&args.runs)?,
    };
    let baselines = names_of(&args.baselines);
    for name in baselines.iter().chain(tuned.choices.iter().map(|c| &c.chosen)) {
        if !names.contains(name) {
            names.push(name.clone());
        }
    }
    if names.is_empty() {
        return Err(EvalError::invalid(
            "report: the runs hold no scores; run cb-eval rank first",
        ));
    }
    let mut runs = load_runs(&args.runs, &names, Split::Test)?;
    runs.sort_by(|a, b| a.label.cmp(&b.label));
    let run_labels: Vec<String> = runs.iter().map(|r| r.label.clone()).collect();
    for choice in &tuned.choices {
        // The same runs named in another order are the same runs.
        let mut chosen_on = choice.runs.clone();
        chosen_on.sort();
        if chosen_on != run_labels {
            notes.push(format!(
                "choice {} was made on the validation questions of runs {:?}; this report covers {:?}",
                choice.label, choice.runs, run_labels
            ));
        }
    }

    let featured_extra = names_of(&args.feature);
    let listed = list_methods(&runs, &names, &baselines, &tuned, &featured_extra)?;
    let threads = threads_to_use(args.threads);

    let mut groups = Vec::new();
    for group in Group::ALL {
        let holding = runs
            .iter()
            .filter(|run| run.questions.iter().any(|q| group.holds(q.class)))
            .count();
        if holding == 0 {
            continue;
        }
        if holding < runs.len() {
            notes.push(format!(
                "group {group} is left out: only {holding} of {} runs have test questions of it",
                runs.len()
            ));
            continue;
        }
        let layout = GroupLayout::of(&runs, group)?
            .ok_or_else(|| EvalError::invalid(format!("group {group}: no questions")))?;
        groups.push(summarise_group(
            args, &runs, layout, &listed, &baselines, threads,
        )?);
    }
    let summary = Summary {
        name: args.name.clone(),
        made_by: made_by(),
        split: Split::Test,
        seed: args.seed,
        draws: args.draws,
        statistics: format!(
            "Means over the questions of each run, then over runs. 95% percentile intervals from {} bootstrap \
             resamples of clusters (a thread, or a question without a thread), each run resampled on its own. \
             Differences are method minus baseline over the same resamples. p is a two-sided paired \
             randomization test swapping the two methods' answers cluster by cluster, {} permutations, \
             p = (1 + b) / (1 + {}). Draws come from ChaCha20 keyed by seed {} and \
             bootstrap/<group>/<run> or randomization/<group>/<run>.",
            args.draws, args.draws, args.draws, args.seed
        ),
        baselines,
        runs: runs
            .iter()
            .map(|run| RunSummary {
                label: run.label.clone(),
                inputs: run.inputs.clone(),
            })
            .collect(),
        tuned: tuned.choices,
        groups,
    };
    Ok((summary, notes))
}

/// The methods in listing order, with what decides whether they are shown in the main tables.
///
/// Featured: the baselines, every method a choice chose, every method that
/// was not a candidate of any choice, and the ones asked for. A candidate
/// that was not chosen is reported everywhere except the main tables.
fn list_methods(
    runs: &[Run],
    names: &[String],
    baselines: &[String],
    tuned: &Tuned,
    extra: &[String],
) -> Result<Vec<Listed>, EvalError> {
    let first = runs
        .first()
        .ok_or_else(|| EvalError::invalid("report: no run directory was given"))?;
    let mut listed = Vec::with_capacity(names.len());
    for name in names {
        let loaded = first
            .methods
            .get(name)
            .ok_or_else(|| EvalError::invalid(format!("run {}: no scores for method {name}", first.label)))?;
        let chosen_by: Vec<String> = tuned
            .choices
            .iter()
            .filter(|c| c.chosen == *name)
            .map(|c| c.label.clone())
            .collect();
        let a_candidate = tuned
            .choices
            .iter()
            .any(|c| c.candidates.iter().any(|candidate| candidate.method == *name));
        listed.push(Listed {
            name: name.clone(),
            text: loaded.text.clone(),
            spec: loaded.spec,
            featured: baselines.contains(name)
                || !chosen_by.is_empty()
                || !a_candidate
                || extra.contains(name),
            chosen_by,
        });
    }
    listed.sort_by(|a, b| a.spec.cmp_listing(&b.spec).then_with(|| a.name.cmp(&b.name)));
    Ok(listed)
}

/// One method's statistics of one measure on one group, before they are summarised.
struct Draws {
    mean: f64,
    per_run: Vec<f64>,
    /// The statistic of every bootstrap resample.
    resampled: Vec<f64>,
    /// The statistic under every label swap ([`LabelSwaps::signed_means`]).
    swapped: Vec<f64>,
}

struct GroupDraws<'a> {
    runs: &'a [Run],
    layout: GroupLayout,
    questions_per_cluster: Vec<Vec<u32>>,
    resamples: Resamples,
    swaps: LabelSwaps,
}

impl GroupDraws<'_> {
    fn of(&self, method: &str, measure: Measure) -> Result<Draws, EvalError> {
        let totals = self.layout.totals(self.runs, method, measure)?;
        Ok(Draws {
            mean: mean_over_strata(&totals, &self.questions_per_cluster),
            per_run: means_per_run(&totals, &self.questions_per_cluster),
            resampled: self.resamples.means(&totals)?,
            swapped: self.swaps.signed_means(&totals)?,
        })
    }
}

fn interval_of(statistics: &[f64]) -> Result<Interval, EvalError> {
    percentile_interval(statistics).ok_or_else(|| EvalError::invalid("report: no resamples were drawn"))
}

fn summarise_group(
    args: &ReportArgs,
    runs: &[Run],
    layout: GroupLayout,
    listed: &[Listed],
    baselines: &[String],
    threads: usize,
) -> Result<GroupSummary, EvalError> {
    let group = layout.group;
    let strata = layout.strata();
    let sizes = layout.sizes();
    let draws = GroupDraws {
        runs,
        questions_per_cluster: layout.questions_per_cluster(),
        resamples: Resamples::draw(args.seed, group.name(), &strata, args.draws)?,
        swaps: LabelSwaps::draw(args.seed, group.name(), &strata, args.draws)?,
        layout,
    };
    let baseline_draws: Vec<(&String, Vec<Draws>)> = baselines
        .iter()
        .map(|name| {
            let per_measure = Measure::ALL
                .iter()
                .map(|measure| draws.of(name, *measure))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((name, per_measure))
        })
        .collect::<Result<_, EvalError>>()?;
    let methods = map_in_parallel(listed, threads, |method| {
        let mut measures = Vec::with_capacity(Measure::ALL.len());
        for (index, measure) in Measure::ALL.iter().enumerate() {
            let own = draws.of(&method.name, *measure)?;
            let mut versus = Vec::new();
            for (baseline, baseline_draws) in &baseline_draws {
                if **baseline == method.name {
                    continue;
                }
                let theirs = &baseline_draws[index];
                let difference = own.mean - theirs.mean;
                let resampled: Vec<f64> = own
                    .resampled
                    .iter()
                    .zip(&theirs.resampled)
                    .map(|(a, b)| a - b)
                    .collect();
                versus.push(Comparison {
                    baseline: (*baseline).clone(),
                    difference,
                    ci: interval_of(&resampled)?,
                    p: two_sided_p_of_difference(difference, &own.swapped, &theirs.swapped),
                });
            }
            measures.push(MeasureSummary {
                measure: *measure,
                mean: own.mean,
                ci: interval_of(&own.resampled)?,
                per_run: own.per_run,
                versus,
            });
        }
        Ok(MethodSummary {
            method: method.name.clone(),
            text: method.text.clone(),
            spec: method.spec,
            featured: method.featured,
            chosen_by: method.chosen_by.clone(),
            measures,
        })
    })?;
    Ok(GroupSummary {
        group,
        questions: sizes.iter().map(|(questions, _)| *questions).collect(),
        clusters: sizes.iter().map(|(_, clusters)| *clusters).collect(),
        methods,
    })
}

// ---- figures ---------------------------------------------------------------------

fn write_figures(out: &Path, summary: &Summary) -> Result<Vec<PathBuf>, EvalError> {
    let dir = out.join(FIGURES_DIR);
    let mut files = Vec::new();
    let mut write = |name: String, text: String| -> Result<(), EvalError> {
        let path = dir.join(name);
        write_text_file(&path, &text)?;
        files.push(path);
        Ok(())
    };

    let mut results = csv_line(&[
        "name",
        "group",
        "method",
        "text",
        "kind",
        "featured",
        "measure",
        "mean",
        "lo",
        "hi",
        "questions",
        "clusters",
    ]);
    let mut comparisons = csv_line(&[
        "name",
        "group",
        "method",
        "text",
        "baseline",
        "measure",
        "difference",
        "lo",
        "hi",
        "p",
    ]);
    let mut per_run = csv_line(&["name", "run", "group", "method", "measure", "mean"]);
    for group in &summary.groups {
        let questions: usize = group.questions.iter().sum();
        let clusters: usize = group.clusters.iter().sum();
        for method in &group.methods {
            for m in &method.measures {
                results.push_str(&csv_line(&[
                    summary.name.clone(),
                    group.group.name().to_string(),
                    method.method.clone(),
                    method.text.clone(),
                    method.spec.kind().to_string(),
                    method.featured.to_string(),
                    m.measure.name().to_string(),
                    csv_number(m.mean),
                    csv_number(m.ci.lo),
                    csv_number(m.ci.hi),
                    questions.to_string(),
                    clusters.to_string(),
                ]));
                for c in &m.versus {
                    comparisons.push_str(&csv_line(&[
                        summary.name.clone(),
                        group.group.name().to_string(),
                        method.method.clone(),
                        method.text.clone(),
                        c.baseline.clone(),
                        m.measure.name().to_string(),
                        csv_number(c.difference),
                        csv_number(c.ci.lo),
                        csv_number(c.ci.hi),
                        csv_number(c.p),
                    ]));
                }
                for (run, mean) in summary.runs.iter().zip(&m.per_run) {
                    per_run.push_str(&csv_line(&[
                        summary.name.clone(),
                        run.label.clone(),
                        group.group.name().to_string(),
                        method.method.clone(),
                        m.measure.name().to_string(),
                        csv_number(*mean),
                    ]));
                }
            }
        }
    }
    write("results.csv".to_string(), results)?;
    write("comparisons.csv".to_string(), comparisons)?;
    write("per_run.csv".to_string(), per_run)?;

    // One file per kind with every parameter as a column, for heat maps and sweeps.
    for kind in KINDS {
        let mut text = String::new();
        for group in &summary.groups {
            for method in group.methods.iter().filter(|m| m.spec.kind() == kind) {
                let parameters = method.spec.parameters();
                if text.is_empty() {
                    let mut header = vec!["group".to_string(), "measure".to_string(), "method".to_string()];
                    header.extend(parameters.iter().map(|(name, _)| name.to_string()));
                    header.extend(["mean", "lo", "hi"].map(String::from));
                    text.push_str(&csv_line(&header));
                }
                for m in &method.measures {
                    let mut row = vec![
                        group.group.name().to_string(),
                        m.measure.name().to_string(),
                        method.method.clone(),
                    ];
                    row.extend(parameters.iter().map(|(_, value)| csv_number(*value)));
                    row.extend([csv_number(m.mean), csv_number(m.ci.lo), csv_number(m.ci.hi)]);
                    text.push_str(&csv_line(&row));
                }
            }
        }
        if !text.is_empty() {
            write(format!("{kind}.csv"), text)?;
        }
    }
    Ok(files)
}

// ---- tables ----------------------------------------------------------------------

/// `\cbci{mean}{lo}{hi}` and `\cbdiff{difference}{lo}{hi}{p}`: how a table cell shows a
/// number with its interval. Defined here only when the paper has not defined them.
const CELL_COMMANDS: [&str; 2] = [
    "\\providecommand{\\cbci}[3]{#1\\,{\\scriptsize[#2,\\,#3]}}",
    "\\providecommand{\\cbdiff}[4]{#1\\,{\\scriptsize[#2,\\,#3]\\,$p$=#4}}",
];

fn unit_note(measure: Measure, difference: bool) -> &'static str {
    match (measure.unit(), difference) {
        (Unit::Percent, false) => " (\\%)",
        (Unit::Percent, true) => " (points)",
        (Unit::Fraction, _) => "",
    }
}

fn method_cell(method: &MethodSummary) -> String {
    let mut cell = format!("\\texttt{{{}}}", escape(&method.text));
    if !method.chosen_by.is_empty() {
        cell.push_str("$^{\\dagger}$");
    }
    cell
}

fn ci_cell(m: &MeasureSummary) -> String {
    let unit = m.measure.unit();
    format!(
        "\\cbci{{{}}}{{{}}}{{{}}}",
        format_value(m.mean, unit),
        format_value(m.ci.lo, unit),
        format_value(m.ci.hi, unit)
    )
}

fn diff_cell(measure: Measure, c: &Comparison) -> String {
    let unit = measure.unit();
    format!(
        "\\cbdiff{{{}}}{{{}}}{{{}}}{{{}}}",
        format_difference(c.difference, unit),
        format_difference(c.ci.lo, unit),
        format_difference(c.ci.hi, unit),
        format_p(c.p)
    )
}

fn table_comments(summary: &Summary, group: &GroupSummary, what: &str) -> Vec<String> {
    let runs: Vec<&str> = summary.runs.iter().map(|r| r.label.as_str()).collect();
    vec![
        format!(
            "Written by {} (cb-eval report). Do not edit; run the report again.",
            summary.made_by
        ),
        format!(
            "{}: {what}; {} questions of the test split, {} questions in {} clusters over runs {}.",
            summary.name,
            group.group.name(),
            group.questions.iter().sum::<usize>(),
            group.clusters.iter().sum::<usize>(),
            runs.join(", ")
        ),
        format!(
            "Means over questions per run, then over runs; 95% percentile bootstrap over clusters, {} resamples, seed {}.",
            summary.draws, summary.seed
        ),
        "\\dagger: parameters chosen on the validation split (tuned.json in summary.json).".to_string(),
    ]
}

fn write_tables(out: &Path, summary: &Summary, columns: &[Measure]) -> Result<Vec<PathBuf>, EvalError> {
    let dir = out.join(TABLES_DIR);
    let mut files = Vec::new();
    let mut write = |name: String, table: Table| -> Result<(), EvalError> {
        let path = dir.join(name);
        write_text_file(&path, &table.render())?;
        files.push(path);
        Ok(())
    };
    let preamble: Vec<String> = CELL_COMMANDS.iter().map(|c| c.to_string()).collect();
    for group in &summary.groups {
        let featured: Vec<&MethodSummary> = group.methods.iter().filter(|m| m.featured).collect();
        // Means with intervals.
        let mut header = vec!["Method".to_string()];
        header.extend(
            columns
                .iter()
                .map(|m| format!("{}{}", escape(m.name()), unit_note(*m, false))),
        );
        let rows = featured
            .iter()
            .map(|method| {
                let mut row = vec![method_cell(method)];
                row.extend(
                    columns
                        .iter()
                        .map(|m| method.measure(*m).map(ci_cell).unwrap_or_default()),
                );
                row
            })
            .collect();
        write(
            format!("{}.tex", group.group.name()),
            Table {
                comments: table_comments(summary, group, "means with 95% intervals"),
                preamble: preamble.clone(),
                columns: format!("l{}", "r".repeat(columns.len())),
                header,
                rows,
            },
        )?;
        // Differences against each baseline.
        for baseline in &summary.baselines {
            let mut header = vec!["Method".to_string()];
            header.extend(
                columns
                    .iter()
                    .map(|m| format!("{}{}", escape(m.name()), unit_note(*m, true))),
            );
            let rows = featured
                .iter()
                .filter(|method| method.method != *baseline)
                .map(|method| {
                    let mut row = vec![method_cell(method)];
                    row.extend(columns.iter().map(|m| {
                        method
                            .measure(*m)
                            .and_then(|s| s.versus(baseline))
                            .map(|c| diff_cell(*m, c))
                            .unwrap_or_default()
                    }));
                    row
                })
                .collect();
            write(
                format!("{}-minus-{baseline}.tex", group.group.name()),
                Table {
                    comments: table_comments(
                        summary,
                        group,
                        &format!("method minus {baseline}, 95% intervals and paired randomization p"),
                    ),
                    preamble: preamble.clone(),
                    columns: format!("l{}", "r".repeat(columns.len())),
                    header,
                    rows,
                },
            )?;
        }
        // Families of one kind that differ in one or two parameters.
        for family in families(&group.methods) {
            let comments = table_comments(
                summary,
                group,
                &format!("{} by {}", family.title, family.axes.join(" and ")),
            );
            match family.axes.as_slice() {
                [axis] => {
                    let mut header = vec![escape(axis)];
                    header.extend(
                        columns
                            .iter()
                            .map(|m| format!("{}{}", escape(m.name()), unit_note(*m, false))),
                    );
                    let rows = family
                        .members
                        .iter()
                        .map(|(values, method)| {
                            let mut row = vec![values[0].to_string()];
                            row.extend(
                                columns
                                    .iter()
                                    .map(|m| method.measure(*m).map(ci_cell).unwrap_or_default()),
                            );
                            row
                        })
                        .collect();
                    write(
                        format!("{}-{}.tex", family.file_stem, group.group.name()),
                        Table {
                            comments,
                            preamble: preamble.clone(),
                            columns: format!("r{}", "r".repeat(columns.len())),
                            header,
                            rows,
                        },
                    )?;
                }
                [row_axis, column_axis] => {
                    let row_values = distinct(family.members.iter().map(|(values, _)| values[0]));
                    let column_values = distinct(family.members.iter().map(|(values, _)| values[1]));
                    for measure in columns {
                        let mut header = vec![format!(
                            "{} \\textbackslash{{}} {}",
                            escape(row_axis),
                            escape(column_axis)
                        )];
                        header.extend(column_values.iter().map(f64::to_string));
                        let rows = row_values
                            .iter()
                            .map(|row_value| {
                                let mut row = vec![row_value.to_string()];
                                row.extend(column_values.iter().map(|column_value| {
                                    family
                                        .members
                                        .iter()
                                        .find(|(values, _)| {
                                            values[0] == *row_value && values[1] == *column_value
                                        })
                                        .and_then(|(_, method)| method.measure(*measure))
                                        .map(|m| format_value(m.mean, measure.unit()))
                                        .unwrap_or_default()
                                }));
                                row
                            })
                            .collect();
                        let mut comments = comments.clone();
                        comments.push(format!(
                            "Cells: mean {}{}; intervals are in summary.json and figures/{}.csv.",
                            measure.name(),
                            unit_note(*measure, false).replace("\\%", "%"),
                            family.kind
                        ));
                        write(
                            format!(
                                "{}-{}-{}.tex",
                                family.file_stem,
                                group.group.name(),
                                measure.name()
                            ),
                            Table {
                                comments,
                                preamble: Vec::new(),
                                columns: format!("r{}", "r".repeat(column_values.len())),
                                header,
                                rows,
                            },
                        )?;
                    }
                }
                _ => {}
            }
        }
    }
    Ok(files)
}

/// Distinct values in ascending order.
fn distinct(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut out: Vec<f64> = values.collect();
    out.sort_by(f64::total_cmp);
    out.dedup();
    out
}

/// Methods of one kind that differ in one or two parameters (the axes), the
/// others held fixed.
struct Family<'a> {
    kind: &'static str,
    /// `<kind>` or `<kind>-<parameter><value>…` for the further parameters held fixed.
    file_stem: String,
    title: String,
    axes: Vec<&'static str>,
    /// Each member's values on the axes, in ascending order.
    members: Vec<(Vec<f64>, &'a MethodSummary)>,
}

/// For each kind with more than one method: the first two parameters whose
/// values differ are the axes; when further parameters differ too, the
/// methods are split by those values, one family each.
fn families(methods: &[MethodSummary]) -> Vec<Family<'_>> {
    let values_of = |m: &MethodSummary| -> Vec<f64> { m.spec.parameters().iter().map(|(_, v)| *v).collect() };
    let mut out = Vec::new();
    for kind in KINDS {
        let members: Vec<&MethodSummary> = methods.iter().filter(|m| m.spec.kind() == kind).collect();
        let Some(&first) = members.first() else {
            continue;
        };
        let names: Vec<&'static str> = first.spec.parameters().iter().map(|(name, _)| *name).collect();
        let first_values = values_of(first);
        let varying: Vec<usize> = (0..names.len())
            .filter(|i| {
                members
                    .iter()
                    .any(|m| values_of(m)[*i].total_cmp(&first_values[*i]).is_ne())
            })
            .collect();
        if varying.is_empty() {
            continue;
        }
        let axes: Vec<usize> = varying.iter().copied().take(2).collect();
        let held: Vec<usize> = varying.iter().copied().skip(2).collect();
        let mut by_held: BTreeMap<String, Vec<&MethodSummary>> = BTreeMap::new();
        for &member in &members {
            let own = values_of(member);
            let mut key = String::new();
            for i in &held {
                let _ = write!(key, "-{}{}", names[*i], own[*i]);
            }
            by_held.entry(key).or_default().push(member);
        }
        for (suffix, group) in by_held {
            let mut members: Vec<(Vec<f64>, &MethodSummary)> = group
                .into_iter()
                .map(|m| {
                    let own = values_of(m);
                    (axes.iter().map(|i| own[*i]).collect(), m)
                })
                .collect();
            members.sort_by(|a, b| {
                a.0.iter()
                    .zip(&b.0)
                    .map(|(x, y)| x.total_cmp(y))
                    .find(|o| o.is_ne())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            out.push(Family {
                kind,
                file_stem: format!("{kind}{suffix}"),
                title: format!("{kind}{}", suffix.replace('-', " ")),
                axes: axes.iter().map(|i| names[*i]).collect(),
                members,
            });
        }
    }
    out
}

// ---- numbers -----------------------------------------------------------------------

/// Every number of the report as a macro. Keys:
/// `<name>/seed`, `<name>/draws`, `<name>/runs`;
/// `<name>/<group>/questions`, `<name>/<group>/clusters`;
/// `<name>/<method>/<group>/<measure>` (the mean) with `/lo` and `/hi`;
/// `<name>/<method>/<group>/<measure>/minus/<baseline>` (the difference) with `/lo`, `/hi` and `/p`;
/// `<name>/tuned/<label>` (the chosen method) with `/<parameter>` for each of its parameters.
pub fn numbers_of(summary: &Summary) -> Result<Numbers, EvalError> {
    let name = &summary.name;
    let mut numbers = Numbers::new(vec![
        format!(
            "Written by {} (cb-eval report). Do not edit; run the report again.",
            summary.made_by
        ),
        format!(
            "{name}: test split of runs {}; seed {}, {} draws.",
            summary
                .runs
                .iter()
                .map(|r| r.label.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            summary.seed,
            summary.draws
        ),
        "Shares are in percent with one decimal (differences in points); ndcg and mrr with three decimals."
            .to_string(),
    ]);
    numbers.push(format!("{name}/seed"), summary.seed.to_string())?;
    numbers.push(format!("{name}/draws"), summary.draws.to_string())?;
    numbers.push(format!("{name}/runs"), summary.runs.len().to_string())?;
    for choice in &summary.tuned {
        let key = format!("{name}/tuned/{}", choice.label);
        let chosen = choice.candidates.iter().find(|c| c.method == choice.chosen);
        numbers.push(
            key.clone(),
            format!(
                "\\texttt{{{}}}",
                escape(chosen.map_or(choice.chosen.as_str(), |c| c.text.as_str()))
            ),
        )?;
        let spec = summary
            .groups
            .iter()
            .flat_map(|g| g.methods.iter())
            .find(|m| m.method == choice.chosen)
            .map(|m| m.spec);
        if let Some(spec) = spec {
            for (parameter, value) in spec.parameters() {
                numbers.push(format!("{key}/{parameter}"), value.to_string())?;
            }
        }
    }
    for group in &summary.groups {
        let g = group.group.name();
        numbers.push(
            format!("{name}/{g}/questions"),
            group.questions.iter().sum::<usize>().to_string(),
        )?;
        numbers.push(
            format!("{name}/{g}/clusters"),
            group.clusters.iter().sum::<usize>().to_string(),
        )?;
        for method in &group.methods {
            for m in &method.measures {
                let unit = m.measure.unit();
                let key = format!("{name}/{}/{g}/{}", method.method, m.measure.name());
                numbers.push(key.clone(), format_value(m.mean, unit))?;
                numbers.push(format!("{key}/lo"), format_value(m.ci.lo, unit))?;
                numbers.push(format!("{key}/hi"), format_value(m.ci.hi, unit))?;
                for c in &m.versus {
                    let key = format!("{key}/minus/{}", c.baseline);
                    numbers.push(key.clone(), format_difference(c.difference, unit))?;
                    numbers.push(format!("{key}/lo"), format_difference(c.ci.lo, unit))?;
                    numbers.push(format!("{key}/hi"), format_difference(c.ci.hi, unit))?;
                    numbers.push(format!("{key}/p"), format_p(c.p))?;
                }
            }
        }
    }
    Ok(numbers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::tune::{self, GuardArgs, TuneArgs};
    use crate::files::test_dir;
    use crate::scores::testing::{Row, write_scores};
    use cb_types::QueryClass;
    use std::fs;

    // Threads inc-0001 and inc-0003 are test threads, inc-0006 a validation thread; q-ord-1 and
    // q-ord-2 hash to test, q-ord-6 to validation (see aggregate.rs).
    fn rows(test: [f64; 3], ordinary: f64) -> Vec<Row<'static>> {
        vec![
            ("t1", QueryClass::Changing, Some("inc-0001"), test[0]),
            ("t2", QueryClass::Changing, Some("inc-0001"), test[1]),
            ("t3", QueryClass::Changing, Some("inc-0003"), test[2]),
            ("v1", QueryClass::Changing, Some("inc-0006"), 1.0),
            ("q-ord-1", QueryClass::Ordinary, None, ordinary),
            ("q-ord-2", QueryClass::Ordinary, None, ordinary),
            ("q-ord-6", QueryClass::Ordinary, None, 1.0),
        ]
    }

    fn args(runs: Vec<PathBuf>, out: &Path) -> ReportArgs {
        ReportArgs {
            runs,
            out: out.to_path_buf(),
            name: "tiny".to_string(),
            methods: None,
            baselines: vec!["product".parse().unwrap()],
            tuned: None,
            feature: Vec::new(),
            columns: vec![Measure::Current1, Measure::Hit4],
            seed: 7,
            draws: 2_000,
            threads: 2,
        }
    }

    #[test]
    fn means_differences_and_intervals_follow_the_scores() {
        let dir = test_dir("report-means");
        // Changing test questions: t1, t2 in thread inc-0001 and t3 in inc-0003.
        // product scores 0, 0, 0 → mean 0; rrf_td scores 1, 0, 1 → mean 2/3.
        write_scores(&dir, "product", &rows([0.0, 0.0, 0.0], 1.0), "a");
        write_scores(&dir, "rrf_td(k=60,h=1800)", &rows([1.0, 0.0, 1.0], 0.5), "a");
        let (summary, notes) = summarise(&args(vec![dir.clone()], &dir)).unwrap();
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(summary.split, Split::Test);
        let groups: Vec<Group> = summary.groups.iter().map(|g| g.group).collect();
        assert_eq!(groups, [Group::Changing, Group::Ordinary, Group::OrdinaryAll]);
        let changing = summary.group(Group::Changing).unwrap();
        assert_eq!(
            (changing.questions.as_slice(), changing.clusters.as_slice()),
            ([3].as_slice(), [2].as_slice())
        );
        let product = changing
            .method("product")
            .unwrap()
            .measure(Measure::Current1)
            .unwrap();
        assert_eq!((product.mean, product.ci.lo, product.ci.hi), (0.0, 0.0, 0.0));
        assert!(product.versus.is_empty());
        let paper = changing
            .method("rrf_td-k60-h1800")
            .unwrap()
            .measure(Measure::Current1)
            .unwrap();
        assert_eq!(paper.mean, 2.0 / 3.0);
        assert_eq!(paper.per_run, [2.0 / 3.0]);
        // A resample draws two of the two threads: inc-0001 (2 questions, total 1) and inc-0003
        // (1 question, total 1). Both twice: 2/4 = 0.5; once each: 2/3; inc-0003 twice: 2/2 = 1.
        assert_eq!((paper.ci.lo, paper.ci.hi), (0.5, 1.0));
        let versus = paper.versus("product").unwrap();
        assert_eq!(versus.difference, 2.0 / 3.0);
        assert_eq!((versus.ci.lo, versus.ci.hi), (0.5, 1.0));
        // Both clusters differ by +1 in total. The four labellings give (1 + 1)/3 = 2/3 (none swapped),
        // (-1 + 1)/3 = 0 and (1 - 1)/3 = 0 (one swapped) and -2/3 (both): half are as far from zero as
        // the observed 2/3, so p is about 0.5.
        assert!((versus.p - 0.5).abs() < 0.05, "{}", versus.p);
        // Ordinary questions without a thread are each their own cluster.
        let ordinary = summary.group(Group::OrdinaryAll).unwrap();
        assert_eq!(
            (ordinary.questions.as_slice(), ordinary.clusters.as_slice()),
            ([2].as_slice(), [2].as_slice())
        );
        let hit = ordinary
            .method("rrf_td-k60-h1800")
            .unwrap()
            .measure(Measure::Hit4)
            .unwrap();
        assert_eq!((hit.mean, hit.versus("product").unwrap().difference), (0.5, -0.5));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_report_is_byte_for_byte_the_same_whatever_the_threads_and_run_order() {
        let root = test_dir("report-repeat");
        let (a, b) = (root.join("seed-1"), root.join("seed-2"));
        for (dir, tag) in [(&a, "a"), (&b, "b")] {
            write_scores(dir, "product", &rows([0.0, 1.0, 0.0], 1.0), tag);
            write_scores(dir, "dense", &rows([1.0, 1.0, 0.0], 0.0), tag);
            write_scores(dir, "rrf_td(k=60,h=1800)", &rows([1.0, 0.0, 1.0], 0.5), tag);
        }
        let read = |out: &Path| -> Vec<(String, Vec<u8>)> {
            let mut files: Vec<(String, Vec<u8>)> = Vec::new();
            for sub in ["", FIGURES_DIR, TABLES_DIR] {
                let mut entries: Vec<PathBuf> = fs::read_dir(out.join(sub))
                    .unwrap()
                    .map(|e| e.unwrap().path())
                    .filter(|p| p.is_file())
                    .collect();
                entries.sort();
                for path in entries {
                    files.push((
                        path.file_name().unwrap().to_string_lossy().into_owned(),
                        fs::read(&path).unwrap(),
                    ));
                }
            }
            files
        };
        let mut one = args(vec![a.clone(), b.clone()], &root.join("one"));
        one.threads = 1;
        run(&one).unwrap();
        let mut many = args(vec![b.clone(), a.clone()], &root.join("many"));
        many.threads = 3;
        run(&many).unwrap();
        let (first, second) = (read(&root.join("one")), read(&root.join("many")));
        assert!(first.len() > 10, "{}", first.len());
        assert_eq!(first, second);
        // Another seed moves the intervals.
        let mut reseeded = args(vec![a, b], &root.join("reseeded"));
        reseeded.seed = 8;
        run(&reseeded).unwrap();
        let summary = |dir: &str| fs::read(root.join(dir).join(SUMMARY)).unwrap();
        assert_ne!(summary("one"), summary("reseeded"));
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn tuned_choices_decide_which_candidates_are_featured_and_their_parameters_become_numbers() {
        let dir = test_dir("report-tuned");
        // On validation (thread inc-0006) h=60 scores 1, h=300 scores 0: h=60 is chosen.
        let with_validation = |test: [f64; 3], validation: f64| -> Vec<Row<'static>> {
            let mut r = rows(test, 0.0);
            r[3].3 = validation;
            r
        };
        write_scores(&dir, "product", &with_validation([0.0, 0.0, 0.0], 0.0), "a");
        write_scores(
            &dir,
            "product_decay(h=60)",
            &with_validation([0.0, 1.0, 0.0], 1.0),
            "a",
        );
        write_scores(
            &dir,
            "product_decay(h=300)",
            &with_validation([1.0, 1.0, 1.0], 0.0),
            "a",
        );
        let choice = tune::run(&TuneArgs {
            runs: vec![dir.clone()],
            out: dir.clone(),
            label: "product_decay".to_string(),
            candidates: MethodSpec::parse_grid("product_decay(h=60|300)").unwrap(),
            measure: Measure::Current1,
            group: Group::Changing,
            higher_is_better: None,
            guard: None::<GuardArgs>,
        })
        .unwrap();
        assert_eq!(choice.chosen, "product_decay-h60");
        let mut with_tuned = args(vec![dir.clone()], &dir);
        with_tuned.tuned = Some(dir.join(crate::files::TUNED));
        let outcome = run(&with_tuned).unwrap();
        let changing = outcome.summary.group(Group::Changing).unwrap();
        let featured: Vec<(&str, bool)> = changing
            .methods
            .iter()
            .map(|m| (m.method.as_str(), m.featured))
            .collect();
        // The candidate that was not chosen is reported but not featured, even though it scores
        // better on the test split.
        assert_eq!(
            featured,
            [
                ("product", true),
                ("product_decay-h60", true),
                ("product_decay-h300", false)
            ]
        );
        assert_eq!(changing.methods[1].chosen_by, ["product_decay"]);
        let main = fs::read_to_string(dir.join(TABLES_DIR).join("changing.tex")).unwrap();
        assert!(
            main.contains("\\texttt{product\\_decay(h=60)}$^{\\dagger}$"),
            "{main}"
        );
        assert!(!main.contains("h=300"), "{main}");
        let numbers = numbers_of(&outcome.summary).unwrap();
        assert_eq!(
            numbers.get("tiny/tuned/product_decay"),
            Some("\\texttt{product\\_decay(h=60)}")
        );
        assert_eq!(numbers.get("tiny/tuned/product_decay/h"), Some("60"));
        assert_eq!(numbers.get("tiny/tuned/product_decay/depth"), Some("10"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_output_is_written_with_the_expected_names_and_numbers() {
        let dir = test_dir("report-files");
        write_scores(&dir, "product", &rows([0.0, 0.0, 1.0], 1.0), "a");
        for k in [10, 60] {
            for h in [60, 1800] {
                write_scores(
                    &dir,
                    &format!("rrf_td(k={k},h={h})"),
                    &rows([1.0, 0.0, 1.0], 0.5),
                    "a",
                );
            }
        }
        write_scores(&dir, "cc(alpha=0.2)", &rows([1.0, 1.0, 1.0], 0.5), "a");
        write_scores(&dir, "cc(alpha=0.7)", &rows([0.0, 1.0, 1.0], 0.5), "a");
        let outcome = run(&args(vec![dir.clone()], &dir)).unwrap();
        let names: Vec<String> = outcome
            .files
            .iter()
            .map(|p| p.strip_prefix(&dir).unwrap().to_string_lossy().into_owned())
            .collect();
        for expected in [
            "summary.json",
            "figures/results.csv",
            "figures/comparisons.csv",
            "figures/per_run.csv",
            "figures/rrf_td.csv",
            "figures/cc.csv",
            "figures/product.csv",
            "tables/changing.tex",
            "tables/changing-minus-product.tex",
            "tables/cc-changing.tex",
            "tables/rrf_td-changing-current@1.tex",
            "tables/rrf_td-ordinary_all-hit@4.tex",
            "numbers.tex",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "{expected} missing from {names:?}"
            );
        }
        // The k x h grid: rows k = 10, 60; columns h = 60, 1800. t1, t3 score 1, t2 0: 66.7%.
        let grid = fs::read_to_string(dir.join("tables/rrf_td-changing-current@1.tex")).unwrap();
        assert!(grid.contains("k \\textbackslash{} h & 60 & 1800 \\\\"), "{grid}");
        assert!(grid.contains("10 & 66.7 & 66.7 \\\\"), "{grid}");
        // The sweep of cc over alpha.
        let sweep = fs::read_to_string(dir.join("tables/cc-changing.tex")).unwrap();
        assert!(sweep.contains("0.2 & \\cbci{100.0}{100.0}{100.0}"), "{sweep}");
        // Numbers: product's current@1 on changing is 1/3 = 33.3%; the paper's setting minus product
        // is 66.7 - 33.3 = +33.3 points.
        let numbers = fs::read_to_string(dir.join(NUMBERS)).unwrap();
        assert!(numbers.contains("\\csname cbnum/tiny/product/changing/current@1\\endcsname{33.3}"));
        assert!(numbers.contains(
            "\\csname cbnum/tiny/rrf_td-k60-h1800/changing/current@1/minus/product\\endcsname{+33.3}"
        ));
        assert!(numbers.contains("\\csname cbnum/tiny/changing/questions\\endcsname{3}"));
        assert!(numbers.contains("\\csname cbnum/tiny/seed\\endcsname{7}"));
        let results = fs::read_to_string(dir.join("figures/results.csv")).unwrap();
        assert!(
            results
                .starts_with("name,group,method,text,kind,featured,measure,mean,lo,hi,questions,clusters\n")
        );
        assert!(results.contains(
            "tiny,changing,rrf_td-k60-h1800,\"rrf_td(k=60,h=1800)\",rrf_td,true,current@1,0.6666666666666666,"
        ));
        let kind = fs::read_to_string(dir.join("figures/rrf_td.csv")).unwrap();
        assert!(
            kind.starts_with("group,measure,method,k,h,w_dense,w_sparse,w_time,list,eps,theta,mean,lo,hi\n")
        );
        assert_eq!(
            outcome.numbers,
            numbers.lines().filter(|l| l.starts_with("\\expandafter")).count()
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_report_needs_a_plain_name_draws_and_scores() {
        let dir = test_dir("report-refuse");
        write_scores(&dir, "product", &rows([0.0, 0.0, 1.0], 1.0), "a");
        let mut bad = args(vec![dir.clone()], &dir);
        bad.name = "tiny/run".to_string();
        assert!(summarise(&bad).is_err());
        let mut no_draws = args(vec![dir.clone()], &dir);
        no_draws.draws = 0;
        assert!(summarise(&no_draws).is_err());
        let mut missing_baseline = args(vec![dir.clone()], &dir);
        missing_baseline.baselines = vec!["dense".parse().unwrap()];
        assert!(
            summarise(&missing_baseline)
                .unwrap_err()
                .to_string()
                .contains("dense.json")
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_group_held_by_only_some_runs_is_left_out_with_a_note() {
        let root = test_dir("report-partial");
        let (a, b) = (root.join("seed-1"), root.join("seed-2"));
        write_scores(&a, "product", &rows([0.0, 0.0, 1.0], 1.0), "a");
        // seed-2 has no ordinary questions at all.
        write_scores(&b, "product", &rows([0.0, 0.0, 1.0], 1.0)[..4], "b");
        let (summary, notes) = summarise(&args(vec![a, b], &root)).unwrap();
        let groups: Vec<Group> = summary.groups.iter().map(|g| g.group).collect();
        assert_eq!(groups, [Group::Changing]);
        assert_eq!(notes.len(), 2, "{notes:?}");
        assert!(notes[0].contains("group ordinary is left out"));
        fs::remove_dir_all(&root).unwrap();
    }
}

#[cfg(test)]
mod speed {
    use super::*;
    use crate::aggregate::Cluster;
    use crate::measures::Measures;
    use crate::scores::{Asked, LoadedMethod};
    use crate::stats::{DEFAULT_DRAWS, DEFAULT_SEED, stream, uniform_below};
    use cb_types::QueryClass;
    use std::time::Instant;

    /// Five runs of 400 incidents with 10 changing questions each and 40 methods, at the
    /// default 10 000 draws: the size of the generated corpus's test split.
    #[test]
    #[ignore = "a timing at full size: cargo test --release -p cb-eval --lib speed -- --ignored --nocapture"]
    fn one_group_at_full_size() {
        let (runs_wanted, threads, per_thread, methods) = (5, 400, 10, 40u32);
        let mut rng = stream(1, "speed");
        let specs: Vec<MethodSpec> = (0..methods)
            .map(|m| MethodSpec::Product {
                depth: 10 + m,
                recent: 0,
            })
            .collect();
        let runs: Vec<Run> = (0..runs_wanted)
            .map(|r| Run {
                label: format!("seed-{r}"),
                dir: PathBuf::new(),
                split: Split::Test,
                inputs: BTreeMap::new(),
                questions: (0..threads * per_thread)
                    .map(|q| Asked {
                        query: format!("q{q}"),
                        class: QueryClass::Changing,
                        cluster: Cluster::Thread(format!("t{}", q / per_thread)),
                    })
                    .collect(),
                methods: specs
                    .iter()
                    .map(|spec| {
                        let measures = (0..threads * per_thread)
                            .map(|_| Measures::from_fn(|_| uniform_below(&mut rng, 2) as f64))
                            .collect();
                        (
                            spec.name(),
                            LoadedMethod {
                                text: spec.to_string(),
                                spec: *spec,
                                measures,
                            },
                        )
                    })
                    .collect(),
            })
            .collect();
        let listed: Vec<Listed> = specs
            .iter()
            .map(|spec| Listed {
                name: spec.name(),
                text: spec.to_string(),
                spec: *spec,
                featured: true,
                chosen_by: Vec::new(),
            })
            .collect();
        let args = ReportArgs {
            runs: Vec::new(),
            out: PathBuf::new(),
            name: "speed".to_string(),
            methods: None,
            baselines: Vec::new(),
            tuned: None,
            feature: Vec::new(),
            columns: DEFAULT_COLUMNS.to_vec(),
            seed: DEFAULT_SEED,
            draws: DEFAULT_DRAWS,
            threads: 0,
        };
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let started = Instant::now();
        let layout = GroupLayout::of(&runs, Group::Changing).unwrap().unwrap();
        let summary =
            summarise_group(&args, &runs, layout, &listed, &["product".to_string()], cores).unwrap();
        let elapsed = started.elapsed();
        assert_eq!(summary.methods.len(), methods as usize);
        println!(
            "{} methods x {} measures, {} runs x {threads} clusters, {} draws, {cores} threads: {:.2} s",
            methods,
            Measure::ALL.len(),
            runs_wanted,
            DEFAULT_DRAWS,
            elapsed.as_secs_f64()
        );
    }
}
