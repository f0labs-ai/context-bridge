//! `cb-eval gate`: the two-conversation rule over a run's extractions
//! (DESIGN.md section 6), every variant, the likeness bar chosen on the
//! validation split and the test split reported.
//!
//! Inputs: `records.jsonl` (the planted truth) and the transcripts the
//! extractions were made from (`transcripts.jsonl` of the data set unless
//! named) from the data set; `extractions.jsonl` and
//! `extraction_vectors.f32` + `extraction_vectors.json` (one vector per
//! extraction, keyed by its transcript id) from the run.
//!
//! Splits are by incident, as for questions: the extractions of each split
//! are replayed on their own. Outputs, under `<out>/gate/`: `summary.json`,
//! `variants.csv`, `extractions.jsonl` (the verdict on every extraction),
//! `judge_sample.jsonl` (the extractions a language-model judge re-scores),
//! `table.tex` and `numbers.tex` (test split only).

use crate::error::EvalError;
use crate::files::{
    EXTRACTION_VECTORS, EXTRACTIONS, RECORDS, TRANSCRIPTS, made_by, read_jsonl_file, read_vectors,
    sha256_of_file, write_json_file, write_jsonl_file, write_text_file,
};
use crate::gate::{
    Agreement, GateOutcome, GateRule, IncidentTally, Independence, Item, JudgeAgreement, JudgeVerdict,
    Likeness, MAX_CONVERSATIONS, PairCounts, Stated, Truth, agreement_of, call_end, group_is_correct,
    judge_group, merged_parts, normalise, pair_counts, precision_gain_interval, predicted_precision_two,
    replay, score, split_of_thread, stated_by, statement_moment,
};
use crate::latex::{Numbers, Table, csv_line, csv_number, escape, format_value, is_plain_name};
use crate::measures::Unit;
use crate::parallel::{map_in_parallel, threads_to_use};
use crate::stats::{DEFAULT_DRAWS, Interval, stream, uniform_below};
use cb_types::{Extraction, Record, Split, Transcript};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// The likeness bar of the first paper.
pub const PAPER_BAR: f64 = 0.92;
/// The confidence cut of the first paper.
pub const PAPER_CONFIDENCE_CUT: f64 = 0.85;
/// The share of true states a chosen bar must validate on the validation split (claim 3's pass line).
pub const DEFAULT_MIN_SHARE: f64 = 0.70;
/// How many extractions the language-model judge re-scores.
pub const DEFAULT_JUDGE_SAMPLE: usize = 500;
/// Where the outputs go, under the output directory.
pub const GATE_DIR: &str = "gate";

/// The bars tried on the validation split: 0.80, 0.81, …, 0.98.
pub fn default_bars() -> Vec<f64> {
    (80..=98).map(|b| f64::from(b) / 100.0).collect()
}

#[derive(Clone, Debug)]
pub struct GateArgs {
    pub data: PathBuf,
    pub run: PathBuf,
    /// The transcripts the extractions were made from; `<data>/transcripts.jsonl` when `None`.
    pub transcripts: Option<PathBuf>,
    pub out: PathBuf,
    /// The first part of every key in `numbers.tex`.
    pub name: String,
    /// How many conversations must agree, each a variant: 1, 2, 3 by default.
    pub conversations: Vec<usize>,
    /// The bars tried on the validation split.
    pub bars: Vec<f64>,
    pub min_share: f64,
    pub seed: u64,
    pub judge_sample: usize,
    /// The judge's verdicts, one [`JudgeVerdict`] per line.
    pub judge: Option<PathBuf>,
    /// The judge's verdicts on `judge_sample_promoted.jsonl`, by sample id.
    pub judge_promoted: Option<PathBuf>,
    pub threads: usize,
    /// The share of calls kept, above 0 and at most 1: a lower call rate, for the wait
    /// for a second conversation. 1 keeps every call.
    pub keep_calls: f64,
}

/// Whether a call stays when calls are thinned to the share `keep`: a seeded hash of its
/// id below `keep` of the hash's range. The same calls stay on every run, and the calls a
/// smaller share keeps are among those a larger share keeps.
pub fn call_is_kept(seed: u64, keep: f64, id: &str) -> bool {
    if keep >= 1.0 {
        return true;
    }
    let h = mix64(cb_types::fnv1a64(format!("{seed}/keep-calls/{id}").as_bytes()));
    (h as f64) < keep * (u64::MAX as f64)
}

/// MurmurHash3's 64-bit finalizer: every input bit reaches every output bit. FNV-1a alone
/// leaves the high bits of ids that differ in their last digits close together (2,000
/// ids kept at 0.5 came out 820).
fn mix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// Where a variant's bar came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BarChoice {
    /// One conversation: no likeness is compared.
    NotUsed,
    /// The first paper's 0.92.
    Paper,
    /// The bar of the grid that did best on the validation split.
    ChosenOnValidation,
}

/// One bar tried on the validation split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridPoint {
    pub bar: f64,
    pub promoted: usize,
    pub precision: Option<f64>,
    pub share_validated: Option<f64>,
    /// Over the true records stated in two or more calls: what the bar is chosen by.
    #[serde(default)]
    pub share_validated_in_two_calls: Option<f64>,
}

/// One variant with what it achieved on each split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    pub name: String,
    /// The rule, with the bar used on the test split.
    pub rule: GateRule,
    pub bar_choice: BarChoice,
    /// For a chosen bar: every bar tried on the validation split.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub validation_grid: Vec<GridPoint>,
    pub validation: GateOutcome,
    pub test: GateOutcome,
}

/// What one split holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitSummary {
    pub split: Split,
    pub extractions: usize,
    /// Extractions that state a true record of their call and no wrong statement.
    pub correct_extractions: usize,
    /// Calls of the split's incidents in the transcripts file without an extraction.
    pub calls_without_extraction: usize,
    /// True records the covered calls state.
    pub states: usize,
    /// Of those, the ones two or more covered calls state (the pass line's denominator).
    #[serde(default)]
    pub states_in_two_calls: usize,
    /// Statements whose moment is the end of the call because the key terms never all appeared.
    pub statement_moments_at_call_end: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeSummary {
    /// How many extractions were drawn for the judge.
    pub sample: usize,
    /// The judge's verdicts on the sample, when a judge file was given.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agreement: Option<JudgeAgreement>,
    /// Verdicts on extractions outside the sample (left out of the agreement).
    pub verdicts_outside_sample: usize,
}

fn every_call() -> f64 {
    1.0
}

/// `gate/summary.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateSummary {
    pub name: String,
    pub made_by: String,
    pub seed: u64,
    /// File name → sha256 of every input.
    pub inputs: BTreeMap<String, String>,
    pub splits: Vec<SplitSummary>,
    /// Extractions that found nothing (no problem, no solution): no record, left out of
    /// every split. Their calls count among `calls_without_extraction`.
    #[serde(default)]
    pub extractions_that_found_nothing: usize,
    /// The share of calls kept, and the calls thinned out (with their extractions).
    #[serde(default = "every_call")]
    pub keep_calls: f64,
    #[serde(default)]
    pub calls_thinned_out: usize,
    /// Planted wrong statements without key terms: never recognised.
    pub wrong_statements_without_key_terms: usize,
    /// True records without key terms: stated by any extraction of a call that states them.
    pub true_records_without_key_terms: usize,
    pub bars_tried: Vec<f64>,
    pub min_share: f64,
    /// How a bar is chosen, in words.
    pub choice: String,
    pub judge: JudgeSummary,
    /// The judge on a sample of the headline variant's promoted groups (exploratory).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_promoted: Option<PromotedJudgeSummary>,
    /// The supplement's error model with measured inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_model: Option<ErrorModel>,
    /// Every variant's gain in precision over [`SINGLE_VARIANT`], with its interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision_gains: Option<PrecisionGains>,
    pub variants: Vec<Variant>,
}

/// The variant of every extraction on its own: one conversation, no confidence cut.
pub const SINGLE_VARIANT: &str = "n1-nocut";

/// On the test split: the inputs of the supplement's error model, measured at the headline
/// variant's bar, and what the model predicts from them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ErrorModel {
    pub variant: String,
    pub bar: f64,
    pub pairs: PairCounts,
    /// The precision of one extraction ([`SINGLE_VARIANT`]).
    pub precision_single: Option<f64>,
    /// The shares of pairs more alike than the bar: both correct with one state (`m_c`),
    /// both wrong (`m_w`), one of each (`m_x`).
    pub match_rate_same_state: Option<f64>,
    pub match_rate_wrong_wrong: Option<f64>,
    pub match_rate_correct_wrong: Option<f64>,
    /// Equation (p2) with those inputs, and the headline variant's measured precision.
    pub precision_two_predicted: Option<f64>,
    pub precision_two: Option<f64>,
}

/// On the test split: each variant's precision minus [`SINGLE_VARIANT`]'s, with a 95%
/// interval by a bootstrap over incidents. Every variant is measured on the same resamples,
/// and a group counts under the incident of its first member.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrecisionGains {
    pub versus: String,
    pub incidents: usize,
    pub draws: usize,
    pub variants: Vec<PrecisionGain>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrecisionGain {
    pub variant: String,
    pub gain: Option<f64>,
    pub interval: Option<Interval>,
}

impl PrecisionGains {
    pub fn of(&self, variant: &str) -> Option<&PrecisionGain> {
        self.variants.iter().find(|g| g.variant == variant)
    }
}

/// The verdict on one extraction: `gate/extractions.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractionVerdict {
    pub transcript: String,
    pub thread: String,
    pub split: Split,
    pub agent: String,
    /// Seconds since the epoch.
    pub arrives_at: f64,
    /// The product of the three parts.
    pub confidence: f32,
    /// `true`, `wrong` or `nothing`.
    pub stated: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record: Option<String>,
    pub correct: bool,
}

/// One extraction as the judge is shown it: `gate/judge_sample.jsonl`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeItem {
    pub transcript: String,
    pub problem: String,
    pub solution: String,
    /// For a promoted group: the calls of its members, whose true statements the judge
    /// is shown together. Empty for one extraction (`transcript` is its call).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<String>,
}

/// The variant whose promoted groups the judge also re-scores: two conversations, every
/// pair above the bar, different calls, no confidence cut, bar chosen on validation.
pub const HEADLINE_VARIANT: &str = "n2-pairs-calls-nocut-chosen";

/// What the judge said about a sample of the headline variant's promoted groups on the
/// test split (exploratory; the key-term rule decides the claim).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromotedJudgeSummary {
    pub variant: String,
    /// Groups the variant promoted on the test split, and how many were drawn.
    pub groups: usize,
    pub sample: usize,
    /// With the judge's verdicts: how many of the sample it judged, found correct, and
    /// found correct among the groups that do not mix two statements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judged: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_correct: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_correct_not_mixed: Option<usize>,
    /// The key-term rule against the judge on the sample (a mixed group is never
    /// correct for the rule).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agreement: Option<JudgeAgreement>,
}

/// The extractions of one split, ready to replay.
struct SplitData {
    split: Split,
    items: Vec<Item>,
    order: Vec<usize>,
    states: BTreeSet<usize>,
    /// The true records two or more covered calls state (the pass line's denominator).
    stated_twice: BTreeSet<usize>,
    first_statement: HashMap<usize, f64>,
    likeness: Likeness,
    summary: SplitSummary,
}

impl SplitData {
    fn outcome(&self, rule: &GateRule, truth: &Truth) -> GateOutcome {
        let promoted = replay(rule, &self.items, &self.order, &self.likeness);
        score(
            rule,
            &self.items,
            &promoted,
            truth,
            (&self.states, &self.stated_twice),
            &self.first_statement,
        )
    }
}

/// What `gate` did.
#[derive(Debug)]
pub struct GateRun {
    pub summary: GateSummary,
    pub files: Vec<PathBuf>,
    pub numbers: usize,
}

fn check_args(args: &GateArgs) -> Result<(), EvalError> {
    if !is_plain_name(&args.name) {
        return Err(EvalError::invalid(format!(
            "gate: name {:?} may hold letters, digits, '-', '_' and '.' only",
            args.name
        )));
    }
    if args.conversations.is_empty()
        || args
            .conversations
            .iter()
            .any(|n| !(1..=MAX_CONVERSATIONS).contains(n))
    {
        return Err(EvalError::invalid(format!(
            "gate: conversations must be between 1 and {MAX_CONVERSATIONS}"
        )));
    }
    if args.bars.is_empty() || args.bars.iter().any(|b| !(-1.0..=1.0).contains(b)) {
        return Err(EvalError::invalid(
            "gate: bars must be between -1 and 1, and at least one is needed",
        ));
    }
    if !(0.0..=1.0).contains(&args.min_share) {
        return Err(EvalError::invalid(
            "gate: the minimum share must be between 0 and 1",
        ));
    }
    if !(args.keep_calls > 0.0 && args.keep_calls <= 1.0) {
        return Err(EvalError::invalid(
            "gate: the share of calls kept is above 0 and at most 1",
        ));
    }
    Ok(())
}

/// Read the inputs, judge every extraction and replay every variant.
pub fn run(args: &GateArgs) -> Result<GateRun, EvalError> {
    check_args(args)?;
    let records_path = args.data.join(RECORDS);
    let transcripts_path = args
        .transcripts
        .clone()
        .unwrap_or_else(|| args.data.join(TRANSCRIPTS));
    let extractions_path = args.run.join(EXTRACTIONS);
    let vectors_stem = args.run.join(EXTRACTION_VECTORS);
    let mut inputs = BTreeMap::new();
    for path in [
        &records_path,
        &transcripts_path,
        &extractions_path,
        &vectors_stem.with_extension("json"),
        &vectors_stem.with_extension("f32"),
    ] {
        let name = path
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        inputs.insert(name, sha256_of_file(path)?);
    }

    let records: Vec<Record> = read_jsonl_file(&records_path)?;
    let truth = Truth::new(&records)?;
    drop(records);
    // Calls thinned out never happened: their transcripts and extractions are dropped
    // together (`keep_calls`, decided by `call_is_kept`).
    let all_calls: Vec<Transcript> = read_jsonl_file(&transcripts_path)?;
    let thinned_out: HashSet<String> = all_calls
        .iter()
        .filter(|t| !call_is_kept(args.seed, args.keep_calls, &t.id))
        .map(|t| t.id.clone())
        .collect();
    let transcripts: Vec<Transcript> = all_calls
        .into_iter()
        .filter(|t| !thinned_out.contains(&t.id))
        .collect();
    let mut call: HashMap<&str, &Transcript> = HashMap::with_capacity(transcripts.len());
    for t in &transcripts {
        if call.insert(t.id.as_str(), t).is_some() {
            return Err(EvalError::invalid(format!(
                "{}: transcript {:?} appears twice",
                transcripts_path.display(),
                t.id
            )));
        }
    }
    // An extraction that found nothing writes no record: it is counted, never grouped or
    // judged, and needs no vector. Its call must still be one of the transcripts.
    let mut extractions_that_found_nothing = 0;
    let mut extractions: Vec<Extraction> = Vec::new();
    for e in read_jsonl_file::<Extraction>(&extractions_path)? {
        if thinned_out.contains(&e.transcript) {
            continue;
        }
        if !(e.problem.trim().is_empty() && e.solution.trim().is_empty()) {
            extractions.push(e);
        } else if call.contains_key(e.transcript.as_str()) {
            extractions_that_found_nothing += 1;
        } else {
            return Err(EvalError::invalid(format!(
                "{}: transcript {:?} is not in {}",
                extractions_path.display(),
                e.transcript,
                transcripts_path.display()
            )));
        }
    }
    let vectors = read_vectors(&vectors_stem)?;
    let vector_rows = vectors.rows_by_id()?;

    let threads = threads_to_use(args.threads);
    let mut seen = BTreeSet::new();
    let mut by_split: HashMap<Split, (Vec<Item>, Vec<usize>)> = HashMap::new();
    let mut verdicts = Vec::with_capacity(extractions.len());
    for e in &extractions {
        let wrong = |what: String| EvalError::invalid(format!("{}: {what}", extractions_path.display()));
        if !seen.insert(e.transcript.as_str()) {
            return Err(wrong(format!(
                "transcript {:?} has two extractions",
                e.transcript
            )));
        }
        let t = call.get(e.transcript.as_str()).ok_or_else(|| {
            wrong(format!(
                "transcript {:?} is not in {}",
                e.transcript,
                transcripts_path.display()
            ))
        })?;
        let row = *vector_rows
            .get(e.transcript.as_str())
            .ok_or_else(|| wrong(format!("no vector for the extraction of {:?}", e.transcript)))?;
        let parts = [e.confidence.problem, e.confidence.solution, e.confidence.time];
        if parts.iter().any(|p| !p.is_finite()) || !e.latency_ms.is_finite() || e.latency_ms < 0.0 {
            return Err(wrong(format!(
                "the extraction of {:?} has a confidence or latency that is not a number",
                e.transcript
            )));
        }
        let mut true_states = Vec::new();
        for id in &t.states {
            let index = truth.index_of(id).ok_or_else(|| {
                EvalError::invalid(format!(
                    "transcript {:?} states {id:?}, which has no planted truth",
                    t.id
                ))
            })?;
            if truth.row(index).correct {
                true_states.push(index);
            }
        }
        let text = normalise(&format!("{} {}", e.problem, e.solution));
        let stated = stated_by(&text, &true_states, truth.wrong_of(&t.thread), &truth);
        let split = split_of_thread(&t.thread);
        let item = Item {
            transcript: t.id.clone(),
            agent: t.agent.clone(),
            thread: t.thread.clone(),
            arrives_at: call_end(t) + e.latency_ms / 1000.0,
            confidence: e.confidence,
            problem: normalise(&e.problem),
            solution: normalise(&e.solution),
            true_states,
            stated,
        };
        verdicts.push(ExtractionVerdict {
            transcript: item.transcript.clone(),
            thread: item.thread.clone(),
            split,
            agent: item.agent.clone(),
            arrives_at: item.arrives_at,
            confidence: e.confidence.product(),
            stated: match stated {
                Stated::True(_) => "true",
                Stated::Wrong(_) => "wrong",
                Stated::Nothing => "nothing",
            }
            .to_string(),
            record: stated.record().map(|r| truth.row(r).id.clone()),
            correct: item.correct(),
        });
        let (items, rows) = by_split.entry(split).or_default();
        items.push(item);
        rows.push(row);
    }

    let mut splits = Vec::new();
    for split in [Split::Validation, Split::Test] {
        let (items, rows) = by_split.remove(&split).unwrap_or_default();
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by(|a, b| {
            items[*a]
                .arrives_at
                .total_cmp(&items[*b].arrives_at)
                .then_with(|| items[*a].transcript.cmp(&items[*b].transcript))
        });
        let mut states = BTreeSet::new();
        let mut first_statement: HashMap<usize, f64> = HashMap::new();
        let mut at_call_end = 0;
        for item in &items {
            let t = call.get(item.transcript.as_str()).copied();
            for record in &item.true_states {
                states.insert(*record);
                let moment = t
                    .and_then(|t| statement_moment(t, &truth.row(*record).key_terms))
                    .unwrap_or_else(|| {
                        at_call_end += 1;
                        t.map_or(item.arrives_at, call_end)
                    });
                let first = first_statement.entry(*record).or_insert(moment);
                *first = first.min(moment);
            }
        }
        let covered: BTreeSet<&str> = items.iter().map(|i| i.transcript.as_str()).collect();
        let calls_without_extraction = transcripts
            .iter()
            .filter(|t| split_of_thread(&t.thread) == split && !covered.contains(t.id.as_str()))
            .count();
        // Each covered call is one item, so counting items counts different calls.
        let mut calls_stating: HashMap<usize, usize> = HashMap::new();
        for item in &items {
            for record in &item.true_states {
                *calls_stating.entry(*record).or_default() += 1;
            }
        }
        let stated_twice: BTreeSet<usize> = calls_stating
            .into_iter()
            .filter(|(_, calls)| *calls >= 2)
            .map(|(record, _)| record)
            .collect();
        let summary = SplitSummary {
            split,
            extractions: items.len(),
            correct_extractions: items.iter().filter(|i| i.correct()).count(),
            calls_without_extraction,
            states: states.len(),
            states_in_two_calls: stated_twice.len(),
            statement_moments_at_call_end: at_call_end,
        };
        splits.push(SplitData {
            split,
            likeness: Likeness::compute(&vectors, &rows, threads)?,
            items,
            order,
            states,
            stated_twice,
            first_statement,
            summary,
        });
    }

    let variants = replay_every_variant(args, &truth, &splits, threads)?;
    let test_split = splits
        .iter()
        .find(|s| s.split == Split::Test)
        .ok_or_else(|| EvalError::invalid("gate: no test split"))?;
    let promoted = promoted_judge_sample(args, test_split, &variants, &truth)?;
    let error_model = error_model(test_split, &variants);
    let precision_gains = precision_gains(args, test_split, &variants, &truth, threads)?;
    let judge = judge_sample_and_agreement(args, &extractions, &verdicts)?;
    let summary = GateSummary {
        name: args.name.clone(),
        made_by: made_by(),
        seed: args.seed,
        inputs,
        splits: splits.iter().map(|s| s.summary.clone()).collect(),
        extractions_that_found_nothing,
        keep_calls: args.keep_calls,
        calls_thinned_out: thinned_out.len(),
        wrong_statements_without_key_terms: truth.wrong_without_terms,
        true_records_without_key_terms: truth.true_without_terms,
        bars_tried: args.bars.clone(),
        min_share: args.min_share,
        choice: format!(
            "On the validation split, the bar with the highest precision among those that validate at least \
             {} of the true states stated in two or more calls (the pass line's denominator); \
             when none does, the bar that validates the most. Ties go to the higher share, then to the lower \
             bar.",
            args.min_share
        ),
        judge: judge.0,
        judge_promoted: promoted.as_ref().map(|(summary, _)| summary.clone()),
        error_model,
        precision_gains,
        variants,
    };

    let dir = args.out.join(GATE_DIR);
    let mut files = Vec::new();
    let mut put = |name: &str, write: &dyn Fn(&Path) -> Result<(), EvalError>| -> Result<(), EvalError> {
        let path = dir.join(name);
        write(&path)?;
        files.push(path);
        Ok(())
    };
    put("summary.json", &|p| write_json_file(p, &summary))?;
    put("variants.csv", &|p| write_text_file(p, &variants_csv(&summary)))?;
    put("extractions.jsonl", &|p| write_jsonl_file(p, &verdicts))?;
    put("judge_sample.jsonl", &|p| write_jsonl_file(p, &judge.1))?;
    if let Some((_, items)) = &promoted {
        put("judge_sample_promoted.jsonl", &|p| write_jsonl_file(p, items))?;
    }
    put("table.tex", &|p| {
        write_text_file(p, &test_table(&summary).render())
    })?;
    let numbers = numbers_of(&summary)?;
    put("numbers.tex", &|p| write_text_file(p, &numbers.render()))?;
    Ok(GateRun {
        summary,
        files,
        numbers: numbers.len(),
    })
}

/// The name of a variant, e.g. `n2-pairs-calls-cut0.85-bar0.92` (the paper's
/// setting) or `n1-nocut` (every extraction on its own).
pub fn variant_name(rule: &GateRule, choice: BarChoice) -> String {
    let cut = rule
        .confidence_cut
        .map_or_else(|| "nocut".to_string(), |c| format!("cut{c}"));
    if rule.conversations == 1 {
        return format!("n1-{cut}");
    }
    let agreement = match rule.agreement {
        Agreement::Pairs => "pairs",
        Agreement::Average => "average",
    };
    let independence = match rule.independence {
        Independence::Calls => "calls",
        Independence::Agents => "agents",
    };
    let bar = match choice {
        BarChoice::Paper => format!("bar{PAPER_BAR}"),
        BarChoice::ChosenOnValidation | BarChoice::NotUsed => "chosen".to_string(),
    };
    format!("n{}-{agreement}-{independence}-{cut}-{bar}", rule.conversations)
}

/// The bar of the grid that did best on validation: the highest precision
/// among bars that validate at least `min_share` of the true states stated in
/// two or more calls (the pass line's denominator); when none
/// does, the highest share. Ties go to the higher share (resp. precision),
/// then to the lower bar.
pub fn choose_bar(grid: &[GridPoint], min_share: f64) -> Option<f64> {
    let share = |p: &GridPoint| p.share_validated_in_two_calls.unwrap_or(0.0);
    let precision = |p: &GridPoint| p.precision.unwrap_or(0.0);
    let lower_bar_wins = |a: &GridPoint, b: &GridPoint| b.bar.total_cmp(&a.bar);
    let qualifying: Vec<&GridPoint> = grid.iter().filter(|p| share(p) >= min_share).collect();
    let best = if qualifying.is_empty() {
        grid.iter().max_by(|a, b| {
            share(a)
                .total_cmp(&share(b))
                .then_with(|| precision(a).total_cmp(&precision(b)))
                .then_with(|| lower_bar_wins(a, b))
        })
    } else {
        qualifying.into_iter().max_by(|a, b| {
            precision(a)
                .total_cmp(&precision(b))
                .then_with(|| share(a).total_cmp(&share(b)))
                .then_with(|| lower_bar_wins(a, b))
        })
    };
    best.map(|p| p.bar)
}

/// The variants of DESIGN.md section 6: for one conversation, with and
/// without the confidence cut; for more, every agreement, independence and
/// cut, each with the paper's bar and with the bar chosen on validation.
fn replay_every_variant(
    args: &GateArgs,
    truth: &Truth,
    splits: &[SplitData],
    threads: usize,
) -> Result<Vec<Variant>, EvalError> {
    let validation = splits
        .iter()
        .find(|s| s.split == Split::Validation)
        .ok_or_else(|| EvalError::invalid("gate: no validation split"))?;
    let test = splits
        .iter()
        .find(|s| s.split == Split::Test)
        .ok_or_else(|| EvalError::invalid("gate: no test split"))?;
    let mut bars = args.bars.clone();
    bars.sort_by(f64::total_cmp);
    bars.dedup();
    let mut planned: Vec<(GateRule, BarChoice)> = Vec::new();
    let mut conversations = args.conversations.clone();
    conversations.sort_unstable();
    conversations.dedup();
    for n in conversations {
        for cut in [Some(PAPER_CONFIDENCE_CUT), None] {
            if n == 1 {
                let rule = GateRule {
                    conversations: 1,
                    agreement: Agreement::Pairs,
                    independence: Independence::Calls,
                    confidence_cut: cut,
                    bar: PAPER_BAR,
                };
                planned.push((rule, BarChoice::NotUsed));
                continue;
            }
            for agreement in [Agreement::Pairs, Agreement::Average] {
                for independence in [Independence::Calls, Independence::Agents] {
                    let rule = GateRule {
                        conversations: n,
                        agreement,
                        independence,
                        confidence_cut: cut,
                        bar: PAPER_BAR,
                    };
                    planned.push((rule, BarChoice::Paper));
                    if !validation.items.is_empty() {
                        planned.push((rule, BarChoice::ChosenOnValidation));
                    }
                }
            }
        }
    }
    map_in_parallel(&planned, threads, |(rule, choice)| {
        let mut rule = *rule;
        let mut validation_grid = Vec::new();
        if *choice == BarChoice::ChosenOnValidation {
            for bar in &bars {
                let tried = GateRule { bar: *bar, ..rule };
                let outcome = validation.outcome(&tried, truth);
                validation_grid.push(GridPoint {
                    bar: *bar,
                    promoted: outcome.promoted,
                    precision: outcome.precision,
                    share_validated: outcome.share_validated,
                    share_validated_in_two_calls: outcome.share_validated_in_two_calls,
                });
            }
            rule.bar = choose_bar(&validation_grid, args.min_share)
                .ok_or_else(|| EvalError::invalid("gate: no bar to choose from"))?;
        }
        Ok(Variant {
            name: variant_name(&rule, *choice),
            rule,
            bar_choice: *choice,
            validation_grid,
            validation: validation.outcome(&rule, truth),
            test: test.outcome(&rule, truth),
        })
    })
}

/// Draw the judge's sample (uniform, without replacement, seeded) and, when
/// the judge's verdicts are given, measure agreement with the key-term rule.
fn judge_sample_and_agreement(
    args: &GateArgs,
    extractions: &[Extraction],
    verdicts: &[ExtractionVerdict],
) -> Result<(JudgeSummary, Vec<JudgeItem>), EvalError> {
    let mut pool: Vec<usize> = (0..extractions.len()).collect();
    pool.sort_by(|a, b| extractions[*a].transcript.cmp(&extractions[*b].transcript));
    let take = args.judge_sample.min(pool.len());
    let mut rng = stream(args.seed, "gate/judge-sample");
    for i in 0..take {
        let j = i + uniform_below(&mut rng, (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    let mut sample: Vec<usize> = pool[..take].to_vec();
    sample.sort_by(|a, b| extractions[*a].transcript.cmp(&extractions[*b].transcript));
    let items: Vec<JudgeItem> = sample
        .iter()
        .map(|i| JudgeItem {
            transcript: extractions[*i].transcript.clone(),
            problem: extractions[*i].problem.clone(),
            solution: extractions[*i].solution.clone(),
            calls: Vec::new(),
        })
        .collect();
    let mut summary = JudgeSummary {
        sample: take,
        agreement: None,
        verdicts_outside_sample: 0,
    };
    if let Some(path) = &args.judge {
        let judged: Vec<JudgeVerdict> = read_jsonl_file(path)?;
        let mut by_transcript: HashMap<&str, bool> = HashMap::with_capacity(judged.len());
        for verdict in &judged {
            if by_transcript
                .insert(verdict.transcript.as_str(), verdict.correct)
                .is_some()
            {
                return Err(EvalError::invalid(format!(
                    "{}: transcript {:?} is judged twice",
                    path.display(),
                    verdict.transcript
                )));
            }
        }
        let rule_of: HashMap<&str, bool> = verdicts
            .iter()
            .map(|v| (v.transcript.as_str(), v.correct))
            .collect();
        let in_sample: BTreeSet<&str> = items.iter().map(|i| i.transcript.as_str()).collect();
        let pairs: Vec<(bool, bool)> = items
            .iter()
            .filter_map(|i| {
                let judge = by_transcript.get(i.transcript.as_str())?;
                let rule = rule_of.get(i.transcript.as_str())?;
                Some((*rule, *judge))
            })
            .collect();
        summary.verdicts_outside_sample = by_transcript.keys().filter(|t| !in_sample.contains(*t)).count();
        summary.agreement = Some(agreement_of(&pairs));
    }
    Ok((summary, items))
}

/// The judge's sample of the headline variant's promoted groups on the test split, and
/// what the judge said about it when its verdicts are given. A group's sample id is
/// `group-<n>`, its place among the promoted groups.
fn promoted_judge_sample(
    args: &GateArgs,
    test: &SplitData,
    variants: &[Variant],
    truth: &Truth,
) -> Result<Option<(PromotedJudgeSummary, Vec<JudgeItem>)>, EvalError> {
    let Some(variant) = variants.iter().find(|v| v.name == HEADLINE_VARIANT) else {
        return Ok(None);
    };
    let groups = replay(&variant.rule, &test.items, &test.order, &test.likeness);
    // The key-term verdict of each group, exactly as `score` counts it.
    let verdicts: Vec<(Stated, bool)> = groups
        .iter()
        .map(|group| judge_group(&group.members, &test.items, truth))
        .collect();
    let rule_correct: Vec<bool> = verdicts
        .iter()
        .map(|v| matches!(v, (Stated::True(_), false)))
        .collect();
    let mixes: Vec<bool> = verdicts.iter().map(|(_, mixes)| *mixes).collect();
    let mut pool: Vec<usize> = (0..groups.len()).collect();
    let take = args.judge_sample.min(pool.len());
    let mut rng = stream(args.seed, "gate/judge-sample-promoted");
    for i in 0..take {
        let j = i + uniform_below(&mut rng, (pool.len() - i) as u64) as usize;
        pool.swap(i, j);
    }
    let mut sample: Vec<usize> = pool[..take].to_vec();
    sample.sort_unstable();
    let items: Vec<JudgeItem> = sample
        .iter()
        .map(|g| {
            let (problem, solution) = merged_parts(&groups[*g].members, &test.items);
            JudgeItem {
                transcript: format!("group-{g:05}"),
                problem: problem.to_string(),
                solution: solution.to_string(),
                calls: groups[*g]
                    .members
                    .iter()
                    .map(|m| test.items[*m].transcript.clone())
                    .collect(),
            }
        })
        .collect();
    let mut summary = PromotedJudgeSummary {
        variant: variant.name.clone(),
        groups: groups.len(),
        sample: take,
        judged: None,
        judge_correct: None,
        judge_correct_not_mixed: None,
        agreement: None,
    };
    if let Some(path) = &args.judge_promoted {
        let judged: Vec<JudgeVerdict> = read_jsonl_file(path)?;
        let by_id: HashMap<&str, bool> = judged
            .iter()
            .map(|v| (v.transcript.as_str(), v.correct))
            .collect();
        let mut pairs = Vec::new();
        let (mut correct, mut correct_not_mixed) = (0, 0);
        for g in &sample {
            if let Some(judge) = by_id.get(format!("group-{g:05}").as_str()) {
                pairs.push((rule_correct[*g], *judge));
                correct += usize::from(*judge);
                correct_not_mixed += usize::from(*judge && !mixes[*g]);
            }
        }
        summary.judged = Some(pairs.len());
        summary.judge_correct = Some(correct);
        summary.judge_correct_not_mixed = Some(correct_not_mixed);
        summary.agreement = Some(agreement_of(&pairs));
    }
    Ok(Some((summary, items)))
}

/// [`ErrorModel`] on the test split; `None` while the headline or the single variant was
/// not replayed (fewer conversations asked for, or no validation split to choose a bar on).
fn error_model(test: &SplitData, variants: &[Variant]) -> Option<ErrorModel> {
    let headline = variants.iter().find(|v| v.name == HEADLINE_VARIANT)?;
    let single = variants.iter().find(|v| v.name == SINGLE_VARIANT)?;
    let pairs = pair_counts(&test.items, &test.likeness, headline.rule.bar);
    let rate = |matched: usize, all: usize| (all > 0).then(|| matched as f64 / all as f64);
    let m_c = rate(pairs.same_state_matched, pairs.same_state);
    let m_w = rate(pairs.wrong_wrong_matched, pairs.wrong_wrong);
    let m_x = rate(pairs.correct_wrong_matched, pairs.correct_wrong);
    let p = single.test.precision;
    let predicted = match (p, m_c, m_w, m_x) {
        (Some(p), Some(m_c), Some(m_w), Some(m_x)) => predicted_precision_two(p, m_c, m_w, m_x),
        _ => None,
    };
    Some(ErrorModel {
        variant: headline.name.clone(),
        bar: headline.rule.bar,
        pairs,
        precision_single: p,
        match_rate_same_state: m_c,
        match_rate_wrong_wrong: m_w,
        match_rate_correct_wrong: m_x,
        precision_two_predicted: predicted,
        precision_two: headline.test.precision,
    })
}

/// [`PrecisionGains`] on the test split; `None` while the single variant was not replayed.
fn precision_gains(
    args: &GateArgs,
    test: &SplitData,
    variants: &[Variant],
    truth: &Truth,
    threads: usize,
) -> Result<Option<PrecisionGains>, EvalError> {
    let Some(single) = variants.iter().find(|v| v.name == SINGLE_VARIANT) else {
        return Ok(None);
    };
    // Promoted and correct groups per incident, under the incident of a group's first member.
    let per_incident = |rule: &GateRule| -> BTreeMap<&str, (u32, u32)> {
        let mut out: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
        for group in replay(rule, &test.items, &test.order, &test.likeness) {
            let tally = out
                .entry(test.items[group.members[0]].thread.as_str())
                .or_default();
            tally.0 += 1;
            tally.1 += u32::from(group_is_correct(&group.members, &test.items, truth));
        }
        out
    };
    // The single variant promotes every extraction, so its incidents are all incidents
    // with one: every variant's tallies line up with them and share the resamples.
    let one = per_incident(&single.rule);
    let others: Vec<&Variant> = variants.iter().filter(|v| v.name != SINGLE_VARIANT).collect();
    let gains = map_in_parallel(&others, threads, |variant| {
        let two = per_incident(&variant.rule);
        if let Some(thread) = two.keys().find(|t| !one.contains_key(*t)) {
            return Err(EvalError::invalid(format!(
                "gate: variant {} promoted a group in incident {thread:?}, where {SINGLE_VARIANT} promoted none",
                variant.name
            )));
        }
        let tallies: Vec<IncidentTally> = one
            .iter()
            .map(|(thread, (one_promoted, one_correct))| {
                let (two_promoted, two_correct) = two.get(thread).copied().unwrap_or_default();
                IncidentTally {
                    one_promoted: *one_promoted,
                    one_correct: *one_correct,
                    two_promoted,
                    two_correct,
                }
            })
            .collect();
        Ok(PrecisionGain {
            variant: variant.name.clone(),
            gain: variant
                .test
                .precision
                .zip(single.test.precision)
                .map(|(a, b)| a - b),
            interval: precision_gain_interval(&tallies, args.seed, DEFAULT_DRAWS),
        })
    })?;
    Ok(Some(PrecisionGains {
        versus: SINGLE_VARIANT.to_string(),
        incidents: one.len(),
        draws: DEFAULT_DRAWS,
        variants: gains,
    }))
}

/// The name `summary.json` uses for a bar choice.
fn bar_choice_name(choice: BarChoice) -> &'static str {
    match choice {
        BarChoice::NotUsed => "not_used",
        BarChoice::Paper => "paper",
        BarChoice::ChosenOnValidation => "chosen_on_validation",
    }
}

fn variants_csv(summary: &GateSummary) -> String {
    let mut text = csv_line(&[
        "name",
        "split",
        "variant",
        "conversations",
        "agreement",
        "independence",
        "confidence_cut",
        "bar",
        "bar_choice",
        "considered",
        "promoted",
        "correct",
        "precision",
        "false_merges",
        "states",
        "states_validated",
        "share_validated",
        "states_in_two_calls",
        "states_in_two_calls_validated",
        "share_validated_in_two_calls",
        "wait_median_s",
        "wait_p90_s",
    ]);
    let optional = |value: Option<f64>| value.map_or_else(String::new, csv_number);
    for variant in &summary.variants {
        for (split, outcome) in [("validation", &variant.validation), ("test", &variant.test)] {
            let rule = &variant.rule;
            text.push_str(&csv_line(&[
                summary.name.clone(),
                split.to_string(),
                variant.name.clone(),
                rule.conversations.to_string(),
                match rule.agreement {
                    Agreement::Pairs => "pairs",
                    Agreement::Average => "average",
                }
                .to_string(),
                match rule.independence {
                    Independence::Calls => "calls",
                    Independence::Agents => "agents",
                }
                .to_string(),
                optional(rule.confidence_cut),
                if rule.conversations == 1 {
                    String::new()
                } else {
                    csv_number(rule.bar)
                },
                bar_choice_name(variant.bar_choice).to_string(),
                outcome.considered.to_string(),
                outcome.promoted.to_string(),
                outcome.correct.to_string(),
                optional(outcome.precision),
                outcome.false_merges.to_string(),
                outcome.states.to_string(),
                outcome.states_validated.to_string(),
                optional(outcome.share_validated),
                outcome.states_in_two_calls.to_string(),
                outcome.states_in_two_calls_validated.to_string(),
                optional(outcome.share_validated_in_two_calls),
                optional(outcome.wait_median_s),
                optional(outcome.wait_p90_s),
            ]));
        }
    }
    text
}

/// A share in percent, a wait in minutes, or `--` when there is none.
fn percent_or_dash(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), |v| format_value(v, Unit::Percent))
}

fn minutes_or_dash(seconds: Option<f64>) -> String {
    seconds.map_or_else(|| "--".to_string(), |s| format!("{:.1}", s / 60.0))
}

fn test_table(summary: &GateSummary) -> Table {
    let test = summary.splits.iter().find(|s| s.split == Split::Test);
    Table {
        comments: vec![
            format!(
                "Written by {} (cb-eval gate). Do not edit; run the gate again.",
                summary.made_by
            ),
            format!(
                "{}: the two-conversation rule on the test split ({} extractions, {} true states).",
                summary.name,
                test.map_or(0, |s| s.extractions),
                test.map_or(0, |s| s.states)
            ),
            "Precision of promoted records; share of true states ever validated, of all and of those stated \
             in two or more calls; wait from a state's first statement to its validation, in minutes."
                .to_string(),
        ],
        preamble: Vec::new(),
        columns: "lrrrrrrrr".to_string(),
        header: [
            "Variant",
            "Bar",
            "Promoted",
            "Precision (\\%)",
            "Validated (\\%)",
            "Validated, 2+ calls (\\%)",
            "Wait median (min)",
            "Wait P90 (min)",
            "False merges",
        ]
        .map(String::from)
        .to_vec(),
        rows: summary
            .variants
            .iter()
            .map(|v| {
                vec![
                    format!("\\texttt{{{}}}", escape(&v.name)),
                    if v.rule.conversations == 1 {
                        "--".to_string()
                    } else {
                        format!("{}", v.rule.bar)
                    },
                    v.test.promoted.to_string(),
                    percent_or_dash(v.test.precision),
                    percent_or_dash(v.test.share_validated),
                    percent_or_dash(v.test.share_validated_in_two_calls),
                    minutes_or_dash(v.test.wait_median_s),
                    minutes_or_dash(v.test.wait_p90_s),
                    v.test.false_merges.to_string(),
                ]
            })
            .collect(),
    }
}

/// The test split's numbers. Keys: `<name>/gate/<variant>/precision` (and
/// `/promoted`, `/correct`, `/false-merges`, `/validated`, `/states-validated`,
/// `/wait-median-min`, `/wait-p90-min`, `/bar`); `<name>/gate/test/extractions`
/// (and `/states`, `/correct-extractions`); `<name>/gate/judge/sample` (and,
/// with verdicts, `/judged`, `/agreement`, `/kappa`).
pub fn numbers_of(summary: &GateSummary) -> Result<Numbers, EvalError> {
    let name = &summary.name;
    let mut numbers = Numbers::new(vec![
        format!(
            "Written by {} (cb-eval gate). Do not edit; run the gate again.",
            summary.made_by
        ),
        format!(
            "{name}: the two-conversation rule, test split; seed {}.",
            summary.seed
        ),
        "Shares in percent with one decimal; waits in minutes with one decimal; -- where there is none."
            .to_string(),
    ]);
    if let Some(test) = summary.splits.iter().find(|s| s.split == Split::Test) {
        numbers.push(
            format!("{name}/gate/test/extractions"),
            test.extractions.to_string(),
        )?;
        numbers.push(
            format!("{name}/gate/test/correct-extractions"),
            test.correct_extractions.to_string(),
        )?;
        numbers.push(format!("{name}/gate/test/states"), test.states.to_string())?;
        numbers.push(
            format!("{name}/gate/test/states-in-two-calls"),
            test.states_in_two_calls.to_string(),
        )?;
    }
    numbers.push(
        format!("{name}/gate/judge/sample"),
        summary.judge.sample.to_string(),
    )?;
    if let Some(agreement) = &summary.judge.agreement {
        numbers.push(format!("{name}/gate/judge/judged"), agreement.judged.to_string())?;
        numbers.push(
            format!("{name}/gate/judge/agreement"),
            percent_or_dash(agreement.agreement),
        )?;
        numbers.push(
            format!("{name}/gate/judge/kappa"),
            agreement
                .kappa
                .map_or_else(|| "--".to_string(), |k| format_value(k, Unit::Fraction)),
        )?;
    }
    for variant in &summary.variants {
        let key = format!("{name}/gate/{}", variant.name);
        let t = &variant.test;
        numbers.push(format!("{key}/precision"), percent_or_dash(t.precision))?;
        numbers.push(format!("{key}/promoted"), t.promoted.to_string())?;
        numbers.push(format!("{key}/correct"), t.correct.to_string())?;
        numbers.push(format!("{key}/false-merges"), t.false_merges.to_string())?;
        numbers.push(format!("{key}/validated"), percent_or_dash(t.share_validated))?;
        numbers.push(
            format!("{key}/validated-two-calls"),
            percent_or_dash(t.share_validated_in_two_calls),
        )?;
        numbers.push(format!("{key}/states-validated"), t.states_validated.to_string())?;
        numbers.push(format!("{key}/wait-median-min"), minutes_or_dash(t.wait_median_s))?;
        numbers.push(format!("{key}/wait-p90-min"), minutes_or_dash(t.wait_p90_s))?;
        if variant.rule.conversations > 1 {
            numbers.push(format!("{key}/bar"), format!("{}", variant.rule.bar))?;
        }
    }
    Ok(numbers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{read_json_file, test_dir, write_vectors};
    use crate::vectors::VectorIndex;
    use cb_types::{Confidence, Phase, RecordTruth, Speaker, Tier, Turn, write_jsonl};
    use std::fs;

    /// A bar's validation result; `share` is over the states stated in two or more calls,
    /// what the bar is chosen by. The share of all states is set apart so a test that
    /// chose by it would fail.
    fn point(bar: f64, precision: f64, share: f64) -> GridPoint {
        GridPoint {
            bar,
            promoted: 1,
            precision: Some(precision),
            share_validated: Some(share / 2.0),
            share_validated_in_two_calls: Some(share),
        }
    }

    #[test]
    fn the_bar_is_the_most_precise_of_those_that_validate_enough() {
        let grid = [
            point(0.85, 0.70, 0.90),
            point(0.88, 0.80, 0.75),
            point(0.92, 0.95, 0.50),
            point(0.95, 0.80, 0.72),
        ];
        // 0.92 is the most precise but validates only half; of the rest 0.88 and 0.95 tie on
        // precision, and 0.88 validates more.
        assert_eq!(choose_bar(&grid, 0.70), Some(0.88));
        // With no share asked for, the most precise.
        assert_eq!(choose_bar(&grid, 0.0), Some(0.92));
        // When none validates enough, the one that validates most.
        assert_eq!(choose_bar(&grid, 0.95), Some(0.85));
        // A full tie goes to the lower bar.
        assert_eq!(
            choose_bar(&[point(0.9, 0.8, 0.8), point(0.8, 0.8, 0.8)], 0.7),
            Some(0.8)
        );
        assert_eq!(choose_bar(&[], 0.7), None);
    }

    #[test]
    fn thinning_keeps_a_seeded_share_and_a_smaller_share_keeps_a_subset() {
        let ids: Vec<String> = (0..2_000).map(|i| format!("call-{i:05}")).collect();
        let kept =
            |keep: f64| -> Vec<&String> { ids.iter().filter(|id| call_is_kept(7, keep, id)).collect() };
        let (half, quarter) = (kept(0.5), kept(0.25));
        // About the share asked for.
        assert!((900..=1_100).contains(&half.len()), "{}", half.len());
        assert!((400..=600).contains(&quarter.len()), "{}", quarter.len());
        // Nested, and the same on every call.
        assert!(quarter.iter().all(|id| half.contains(id)));
        assert_eq!(kept(0.5), half);
        assert_eq!(kept(1.0).len(), ids.len());
    }

    #[test]
    fn variant_names_say_what_was_run() {
        let rule = GateRule {
            conversations: 2,
            agreement: Agreement::Pairs,
            independence: Independence::Calls,
            confidence_cut: Some(PAPER_CONFIDENCE_CUT),
            bar: PAPER_BAR,
        };
        assert_eq!(
            variant_name(&rule, BarChoice::Paper),
            "n2-pairs-calls-cut0.85-bar0.92"
        );
        let other = GateRule {
            conversations: 3,
            agreement: Agreement::Average,
            independence: Independence::Agents,
            confidence_cut: None,
            bar: 0.87,
        };
        assert_eq!(
            variant_name(&other, BarChoice::ChosenOnValidation),
            "n3-average-agents-nocut-chosen"
        );
        let single = GateRule {
            conversations: 1,
            confidence_cut: None,
            ..rule
        };
        assert_eq!(variant_name(&single, BarChoice::NotUsed), "n1-nocut");
    }

    /// One incident's records: two true states and one planted wrong statement.
    fn incident(thread: &str, start: i64) -> Vec<Record> {
        let make = |state: &str, at: i64, correct: bool, terms: &[&str], superseded: Option<i64>| Record {
            id: format!("{thread}.{state}"),
            text: String::new(),
            written_at: at,
            valid_until: None,
            thread: Some(thread.to_string()),
            tier: Tier::Changing,
            title: None,
            source: "test".to_string(),
            truth: Some(RecordTruth {
                correct,
                phase: Phase::Investigating,
                key_terms: terms.iter().map(|t| t.to_string()).collect(),
                superseded_at: superseded,
            }),
        };
        vec![
            make("s1", start, true, &["fibre cut"], Some(start + 3_600)),
            make("s2", start + 3_600, true, &["fibre cut", "back by noon"], None),
            make("d1", start + 600, false, &["your own cable"], None),
        ]
    }

    /// A call by `agent` that states `states`, `minutes` after `start`, with the agent saying `said`.
    fn call(id: &str, thread: &str, agent: &str, start: i64, states: &[&str], said: &str) -> Transcript {
        Transcript {
            id: id.to_string(),
            thread: thread.to_string(),
            started_at: start,
            agent: agent.to_string(),
            persona: String::new(),
            turns: vec![
                Turn {
                    who: Speaker::Customer,
                    text: "Is it down?".to_string(),
                    ends_at_s: 5.0,
                },
                Turn {
                    who: Speaker::Agent,
                    text: said.to_string(),
                    ends_at_s: 30.0,
                },
                Turn {
                    who: Speaker::Customer,
                    text: "Thanks.".to_string(),
                    ends_at_s: 60.0,
                },
            ],
            states: states.iter().map(|s| format!("{thread}.{s}")).collect(),
            asr_error_rate: 0.0,
        }
    }

    fn extraction(transcript: &str, solution: &str, confidence: f32) -> Extraction {
        Extraction {
            transcript: transcript.to_string(),
            problem: "internet down".to_string(),
            solution: solution.to_string(),
            t_start: None,
            t_eta: None,
            times_unread: vec![],
            confidence: Confidence {
                problem: confidence,
                solution: 1.0,
                time: 1.0,
            },
            model: "test".to_string(),
            prompt_version: "1".to_string(),
            latency_ms: 2_000.0,
            tokens_in: 0,
            tokens_out: 0,
        }
    }

    /// inc-0006 (validation) and inc-0001 (test), four calls each, ten minutes apart:
    /// three state s1 correctly, one states the wrong statement d1.
    fn write_scene(dir: &Path) -> GateArgs {
        let (data, run) = (dir.join("data"), dir.join("run"));
        fs::create_dir_all(&data).unwrap();
        fs::create_dir_all(&run).unwrap();
        let mut records = incident("inc-0006", 10_000);
        records.extend(incident("inc-0001", 50_000));
        let mut transcripts = Vec::new();
        let mut extractions = Vec::new();
        let mut ids = Vec::new();
        let mut values: Vec<f32> = Vec::new();
        for (thread, start) in [("inc-0006", 10_000), ("inc-0001", 50_000)] {
            for (i, agent) in ["a", "b", "a", "c"].iter().enumerate() {
                let id = format!("{thread}-call{i}");
                let begins = start + 600 * i as i64;
                let (states, said, solution, vector) = if i == 3 {
                    (
                        vec!["d1"],
                        "It is your own cable.",
                        "check your own cable",
                        [0.0f32, 1.0],
                    )
                } else {
                    (
                        vec!["s1"],
                        "There is a fibre cut.",
                        "a fibre cut in the area",
                        [1.0, 0.02 * i as f32],
                    )
                };
                transcripts.push(call(&id, thread, agent, begins, &states, said));
                extractions.push(extraction(&id, solution, if i == 2 { 0.5 } else { 0.95 }));
                ids.push(id);
                values.extend(vector);
            }
        }
        let write = |path: PathBuf, f: &dyn Fn(&mut Vec<u8>)| {
            let mut bytes = Vec::new();
            f(&mut bytes);
            fs::write(path, bytes).unwrap();
        };
        write(data.join(RECORDS), &|b| write_jsonl(b, &records).unwrap());
        write(data.join(TRANSCRIPTS), &|b| write_jsonl(b, &transcripts).unwrap());
        write(run.join(EXTRACTIONS), &|b| write_jsonl(b, &extractions).unwrap());
        write_vectors(
            &run.join(EXTRACTION_VECTORS),
            &VectorIndex { dim: 2, ids },
            &values,
        )
        .unwrap();
        GateArgs {
            data,
            run: run.clone(),
            transcripts: None,
            out: run,
            name: "tiny".to_string(),
            conversations: vec![1, 2, 3],
            bars: vec![0.9, 0.95],
            min_share: DEFAULT_MIN_SHARE,
            seed: 7,
            judge_sample: 3,
            judge: None,
            judge_promoted: None,
            threads: 2,
            keep_calls: 1.0,
        }
    }

    fn variant<'a>(summary: &'a GateSummary, name: &str) -> &'a Variant {
        summary.variants.iter().find(|v| v.name == name).unwrap()
    }

    #[test]
    fn the_gate_reports_every_variant_from_the_files() {
        let dir = test_dir("gate-files");
        let args = write_scene(&dir);
        let gate = run(&args).unwrap();
        let summary = &gate.summary;
        // n = 1: 2 variants; n = 2 and 3: 2 agreements x 2 independences x 2 cuts x 2 bars each.
        assert_eq!(summary.variants.len(), 2 + 16 + 16);
        let test = summary.splits.iter().find(|s| s.split == Split::Test).unwrap();
        assert_eq!(
            (test.extractions, test.correct_extractions, test.states),
            (4, 3, 1)
        );
        assert_eq!(
            (test.calls_without_extraction, test.statement_moments_at_call_end),
            (0, 0)
        );

        // Every extraction on its own: 3 of 4 correct. s1 is first stated at the end of the agent's
        // turn of call 0 (50 000 + 30) and promoted when that extraction arrives (50 000 + 60 + 2):
        // a wait of 32 s.
        let single = variant(summary, "n1-nocut");
        assert_eq!((single.test.promoted, single.test.correct), (4, 3));
        assert_eq!(single.test.precision, Some(0.75));
        assert_eq!(single.test.wait_median_s, Some(32.0));
        // The cut drops call 2 (confidence 0.5): 3 promoted, 2 correct.
        let cut = variant(summary, "n1-cut0.85");
        assert_eq!((cut.test.promoted, cut.test.correct), (3, 2));

        // The paper's setting: call 2 is cut; call 1 agrees with call 0 (cos 0.9998) and promotes
        // both at 50 600 + 62; s1 waits 50 662 - 50 030 = 632 s. Call 3 (the wrong statement) waits
        // alone.
        let paper = variant(summary, "n2-pairs-calls-cut0.85-bar0.92");
        assert_eq!(
            (paper.test.promoted, paper.test.correct, paper.test.false_merges),
            (1, 1, 0)
        );
        assert_eq!(paper.test.precision, Some(1.0));
        assert_eq!(paper.test.share_validated, Some(1.0));
        assert_eq!(paper.test.wait_median_s, Some(632.0));
        // Different agents: calls 0 and 2 are by agent a; without the cut, call 1 (b) pairs with 0.
        let agents = variant(summary, "n2-pairs-agents-nocut-bar0.92");
        assert_eq!(agents.test.promoted, 1);
        // Three conversations without the cut: 0, 1 and 2 agree pair by pair; promoted at call 2.
        let three = variant(summary, "n3-pairs-calls-nocut-bar0.92");
        assert_eq!((three.test.promoted, three.test.correct), (1, 1));
        assert_eq!(three.test.wait_median_s, Some(1_232.0));
        // A chosen bar keeps its validation grid.
        let chosen = variant(summary, "n2-pairs-calls-cut0.85-chosen");
        assert_eq!(chosen.bar_choice, BarChoice::ChosenOnValidation);
        assert_eq!(chosen.validation_grid.len(), 2);

        for file in [
            "summary.json",
            "variants.csv",
            "extractions.jsonl",
            "judge_sample.jsonl",
            "table.tex",
            "numbers.tex",
        ] {
            assert!(dir.join("run/gate").join(file).exists(), "{file}");
        }
        let back: GateSummary = read_json_file(&dir.join("run/gate/summary.json")).unwrap();
        assert_eq!(&back, summary);
        let numbers = fs::read_to_string(dir.join("run/gate/numbers.tex")).unwrap();
        assert!(
            numbers.contains(
                "\\csname cbnum/tiny/gate/n2-pairs-calls-cut0.85-bar0.92/precision\\endcsname{100.0}"
            )
        );
        assert!(numbers.contains("\\csname cbnum/tiny/gate/n1-nocut/precision\\endcsname{75.0}"));
        // 632 s is 10.5 minutes.
        assert!(numbers.contains(
            "\\csname cbnum/tiny/gate/n2-pairs-calls-cut0.85-bar0.92/wait-median-min\\endcsname{10.5}"
        ));
        let verdicts: Vec<ExtractionVerdict> =
            read_jsonl_file(&dir.join("run/gate/extractions.jsonl")).unwrap();
        assert_eq!(verdicts.len(), 8);
        let wrong = verdicts
            .iter()
            .find(|v| v.transcript == "inc-0001-call3")
            .unwrap();
        assert_eq!(
            (wrong.stated.as_str(), wrong.record.as_deref(), wrong.correct),
            ("wrong", Some("inc-0001.d1"), false)
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_extraction_that_found_nothing_is_counted_and_never_grouped() {
        let dir = test_dir("gate-nothing");
        let args = write_scene(&dir);
        // A fifth call of inc-0001 whose extraction found nothing: no vector is written for it.
        let mut transcripts: Vec<Transcript> = read_jsonl_file(&args.data.join(TRANSCRIPTS)).unwrap();
        transcripts.push(call(
            "inc-0001-call4",
            "inc-0001",
            "d",
            52_400,
            &["s1"],
            "There is a fibre cut.",
        ));
        let mut lines = Vec::new();
        write_jsonl(&mut lines, &transcripts).unwrap();
        fs::write(args.data.join(TRANSCRIPTS), lines).unwrap();
        let mut extractions: Vec<Extraction> = read_jsonl_file(&args.run.join(EXTRACTIONS)).unwrap();
        let mut nothing = extraction("inc-0001-call4", "", 0.0);
        nothing.problem = String::new();
        extractions.push(nothing);
        let mut lines = Vec::new();
        write_jsonl(&mut lines, &extractions).unwrap();
        fs::write(args.run.join(EXTRACTIONS), lines).unwrap();
        let gate = run(&args).unwrap();
        assert_eq!(gate.summary.extractions_that_found_nothing, 1);
        let test = gate
            .summary
            .splits
            .iter()
            .find(|s| s.split == Split::Test)
            .unwrap();
        // The same four extractions as before; the fifth call has none that counts.
        assert_eq!((test.extractions, test.correct_extractions), (4, 3));
        assert_eq!(test.calls_without_extraction, 1);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_judge_sample_is_seeded_and_agreement_is_measured_on_it() {
        let dir = test_dir("gate-judge");
        let mut args = write_scene(&dir);
        let first: Vec<JudgeItem> = {
            run(&args).unwrap();
            read_jsonl_file(&dir.join("run/gate/judge_sample.jsonl")).unwrap()
        };
        assert_eq!(first.len(), 3);
        run(&args).unwrap();
        let again: Vec<JudgeItem> = read_jsonl_file(&dir.join("run/gate/judge_sample.jsonl")).unwrap();
        assert_eq!(first, again);
        // A judge that calls everything correct, including one extraction outside the sample.
        let mut verdicts: Vec<JudgeVerdict> = first
            .iter()
            .map(|item| JudgeVerdict {
                transcript: item.transcript.clone(),
                correct: true,
            })
            .collect();
        let outside = [
            "inc-0006-call0",
            "inc-0006-call1",
            "inc-0006-call2",
            "inc-0006-call3",
            "inc-0001-call0",
            "inc-0001-call1",
            "inc-0001-call2",
            "inc-0001-call3",
        ]
        .into_iter()
        .find(|t| first.iter().all(|i| i.transcript != *t))
        .unwrap();
        verdicts.push(JudgeVerdict {
            transcript: outside.to_string(),
            correct: true,
        });
        let path = dir.join("judge.jsonl");
        let mut bytes = Vec::new();
        write_jsonl(&mut bytes, &verdicts).unwrap();
        fs::write(&path, bytes).unwrap();
        args.judge = Some(path);
        let judged = run(&args).unwrap();
        let agreement = judged.summary.judge.agreement.clone().unwrap();
        assert_eq!(agreement.judged, 3);
        assert_eq!(judged.summary.judge.verdicts_outside_sample, 1);
        // The rule calls the sampled wrong statements wrong; the judge does not.
        let rule_correct = agreement.both_correct + agreement.only_rule_correct;
        assert_eq!(agreement.only_judge_correct, 3 - rule_correct);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn inputs_that_break_the_contract_are_refused() {
        let dir = test_dir("gate-refuse");
        let args = write_scene(&dir);
        let mut bad = args.clone();
        bad.conversations = vec![0];
        assert!(run(&bad).is_err());
        let mut bad = args.clone();
        bad.name = "a/b".to_string();
        assert!(run(&bad).is_err());
        // An extraction of a call the transcripts file does not hold.
        let mut extra: Vec<Extraction> = read_jsonl_file(&args.run.join(EXTRACTIONS)).unwrap();
        extra.push(extraction("nowhere", "x", 1.0));
        let mut bytes = Vec::new();
        write_jsonl(&mut bytes, &extra).unwrap();
        fs::write(args.run.join(EXTRACTIONS), &bytes).unwrap();
        assert!(
            run(&args)
                .unwrap_err()
                .to_string()
                .contains("\"nowhere\" is not in")
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
