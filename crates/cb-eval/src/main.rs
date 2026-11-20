//! `cb-eval` — computes every reported number of the ContextBridge proof from run files.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use anyhow::Context;
use cb_eval::aggregate::Group;
use cb_eval::commands::report::{self, DEFAULT_COLUMNS, Summary};
use cb_eval::commands::{gate, misses, rank, speech, trips, tune};
use cb_eval::files::TUNED;
use cb_eval::latex::{format_difference, format_p, format_value};
use cb_eval::measures::Measure;
use cb_eval::scores::label_of;
use cb_eval::spec::{MethodSpec, parse_methods, parse_methods_file};
use cb_eval::stats::{DEFAULT_DRAWS, DEFAULT_SEED};
use cb_types::Split;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "cb-eval",
    version,
    about = "Computes every reported number of the ContextBridge proof from run files"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run ranking methods over every question of a run and score each answer.
    Rank(RankCli),
    /// Choose a method's parameters on the validation split and record the choice in tuned.json.
    Tune(TuneCli),
    /// Write summary.json, figures/*.csv, tables/*.tex and numbers.tex for the test split.
    Report(ReportCli),
    /// Replay the two-conversation rule over a run's extractions, every variant, into gate/.
    Gate(GateCli),
    /// Percentiles over whole trips and medians of stage times from trips.jsonl, into trips/.
    Trips(TripsCli),
    /// Put together trips from the end of speech to the first hit, per call, from stage times.
    SpeechTrips(SpeechTripsCli),
    /// Where a product method with the newest list misses the current record, into misses/.
    Misses(MissesCli),
}

#[derive(Args)]
struct MissesCli {
    /// The data set directory (records.jsonl, queries.jsonl).
    #[arg(long, value_name = "DIR")]
    data: PathBuf,
    /// The run directory (candidates.jsonl, newest.jsonl).
    #[arg(long, value_name = "DIR")]
    run: PathBuf,
    /// The method, in its text form, e.g. 'product_bonus(beta=1,h=3600,recent=16)'.
    #[arg(value_name = "METHOD")]
    method: MethodSpec,
}

#[derive(Args)]
struct SpeechTripsCli {
    /// transcribe_many's output: {"id", "decode_ms", …} per voice line.
    #[arg(long, value_name = "FILE")]
    stt: PathBuf,
    /// The voice lines: {"id", "transcript", …} per line (refs.jsonl).
    #[arg(long, value_name = "FILE")]
    voice: PathBuf,
    /// mask_transcripts' timings: {"id", "mask_us", …} per call.
    #[arg(long, value_name = "FILE")]
    mask: PathBuf,
    /// cb-run extract's extractions, made one call at a time.
    #[arg(long, value_name = "FILE")]
    extract: PathBuf,
    /// cb-run trips' trips (written_ms, first_hit_ms from the hand-over).
    #[arg(long, value_name = "FILE")]
    upload: PathBuf,
    /// The records the upload run timed, in its order (its data set's records.jsonl):
    /// needed when its failures.jsonl lists records the search never found.
    #[arg(long, value_name = "FILE")]
    upload_records: Option<PathBuf>,
    /// Where trips.jsonl, failures.jsonl and composed.json are written.
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
}

#[derive(Args)]
struct TripsCli {
    /// The run directory (trips.jsonl).
    #[arg(long, value_name = "DIR")]
    run: PathBuf,
    /// Where trips/ is written; the run directory by default.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,
    /// The first part of every number key; the run directory's name by default.
    #[arg(long)]
    name: Option<String>,
}

#[derive(Args)]
struct GateCli {
    /// The data set directory (records.jsonl, and transcripts.jsonl unless --transcripts names another file).
    #[arg(long, value_name = "DIR")]
    data: PathBuf,
    /// The run directory (extractions.jsonl, extraction_vectors.f32 and extraction_vectors.json).
    #[arg(long, value_name = "DIR")]
    run: PathBuf,
    /// The transcripts the extractions were made from, e.g. a file with altered words.
    #[arg(long, value_name = "FILE")]
    transcripts: Option<PathBuf>,
    /// Where gate/ is written; the run directory by default.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,
    /// The first part of every number key; the run directory's name by default.
    #[arg(long)]
    name: Option<String>,
    /// How many conversations must agree, separated by commas; each is a variant.
    #[arg(long, value_delimiter = ',', default_values_t = [1usize, 2, 3])]
    conversations: Vec<usize>,
    /// The likeness bars tried on the validation split, separated by commas (0.80 to 0.98 by default).
    #[arg(long, value_delimiter = ',')]
    bars: Vec<f64>,
    /// The share of true states a chosen bar must validate on the validation split.
    #[arg(long, default_value_t = gate::DEFAULT_MIN_SHARE)]
    min_share: f64,
    /// The seed of the judge's sample; it is written into every output.
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,
    /// How many extractions to draw for the language-model judge.
    #[arg(long, default_value_t = gate::DEFAULT_JUDGE_SAMPLE)]
    judge_sample: usize,
    /// The judge's verdicts on the sample, one {"transcript": …, "correct": …} per line.
    #[arg(long, value_name = "FILE")]
    judge: Option<PathBuf>,
    /// The judge's verdicts on gate/judge_sample_promoted.jsonl (groups of the headline variant).
    #[arg(long, value_name = "FILE")]
    judge_promoted: Option<PathBuf>,
    /// Worker threads (0 = every core). The numbers do not depend on it.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// The share of calls kept (above 0, at most 1): a lower call rate. A seeded hash of
    /// each call's id decides, and a smaller share keeps a subset of a larger one's calls.
    #[arg(long, default_value_t = 1.0)]
    keep_calls: f64,
}

#[derive(Args)]
struct MethodList {
    /// A file with one method per line; blank lines and lines starting with # are skipped.
    /// Repeat for several; a method named in more than one place is kept once, at its first.
    #[arg(long, value_name = "FILE")]
    methods_file: Vec<PathBuf>,
    /// Methods in their text form, e.g. dense 'rrf_td(k=60,h=1800)' 'cc(alpha=0|0.5|1)'.
    /// A parameter may list alternatives with |; every combination is one method.
    #[arg(value_name = "METHOD")]
    methods: Vec<String>,
}

impl MethodList {
    fn read(&self) -> anyhow::Result<Vec<MethodSpec>> {
        let mut methods = Vec::new();
        let mut add = |found: Vec<MethodSpec>| {
            for spec in found {
                if !methods.contains(&spec) {
                    methods.push(spec);
                }
            }
        };
        for path in &self.methods_file {
            let text =
                std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
            add(parse_methods_file(&text).with_context(|| format!("in {}", path.display()))?);
        }
        add(parse_methods(&self.methods)?);
        Ok(methods)
    }
}

#[derive(Args)]
struct RankCli {
    /// The data set directory (records.jsonl, queries.jsonl).
    #[arg(long, value_name = "DIR")]
    data: PathBuf,
    /// The run directory (candidates.jsonl; vectors.f32 and vectors.json for product_newer_wins).
    #[arg(long, value_name = "DIR")]
    run: PathBuf,
    /// How many records of each answer to write to rankings/ (0 = all). Measures always see the whole answer.
    #[arg(long, default_value_t = rank::DEFAULT_KEEP)]
    keep: usize,
    #[command(flatten)]
    methods: MethodList,
}

#[derive(Args)]
struct TuneCli {
    /// A run directory that `rank` has scored; repeat for several runs (the seeds of a corpus).
    #[arg(long = "run", value_name = "DIR", required = true)]
    runs: Vec<PathBuf>,
    /// Where tuned.json is kept; the run directory when there is one run.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,
    /// The name of this choice, e.g. product_decay; a later choice with the same label replaces it.
    #[arg(long)]
    label: String,
    /// The measure the choice is made by, e.g. current@1.
    #[arg(long)]
    measure: Measure,
    /// The group of validation questions it is measured on, e.g. changing.
    #[arg(long)]
    group: Group,
    /// Prefer the smallest value instead of the measure's own direction.
    #[arg(long, conflicts_with = "higher_is_better")]
    lower_is_better: bool,
    /// Prefer the largest value instead of the measure's own direction.
    #[arg(long)]
    higher_is_better: bool,
    /// A second measure that may not fall more than --guard-margin below the baseline's, e.g. hit@4.
    #[arg(long, requires_all = ["guard_group", "guard_baseline"])]
    guard_measure: Option<Measure>,
    /// The group the guard is measured on, e.g. ordinary_all.
    #[arg(long, requires = "guard_measure")]
    guard_group: Option<Group>,
    /// The method the guard compares with, e.g. product.
    #[arg(long, requires = "guard_measure")]
    guard_baseline: Option<MethodSpec>,
    /// How far below the baseline the guard measure may fall, as a fraction (0.01 = 1 point).
    #[arg(long, default_value_t = 0.01, requires = "guard_measure")]
    guard_margin: f64,
    /// The candidates.
    #[command(flatten)]
    methods: MethodList,
}

#[derive(Args)]
struct ReportCli {
    /// A run directory that `rank` has scored; repeat for several runs (the seeds of a corpus).
    #[arg(long = "run", value_name = "DIR", required = true)]
    runs: Vec<PathBuf>,
    /// Where the outputs go; the run directory when there is one run.
    #[arg(long, value_name = "DIR")]
    out: Option<PathBuf>,
    /// The first part of every number key; the run directory's name when there is one run.
    #[arg(long)]
    name: Option<String>,
    /// A method every other is compared with; repeat for several.
    #[arg(long = "baseline", value_name = "METHOD", default_value = "product")]
    baselines: Vec<MethodSpec>,
    /// Compare with no baseline.
    #[arg(long, conflicts_with = "baselines")]
    no_baseline: bool,
    /// The parameter choices to apply; <out>/tuned.json when it exists.
    #[arg(long, value_name = "FILE")]
    tuned: Option<PathBuf>,
    /// Apply no parameter choices, even when <out>/tuned.json exists.
    #[arg(long, conflicts_with = "tuned")]
    no_tuned: bool,
    /// A method to show in the main tables besides the ones shown anyway; repeat for several.
    #[arg(long, value_name = "METHOD")]
    feature: Vec<MethodSpec>,
    /// The measures the tables show, separated by commas.
    #[arg(long, value_delimiter = ',')]
    columns: Vec<Measure>,
    /// The seed of every random draw; it is written into every output.
    #[arg(long, default_value_t = DEFAULT_SEED)]
    seed: u64,
    /// Bootstrap resamples and permutations.
    #[arg(long, default_value_t = DEFAULT_DRAWS)]
    draws: usize,
    /// Worker threads (0 = every core). The numbers do not depend on it.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// The methods to report; every method the runs hold scores for when none is named.
    #[command(flatten)]
    methods: MethodList,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Rank(cli) => run_rank(cli),
        Command::Tune(cli) => run_tune(cli),
        Command::Report(cli) => run_report(cli),
        Command::Gate(cli) => run_gate(cli),
        Command::Trips(cli) => run_trips(cli),
        Command::SpeechTrips(cli) => run_speech_trips(cli),
        Command::Misses(cli) => run_misses(cli),
    }
}

fn run_misses(cli: MissesCli) -> anyhow::Result<()> {
    let m = misses::run(&misses::MissesArgs {
        data: cli.data,
        run: cli.run.clone(),
        method: cli.method,
    })?;
    let share = |n: usize| 100.0 * n as f64 / m.questions.max(1) as f64;
    println!(
        "{}: {} changing questions (test split): first {:.1}%, not a candidate {:.1}%, under the cut {:.1}%, \
         outranked {:.1}%; current record among the newest read {:.1}%",
        m.text,
        m.questions,
        share(m.first),
        share(m.not_a_candidate),
        share(m.under_the_cut),
        share(m.outranked),
        share(m.current_among_newest)
    );
    println!(
        "  without the newest list: current record in the first 64 by meaning {:.1}%, by keywords {:.1}%, a candidate of the product {:.1}%",
        share(m.current_in_dense_64),
        share(m.current_in_sparse_64),
        share(m.current_a_product_candidate)
    );
    Ok(())
}

fn run_speech_trips(cli: SpeechTripsCli) -> anyhow::Result<()> {
    let composed = speech::run(&speech::SpeechTripsArgs {
        stt: cli.stt,
        voice: cli.voice,
        mask: cli.mask,
        extract: cli.extract,
        upload: cli.upload,
        upload_records: cli.upload_records,
        out: cli.out.clone(),
    })?;
    println!(
        "{} trips under {}, {} of them with an upload never found ({} voice lines without an extraction, \
         {} uploads left over); run cb-eval trips --run {} next",
        composed.trips,
        display(&cli.out),
        composed.trips_never_found,
        composed.voice_lines_without_extraction,
        composed.uploads_left_over,
        display(&cli.out)
    );
    Ok(())
}

fn display(path: &Path) -> String {
    path.display().to_string()
}

/// The output directory: the one given, or the run directory when there is one run.
fn out_dir(out: Option<PathBuf>, runs: &[PathBuf]) -> anyhow::Result<PathBuf> {
    match (out, runs) {
        (Some(out), _) => Ok(out),
        (None, [run]) => Ok(run.clone()),
        (None, _) => anyhow::bail!("several runs were given: name the output directory with --out"),
    }
}

fn percent(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |v| format!("{:.1}", 100.0 * v))
}

fn run_rank(cli: RankCli) -> anyhow::Result<()> {
    let methods = cli.methods.read()?;
    anyhow::ensure!(!methods.is_empty(), "no method was named");
    let done = rank::run(&rank::RankArgs {
        data: cli.data,
        run: cli.run.clone(),
        methods,
        keep: cli.keep,
    })?;
    println!(
        "{:<44} {:>9} {:>20} {:>22}",
        "method", "questions", "current@1 changing %", "hit@4 ordinary_all %"
    );
    for method in &done.methods {
        let test_mean = |group: Group| {
            method
                .means
                .iter()
                .find(|m| m.split == Split::Test && m.group == group)
        };
        println!(
            "{:<44} {:>9} {:>20} {:>22}",
            method.text,
            method.questions,
            percent(test_mean(Group::Changing).map(|m| m.means.current_1)),
            percent(test_mean(Group::OrdinaryAll).map(|m| m.means.hit_4)),
        );
    }
    println!(
        "test split only; wrote {} methods under {}",
        done.methods.len(),
        display(&cli.run)
    );
    // Holds no measure, so it goes where a caller hiding the test numbers still sees it.
    eprintln!(
        "{}: {} record(s) in the first 64 places of a `now` list had no relevance score, in {} question(s); \
         the product's methods left them out",
        display(&cli.run),
        done.unscored_records,
        done.questions_with_unscored
    );
    Ok(())
}

fn run_tune(cli: TuneCli) -> anyhow::Result<()> {
    let candidates = cli.methods.read()?;
    let out = out_dir(cli.out, &cli.runs)?;
    let guard = match (cli.guard_measure, cli.guard_group, cli.guard_baseline) {
        (Some(measure), Some(group), Some(baseline)) => Some(tune::GuardArgs {
            measure,
            group,
            baseline,
            margin: cli.guard_margin,
        }),
        _ => None,
    };
    let higher_is_better = match (cli.lower_is_better, cli.higher_is_better) {
        (true, _) => Some(false),
        (_, true) => Some(true),
        _ => None,
    };
    let choice = tune::run(&tune::TuneArgs {
        runs: cli.runs,
        out: out.clone(),
        label: cli.label,
        candidates,
        measure: cli.measure,
        group: cli.group,
        higher_is_better,
        guard,
    })?;
    let unit = choice.measure.unit();
    println!(
        "{}: {} on {} validation questions of group {} (runs {})",
        choice.label,
        choice.measure,
        choice.questions,
        choice.group,
        choice.runs.join(", ")
    );
    for candidate in &choice.candidates {
        let guard = match (&choice.guard, candidate.guard_value) {
            (Some(guard), Some(value)) => format!(
                "  {} {} {}",
                guard.measure,
                format_value(value, guard.measure.unit()),
                if candidate.within_guard {
                    "within guard"
                } else {
                    "outside guard"
                }
            ),
            _ => String::new(),
        };
        let mark = if candidate.method == choice.chosen {
            "*"
        } else {
            " "
        };
        println!(
            "{mark} {:<44} {:>8}{guard}",
            candidate.text,
            format_value(candidate.value, unit)
        );
    }
    println!(
        "chose {} (written to {})",
        choice.chosen,
        display(&out.join(TUNED))
    );
    Ok(())
}

fn run_report(cli: ReportCli) -> anyhow::Result<()> {
    let out = out_dir(cli.out, &cli.runs)?;
    let name = match (cli.name, cli.runs.as_slice()) {
        (Some(name), _) => name,
        (None, [run]) => label_of(run),
        (None, _) => anyhow::bail!("several runs were given: name the report with --name"),
    };
    let tuned = if cli.no_tuned {
        None
    } else {
        cli.tuned
            .or_else(|| Some(out.join(TUNED)).filter(|path| path.exists()))
    };
    let methods = cli.methods.read()?;
    let outcome = report::run(&report::ReportArgs {
        runs: cli.runs,
        out: out.clone(),
        name,
        methods: if methods.is_empty() { None } else { Some(methods) },
        baselines: if cli.no_baseline {
            Vec::new()
        } else {
            cli.baselines
        },
        tuned: tuned.clone(),
        feature: cli.feature,
        columns: if cli.columns.is_empty() {
            DEFAULT_COLUMNS.to_vec()
        } else {
            cli.columns
        },
        seed: cli.seed,
        draws: cli.draws,
        threads: cli.threads,
    })?;
    print_headline(&outcome.summary);
    for note in &outcome.notes {
        println!("note: {note}");
    }
    if let Some(tuned) = tuned {
        println!("applied the choices of {}", display(&tuned));
    }
    println!(
        "wrote {} files and {} numbers under {} (seed {}, {} draws)",
        outcome.files.len(),
        outcome.numbers,
        display(&out),
        outcome.summary.seed,
        outcome.summary.draws
    );
    Ok(())
}

/// The preregistered measures of claim 2: current@1 on changing questions, hit@4 on ordinary ones.
fn print_headline(summary: &Summary) {
    let baseline = summary.baselines.first();
    for (group, measure) in [
        (Group::Changing, Measure::Current1),
        (Group::OrdinaryAll, Measure::Hit4),
    ] {
        let Some(found) = summary.group(group) else {
            continue;
        };
        let unit = measure.unit();
        println!(
            "{group}: {measure}, {} questions in {} clusters (test split)",
            found.questions.iter().sum::<usize>(),
            found.clusters.iter().sum::<usize>()
        );
        for method in found.methods.iter().filter(|m| m.featured) {
            let Some(m) = method.measure(measure) else {
                continue;
            };
            let versus = baseline
                .and_then(|b| m.versus(b).map(|c| (b, c)))
                .map(|(b, c)| {
                    format!(
                        "  minus {b} {} [{}, {}] p={}",
                        format_difference(c.difference, unit),
                        format_difference(c.ci.lo, unit),
                        format_difference(c.ci.hi, unit),
                        format_p(c.p)
                    )
                })
                .unwrap_or_default();
            println!(
                "  {:<44} {:>6} [{}, {}]{versus}",
                method.text,
                format_value(m.mean, unit),
                format_value(m.ci.lo, unit),
                format_value(m.ci.hi, unit)
            );
        }
    }
}

/// The name a single run directory gives its outputs unless one is named.
fn name_or_label(name: Option<String>, run: &Path) -> String {
    name.unwrap_or_else(|| label_of(run))
}

fn run_gate(cli: GateCli) -> anyhow::Result<()> {
    let out = cli.out.unwrap_or_else(|| cli.run.clone());
    let done = gate::run(&gate::GateArgs {
        name: name_or_label(cli.name, &cli.run),
        data: cli.data,
        run: cli.run,
        transcripts: cli.transcripts,
        out: out.clone(),
        conversations: cli.conversations,
        bars: if cli.bars.is_empty() {
            gate::default_bars()
        } else {
            cli.bars
        },
        min_share: cli.min_share,
        seed: cli.seed,
        judge_sample: cli.judge_sample,
        judge: cli.judge,
        judge_promoted: cli.judge_promoted,
        threads: cli.threads,
        keep_calls: cli.keep_calls,
    })?;
    let summary = &done.summary;
    if summary.calls_thinned_out > 0 {
        println!(
            "calls kept: {} of every 1 ({} thinned out with their extractions)",
            summary.keep_calls, summary.calls_thinned_out
        );
    }
    for split in &summary.splits {
        println!(
            "{:?}: {} extractions ({} correct), {} true states ({} stated in two or more calls), {} calls without an extraction",
            split.split,
            split.extractions,
            split.correct_extractions,
            split.states,
            split.states_in_two_calls,
            split.calls_without_extraction
        );
    }
    println!(
        "{:<36} {:>5} {:>8} {:>10} {:>10} {:>11} {:>9} {:>9} {:>6}",
        "test split",
        "bar",
        "promoted",
        "precision%",
        "validated%",
        "2+ calls %",
        "wait p50",
        "wait p90",
        "mixed"
    );
    let percent = |v: Option<f64>| v.map_or_else(|| "--".to_string(), |v| format!("{:.1}", 100.0 * v));
    let minutes = |v: Option<f64>| v.map_or_else(|| "--".to_string(), |v| format!("{:.1}m", v / 60.0));
    for variant in &summary.variants {
        let t = &variant.test;
        println!(
            "{:<36} {:>5} {:>8} {:>10} {:>10} {:>11} {:>9} {:>9} {:>6}",
            variant.name,
            if variant.rule.conversations == 1 {
                "--".to_string()
            } else {
                variant.rule.bar.to_string()
            },
            t.promoted,
            percent(t.precision),
            percent(t.share_validated),
            percent(t.share_validated_in_two_calls),
            minutes(t.wait_median_s),
            minutes(t.wait_p90_s),
            t.false_merges
        );
    }
    if let Some(agreement) = &summary.judge.agreement {
        println!(
            "judge: {} judged, agreement {}, kappa {}",
            agreement.judged,
            percent(agreement.agreement),
            agreement
                .kappa
                .map_or_else(|| "--".to_string(), |k| format!("{k:.3}"))
        );
    }
    println!(
        "wrote {} files and {} numbers under {}",
        done.files.len(),
        done.numbers,
        display(&out.join(gate::GATE_DIR))
    );
    Ok(())
}

fn run_trips(cli: TripsCli) -> anyhow::Result<()> {
    let out = cli.out.unwrap_or_else(|| cli.run.clone());
    let done = trips::run(&trips::TripsArgs {
        name: name_or_label(cli.name, &cli.run),
        run: cli.run,
        out: out.clone(),
    })?;
    let summary = &done.summary;
    let ms = |v: Option<&f64>| v.map_or_else(|| "--".to_string(), |v| format!("{v:.0}"));
    println!(
        "{:<24} {:>7} {:>9} {:>9} {:>9} {:>9}",
        "path (whole trips)", "trips", "p50 ms", "p90 ms", "p95 ms", "p99 ms"
    );
    for path in &summary.paths {
        println!(
            "{:<24} {:>7} {:>9} {:>9} {:>9} {:>9}",
            path.path,
            path.trips,
            ms(path.percentiles_ms.get(&500)),
            ms(path.percentiles_ms.get(&900)),
            ms(path.percentiles_ms.get(&950)),
            ms(path.percentiles_ms.get(&990))
        );
    }
    for stage in &summary.stages {
        println!(
            "stage {:<18} {:>7} records, median {} ms",
            stage.stage,
            stage.records,
            stage
                .median_ms
                .map_or_else(|| "--".to_string(), |v| format!("{v:.1}"))
        );
    }
    let line = &summary.pass_line;
    println!(
        "claim 1: P95 of {} = {} ms over {} trips (limit {} ms, at least {} trips): {}",
        line.path,
        ms(line.value_ms.as_ref()),
        line.records,
        line.limit_ms,
        line.records_required,
        if line.met { "met" } else { "not met" }
    );
    println!(
        "wrote {} files under {}",
        done.files.len(),
        display(&out.join(trips::TRIPS_DIR))
    );
    Ok(())
}
