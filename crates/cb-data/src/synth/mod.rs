//! The generated incident corpus (docs/SYNTH_SPEC.md): 500 incidents of a
//! fictional provider, five cumulative states each, planted wrong statements,
//! 10,000 call transcripts, 320 help-article sections and five classes of
//! questions with graded gold. Everything is drawn from one ChaCha20 generator
//! seeded from `--seed`, so a seed reproduces every file byte for byte.
//! Decisions the specification leaves open are listed in `NOTES.md` next to
//! this file.

pub mod asr;
pub mod calls;
pub mod clock;
pub mod dialogue;
pub mod draw;
pub mod incidents;
pub mod pools;
pub mod questions;
pub mod records;
pub mod text;

use crate::files::{self, DataSet, Provenance};
use calls::{Agent, Call, Knowledge};
use cb_types::{QueryClass, Transcript, write_jsonl};
use dialogue::{CallSetting, WrittenCall};
use incidents::{Incident, LengthClass, PlantedPair};
use pools::Pools;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// The counts docs/SYNTH_SPEC.md fixes. `cb-data check` holds a generated data
/// set to them.
pub mod spec {
    pub const INCIDENTS: usize = 500;
    pub const SHORT: usize = 200;
    pub const MEDIUM: usize = 200;
    pub const LONG: usize = 100;
    pub const STATES: usize = 5;
    pub const TRUE_RECORDS: usize = INCIDENTS * STATES;
    pub const INCIDENTS_WITH_THREE_WRONG: usize = 250;
    pub const WRONG_RECORDS: usize = 1_250;
    pub const INCIDENTS_WITH_NEXT_UPDATE: usize = 250;
    pub const INCIDENTS_WITHOUT_WORKAROUND: usize = 100;
    pub const PLANTED_PAIRS: usize = 60;
    pub const STANDING_RECORDS: usize = 320;
    pub const TRANSCRIPTS: usize = 10_000;
    pub const MIN_CALLS_PER_INCIDENT: usize = 6;
    pub const CHANGING_QUESTIONS: usize = 4_000;
    pub const POST_INCIDENT_QUESTIONS: usize = 500;
    pub const AMBIGUOUS_QUESTIONS: usize = PLANTED_PAIRS;
    pub const ORDINARY_QUESTIONS: usize = STANDING_RECORDS;
    pub const ORDINARY_NEAR_QUESTIONS: usize = 100;
    /// Mean transcript length in tokens (words × 1.33) and the allowed distance from it.
    pub const MEAN_TOKENS: f64 = 850.0;
    pub const MEAN_TOKENS_TOLERANCE: f64 = 50.0;
    /// `Record.source` of every generated record.
    pub const SOURCE: &str = "synthetic";
    /// How the manifest's `made_by` begins.
    pub const MADE_BY: &str = "cb-data synth";
}

#[derive(Debug, Error)]
pub enum SynthError {
    #[error("asset {file}: {source}")]
    Asset {
        file: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("content pool: {0}")]
    Pool(String),
    #[error("template {template:?}: {problem}")]
    Template { template: String, problem: String },
    #[error("generation: {0}")]
    Plan(String),
    #[error("moment {0} cannot be shown as a date")]
    Time(i64),
    #[error("records {first} and {second} have the same text")]
    SharedText { first: String, second: String },
}

/// `cb-data synth` arguments.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// Seed of the one ChaCha20 generator everything is drawn from.
    #[arg(long)]
    pub seed: u64,
    /// Directory to write the data set into; created when absent.
    #[arg(long)]
    pub out: PathBuf,
    /// Also write the transcripts with this share of words altered by
    /// speech-recognition-style errors (transcripts.asrNN.jsonl). Repeat for
    /// several rates; each copy is drawn independently of the others.
    #[arg(long, value_name = "P")]
    pub asr: Vec<f64>,
}

/// The name a question class has in the files.
pub fn class_name(class: QueryClass) -> &'static str {
    match class {
        QueryClass::Changing => "changing",
        QueryClass::PostIncident => "post_incident",
        QueryClass::Ambiguous => "ambiguous",
        QueryClass::Ordinary => "ordinary",
        QueryClass::OrdinaryNear => "ordinary_near",
    }
}

/// What it took to make the calls.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Calls in which the agent did not yet know the cause.
    pub eligible_calls: usize,
    /// Calls in which the agent stated a wrong statement.
    pub wrong_calls: usize,
    /// Calls moved early so every wrong statement is stated at least once.
    pub moved_calls: usize,
}

/// A generated corpus with the plan it came from.
pub struct Corpus {
    pub seed: u64,
    pub data: DataSet,
    pub pools: Pools,
    pub incidents: Vec<Incident>,
    pub pairs: Vec<PlantedPair>,
    pub agents: Vec<Agent>,
    /// The planned calls, in the order of `data.transcripts`.
    pub calls: Vec<Call>,
    pub knowledge: Knowledge,
    pub stats: Stats,
    /// The generator's state after the corpus was drawn; every copy with
    /// speech-recognition errors starts from it.
    after: ChaCha20Rng,
}

impl Corpus {
    /// A copy of the transcripts with speech-recognition-style errors at rate `p`.
    pub fn transcripts_with_speech_errors(&self, p: f64) -> Vec<Transcript> {
        let mut rng = self.after.clone();
        asr::with_speech_errors(&self.data.transcripts, &self.pools.asr, p, &mut rng)
    }

    /// Counts for the manifest beyond records, questions and transcripts.
    pub fn manifest_counts(&self) -> BTreeMap<String, u64> {
        let mut counts = BTreeMap::new();
        let mut put = |name: String, n: usize| {
            counts.insert(name, n as u64);
        };
        put("incidents".into(), self.incidents.len());
        for class in [LengthClass::Short, LengthClass::Medium, LengthClass::Long] {
            let n = self.incidents.iter().filter(|i| i.class == class).count();
            put(format!("incidents_{}", class.name()), n);
        }
        put("planted_pairs".into(), self.pairs.len());
        let records = &self.data.records;
        let with_truth = |correct: bool| {
            records
                .iter()
                .filter(|r| r.truth.as_ref().is_some_and(|t| t.correct == correct))
                .count()
        };
        put("records_true".into(), with_truth(true));
        put("records_wrong".into(), with_truth(false));
        put(
            "records_standing".into(),
            records.iter().filter(|r| r.truth.is_none()).count(),
        );
        for class in [
            QueryClass::Changing,
            QueryClass::PostIncident,
            QueryClass::Ambiguous,
            QueryClass::Ordinary,
            QueryClass::OrdinaryNear,
        ] {
            let n = self.data.queries.iter().filter(|q| q.class == class).count();
            put(format!("queries_{}", class_name(class)), n);
        }
        put("calls_before_cause_known".into(), self.stats.eligible_calls);
        put("calls_with_wrong_statement".into(), self.stats.wrong_calls);
        put("calls_moved_early".into(), self.stats.moved_calls);
        counts
    }
}

/// Draw the whole corpus from `seed`.
pub fn generate(seed: u64) -> Result<Corpus, SynthError> {
    let pools = Pools::load()?;
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let plan = incidents::plan_incidents(&pools, &mut rng)?;
    let mut incidents = plan.incidents;
    let agents = calls::hire_agents(&pools, &mut rng);
    let knowledge = Knowledge::draw(&incidents, agents.len(), &mut rng);
    let schedule = calls::schedule_calls(&mut incidents, &knowledge, agents.len(), &mut rng)?;

    // Write every call.
    let mut written: Vec<(usize, WrittenCall)> = Vec::with_capacity(schedule.calls.len());
    for call in &schedule.calls {
        let incident = &incidents[call.incident];
        let known = knowledge.latest_known(call.incident, call.agent, call.started_at);
        let wrong = match call.wrong {
            Some(j) => Some(
                incident
                    .wrong
                    .get(j)
                    .ok_or_else(|| SynthError::Plan(format!("{}: no wrong statement {j}", incident.id)))?,
            ),
            None => None,
        };
        let setting = CallSetting {
            pools: &pools,
            incident,
            agent: &agents[call.agent],
            started_at: call.started_at,
            known,
            wrong,
        };
        written.push((known, dialogue::write_call(&setting, &mut rng)?));
    }

    // A wrong statement is written when it is first said; its id follows that order.
    let mut first_said: Vec<Vec<Option<i64>>> = incidents.iter().map(|i| vec![None; i.wrong.len()]).collect();
    for (call, (_, w)) in schedule.calls.iter().zip(&written) {
        if let (Some(j), Some(at)) = (call.wrong, w.wrong_said_at_s)
            && let Some(slot) = first_said.get_mut(call.incident).and_then(|v| v.get_mut(j))
        {
            let moment = call.started_at + at.floor() as i64;
            *slot = Some(slot.map_or(moment, |m| m.min(moment)));
        }
    }
    let mut wrong_ids: Vec<Vec<String>> = Vec::with_capacity(incidents.len());
    for (i, incident) in incidents.iter().enumerate() {
        let mut order: Vec<usize> = (0..incident.wrong.len()).collect();
        order.sort_by_key(|j| (first_said[i][*j], *j));
        let mut ids = vec![String::new(); incident.wrong.len()];
        for (rank, j) in order.into_iter().enumerate() {
            ids[j] = format!("{}.d{}", incident.id, rank + 1);
        }
        wrong_ids.push(ids);
    }

    let transcripts: Vec<Transcript> = schedule
        .calls
        .iter()
        .zip(written)
        .enumerate()
        .map(|(n, (call, (known, w)))| {
            let incident = &incidents[call.incident];
            let mut states = Vec::new();
            if known >= 1 {
                states.push(incident.state_id(known));
            }
            if let Some(j) = call.wrong {
                states.push(wrong_ids[call.incident][j].clone());
            }
            let agent = &agents[call.agent];
            Transcript {
                id: format!("call-{:05}", n + 1),
                thread: incident.id.clone(),
                started_at: call.started_at,
                agent: agent.id.clone(),
                persona: agent.style.name().to_string(),
                turns: w.turns,
                states,
                asr_error_rate: 0.0,
            }
        })
        .collect();

    let mut records = Vec::with_capacity(spec::TRUE_RECORDS + spec::WRONG_RECORDS + spec::STANDING_RECORDS);
    for (i, incident) in incidents.iter().enumerate() {
        records.extend(records::state_records(&pools, incident, &mut rng)?);
        let mut notes = Vec::with_capacity(incident.wrong.len());
        for (j, statement) in incident.wrong.iter().enumerate() {
            let written_at = first_said[i][j].ok_or_else(|| {
                SynthError::Plan(format!("{}: a wrong statement is never said", incident.id))
            })?;
            notes.push(records::wrong_record(
                &pools,
                incident,
                statement,
                wrong_ids[i][j].clone(),
                written_at,
                &mut rng,
            )?);
        }
        notes.sort_by(|a, b| a.id.cmp(&b.id));
        records.extend(notes);
    }
    records.extend(records::standing_records(&pools, &mut rng));
    if let Some((first, second)) = records::first_shared_text(&records) {
        return Err(SynthError::SharedText { first, second });
    }

    let queries = questions::write_questions(&pools, &incidents, &plan.pairs, &wrong_ids, &mut rng)?;
    Ok(Corpus {
        seed,
        data: DataSet {
            records,
            queries,
            transcripts,
        },
        pools,
        incidents,
        pairs: plan.pairs,
        agents,
        calls: schedule.calls,
        knowledge,
        stats: Stats {
            eligible_calls: schedule.eligible_calls,
            wrong_calls: schedule.wrong_calls,
            moved_calls: schedule.moved_calls,
        },
        after: rng,
    })
}

fn write_jsonl_file<T: serde::Serialize>(path: &Path, items: &[T]) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_jsonl(&mut w, items)?;
    w.flush()?;
    Ok(())
}

/// Generate the corpus for `seed` and write it, the speech-error copies asked
/// for, and a manifest with every file's hash into `out`.
pub fn write_corpus(seed: u64, out: &Path, asr_rates: &[f64]) -> anyhow::Result<Corpus> {
    check_rates(asr_rates)?;
    let corpus = generate(seed)?;
    write_generated(&corpus, out, asr_rates)?;
    Ok(corpus)
}

fn check_rates(asr_rates: &[f64]) -> anyhow::Result<()> {
    for p in asr_rates {
        anyhow::ensure!((0.0..=1.0).contains(p), "--asr must be between 0 and 1, got {p}");
    }
    Ok(())
}

/// Write a generated corpus, the speech-error copies asked for, and a manifest
/// with every file's hash into `out`.
pub fn write_generated(corpus: &Corpus, out: &Path, asr_rates: &[f64]) -> anyhow::Result<()> {
    check_rates(asr_rates)?;
    let seed = corpus.seed;
    std::fs::create_dir_all(out)?;
    let mut extra: Vec<String> = Vec::new();
    for p in asr_rates {
        let name = asr::file_name(*p);
        anyhow::ensure!(!extra.contains(&name), "two --asr rates would both write {name}");
        write_jsonl_file(&out.join(&name), &corpus.transcripts_with_speech_errors(*p))?;
        extra.push(name);
    }
    let extra_names: Vec<&str> = extra.iter().map(String::as_str).collect();
    let provenance = Provenance {
        name: out
            .file_name()
            .map_or_else(|| format!("synth-{seed}"), |n| n.to_string_lossy().into_owned()),
        made_by: format!("{} {}", spec::MADE_BY, env!("CARGO_PKG_VERSION")),
        seed: Some(seed),
        sources: vec![
            "Generated by cb-data synth to docs/SYNTH_SPEC.md; every company, place, person, \
             account and number in it is invented."
                .to_string(),
        ],
    };
    let mut manifest = files::write_data_set(out, &corpus.data, &provenance, &extra_names)?;
    manifest.counts.extend(corpus.manifest_counts());
    let mut w = BufWriter::new(File::create(out.join(files::MANIFEST))?);
    serde_json::to_writer_pretty(&mut w, &manifest)?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(())
}

/// Mean, standard deviation, smallest and largest transcript length in tokens.
pub fn transcript_token_stats(transcripts: &[Transcript]) -> (f64, f64, f64, f64) {
    let tokens: Vec<f64> = transcripts
        .iter()
        .map(|t| text::tokens_of_words(t.turns.iter().map(|turn| text::word_count(&turn.text)).sum()))
        .collect();
    if tokens.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = tokens.len() as f64;
    let mean = tokens.iter().sum::<f64>() / n;
    let sd = (tokens.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n).sqrt();
    let min = tokens.iter().copied().fold(f64::INFINITY, f64::min);
    let max = tokens.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mean, sd, min, max)
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let corpus = write_corpus(args.seed, &args.out, &args.asr)?;
    let (mean, sd, min, max) = transcript_token_stats(&corpus.data.transcripts);
    println!("wrote {}", args.out.display());
    println!(
        "  records {}  queries {}  transcripts {}",
        corpus.data.records.len(),
        corpus.data.queries.len(),
        corpus.data.transcripts.len()
    );
    for (name, n) in corpus.manifest_counts() {
        println!("  {name}: {n}");
    }
    println!("  transcript tokens (words x 1.33): mean {mean:.1}, sd {sd:.1}, min {min:.0}, max {max:.0}");
    Ok(())
}

/// One corpus (seed 42) shared by every test of the crate, drawn once.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::{Corpus, generate};
    use std::sync::OnceLock;

    pub(crate) fn shared_corpus() -> &'static Corpus {
        static CORPUS: OnceLock<Corpus> = OnceLock::new();
        CORPUS.get_or_init(|| generate(42).expect("seed 42 generates"))
    }
}

#[cfg(test)]
mod tests;
