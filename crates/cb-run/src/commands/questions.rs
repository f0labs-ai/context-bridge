//! `cb-run questions`: ordinary questions for a data set's standing records, written by a
//! language model, added to the data set's `queries.jsonl`.
//!
//! What the model writes goes to `<run>/questions.jsonl` first, line by line as it comes,
//! so a stopped run is picked up where it stopped. At the end the data set's
//! `queries.jsonl` is replaced in one step by itself plus the new questions, and the data
//! set's manifest is rewritten to match: the data set is never left half-changed.

use super::{
    DATA_MANIFEST_FILE, Prompt, QUERIES_FILE, Shared, belonging, extra_body_identity, in_order, read_queries,
    read_records,
};
use crate::atomic::{sha256_of_file, write_atomically, write_json_atomically};
use crate::error::{FailureKind, RunError};
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::model::{Ask, ExtraBody, ModelAnswer, ModelEndpoint};
use crate::retry::with_retries;
use crate::run::{About, ModelRecord, Report, Session, VERSION};
use crate::seeded::Draws;
use crate::text::read_questions;
use cb_types::{Gold, Query, QueryClass, Record, Tier, UnixSeconds, fnv1a64};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

/// The run file the model's questions are written to before they join the data set.
pub const QUESTIONS_FILE: &str = "questions.jsonl";
/// Where the run's files go, inside the data set, when `--run` is not given.
pub const DEFAULT_RUN_DIR: &str = "run-questions";
const COMMAND: &str = "questions";
const BUILT_IN_PROMPT: &str = include_str!("../../prompts/questions.v1.md");
const BUILT_IN_PROMPT_VERSION: &str = "questions.v1";
/// The id the failure "fewer standing records than asked for" is filed under.
const SAMPLE: &str = "sample";

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct QuestionsArgs {
    /// The data set directory; records.jsonl is read, queries.jsonl and manifest.json are rewritten.
    #[arg(long)]
    pub data: PathBuf,
    /// The model's name, as the endpoint knows it.
    #[arg(long)]
    pub model: String,
    /// Questions written per sampled record.
    #[arg(long, default_value_t = 1)]
    pub per_section: usize,
    /// How many standing records to sample.
    #[arg(long)]
    pub sample: usize,
    /// The seed of the sample and of every question's moment.
    #[arg(long)]
    pub seed: u64,
    /// Where failures.jsonl and the record of the run go. Default: <data>/run-questions.
    #[arg(long)]
    pub run: Option<PathBuf>,
    /// A prompt file in place of the built-in one ({{count}}, {{title}}, {{text}}).
    #[arg(long)]
    pub prompt: Option<PathBuf>,
    /// Calls in flight at once.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// A file of product words or phrases, one per line: a question is "near" an incident
    /// only when both titles hold one of them. Without it, any title word of three
    /// letters or more that is not a common word counts.
    #[arg(long)]
    pub product_words: Option<PathBuf>,
    /// A bound on the answer's length, sent as max_tokens. Not sent when absent.
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// A JSON object whose fields are added at the top level of every request body, for
    /// an endpoint that needs them (e.g. '{"chat_template_kwargs":{"enable_thinking":false}}').
    /// Fields cb-run sets itself are refused. Recorded in the run manifest: no secrets.
    #[arg(long)]
    pub extra_body: Option<ExtraBody>,
}

// ---- incidents and the "near" rule -----------------------------------------------------

/// An incident of the data set: the changing records of one thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Incident {
    pub thread: String,
    pub title: Option<String>,
    /// Its first record's moment.
    pub from: UnixSeconds,
    /// Its last record's moment.
    pub to: UnixSeconds,
}

/// The incidents of a data set, in thread order.
pub fn incidents_of(records: &[Record]) -> Vec<Incident> {
    let mut by_thread: BTreeMap<&str, Incident> = BTreeMap::new();
    for record in records.iter().filter(|r| r.tier == Tier::Changing) {
        let Some(thread) = record.thread.as_deref() else {
            continue;
        };
        let incident = by_thread.entry(thread).or_insert_with(|| Incident {
            thread: thread.to_string(),
            title: None,
            from: record.written_at,
            to: record.written_at,
        });
        incident.from = incident.from.min(record.written_at);
        incident.to = incident.to.max(record.written_at);
        if incident.title.is_none() {
            incident.title.clone_from(&record.title);
        }
    }
    by_thread.into_values().collect()
}

/// Words that appear in titles without naming a product.
const COMMON_WORDS: &[&str] = &[
    "about",
    "affected",
    "all",
    "and",
    "any",
    "are",
    "availability",
    "can",
    "customers",
    "degradation",
    "degraded",
    "delay",
    "delayed",
    "delays",
    "disruption",
    "down",
    "elevated",
    "error",
    "errors",
    "experiencing",
    "failing",
    "failure",
    "failures",
    "for",
    "from",
    "get",
    "had",
    "has",
    "have",
    "how",
    "impact",
    "impacting",
    "incident",
    "incidents",
    "increased",
    "intermittent",
    "into",
    "investigating",
    "issue",
    "issues",
    "its",
    "latency",
    "new",
    "not",
    "our",
    "outage",
    "outages",
    "partial",
    "performance",
    "problem",
    "problems",
    "request",
    "requests",
    "service",
    "services",
    "set",
    "slow",
    "some",
    "status",
    "that",
    "the",
    "their",
    "this",
    "timeouts",
    "unavailable",
    "update",
    "use",
    "used",
    "users",
    "using",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
];

/// The lower-case words of a title.
fn words_of(title: &str) -> Vec<String> {
    title
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// What counts as a product word when two titles are compared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProductWords {
    /// Only these words or phrases (lower case, single spaces).
    Listed(Vec<String>),
    /// Any title word of three letters or more that is not a common word.
    AnyTitleWord,
}

impl ProductWords {
    /// A file of words or phrases, one per line; empty lines and lines starting with `#` are passed over.
    pub fn read(path: &Path) -> Result<Self, RunError> {
        let text = std::fs::read_to_string(path).map_err(|e| RunError::io(path, e))?;
        let listed: Vec<String> = text
            .lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .map(|line| words_of(line).join(" "))
            .filter(|phrase| !phrase.is_empty())
            .collect();
        if listed.is_empty() {
            return Err(RunError::file(path, "no product word in the file"));
        }
        Ok(Self::Listed(listed))
    }

    /// Whether two titles share a product word.
    pub fn shared_by(&self, a: &str, b: &str) -> bool {
        let (a, b) = (words_of(a), words_of(b));
        match self {
            Self::Listed(listed) => {
                let padded = |words: &[String]| format!(" {} ", words.join(" "));
                let (a, b) = (padded(&a), padded(&b));
                listed.iter().any(|phrase| {
                    let phrase = format!(" {phrase} ");
                    a.contains(&phrase) && b.contains(&phrase)
                })
            }
            Self::AnyTitleWord => a.iter().any(|word| {
                word.chars().count() >= 3 && !COMMON_WORDS.contains(&word.as_str()) && b.contains(word)
            }),
        }
    }
}

// ---- the plan --------------------------------------------------------------------------

/// A question that will be asked for: its id, its moment and its class. Its words come
/// from the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    pub id: String,
    pub asked_at: UnixSeconds,
    pub class: QueryClass,
}

/// The id of the `k`-th question written for a record: fixed by the record, valid as an
/// id whatever the record's own id looks like, and the same on every run.
pub fn question_id(record_id: &str, k: usize) -> String {
    format!("oq-{:016x}-{k}", fnv1a64(record_id.as_bytes()))
}

/// The questions planned for one record: each asked at a moment drawn inside a randomly
/// chosen incident, never before the record itself was written.
fn plan_for(
    record: &Record,
    incidents: &[Incident],
    words: &ProductWords,
    per_section: usize,
    draws: &mut Draws,
) -> Result<Vec<Planned>, String> {
    let open: Vec<&Incident> = incidents.iter().filter(|i| i.to >= record.written_at).collect();
    if open.is_empty() {
        return Err("every incident of the data set ended before this record was written".into());
    }
    let mut planned = Vec::with_capacity(per_section);
    for k in 1..=per_section {
        let incident = open[draws.below(open.len() as u64) as usize];
        let asked_at = draws.between(incident.from.max(record.written_at), incident.to);
        let near = match (&incident.title, &record.title) {
            (Some(incident_title), Some(record_title)) => words.shared_by(incident_title, record_title),
            _ => false,
        };
        planned.push(Planned {
            id: question_id(&record.id, k),
            asked_at,
            class: if near {
                QueryClass::OrdinaryNear
            } else {
                QueryClass::Ordinary
            },
        });
    }
    Ok(planned)
}

/// The seeded sample of standing records with the questions planned for each. Every draw
/// is made here, before any call to the model, so no call's outcome can move another
/// question's moment.
///
/// The sample is the first `sample` picks of one seeded shuffle, so a larger sample holds
/// a smaller one. A record's questions are drawn from a stream of the seed and the
/// record's own id, so they are the same whatever else is sampled: a run can be extended
/// without moving a question already written.
pub fn plan<'a>(
    records: &'a [Record],
    words: &ProductWords,
    sample: usize,
    per_section: usize,
    seed: u64,
) -> Vec<(&'a Record, Result<Vec<Planned>, String>)> {
    let incidents = incidents_of(records);
    let mut standing: Vec<&Record> = records.iter().filter(|r| r.tier == Tier::Standing).collect();
    standing.sort_by(|a, b| a.id.cmp(&b.id));
    let sampled = Draws::new(seed).sample(standing, sample);
    sampled
        .into_iter()
        .map(|record| {
            let mut draws = Draws::keyed(seed, &record.id);
            (
                record,
                plan_for(record, &incidents, words, per_section, &mut draws),
            )
        })
        .collect()
}

/// Refuses when a question that `planned` holds is already held (in the data set or
/// this run's file) with another moment or class.
fn check_plan_against<'a>(
    planned: &[(&Record, Result<Vec<Planned>, String>)],
    held: impl Iterator<Item = &'a Query>,
    queries_file: &Path,
) -> Result<(), RunError> {
    let by_id: BTreeMap<&str, &Planned> = planned
        .iter()
        .filter_map(|(_, questions)| questions.as_ref().ok())
        .flatten()
        .map(|question| (question.id.as_str(), question))
        .collect();
    for query in held {
        if let Some(plan) = by_id.get(query.id.as_str())
            && (query.asked_at != plan.asked_at || query.class != plan.class)
        {
            return Err(RunError::file(
                queries_file,
                format!(
                    "question {} is held as asked at {} ({:?}) and this run plans it at {} ({:?}); \
                     it was planned with another seed or product-word list",
                    query.id, query.asked_at, query.class, plan.asked_at, plan.class
                ),
            ));
        }
    }
    Ok(())
}

// ---- the prompt ------------------------------------------------------------------------

/// A template with each placeholder replaced once, in one pass: a value is never read
/// for placeholders, so a passage that holds `{{title}}` in its text stays as it is.
fn fill(template: &str, values: &[(&str, &str)]) -> String {
    let mut filled = String::with_capacity(template.len());
    let mut rest = template;
    'template: while !rest.is_empty() {
        for (placeholder, value) in values {
            if let Some(after) = rest.strip_prefix(placeholder) {
                filled.push_str(value);
                rest = after;
                continue 'template;
            }
        }
        let mut chars = rest.chars();
        if let Some(c) = chars.next() {
            filled.push(c);
        }
        rest = chars.as_str();
    }
    filled
}

fn message_for(prompt: &Prompt, record: &Record, count: usize) -> String {
    fill(
        &prompt.text,
        &[
            ("{{count}}", &count.to_string()),
            ("{{title}}", record.title.as_deref().unwrap_or("(none)")),
            ("{{text}}", &record.text),
        ],
    )
}

/// One try for one record: the call, then the answer read strictly.
async fn ask_for_questions<M: ModelEndpoint>(
    model: &M,
    args: &QuestionsArgs,
    record_id: &str,
    message: &str,
    count: usize,
) -> Result<(Vec<String>, ModelAnswer), RunError> {
    let answer = model
        .ask(&Ask {
            model: &args.model,
            message,
            max_tokens: args.max_tokens,
            extra: args.extra_body.as_ref(),
        })
        .await?;
    match read_questions(&answer.content, count) {
        Ok(questions) => Ok((questions, answer)),
        Err(reason) => Err(RunError::AnswerNotAsked {
            id: record_id.to_string(),
            url: model.address(),
            reason,
            raw: answer.content,
        }),
    }
}

// ---- the data set ----------------------------------------------------------------------

/// Adds `new` to the data set's `queries.jsonl` (replaced in one step by its old bytes
/// plus the new lines) and rewrites the data set's manifest: the count and the hash of
/// the questions file, and one line that says where the added questions came from.
fn add_to_data_set(data: &Path, new: &[&Query], provenance: &str) -> Result<(), RunError> {
    let queries_path = data.join(QUERIES_FILE);
    let manifest_path = data.join(DATA_MANIFEST_FILE);
    let manifest_bytes = std::fs::read(&manifest_path).map_err(|e| RunError::io(&manifest_path, e))?;
    let mut manifest: cb_types::Manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| RunError::file(&manifest_path, format!("not a data set manifest: {e}")))?;
    if !new.is_empty() {
        let old = std::fs::read(&queries_path).map_err(|e| RunError::io(&queries_path, e))?;
        write_atomically(&queries_path, |writer| {
            writer.write_all(&old)?;
            if !old.is_empty() && old.last() != Some(&b'\n') {
                writer.write_all(b"\n")?;
            }
            for query in new {
                serde_json::to_writer(&mut *writer, query).map_err(std::io::Error::other)?;
                writer.write_all(b"\n")?;
            }
            Ok(())
        })?;
        if !manifest.sources.iter().any(|line| line == provenance) {
            manifest.sources.push(provenance.to_string());
        }
    }
    // Always: a run stopped between the two files leaves the next run to finish the job.
    let total = read_queries(data)?.len() as u64;
    manifest.counts.insert("queries".to_string(), total);
    manifest
        .sha256
        .insert(QUERIES_FILE.to_string(), sha256_of_file(&queries_path)?);
    write_json_atomically(&manifest_path, &manifest)
}

// ---- the command -----------------------------------------------------------------------

/// Writes the questions the data set does not hold yet.
pub async fn questions<M: ModelEndpoint>(
    model: &M,
    args: &QuestionsArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    if args.per_section == 0 {
        return Err(RunError::Setting {
            name: "--per-section",
            reason: "at least one question per record".into(),
        });
    }
    let prompt = match &args.prompt {
        Some(path) => Prompt::read(path)?,
        None => Prompt::of(BUILT_IN_PROMPT_VERSION, BUILT_IN_PROMPT),
    };
    let words = match &args.product_words {
        Some(path) => ProductWords::read(path)?,
        None => ProductWords::AnyTitleWord,
    };
    let records = read_records(&args.data)?;
    let queries = read_queries(&args.data)?;
    let manifest_path = args.data.join(DATA_MANIFEST_FILE);
    if !manifest_path.is_file() {
        return Err(RunError::file(
            &manifest_path,
            "a data set has a manifest; build it with cb-data first",
        ));
    }
    if incidents_of(&records).is_empty() {
        return Err(RunError::file(
            args.data.join(super::RECORDS_FILE),
            "no incident (a changing record with a thread) to ask a question in",
        ));
    }
    let run = args
        .run
        .clone()
        .unwrap_or_else(|| args.data.join(DEFAULT_RUN_DIR));
    let about = About {
        model: Some(ModelRecord {
            url: model.address(),
            name: args.model.clone(),
            reported_names: Vec::new(),
            prompt_version: prompt.version.clone(),
            prompt_sha256: prompt.sha256.clone(),
            json_mode: None,
            extra_body: args.extra_body.clone(),
        }),
        machine_label: shared.machine_label.clone(),
        ..About::default()
    };
    let mut session = Session::begin(&run, COMMAND, args, about)?;
    match work(
        model,
        args,
        shared,
        (&prompt, &words),
        (&records, &queries),
        &mut session,
    )
    .await
    {
        // The shortfall is judged anew on every run; a record is done when its questions exist.
        Ok(done) => session.finish(|id| id == SAMPLE || done.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn work<M: ModelEndpoint>(
    model: &M,
    args: &QuestionsArgs,
    shared: &Shared,
    (prompt, words): (&Prompt, &ProductWords),
    (records, queries): (&[Record], &[Query]),
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    let product_words_sha256 = match &args.product_words {
        Some(file) => sha256_of_file(file)?,
        None => "none".to_string(),
    };
    session.claim(
        QUESTIONS_FILE,
        belonging(&[
            ("model_endpoint", &model.address()),
            ("model", &args.model),
            ("prompt_sha256", &prompt.sha256),
            (
                "records_sha256",
                &sha256_of_file(&args.data.join(super::RECORDS_FILE))?,
            ),
            ("seed", &args.seed.to_string()),
            ("product_words_sha256", &product_words_sha256),
            ("extra_body", &extra_body_identity(args.extra_body.as_ref())),
        ]),
    )?;
    let standing = records.iter().filter(|r| r.tier == Tier::Standing).count();
    session.set("standing_records", standing as u64);
    if standing < args.sample {
        let error = RunError::item(
            SAMPLE,
            format!(
                "{} standing records were asked for and the data set has {standing}",
                args.sample
            ),
        );
        session.fail_as(SAMPLE, FailureKind::Shortfall, &error, 0)?;
    }
    let planned = plan(records, words, args.sample, args.per_section, args.seed);

    let in_data: HashSet<&str> = queries.iter().map(|q| q.id.as_str()).collect();
    let path = session.file(QUESTIONS_FILE);
    let existing = read_resumable::<Query>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let mut written: Vec<Query> = existing.lines;
    // A question this run plans that the data set (or this run's file) already holds
    // must be the same question: asked at the same moment, of the same class. Otherwise
    // it was planned with another seed or another product-word list, and skipping it as
    // "already there" would mix two plans in one data set without a word.
    check_plan_against(
        &planned,
        queries.iter().chain(written.iter()),
        &args.data.join(QUERIES_FILE),
    )?;
    let mut have: HashSet<String> = in_data.iter().map(|id| id.to_string()).collect();
    have.extend(written.iter().map(|q| q.id.clone()));
    let mut out = Appender::open(&path)?;

    let mut done: HashSet<String> = HashSet::new();
    let mut todo = Vec::new();
    for (record, plan) in &planned {
        session.add("selected", 1);
        match plan {
            Err(reason) => session.fail(&record.id, &RunError::item(&record.id, reason.clone()), 0)?,
            Ok(questions) if questions.iter().all(|q| have.contains(&q.id)) => {
                session.add("already_there", 1);
                done.insert(record.id.clone());
            }
            Ok(questions) => todo.push((*record, questions)),
        }
    }

    let retry = &shared.retry;
    let mut reported: BTreeSet<String> = BTreeSet::new();
    in_order(
        todo,
        args.concurrency,
        |(record, questions)| async move {
            let message = message_for(prompt, record, questions.len());
            let outcome = with_retries(retry, |_| {
                ask_for_questions(model, args, &record.id, &message, questions.len())
            })
            .await;
            (record, questions, outcome)
        },
        |(record, questions, outcome)| {
            match outcome {
                Ok(tried) => {
                    let (texts, answer) = tried.value;
                    if let Some(name) = answer.reported_model {
                        reported.insert(name);
                    }
                    if answer.tokens_in.is_none() || answer.tokens_out.is_none() {
                        session.add("answers_without_token_counts", 1);
                    }
                    session.add("tokens_in", u64::from(answer.tokens_in.unwrap_or(0)));
                    session.add("tokens_out", u64::from(answer.tokens_out.unwrap_or(0)));
                    let lines: Vec<Query> = questions
                        .iter()
                        .zip(texts)
                        .filter(|(planned, _)| !have.contains(&planned.id))
                        .map(|(planned, text)| Query {
                            id: planned.id.clone(),
                            text,
                            asked_at: planned.asked_at,
                            class: planned.class,
                            gold: vec![Gold {
                                record: record.id.clone(),
                                grade: 3,
                            }],
                            thread: None,
                            harmful: Vec::new(),
                        })
                        .collect();
                    out.append(&lines)?;
                    session.add("questions_written", lines.len() as u64);
                    for line in lines {
                        have.insert(line.id.clone());
                        written.push(line);
                    }
                    done.insert(record.id.clone());
                    session.wrote();
                }
                Err(gave_up) => session.fail(&record.id, &gave_up.error, gave_up.tries)?,
            }
            Ok(())
        },
    )
    .await?;
    out.sync()?;

    let new: Vec<&Query> = written
        .iter()
        .filter(|q| !in_data.contains(q.id.as_str()))
        .collect();
    let provenance = format!(
        "ordinary questions written by cb-run {VERSION} questions: model {}, prompt {} (sha256 {}), seed {}",
        args.model, prompt.version, prompt.sha256, args.seed
    );
    add_to_data_set(&args.data, &new, &provenance)?;
    session.set("questions_added_to_data_set", new.len() as u64);
    if let Some(record) = session.model_mut() {
        record.reported_names = reported.into_iter().collect();
        record.json_mode = Some(model.json_mode());
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, tier: Tier, thread: Option<&str>, title: Option<&str>, written_at: i64) -> Record {
        Record {
            id: id.into(),
            text: format!("Text of {id}."),
            written_at,
            valid_until: None,
            thread: thread.map(str::to_string),
            tier,
            title: title.map(str::to_string),
            source: "test".into(),
            truth: None,
        }
    }

    fn data() -> Vec<Record> {
        let mut records = vec![
            record(
                "gh-1.1",
                Tier::Changing,
                Some("gh-1"),
                Some("Incident with Actions"),
                1_000,
            ),
            record(
                "gh-1.2",
                Tier::Changing,
                Some("gh-1"),
                Some("Incident with Actions"),
                1_600,
            ),
            record(
                "gh-2.1",
                Tier::Changing,
                Some("gh-2"),
                Some("Disruption with some GitHub services"),
                5_000,
            ),
            record("gh-2.2", Tier::Changing, Some("gh-2"), None, 5_900),
        ];
        for n in 0..40 {
            let title = if n % 2 == 0 {
                "About GitHub Actions"
            } else {
                "Managing your profile"
            };
            records.push(record(
                &format!("ghdoc-{n:02}"),
                Tier::Standing,
                None,
                Some(title),
                10,
            ));
        }
        records
    }

    #[test]
    fn an_incident_is_a_thread_of_changing_records_from_its_first_to_its_last() {
        let incidents = incidents_of(&data());
        assert_eq!(
            incidents,
            vec![
                Incident {
                    thread: "gh-1".into(),
                    title: Some("Incident with Actions".into()),
                    from: 1_000,
                    to: 1_600
                },
                Incident {
                    thread: "gh-2".into(),
                    title: Some("Disruption with some GitHub services".into()),
                    from: 5_000,
                    to: 5_900
                },
            ]
        );
    }

    #[test]
    fn two_titles_are_near_when_they_share_a_product_word_and_not_a_common_one() {
        let any = ProductWords::AnyTitleWord;
        assert!(any.shared_by("Incident with Actions", "About GitHub Actions"));
        assert!(!any.shared_by("Incident with Actions", "Managing your profile"));
        // "with", "some", "services" are common words; "github" is shared and counts
        // unless a list says what a product is.
        assert!(!any.shared_by("Disruption with some services", "Working with some services"));
        assert!(any.shared_by("Disruption with some GitHub services", "About GitHub Actions"));
        let listed = ProductWords::Listed(vec!["actions".into(), "pull requests".into()]);
        assert!(listed.shared_by("Incident with Actions", "About GitHub ACTIONS"));
        assert!(!listed.shared_by("Disruption with some GitHub services", "About GitHub Actions"));
        assert!(listed.shared_by("Incident with Pull Requests", "Reviewing pull requests"));
        assert!(!listed.shared_by("Incident with Pull Requests", "Requests you can pull"));
        // A listed word matches a whole word only.
        assert!(!listed.shared_by("Transactions are slow", "Actions overview"));
    }

    #[test]
    fn the_plan_is_fixed_by_the_seed_and_asks_inside_an_incident() {
        let records = data();
        let first = plan(&records, &ProductWords::AnyTitleWord, 10, 1, 42);
        let again = plan(&records, &ProductWords::AnyTitleWord, 10, 1, 42);
        assert_eq!(first, again);
        assert_ne!(first, plan(&records, &ProductWords::AnyTitleWord, 10, 1, 43));
        assert_eq!(first.len(), 10);
        let incidents = incidents_of(&records);
        for (record, planned) in &first {
            assert_eq!(record.tier, Tier::Standing);
            let planned = planned.as_ref().unwrap();
            assert_eq!(planned.len(), 1);
            let question = &planned[0];
            assert_eq!(question.id, question_id(&record.id, 1));
            let inside = incidents
                .iter()
                .find(|i| (i.from..=i.to).contains(&question.asked_at))
                .unwrap();
            let near = ProductWords::AnyTitleWord
                .shared_by(inside.title.as_deref().unwrap(), record.title.as_deref().unwrap());
            assert_eq!(
                question.class == QueryClass::OrdinaryNear,
                near,
                "{question:?} in {inside:?}"
            );
        }
        // Both classes come up in a sample of this size.
        let classes: HashSet<QueryClass> = plan(&records, &ProductWords::AnyTitleWord, 40, 1, 42)
            .iter()
            .map(|(_, p)| p.as_ref().unwrap()[0].class)
            .collect();
        assert_eq!(classes.len(), 2);
    }

    #[test]
    fn a_larger_sample_holds_the_smaller_one_with_the_same_questions() {
        let records = data();
        let small = plan(&records, &ProductWords::AnyTitleWord, 5, 1, 42);
        let large = plan(&records, &ProductWords::AnyTitleWord, 12, 2, 42);
        for (k, (record, questions)) in small.iter().enumerate() {
            let (same_record, more) = &large[k];
            assert_eq!(record.id, same_record.id, "the same picks, in the same order");
            // The first question of a record is the same however many are asked for.
            assert_eq!(questions.as_ref().unwrap()[0], more.as_ref().unwrap()[0]);
        }
    }

    #[test]
    fn a_held_question_planned_otherwise_is_refused() {
        let records = data();
        let planned = plan(&records, &ProductWords::AnyTitleWord, 3, 1, 42);
        let first = planned[0].1.as_ref().unwrap()[0].clone();
        let held = |asked_at: i64, class: QueryClass| Query {
            id: first.id.clone(),
            text: "How do I reset my password?".into(),
            asked_at,
            class,
            gold: vec![],
            thread: None,
            harmful: vec![],
        };
        let file = Path::new("queries.jsonl");
        assert!(check_plan_against(&planned, [held(first.asked_at, first.class)].iter(), file).is_ok());
        let error =
            check_plan_against(&planned, [held(first.asked_at + 1, first.class)].iter(), file).unwrap_err();
        assert!(error.to_string().contains(&first.id), "{error}");
        let other_class = if first.class == QueryClass::Ordinary {
            QueryClass::OrdinaryNear
        } else {
            QueryClass::Ordinary
        };
        assert!(check_plan_against(&planned, [held(first.asked_at, other_class)].iter(), file).is_err());
    }

    #[test]
    fn a_question_is_never_asked_before_its_record_was_written() {
        let mut records = data();
        // Written in the middle of the second incident, after the first one ended.
        records.push(record("late-doc", Tier::Standing, None, Some("Late"), 5_500));
        for seed in 0..20 {
            let planned = plan(&records, &ProductWords::AnyTitleWord, 100, 2, seed);
            let (_, late) = planned.iter().find(|(r, _)| r.id == "late-doc").unwrap();
            for question in late.as_ref().unwrap() {
                assert!((5_500..=5_900).contains(&question.asked_at), "{question:?}");
            }
        }
        // Written after every incident: no moment to ask in, and that is said.
        records.push(record("too-late", Tier::Standing, None, None, 9_000));
        let planned = plan(&records, &ProductWords::AnyTitleWord, 100, 1, 1);
        let (_, none) = planned.iter().find(|(r, _)| r.id == "too-late").unwrap();
        assert!(none.as_ref().unwrap_err().contains("ended before"));
    }

    #[test]
    fn a_question_id_is_a_valid_id_whatever_the_record_id() {
        let long = "x".repeat(cb_types::ids::MAX_ID_LEN);
        for record_id in ["ghdoc-about-actions.3", long.as_str()] {
            let id = question_id(record_id, 2);
            assert!(cb_types::ids::check(&id).is_ok(), "{id}");
            assert!(id.ends_with("-2"));
        }
        assert_ne!(question_id("a", 1), question_id("b", 1));
        assert_eq!(question_id("a", 1), question_id("a", 1));
    }

    #[test]
    fn a_value_is_never_read_for_placeholders() {
        let filled = fill(
            "Write {{count}} for {{title}}:\n{{text}}",
            &[
                ("{{count}}", "2"),
                ("{{title}}", "A {{text}} title"),
                ("{{text}}", "Body with {{count}}."),
            ],
        );
        assert_eq!(filled, "Write 2 for A {{text}} title:\nBody with {{count}}.");
    }

    #[test]
    fn the_built_in_prompt_has_its_three_placeholders() {
        for placeholder in ["{{count}}", "{{title}}", "{{text}}"] {
            assert!(BUILT_IN_PROMPT.contains(placeholder), "{placeholder}");
        }
    }
}
