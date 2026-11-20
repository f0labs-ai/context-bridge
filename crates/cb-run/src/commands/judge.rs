//! `cb-run judge`: a language model's verdict on each extraction of the gate's judge
//! sample (`<run>/gate/judge_sample.jsonl`, drawn by `cb-eval gate`), written to
//! `<run>/judge.jsonl` one `{"transcript", "correct"}` per line (DESIGN.md section 6).
//!
//! The judge is shown the statements the call conveys that are true, the incident's
//! wrong statements, and the extracted record, and says whether the record states a
//! true one and none of the wrong ones. It is not shown the key terms the gate's own
//! rule matches, so the two verdicts are made apart and their agreement means something.

use super::{Prompt, Shared, belonging, extra_body_identity, in_order, read_records, read_transcripts};
use crate::atomic::sha256_of_file;
use crate::error::RunError;
use crate::http::millis;
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::model::{Ask, ExtraBody, ModelAnswer, ModelEndpoint};
use crate::retry::with_retries;
use crate::run::{About, ModelRecord, Report, Session};
use crate::text::{JUDGE_PLACEHOLDERS, judge_message, read_judgement};
use cb_types::{Record, Transcript};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

pub const JUDGE_FILE: &str = "judge.jsonl";
const COMMAND: &str = "judge";

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct JudgeArgs {
    /// The judge sample: one {"transcript", "problem", "solution"} per line.
    #[arg(long)]
    pub sample: PathBuf,
    /// The data set directory; its records.jsonl and transcripts.jsonl are read.
    #[arg(long)]
    pub data: PathBuf,
    /// The transcripts the extractions were made from, in place of <data>/transcripts.jsonl.
    #[arg(long)]
    pub transcripts: Option<PathBuf>,
    /// The judge model's name, as the endpoint knows it.
    #[arg(long)]
    pub model: String,
    /// The prompt file; its name without the extension is the prompt's version.
    #[arg(long)]
    pub prompt: PathBuf,
    /// The run directory; judge.jsonl is appended to.
    #[arg(long)]
    pub run: PathBuf,
    /// Calls in flight at once.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// A bound on the answer's length, sent as max_tokens. Not sent when absent.
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// A JSON object whose fields are added at the top level of every request body
    /// (e.g. '{"chat_template_kwargs":{"enable_thinking":false}}'). No secrets.
    #[arg(long)]
    pub extra_body: Option<ExtraBody>,
}

/// One extraction of the judge sample, or one promoted group (`calls` then names its
/// members' calls and `transcript` is the group's sample id).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct SampleItem {
    pub transcript: String,
    pub problem: String,
    pub solution: String,
    #[serde(default)]
    pub calls: Vec<String>,
}

/// One line of `judge.jsonl`: what `cb-eval gate --judge` reads (`transcript`, `correct`)
/// and the call's own facts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeLine {
    pub transcript: String,
    pub correct: bool,
    pub model: String,
    pub prompt_version: String,
    pub latency_ms: f64,
    pub tokens_in: u32,
    pub tokens_out: u32,
}

/// What the judge is told about one call: the texts of the true records it states, and
/// of every wrong statement planted in its incident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statements<'a> {
    pub true_ones: Vec<&'a str>,
    pub wrong_ones: Vec<&'a str>,
}

/// The planted truth of a data set, read for the judge.
pub struct Truth<'a> {
    by_id: HashMap<&'a str, &'a Record>,
    wrong_by_thread: HashMap<&'a str, Vec<&'a str>>,
}

impl<'a> Truth<'a> {
    pub fn new(records: &'a [Record]) -> Self {
        let mut by_id = HashMap::with_capacity(records.len());
        let mut wrong_by_thread: HashMap<&str, Vec<&str>> = HashMap::new();
        for record in records {
            by_id.insert(record.id.as_str(), record);
            if let (Some(thread), Some(truth)) = (record.thread.as_deref(), record.truth.as_ref())
                && !truth.correct
            {
                wrong_by_thread
                    .entry(thread)
                    .or_default()
                    .push(record.text.as_str());
            }
        }
        Self {
            by_id,
            wrong_by_thread,
        }
    }

    /// The call's statements, or the id of a stated record the data set does not hold.
    pub fn of(&self, call: &Transcript) -> Result<Statements<'a>, String> {
        let mut true_ones = Vec::new();
        for id in &call.states {
            let record = self
                .by_id
                .get(id.as_str())
                .ok_or_else(|| format!("the call states record {id}, which the data set does not hold"))?;
            if record.truth.as_ref().is_some_and(|t| t.correct) {
                true_ones.push(record.text.as_str());
            }
        }
        let wrong_ones = self
            .wrong_by_thread
            .get(call.thread.as_str())
            .cloned()
            .unwrap_or_default();
        Ok(Statements {
            true_ones,
            wrong_ones,
        })
    }

    /// The statements of several calls together, each statement once in the order first
    /// met: what a group merged from those calls is judged against.
    pub fn of_calls(&self, calls: &[&Transcript]) -> Result<Statements<'a>, String> {
        let mut together = Statements {
            true_ones: Vec::new(),
            wrong_ones: Vec::new(),
        };
        for call in calls {
            let one = self.of(call)?;
            for text in one.true_ones {
                if !together.true_ones.contains(&text) {
                    together.true_ones.push(text);
                }
            }
            for text in one.wrong_ones {
                if !together.wrong_ones.contains(&text) {
                    together.wrong_ones.push(text);
                }
            }
        }
        Ok(together)
    }
}

/// One try for one extraction: the call, then the answer read strictly.
async fn ask_for_verdict<M: ModelEndpoint>(
    model: &M,
    args: &JudgeArgs,
    transcript_id: &str,
    message: &str,
) -> Result<(bool, ModelAnswer), RunError> {
    let answer = model
        .ask(&Ask {
            model: &args.model,
            message,
            max_tokens: args.max_tokens,
            extra: args.extra_body.as_ref(),
        })
        .await?;
    match read_judgement(&answer.content) {
        Ok(correct) => Ok((correct, answer)),
        Err(reason) => Err(RunError::AnswerNotAsked {
            id: transcript_id.to_string(),
            url: model.address(),
            reason,
            raw: answer.content,
        }),
    }
}

/// Judges the sample's extractions that `judge.jsonl` does not hold yet.
pub async fn judge<M: ModelEndpoint>(
    model: &M,
    args: &JudgeArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let prompt = Prompt::read(&args.prompt)?;
    if let Some(missing) = JUDGE_PLACEHOLDERS
        .iter()
        .find(|p| prompt.text.matches(*p).count() != 1)
    {
        return Err(RunError::file(
            &args.prompt,
            format!("the prompt must hold {missing} exactly once"),
        ));
    }
    let sample: Vec<SampleItem> = crate::jsonl::read_all(&args.sample)?;
    let records = read_records(&args.data)?;
    let transcripts_path = args
        .transcripts
        .clone()
        .unwrap_or_else(|| args.data.join("transcripts.jsonl"));
    let transcripts = read_transcripts(&transcripts_path)?;
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
    let mut session = Session::begin(&args.run, COMMAND, args, about)?;
    let inputs = (&sample[..], &records[..], &transcripts[..], &transcripts_path);
    match work(model, args, shared, &prompt, inputs, &mut session).await {
        Ok(done) => session.finish(|id| done.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

type Inputs<'a> = (&'a [SampleItem], &'a [Record], &'a [Transcript], &'a PathBuf);

async fn work<M: ModelEndpoint>(
    model: &M,
    args: &JudgeArgs,
    shared: &Shared,
    prompt: &Prompt,
    (sample, records, transcripts, transcripts_path): Inputs<'_>,
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    session.claim(
        JUDGE_FILE,
        belonging(&[
            ("model_endpoint", &model.address()),
            ("model", &args.model),
            ("prompt_sha256", &prompt.sha256),
            ("sample_sha256", &sha256_of_file(&args.sample)?),
            (
                "records_sha256",
                &sha256_of_file(&args.data.join("records.jsonl"))?,
            ),
            ("transcripts_sha256", &sha256_of_file(transcripts_path)?),
            ("extra_body", &extra_body_identity(args.extra_body.as_ref())),
        ]),
    )?;
    let path = session.file(JUDGE_FILE);
    let existing = read_resumable::<JudgeLine>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let mut done: HashSet<String> = existing.lines.into_iter().map(|line| line.transcript).collect();
    let mut out = Appender::open(&path)?;
    let truth = Truth::new(records);
    let calls: HashMap<&str, &Transcript> = transcripts.iter().map(|t| (t.id.as_str(), t)).collect();

    let mut todo: Vec<(&SampleItem, String)> = Vec::new();
    for item in sample {
        session.add("selected", 1);
        if done.contains(&item.transcript) {
            session.add("already_there", 1);
            continue;
        }
        // One extraction is judged against its own call; a group against all its calls.
        let ids: Vec<&str> = if item.calls.is_empty() {
            vec![item.transcript.as_str()]
        } else {
            item.calls.iter().map(String::as_str).collect()
        };
        let statements = ids
            .iter()
            .map(|id| {
                calls
                    .get(id)
                    .copied()
                    .ok_or_else(|| format!("the transcripts file holds no call {id}"))
            })
            .collect::<Result<Vec<&Transcript>, String>>()
            .and_then(|found| truth.of_calls(&found));
        match statements {
            Ok(s) => todo.push((
                item,
                judge_message(
                    &prompt.text,
                    &s.true_ones,
                    &s.wrong_ones,
                    &item.problem,
                    &item.solution,
                ),
            )),
            Err(reason) => session.fail(&item.transcript, &RunError::item(&item.transcript, reason), 0)?,
        }
    }

    let retry = &shared.retry;
    let mut reported: BTreeSet<String> = BTreeSet::new();
    in_order(
        todo,
        args.concurrency,
        |(item, message)| async move {
            let outcome = with_retries(retry, |_| {
                ask_for_verdict(model, args, &item.transcript, &message)
            })
            .await;
            (item, outcome)
        },
        |(item, outcome)| {
            match outcome {
                Ok(tried) => {
                    let (correct, answer) = tried.value;
                    if answer.tokens_in.is_none() || answer.tokens_out.is_none() {
                        session.add("answers_without_token_counts", 1);
                    }
                    if let Some(name) = &answer.reported_model {
                        reported.insert(name.clone());
                    }
                    session.add(
                        if correct {
                            "judged_correct"
                        } else {
                            "judged_not_correct"
                        },
                        1,
                    );
                    out.append(&[JudgeLine {
                        transcript: item.transcript.clone(),
                        correct,
                        model: args.model.clone(),
                        prompt_version: prompt.version.clone(),
                        latency_ms: millis(answer.wall),
                        tokens_in: answer.tokens_in.unwrap_or(0),
                        tokens_out: answer.tokens_out.unwrap_or(0),
                    }])?;
                    done.insert(item.transcript.clone());
                    session.wrote();
                }
                Err(gave_up) => session.fail(&item.transcript, &gave_up.error, gave_up.tries)?,
            }
            Ok(())
        },
    )
    .await?;
    out.sync()?;
    if let Some(record) = session.model_mut() {
        record.reported_names = reported.into_iter().collect();
        record.json_mode = Some(model.json_mode());
    }
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::{Phase, RecordTruth, Tier};

    fn record(id: &str, text: &str, thread: &str, correct: bool) -> Record {
        Record {
            id: id.into(),
            text: text.into(),
            written_at: 1,
            valid_until: None,
            thread: Some(thread.into()),
            tier: Tier::Changing,
            title: None,
            source: "test".into(),
            truth: Some(RecordTruth {
                correct,
                phase: Phase::Investigating,
                key_terms: vec![],
                superseded_at: None,
            }),
        }
    }

    fn call(id: &str, thread: &str, states: &[&str]) -> Transcript {
        Transcript {
            id: id.into(),
            thread: thread.into(),
            started_at: 1,
            agent: "a".into(),
            persona: "p".into(),
            turns: vec![],
            states: states.iter().map(|s| s.to_string()).collect(),
            asr_error_rate: 0.0,
        }
    }

    #[test]
    fn the_judge_is_told_the_calls_true_statements_and_its_incidents_wrong_ones() {
        let records = vec![
            record("inc-1.s1", "A fibre cut in Northgate.", "inc-1", true),
            record("inc-1.s2", "Fixed at 14:30.", "inc-1", true),
            record("inc-1.d1", "It is your own router.", "inc-1", false),
            record("inc-2.d1", "Another incident's guess.", "inc-2", false),
        ];
        let truth = Truth::new(&records);
        // The call states s1 and the wrong d1: only s1 is a true statement; every wrong
        // statement of its incident is listed, whether this call says it or not.
        let told = truth.of(&call("c1", "inc-1", &["inc-1.s1", "inc-1.d1"])).unwrap();
        assert_eq!(told.true_ones, ["A fibre cut in Northgate."]);
        assert_eq!(told.wrong_ones, ["It is your own router."]);
        // A call that states nothing true: no true statement at all.
        assert!(truth.of(&call("c2", "inc-1", &[])).unwrap().true_ones.is_empty());
        // A stated record the data set lacks is named, not skipped.
        let error = truth.of(&call("c3", "inc-1", &["inc-1.s9"])).unwrap_err();
        assert!(error.contains("inc-1.s9"), "{error}");
        // A group of calls: their true statements together, each once.
        let (a, b) = (
            call("c4", "inc-1", &["inc-1.s1"]),
            call("c5", "inc-1", &["inc-1.s1", "inc-1.s2"]),
        );
        let group = truth.of_calls(&[&a, &b]).unwrap();
        assert_eq!(group.true_ones, ["A fibre cut in Northgate.", "Fixed at 14:30."]);
        assert_eq!(group.wrong_ones, ["It is your own router."]);
    }

    #[test]
    fn the_built_in_prompt_holds_every_placeholder_once() {
        let text = include_str!("../../../../prompts/judge.v1.md");
        for placeholder in JUDGE_PLACEHOLDERS {
            assert_eq!(text.matches(placeholder).count(), 1, "{placeholder}");
        }
    }
}
