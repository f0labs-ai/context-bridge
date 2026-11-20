//! `cb-run extract`: one model call per transcript, its answer read strictly and written
//! to `<run>/extractions.jsonl`.

use super::{Prompt, Shared, belonging, extra_body_identity, in_order, read_transcripts};
use crate::atomic::sha256_of_file;
use crate::error::RunError;
use crate::http::millis;
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::model::{Ask, ExtraBody, ModelAnswer, ModelEndpoint};
use crate::retry::{GaveUp, with_retries};
use crate::run::{About, ModelRecord, Report, Session};
use crate::text::{Extracted, extraction_message, read_extraction};
use cb_types::{Extraction, Transcript, Turn};
use serde::Serialize;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

pub const EXTRACTIONS_FILE: &str = "extractions.jsonl";
const COMMAND: &str = "extract";

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct ExtractArgs {
    /// A JSON Lines file of transcripts.
    #[arg(long)]
    pub transcripts: PathBuf,
    /// The model's name, as the endpoint knows it.
    #[arg(long)]
    pub model: String,
    /// The prompt file; its name without the extension is the prompt's version.
    #[arg(long)]
    pub prompt: PathBuf,
    /// The run directory; extractions.jsonl is appended to.
    #[arg(long)]
    pub run: PathBuf,
    /// Calls in flight at once.
    #[arg(long, default_value_t = 8)]
    pub concurrency: usize,
    /// Read only the first N transcripts of the file.
    #[arg(long)]
    pub limit: Option<usize>,
    /// A file of the same transcripts (same ids, same turns) whose text is already
    /// masked; the model is given that text.
    #[arg(long)]
    pub masked_with: Option<PathBuf>,
    /// A bound on the answer's length, sent as max_tokens. Not sent when absent.
    #[arg(long)]
    pub max_tokens: Option<u32>,
    /// A JSON object whose fields are added at the top level of every request body, for
    /// an endpoint that needs them (e.g. '{"chat_template_kwargs":{"enable_thinking":false}}').
    /// Fields cb-run sets itself are refused. Recorded in the run manifest: no secrets.
    #[arg(long)]
    pub extra_body: Option<ExtraBody>,
}

/// Whether two renderings of a call have the same turns by the same speakers.
fn same_speakers(a: &[Turn], b: &[Turn]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.who == y.who)
}

/// One try for one transcript: the call, then the answer read strictly. An answer that
/// is not the asked JSON is an error that keeps the answer whole, and counts as a try.
async fn ask_for_extraction<M: ModelEndpoint>(
    model: &M,
    args: &ExtractArgs,
    transcript_id: &str,
    message: &str,
) -> Result<(Extracted, ModelAnswer), RunError> {
    let answer = model
        .ask(&Ask {
            model: &args.model,
            message,
            max_tokens: args.max_tokens,
            extra: args.extra_body.as_ref(),
        })
        .await?;
    match read_extraction(&answer.content) {
        Ok(extracted) => Ok((extracted, answer)),
        Err(reason) => Err(RunError::AnswerNotAsked {
            id: transcript_id.to_string(),
            url: model.address(),
            reason,
            raw: answer.content,
        }),
    }
}

/// Reads the transcripts `extractions.jsonl` does not hold yet.
pub async fn extract<M: ModelEndpoint>(
    model: &M,
    args: &ExtractArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let prompt = Prompt::read(&args.prompt)?;
    let transcripts = read_transcripts(&args.transcripts)?;
    let masked = args.masked_with.as_deref().map(read_transcripts).transpose()?;
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
    match work(
        model,
        args,
        shared,
        &prompt,
        (&transcripts, masked.as_deref()),
        &mut session,
    )
    .await
    {
        Ok(done) => session.finish(|id| done.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn work<M: ModelEndpoint>(
    model: &M,
    args: &ExtractArgs,
    shared: &Shared,
    prompt: &Prompt,
    (transcripts, masked): (&[Transcript], Option<&[Transcript]>),
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    let masked_sha256 = match &args.masked_with {
        Some(masked_file) => sha256_of_file(masked_file)?,
        None => "none".to_string(),
    };
    session.claim(
        EXTRACTIONS_FILE,
        belonging(&[
            ("model_endpoint", &model.address()),
            ("model", &args.model),
            ("prompt_sha256", &prompt.sha256),
            ("transcripts_sha256", &sha256_of_file(&args.transcripts)?),
            ("masked_sha256", &masked_sha256),
            ("extra_body", &extra_body_identity(args.extra_body.as_ref())),
        ]),
    )?;
    let path = session.file(EXTRACTIONS_FILE);
    let existing = read_resumable::<Extraction>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let mut done: HashSet<String> = existing.lines.into_iter().map(|line| line.transcript).collect();
    let mut out = Appender::open(&path)?;
    let masked_turns: Option<HashMap<&str, &[Turn]>> =
        masked.map(|all| all.iter().map(|t| (t.id.as_str(), t.turns.as_slice())).collect());

    // `--limit N` is the first N transcripts of the file, whatever is done already.
    let mut todo: Vec<(&Transcript, &[Turn])> = Vec::new();
    for transcript in transcripts.iter().take(args.limit.unwrap_or(usize::MAX)) {
        session.add("selected", 1);
        if done.contains(&transcript.id) {
            session.add("already_there", 1);
            continue;
        }
        let turns = match &masked_turns {
            None => transcript.turns.as_slice(),
            // Masked text was asked for: the unmasked text is never sent in its place.
            Some(by_id) => match by_id.get(transcript.id.as_str()) {
                Some(&turns) if same_speakers(turns, &transcript.turns) => turns,
                Some(_) => {
                    let error = RunError::item(
                        &transcript.id,
                        "the masked transcript has other turns or speakers than the original",
                    );
                    session.fail(&transcript.id, &error, 0)?;
                    continue;
                }
                None => {
                    let error =
                        RunError::item(&transcript.id, "the masked file has no transcript with this id");
                    session.fail(&transcript.id, &error, 0)?;
                    continue;
                }
            },
        };
        todo.push((transcript, turns));
    }

    let retry = &shared.retry;
    let mut reported: BTreeSet<String> = BTreeSet::new();
    in_order(
        todo,
        args.concurrency,
        |(transcript, turns)| async move {
            let outcome = match extraction_message(&prompt.text, transcript.started_at, turns) {
                Some(message) => {
                    with_retries(retry, |_| {
                        ask_for_extraction(model, args, &transcript.id, &message)
                    })
                    .await
                }
                None => Err(GaveUp {
                    error: RunError::item(
                        &transcript.id,
                        "started_at is not a moment that can be written as a date",
                    ),
                    tries: 0,
                }),
            };
            (transcript, outcome)
        },
        |(transcript, outcome)| {
            match outcome {
                Ok(tried) => {
                    let (extracted, answer) = tried.value;
                    if extracted.problem.is_empty() {
                        session.add("nothing_found", 1);
                    }
                    // The endpoint's `usage`, as reported; zero when it reported none, and
                    // such answers counted, so a zero is never taken for a measurement.
                    if answer.tokens_in.is_none() || answer.tokens_out.is_none() {
                        session.add("answers_without_token_counts", 1);
                    }
                    session.add("tokens_in", u64::from(answer.tokens_in.unwrap_or(0)));
                    session.add("tokens_out", u64::from(answer.tokens_out.unwrap_or(0)));
                    if let Some(name) = &answer.reported_model {
                        reported.insert(name.clone());
                    }
                    out.append(&[Extraction {
                        transcript: transcript.id.clone(),
                        problem: extracted.problem,
                        solution: extracted.solution,
                        t_start: extracted.t_start,
                        t_eta: extracted.t_eta,
                        times_unread: extracted.times_unread,
                        confidence: extracted.confidence,
                        model: args.model.clone(),
                        prompt_version: prompt.version.clone(),
                        // The wall time of the call that gave this answer, and of that call only.
                        latency_ms: millis(answer.wall),
                        tokens_in: answer.tokens_in.unwrap_or(0),
                        tokens_out: answer.tokens_out.unwrap_or(0),
                    }])?;
                    done.insert(transcript.id.clone());
                    session.wrote();
                }
                Err(gave_up) => session.fail(&transcript.id, &gave_up.error, gave_up.tries)?,
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
    use cb_types::Speaker;

    fn turn(who: Speaker, text: &str) -> Turn {
        Turn {
            who,
            text: text.into(),
            ends_at_s: 0.0,
        }
    }

    #[test]
    fn masked_turns_must_line_up_with_the_original() {
        let original = [
            turn(Speaker::Customer, "I am Ada Lovelace."),
            turn(Speaker::Agent, "Thanks."),
        ];
        let masked = [
            turn(Speaker::Customer, "I am [NAME]."),
            turn(Speaker::Agent, "Thanks."),
        ];
        assert!(same_speakers(&original, &masked));
        assert!(!same_speakers(&original, &masked[..1]));
        let swapped = [
            turn(Speaker::Agent, "I am [NAME]."),
            turn(Speaker::Agent, "Thanks."),
        ];
        assert!(!same_speakers(&original, &swapped));
    }
}
