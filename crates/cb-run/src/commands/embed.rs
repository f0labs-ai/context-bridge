//! `cb-run embed`: texts embedded through the knowledge service's `/v1/embed`, written as
//! `<run>/<name>.f32` + `<run>/<name>.json`.

use super::{Shared, about_knowledge_service, ask_what_the_service_is, belonging};
use crate::error::RunError;
use crate::jsonl::read_all;
use crate::knowledge::KnowledgeService;
use crate::retry::with_retries;
use crate::run::{Report, Session};
use crate::vector_file::VectorTable;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

const COMMAND: &str = "embed";

/// Texts per request (docs/RUN_TOOLS.md). The service takes at most 128.
pub const BATCH: usize = 64;
/// The vector file is written anew after this many batches, so a crash loses little.
const BATCHES_PER_CHECKPOINT: usize = 16;

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct EmbedArgs {
    /// A JSON Lines file of {"id", "text"}.
    #[arg(long)]
    pub texts: PathBuf,
    /// The run directory the vector file is written into.
    #[arg(long)]
    pub run: PathBuf,
    /// The vector file's name: <name>.f32 and <name>.json.
    #[arg(long)]
    pub name: String,
}

/// One text to embed.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct TextLine {
    pub id: String,
    pub text: String,
}

/// A name that stays inside the run directory: letters, digits, `-`, `_` and `.`,
/// not starting with a dot.
fn check_name(name: &str) -> Result<(), RunError> {
    let ok = !name.is_empty()
        && !name.starts_with('.')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if ok {
        Ok(())
    } else {
        Err(RunError::Setting {
            name: "--name",
            reason: "a file name of letters, digits, -, _ and . (no path)".into(),
        })
    }
}

/// Embeds the texts the vector file does not hold yet.
pub async fn embed<K: KnowledgeService>(
    kb: &K,
    args: &EmbedArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    check_name(&args.name)?;
    let texts: Vec<TextLine> = read_all(&args.texts)?;
    let mut seen = HashSet::new();
    if let Some(twice) = texts.iter().find(|t| !seen.insert(t.id.as_str())) {
        return Err(RunError::file(
            &args.texts,
            format!("id {:?} appears twice", twice.id),
        ));
    }
    let mut session = Session::begin(&args.run, COMMAND, args, about_knowledge_service(kb, shared))?;
    match work(kb, args, shared, &texts, &mut session).await {
        Ok(table) => session.finish(|id| table.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn work<K: KnowledgeService>(
    kb: &K,
    args: &EmbedArgs,
    shared: &Shared,
    texts: &[TextLine],
    session: &mut Session,
) -> Result<VectorTable, RunError> {
    session.claim(
        &format!("{}.json", args.name),
        belonging(&[("knowledge_service", &kb.address())]),
    )?;
    // `/v1/embed` is older than the experiment branch: no feature to ask for.
    ask_what_the_service_is(kb, shared, session, &[]).await?;
    let mut table = VectorTable::read(session.dir(), &args.name)?.unwrap_or_default();
    let todo: Vec<&TextLine> = texts.iter().filter(|t| !table.contains(&t.id)).collect();
    session.set("already_there", (texts.len() - todo.len()) as u64);

    let mut models: Vec<String> = Vec::new();
    for (n, batch) in todo.chunks(BATCH).enumerate() {
        let sent: Vec<String> = batch.iter().map(|t| t.text.clone()).collect();
        match with_retries(&shared.retry, |_| kb.embed(&sent)).await {
            Ok(embedded) => {
                note_model(&mut models, &embedded.value.value.model);
                for (line, vector) in batch.iter().zip(&embedded.value.value.vectors) {
                    keep(&mut table, session, line, vector)?;
                }
            }
            // The service refused the batch (a 4xx is about what was sent). One text it
            // will not take must not cost the other 63 their vectors: each goes alone.
            Err(gave_up) if !gave_up.error.is_retryable() && batch.len() > 1 => {
                session.add("batches_sent_one_by_one", 1);
                for line in batch {
                    match with_retries(&shared.retry, |_| kb.embed(std::slice::from_ref(&line.text))).await {
                        Ok(embedded) => {
                            note_model(&mut models, &embedded.value.value.model);
                            match embedded.value.value.vectors.first() {
                                Some(vector) => keep(&mut table, session, line, vector)?,
                                None => session.fail(
                                    &line.id,
                                    &RunError::item(&line.id, "no vector came back"),
                                    1,
                                )?,
                            }
                        }
                        Err(alone) => session.fail(&line.id, &alone.error, alone.tries)?,
                    }
                }
            }
            Err(gave_up) => {
                for line in batch {
                    session.fail(&line.id, &gave_up.error, gave_up.tries)?;
                }
            }
        }
        if (n + 1) % BATCHES_PER_CHECKPOINT == 0 {
            table.write(session.dir(), &args.name)?;
        }
    }
    table.write(session.dir(), &args.name)?;
    session.set("rows", table.len() as u64);
    session.set("dim", table.dim() as u64);
    if let Some(record) = session.knowledge_service_mut() {
        record.embedding_model = (!models.is_empty()).then(|| models.join(", "));
    }
    Ok(table)
}

fn note_model(models: &mut Vec<String>, model: &str) {
    if !models.iter().any(|m| m == model) {
        models.push(model.to_string());
    }
}

/// Puts one text's vector in the table, or records why it cannot be there (a vector of
/// another width than the rest).
fn keep(
    table: &mut VectorTable,
    session: &mut Session,
    line: &TextLine,
    vector: &[f32],
) -> Result<(), RunError> {
    match table.push(&line.id, vector) {
        Ok(()) => {
            session.wrote();
            Ok(())
        }
        Err(error) => session.fail(&line.id, &error, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_cannot_leave_the_run_directory() {
        for good in ["extractions", "extractions.asr05", "q_vectors-2"] {
            assert!(check_name(good).is_ok(), "{good}");
        }
        for bad in ["", "../x", "a/b", ".hidden", "a b", "a\\b"] {
            assert!(check_name(bad).is_err(), "{bad:?}");
        }
    }
}
