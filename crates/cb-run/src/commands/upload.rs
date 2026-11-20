//! `cb-run upload`: every record of a data set becomes one document of the knowledge
//! service, and what the service said about each is written to `<run>/uploads.jsonl`.

use super::{
    Shared, about_knowledge_service, ask_what_the_service_is, belongs_to_records, in_order, read_records,
};
use crate::error::{FailureKind, RunError};
use crate::http::millis;
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::knowledge::{Document, KnowledgeService, Label, Tenant, Timed, Uploaded, features};
use crate::retry::{Tried, with_retries};
use crate::run::{Report, Session};
use cb_types::{Record, Tier};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

pub const UPLOADS_FILE: &str = "uploads.jsonl";
const COMMAND: &str = "upload";

/// Which kind of record to upload.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TierChoice {
    Changing,
    Standing,
}

impl TierChoice {
    fn takes(self, tier: Tier) -> bool {
        matches!(
            (self, tier),
            (Self::Changing, Tier::Changing) | (Self::Standing, Tier::Standing)
        )
    }
}

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct UploadArgs {
    /// The data set directory; its records.jsonl is read.
    #[arg(long)]
    pub data: PathBuf,
    /// The tenant of the knowledge service the records are uploaded under.
    #[arg(long)]
    pub tenant: String,
    /// The collection label the documents are filed under.
    #[arg(long)]
    pub collection: String,
    /// The run directory; uploads.jsonl is appended to.
    #[arg(long)]
    pub run: PathBuf,
    /// Uploads in flight at once.
    #[arg(long, default_value_t = 4)]
    pub concurrency: usize,
    /// Upload only the records of one kind.
    #[arg(long, value_enum)]
    pub only: Option<TierChoice>,
}

/// One line of `uploads.jsonl`: what the service said about one record's document.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UploadLine {
    pub record: String,
    /// Passages the service indexed for the document (its `chunks_indexed`).
    pub passages_indexed: u64,
    /// Where the upload's time went, as the service measured it.
    pub timings_ms: BTreeMap<String, f64>,
    /// The call's wall time as this side saw it: the successful try only.
    pub wall_ms: f64,
    /// Tries the upload took.
    pub tries: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes_processed: Option<u64>,
    /// Whether the service's language model rewrote the document before indexing it.
    #[serde(default)]
    pub refined: bool,
    #[serde(default)]
    pub task_extracted: bool,
    #[serde(default)]
    pub symptom_chunks: u64,
    #[serde(default)]
    pub spoken_chunks: u64,
    /// What the service masked, by pattern name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub redactions: BTreeMap<String, u64>,
}

impl UploadLine {
    fn of(record: &Record, tried: Tried<Timed<Uploaded>>) -> Self {
        let Timed { value, wall } = tried.value;
        Self {
            record: record.id.clone(),
            passages_indexed: value.passages_indexed,
            timings_ms: value.timings_ms,
            wall_ms: millis(wall),
            tries: tried.tries,
            source_id: value.source_id,
            bytes_processed: value.bytes_processed,
            refined: value.refined,
            task_extracted: value.task_extracted,
            symptom_chunks: value.symptom_chunks,
            spoken_chunks: value.spoken_chunks,
            redactions: value.redactions,
        }
    }
}

/// For each record whose text an EARLIER record of the data set already has, byte for
/// byte: that earlier record's id.
///
/// The index keys a passage by its text (`point id = uuid5(tenant|collection|sha256(text))`,
/// `kb-domain/src/ids.rs`; finding 1 of the service's `docs/EXPERIMENT_CONTEXTBRIDGE.md`).
/// Two records with the same text would share their points, and the later upload would
/// take them from the earlier record without any error. So the later record is not
/// uploaded and is written to `failures.jsonl` instead: the hole is named, the same record
/// keeps the points on every run, and the earlier record stays searchable. Only identical
/// texts are held back here, because only for those is the collision certain; texts that
/// share a passage, or differ in white space only, are caught by the check after the
/// uploads ([`check_passages_in_index`]).
pub fn earlier_records_with_the_same_text(records: &[Record]) -> HashMap<&str, &str> {
    let mut first_with: HashMap<&str, &str> = HashMap::new();
    let mut same_as = HashMap::new();
    for record in records {
        match first_with.get(record.text.as_str()) {
            Some(first) => {
                same_as.insert(record.id.as_str(), *first);
            }
            None => {
                first_with.insert(record.text.as_str(), record.id.as_str());
            }
        }
    }
    same_as
}

/// Asks the index, for every record that has an upload line, how many of its document's
/// passages it holds, and compares that with what the upload indexed. Answers the
/// records whose passages are all there.
///
/// An upload that indexed a passage another record's document also has (the same
/// sentence in two incident updates, say) takes that point over: from then on it carries
/// the other document's path and dates, and the first record has lost it without any
/// error. The service's own advice is to check every record after loading
/// (`GET /v1/tenants/{tenant}/sources/count`). A record whose count differs is a failure
/// on every run until the data set is fixed, so the run can never pass for complete.
pub async fn check_passages_in_index<K: KnowledgeService>(
    kb: &K,
    (tenant, label): (&Tenant, &Label),
    uploaded: &BTreeMap<String, u64>,
    retry: &crate::retry::RetryPolicy,
    concurrency: usize,
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    let mut whole = HashSet::new();
    in_order(
        uploaded.iter(),
        concurrency,
        |(record, indexed)| async move {
            let path = cb_types::ids::source_path(record);
            let counted = with_retries(retry, |_| kb.count_passages(tenant, &path, label)).await;
            (record, *indexed, counted)
        },
        |(record, indexed, counted)| {
            session.add("checked_in_index", 1);
            match counted {
                Ok(held) if held.value == indexed => {
                    whole.insert(record.clone());
                }
                Ok(held) => {
                    let error = RunError::Contract {
                        id: record.clone(),
                        url: kb.address(),
                        reason: format!(
                            "its upload indexed {indexed} passage(s) and the index now holds {} for its document; \
                             another document with a passage of the same text has most likely taken them over",
                            held.value
                        ),
                    };
                    session.fail_as(record, FailureKind::NotWholeInIndex, &error, held.tries)?;
                }
                Err(gave_up) => session.fail(record, &gave_up.error, gave_up.tries)?,
            }
            Ok(())
        },
    )
    .await?;
    Ok(whole)
}

/// Uploads the records of `args.data` that `uploads.jsonl` does not hold yet.
pub async fn upload<K: KnowledgeService>(
    kb: &K,
    args: &UploadArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let tenant = Tenant::parse(&args.tenant)?;
    let label = Label::parse("--collection", &args.collection)?;
    let records = read_records(&args.data)?;
    let mut session = Session::begin(&args.run, COMMAND, args, about_knowledge_service(kb, shared))?;
    match work(kb, args, shared, (&tenant, &label), &records, &mut session).await {
        // Done: uploaded, and every passage of it still in the index.
        Ok(done) => session.finish(|id| done.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn work<K: KnowledgeService>(
    kb: &K,
    args: &UploadArgs,
    shared: &Shared,
    (tenant, label): (&Tenant, &Label),
    records: &[Record],
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    session.claim(
        UPLOADS_FILE,
        belongs_to_records(kb, &args.data, tenant.as_str(), label.as_str())?,
    )?;
    ask_what_the_service_is(
        kb,
        shared,
        session,
        &[features::DOCUMENT_TIMES, features::INGEST_TIMINGS],
    )
    .await?;
    let path = session.file(UPLOADS_FILE);
    let existing = read_resumable::<UploadLine>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    // Every record with an upload line, and the passages its upload indexed.
    let mut uploaded: BTreeMap<String, u64> = BTreeMap::new();
    for line in existing.lines {
        if uploaded
            .insert(line.record.clone(), line.passages_indexed)
            .is_some()
        {
            return Err(RunError::file(
                &path,
                format!("record {} has two upload lines", line.record),
            ));
        }
    }
    let mut out = Appender::open(&path)?;

    let same_text_as = earlier_records_with_the_same_text(records);
    let mut todo = Vec::new();
    for record in records
        .iter()
        .filter(|r| args.only.is_none_or(|only| only.takes(r.tier)))
    {
        session.add("selected", 1);
        if uploaded.contains_key(&record.id) {
            session.add("already_there", 1);
        } else if let Err(e) = cb_types::ids::check(&record.id) {
            session.fail(&record.id, &RunError::item(&record.id, e.to_string()), 0)?;
        } else if let Some(first) = same_text_as.get(record.id.as_str()) {
            let error = RunError::item(
                &record.id,
                format!(
                    "the same text as record {first}: the index keys a passage by its text, \
                     so the two cannot both be in it; not uploaded"
                ),
            );
            session.fail_as(&record.id, FailureKind::DuplicateText, &error, 0)?;
        } else {
            todo.push(record);
        }
    }

    let retry = &shared.retry;
    in_order(
        todo,
        args.concurrency,
        |record| async move {
            let document = Document::of(record, label);
            let outcome = with_retries(retry, |_| kb.upload(tenant, &document)).await;
            (record, outcome)
        },
        |(record, outcome)| {
            match outcome {
                Ok(tried) => {
                    let line = UploadLine::of(record, tried);
                    out.append(std::slice::from_ref(&line))?;
                    uploaded.insert(line.record, line.passages_indexed);
                    session.wrote();
                }
                Err(gave_up) => session.fail(&record.id, &gave_up.error, gave_up.tries)?,
            }
            Ok(())
        },
    )
    .await?;
    out.sync()?;
    // Every record uploaded so far, on this run or before, whatever `--only` selected:
    // a later upload can take passages from any of them.
    check_passages_in_index(kb, (tenant, label), &uploaded, retry, args.concurrency, session).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(id: &str, text: &str) -> Record {
        Record {
            id: id.into(),
            text: text.into(),
            written_at: 1,
            valid_until: None,
            thread: None,
            tier: Tier::Standing,
            title: None,
            source: "test".into(),
            truth: None,
        }
    }

    #[test]
    fn a_later_record_with_an_earlier_records_text_is_found() {
        let records = [
            record("a", "We are investigating."),
            record("b", "The cause is a cut fibre."),
            record("c", "We  are\ninvestigating. "),
            record("d", "We are investigating."),
            record("e", "We are investigating."),
        ];
        let same = earlier_records_with_the_same_text(&records);
        assert_eq!(same.len(), 2);
        // The first in file order keeps the text, on every run.
        assert_eq!(same.get("d"), Some(&"a"));
        assert_eq!(same.get("e"), Some(&"a"));
        // White space differs: not held back on a guess; the check after the uploads
        // finds it if the index makes one point of the two.
        assert_eq!(same.get("c"), None);
    }

    #[test]
    fn only_takes_the_named_kind() {
        assert!(TierChoice::Changing.takes(Tier::Changing));
        assert!(!TierChoice::Changing.takes(Tier::Standing));
        assert!(TierChoice::Standing.takes(Tier::Standing));
    }
}
