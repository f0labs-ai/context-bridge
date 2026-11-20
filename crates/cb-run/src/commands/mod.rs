//! The commands of docs/RUN_TOOLS.md, one module each, and what they share.
//!
//! Every command has the same shape: read the inputs, see what the run files already
//! hold, do the rest with bounded retries, write each result or failure the moment it is
//! known, and end by settling `failures.jsonl` and the manifest ([`crate::run::Session`]).

pub mod candidates;
pub mod control;
pub mod embed;
pub mod extract;
pub mod judge;
pub mod newest;
pub mod questions;
pub mod trips;
pub mod upload;
pub mod vectors;

use crate::error::RunError;
use crate::index::{IndexReader, Page, Point, PointId, Want};
use crate::jsonl::read_all;
use crate::knowledge::{KnowledgeService, ServiceInfo, missing_feature};
use crate::retry::{GaveUp, RetryPolicy, with_retries};
use crate::run::{About, ServiceRecord, Session};
use cb_types::{Query, Record, Transcript};
use futures::stream::{self, StreamExt};
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, HashSet};
use std::future::Future;
use std::path::Path;

pub const RECORDS_FILE: &str = "records.jsonl";
pub const QUERIES_FILE: &str = "queries.jsonl";
/// The manifest of a DATA SET (`cb_types::Manifest`), not of a run.
pub const DATA_MANIFEST_FILE: &str = "manifest.json";

/// What every command is given besides its own flags and its services.
#[derive(Clone, Debug)]
pub struct Shared {
    pub retry: RetryPolicy,
    /// What the operator calls the machine, for the manifest.
    pub machine_label: Option<String>,
}

/// A prompt: its text, its version (the file's name without the extension, `extract.v1`)
/// and the sha256 of its text, so a manifest says exactly which words a model was given.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub text: String,
    pub version: String,
    pub sha256: String,
}

impl Prompt {
    pub fn read(path: &Path) -> Result<Self, RunError> {
        let text = std::fs::read_to_string(path).map_err(|e| RunError::io(path, e))?;
        let version = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .filter(|stem| !stem.is_empty())
            .ok_or_else(|| RunError::file(path, "a prompt file has a name"))?;
        Ok(Self::of(version, text))
    }

    pub fn of(version: impl Into<String>, text: impl Into<String>) -> Self {
        let text = text.into();
        Self {
            sha256: crate::atomic::sha256_of_bytes(text.as_bytes()),
            version: version.into(),
            text,
        }
    }
}

/// Does `work` for every item, at most `concurrency` at a time, and hands each result to
/// `keep` in the items' own order. So a run without failures writes its file in input
/// order whatever the concurrency; what a crash loses is at most the few results that
/// were finished but not yet handed over, and a resume does those again.
pub async fn in_order<I, O, W, Fut>(
    items: impl IntoIterator<Item = I>,
    concurrency: usize,
    work: W,
    mut keep: impl FnMut(O) -> Result<(), RunError>,
) -> Result<(), RunError>
where
    W: FnMut(I) -> Fut,
    Fut: Future<Output = O>,
{
    let mut results = std::pin::pin!(stream::iter(items).map(work).buffered(concurrency.max(1)));
    while let Some(result) = results.next().await {
        keep(result)?;
    }
    Ok(())
}

/// How an extra body counts in what a run file belongs to: its fields as one line of
/// JSON, or `none`. Answers asked with other fields (thinking on, thinking off) are
/// answers to another question.
pub fn extra_body_identity(extra: Option<&crate::model::ExtraBody>) -> String {
    extra
        .filter(|extra| !extra.is_empty())
        .map_or_else(|| "none".to_string(), crate::model::ExtraBody::canonical)
}

/// What a run file belongs to, as names and values ([`Session::claim`]).
pub fn belonging(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect()
}

/// What the run files of a command that works through a data set's records for one
/// tenant and collection of the knowledge service belong to.
pub fn belongs_to_records<K: KnowledgeService>(
    kb: &K,
    data: &Path,
    tenant: &str,
    collection: &str,
) -> Result<BTreeMap<String, String>, RunError> {
    let records_sha256 = crate::atomic::sha256_of_file(&data.join(RECORDS_FILE))?;
    Ok(belonging(&[
        ("knowledge_service", &kb.address()),
        ("tenant", tenant),
        ("collection", collection),
        ("records_sha256", &records_sha256),
    ]))
}

/// Reads every point of a tenant, page by page, handing each page to `keep`, and answers
/// how many points were read.
///
/// The tenant's points are counted before the read and the two numbers must agree: a
/// read that came back short, or an index that changed while it was read, is a failure
/// and never a smaller result.
pub async fn read_every_point<I, P>(
    index: &I,
    (collection, tenant): (&str, &str),
    want: Want,
    retry: &RetryPolicy,
    mut keep: impl FnMut(Vec<Point<P>>),
) -> Result<u64, GaveUp>
where
    I: IndexReader,
    P: DeserializeOwned + Send,
{
    let expected = with_retries(retry, |_| index.count(collection, tenant))
        .await?
        .value;
    let mut after: Option<PointId> = None;
    let mut read = 0u64;
    loop {
        let page: Page<P> = with_retries(retry, |_| index.scroll(collection, tenant, want, after.as_ref()))
            .await?
            .value;
        read += page.points.len() as u64;
        keep(page.points);
        match page.next_page_offset {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    if read != expected {
        return Err(GaveUp {
            error: RunError::Contract {
                id: tenant.to_string(),
                url: index.address(),
                reason: format!(
                    "the index counts {expected} passages for the tenant and the read returned {read}; \
                     the index changed while it was read, or the read is not whole"
                ),
            },
            tries: 1,
        });
    }
    Ok(read)
}

/// Writes a whole JSON Lines file in one step.
pub fn write_jsonl_atomically<T: serde::Serialize>(path: &Path, items: &[T]) -> Result<(), RunError> {
    use std::io::Write;
    crate::atomic::write_atomically(path, |writer| {
        for item in items {
            serde_json::to_writer(&mut *writer, item).map_err(std::io::Error::other)?;
            writer.write_all(b"\n")?;
        }
        Ok(())
    })
}

/// The ids of a file's items, or the first id that appears twice.
fn unique_ids<'a>(path: &Path, what: &str, ids: impl Iterator<Item = &'a str>) -> Result<(), RunError> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(RunError::file(path, format!("{what} id {id:?} appears twice")));
        }
    }
    Ok(())
}

/// A data set's records. An id that appears twice is refused: it would name two things.
pub fn read_records(data: &Path) -> Result<Vec<Record>, RunError> {
    let path = data.join(RECORDS_FILE);
    let records: Vec<Record> = read_all(&path)?;
    unique_ids(&path, "record", records.iter().map(|r| r.id.as_str()))?;
    Ok(records)
}

/// A data set's questions. An id that appears twice is refused.
pub fn read_queries(data: &Path) -> Result<Vec<Query>, RunError> {
    let path = data.join(QUERIES_FILE);
    let queries: Vec<Query> = read_all(&path)?;
    unique_ids(&path, "question", queries.iter().map(|q| q.id.as_str()))?;
    Ok(queries)
}

/// A file of transcripts. An id that appears twice is refused.
pub fn read_transcripts(path: &Path) -> Result<Vec<Transcript>, RunError> {
    let transcripts: Vec<Transcript> = read_all(path)?;
    unique_ids(path, "transcript", transcripts.iter().map(|t| t.id.as_str()))?;
    Ok(transcripts)
}

/// The manifest's record of the knowledge service, before the service has been asked.
pub fn about_knowledge_service<K: KnowledgeService>(kb: &K, shared: &Shared) -> About {
    About {
        knowledge_service: Some(ServiceRecord {
            url: kb.address(),
            info: None,
            embedding_model: None,
        }),
        machine_label: shared.machine_label.clone(),
        ..About::default()
    }
}

/// Asks the knowledge service what it is (`/v1/info`), puts the answer in the manifest
/// entry, and checks that the service lists every feature in `needs`.
///
/// The command stops before it does anything when the service cannot say what it is, or
/// is a build without a feature the command relies on: run against such a build, every
/// item would fail (an attribute it refuses, a route it lacks), or worse, be answered in
/// a shape the run files would take for the real one.
pub async fn ask_what_the_service_is<K: KnowledgeService>(
    kb: &K,
    shared: &Shared,
    session: &mut Session,
    needs: &[&str],
) -> Result<(), RunError> {
    let raw = with_retries(&shared.retry, |_| kb.info())
        .await
        .map_err(|gave_up| gave_up.error)?
        .value;
    let info: ServiceInfo = serde_json::from_str(raw.get()).map_err(|e| RunError::Malformed {
        url: kb.address(),
        reason: format!("/v1/info: {e}"),
        body: crate::error::excerpt(raw.get().as_bytes()),
    })?;
    if let Some(record) = session.knowledge_service_mut() {
        record.info = Some(raw);
    }
    match missing_feature(&info, needs) {
        None => Ok(()),
        Some(feature) => Err(RunError::Contract {
            id: session.command().to_string(),
            url: kb.address(),
            reason: format!(
                "the service does not list feature {feature:?} in /v1/info; \
                 this command needs a build of the experiment branch that has it"
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Duration;

    #[tokio::test]
    async fn results_are_kept_in_input_order_whatever_finishes_first() {
        let kept = RefCell::new(Vec::new());
        // The first item is the slowest; with three in flight the others finish before it.
        in_order(
            [30u64, 1, 1, 1, 1],
            3,
            |wait| async move {
                tokio::time::sleep(Duration::from_millis(wait)).await;
                wait
            },
            |wait| {
                kept.borrow_mut().push(wait);
                Ok(())
            },
        )
        .await
        .unwrap();
        assert_eq!(*kept.borrow(), vec![30, 1, 1, 1, 1]);
    }

    #[tokio::test]
    async fn an_error_while_keeping_stops_the_work() {
        let kept = RefCell::new(0);
        let result = in_order(
            0..10,
            1,
            |n| async move { n },
            |n| {
                if n == 3 {
                    return Err(RunError::item("3", "cannot be kept"));
                }
                *kept.borrow_mut() += 1;
                Ok(())
            },
        )
        .await;
        assert!(result.is_err());
        assert_eq!(*kept.borrow(), 3);
    }
}
