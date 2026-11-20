//! The knowledge service: uploads, candidate lists, the live search, embeddings.
//!
//! [`KnowledgeService`] is what the commands need from it; [`HttpKnowledgeService`] is the
//! one real implementation. Route shapes and field names are the service's own
//! (`kb-api/src/routes.rs`, `kb-domain/src/wire.rs`) and docs/KB_EXPERIMENT.md for the
//! experiment branch. Every method makes ONE try; [`crate::retry`] decides about more.

use crate::error::RunError;
use crate::http::{Http, endpoint, parse_json, shown};
use crate::secret::{Secret, usable};
use reqwest::Url;
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::time::Duration;

/// A value from a service, with the wall time of the call that fetched it.
#[derive(Clone, Debug, PartialEq)]
pub struct Timed<T> {
    pub value: T,
    pub wall: Duration,
}

/// A tenant of the knowledge service, checked the way the service checks one, so it can
/// never break out of its place in a route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tenant(String);

impl Tenant {
    pub fn parse(raw: &str) -> Result<Self, RunError> {
        let ok = !raw.is_empty()
            && raw.len() <= 256
            && !raw
                .chars()
                .any(|c| matches!(c, '/' | '\\' | '\0' | ' ' | '\t' | '\r' | '\n'));
        if ok {
            Ok(Self(raw.to_string()))
        } else {
            Err(RunError::Setting {
                name: "--tenant",
                reason: "a tenant is 1 to 256 characters without slashes or spaces".into(),
            })
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A collection label of the knowledge service (the name a tenant files documents
/// under), checked the way the service checks one: `a-z0-9_-`, at most 40.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label(String);

impl Label {
    pub fn parse(name: &'static str, raw: &str) -> Result<Self, RunError> {
        let ok = !raw.is_empty()
            && raw.len() <= 40
            && raw
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-');
        if ok {
            Ok(Self(raw.to_string()))
        } else {
            Err(RunError::Setting {
                name,
                reason: "a collection label is 1 to 40 characters of a-z, 0-9, _ and -".into(),
            })
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

// ---- upload --------------------------------------------------------------------------

/// What a record's document says about itself (the service's `DocumentAttributes`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DocumentAttributes<'a> {
    pub written_at: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<std::borrow::Cow<'a, str>>,
    pub document_ref: &'a str,
}

/// The longest title the service keeps, in characters (its `MAX_TITLE_CHARS`).
pub const MAX_TITLE_CHARS: usize = 300;

/// A record's title as the service takes it. A title is a display name, and the
/// record's text carries its content in full, so a longer one is cut at the last word
/// that fits and ends in "…" rather than refused.
pub fn title_for_service(title: &str) -> std::borrow::Cow<'_, str> {
    if title.chars().count() <= MAX_TITLE_CHARS {
        return std::borrow::Cow::Borrowed(title);
    }
    let head: String = title.chars().take(MAX_TITLE_CHARS - 1).collect();
    let cut = match head.rfind(char::is_whitespace) {
        Some(at) if at > 0 => head[..at].trim_end(),
        _ => head.as_str(),
    };
    std::borrow::Cow::Owned(format!("{cut}…"))
}

/// One record as the document it is uploaded as.
#[derive(Clone, Debug, PartialEq)]
pub struct Document<'a> {
    pub record_id: &'a str,
    /// The document's path in the service: `cb/<record id>.md`.
    pub path: String,
    pub text: &'a str,
    pub collection: &'a Label,
    pub attributes: DocumentAttributes<'a>,
}

impl<'a> Document<'a> {
    pub fn of(record: &'a cb_types::Record, collection: &'a Label) -> Self {
        Self {
            record_id: &record.id,
            path: cb_types::ids::source_path(&record.id),
            text: &record.text,
            collection,
            attributes: DocumentAttributes {
                written_at: record.written_at,
                valid_until: record.valid_until,
                title: record.title.as_deref().map(title_for_service),
                document_ref: &record.id,
            },
        }
    }
}

/// What the service said about one indexed document.
#[derive(Clone, Debug, PartialEq)]
pub struct Uploaded {
    pub source_id: Option<String>,
    /// The service's `chunks_indexed`.
    pub passages_indexed: u64,
    pub bytes_processed: Option<u64>,
    /// Whether the service's language model rewrote the document before indexing it.
    pub refined: bool,
    pub task_extracted: bool,
    pub symptom_chunks: u64,
    pub spoken_chunks: u64,
    /// What the service masked, by pattern name.
    pub redactions: BTreeMap<String, u64>,
    /// Where the upload's time went, as the service measured it.
    pub timings_ms: BTreeMap<String, f64>,
}

#[derive(Debug, Deserialize)]
struct IngestAnswer {
    sources: Vec<IngestedSource>,
    #[serde(default)]
    jobs: Vec<QueuedJob>,
}

#[derive(Debug, Deserialize)]
struct IngestedSource {
    source_path: String,
    source_id: Option<String>,
    chunks_indexed: u64,
    bytes_processed: Option<u64>,
    #[serde(default)]
    refined: bool,
    #[serde(default)]
    task_extracted: bool,
    #[serde(default)]
    symptom_chunks: u64,
    #[serde(default)]
    spoken_chunks: u64,
    redactions: Option<BTreeMap<String, u64>>,
    attributes: Option<EchoedAttributes>,
    timings_ms: Option<BTreeMap<String, f64>>,
}

#[derive(Debug, Deserialize)]
struct EchoedAttributes {
    written_at: Option<i64>,
    valid_until: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct QueuedJob {
    job_id: Option<String>,
    status: Option<String>,
}

/// Reads the service's answer to an upload as "this document is indexed, with these
/// times", or says why it cannot be read that way. Nothing is assumed: a queued
/// document, no passage, a time the service did not keep or missing timings each make the
/// upload a failure, because each would leave a hole that later looks like data.
fn uploaded_from(answer: IngestAnswer, document: &Document<'_>, url: &Url) -> Result<Uploaded, RunError> {
    let unusable = |reason: String| RunError::Contract {
        id: document.record_id.to_string(),
        url: shown(url),
        reason,
    };
    if let Some(job) = answer.jobs.first() {
        return Err(unusable(format!(
            "the service queued the document (job {}, {}) instead of indexing it within the request",
            job.job_id.as_deref().unwrap_or("?"),
            job.status.as_deref().unwrap_or("?"),
        )));
    }
    let count = answer.sources.len();
    let mut sources = answer.sources.into_iter();
    let (Some(source), None) = (sources.next(), sources.next()) else {
        return Err(unusable(format!(
            "one document was uploaded and {count} came back"
        )));
    };
    if source.source_path != document.path {
        return Err(unusable(format!(
            "uploaded as {} and indexed as {}",
            document.path, source.source_path
        )));
    }
    if source.chunks_indexed == 0 {
        return Err(unusable("the service indexed no passage of this document".into()));
    }
    let Some(echo) = source.attributes else {
        return Err(unusable(
            "the service did not echo the document's attributes".into(),
        ));
    };
    if echo.written_at != Some(document.attributes.written_at) {
        return Err(unusable(format!(
            "written_at {} was sent and {:?} was kept",
            document.attributes.written_at, echo.written_at
        )));
    }
    if echo.valid_until != document.attributes.valid_until {
        return Err(unusable(format!(
            "valid_until {:?} was sent and {:?} was kept",
            document.attributes.valid_until, echo.valid_until
        )));
    }
    let Some(timings_ms) = source.timings_ms else {
        return Err(unusable(
            "the service reported no timings_ms for the upload".into(),
        ));
    };
    Ok(Uploaded {
        source_id: source.source_id,
        passages_indexed: source.chunks_indexed,
        bytes_processed: source.bytes_processed,
        refined: source.refined,
        task_extracted: source.task_extracted,
        symptom_chunks: source.symptom_chunks,
        spoken_chunks: source.spoken_chunks,
        redactions: source.redactions.unwrap_or_default(),
        timings_ms,
    })
}

// ---- candidate lists (docs/KB_EXPERIMENT.md section 3) -------------------------------

/// One view of the index a question is asked against (DESIGN.md section 3.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ViewRequest {
    pub name: String,
    /// The view sees passages written up to this moment (a batch index's last refresh).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index_as_of: Option<i64>,
    /// The view sees passages written after this moment (a recency window's start).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub written_from: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RerankRequest {
    pub view: String,
    pub per_list: u32,
    /// How many of the view's newest passages are scored too (feature `eval_newest`).
    /// Not sent when 0, so a service without the feature is asked what it always was.
    #[serde(skip_serializing_if = "is_zero")]
    pub newest: u32,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

/// The body of `POST /v1/tenants/{tenant}/eval/candidates`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CandidatesRequest {
    pub query: String,
    pub as_of: i64,
    pub n: u32,
    pub exact: bool,
    pub collections: Vec<String>,
    pub views: Vec<ViewRequest>,
    /// How many of its newest passages each view also lists (feature `eval_newest`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub newest: Option<u32>,
    pub rerank: RerankRequest,
}

/// A point of the index with the score one ranker gave it.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PointScore {
    pub id: String,
    pub score: f64,
}

/// The two ranked lists of one view, best first, and its newest passages when asked.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ViewLists {
    pub dense: Vec<PointScore>,
    pub sparse: Vec<PointScore>,
    /// Newest first, each point scored with its `written_at`; only when asked for.
    #[serde(default)]
    pub newest: Option<Vec<PointScore>>,
}

/// What the service says about a point that appears in an answer.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PassageInfo {
    pub source_path: String,
    pub kind: Option<String>,
    pub written_at: Option<i64>,
    pub valid_until: Option<i64>,
    pub chars: Option<u64>,
}

/// The answer of `POST /v1/tenants/{tenant}/eval/candidates`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct CandidatesAnswer {
    pub views: HashMap<String, ViewLists>,
    pub rerank: Vec<PointScore>,
    pub passages: HashMap<String, PassageInfo>,
    pub timings_ms: BTreeMap<String, f64>,
}

// ---- live search ---------------------------------------------------------------------

/// The body of `POST /v1/tenants/{tenant}/search`, as a call sends it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SearchRequest {
    pub query: String,
    pub top_k: u32,
    pub score_threshold: f64,
    pub collections: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SearchHit {
    pub source_path: String,
}

/// The part of the live search's answer a trip reads: which documents came back, and
/// whether the service says the answer was degraded.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct SearchAnswer {
    pub hits: Vec<SearchHit>,
    #[serde(default)]
    pub degraded: bool,
    pub mode: Option<String>,
}

impl SearchAnswer {
    /// Whether the answer holds a passage of this record's document.
    pub fn contains_record(&self, record_id: &str) -> bool {
        self.hits
            .iter()
            .any(|hit| cb_types::ids::record_of(&hit.source_path) == Some(record_id))
    }
}

// ---- embeddings ----------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct EmbedRequest<'a> {
    texts: &'a [String],
    /// The service embeds for one purpose only: the way a search embeds a question.
    kind: &'static str,
}

/// The answer of `POST /v1/embed`: one vector per text, in order.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Embedded {
    pub model: String,
    pub dims: usize,
    pub vectors: Vec<Vec<f32>>,
}

// ---- what the service is -------------------------------------------------------------

/// The part of `/v1/info` a command reads: what this build of the service accepts beyond
/// the original contract. The whole answer is kept in the manifest as it came.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct ServiceInfo {
    #[serde(default)]
    pub features: Vec<String>,
}

/// The features of the experiment branch (`docs/EXPERIMENT_CONTEXTBRIDGE.md` section 1 in
/// the service's repository), as `/v1/info` lists them.
pub mod features {
    /// An upload's attributes may carry `written_at` and `valid_until`.
    pub const DOCUMENT_TIMES: &str = "document_times";
    /// Each `sources[]` entry of an upload carries `timings_ms`.
    pub const INGEST_TIMINGS: &str = "ingest_timings";
    /// `POST /v1/tenants/{tenant}/eval/candidates` exists (a build with the `eval` feature).
    pub const EVAL_CANDIDATES: &str = "eval_candidates";
    /// That route takes `newest` and `rerank.newest`: each view's newest passages, scored.
    pub const EVAL_NEWEST: &str = "eval_newest";
}

/// The first of `needed` that the service does not list, if any.
pub fn missing_feature<'a>(info: &ServiceInfo, needed: &[&'a str]) -> Option<&'a str> {
    needed
        .iter()
        .find(|feature| !info.features.iter().any(|listed| listed == *feature))
        .copied()
}

#[derive(Debug, Deserialize)]
struct PassageCount {
    points: u64,
}

// ---- the service ---------------------------------------------------------------------

/// What the commands need from the knowledge service. One try per call.
pub trait KnowledgeService: Sync {
    /// The service's address as it may be shown.
    fn address(&self) -> String;

    /// The service's `/v1/info`, exactly as it answered.
    fn info(&self) -> impl Future<Output = Result<Box<RawValue>, RunError>> + Send;

    /// How many passages the index holds for one document of a collection label
    /// (`GET /v1/tenants/{tenant}/sources/count`).
    fn count_passages(
        &self,
        tenant: &Tenant,
        path: &str,
        collection: &Label,
    ) -> impl Future<Output = Result<u64, RunError>> + Send;

    /// Uploads one record as its own document and answers what the service indexed.
    fn upload(
        &self,
        tenant: &Tenant,
        document: &Document<'_>,
    ) -> impl Future<Output = Result<Timed<Uploaded>, RunError>> + Send;

    /// The candidate lists behind a search, for several views of the index at once.
    fn candidates(
        &self,
        tenant: &Tenant,
        request: &CandidatesRequest,
    ) -> impl Future<Output = Result<Timed<CandidatesAnswer>, RunError>> + Send;

    /// The live search, as a call makes it.
    fn search(
        &self,
        tenant: &Tenant,
        request: &SearchRequest,
    ) -> impl Future<Output = Result<Timed<SearchAnswer>, RunError>> + Send;

    /// Embeds texts the way a search embeds a question.
    fn embed(&self, texts: &[String]) -> impl Future<Output = Result<Timed<Embedded>, RunError>> + Send;
}

/// The knowledge service over HTTP.
#[derive(Clone, Debug)]
pub struct HttpKnowledgeService {
    http: Http,
    base: Url,
    token: Option<Secret>,
}

impl HttpKnowledgeService {
    pub fn new(http: Http, base: Url, token: Option<Secret>) -> Self {
        Self {
            http,
            base,
            token: usable(token.as_ref()).cloned(),
        }
    }

    fn with_token(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.token {
            Some(token) => request.bearer_auth(token.expose()),
            None => request,
        }
    }

    fn tenant_route(&self, tenant: &Tenant, tail: &[&str]) -> Result<Url, RunError> {
        let mut segments = vec!["v1", "tenants", tenant.as_str()];
        segments.extend_from_slice(tail);
        endpoint(&self.base, &segments)
    }
}

impl KnowledgeService for HttpKnowledgeService {
    fn address(&self) -> String {
        shown(&self.base)
    }

    async fn info(&self) -> Result<Box<RawValue>, RunError> {
        let url = endpoint(&self.base, &["v1", "info"])?;
        let answered = self
            .http
            .exchange(&url, self.with_token(self.http.client().get(url.clone())))
            .await?;
        parse_json(&url, &answered.body)
    }

    async fn count_passages(&self, tenant: &Tenant, path: &str, collection: &Label) -> Result<u64, RunError> {
        let mut url = self.tenant_route(tenant, &["sources", "count"])?;
        url.query_pairs_mut()
            .append_pair("source_path", path)
            .append_pair("collection", collection.as_str());
        let answered = self
            .http
            .exchange(&url, self.with_token(self.http.client().get(url.clone())))
            .await?;
        let counted: PassageCount = parse_json(&url, &answered.body)?;
        Ok(counted.points)
    }

    async fn upload(&self, tenant: &Tenant, document: &Document<'_>) -> Result<Timed<Uploaded>, RunError> {
        let url = self.tenant_route(tenant, &["ingest"])?;
        let attributes = serde_json::to_string(&document.attributes)
            .map_err(|e| RunError::item(document.record_id, format!("attributes: {e}")))?;
        let file = Part::bytes(document.text.as_bytes().to_vec())
            .file_name(document.path.clone())
            .mime_str("text/markdown; charset=utf-8")
            .map_err(|e| RunError::item(document.record_id, format!("document part: {e}")))?;
        // The service's own field names: `files` (the file's name is the document's
        // path), `collection`, `attributes` (a JSON string).
        let form = Form::new()
            .text("collection", document.collection.as_str().to_string())
            .text("attributes", attributes)
            .part("files", file);
        let request = self
            .with_token(self.http.client().post(url.clone()))
            .multipart(form);
        let answered = self.http.exchange(&url, request).await?;
        let answer: IngestAnswer = parse_json(&url, &answered.body)?;
        Ok(Timed {
            value: uploaded_from(answer, document, &url)?,
            wall: answered.wall,
        })
    }

    async fn candidates(
        &self,
        tenant: &Tenant,
        request: &CandidatesRequest,
    ) -> Result<Timed<CandidatesAnswer>, RunError> {
        let url = self.tenant_route(tenant, &["eval", "candidates"])?;
        let answered = self
            .http
            .exchange(
                &url,
                self.with_token(self.http.client().post(url.clone()))
                    .json(request),
            )
            .await?;
        Ok(Timed {
            value: parse_json(&url, &answered.body)?,
            wall: answered.wall,
        })
    }

    async fn search(
        &self,
        tenant: &Tenant,
        request: &SearchRequest,
    ) -> Result<Timed<SearchAnswer>, RunError> {
        let url = self.tenant_route(tenant, &["search"])?;
        let answered = self
            .http
            .exchange(
                &url,
                self.with_token(self.http.client().post(url.clone()))
                    .json(request),
            )
            .await?;
        Ok(Timed {
            value: parse_json(&url, &answered.body)?,
            wall: answered.wall,
        })
    }

    async fn embed(&self, texts: &[String]) -> Result<Timed<Embedded>, RunError> {
        let url = endpoint(&self.base, &["v1", "embed"])?;
        let body = EmbedRequest { texts, kind: "query" };
        let answered = self
            .http
            .exchange(
                &url,
                self.with_token(self.http.client().post(url.clone())).json(&body),
            )
            .await?;
        let embedded: Embedded = parse_json(&url, &answered.body)?;
        let sound = embedded.dims > 0
            && embedded.vectors.len() == texts.len()
            && embedded.vectors.iter().all(|v| v.len() == embedded.dims);
        if !sound {
            return Err(RunError::Malformed {
                url: shown(&url),
                reason: format!(
                    "{} vectors for {} texts at a stated width of {}",
                    embedded.vectors.len(),
                    texts.len(),
                    embedded.dims
                ),
                body: String::new(),
            });
        }
        Ok(Timed {
            value: embedded,
            wall: answered.wall,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::{Record, Tier};

    fn record() -> Record {
        Record {
            id: "inc-0001.s4".into(),
            text: "Home internet is down in Northgate.".into(),
            written_at: 1_780_272_000,
            valid_until: Some(1_780_279_200),
            thread: Some("inc-0001".into()),
            tier: Tier::Changing,
            title: Some("Northgate outage".into()),
            source: "synthetic".into(),
            truth: None,
        }
    }

    fn url() -> Url {
        Url::parse("http://kb/v1/tenants/t/ingest").unwrap()
    }

    fn answer(source: serde_json::Value) -> IngestAnswer {
        serde_json::from_value(serde_json::json!({ "sources": [source], "jobs": [] })).unwrap()
    }

    fn good_source() -> serde_json::Value {
        serde_json::json!({
            "source_id": "sha256-abc", "source_path": "cb/inc-0001.s4.md", "chunks_indexed": 2,
            "bytes_processed": 35, "refined": false,
            "attributes": {"written_at": 1_780_272_000, "valid_until": 1_780_279_200, "document_ref": "inc-0001.s4"},
            "timings_ms": {"parse": 0.2, "embed": 8.0, "searchable": 21.3, "total": 25.3},
            "quality": {"anything": "the service adds"},
        })
    }

    #[test]
    fn a_record_becomes_a_document_at_its_own_path_with_its_times_and_title() {
        let label = Label::parse("--collection", "cb").unwrap();
        let record = record();
        let document = Document::of(&record, &label);
        assert_eq!(document.path, "cb/inc-0001.s4.md");
        let attributes = serde_json::to_value(&document.attributes).unwrap();
        assert_eq!(
            attributes,
            serde_json::json!({
                "written_at": 1_780_272_000, "valid_until": 1_780_279_200,
                "title": "Northgate outage", "document_ref": "inc-0001.s4",
            })
        );
        // What is not set is not sent: the service refuses fields it does not expect as null.
        let mut bare = record.clone();
        bare.valid_until = None;
        bare.title = None;
        let attributes = serde_json::to_value(Document::of(&bare, &label).attributes).unwrap();
        assert_eq!(
            attributes,
            serde_json::json!({"written_at": 1_780_272_000, "document_ref": "inc-0001.s4"})
        );
    }

    #[test]
    fn a_title_longer_than_the_service_keeps_is_cut_at_a_word() {
        let short = "Cloud Spanner: node count changes fail";
        assert_eq!(title_for_service(short), short);
        let exactly = "x".repeat(MAX_TITLE_CHARS);
        assert_eq!(title_for_service(&exactly), exactly.as_str());
        let long = "word ".repeat(80); // 400 characters
        let cut = title_for_service(&long);
        assert!(cut.chars().count() <= MAX_TITLE_CHARS, "{}", cut.chars().count());
        assert!(cut.ends_with("word…"), "{cut:?}");
        // One unbroken run of letters is cut where it must be.
        let unbroken = "é".repeat(MAX_TITLE_CHARS + 20);
        let cut = title_for_service(&unbroken);
        assert_eq!(cut.chars().count(), MAX_TITLE_CHARS);
        assert!(cut.ends_with('…'));
    }

    #[test]
    fn an_indexed_document_is_read_with_its_passages_and_timings() {
        let label = Label::parse("--collection", "cb").unwrap();
        let record = record();
        let uploaded = uploaded_from(answer(good_source()), &Document::of(&record, &label), &url()).unwrap();
        assert_eq!(uploaded.passages_indexed, 2);
        assert_eq!(uploaded.timings_ms.get("searchable"), Some(&21.3));
        assert_eq!(uploaded.source_id.as_deref(), Some("sha256-abc"));
    }

    #[test]
    fn an_upload_that_left_a_hole_is_a_failure_that_names_the_record() {
        let label = Label::parse("--collection", "cb").unwrap();
        let record = record();
        let document = Document::of(&record, &label);
        type Change = fn(&mut serde_json::Value);
        let broken = |change: Change| {
            let mut source = good_source();
            change(&mut source);
            uploaded_from(answer(source), &document, &url()).unwrap_err()
        };
        let cases: [(&str, Change); 6] = [
            ("no passage", |s| s["chunks_indexed"] = 0.into()),
            ("did not echo", |s| {
                s.as_object_mut().unwrap().remove("attributes");
            }),
            ("written_at", |s| s["attributes"]["written_at"] = 5.into()),
            ("valid_until", |s| {
                s["attributes"]["valid_until"] = serde_json::Value::Null
            }),
            ("timings_ms", |s| {
                s.as_object_mut().unwrap().remove("timings_ms");
            }),
            ("indexed as", |s| s["source_path"] = "cb/other.md".into()),
        ];
        for (expected, change) in cases {
            let error = broken(change);
            let text = error.to_string();
            assert!(matches!(error, RunError::Contract { .. }), "{text}");
            assert!(text.contains(expected) && text.contains("inc-0001.s4"), "{text}");
            assert!(!error.is_retryable());
        }
    }

    #[test]
    fn a_queued_document_is_not_an_indexed_one() {
        let label = Label::parse("--collection", "cb").unwrap();
        let record = record();
        let queued: IngestAnswer = serde_json::from_value(serde_json::json!({
            "sources": [], "jobs": [{"job_id": "j-7", "source_path": "cb/inc-0001.s4.md", "status": "queued"}],
        }))
        .unwrap();
        let error = uploaded_from(queued, &Document::of(&record, &label), &url()).unwrap_err();
        assert!(error.to_string().contains("job j-7"), "{error}");
        let none: IngestAnswer = serde_json::from_value(serde_json::json!({"sources": []})).unwrap();
        assert!(uploaded_from(none, &Document::of(&record, &label), &url()).is_err());
    }

    #[test]
    fn a_feature_the_service_does_not_list_is_named() {
        let info: ServiceInfo = serde_json::from_value(serde_json::json!({
            "dense_model": "intfloat/multilingual-e5-large", "dense_dim": 1024,
            "features": ["document_titles", "document_times", "search_as_of", "ingest_timings"],
        }))
        .unwrap();
        assert_eq!(
            missing_feature(&info, &[features::DOCUMENT_TIMES, features::INGEST_TIMINGS]),
            None
        );
        assert_eq!(
            missing_feature(&info, &[features::DOCUMENT_TIMES, features::EVAL_CANDIDATES]),
            Some("eval_candidates")
        );
        // A service from before features were listed lists none.
        let old: ServiceInfo = serde_json::from_value(serde_json::json!({"dense_model": "x"})).unwrap();
        assert_eq!(
            missing_feature(&old, &[features::DOCUMENT_TIMES]),
            Some("document_times")
        );
        assert_eq!(missing_feature(&old, &[]), None);
    }

    #[test]
    fn tenants_and_labels_are_checked_as_the_service_checks_them() {
        assert!(Tenant::parse("0c5a9e21-7d3b-4f6a-9b1e-2a8c4d5e6f70").is_ok());
        for bad in ["", "a/b", "a b", "a\\b", "a\nb"] {
            assert!(Tenant::parse(bad).is_err(), "{bad:?}");
        }
        assert!(Label::parse("--collection", "cb").is_ok());
        assert!(Label::parse("--collection", "web-site_1").is_ok());
        for bad in ["", "Cb", "c b", &"x".repeat(41)] {
            assert!(Label::parse("--collection", bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_search_answer_contains_a_record_by_its_document_path() {
        let answer: SearchAnswer = serde_json::from_value(serde_json::json!({
            "hits": [
                {"id": "p1", "text": "never read here", "source_path": "https://docs.example.com/page", "score": 0.9},
                {"id": "p2", "text": "never read here", "source_path": "cb/inc-0001.s4.md", "score": 0.4},
            ],
            "chunks": [], "degraded": false, "mode": "reranked", "latency_ms": 12.5,
        }))
        .unwrap();
        assert!(answer.contains_record("inc-0001.s4"));
        assert!(!answer.contains_record("inc-0001.s3"));
        assert_eq!(answer.mode.as_deref(), Some("reranked"));
    }
}
