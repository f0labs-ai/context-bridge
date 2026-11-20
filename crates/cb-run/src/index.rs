//! The search index (Qdrant), reached directly: reading a tenant's passages with their
//! vectors, and — for the control data only — writing passages into the experiment's
//! collection.
//!
//! Reading and writing are two traits on purpose. A command that only reads is given an
//! [`IndexReader`] and cannot write: the product's index is exported through that type
//! alone. The index's layout is the knowledge service's (`kb-adapters/src/qdrant.rs`): one
//! collection, a tenant per payload key `tenant_id`, named vectors `dense` and `sparse`.

use crate::error::RunError;
use crate::http::{Http, endpoint, parse_json, shown};
use reqwest::Url;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::future::Future;

/// The payload key the index keeps a point's tenant under.
pub const TENANT_KEY: &str = "tenant_id";
/// The index's name for a passage's vector by meaning.
pub const DENSE: &str = "dense";
/// The index's name for a passage's keyword vector.
pub const SPARSE: &str = "sparse";

/// A point's id: the index allows a whole number or a text (a UUID). Kept in the type it
/// came in, so it is written back as the same id.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PointId {
    Number(u64),
    Text(String),
}

impl fmt::Display for PointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Number(n) => write!(f, "{n}"),
            Self::Text(text) => f.write_str(text),
        }
    }
}

/// A passage's keyword vector.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// The vectors of a point, as far as they were asked for.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Vectors {
    pub dense: Option<Vec<f32>>,
    pub sparse: Option<SparseVector>,
}

/// One point of a scroll, with the payload read as `P`.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Point<P> {
    pub id: PointId,
    pub payload: Option<P>,
    pub vector: Option<Vectors>,
}

/// One page of a scroll.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Page<P> {
    pub points: Vec<Point<P>>,
    /// Where the next page starts; none after the last page.
    pub next_page_offset: Option<PointId>,
}

/// Where a passage sits in its document: all that `cb-run vectors` reads of a payload.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct PassagePlace {
    pub source_path: Option<String>,
    pub seq: Option<i64>,
}

/// A passage's payload as the product wrote it.
///
/// Only the keys this tool reads or sets are named. Everything else (`rest`) is the
/// product's own enrichment: carried from the export to the import exactly as it is and
/// never looked into. It is the one place a free-form JSON value travels through this
/// tool, because the payload's shape belongs to the product, not to this contract.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassagePayload {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<String>,
    /// The collection label the passage is filed under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    /// The section the passage was cut from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    /// When the passage became known (Unix seconds). The product's index has none; the
    /// import adds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written_at: Option<i64>,
    #[serde(flatten)]
    pub rest: BTreeMap<String, serde_json::Value>,
}

/// Which parts of a payload a scroll asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PayloadWant {
    All,
    Fields(&'static [&'static str]),
}

/// What a scroll asks for besides ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Want {
    pub payload: PayloadWant,
    /// The named vectors to return.
    pub vectors: &'static [&'static str],
    /// Points per page.
    pub page: u32,
}

/// A point to write into the experiment's collection.
#[derive(Clone, Debug, Serialize)]
pub struct PointToWrite<'a> {
    pub id: &'a PointId,
    pub vector: VectorsToWrite<'a>,
    pub payload: &'a PassagePayload,
}

#[derive(Clone, Debug, Serialize)]
pub struct VectorsToWrite<'a> {
    pub dense: &'a [f32],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sparse: Option<&'a SparseVector>,
}

// ---- wire ----------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MatchValue<'a> {
    value: &'a str,
}

#[derive(Debug, Serialize)]
struct Condition<'a> {
    key: &'static str,
    r#match: MatchValue<'a>,
}

#[derive(Debug, Serialize)]
struct Filter<'a> {
    must: [Condition<'a>; 1],
}

impl<'a> Filter<'a> {
    /// The filter every read carries: one tenant's points and no one else's.
    fn tenant(tenant: &'a str) -> Self {
        Self {
            must: [Condition {
                key: TENANT_KEY,
                r#match: MatchValue { value: tenant },
            }],
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum WithPayload {
    All(bool),
    Fields(&'static [&'static str]),
}

#[derive(Debug, Serialize)]
struct ScrollRequest<'a> {
    filter: Filter<'a>,
    limit: u32,
    with_payload: WithPayload,
    with_vector: &'static [&'static str],
    #[serde(skip_serializing_if = "Option::is_none")]
    offset: Option<&'a PointId>,
}

#[derive(Debug, Serialize)]
struct CountRequest<'a> {
    filter: Filter<'a>,
    exact: bool,
}

#[derive(Debug, Deserialize)]
struct Envelope<T> {
    result: T,
}

#[derive(Debug, Deserialize)]
struct Counted {
    count: u64,
}

#[derive(Debug, Serialize)]
struct UpsertRequest<'a> {
    points: &'a [PointToWrite<'a>],
}

#[derive(Debug, Deserialize)]
struct Upserted {
    status: String,
}

// ---- the index -----------------------------------------------------------------------

/// Reading a tenant's points. One try per call. Nothing here can change the index.
pub trait IndexReader: Sync {
    /// The index's address as it may be shown.
    fn address(&self) -> String;

    /// How many points the tenant has, counted exactly.
    fn count(&self, collection: &str, tenant: &str) -> impl Future<Output = Result<u64, RunError>> + Send;

    /// One page of the tenant's points, starting at `after` (the previous page's
    /// `next_page_offset`), with each payload read as `P`.
    fn scroll<P: DeserializeOwned + Send>(
        &self,
        collection: &str,
        tenant: &str,
        want: Want,
        after: Option<&PointId>,
    ) -> impl Future<Output = Result<Page<P>, RunError>> + Send;
}

/// Writing points. One try per call.
pub trait IndexWriter: Sync {
    /// The index's address as it may be shown.
    fn address(&self) -> String;

    /// Writes the points (new, or in place of points with the same ids) and waits until
    /// the index has applied the write.
    fn upsert(
        &self,
        collection: &str,
        points: &[PointToWrite<'_>],
    ) -> impl Future<Output = Result<(), RunError>> + Send;
}

/// The search index over HTTP.
#[derive(Clone, Debug)]
pub struct HttpSearchIndex {
    http: Http,
    base: Url,
}

impl HttpSearchIndex {
    pub fn new(http: Http, base: Url) -> Self {
        Self { http, base }
    }

    /// The index's base address.
    pub fn base(&self) -> &Url {
        &self.base
    }

    fn points_route(&self, collection: &str, tail: &[&str]) -> Result<Url, RunError> {
        let mut segments = vec!["collections", collection, "points"];
        segments.extend_from_slice(tail);
        endpoint(&self.base, &segments)
    }
}

impl IndexReader for HttpSearchIndex {
    fn address(&self) -> String {
        shown(&self.base)
    }

    async fn count(&self, collection: &str, tenant: &str) -> Result<u64, RunError> {
        let url = self.points_route(collection, &["count"])?;
        let body = CountRequest {
            filter: Filter::tenant(tenant),
            exact: true,
        };
        let answered = self
            .http
            .exchange(&url, self.http.client().post(url.clone()).json(&body))
            .await?;
        let counted: Envelope<Counted> = parse_json(&url, &answered.body)?;
        Ok(counted.result.count)
    }

    async fn scroll<P: DeserializeOwned + Send>(
        &self,
        collection: &str,
        tenant: &str,
        want: Want,
        after: Option<&PointId>,
    ) -> Result<Page<P>, RunError> {
        let url = self.points_route(collection, &["scroll"])?;
        let body = ScrollRequest {
            filter: Filter::tenant(tenant),
            limit: want.page,
            with_payload: match want.payload {
                PayloadWant::All => WithPayload::All(true),
                PayloadWant::Fields(fields) => WithPayload::Fields(fields),
            },
            with_vector: want.vectors,
            offset: after,
        };
        let answered = self
            .http
            .exchange(&url, self.http.client().post(url.clone()).json(&body))
            .await?;
        let page: Envelope<Page<P>> = parse_json(&url, &answered.body)?;
        Ok(page.result)
    }
}

impl IndexWriter for HttpSearchIndex {
    fn address(&self) -> String {
        shown(&self.base)
    }

    async fn upsert(&self, collection: &str, points: &[PointToWrite<'_>]) -> Result<(), RunError> {
        let mut url = endpoint(&self.base, &["collections", collection, "points"])?;
        url.query_pairs_mut().append_pair("wait", "true");
        let body = UpsertRequest { points };
        let answered = self
            .http
            .exchange(&url, self.http.client().put(url.clone()).json(&body))
            .await?;
        let upserted: Envelope<Upserted> = parse_json(&url, &answered.body)?;
        if upserted.result.status != "completed" {
            return Err(RunError::Malformed {
                url: shown(&url),
                reason: format!(
                    "the index answered {:?} for a write it was asked to complete",
                    upserted.result.status
                ),
                body: String::new(),
            });
        }
        Ok(())
    }
}

/// An index address reduced to what tells two servers apart: host and port, with the
/// names of this machine counted as one. Used to refuse a write to the index a data set
/// was exported from.
pub fn same_server(a: &Url, b: &Url) -> bool {
    fn host(url: &Url) -> String {
        let host = url
            .host_str()
            .unwrap_or_default()
            .trim_matches(['[', ']'])
            .to_ascii_lowercase();
        match host.as_str() {
            "localhost" | "127.0.0.1" | "::1" | "0.0.0.0" => "this-machine".to_string(),
            _ => host,
        }
    }
    host(a) == host(b) && a.port_or_known_default() == b.port_or_known_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scroll_asks_for_one_tenant_and_only_what_it_needs() {
        let after = PointId::Text("0b6f1d3e-7c2a-4e59-9a1b-2f4c6d8e0a13".into());
        let body = ScrollRequest {
            filter: Filter::tenant("t-1"),
            limit: 256,
            with_payload: WithPayload::Fields(&["source_path", "seq"]),
            with_vector: &[DENSE],
            offset: Some(&after),
        };
        assert_eq!(
            serde_json::to_value(&body).unwrap(),
            serde_json::json!({
                "filter": {"must": [{"key": "tenant_id", "match": {"value": "t-1"}}]},
                "limit": 256,
                "with_payload": ["source_path", "seq"],
                "with_vector": ["dense"],
                "offset": "0b6f1d3e-7c2a-4e59-9a1b-2f4c6d8e0a13",
            })
        );
        let first = ScrollRequest {
            filter: Filter::tenant("t-1"),
            limit: 2,
            with_payload: WithPayload::All(true),
            with_vector: &[DENSE, SPARSE],
            offset: None,
        };
        let first = serde_json::to_value(&first).unwrap();
        assert_eq!(first["with_payload"], true);
        assert!(first.get("offset").is_none());
    }

    #[test]
    fn a_page_is_read_with_ids_of_either_kind_and_named_vectors() {
        let page: Envelope<Page<PassagePlace>> = serde_json::from_value(serde_json::json!({
            "result": {
                "points": [
                    {"id": "0b6f1d3e-7c2a-4e59-9a1b-2f4c6d8e0a13", "payload": {"source_path": "cb/r-1.md", "seq": 0},
                     "vector": {"dense": [0.25, -0.5]}},
                    {"id": 42, "payload": {"source_path": "https://docs.example.com/a"},
                     "vector": {"dense": [1.0, 0.0], "sparse": {"indices": [7, 9], "values": [0.5, 1.5]}}},
                    {"id": 43, "payload": null},
                ],
                "next_page_offset": 44,
            },
            "status": "ok", "time": 0.002,
        }))
        .unwrap();
        let points = page.result.points;
        assert_eq!(points[0].id.to_string(), "0b6f1d3e-7c2a-4e59-9a1b-2f4c6d8e0a13");
        assert_eq!(points[0].payload.as_ref().unwrap().seq, Some(0));
        assert_eq!(points[1].id, PointId::Number(42));
        assert_eq!(points[1].payload.as_ref().unwrap().seq, None);
        assert_eq!(
            points[1]
                .vector
                .as_ref()
                .unwrap()
                .sparse
                .as_ref()
                .unwrap()
                .indices,
            vec![7, 9]
        );
        assert!(points[2].payload.is_none() && points[2].vector.is_none());
        assert_eq!(page.result.next_page_offset, Some(PointId::Number(44)));
    }

    #[test]
    fn a_payload_keeps_every_key_it_does_not_name() {
        let raw = serde_json::json!({
            "tenant_id": "product", "collection": "website", "text": "Open Settings.", "heading": "Settings",
            "parent_id": "ad6190bf", "kind": "step", "seq": 3, "step_no": 2, "labels": ["a", "b"],
            "spoken_answer": null, "score_hint": 0.1, "nested": {"deep": [1, {"x": true}]},
        });
        let payload: PassagePayload = serde_json::from_value(raw.clone()).unwrap();
        assert_eq!(payload.text.as_deref(), Some("Open Settings."));
        assert_eq!(payload.parent_id.as_deref(), Some("ad6190bf"));
        assert_eq!(payload.rest.len(), 7);
        assert_eq!(serde_json::to_value(&payload).unwrap(), raw);
    }

    #[test]
    fn a_point_is_written_with_its_own_id_and_both_vectors() {
        let id = PointId::Number(7);
        let payload: PassagePayload = serde_json::from_value(
            serde_json::json!({"tenant_id": "exp", "collection": "cb", "written_at": 99}),
        )
        .unwrap();
        let sparse = SparseVector {
            indices: vec![3],
            values: vec![0.5],
        };
        let point = PointToWrite {
            id: &id,
            vector: VectorsToWrite {
                dense: &[0.5, 0.25],
                sparse: Some(&sparse),
            },
            payload: &payload,
        };
        assert_eq!(
            serde_json::to_value(UpsertRequest { points: &[point] }).unwrap(),
            serde_json::json!({"points": [{
                "id": 7,
                "vector": {"dense": [0.5, 0.25], "sparse": {"indices": [3], "values": [0.5]}},
                "payload": {"tenant_id": "exp", "collection": "cb", "written_at": 99},
            }]})
        );
    }

    #[test]
    fn the_names_of_this_machine_are_one_server_and_another_port_is_another() {
        let url = |s: &str| Url::parse(s).unwrap();
        assert!(same_server(
            &url("http://localhost:6339"),
            &url("http://127.0.0.1:6339/")
        ));
        assert!(same_server(
            &url("http://[::1]:6339"),
            &url("http://LOCALHOST:6339")
        ));
        assert!(same_server(
            &url("https://qdrant.example"),
            &url("https://QDRANT.example:443/x")
        ));
        assert!(!same_server(
            &url("http://localhost:6339"),
            &url("http://localhost:6340")
        ));
        assert!(!same_server(
            &url("http://qdrant-a:6333"),
            &url("http://qdrant-b:6333")
        ));
    }
}
