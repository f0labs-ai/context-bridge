//! An in-process stand-in for the three services `cb-run` talks to: the knowledge
//! service, the search index and a language-model endpoint.
//!
//! It listens on `127.0.0.1:0` and keeps the contracts closely enough for the commands to
//! be tested for real, after the experiment branch as built
//! (`docs/EXPERIMENT_CONTEXTBRIDGE.md` in the service's repository):
//!
//! - an upload is read as multipart the way the service reads it, its dates are checked
//!   as the service checks them, and each paragraph of a document is one passage whose
//!   point is keyed by its text, so a later document with the same paragraph takes the
//!   point over (the service's finding 1);
//! - candidate views are cut by time as section 3 of the contract says;
//! - the live search leaves out what is not written yet and what has ended;
//! - the index scrolls in pages.
//!
//! On top of that, any request can be made to go wrong ([`Fake::rule`]): a status, a body
//! that is not JSON, an answer that comes too late.

use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The routes of the stand-in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Info,
    Ingest,
    /// The knowledge service's `GET /v1/tenants/{tenant}/sources/count`.
    PassageCount,
    Candidates,
    Search,
    Embed,
    /// The search index's `POST /collections/{collection}/points/count`.
    Count,
    Scroll,
    Upsert,
    Chat,
}

/// What a request gets in place of its normal answer.
#[derive(Clone, Debug)]
pub enum Action {
    /// This status, with a small JSON body.
    Status(u16),
    /// This status and a `Retry-After` header of that many seconds.
    StatusRetryAfter(u16, u64),
    /// 200 with a body that is not JSON.
    Garbage,
    /// 200 with JSON of another shape than the contract's.
    WrongShape,
    /// The normal answer, after a wait.
    Slow(Duration),
    /// A wait, then this status.
    SlowStatus(Duration, u16),
    /// 200 with exactly this body. For the model route: this text as the answer's content.
    Body(String),
}

struct Rule {
    route: Route,
    contains: String,
    times: usize,
    action: Action,
}

/// A request the stand-in received.
#[derive(Clone, Debug)]
pub struct Seen {
    pub route: Route,
    /// What identifies the request: the uploaded file's name and fields, or the JSON body.
    pub key: String,
    pub body: Value,
    pub authorization: Option<String>,
    pub at: Instant,
}

/// A document the knowledge stand-in holds.
#[derive(Clone, Debug)]
pub struct Stored {
    pub tenant: String,
    pub collection: String,
    pub path: String,
    pub text: String,
    pub attributes: Value,
    /// The point ids of the passages it still holds (a later document with the same
    /// paragraph takes that point over).
    pub points: Vec<String>,
    /// When its upload started.
    pub uploaded_at: Instant,
}

impl Stored {
    fn written_at(&self) -> Option<i64> {
        self.attributes.get("written_at").and_then(Value::as_i64)
    }

    fn valid_until(&self) -> Option<i64> {
        self.attributes.get("valid_until").and_then(Value::as_i64)
    }
}

/// The passages of a document's text: its paragraphs.
fn paragraphs(text: &str) -> Vec<&str> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect()
}

/// A passage's point id: keyed by the passage's text within its tenant and label, as the
/// service keys it (`uuid5(tenant|collection|sha256(text))`).
pub fn point_id(tenant: &str, collection: &str, passage: &str) -> String {
    format!(
        "pt-{:016x}",
        cb_types::fnv1a64(format!("{tenant}|{collection}|{passage}").as_bytes())
    )
}

/// The server's clock, Unix seconds.
fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

/// A point the index stand-in holds.
#[derive(Clone, Debug, PartialEq)]
pub struct FakePoint {
    pub id: Value,
    pub payload: Value,
    pub vector: Value,
}

type ChatAnswer = Box<dyn Fn(&str) -> String + Send + Sync>;

pub struct FakeState {
    rules: Mutex<Vec<Rule>>,
    seen: Mutex<Vec<Seen>>,
    // -- knowledge service --
    /// What `/v1/info` lists under `features`.
    pub features: Mutex<Vec<String>>,
    pub documents: Mutex<Vec<Stored>>,
    /// How long an upload takes to answer.
    pub upload_takes: Mutex<Duration>,
    /// How long after an upload starts the search returns its document.
    pub searchable_after: Mutex<Duration>,
    /// How long a search takes to answer.
    pub search_takes: Mutex<Duration>,
    searches_under_way: AtomicUsize,
    pub most_searches_at_once: AtomicUsize,
    /// Whether an upload's answer carries `timings_ms` (the experiment branch's addition).
    pub reports_timings: AtomicBool,
    // -- search index --
    pub collections: Mutex<BTreeMap<String, BTreeMap<String, FakePoint>>>,
    /// The most points a scroll page holds, whatever the request asks for.
    pub page_cap: AtomicUsize,
    // -- model --
    chat_answer: Mutex<ChatAnswer>,
    /// Whether the endpoint refuses `response_format`, as some OpenAI-compatible servers do.
    pub refuses_response_format: AtomicBool,
    pub reports_usage: AtomicBool,
    /// What a thinking model puts beside its answer, in `reasoning_content` and `reasoning`.
    pub reasoning: Mutex<Option<String>>,
}

/// The stand-in, listening.
pub struct Fake {
    pub url: String,
    pub state: Arc<FakeState>,
    server: tokio::task::JoinHandle<()>,
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.server.abort();
    }
}

/// What the experiment build's `/v1/info` lists (and one feature from before).
pub const EXPERIMENT_FEATURES: [&str; 5] = [
    "document_titles",
    "document_times",
    "search_as_of",
    "ingest_timings",
    "eval_candidates",
];

impl Fake {
    pub async fn start() -> Self {
        let state = Arc::new(FakeState {
            rules: Mutex::new(Vec::new()),
            seen: Mutex::new(Vec::new()),
            features: Mutex::new(EXPERIMENT_FEATURES.iter().map(|f| f.to_string()).collect()),
            documents: Mutex::new(Vec::new()),
            upload_takes: Mutex::new(Duration::ZERO),
            searchable_after: Mutex::new(Duration::ZERO),
            search_takes: Mutex::new(Duration::ZERO),
            searches_under_way: AtomicUsize::new(0),
            most_searches_at_once: AtomicUsize::new(0),
            reports_timings: AtomicBool::new(true),
            collections: Mutex::new(BTreeMap::new()),
            page_cap: AtomicUsize::new(usize::MAX),
            chat_answer: Mutex::new(Box::new(|_| "{\"found\": false}".to_string())),
            refuses_response_format: AtomicBool::new(false),
            reports_usage: AtomicBool::new(true),
            reasoning: Mutex::new(None),
        });
        let router = Router::new()
            .route("/v1/info", get(info))
            .route("/v1/embed", post(embed))
            .route("/v1/tenants/{tenant}/ingest", post(ingest))
            .route("/v1/tenants/{tenant}/sources/count", get(passage_count))
            .route("/v1/tenants/{tenant}/search", post(search))
            .route("/v1/tenants/{tenant}/eval/candidates", post(candidates))
            .route("/collections/{collection}/points/count", post(count))
            .route("/collections/{collection}/points/scroll", post(scroll))
            .route("/collections/{collection}/points", put(upsert))
            .route("/v1/chat/completions", post(chat))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("a free port");
        let url = format!("http://{}", listener.local_addr().expect("the bound address"));
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.expect("the stand-in serves");
        });
        Self { url, state, server }
    }

    /// The next `times` requests to `route` whose key contains `contains` get `action`.
    pub fn rule(&self, route: Route, contains: &str, times: usize, action: Action) {
        self.state.rules.lock().unwrap().push(Rule {
            route,
            contains: contains.to_string(),
            times,
            action,
        });
    }

    /// As [`Fake::rule`], for every such request from now on.
    pub fn always(&self, route: Route, contains: &str, action: Action) {
        self.rule(route, contains, usize::MAX, action);
    }

    pub fn clear_rules(&self) {
        self.state.rules.lock().unwrap().clear();
    }

    /// Forgets the requests seen so far.
    pub fn forget_requests(&self) {
        self.state.seen.lock().unwrap().clear();
    }

    /// The requests a route received, in order.
    pub fn seen(&self, route: Route) -> Vec<Seen> {
        self.state
            .seen
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.route == route)
            .cloned()
            .collect()
    }

    /// How many requests a route received whose key contains `contains`.
    pub fn count(&self, route: Route, contains: &str) -> usize {
        self.seen(route)
            .iter()
            .filter(|s| s.key.contains(contains))
            .count()
    }

    /// Every request received, whatever the route.
    pub fn all_seen(&self) -> Vec<Seen> {
        self.state.seen.lock().unwrap().clone()
    }

    /// Puts a document in the knowledge stand-in without an upload.
    pub fn add_document(&self, tenant: &str, collection: &str, path: &str, text: &str, attributes: Value) {
        store(
            &self.state,
            tenant,
            collection,
            path,
            text,
            attributes,
            Instant::now(),
        );
    }

    /// What `/v1/info` lists under `features`.
    pub fn set_features(&self, features: &[&str]) {
        *self.state.features.lock().unwrap() = features.iter().map(|f| f.to_string()).collect();
    }

    pub fn create_collection(&self, collection: &str) {
        self.state
            .collections
            .lock()
            .unwrap()
            .entry(collection.to_string())
            .or_default();
    }

    /// Puts a point in the index stand-in.
    pub fn add_point(&self, collection: &str, id: &str, payload: Value, vector: Value) {
        self.state
            .collections
            .lock()
            .unwrap()
            .entry(collection.to_string())
            .or_default()
            .insert(
                id.to_string(),
                FakePoint {
                    id: json!(id),
                    payload,
                    vector,
                },
            );
    }

    pub fn points(&self, collection: &str) -> Vec<FakePoint> {
        self.state
            .collections
            .lock()
            .unwrap()
            .get(collection)
            .map(|points| points.values().cloned().collect())
            .unwrap_or_default()
    }

    /// What the model answers, given the message it was sent.
    pub fn chat_answers(&self, answer: impl Fn(&str) -> String + Send + Sync + 'static) {
        *self.state.chat_answer.lock().unwrap() = Box::new(answer);
    }

    pub fn set(&self, what: &Mutex<Duration>, value: Duration) {
        *what.lock().unwrap() = value;
    }
}

/// Files a document as the service's ingest does: a second upload of its path replaces
/// the first (clean replace), and a point another document of the label holds moves to
/// this one (points are keyed by text). Answers the number of passages indexed.
fn store(
    state: &FakeState,
    tenant: &str,
    collection: &str,
    path: &str,
    text: &str,
    attributes: Value,
    uploaded_at: Instant,
) -> usize {
    let points: Vec<String> = paragraphs(text)
        .iter()
        .map(|p| point_id(tenant, collection, p))
        .collect();
    let mut documents = state.documents.lock().unwrap();
    documents.retain(|d| !(d.tenant == tenant && d.collection == collection && d.path == path));
    for other in documents
        .iter_mut()
        .filter(|d| d.tenant == tenant && d.collection == collection)
    {
        other.points.retain(|p| !points.contains(p));
    }
    let indexed = points.len();
    documents.push(Stored {
        tenant: tenant.to_string(),
        collection: collection.to_string(),
        path: path.to_string(),
        text: text.to_string(),
        attributes,
        points,
        uploaded_at,
    });
    indexed
}

impl FakeState {
    /// Records the request and answers the action a rule sets for it, if one does.
    fn receive(&self, route: Route, key: String, body: Value, headers: &HeaderMap) -> Option<Action> {
        self.seen.lock().unwrap().push(Seen {
            route,
            key: key.clone(),
            body,
            authorization: headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string),
            at: Instant::now(),
        });
        let mut rules = self.rules.lock().unwrap();
        let rule = rules
            .iter_mut()
            .find(|r| r.route == route && r.times > 0 && key.contains(&r.contains))?;
        if rule.times != usize::MAX {
            rule.times -= 1;
        }
        Some(rule.action.clone())
    }
}

fn status(code: u16) -> Response {
    let status = StatusCode::from_u16(code).expect("a status code");
    (
        status,
        Json(json!({"detail": {"error": format!("the stand-in answers {code}"), "stage": "test"}})),
    )
        .into_response()
}

fn refuse(detail: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"detail": detail}))).into_response()
}

/// Applies an action that replaces the normal answer; `None` means: answer normally
/// (after the wait, for a slow one).
async fn instead(action: Option<Action>) -> Option<Response> {
    match action? {
        Action::Status(code) => Some(status(code)),
        Action::StatusRetryAfter(code, seconds) => {
            let mut response = status(code);
            response.headers_mut().insert(
                "retry-after",
                seconds.to_string().parse().expect("a header value"),
            );
            Some(response)
        }
        Action::Garbage => Some("<html>the gateway is having a moment</html>".into_response()),
        Action::WrongShape => Some(Json(json!({"unexpected": [1, 2, 3]})).into_response()),
        Action::Slow(wait) => {
            tokio::time::sleep(wait).await;
            None
        }
        Action::SlowStatus(wait, code) => {
            tokio::time::sleep(wait).await;
            Some(status(code))
        }
        Action::Body(body) => Some(([("content-type", "application/json")], body).into_response()),
    }
}

// ---- knowledge service -----------------------------------------------------------------

async fn info(State(state): State<Arc<FakeState>>, headers: HeaderMap) -> Response {
    if let Some(response) = instead(state.receive(Route::Info, String::new(), Value::Null, &headers)).await {
        return response;
    }
    let features = state.features.lock().unwrap().clone();
    Json(json!({
        "dense_model": "fake-e5", "dense_dim": 4, "sparse_model": "fake-bm25", "reranker": "fake-reranker",
        "collection": "fake_index", "boot_id": "boot-1", "features": features,
    }))
    .into_response()
}

/// The checks the service makes of an upload's attributes (`DocumentAttributes::from_json`).
fn check_attributes(attributes: &Value, now: i64) -> Result<(), String> {
    const KNOWN: [&str; 8] = [
        "audience",
        "labels",
        "language",
        "document_ref",
        "title",
        "url",
        "written_at",
        "valid_until",
    ];
    let Some(fields) = attributes.as_object() else {
        return Err("attributes: not an object".into());
    };
    if let Some(unknown) = fields.keys().find(|k| !KNOWN.contains(&k.as_str())) {
        return Err(format!("attributes: unknown field `{unknown}`"));
    }
    let whole = |key: &str| match fields.get(key) {
        None => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("attributes: {key} is not a whole number")),
    };
    let (written, until) = (whole("written_at")?, whole("valid_until")?);
    if let Some(written) = written
        && (written <= 0 || written > now + 86_400)
    {
        return Err(
            "written_at is Unix seconds: after 1970, and at most a day after this server's clock".into(),
        );
    }
    match (written, until) {
        (None, Some(_)) => Err("valid_until needs written_at".into()),
        (Some(written), Some(until)) if until <= written => {
            Err("valid_until is later than written_at".into())
        }
        _ => Ok(()),
    }
}

async fn ingest(
    State(state): State<Arc<FakeState>>,
    Path(tenant): Path<String>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Response {
    let started = Instant::now();
    let (mut collection, mut attributes, mut file) = ("default".to_string(), None, None);
    let mut fields = Vec::new();
    while let Some(field) = multipart.next_field().await.expect("a multipart field") {
        let name = field.name().map(str::to_string);
        fields.push(name.clone().unwrap_or_default());
        match name.as_deref() {
            Some("collection") => collection = field.text().await.expect("collection"),
            Some("attributes") => attributes = Some(field.text().await.expect("attributes")),
            Some("files") => {
                let path = field.file_name().expect("a file name").to_string();
                let text =
                    String::from_utf8(field.bytes().await.expect("file bytes").to_vec()).expect("utf-8 text");
                file = Some((path, text));
            }
            _ => {}
        }
    }
    let key = format!(
        "{} {} {}",
        file.as_ref().map(|(path, _)| path.as_str()).unwrap_or_default(),
        collection,
        attributes.as_deref().unwrap_or_default()
    );
    let body = json!({
        "tenant": tenant, "collection": collection, "fields": fields,
        "path": file.as_ref().map(|(path, _)| path.clone()),
        "text": file.as_ref().map(|(_, text)| text.clone()),
        "attributes": attributes.as_deref().and_then(|raw| serde_json::from_str::<Value>(raw).ok()),
    });
    if let Some(response) = instead(state.receive(Route::Ingest, key, body, &headers)).await {
        return response;
    }
    let Some((path, text)) = file else {
        return refuse("no files uploaded");
    };
    let attributes: Value = match attributes.as_deref().map(serde_json::from_str::<Value>) {
        Some(Ok(value)) => value,
        None => json!({}),
        Some(Err(e)) => return refuse(&format!("attributes: {e}")),
    };
    if let Err(detail) = check_attributes(&attributes, now_unix()) {
        return refuse(&detail);
    }
    let indexed = store(
        &state,
        &tenant,
        &collection,
        &path,
        &text,
        attributes.clone(),
        started,
    );
    let takes = *state.upload_takes.lock().unwrap();
    tokio::time::sleep(takes).await;
    let mut source = json!({
        "source_id": format!("sha256-{:016x}", cb_types::fnv1a64(text.as_bytes())),
        "source_path": path, "chunks_indexed": indexed, "bytes_processed": text.len(), "refined": false,
        "attributes": attributes,
        "quality": {"note": "whatever the service adds"},
    });
    if state.reports_timings.load(Ordering::Relaxed) {
        source["timings_ms"] = json!({
            "parse": 0.2, "mask": 0.1, "models": 0.0, "embed": 8.0, "write": 12.4,
            "searchable": 21.3, "after": 4.0, "total": 25.3,
        });
    }
    Json(json!({"sources": [source], "jobs": []})).into_response()
}

async fn passage_count(
    State(state): State<Arc<FakeState>>,
    Path(tenant): Path<String>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> Response {
    let path = query.get("source_path").cloned().unwrap_or_default();
    let collection = query.get("collection").cloned();
    let key = format!("{path} {}", collection.as_deref().unwrap_or_default());
    if let Some(response) = instead(state.receive(Route::PassageCount, key, json!(query), &headers)).await {
        return response;
    }
    if path.is_empty() {
        return refuse("give source_path, site_id or collection");
    }
    let points: usize = state
        .documents
        .lock()
        .unwrap()
        .iter()
        .filter(|d| {
            d.tenant == tenant && d.path == path && collection.as_ref().is_none_or(|c| *c == d.collection)
        })
        .map(|d| d.points.len())
        .sum();
    Json(json!({"points": points})).into_response()
}

async fn search(
    State(state): State<Arc<FakeState>>,
    Path(tenant): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let arrived = Instant::now();
    if let Some(response) =
        instead(state.receive(Route::Search, body.to_string(), body.clone(), &headers)).await
    {
        return response;
    }
    let under_way = state.searches_under_way.fetch_add(1, Ordering::SeqCst) + 1;
    state.most_searches_at_once.fetch_max(under_way, Ordering::SeqCst);
    let query = body["query"].as_str().unwrap_or_default().to_string();
    let collections: Vec<String> =
        serde_json::from_value(body.get("collections").cloned().unwrap_or(json!([]))).unwrap_or_default();
    let top_k = body["top_k"].as_u64().unwrap_or(8) as usize;
    // Asked as of the caller's moment, else the server's clock: nothing written after it,
    // nothing ended by it.
    let as_of = body.get("as_of").and_then(Value::as_i64).unwrap_or_else(now_unix);
    let searchable_after = *state.searchable_after.lock().unwrap();
    // What the index holds is decided when the search ARRIVES; the answer comes later.
    let hits: Vec<Value> = state
        .documents
        .lock()
        .unwrap()
        .iter()
        .filter(|d| d.tenant == tenant && (collections.is_empty() || collections.contains(&d.collection)))
        .filter(|d| !d.points.is_empty() && arrived.duration_since(d.uploaded_at) >= searchable_after)
        .filter(|d| d.written_at().is_none_or(|w| w <= as_of) && d.valid_until().is_none_or(|u| u > as_of))
        // A document answers a question that names its record.
        .filter(|d| {
            d.attributes
                .get("document_ref")
                .and_then(Value::as_str)
                .is_some_and(|record| query.contains(record))
        })
        .take(top_k)
        .map(|d| {
            json!({
                "id": d.points.first(), "text": d.text, "heading": "", "source_path": d.path, "kind": "prose",
                "score": 0.9, "written_at": d.written_at(), "valid_until": d.valid_until(),
            })
        })
        .collect();
    let takes = *state.search_takes.lock().unwrap();
    tokio::time::sleep(takes).await;
    state.searches_under_way.fetch_sub(1, Ordering::SeqCst);
    Json(json!({"hits": hits, "chunks": [], "degraded": false, "mode": "reranked", "latency_ms": 1.5}))
        .into_response()
}

async fn candidates(
    State(state): State<Arc<FakeState>>,
    Path(tenant): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    if let Some(response) =
        instead(state.receive(Route::Candidates, body.to_string(), body.clone(), &headers)).await
    {
        return response;
    }
    // The route's own checks: the required fields, no field it does not take.
    const TAKES: [&str; 7] = ["query", "as_of", "n", "exact", "collections", "views", "rerank"];
    if let Some(unknown) = body
        .as_object()
        .and_then(|o| o.keys().find(|k| !TAKES.contains(&k.as_str())))
    {
        return refuse(&format!("unknown field `{unknown}`"));
    }
    let (Some(as_of), Some(limit)) = (
        body.get("as_of").and_then(Value::as_i64),
        body.get("n").and_then(Value::as_u64),
    ) else {
        return refuse("missing field `as_of` or `n`");
    };
    let collections: Vec<String> =
        serde_json::from_value(body.get("collections").cloned().unwrap_or(json!([]))).unwrap_or_default();
    let mut documents: Vec<Stored> = state
        .documents
        .lock()
        .unwrap()
        .iter()
        .filter(|d| d.tenant == tenant && (collections.is_empty() || collections.contains(&d.collection)))
        .cloned()
        .collect();
    documents.sort_by(|a, b| a.path.cmp(&b.path));

    let mut views = serde_json::Map::new();
    let mut passages = serde_json::Map::new();
    let mut rerank: Vec<(String, f64)> = Vec::new();
    for view in body["views"].as_array().cloned().unwrap_or_default() {
        let name = view["name"].as_str().unwrap_or_default().to_string();
        let upper = view
            .get("index_as_of")
            .and_then(Value::as_i64)
            .map_or(as_of, |index| index.min(as_of));
        let written_from = view.get("written_from").and_then(Value::as_i64);
        // Section 3 of the contract: `written_from < written_at <= min(as_of, index_as_of)`;
        // a passage without `written_at` is seen by every view that has no `written_from`.
        let visible: Vec<&Stored> = documents
            .iter()
            .filter(|d| match d.written_at() {
                Some(written) => written <= upper && written_from.is_none_or(|from| written > from),
                None => written_from.is_none(),
            })
            .collect();
        // First passages of every document, then second passages: a record with two
        // passages appears twice in a list, apart.
        let most = visible.iter().map(|d| d.points.len()).max().unwrap_or(0);
        let mut ordered: Vec<(&Stored, &String)> = Vec::new();
        for k in 0..most {
            for document in &visible {
                if let Some(point) = document.points.get(k) {
                    ordered.push((document, point));
                }
            }
        }
        let limit = limit as usize;
        let dense: Vec<(&Stored, &String)> = ordered.iter().take(limit).copied().collect();
        let sparse: Vec<(&Stored, &String)> = ordered.iter().rev().take(limit).copied().collect();
        for (document, point) in dense.iter().chain(&sparse) {
            passages.insert(
                (*point).clone(),
                json!({
                    "source_path": document.path, "kind": "prose", "written_at": document.written_at(),
                    "valid_until": document.valid_until(), "chars": document.text.chars().count(),
                }),
            );
        }
        if body["rerank"]["view"].as_str() == Some(name.as_str()) {
            let per_list = body["rerank"]["per_list"].as_u64().unwrap_or(32) as usize;
            let mut scored: Vec<&String> = Vec::new();
            for (_, point) in dense.iter().take(per_list).chain(sparse.iter().take(per_list)) {
                if !scored.contains(point) {
                    scored.push(point);
                }
            }
            // The union in the order asked, scored so the last scores best; the route
            // answers best first.
            rerank = scored
                .iter()
                .enumerate()
                .map(|(i, point)| ((*point).clone(), 0.2 + 0.1 * i as f64))
                .collect();
            rerank.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
        let listed = |list: &[(&Stored, &String)], top: f64, step: f64| -> Vec<Value> {
            list.iter()
                .enumerate()
                .map(|(i, (_, point))| json!({"id": point, "score": top - step * i as f64}))
                .collect()
        };
        views.insert(
            name,
            json!({"dense": listed(&dense, 0.9, 0.01), "sparse": listed(&sparse, 12.5, 0.5)}),
        );
    }
    let rerank: Vec<Value> = rerank
        .into_iter()
        .map(|(id, score)| json!({"id": id, "score": score}))
        .collect();
    Json(json!({
        "views": views, "rerank": rerank, "passages": passages,
        "timings_ms": {"embed": 7.1, "search": 9.4, "rerank": 31.0},
    }))
    .into_response()
}

async fn embed(State(state): State<Arc<FakeState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    if let Some(response) =
        instead(state.receive(Route::Embed, body.to_string(), body.clone(), &headers)).await
    {
        return response;
    }
    let texts: Vec<String> = serde_json::from_value(body["texts"].clone()).unwrap_or_default();
    // The service's own rules (`EmbedRequest::check`).
    let refusal = if body["kind"] != "query" {
        Some("kind must be \"query\"")
    } else if texts.is_empty() || texts.len() > 128 {
        Some("texts: 1 to 128 per request")
    } else if texts
        .iter()
        .any(|t| t.trim().is_empty() || t.chars().count() > 1000)
    {
        Some("each text: 1 to 1000 characters")
    } else {
        None
    };
    if let Some(detail) = refusal {
        return refuse(detail);
    }
    let vectors: Vec<Vec<f32>> = texts.iter().map(|text| vector_of(text)).collect();
    Json(json!({"model": "fake-e5#query", "dims": 4, "vectors": vectors})).into_response()
}

/// The vector the stand-in gives a text: fixed by the text, so a test can say what a row must be.
pub fn vector_of(text: &str) -> Vec<f32> {
    let hash = cb_types::fnv1a64(text.as_bytes());
    vec![
        text.len() as f32,
        (hash % 1000) as f32 / 1000.0,
        ((hash >> 20) % 1000) as f32 / -1000.0,
        0.1,
    ]
}

// ---- search index ----------------------------------------------------------------------

fn tenant_of(body: &Value) -> Option<String> {
    let condition = body["filter"]["must"].as_array()?.first()?;
    (condition["key"] == "tenant_id").then(|| condition["match"]["value"].as_str().map(str::to_string))?
}

fn no_such_collection(collection: &str) -> Response {
    let error = format!("Not found: Collection `{collection}` doesn't exist!");
    (
        StatusCode::NOT_FOUND,
        Json(json!({"status": {"error": error}, "time": 0.0})),
    )
        .into_response()
}

async fn count(
    State(state): State<Arc<FakeState>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let key = format!("{collection} {body}");
    if let Some(response) = instead(state.receive(Route::Count, key, body.clone(), &headers)).await {
        return response;
    }
    let collections = state.collections.lock().unwrap();
    let Some(points) = collections.get(&collection) else {
        return no_such_collection(&collection);
    };
    let tenant = tenant_of(&body);
    let count = points
        .values()
        .filter(|p| p.payload["tenant_id"].as_str().map(str::to_string) == tenant)
        .count();
    Json(json!({"result": {"count": count}, "status": "ok", "time": 0.001})).into_response()
}

async fn scroll(
    State(state): State<Arc<FakeState>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let key = format!("{collection} {body}");
    if let Some(response) = instead(state.receive(Route::Scroll, key, body.clone(), &headers)).await {
        return response;
    }
    let collections = state.collections.lock().unwrap();
    let Some(points) = collections.get(&collection) else {
        return no_such_collection(&collection);
    };
    let tenant = tenant_of(&body);
    let limit = (body["limit"].as_u64().unwrap_or(10) as usize).min(state.page_cap.load(Ordering::Relaxed));
    let from = body.get("offset").and_then(Value::as_str).map(str::to_string);
    let mut matching = points
        .iter()
        .filter(|(_, p)| p.payload["tenant_id"].as_str().map(str::to_string) == tenant)
        .filter(|(id, _)| from.as_ref().is_none_or(|from| *id >= from));
    let mut page = Vec::new();
    for (_, point) in matching.by_ref().take(limit) {
        let payload = match &body["with_payload"] {
            Value::Array(fields) => {
                let mut chosen = serde_json::Map::new();
                for field in fields.iter().filter_map(Value::as_str) {
                    if let Some(value) = point.payload.get(field) {
                        chosen.insert(field.to_string(), value.clone());
                    }
                }
                Value::Object(chosen)
            }
            Value::Bool(true) => point.payload.clone(),
            _ => Value::Null,
        };
        let vector = match &body["with_vector"] {
            Value::Array(names) => {
                let mut chosen = serde_json::Map::new();
                for name in names.iter().filter_map(Value::as_str) {
                    if let Some(value) = point.vector.get(name) {
                        chosen.insert(name.to_string(), value.clone());
                    }
                }
                Value::Object(chosen)
            }
            Value::Bool(true) => point.vector.clone(),
            _ => Value::Null,
        };
        page.push(json!({"id": point.id, "payload": payload, "vector": vector}));
    }
    let next = matching.next().map(|(_, point)| point.id.clone());
    Json(json!({"result": {"points": page, "next_page_offset": next}, "status": "ok", "time": 0.002}))
        .into_response()
}

async fn upsert(
    State(state): State<Arc<FakeState>>,
    Path(collection): Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let ids: Vec<String> = body["points"]
        .as_array()
        .map(|points| {
            points
                .iter()
                .map(|p| p["id"].as_str().unwrap_or_default().to_string())
                .collect()
        })
        .unwrap_or_default();
    let key = format!("{collection} {}", ids.join(" "));
    if let Some(response) = instead(state.receive(Route::Upsert, key, body.clone(), &headers)).await {
        return response;
    }
    let mut collections = state.collections.lock().unwrap();
    let Some(points) = collections.get_mut(&collection) else {
        return no_such_collection(&collection);
    };
    for point in body["points"].as_array().cloned().unwrap_or_default() {
        let id = point["id"].as_str().unwrap_or_default().to_string();
        points.insert(
            id,
            FakePoint {
                id: point["id"].clone(),
                payload: point["payload"].clone(),
                vector: point["vector"].clone(),
            },
        );
    }
    Json(json!({"result": {"operation_id": 7, "status": "completed"}, "status": "ok", "time": 0.003}))
        .into_response()
}

// ---- language model --------------------------------------------------------------------

async fn chat(State(state): State<Arc<FakeState>>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    let message = body["messages"][0]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    let action = state.receive(Route::Chat, message.clone(), body.clone(), &headers);
    if state.refuses_response_format.load(Ordering::Relaxed) && body.get("response_format").is_some() {
        let error = json!({"error": {
            "message": "Unrecognized request argument supplied: response_format",
            "type": "invalid_request_error", "param": "response_format",
        }});
        return (StatusCode::BAD_REQUEST, Json(error)).into_response();
    }
    let content = match action {
        Some(Action::Body(content)) => content,
        other => {
            if let Some(response) = instead(other).await {
                return response;
            }
            (state.chat_answer.lock().unwrap())(&message)
        }
    };
    let mut answer = json!({
        "id": "chatcmpl-1", "object": "chat.completion", "model": "fake-model-2026",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    });
    if let Some(thought) = state.reasoning.lock().unwrap().clone() {
        answer["choices"][0]["message"]["reasoning_content"] = json!(thought);
        answer["choices"][0]["message"]["reasoning"] = json!(thought);
    }
    if state.reports_usage.load(Ordering::Relaxed) {
        answer["usage"] = json!({"prompt_tokens": 321, "completion_tokens": 45, "total_tokens": 366});
    }
    Json(answer).into_response()
}
