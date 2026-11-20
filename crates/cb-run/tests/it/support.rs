//! What every command test needs: a scratch directory, the services pointed at the
//! stand-in, data sets written to disk, and run files read back.

use crate::fake::Fake;
use cb_run::commands::Shared;
use cb_run::http::{Http, HttpSettings, parse_address};
use cb_run::index::HttpSearchIndex;
use cb_run::knowledge::HttpKnowledgeService;
use cb_run::model::HttpModelEndpoint;
use cb_run::retry::RetryPolicy;
use cb_run::run::{FAILURES_FILE, MANIFEST_FILE};
use cb_types::{Gold, Manifest, Query, QueryClass, Record, Tier};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The knowledge service's token in every test. No run file may ever hold it.
pub const KB_TOKEN: &str = "kb-token-never-written";
/// The model endpoint's key in every test. No run file may ever hold it.
pub const MODEL_KEY: &str = "sk-model-key-never-written";
pub const TENANT: &str = "t-proof";
pub const LABEL: &str = "cb";

static NEXT: AtomicU64 = AtomicU64::new(0);

/// A directory of its own for one test, removed when the test ends.
pub struct Scratch {
    path: PathBuf,
}

impl Scratch {
    pub fn new(name: &str) -> Self {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("cb-run-it-{}-{n}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("a scratch directory");
        Self { path }
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Three tries, waits of a millisecond or two: the retry rule, fast.
pub fn shared() -> Shared {
    Shared {
        retry: RetryPolicy {
            max_tries: 3,
            base: Duration::from_millis(1),
            cap: Duration::from_millis(5),
        },
        machine_label: Some("test machine".into()),
    }
}

pub fn http(total_timeout: Duration) -> Http {
    Http::new(&HttpSettings {
        connect_timeout: Duration::from_secs(2),
        total_timeout,
        proxy_from_env: false,
    })
    .expect("an http client")
}

pub fn kb(fake: &Fake) -> HttpKnowledgeService {
    kb_with_timeout(fake, Duration::from_secs(5))
}

pub fn kb_with_timeout(fake: &Fake, total_timeout: Duration) -> HttpKnowledgeService {
    HttpKnowledgeService::new(
        http(total_timeout),
        parse_address("CB_KB_URL", &fake.url).expect("an address"),
        Some(KB_TOKEN.parse().expect("a token")),
    )
}

pub fn index(fake: &Fake) -> HttpSearchIndex {
    HttpSearchIndex::new(
        http(Duration::from_secs(5)),
        parse_address("CB_QDRANT_URL", &fake.url).expect("an address"),
    )
}

pub fn model(fake: &Fake, total_timeout: Duration) -> HttpModelEndpoint {
    HttpModelEndpoint::new(
        http(total_timeout),
        &parse_address("CB_MODEL_URL", &format!("{}/v1", fake.url)).expect("an address"),
        Some(MODEL_KEY.parse().expect("a key")),
    )
    .expect("a model endpoint")
}

// ---- data sets ---------------------------------------------------------------------------

pub fn record(id: &str, text: &str, written_at: i64) -> Record {
    Record {
        id: id.into(),
        text: text.into(),
        written_at,
        valid_until: None,
        thread: None,
        tier: Tier::Standing,
        title: None,
        source: "test".into(),
        truth: None,
    }
}

pub fn changing(id: &str, thread: &str, title: &str, text: &str, written_at: i64) -> Record {
    Record {
        thread: Some(thread.into()),
        tier: Tier::Changing,
        title: Some(title.into()),
        ..record(id, text, written_at)
    }
}

pub fn query(id: &str, text: &str, asked_at: i64, gold: &[&str]) -> Query {
    Query {
        id: id.into(),
        text: text.into(),
        asked_at,
        class: QueryClass::Changing,
        gold: gold
            .iter()
            .map(|record| Gold {
                record: (*record).into(),
                grade: 3,
            })
            .collect(),
        thread: None,
        harmful: vec![],
    }
}

pub fn write_lines<T: Serialize>(path: &Path, items: &[T]) {
    let mut text = String::new();
    for item in items {
        text.push_str(&serde_json::to_string(item).expect("serialisable"));
        text.push('\n');
    }
    std::fs::write(path, text).expect("a written file");
}

/// A data set directory: records, questions and a manifest.
pub fn write_data_set(dir: &Path, records: &[Record], queries: &[Query]) {
    std::fs::create_dir_all(dir).expect("a data set directory");
    write_lines(&dir.join("records.jsonl"), records);
    write_lines(&dir.join("queries.jsonl"), queries);
    let mut counts = BTreeMap::new();
    counts.insert("records".to_string(), records.len() as u64);
    counts.insert("queries".to_string(), queries.len() as u64);
    let mut sha256 = BTreeMap::new();
    for file in ["records.jsonl", "queries.jsonl"] {
        sha256.insert(
            file.to_string(),
            cb_run::atomic::sha256_of_file(&dir.join(file)).expect("a hash"),
        );
    }
    let manifest = Manifest {
        name: dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("data")
            .to_string(),
        made_by: "test".into(),
        seed: Some(1),
        counts,
        sha256,
        sources: vec![],
    };
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).expect("json"),
    )
    .expect("a manifest");
}

// ---- run files ---------------------------------------------------------------------------

/// Every non-empty line of a JSON Lines file, as JSON values; none when it is absent.
pub fn lines(path: &Path) -> Vec<Value> {
    match std::fs::read_to_string(path) {
        Ok(text) => text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("a JSON line"))
            .collect(),
        Err(_) => Vec::new(),
    }
}

pub fn failures(run: &Path) -> Vec<Value> {
    lines(&run.join(FAILURES_FILE))
}

/// The failures of a run as (id, kind, tries).
pub fn failure_summary(run: &Path) -> Vec<(String, String, u64)> {
    failures(run)
        .iter()
        .map(|f| {
            (
                f["id"].as_str().unwrap_or_default().to_string(),
                f["kind"].as_str().unwrap_or_default().to_string(),
                f["tries"].as_u64().unwrap_or_default(),
            )
        })
        .collect()
}

pub fn manifest_entries(run: &Path) -> Vec<Value> {
    let text = std::fs::read_to_string(run.join(MANIFEST_FILE)).expect("a run manifest");
    let manifest: Value = serde_json::from_str(&text).expect("JSON");
    manifest["entries"].as_array().cloned().unwrap_or_default()
}

pub fn last_entry(run: &Path) -> Value {
    manifest_entries(run).pop().expect("an entry")
}

/// Fails the test when any file under `dir` holds a token or a key.
pub fn assert_no_secret_in(dir: &Path) {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(next) = stack.pop() {
        for entry in std::fs::read_dir(&next).expect("a directory").flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let bytes = std::fs::read(&path).expect("a file");
            let text = String::from_utf8_lossy(&bytes);
            for secret in [KB_TOKEN, MODEL_KEY] {
                assert!(!text.contains(secret), "{} holds a secret", path.display());
            }
        }
    }
}
