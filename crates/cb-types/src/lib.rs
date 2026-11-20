//! The file contracts of the ContextBridge proof.
//!
//! Every tool in this repository reads and writes these shapes as JSON Lines
//! (one value per line). They are the only interface between the data tools
//! (`cb-data`), the service drivers (`cb-run`) and the evaluator (`cb-eval`),
//! so a run can be reproduced from its files alone.
//!
//! Time is Unix seconds, UTC, everywhere ([`UnixSeconds`]). A question carries
//! the moment it was asked; a record carries the moment it became known. A
//! method under test may see a record only when `written_at <= asked_at`.

#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use thiserror::Error;

pub mod ids;

/// Seconds since the Unix epoch, UTC.
pub type UnixSeconds = i64;

#[derive(Debug, Error)]
pub enum ContractError {
    #[error("line {line}: {source}")]
    Line {
        line: usize,
        #[source]
        source: serde_json::Error,
    },
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Invalid(String),
}

// ---- knowledge -----------------------------------------------------------------

/// Whether a record is expected to change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Knowledge about something in progress (an incident update, a mined
    /// record): a later record on the same matter replaces it.
    Changing,
    /// Standing documentation: true until someone edits it.
    Standing,
}

/// Where in an incident's life a record was written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Investigating,
    RootCause,
    Workaround,
    FixInProgress,
    Resolved,
}

/// What a generator knows about a record it planted. Never shown to a method
/// under test.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordTruth {
    /// False for a planted wrong statement (an early hypothesis that was not the cause).
    pub correct: bool,
    pub phase: Phase,
    /// Words that only this statement uses, for scoring an extraction against it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub key_terms: Vec<String>,
    /// When a later record on the same matter replaced this one; `None` while
    /// it is still the latest. A record is valid from `written_at` until then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub superseded_at: Option<UnixSeconds>,
}

/// One piece of knowledge the agent's search may return.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// Unique within its dataset (see [`ids::check`]).
    pub id: String,
    pub text: String,
    /// When it became known: an update's own timestamp, a document's publication.
    pub written_at: UnixSeconds,
    /// A stated end, when the source states one. A search asked at or after
    /// this moment must not return the record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<UnixSeconds>,
    /// The source's own grouping (an incident's id). `None` for standing
    /// documentation. A method may use it only in a run marked "keyed".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
    pub tier: Tier,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// The data set it comes from ("synthetic", "github-status", …).
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truth: Option<RecordTruth>,
}

// ---- questions -----------------------------------------------------------------

/// What kind of question it is. Results are always reported per class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryClass {
    /// About a matter in progress; the right answer changes while it lasts.
    Changing,
    /// About a matter that has ended.
    PostIncident,
    /// Fits more than one matter in progress.
    Ambiguous,
    /// Answered by standing documentation, on a subject no incident touches.
    Ordinary,
    /// Answered by standing documentation about the same product area as an
    /// incident in progress.
    OrdinaryNear,
}

/// How right a record is for a question.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Gold {
    pub record: String,
    /// 3 = the answer (for a changing question: the record current when it was
    /// asked); 1 = about the right matter but replaced since. Anything not
    /// listed is 0.
    pub grade: u8,
}

/// One question, with the moment it was asked.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub id: String,
    pub text: String,
    pub asked_at: UnixSeconds,
    pub class: QueryClass,
    pub gold: Vec<Gold>,
    /// The matter it is about, when it is about one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread: Option<String>,
    /// Records that would mislead if returned (planted wrong statements on the
    /// same matter). Graded 0; counted separately.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub harmful: Vec<String>,
}

// ---- conversations -------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Speaker {
    Customer,
    Agent,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    pub who: Speaker,
    pub text: String,
    /// Seconds from the start of the call to the end of this turn.
    pub ends_at_s: f32,
}

/// One call between a customer and a human support agent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    pub id: String,
    /// The incident the customer called about.
    pub thread: String,
    pub started_at: UnixSeconds,
    /// An opaque agent id; two calls by one agent are not independent.
    pub agent: String,
    pub persona: String,
    pub turns: Vec<Turn>,
    /// The records this agent states in this call (planted truth; correct or not).
    pub states: Vec<String>,
    /// Share of words altered to imitate speech-recognition errors.
    pub asr_error_rate: f32,
}

/// The extractor's self-reported confidence, as the paper defines it: the
/// product of the three parts.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Confidence {
    pub problem: f32,
    pub solution: f32,
    pub time: f32,
}

impl Confidence {
    pub fn product(self) -> f32 {
        self.problem * self.solution * self.time
    }
}

/// What a model pulled out of one transcript (the paper's Problem–Solution–Timeline triple).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Extraction {
    pub transcript: String,
    pub problem: String,
    pub solution: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_start: Option<UnixSeconds>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_eta: Option<UnixSeconds>,
    /// Times the model wrote that are not a moment ("an hour or more ago"), kept as
    /// written, field name first; each is read as "no time stated".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub times_unread: Vec<String>,
    pub confidence: Confidence,
    pub model: String,
    pub prompt_version: String,
    /// Wall time of the model call, milliseconds.
    pub latency_ms: f64,
    pub tokens_in: u32,
    pub tokens_out: u32,
}

// ---- search output -------------------------------------------------------------

/// A record with the score one ranker gave it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    pub record: String,
    pub score: f64,
}

/// Everything the search service produced for one question, before any
/// ranking rule under test is applied: the two ranked lists it merges and the
/// relevance model's score for each record in either list.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Candidates {
    pub query: String,
    /// Which records the index could see for this list ([`views`]); "now" is
    /// everything written up to `as_of`.
    pub view: String,
    /// The moment the search was asked as of (the question's `asked_at`).
    pub as_of: UnixSeconds,
    /// By meaning, best first.
    pub dense: Vec<Scored>,
    /// By keywords, best first.
    pub sparse: Vec<Scored>,
    /// The relevance model's score (0 to 1) for records in either list.
    pub rerank: Vec<Scored>,
    /// Milliseconds per stage, as the service measured them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub timings_ms: BTreeMap<String, f64>,
}

/// The newest records the index held when a question was asked, and the relevance
/// model's score of each: the recency list of the product's path (`recent` of
/// DESIGN.md section 4.4). Collected by
/// `cb-run newest` in a pass of its own beside [`Candidates`], whose lines it leaves
/// as they are.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Newest {
    pub query: String,
    /// The moment the list was taken as of (the question's `asked_at`).
    pub as_of: UnixSeconds,
    /// Newest first: the records of the index's newest passages that the `now` view
    /// could see, each once, scored with its `written_at`.
    pub newest: Vec<Scored>,
    /// The relevance model's score (0 to 1) for each record of `newest`.
    pub rerank: Vec<Scored>,
    /// Milliseconds per stage, as the service measured them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub timings_ms: BTreeMap<String, f64>,
}

/// One method's answer to one question.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Ranking {
    pub query: String,
    pub method: String,
    /// Best first.
    pub ranked: Vec<Scored>,
}

/// The views of a question a run collects (DESIGN.md section 3.1).
pub mod views {
    use crate::UnixSeconds;

    pub const NOW: &str = "now";
    /// Batch periods the paper compares against, in seconds.
    pub const BATCH_PERIODS: [i64; 3] = [300, 1800, 3600];
    /// Recency windows of the hard cut-off baseline, in seconds.
    pub const WINDOWS: [i64; 3] = [3600, 14_400, 86_400];

    pub fn batch(period_s: i64) -> String {
        format!("batch_{period_s}")
    }

    pub fn window(window_s: i64) -> String {
        format!("window_{window_s}")
    }

    /// The last moment a batch index refreshed before `asked_at`.
    pub fn batch_boundary(asked_at: UnixSeconds, period_s: i64) -> UnixSeconds {
        asked_at.div_euclid(period_s) * period_s
    }
}

/// Which part of a data set a question belongs to. Parameters are chosen on
/// validation; only test is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Validation,
    Test,
}

/// The split of a question: by its thread, so that questions about one
/// incident never straddle the two parts; a question without a thread goes by
/// its own id. One thread in five is validation.
pub fn split_of(query: &Query) -> Split {
    let key = query.thread.as_deref().unwrap_or(&query.id);
    if fnv1a64(key.as_bytes()).is_multiple_of(5) {
        Split::Validation
    } else {
        Split::Test
    }
}

/// FNV-1a, 64 bit: a fixed, dependency-free hash so the split never changes.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---- timing --------------------------------------------------------------------

/// One record's trip from speech to search, each mark in milliseconds after the
/// end of the utterance that stated it. A mark the run did not take is absent.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PipelineTrip {
    pub record: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub masked_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extracted_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub written_ms: Option<f64>,
    /// The first moment the search service returned the record for a question about it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_hit_ms: Option<f64>,
}

// ---- data set manifest ---------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub name: String,
    /// Tool and version that wrote the files.
    pub made_by: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub counts: BTreeMap<String, u64>,
    /// File name → sha256 hex.
    pub sha256: BTreeMap<String, String>,
    /// Where the source material came from and under what terms.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
}

// ---- JSON Lines ----------------------------------------------------------------

/// Read one value per non-empty line.
pub fn read_jsonl<T: DeserializeOwned>(reader: impl BufRead) -> Result<Vec<T>, ContractError> {
    let mut out = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).map_err(|source| ContractError::Line { line: i + 1, source })?);
    }
    Ok(out)
}

/// Write one value per line.
pub fn write_jsonl<T: Serialize>(mut writer: impl Write, items: &[T]) -> Result<(), ContractError> {
    for item in items {
        serde_json::to_writer(&mut writer, item).map_err(|source| ContractError::Line { line: 0, source })?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_round_trips_and_leaves_absent_fields_out() {
        let r = Record {
            id: "r-1".into(),
            text: "The outage is fixed.".into(),
            written_at: 1_700_000_000,
            valid_until: None,
            thread: Some("inc-1".into()),
            tier: Tier::Changing,
            title: None,
            source: "synthetic".into(),
            truth: None,
        };
        let line = serde_json::to_string(&r).unwrap_or_default();
        assert!(!line.contains("valid_until"));
        assert!(line.contains("\"tier\":\"changing\""));
        let back: Result<Record, _> = serde_json::from_str(&line);
        assert_eq!(back.ok(), Some(r));
    }

    #[test]
    fn jsonl_reads_what_it_wrote_and_names_a_bad_line() {
        let items = vec![
            Scored {
                record: "a".into(),
                score: 0.5,
            },
            Scored {
                record: "b".into(),
                score: 0.25,
            },
        ];
        let mut buf = Vec::new();
        assert!(write_jsonl(&mut buf, &items).is_ok());
        let back: Result<Vec<Scored>, _> = read_jsonl(buf.as_slice());
        assert_eq!(back.ok(), Some(items));
        let bad: Result<Vec<Scored>, _> =
            read_jsonl("{\"record\":\"a\",\"score\":1}\n\nnot json\n".as_bytes());
        assert!(matches!(bad, Err(ContractError::Line { line: 3, .. })));
    }

    #[test]
    fn the_split_follows_the_thread_and_the_hash_is_fixed() {
        // FNV-1a test vectors.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        let q = |id: &str, thread: Option<&str>| Query {
            id: id.into(),
            text: String::new(),
            asked_at: 0,
            class: QueryClass::Changing,
            gold: vec![],
            thread: thread.map(str::to_string),
            harmful: vec![],
        };
        assert_eq!(
            split_of(&q("q-1", Some("inc-7"))),
            split_of(&q("q-2", Some("inc-7")))
        );
        assert_eq!(views::batch_boundary(3_725, 3_600), 3_600);
        assert_eq!(views::batch_boundary(-1, 300), -300);
        assert_eq!(views::batch(300), "batch_300");
    }

    #[test]
    fn confidence_is_the_product_of_its_parts() {
        let c = Confidence {
            problem: 0.9,
            solution: 0.8,
            time: 0.5,
        };
        assert!((c.product() - 0.36).abs() < 1e-6);
    }
}
