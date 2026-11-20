//! What can go wrong in a run. Every error names the address, the id, the setting or the
//! file it concerns, so a line in `failures.jsonl` can always be traced to one thing.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// The longest piece of a service's answer kept with an error, in bytes.
pub const BODY_EXCERPT_BYTES: usize = 2_000;

/// Why an item is in `failures.jsonl`, in one word. Logs carry this word and never an
/// error's text, which may quote what a service sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// No answer within the time limit.
    Timeout,
    /// The request did not get through, or the answer broke off.
    Transport,
    /// The service answered with a status that is not a success.
    HttpStatus,
    /// A success whose body is not the contract's shape.
    MalformedAnswer,
    /// A well-formed answer that cannot be used.
    Contract,
    /// The model answered, but not with the asked JSON.
    AnswerNotAsked,
    /// The item itself cannot be processed.
    Input,
    /// A record's text is the same as an earlier record's, and the index keys a passage by its text.
    DuplicateText,
    /// The index holds another number of a record's passages than its upload indexed.
    NotWholeInIndex,
    /// A record was searchable before its upload started, so its trip cannot be timed.
    AlreadySearchable,
    /// A record was not found by the search within the time allowed.
    NeverFound,
    /// Fewer items exist than were asked for.
    Shortfall,
    /// A write this tool refused to make.
    Refused,
    /// A file could not be read or written.
    File,
    /// A flag or environment variable is missing or cannot be used.
    Setting,
}

impl FailureKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::HttpStatus => "http_status",
            Self::MalformedAnswer => "malformed_answer",
            Self::Contract => "contract",
            Self::AnswerNotAsked => "answer_not_asked",
            Self::Input => "input",
            Self::DuplicateText => "duplicate_text",
            Self::NotWholeInIndex => "not_whole_in_index",
            Self::AlreadySearchable => "already_searchable",
            Self::NeverFound => "never_found",
            Self::Shortfall => "shortfall",
            Self::Refused => "refused",
            Self::File => "file",
            Self::Setting => "setting",
        }
    }
}

#[derive(Debug, Error)]
pub enum RunError {
    /// A flag or environment variable is missing or cannot be used.
    #[error("{name}: {reason}")]
    Setting { name: &'static str, reason: String },

    /// The service did not answer in time (connecting, or the whole request).
    #[error("{url}: no answer within {limit_ms} ms")]
    Timeout { url: String, limit_ms: u64 },

    /// The request did not get through, or the answer broke off.
    #[error("{url}: the request did not get through: {reason}")]
    Transport { url: String, reason: String },

    /// The service answered with a status that is not a success.
    #[error("{url}: answered {status}")]
    Status {
        url: String,
        status: u16,
        /// The start of the answer's body.
        body: String,
        /// What the service asked for in `Retry-After`, in seconds.
        retry_after_s: Option<u64>,
    },

    /// A success whose body is not the shape the contract promises.
    #[error("{url}: the answer does not keep the contract: {reason}")]
    Malformed {
        url: String,
        reason: String,
        /// The start of the answer's body.
        body: String,
    },

    /// A well-formed answer that cannot be used for this item (a missing view, an
    /// upload the service queued instead of indexing, a time that differs from the data set).
    #[error("{id}: the service's answer cannot be used: {reason}")]
    Contract { id: String, url: String, reason: String },

    /// The model answered, but not with the JSON it was asked for. The answer is kept whole.
    #[error("{id}: the model did not answer with the asked JSON: {reason}")]
    AnswerNotAsked {
        id: String,
        url: String,
        reason: String,
        raw: String,
    },

    /// An input item that cannot be processed as it is.
    #[error("{id}: {reason}")]
    Item { id: String, reason: String },

    /// A write that this tool refuses to make.
    #[error("{url}: refusing to write: {reason}")]
    Refused { url: String, reason: String },

    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("{path} line {line}: {reason}")]
    Line {
        path: PathBuf,
        line: usize,
        reason: String,
    },

    #[error("{path}: {reason}")]
    File { path: PathBuf, reason: String },
}

impl RunError {
    /// Whether trying the same call again can help: no answer, a broken answer, "too many
    /// requests" or a server-side failure. Any other refusal (a 4xx) is final.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Timeout { .. }
            | Self::Transport { .. }
            | Self::Malformed { .. }
            | Self::AnswerNotAsked { .. } => true,
            Self::Status { status, .. } => *status == 429 || (500..=599).contains(status),
            Self::Setting { .. }
            | Self::Contract { .. }
            | Self::Item { .. }
            | Self::Refused { .. }
            | Self::Io { .. }
            | Self::Line { .. }
            | Self::File { .. } => false,
        }
    }

    /// The error in one word, for `failures.jsonl` and for logs.
    pub fn kind(&self) -> FailureKind {
        match self {
            Self::Setting { .. } => FailureKind::Setting,
            Self::Timeout { .. } => FailureKind::Timeout,
            Self::Transport { .. } => FailureKind::Transport,
            Self::Status { .. } => FailureKind::HttpStatus,
            Self::Malformed { .. } => FailureKind::MalformedAnswer,
            Self::Contract { .. } => FailureKind::Contract,
            Self::AnswerNotAsked { .. } => FailureKind::AnswerNotAsked,
            Self::Item { .. } => FailureKind::Input,
            Self::Refused { .. } => FailureKind::Refused,
            Self::Io { .. } | Self::Line { .. } | Self::File { .. } => FailureKind::File,
        }
    }

    /// The address the error concerns, when it concerns one.
    pub fn url(&self) -> Option<&str> {
        match self {
            Self::Timeout { url, .. }
            | Self::Transport { url, .. }
            | Self::Status { url, .. }
            | Self::Malformed { url, .. }
            | Self::Contract { url, .. }
            | Self::AnswerNotAsked { url, .. }
            | Self::Refused { url, .. } => Some(url),
            Self::Setting { .. }
            | Self::Item { .. }
            | Self::Io { .. }
            | Self::Line { .. }
            | Self::File { .. } => None,
        }
    }

    /// The status the service answered with, when it answered.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Status { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// What the service or the model sent back, when something came back.
    pub fn raw_answer(&self) -> Option<&str> {
        match self {
            Self::Status { body, .. } | Self::Malformed { body, .. } if !body.is_empty() => Some(body),
            Self::AnswerNotAsked { raw, .. } => Some(raw),
            _ => None,
        }
    }

    /// How long the service asked to be left alone, when it said.
    pub fn retry_after_s(&self) -> Option<u64> {
        match self {
            Self::Status { retry_after_s, .. } => *retry_after_s,
            _ => None,
        }
    }

    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }

    pub fn file(path: impl Into<PathBuf>, reason: impl Into<String>) -> Self {
        Self::File {
            path: path.into(),
            reason: reason.into(),
        }
    }

    pub fn item(id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::Item {
            id: id.into(),
            reason: reason.into(),
        }
    }
}

/// The start of a body as text, cut at a character boundary.
pub fn excerpt(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(body);
    if text.len() <= BODY_EXCERPT_BYTES {
        return text.into_owned();
    }
    let mut end = BODY_EXCERPT_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(code: u16) -> RunError {
        RunError::Status {
            url: "http://kb/v1".into(),
            status: code,
            body: String::new(),
            retry_after_s: None,
        }
    }

    #[test]
    fn only_no_answer_too_many_requests_and_server_failures_are_tried_again() {
        assert!(status(429).is_retryable());
        assert!(status(500).is_retryable());
        assert!(status(503).is_retryable());
        for refused in [400, 401, 404, 408, 409, 413, 422] {
            assert!(!status(refused).is_retryable(), "{refused}");
        }
        assert!(!status(302).is_retryable());
        assert!(
            RunError::Timeout {
                url: "u".into(),
                limit_ms: 1
            }
            .is_retryable()
        );
        assert!(!RunError::item("r-1", "bad id").is_retryable());
    }

    #[test]
    fn a_long_body_is_cut_at_a_character_boundary() {
        let body = "é".repeat(BODY_EXCERPT_BYTES);
        let cut = excerpt(body.as_bytes());
        assert!(cut.len() <= BODY_EXCERPT_BYTES);
        assert!(cut.chars().all(|c| c == 'é'));
        assert_eq!(excerpt(b"short"), "short");
    }
}
