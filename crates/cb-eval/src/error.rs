//! What can go wrong while evaluating a run.

use crate::spec::SpecError;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvalError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Contract {
        path: PathBuf,
        #[source]
        source: cb_types::ContractError,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Spec(#[from] SpecError),
    /// An input file breaks its contract. The message names the file's item.
    #[error("{0}")]
    Invalid(String),
    #[error("question {query}: the run holds no \"{view}\" candidate lists")]
    MissingView { query: String, view: String },
    #[error("question {query}: the run holds no relevance score for record {record}")]
    MissingRelevance { query: String, record: String },
    #[error("question {query}: the run holds no newest list (newest.jsonl, written by cb-run newest)")]
    MissingNewest { query: String },
    #[error("question {query}: no passage vector for record {record}")]
    MissingVector { query: String, record: String },
    #[error("method {method} needs passage vectors (vectors.f32 and vectors.json)")]
    VectorsNotLoaded { method: String },
    #[error("question {query}: method {method} gave a score that is not a number for record {record}")]
    ScoreNotFinite {
        query: String,
        method: String,
        record: String,
    },
    #[error("question {query}: method {method} listed record {record} twice")]
    ListedTwice {
        query: String,
        method: String,
        record: String,
    },
}

impl EvalError {
    pub fn invalid(message: impl Into<String>) -> Self {
        EvalError::Invalid(message.into())
    }
}
