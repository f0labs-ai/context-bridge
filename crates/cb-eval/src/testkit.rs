//! Small builders shared by the unit tests.

use crate::context::{Graded, PassageVectors, QuestionContext};
use crate::error::EvalError;
use crate::methods;
use crate::records::RecordTable;
use crate::spec::MethodSpec;
use crate::vectors::{VectorIndex, Vectors};
use cb_types::{Candidates, Gold, Newest, Query, QueryClass, Record, RecordTruth, Scored, Tier};
use std::collections::BTreeMap;

/// A changing record without a thread.
pub(crate) fn record(id: &str, written_at: i64) -> Record {
    Record {
        id: id.to_string(),
        text: String::new(),
        written_at,
        valid_until: None,
        thread: None,
        tier: Tier::Changing,
        title: None,
        source: "test".to_string(),
        truth: None,
    }
}

/// A changing record of an incident.
pub(crate) fn threaded(id: &str, written_at: i64, thread: &str) -> Record {
    Record {
        thread: Some(thread.to_string()),
        ..record(id, written_at)
    }
}

/// A piece of standing documentation.
pub(crate) fn standing(id: &str, written_at: i64) -> Record {
    Record {
        tier: Tier::Standing,
        ..record(id, written_at)
    }
}

/// A record with a stated end.
pub(crate) fn ending(id: &str, written_at: i64, valid_until: i64) -> Record {
    Record {
        valid_until: Some(valid_until),
        ..record(id, written_at)
    }
}

pub(crate) fn scored(pairs: &[(&str, f64)]) -> Vec<Scored> {
    pairs
        .iter()
        .map(|(record, score)| Scored {
            record: record.to_string(),
            score: *score,
        })
        .collect()
}

pub(crate) fn candidates(
    query: &str,
    view: &str,
    as_of: i64,
    dense: &[Scored],
    sparse: &[Scored],
    rerank: &[Scored],
) -> Candidates {
    Candidates {
        query: query.to_string(),
        view: view.to_string(),
        as_of,
        dense: dense.to_vec(),
        sparse: sparse.to_vec(),
        rerank: rerank.to_vec(),
        timings_ms: BTreeMap::new(),
    }
}

pub(crate) fn question(
    id: &str,
    asked_at: i64,
    class: QueryClass,
    gold: &[(&str, u8)],
    thread: Option<&str>,
) -> Query {
    Query {
        id: id.to_string(),
        text: String::new(),
        asked_at,
        class,
        gold: gold
            .iter()
            .map(|(record, grade)| Gold {
                record: record.to_string(),
                grade: *grade,
            })
            .collect(),
        thread: thread.map(str::to_string),
        harmful: Vec::new(),
    }
}

/// A vector length and the vector of each named record.
pub(crate) type TestVectors = (usize, Vec<(String, Vec<f32>)>);

/// One question with its records and candidate lists, ready to be ranked.
pub(crate) struct Scene {
    pub records: Vec<Record>,
    pub query: Query,
    pub lines: Vec<Candidates>,
    pub newest: Option<Newest>,
    pub vectors: Option<TestVectors>,
}

impl Scene {
    pub fn new(asked_at: i64, records: Vec<Record>) -> Self {
        Self {
            records,
            query: question("q", asked_at, QueryClass::Changing, &[], None),
            lines: Vec::new(),
            newest: None,
            vectors: None,
        }
    }

    /// Set the newest list (newest first; each record is scored with its own
    /// `written_at`) and the relevance scores of its records.
    pub fn newest(mut self, records: &[&str], rerank: &[(&str, f64)]) -> Self {
        let newest: Vec<Scored> = records
            .iter()
            .map(|id| {
                let record = self
                    .records
                    .iter()
                    .find(|r| r.id == *id)
                    .expect("a record of the scene");
                Scored {
                    record: id.to_string(),
                    score: record.written_at as f64,
                }
            })
            .collect();
        self.newest = Some(Newest {
            query: "q".to_string(),
            as_of: self.query.asked_at,
            newest,
            rerank: scored(rerank),
            timings_ms: BTreeMap::new(),
        });
        self
    }

    /// Add the two lists of a view.
    pub fn view(mut self, view: &str, dense: &[(&str, f64)], sparse: &[(&str, f64)]) -> Self {
        self.lines.push(candidates(
            "q",
            view,
            self.query.asked_at,
            &scored(dense),
            &scored(sparse),
            &[],
        ));
        self
    }

    /// Add the `now` view.
    pub fn now(self, dense: &[(&str, f64)], sparse: &[(&str, f64)]) -> Self {
        self.view("now", dense, sparse)
    }

    /// Set the relevance scores of the `now` view.
    pub fn relevance(mut self, rerank: &[(&str, f64)]) -> Self {
        let now = self
            .lines
            .iter_mut()
            .find(|line| line.view == "now")
            .expect("add the now view first");
        now.rerank = scored(rerank);
        self
    }

    pub fn gold(mut self, gold: &[(&str, u8)]) -> Self {
        self.query.gold = question("q", 0, QueryClass::Changing, gold, None).gold;
        self
    }

    pub fn vectors(mut self, dim: usize, rows: &[(&str, &[f32])]) -> Self {
        self.vectors = Some((
            dim,
            rows.iter().map(|(id, v)| (id.to_string(), v.to_vec())).collect(),
        ));
        self
    }

    /// Rank with the method written in its text form; ids and scores, best first.
    pub fn rank(&self, spec: &str) -> Result<Vec<(String, f64)>, EvalError> {
        let spec: MethodSpec = spec.parse()?;
        let table = RecordTable::from_records(&self.records)?;
        let gold: Vec<Graded> = self
            .query
            .gold
            .iter()
            .filter_map(|g| {
                Some(Graded {
                    record: table.index_of(&g.record)?,
                    grade: g.grade,
                })
            })
            .collect();
        let vectors = match &self.vectors {
            Some((dim, rows)) => {
                let index = VectorIndex {
                    dim: *dim,
                    ids: rows.iter().map(|(id, _)| id.clone()).collect(),
                };
                let data = rows.iter().flat_map(|(_, v)| v.iter().copied()).collect();
                Some(PassageVectors::new(Vectors::new(index, data)?, &table)?)
            }
            None => None,
        };
        let lines: Vec<&Candidates> = self.lines.iter().collect();
        let ctx = QuestionContext::new(
            &self.query,
            &lines,
            self.newest.as_ref(),
            &table,
            vectors.as_ref(),
            &gold,
        )?;
        let ranked = methods::rank(&spec, &ctx)?;
        Ok(ranked
            .iter()
            .map(|r| (table.id(r.record).to_string(), r.score))
            .collect())
    }

    /// The ids of [`Scene::rank`], best first.
    pub fn order(&self, spec: &str) -> Vec<String> {
        self.rank(spec)
            .expect("the method ranks this scene")
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }
}

/// The planted truth of a record.
pub(crate) fn truth(correct: bool, key_terms: &[&str], superseded_at: Option<i64>) -> Option<RecordTruth> {
    Some(RecordTruth {
        correct,
        phase: cb_types::Phase::Investigating,
        key_terms: key_terms.iter().map(|t| t.to_string()).collect(),
        superseded_at,
    })
}
