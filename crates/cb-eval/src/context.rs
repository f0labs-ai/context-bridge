//! What one question brings to a ranking method: the moment it was asked, its
//! candidate lists per view, its newest list when the run has one, the record
//! table and (for one method) passage vectors.
//!
//! Building a [`QuestionContext`] checks the run files against their contract,
//! so a method never sees a record the index could not have held, a list that
//! is not best-first, or a score that is not a number.

use crate::error::EvalError;
use crate::records::{RecordIdx, RecordTable};
use crate::vectors::Vectors;
use cb_types::{Candidates, Newest, Query, Scored, UnixSeconds, views};
use std::collections::{HashMap, HashSet};

/// Which records an index could see (DESIGN.md section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum View {
    /// Everything written up to the moment the question was asked.
    Now,
    /// An index that refreshes every so many seconds.
    Batch(i64),
    /// Only records written within the last so many seconds.
    Window(i64),
}

impl View {
    /// The name the run file uses, e.g. `batch_300`.
    pub fn name(self) -> String {
        match self {
            View::Now => views::NOW.to_string(),
            View::Batch(period) => views::batch(period),
            View::Window(window) => views::window(window),
        }
    }

    pub fn parse(name: &str) -> Option<View> {
        if name == views::NOW {
            return Some(View::Now);
        }
        let seconds = |text: &str| text.parse::<i64>().ok().filter(|s| *s > 0);
        if let Some(period) = name.strip_prefix("batch_") {
            return seconds(period).map(View::Batch);
        }
        name.strip_prefix("window_").and_then(seconds).map(View::Window)
    }

    /// Whether this view may hold a record written at `written_at` for a question asked at `asked_at`.
    pub fn sees(self, written_at: UnixSeconds, asked_at: UnixSeconds) -> bool {
        match self {
            View::Now => written_at <= asked_at,
            View::Batch(period) => written_at <= views::batch_boundary(asked_at, period),
            View::Window(window) => written_at <= asked_at && written_at > asked_at.saturating_sub(window),
        }
    }
}

/// A record in a candidate list with the score its ranker gave it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Listed {
    pub record: RecordIdx,
    pub score: f64,
    /// Its place in the list as the service returned it, from 0, before records with
    /// equal scores were put in order. What the service cut its list at (the first 64
    /// of each list reach its relevance model) is counted in these places.
    pub served: usize,
}

/// The two ranked lists of one view, each best first as the service returned it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ViewLists {
    /// By meaning, with cosine scores.
    pub dense: Vec<Listed>,
    /// By keywords, with BM25 scores.
    pub sparse: Vec<Listed>,
}

/// A graded answer of a question. Only the oracle reads these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Graded {
    pub record: RecordIdx,
    pub grade: u8,
}

/// Passage vectors keyed by record.
#[derive(Debug)]
pub struct PassageVectors {
    vectors: Vectors,
    row_of_record: Vec<Option<usize>>,
}

impl PassageVectors {
    /// Tie each vector to its record. A vector of an unknown record is a contract error.
    pub fn new(vectors: Vectors, records: &RecordTable) -> Result<Self, EvalError> {
        let mut row_of_record = vec![None; records.len()];
        for (id, row) in vectors.rows_by_id()? {
            let record = records.index_of(id).ok_or_else(|| {
                EvalError::invalid(format!("vectors: {id:?} is not a record of the data set"))
            })?;
            row_of_record[record.row()] = Some(row);
        }
        Ok(Self {
            vectors,
            row_of_record,
        })
    }

    /// `cos` of two records' passages; `Err(record)` names the one without a vector.
    pub fn cosine(&self, a: RecordIdx, b: RecordIdx) -> Result<f64, RecordIdx> {
        let row_a = self.row_of_record[a.row()].ok_or(a)?;
        let row_b = self.row_of_record[b.row()].ok_or(b)?;
        Ok(self.vectors.cosine(row_a, row_b))
    }
}

/// Everything a method may read for one question.
#[derive(Debug)]
pub struct QuestionContext<'a> {
    query: &'a str,
    asked_at: UnixSeconds,
    records: &'a RecordTable,
    views: Vec<(View, ViewLists)>,
    relevance: HashMap<RecordIdx, f64>,
    /// The newest records the `now` view could show, newest first (then by id),
    /// each scored with its `written_at`; None when the run has no newest list.
    newest: Option<Vec<Listed>>,
    /// The relevance model's scores of the newest pass, apart from `relevance`:
    /// a method that does not read the newest list never sees them.
    newest_relevance: HashMap<RecordIdx, f64>,
    vectors: Option<&'a PassageVectors>,
    gold: &'a [Graded],
}

impl<'a> QuestionContext<'a> {
    /// Check the question's candidate lines against the contract and resolve their records.
    ///
    /// Refused: a line of another question, an unknown or repeated view, an
    /// `as_of` that is not the question's `asked_at`, a record that is not in
    /// the data set, a record listed twice in one list, a record the view
    /// could not have seen, a score that is not a number, a list whose scores
    /// rise, and a relevance score outside 0 to 1.
    ///
    /// Within a list, records with exactly the same score are put newer
    /// first, then by id, so no method depends on the order in which the
    /// service happened to return a tie.
    ///
    /// `newest` is the question's line of `newest.jsonl`, when the run has one;
    /// `newest_list` below says what it must hold.
    pub fn new(
        query: &'a Query,
        candidates: &[&Candidates],
        newest: Option<&Newest>,
        records: &'a RecordTable,
        vectors: Option<&'a PassageVectors>,
        gold: &'a [Graded],
    ) -> Result<Self, EvalError> {
        let mut resolved: Vec<(View, ViewLists)> = Vec::with_capacity(candidates.len());
        let mut relevance = HashMap::new();
        for line in candidates {
            let wrong =
                |what: String| EvalError::invalid(format!("candidates of question {}: {what}", query.id));
            if line.query != query.id {
                return Err(wrong(format!("a line of question {} was passed", line.query)));
            }
            let view =
                View::parse(&line.view).ok_or_else(|| wrong(format!("unknown view {:?}", line.view)))?;
            if line.as_of != query.asked_at {
                return Err(wrong(format!(
                    "view {} was asked as of {}, the question at {}",
                    line.view, line.as_of, query.asked_at
                )));
            }
            if resolved.iter().any(|(seen, _)| *seen == view) {
                return Err(wrong(format!("view {} appears twice", line.view)));
            }
            let list = |which: &str, list: &[Scored]| -> Result<Vec<Listed>, EvalError> {
                let mut seen = HashSet::with_capacity(list.len());
                let mut out: Vec<Listed> = Vec::with_capacity(list.len());
                for item in list {
                    let at = format!("view {} {which} list, record {}", line.view, item.record);
                    let record = records
                        .index_of(&item.record)
                        .ok_or_else(|| wrong(format!("{at}: not a record of the data set")))?;
                    if !item.score.is_finite() {
                        return Err(wrong(format!("{at}: its score is not a number")));
                    }
                    if !seen.insert(record) {
                        return Err(wrong(format!("{at}: listed twice")));
                    }
                    if !view.sees(records.written_at(record), query.asked_at) {
                        return Err(wrong(format!(
                            "{at}: written at {}, which this view could not see",
                            records.written_at(record)
                        )));
                    }
                    if out.last().is_some_and(|previous| item.score > previous.score) {
                        return Err(wrong(format!("{at}: scored above the record before it")));
                    }
                    out.push(Listed {
                        record,
                        score: item.score,
                        served: out.len(),
                    });
                }
                // Records the service scored exactly alike go newer first, then by id
                // (DESIGN.md section 4); records with different scores keep the service's order.
                out.sort_by(|a, b| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| records.cmp_newer_then_id(a.record, b.record))
                });
                Ok(out)
            };
            let lists = ViewLists {
                dense: list("dense", &line.dense)?,
                sparse: list("sparse", &line.sparse)?,
            };
            if view == View::Now {
                for item in &line.rerank {
                    let at = format!("relevance score of record {}", item.record);
                    let record = records
                        .index_of(&item.record)
                        .ok_or_else(|| wrong(format!("{at}: not a record of the data set")))?;
                    if !(0.0..=1.0).contains(&item.score) {
                        return Err(wrong(format!("{at}: {} is not between 0 and 1", item.score)));
                    }
                    if relevance.insert(record, item.score).is_some() {
                        return Err(wrong(format!("{at}: given twice")));
                    }
                }
            }
            resolved.push((view, lists));
        }
        resolved.sort_by_key(|(view, _)| *view);
        let (newest, newest_relevance) = match newest {
            Some(line) => {
                let (list, scores) = newest_list(query, line, records)?;
                (Some(list), scores)
            }
            None => (None, HashMap::new()),
        };
        Ok(Self {
            query: &query.id,
            asked_at: query.asked_at,
            records,
            views: resolved,
            relevance,
            newest,
            newest_relevance,
            vectors,
            gold,
        })
    }

    pub fn query(&self) -> &str {
        self.query
    }

    /// The moment the question was asked.
    pub fn asked_at(&self) -> UnixSeconds {
        self.asked_at
    }

    pub fn records(&self) -> &'a RecordTable {
        self.records
    }

    /// The lists of one view, or an error naming the view the run did not collect.
    pub fn view(&self, view: View) -> Result<&ViewLists, EvalError> {
        self.views
            .iter()
            .find(|(held, _)| *held == view)
            .map(|(_, lists)| lists)
            .ok_or_else(|| EvalError::MissingView {
                query: self.query.to_string(),
                view: view.name(),
            })
    }

    /// The lists of the `now` view: `D` and `S` of DESIGN.md section 4.
    pub fn now(&self) -> Result<&ViewLists, EvalError> {
        self.view(View::Now)
    }

    /// `age(r) = asked_at - written_at(r)`, seconds.
    pub fn age_of(&self, record: RecordIdx) -> i64 {
        self.asked_at.saturating_sub(self.records.written_at(record))
    }

    /// Whether the candidate lists' relevance pass scored the record.
    pub fn has_relevance(&self, record: RecordIdx) -> bool {
        self.relevance.contains_key(&record)
    }

    /// How many records the service listed in the first `places` of the `now` view's
    /// two lists that the relevance pass did not score. The product's methods leave them
    /// out; a few come from equal scores across the cut or from records of several
    /// passages, and many would mean a broken run, so `rank` reports the count.
    pub fn unscored_in_first(&self, places: usize) -> Result<usize, EvalError> {
        let now = self.now()?;
        let unscored: HashSet<RecordIdx> = now
            .dense
            .iter()
            .chain(&now.sparse)
            .filter(|l| l.served < places && !self.has_relevance(l.record))
            .map(|l| l.record)
            .collect();
        Ok(unscored.len())
    }

    /// `rho(r)`: the relevance model's score. A record the run did not score is an error, never 0.
    pub fn relevance_of(&self, record: RecordIdx) -> Result<f64, EvalError> {
        self.relevance
            .get(&record)
            .copied()
            .ok_or_else(|| EvalError::MissingRelevance {
                query: self.query.to_string(),
                record: self.records.id(record).to_string(),
            })
    }

    /// `rho(r)` over the candidate lists and the newest list together: the higher of the
    /// two passes' scores. A record's score is that of its best-scored passage, and the
    /// two passes may have scored different passages of one record; a pool that holds
    /// both lists holds both passages. Only a method that reads the newest list asks this.
    pub fn relevance_with_newest(&self, record: RecordIdx) -> Result<f64, EvalError> {
        let found = [self.relevance.get(&record), self.newest_relevance.get(&record)]
            .into_iter()
            .flatten()
            .copied()
            .reduce(f64::max);
        found.ok_or_else(|| EvalError::MissingRelevance {
            query: self.query.to_string(),
            record: self.records.id(record).to_string(),
        })
    }

    /// The newest records the `now` view could show, newest first, or an error naming
    /// the question when the run holds no newest list.
    pub fn newest(&self) -> Result<&[Listed], EvalError> {
        self.newest.as_deref().ok_or_else(|| EvalError::MissingNewest {
            query: self.query.to_string(),
        })
    }

    /// `cos(passage, passage)` of two records, for the method named `method`.
    pub fn passage_cosine(&self, a: RecordIdx, b: RecordIdx, method: &str) -> Result<f64, EvalError> {
        let vectors = self.vectors.ok_or_else(|| EvalError::VectorsNotLoaded {
            method: method.to_string(),
        })?;
        vectors.cosine(a, b).map_err(|missing| EvalError::MissingVector {
            query: self.query.to_string(),
            record: self.records.id(missing).to_string(),
        })
    }

    /// **The oracle only.** The question's graded answers.
    pub fn gold_for_oracle(&self) -> &[Graded] {
        self.gold
    }
}

/// A question's newest list, checked against its contract, and its relevance scores.
///
/// Refused: a line of another question or taken as of another moment, a record that is
/// not in the data set, listed twice, written after the question, scored with anything
/// but its `written_at`, or newer than the record before it; a relevance score for a
/// record the list does not hold, outside 0 to 1, or given twice. Records of one second
/// are put in id order, as the other lists' ties are.
fn newest_list(
    query: &Query,
    line: &Newest,
    records: &RecordTable,
) -> Result<(Vec<Listed>, HashMap<RecordIdx, f64>), EvalError> {
    let wrong = |what: String| EvalError::invalid(format!("newest list of question {}: {what}", query.id));
    if line.query != query.id {
        return Err(wrong(format!("a line of question {} was passed", line.query)));
    }
    if line.as_of != query.asked_at {
        return Err(wrong(format!(
            "taken as of {}, the question asked at {}",
            line.as_of, query.asked_at
        )));
    }
    let resolve = |at: &str, id: &str| {
        records
            .index_of(id)
            .ok_or_else(|| wrong(format!("{at}: not a record of the data set")))
    };
    let mut list: Vec<Listed> = Vec::with_capacity(line.newest.len());
    let mut listed = HashSet::with_capacity(line.newest.len());
    for item in &line.newest {
        let at = format!("record {}", item.record);
        let record = resolve(&at, &item.record)?;
        if !listed.insert(record) {
            return Err(wrong(format!("{at}: listed twice")));
        }
        let written_at = records.written_at(record);
        if !View::Now.sees(written_at, query.asked_at) {
            return Err(wrong(format!(
                "{at}: written at {written_at}, after the question"
            )));
        }
        if item.score != written_at as f64 {
            return Err(wrong(format!(
                "{at}: scored {}, not its written_at {written_at}",
                item.score
            )));
        }
        if list
            .last()
            .is_some_and(|previous| records.written_at(previous.record) < written_at)
        {
            return Err(wrong(format!("{at}: newer than the record before it")));
        }
        list.push(Listed {
            record,
            score: item.score,
            served: list.len(),
        });
    }
    list.sort_by(|a, b| records.cmp_newer_then_id(a.record, b.record));
    let mut relevance = HashMap::with_capacity(line.rerank.len());
    for item in &line.rerank {
        let at = format!("relevance score of record {}", item.record);
        let record = resolve(&at, &item.record)?;
        if !listed.contains(&record) {
            return Err(wrong(format!("{at}: the newest list does not hold it")));
        }
        if !(0.0..=1.0).contains(&item.score) {
            return Err(wrong(format!("{at}: {} is not between 0 and 1", item.score)));
        }
        if relevance.insert(record, item.score).is_some() {
            return Err(wrong(format!("{at}: given twice")));
        }
    }
    Ok((list, relevance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{candidates, question, record, scored};
    use cb_types::QueryClass;

    fn table() -> RecordTable {
        RecordTable::from_records(&[record("a", 100), record("b", 200), record("c", 3_700)]).unwrap()
    }

    fn refusal(lines: &[Candidates]) -> String {
        let table = table();
        let query = question("q", 4_000, QueryClass::Changing, &[("a", 3)], None);
        let lines: Vec<&Candidates> = lines.iter().collect();
        match QuestionContext::new(&query, &lines, None, &table, None, &[]) {
            Err(EvalError::Invalid(message)) => message,
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn view_names_round_trip_and_bad_names_are_none() {
        for view in [View::Now, View::Batch(300), View::Window(86_400)] {
            assert_eq!(View::parse(&view.name()), Some(view));
        }
        assert_eq!(View::Batch(1800).name(), "batch_1800");
        assert_eq!(View::parse("batch_0"), None);
        assert_eq!(View::parse("window_-5"), None);
        assert_eq!(View::parse("later"), None);
    }

    #[test]
    fn each_view_sees_what_design_section_3_1_says() {
        // Asked at 3 725. now: written_at <= 3 725.
        assert!(View::Now.sees(3_725, 3_725));
        assert!(!View::Now.sees(3_726, 3_725));
        // batch_3600: the index last refreshed at floor(3725 / 3600) * 3600 = 3 600.
        assert!(View::Batch(3_600).sees(3_600, 3_725));
        assert!(!View::Batch(3_600).sees(3_601, 3_725));
        // window_300: 3 725 - 300 = 3 425 < written_at <= 3 725.
        assert!(View::Window(300).sees(3_426, 3_725));
        assert!(!View::Window(300).sees(3_425, 3_725));
        assert!(View::Window(300).sees(3_725, 3_725));
        assert!(!View::Window(300).sees(3_726, 3_725));
    }

    #[test]
    fn a_good_line_resolves_and_missing_things_are_named() {
        let table = table();
        let query = question("q", 4_000, QueryClass::Changing, &[("a", 3)], None);
        let line = candidates(
            "q",
            "now",
            4_000,
            &scored(&[("c", 0.9), ("a", 0.8)]),
            &scored(&[("b", 7.0)]),
            &scored(&[("a", 0.5)]),
        );
        let ctx = QuestionContext::new(&query, &[&line], None, &table, None, &[]).unwrap();
        let now = ctx.now().unwrap();
        assert_eq!(now.dense.len(), 2);
        assert_eq!(table.id(now.dense[0].record), "c");
        assert_eq!(now.sparse[0].score, 7.0);
        let a = table.index_of("a").unwrap();
        let b = table.index_of("b").unwrap();
        assert_eq!(ctx.age_of(a), 3_900);
        assert_eq!(ctx.relevance_of(a).unwrap(), 0.5);
        assert!(
            matches!(ctx.relevance_of(b), Err(EvalError::MissingRelevance { record, .. }) if record == "b")
        );
        assert!(matches!(
            ctx.view(View::Batch(300)),
            Err(EvalError::MissingView { view, .. }) if view == "batch_300"
        ));
        assert!(matches!(
            ctx.passage_cosine(a, b, "product_newer_wins"),
            Err(EvalError::VectorsNotLoaded { .. })
        ));
    }

    #[test]
    fn lines_that_break_the_contract_are_refused() {
        let none = scored(&[]);
        let line = |view: &str, as_of, dense: &[(&str, f64)]| {
            candidates("q", view, as_of, &scored(dense), &none, &none)
        };
        assert!(refusal(&[line("soon", 4_000, &[])]).contains("unknown view"));
        assert!(refusal(&[line("now", 3_999, &[])]).contains("asked as of 3999"));
        assert!(refusal(&[line("now", 4_000, &[]), line("now", 4_000, &[])]).contains("appears twice"));
        assert!(refusal(&[line("now", 4_000, &[("zz", 0.5)])]).contains("not a record"));
        assert!(refusal(&[line("now", 4_000, &[("a", 0.5), ("a", 0.4)])]).contains("listed twice"));
        assert!(refusal(&[line("now", 4_000, &[("a", f64::NAN)])]).contains("not a number"));
        assert!(refusal(&[line("now", 4_000, &[("a", 0.4), ("b", 0.5)])]).contains("scored above"));
        // batch_3600 at 4 000 last refreshed at 3 600: record c (3 700) was not in it.
        assert!(refusal(&[line("batch_3600", 4_000, &[("c", 0.5)])]).contains("could not see"));
        // window_300 at 4 000 holds records after 3 700 only: c (3 700) is just outside.
        assert!(refusal(&[line("window_300", 4_000, &[("c", 0.5)])]).contains("could not see"));
        let other = candidates("p", "now", 4_000, &none, &none, &none);
        assert!(refusal(&[other]).contains("a line of question p"));
        let logit = candidates("q", "now", 4_000, &none, &none, &scored(&[("a", 3.2)]));
        assert!(refusal(&[logit]).contains("not between 0 and 1"));
        let twice = candidates(
            "q",
            "now",
            4_000,
            &none,
            &none,
            &scored(&[("a", 0.2), ("a", 0.3)]),
        );
        assert!(refusal(&[twice]).contains("given twice"));
    }

    fn newest_line(newest: &[(&str, f64)], rerank: &[(&str, f64)]) -> Newest {
        Newest {
            query: "q".into(),
            as_of: 4_000,
            newest: scored(newest),
            rerank: scored(rerank),
            timings_ms: Default::default(),
        }
    }

    #[test]
    fn a_newest_list_resolves_and_its_scores_stay_apart_from_the_candidate_lists() {
        let table = table();
        let query = question("q", 4_000, QueryClass::Changing, &[("a", 3)], None);
        let none = scored(&[]);
        let line = candidates("q", "now", 4_000, &none, &none, &scored(&[("a", 0.3)]));
        let newest = newest_line(&[("c", 3_700.0), ("a", 100.0)], &[("c", 0.7), ("a", 0.6)]);
        let ctx = QuestionContext::new(&query, &[&line], Some(&newest), &table, None, &[]).unwrap();
        let at = |id: &str| table.index_of(id).unwrap();
        assert_eq!(
            ctx.newest().unwrap().iter().map(|l| l.record).collect::<Vec<_>>(),
            [at("c"), at("a")]
        );
        // The candidate lists' score stands on its own; with the newest list, the higher one.
        assert_eq!(ctx.relevance_of(at("a")).unwrap(), 0.3);
        assert_eq!(ctx.relevance_with_newest(at("a")).unwrap(), 0.6);
        assert!(ctx.relevance_of(at("c")).is_err());
        assert_eq!(ctx.relevance_with_newest(at("c")).unwrap(), 0.7);
        // Without a newest list, asking for it names the question.
        let bare = QuestionContext::new(&query, &[&line], None, &table, None, &[]).unwrap();
        assert!(matches!(bare.newest(), Err(EvalError::MissingNewest { query }) if query == "q"));
    }

    #[test]
    fn a_newest_list_that_breaks_the_contract_is_refused() {
        let table = table();
        let query = question("q", 4_000, QueryClass::Changing, &[], None);
        let refused =
            |newest: Newest| match QuestionContext::new(&query, &[], Some(&newest), &table, None, &[]) {
                Err(EvalError::Invalid(message)) => message,
                other => panic!("expected a refusal, got {other:?}"),
            };
        let mut other_question = newest_line(&[], &[]);
        other_question.query = "p".into();
        assert!(refused(other_question).contains("a line of question p"));
        let mut other_moment = newest_line(&[], &[]);
        other_moment.as_of = 3_999;
        assert!(refused(other_moment).contains("taken as of 3999"));
        assert!(refused(newest_line(&[("zz", 1.0)], &[])).contains("not a record"));
        assert!(refused(newest_line(&[("a", 100.0), ("a", 100.0)], &[])).contains("listed twice"));
        assert!(refused(newest_line(&[("b", 150.0)], &[])).contains("not its written_at 200"));
        assert!(
            refused(newest_line(&[("a", 100.0), ("b", 200.0)], &[])).contains("newer than the record before")
        );
        assert!(refused(newest_line(&[("a", 100.0)], &[("b", 0.5)])).contains("does not hold it"));
        assert!(refused(newest_line(&[("a", 100.0)], &[("a", 1.5)])).contains("not between 0 and 1"));
        assert!(refused(newest_line(&[("a", 100.0)], &[("a", 0.2), ("a", 0.3)])).contains("given twice"));
        // Asked at 3 000, record c (3 700) was not written yet.
        let early = question("q", 3_000, QueryClass::Changing, &[], None);
        let mut line = newest_line(&[("c", 3_700.0)], &[]);
        line.as_of = 3_000;
        match QuestionContext::new(&early, &[], Some(&line), &table, None, &[]) {
            Err(EvalError::Invalid(message)) => assert!(message.contains("after the question"), "{message}"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn passage_vectors_follow_their_records() {
        use crate::vectors::VectorIndex;
        let table = table();
        let index = VectorIndex {
            dim: 2,
            ids: vec!["b".into(), "a".into()],
        };
        let vectors =
            PassageVectors::new(Vectors::new(index, vec![0.0, 1.0, 1.0, 0.0]).unwrap(), &table).unwrap();
        let at = |id: &str| table.index_of(id).unwrap();
        assert_eq!(vectors.cosine(at("a"), at("b")), Ok(0.0));
        assert_eq!(vectors.cosine(at("a"), at("a")), Ok(1.0));
        assert_eq!(vectors.cosine(at("a"), at("c")), Err(at("c")));
        let stranger = VectorIndex {
            dim: 1,
            ids: vec!["zz".into()],
        };
        assert!(PassageVectors::new(Vectors::new(stranger, vec![1.0]).unwrap(), &table).is_err());
    }
}
