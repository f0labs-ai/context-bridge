//! The measures of DESIGN.md section 5, for one method's answer to one question.
//!
//! Grades: 3 = the answer, 1 = about the right matter but replaced since,
//! anything else 0. "The first n" means the first n places of the answer; a
//! place the method left empty counts as a miss, so every share below divides
//! by n itself (`filled@n` reports how many places were filled).

use crate::context::Graded;
use crate::error::EvalError;
use crate::records::{RecordIdx, RecordTable};
use cb_types::{Query, Record, Tier, UnixSeconds};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::str::FromStr;

/// The grade of the record that answers the question.
pub const ANSWER: u8 = 3;
/// The grade of a record about the right matter that has been replaced since.
pub const REPLACED: u8 = 1;
/// How deep `mrr` and `recall@20` look.
pub const DEEPEST_CUT: usize = 20;

/// How a measure is printed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// A value between 0 and 1 printed with three decimals, e.g. `0.712`.
    Fraction,
    /// A share printed in percent with one decimal, e.g. `41.2`.
    Percent,
}

/// One measure of section 5, by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Measure {
    /// `DCG@5 / IDCG@5` with gain `2^grade - 1` and discount `log2(place + 1)`.
    #[serde(rename = "ndcg@5")]
    Ndcg5,
    #[serde(rename = "ndcg@10")]
    Ndcg10,
    /// The reciprocal of the place of the first grade-3 record within the first 20; 0 when there is none.
    #[serde(rename = "mrr")]
    Mrr,
    /// The share of the question's grade-3 records that are within the first 20.
    #[serde(rename = "recall@20")]
    Recall20,
    /// 1 when the first record has grade 3.
    #[serde(rename = "current@1")]
    Current1,
    /// 1 when a grade-3 record is within the first 4 (the agent receives 4 passages).
    #[serde(rename = "hit@4")]
    Hit4,
    /// The share of the first 4 places holding a grade-1 record.
    #[serde(rename = "stale@4")]
    Stale4,
    /// The share of the first 4 places holding a record of `Query.harmful`.
    #[serde(rename = "harmful@4")]
    Harmful4,
    /// The share of the first 4 places holding a `Tier::Changing` record.
    #[serde(rename = "changing@4")]
    Changing4,
    /// The share of the first 10 places holding a record that was valid when the question was asked.
    #[serde(rename = "temporal_precision@10")]
    TemporalPrecision10,
    /// The share of the first 4 places the method filled.
    #[serde(rename = "filled@4")]
    Filled4,
    /// The share of the first 10 places the method filled.
    #[serde(rename = "filled@10")]
    Filled10,
}

impl Measure {
    pub const ALL: [Measure; 12] = [
        Measure::Ndcg5,
        Measure::Ndcg10,
        Measure::Mrr,
        Measure::Recall20,
        Measure::Current1,
        Measure::Hit4,
        Measure::Stale4,
        Measure::Harmful4,
        Measure::Changing4,
        Measure::TemporalPrecision10,
        Measure::Filled4,
        Measure::Filled10,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Measure::Ndcg5 => "ndcg@5",
            Measure::Ndcg10 => "ndcg@10",
            Measure::Mrr => "mrr",
            Measure::Recall20 => "recall@20",
            Measure::Current1 => "current@1",
            Measure::Hit4 => "hit@4",
            Measure::Stale4 => "stale@4",
            Measure::Harmful4 => "harmful@4",
            Measure::Changing4 => "changing@4",
            Measure::TemporalPrecision10 => "temporal_precision@10",
            Measure::Filled4 => "filled@4",
            Measure::Filled10 => "filled@10",
        }
    }

    pub fn unit(self) -> Unit {
        match self {
            Measure::Ndcg5 | Measure::Ndcg10 | Measure::Mrr => Unit::Fraction,
            _ => Unit::Percent,
        }
    }

    /// Whether a larger value is the better one when a parameter is chosen by this measure.
    /// `stale@4`, `harmful@4` and `changing@4` count unwanted records.
    pub fn higher_is_better(self) -> bool {
        !matches!(self, Measure::Stale4 | Measure::Harmful4 | Measure::Changing4)
    }
}

impl fmt::Display for Measure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Measure {
    type Err = EvalError;

    fn from_str(text: &str) -> Result<Self, EvalError> {
        Measure::ALL
            .into_iter()
            .find(|m| m.name() == text)
            .ok_or_else(|| {
                let known: Vec<&str> = Measure::ALL.iter().map(|m| m.name()).collect();
                EvalError::invalid(format!("{text:?} is not a measure (known: {})", known.join(", ")))
            })
    }
}

/// Every measure of one answer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Measures {
    #[serde(rename = "ndcg@5")]
    pub ndcg_5: f64,
    #[serde(rename = "ndcg@10")]
    pub ndcg_10: f64,
    pub mrr: f64,
    #[serde(rename = "recall@20")]
    pub recall_20: f64,
    #[serde(rename = "current@1")]
    pub current_1: f64,
    #[serde(rename = "hit@4")]
    pub hit_4: f64,
    #[serde(rename = "stale@4")]
    pub stale_4: f64,
    #[serde(rename = "harmful@4")]
    pub harmful_4: f64,
    #[serde(rename = "changing@4")]
    pub changing_4: f64,
    #[serde(rename = "temporal_precision@10")]
    pub temporal_precision_10: f64,
    #[serde(rename = "filled@4")]
    pub filled_4: f64,
    #[serde(rename = "filled@10")]
    pub filled_10: f64,
}

impl Measures {
    pub fn get(&self, measure: Measure) -> f64 {
        match measure {
            Measure::Ndcg5 => self.ndcg_5,
            Measure::Ndcg10 => self.ndcg_10,
            Measure::Mrr => self.mrr,
            Measure::Recall20 => self.recall_20,
            Measure::Current1 => self.current_1,
            Measure::Hit4 => self.hit_4,
            Measure::Stale4 => self.stale_4,
            Measure::Harmful4 => self.harmful_4,
            Measure::Changing4 => self.changing_4,
            Measure::TemporalPrecision10 => self.temporal_precision_10,
            Measure::Filled4 => self.filled_4,
            Measure::Filled10 => self.filled_10,
        }
    }

    fn slot(&mut self, measure: Measure) -> &mut f64 {
        match measure {
            Measure::Ndcg5 => &mut self.ndcg_5,
            Measure::Ndcg10 => &mut self.ndcg_10,
            Measure::Mrr => &mut self.mrr,
            Measure::Recall20 => &mut self.recall_20,
            Measure::Current1 => &mut self.current_1,
            Measure::Hit4 => &mut self.hit_4,
            Measure::Stale4 => &mut self.stale_4,
            Measure::Harmful4 => &mut self.harmful_4,
            Measure::Changing4 => &mut self.changing_4,
            Measure::TemporalPrecision10 => &mut self.temporal_precision_10,
            Measure::Filled4 => &mut self.filled_4,
            Measure::Filled10 => &mut self.filled_10,
        }
    }

    /// Build from one value per measure.
    pub fn from_fn(mut value: impl FnMut(Measure) -> f64) -> Self {
        let mut out = Measures::default();
        for measure in Measure::ALL {
            *out.slot(measure) = value(measure);
        }
        out
    }
}

/// What scoring knows about every record: its tier and when it was valid.
/// Built from the planted truth; never handed to a method.
#[derive(Debug)]
pub struct ScoringKey {
    tier: Vec<Tier>,
    written_at: Vec<UnixSeconds>,
    superseded_at: Vec<Option<UnixSeconds>>,
}

impl ScoringKey {
    /// `records` must be the records `table` was built from.
    pub fn new(records: &[Record], table: &RecordTable) -> Result<Self, EvalError> {
        let mut key = Self {
            tier: vec![Tier::Standing; table.len()],
            written_at: vec![0; table.len()],
            superseded_at: vec![None; table.len()],
        };
        for record in records {
            let row = table
                .index_of(&record.id)
                .ok_or_else(|| {
                    EvalError::invalid(format!("record {:?} is not in the record table", record.id))
                })?
                .row();
            key.tier[row] = record.tier;
            key.written_at[row] = record.written_at;
            key.superseded_at[row] = record.truth.as_ref().and_then(|truth| truth.superseded_at);
        }
        Ok(key)
    }

    /// Whether the record was valid at `moment`: a standing record always; a
    /// changing record from `written_at` until `truth.superseded_at` (the
    /// moment a later record replaced it), or for good while nothing has.
    pub fn was_valid_at(&self, record: RecordIdx, moment: UnixSeconds) -> bool {
        let row = record.row();
        match self.tier[row] {
            Tier::Standing => true,
            Tier::Changing => {
                self.written_at[row] <= moment
                    && self.superseded_at[row].is_none_or(|replaced| moment < replaced)
            }
        }
    }

    pub fn is_changing(&self, record: RecordIdx) -> bool {
        self.tier[record.row()] == Tier::Changing
    }
}

/// The graded answers of one question, checked against the contract.
#[derive(Debug)]
pub struct QuestionKey {
    asked_at: UnixSeconds,
    gold: Vec<Graded>,
    grade_of: HashMap<RecordIdx, u8>,
    harmful: HashSet<RecordIdx>,
    /// The question's grades, highest first: the ideal answer.
    ideal: Vec<u8>,
    answers: usize,
}

impl QuestionKey {
    /// Refused: an answer or harmful record that is not in the data set, a
    /// record graded twice, a grade above 3, a harmful record that also has a
    /// grade, and a question without any grade-3 record (nothing could score on it).
    pub fn new(query: &Query, table: &RecordTable) -> Result<Self, EvalError> {
        let wrong = |what: String| EvalError::invalid(format!("question {}: {what}", query.id));
        let mut gold = Vec::with_capacity(query.gold.len());
        let mut grade_of = HashMap::with_capacity(query.gold.len());
        for answer in &query.gold {
            let record = table.index_of(&answer.record).ok_or_else(|| {
                wrong(format!(
                    "answer {:?} is not a record of the data set",
                    answer.record
                ))
            })?;
            if answer.grade > ANSWER {
                return Err(wrong(format!(
                    "answer {:?} has grade {}, above 3",
                    answer.record, answer.grade
                )));
            }
            if grade_of.insert(record, answer.grade).is_some() {
                return Err(wrong(format!("answer {:?} is graded twice", answer.record)));
            }
            gold.push(Graded {
                record,
                grade: answer.grade,
            });
        }
        let mut harmful = HashSet::with_capacity(query.harmful.len());
        for id in &query.harmful {
            let record = table
                .index_of(id)
                .ok_or_else(|| wrong(format!("harmful record {id:?} is not a record of the data set")))?;
            if grade_of.get(&record).is_some_and(|grade| *grade > 0) {
                return Err(wrong(format!("record {id:?} is both an answer and harmful")));
            }
            harmful.insert(record);
        }
        let mut ideal: Vec<u8> = gold.iter().map(|g| g.grade).collect();
        ideal.sort_unstable_by(|a, b| b.cmp(a));
        let answers = ideal.iter().filter(|grade| **grade == ANSWER).count();
        if answers == 0 {
            return Err(wrong("it has no grade-3 answer".to_string()));
        }
        Ok(Self {
            asked_at: query.asked_at,
            gold,
            grade_of,
            harmful,
            ideal,
            answers,
        })
    }

    /// The graded answers as listed. Passed to the oracle only.
    pub fn gold(&self) -> &[Graded] {
        &self.gold
    }

    pub fn grade_of(&self, record: RecordIdx) -> u8 {
        self.grade_of.get(&record).copied().unwrap_or(0)
    }
}

/// `2^grade - 1`.
pub fn gain(grade: u8) -> f64 {
    2f64.powi(i32::from(grade)) - 1.0
}

/// `DCG@n = sum over the first n places i of (2^grade_i - 1) / log2(i + 1)`, places counted from 1.
pub fn dcg(grades: &[u8], n: usize) -> f64 {
    grades
        .iter()
        .take(n)
        .zip(1u32..)
        .map(|(grade, place)| gain(*grade) / f64::from(place + 1).log2())
        .sum()
}

/// `NDCG@n = DCG@n / IDCG@n`, where `IDCG@n` is the `DCG@n` of the question's
/// own grades from highest to lowest (`ideal`). `ideal` must hold a positive grade.
pub fn ndcg(grades: &[u8], ideal: &[u8], n: usize) -> f64 {
    dcg(grades, n) / dcg(ideal, n)
}

/// The reciprocal of the place of the first grade-3 record within the first `n`; 0 when there is none.
pub fn reciprocal_rank_of_first_answer(grades: &[u8], n: usize) -> f64 {
    grades
        .iter()
        .take(n)
        .zip(1u32..)
        .find(|(grade, _)| **grade == ANSWER)
        .map_or(0.0, |(_, place)| 1.0 / f64::from(place))
}

/// How many of the first `n` places satisfy `wanted`, divided by `n`.
pub fn share_of_first<T>(answer: &[T], n: usize, wanted: impl Fn(&T) -> bool) -> f64 {
    answer.iter().take(n).filter(|item| wanted(item)).count() as f64 / n as f64
}

/// Every measure of section 5 for one answer (best first).
pub fn score_answer(answer: &[RecordIdx], question: &QuestionKey, key: &ScoringKey) -> Measures {
    let grades: Vec<u8> = answer
        .iter()
        .take(DEEPEST_CUT)
        .map(|r| question.grade_of(*r))
        .collect();
    let is_answer = |grade: &u8| *grade == ANSWER;
    Measures {
        ndcg_5: ndcg(&grades, &question.ideal, 5),
        ndcg_10: ndcg(&grades, &question.ideal, 10),
        mrr: reciprocal_rank_of_first_answer(&grades, DEEPEST_CUT),
        recall_20: grades.iter().filter(|g| is_answer(g)).count() as f64 / question.answers as f64,
        current_1: share_of_first(&grades, 1, is_answer),
        hit_4: if grades.iter().take(4).any(is_answer) {
            1.0
        } else {
            0.0
        },
        stale_4: share_of_first(&grades, 4, |grade| *grade == REPLACED),
        harmful_4: share_of_first(answer, 4, |r| question.harmful.contains(r)),
        changing_4: share_of_first(answer, 4, |r| key.is_changing(*r)),
        temporal_precision_10: share_of_first(answer, 10, |r| key.was_valid_at(*r, question.asked_at)),
        filled_4: share_of_first(answer, 4, |_| true),
        filled_10: share_of_first(answer, 10, |_| true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{question, record, standing, truth};
    use cb_types::QueryClass;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12
    }

    /// An incident with three states, a wrong statement, two help sections and fillers.
    fn records() -> Vec<Record> {
        let state = |id: &str, written_at: i64, superseded_at: Option<i64>| Record {
            truth: truth(true, &[], superseded_at),
            ..record(id, written_at)
        };
        let mut records = vec![
            state("s1", 1_000, Some(2_000)),
            state("s2", 2_000, Some(3_000)),
            state("s3", 3_000, None),
            Record {
                truth: truth(false, &[], None),
                ..record("wrong", 1_500)
            },
            standing("doc1", 10),
            standing("doc2", 10),
            record("untold", 500),
        ];
        records.extend((1..=25).map(|i| standing(&format!("pad{i:02}"), 10)));
        records
    }

    struct Fixture {
        table: RecordTable,
        key: ScoringKey,
        question: QuestionKey,
    }

    /// Asked at 3 500: s3 is the answer, s1 and s2 were replaced, "wrong" would mislead.
    fn fixture() -> Fixture {
        let records = records();
        let table = RecordTable::from_records(&records).unwrap();
        let key = ScoringKey::new(&records, &table).unwrap();
        let mut query = question(
            "q",
            3_500,
            QueryClass::Changing,
            &[("s3", 3), ("s1", 1), ("s2", 1)],
            Some("i"),
        );
        query.harmful = vec!["wrong".to_string()];
        let question = QuestionKey::new(&query, &table).unwrap();
        Fixture { table, key, question }
    }

    fn score(fixture: &Fixture, answer: &[&str]) -> Measures {
        let answer: Vec<RecordIdx> = answer
            .iter()
            .map(|id| fixture.table.index_of(id).unwrap())
            .collect();
        score_answer(&answer, &fixture.question, &fixture.key)
    }

    #[test]
    fn gain_and_dcg_follow_their_formulas() {
        // 2^3 - 1 = 7, 2^1 - 1 = 1, 2^0 - 1 = 0, 2^2 - 1 = 3
        assert_eq!([gain(3), gain(1), gain(0), gain(2)], [7.0, 1.0, 0.0, 3.0]);
        // grades 1, 3, 0, 1: 1/log2(2) + 7/log2(3) + 0/log2(4) + 1/log2(5)
        //                  = 1 + 7/1.5849625 + 0 + 1/2.3219281 = 1 + 4.4165083 + 0.4306766 = 5.8471848
        assert!(close(
            dcg(&[1, 3, 0, 1], 10),
            1.0 + 7.0 / 3f64.log2() + 1.0 / 5f64.log2()
        ));
        assert!((dcg(&[1, 3, 0, 1], 10) - 5.8471848).abs() < 1e-7);
        // Cut at 2 places: 1 + 4.4165083 = 5.4165083
        assert!((dcg(&[1, 3, 0, 1], 2) - 5.4165083).abs() < 1e-7);
        assert_eq!(dcg(&[], 10), 0.0);
        // A grade 3 in place 7: 7/log2(8) = 7/3
        assert!(close(dcg(&[0, 0, 0, 0, 0, 0, 3], 10), 7.0 / 3.0));
    }

    #[test]
    fn ndcg_of_a_worked_example() {
        let f = fixture();
        let m = score(&f, &["s1", "s3", "doc1", "s2", "doc2"]);
        // DCG  = 1/log2(2) + 7/log2(3) + 0 + 1/log2(5) + 0          = 5.8471848
        // IDCG = 7/log2(2) + 1/log2(3) + 1/log2(4) = 7 + 0.6309298 + 0.5 = 8.1309298
        // NDCG = 5.8471848 / 8.1309298 = 0.7191287
        let expected = (1.0 + 7.0 / 3f64.log2() + 1.0 / 5f64.log2()) / (7.0 + 1.0 / 3f64.log2() + 0.5);
        assert!(close(m.ndcg_5, expected));
        assert!((m.ndcg_5 - 0.7191287).abs() < 1e-7);
        // Nothing relevant lies between places 6 and 10, so NDCG@10 is the same.
        assert!(close(m.ndcg_10, expected));
    }

    #[test]
    fn ndcg_at_5_and_at_10_differ_when_the_answer_is_in_place_7() {
        let f = fixture();
        let m = score(&f, &["pad01", "pad02", "pad03", "pad04", "pad05", "pad06", "s3"]);
        assert_eq!(m.ndcg_5, 0.0);
        // DCG@10 = 7/log2(8) = 7/3 = 2.3333333; IDCG@10 = 8.1309298; NDCG@10 = 0.28697005
        assert!(close(m.ndcg_10, (7.0 / 3.0) / (7.0 + 1.0 / 3f64.log2() + 0.5)));
        assert!((m.ndcg_10 - 0.28697005).abs() < 1e-7);
    }

    #[test]
    fn the_gold_order_scores_one_everywhere_it_can() {
        let f = fixture();
        let m = score(&f, &["s3", "s2", "s1"]);
        assert_eq!(
            (m.ndcg_5, m.ndcg_10, m.mrr, m.recall_20, m.current_1, m.hit_4),
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        );
        // The two replaced states fill two of the first four places.
        assert_eq!(m.stale_4, 0.5);
        assert_eq!(m.harmful_4, 0.0);
        assert_eq!((m.filled_4, m.filled_10), (0.75, 0.3));
    }

    #[test]
    fn mrr_current_and_hit_follow_the_place_of_the_answer() {
        let f = fixture();
        let first = score(&f, &["s3", "doc1"]);
        assert_eq!((first.mrr, first.current_1, first.hit_4), (1.0, 1.0, 1.0));
        let second = score(&f, &["s1", "s3", "doc1", "s2", "doc2"]);
        assert_eq!((second.mrr, second.current_1, second.hit_4), (0.5, 0.0, 1.0));
        let fourth = score(&f, &["doc1", "doc2", "s1", "s3"]);
        assert_eq!((fourth.mrr, fourth.current_1, fourth.hit_4), (0.25, 0.0, 1.0));
        // Fifth place: outside the four passages the agent receives. 1/5 = 0.2
        let fifth = score(&f, &["doc1", "doc2", "s1", "s2", "s3"]);
        assert_eq!((fifth.mrr, fifth.current_1, fifth.hit_4), (0.2, 0.0, 0.0));
        let never = score(&f, &["doc1", "s1"]);
        assert_eq!(
            (never.mrr, never.current_1, never.hit_4, never.recall_20),
            (0.0, 0.0, 0.0, 0.0)
        );
        let nothing = score(&f, &[]);
        assert_eq!(nothing, Measures::default());
    }

    #[test]
    fn only_the_first_twenty_count_for_mrr_and_recall() {
        let f = fixture();
        let pads: Vec<String> = (1..=20).map(|i| format!("pad{i:02}")).collect();
        let mut twentieth: Vec<&str> = pads[..19].iter().map(String::as_str).collect();
        twentieth.push("s3");
        let m = score(&f, &twentieth);
        // Place 20: 1/20 = 0.05
        assert_eq!((m.mrr, m.recall_20), (0.05, 1.0));
        let mut twenty_first: Vec<&str> = pads.iter().map(String::as_str).collect();
        twenty_first.push("s3");
        let m = score(&f, &twenty_first);
        assert_eq!((m.mrr, m.recall_20), (0.0, 0.0));
    }

    #[test]
    fn recall_is_the_share_of_grade_3_records_found() {
        let records = records();
        let table = RecordTable::from_records(&records).unwrap();
        let key = ScoringKey::new(&records, &table).unwrap();
        // A question that fits two matters has two grade-3 records.
        let query = question(
            "q",
            3_500,
            QueryClass::Ambiguous,
            &[("s3", 3), ("doc1", 3), ("s1", 1)],
            None,
        );
        let question = QuestionKey::new(&query, &table).unwrap();
        let at = |id: &str| table.index_of(id).unwrap();
        let one = score_answer(&[at("pad01"), at("doc1")], &question, &key);
        assert_eq!(one.recall_20, 0.5);
        let both = score_answer(&[at("s3"), at("pad01"), at("doc1")], &question, &key);
        assert_eq!(both.recall_20, 1.0);
        // Ideal: 7/log2(2) + 7/log2(3) + 1/log2(4) = 7 + 4.4165083 + 0.5 = 11.9165083
        // Got:   7/log2(2) + 0 + 7/log2(4) = 7 + 3.5 = 10.5 → 10.5 / 11.9165083 = 0.8811306
        assert!((both.ndcg_10 - 0.8811306).abs() < 1e-7);
    }

    #[test]
    fn shares_of_the_first_four_divide_by_four() {
        let f = fixture();
        // s1 and s2 are replaced (grade 1), "wrong" is harmful, doc1 is standing.
        let m = score(&f, &["s1", "wrong", "s2", "doc1", "s3"]);
        assert_eq!(m.stale_4, 0.5); // s1, s2
        assert_eq!(m.harmful_4, 0.25); // wrong
        assert_eq!(m.changing_4, 0.75); // s1, wrong, s2 are changing records; doc1 is not
        assert_eq!(m.filled_4, 1.0);
        // A short answer leaves places empty: one stale record in four places is 0.25, not 1.
        let short = score(&f, &["s1"]);
        assert_eq!(
            (short.stale_4, short.changing_4, short.filled_4, short.filled_10),
            (0.25, 0.25, 0.25, 0.1)
        );
        // Fifth place and beyond do not count.
        let late = score(&f, &["doc1", "doc2", "pad01", "pad02", "wrong", "s1"]);
        assert_eq!((late.stale_4, late.harmful_4, late.changing_4), (0.0, 0.0, 0.0));
    }

    #[test]
    fn temporal_precision_counts_records_valid_when_the_question_was_asked() {
        let f = fixture();
        // Asked at 3 500. s3 (from 3 000, never replaced): valid. s2 (2 000 to 3 000): replaced.
        // s1 (1 000 to 2 000): replaced. doc1, doc2: standing, always valid. "untold" is a changing
        // record without planted truth: valid from the moment it was written. "wrong" was never
        // replaced by a later record, so by the same rule it counts as valid.
        let m = score(&f, &["s3", "s2", "s1", "doc1", "doc2", "untold", "wrong"]);
        // valid: s3, doc1, doc2, untold, wrong = 5 of 10 places
        assert_eq!(m.temporal_precision_10, 0.5);
        assert_eq!(m.filled_10, 0.7);
        let key = &f.key;
        let at = |id: &str| f.table.index_of(id).unwrap();
        // s1 is valid from 1 000 up to, not including, 2 000.
        assert!(!key.was_valid_at(at("s1"), 999));
        assert!(key.was_valid_at(at("s1"), 1_000));
        assert!(key.was_valid_at(at("s1"), 1_999));
        assert!(!key.was_valid_at(at("s1"), 2_000));
        assert!(key.was_valid_at(at("doc1"), 0));
        assert!(key.was_valid_at(at("s3"), i64::MAX));
    }

    #[test]
    fn a_question_that_breaks_the_contract_is_refused() {
        let records = records();
        let table = RecordTable::from_records(&records).unwrap();
        let refusal = |gold: &[(&str, u8)], harmful: &[&str]| {
            let mut query = question("q", 3_500, QueryClass::Changing, gold, None);
            query.harmful = harmful.iter().map(|s| s.to_string()).collect();
            match QuestionKey::new(&query, &table) {
                Err(EvalError::Invalid(message)) => message,
                other => panic!("expected a refusal, got {other:?}"),
            }
        };
        assert!(refusal(&[("nope", 3)], &[]).contains("not a record"));
        assert!(refusal(&[("s3", 3), ("s3", 1)], &[]).contains("graded twice"));
        assert!(refusal(&[("s3", 4)], &[]).contains("above 3"));
        assert!(refusal(&[("s1", 1)], &[]).contains("no grade-3 answer"));
        assert!(refusal(&[], &[]).contains("no grade-3 answer"));
        assert!(refusal(&[("s3", 3)], &["nope"]).contains("not a record"));
        assert!(refusal(&[("s3", 3), ("s1", 1)], &["s1"]).contains("both an answer and harmful"));
    }

    #[test]
    fn measures_have_names_units_and_a_direction() {
        for measure in Measure::ALL {
            assert_eq!(measure.name().parse::<Measure>().unwrap(), measure);
            assert_eq!(
                serde_json::to_string(&measure).unwrap(),
                format!("\"{}\"", measure.name())
            );
        }
        assert!("ndcg@7".parse::<Measure>().is_err());
        assert_eq!(Measure::Ndcg10.unit(), Unit::Fraction);
        assert_eq!(Measure::Current1.unit(), Unit::Percent);
        assert!(Measure::Hit4.higher_is_better());
        assert!(!Measure::Stale4.higher_is_better());
        // The struct and the enum name the same twelve values in the same order.
        let numbered = Measures::from_fn(|m| Measure::ALL.iter().position(|x| *x == m).unwrap() as f64);
        let json = serde_json::to_value(numbered).unwrap();
        for (i, measure) in Measure::ALL.iter().enumerate() {
            assert_eq!(numbered.get(*measure), i as f64);
            assert_eq!(json[measure.name()], serde_json::json!(i as f64));
        }
        assert_eq!(json.as_object().unwrap().len(), Measure::ALL.len());
    }
}
