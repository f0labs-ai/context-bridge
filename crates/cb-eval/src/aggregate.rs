//! Per-question scores gathered into groups of questions and the two splits.
//!
//! Results are reported for every class of question. One further group pools the two ordinary classes, because the guard
//! measure of claim 2 is defined on "ordinary and near-ordinary questions".

use crate::error::EvalError;
use crate::measures::{Measure, Measures};
use cb_types::{Query, QueryClass, Split, split_of};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// A set of questions a result is reported for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Group {
    Changing,
    PostIncident,
    Ambiguous,
    Ordinary,
    OrdinaryNear,
    /// `ordinary` and `ordinary_near` together.
    OrdinaryAll,
}

impl Group {
    pub const ALL: [Group; 6] = [
        Group::Changing,
        Group::PostIncident,
        Group::Ambiguous,
        Group::Ordinary,
        Group::OrdinaryNear,
        Group::OrdinaryAll,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Group::Changing => "changing",
            Group::PostIncident => "post_incident",
            Group::Ambiguous => "ambiguous",
            Group::Ordinary => "ordinary",
            Group::OrdinaryNear => "ordinary_near",
            Group::OrdinaryAll => "ordinary_all",
        }
    }

    /// Whether a question of `class` belongs to this group.
    pub fn holds(self, class: QueryClass) -> bool {
        match self {
            Group::Changing => class == QueryClass::Changing,
            Group::PostIncident => class == QueryClass::PostIncident,
            Group::Ambiguous => class == QueryClass::Ambiguous,
            Group::Ordinary => class == QueryClass::Ordinary,
            Group::OrdinaryNear => class == QueryClass::OrdinaryNear,
            Group::OrdinaryAll => matches!(class, QueryClass::Ordinary | QueryClass::OrdinaryNear),
        }
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Group {
    type Err = EvalError;

    fn from_str(text: &str) -> Result<Self, EvalError> {
        Group::ALL.into_iter().find(|g| g.name() == text).ok_or_else(|| {
            let known: Vec<&str> = Group::ALL.iter().map(|g| g.name()).collect();
            EvalError::invalid(format!(
                "{text:?} is not a group of questions (known: {})",
                known.join(", ")
            ))
        })
    }
}

/// The unit that is resampled: questions about one incident are not
/// independent, so they stand or fall together; a question without a thread
/// is a unit of its own.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Cluster {
    Thread(String),
    Question(String),
}

impl Cluster {
    pub fn of(query: &Query) -> Self {
        match &query.thread {
            Some(thread) => Cluster::Thread(thread.clone()),
            None => Cluster::Question(query.id.clone()),
        }
    }
}

/// One method's measures on one question, with what is needed to group it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuestionScore {
    pub query: String,
    pub class: QueryClass,
    pub split: Split,
    pub cluster: Cluster,
    #[serde(flatten)]
    pub measures: Measures,
}

impl QuestionScore {
    pub fn new(query: &Query, measures: Measures) -> Self {
        Self {
            query: query.id.clone(),
            class: query.class,
            split: split_of(query),
            cluster: Cluster::of(query),
            measures,
        }
    }
}

/// The mean of every measure over the questions of one group in one split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupMean {
    pub split: Split,
    pub group: Group,
    pub questions: usize,
    #[serde(flatten)]
    pub means: Measures,
}

/// The plain mean of each measure over `scores`, added up in the order given.
/// `None` for no scores.
pub fn mean_of(scores: &[&QuestionScore]) -> Option<Measures> {
    if scores.is_empty() {
        return None;
    }
    Some(Measures::from_fn(|measure: Measure| {
        scores.iter().map(|s| s.measures.get(measure)).sum::<f64>() / scores.len() as f64
    }))
}

/// Means per split and group, validation before test and groups in their
/// fixed order. A group without questions in a split is left out.
pub fn means_by_split_and_group(scores: &[QuestionScore]) -> Vec<GroupMean> {
    let mut out = Vec::new();
    for split in [Split::Validation, Split::Test] {
        for group in Group::ALL {
            let members: Vec<&QuestionScore> = scores
                .iter()
                .filter(|s| s.split == split && group.holds(s.class))
                .collect();
            if let Some(means) = mean_of(&members) {
                out.push(GroupMean {
                    split,
                    group,
                    questions: members.len(),
                    means,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::question;
    use cb_types::fnv1a64;

    fn score(id: &str, class: QueryClass, thread: Option<&str>, hit: f64) -> QuestionScore {
        let query = question(id, 0, class, &[], thread);
        QuestionScore::new(
            &query,
            Measures {
                hit_4: hit,
                ..Measures::default()
            },
        )
    }

    #[test]
    fn the_split_is_one_thread_in_five_by_a_fixed_hash() {
        // FNV-1a 64 of the thread name, modulo 5: 0 is validation, anything else test.
        // The values are fixed; if the hash or the rule changed, every reported split would move.
        let expected = [
            // 10 535 700 299 149 982 129 ends in 9: remainder 4
            ("inc-0001", 0x9236_5399_fecf_b9b1_u64, Split::Test),
            // 10 535 701 398 661 610 340 ends in 0: divisible by 5
            ("inc-0006", 0x9236_5499_fecf_bb64_u64, Split::Validation),
            // 10 535 698 100 126 725 707 ends in 7: remainder 2
            ("inc-0003", 0x9236_5199_fecf_b64b_u64, Split::Test),
        ];
        for (thread, hash, split) in expected {
            assert_eq!(fnv1a64(thread.as_bytes()), hash, "{thread}");
            assert_eq!(hash % 5 == 0, split == Split::Validation, "{thread}");
            let asked = score("q-1", QueryClass::Changing, Some(thread), 0.0);
            assert_eq!(asked.split, split, "{thread}");
        }
    }

    #[test]
    fn questions_of_one_thread_share_a_split_and_a_cluster_and_others_go_by_their_own_id() {
        let a = score("q-1", QueryClass::Changing, Some("inc-0002"), 0.0);
        let b = score("q-2", QueryClass::PostIncident, Some("inc-0002"), 0.0);
        assert_eq!(a.split, b.split);
        assert_eq!(a.cluster, b.cluster);
        assert_eq!(a.cluster, Cluster::Thread("inc-0002".to_string()));
        // Without a thread the question's own id is hashed and the question is its own cluster.
        let alone = score("inc-0002", QueryClass::Ordinary, None, 0.0);
        assert_eq!(alone.split, a.split);
        assert_eq!(alone.cluster, Cluster::Question("inc-0002".to_string()));
        assert_ne!(alone.cluster, a.cluster);
    }

    #[test]
    fn means_are_taken_per_split_and_group() {
        let scores = vec![
            score("q-1", QueryClass::Changing, Some("inc-0001"), 1.0), // test
            score("q-2", QueryClass::Changing, Some("inc-0001"), 0.0), // test
            score("q-3", QueryClass::Changing, Some("inc-0003"), 1.0), // test
            score("q-4", QueryClass::Changing, Some("inc-0006"), 1.0), // validation
            score("q-5", QueryClass::Ordinary, Some("inc-0001"), 1.0), // test
            score("q-6", QueryClass::OrdinaryNear, Some("inc-0003"), 0.0), // test
        ];
        let means = means_by_split_and_group(&scores);
        let found: Vec<(Split, Group, usize, f64)> = means
            .iter()
            .map(|m| (m.split, m.group, m.questions, m.means.hit_4))
            .collect();
        assert_eq!(
            found,
            [
                (Split::Validation, Group::Changing, 1, 1.0),
                // (1 + 0 + 1) / 3
                (Split::Test, Group::Changing, 3, 2.0 / 3.0),
                (Split::Test, Group::Ordinary, 1, 1.0),
                (Split::Test, Group::OrdinaryNear, 1, 0.0),
                // the two ordinary classes pooled: (1 + 0) / 2
                (Split::Test, Group::OrdinaryAll, 2, 0.5),
            ]
        );
    }

    #[test]
    fn groups_have_names_and_hold_their_classes() {
        for group in Group::ALL {
            assert_eq!(group.name().parse::<Group>().unwrap(), group);
            assert_eq!(
                serde_json::to_string(&group).unwrap(),
                format!("\"{}\"", group.name())
            );
        }
        assert!("all".parse::<Group>().is_err());
        assert!(Group::OrdinaryAll.holds(QueryClass::Ordinary));
        assert!(Group::OrdinaryAll.holds(QueryClass::OrdinaryNear));
        assert!(!Group::OrdinaryAll.holds(QueryClass::Changing));
        assert!(!Group::Ordinary.holds(QueryClass::OrdinaryNear));
    }

    #[test]
    fn a_question_score_is_one_flat_json_object() {
        let s = score("q-1", QueryClass::Changing, Some("inc-0001"), 1.0);
        let json = serde_json::to_value(&s).unwrap();
        assert_eq!(json["query"], "q-1");
        assert_eq!(json["class"], "changing");
        assert_eq!(json["split"], "test");
        assert_eq!(json["cluster"]["thread"], "inc-0001");
        assert_eq!(json["hit@4"], 1.0);
        assert_eq!(serde_json::from_value::<QuestionScore>(json).unwrap(), s);
    }
}
