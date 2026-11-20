//! The two-conversation rule (DESIGN.md section 6): when is an extracted
//! record correct, and what a gate that waits for agreeing conversations
//! lets through, and how late.
//!
//! An extraction is **correct** when it states a true record of its
//! transcript and no planted wrong statement: every key term of one true
//! state its transcript states appears in `problem + " " + solution`, and no
//! wrong statement of its incident has all of its key terms there. A key term
//! "appears" when, after both texts are case-folded and their whitespace
//! collapsed, it occurs as whole words (no letter or digit just before or
//! just after it).
//!
//! The gate replays the extractions in the order they become available
//! (the end of the call plus the model's own time) through the paper's
//! Algorithm 2: an extraction below the confidence cut is dropped; otherwise
//! it is promoted together with earlier pending extractions that agree with
//! it, once enough independent conversations agree, or else it waits.

use crate::error::EvalError;
use crate::stats::{Interval, median, nearest_rank, percentile_interval, sorted, stream, uniform_below};
use crate::vectors::Vectors;
use cb_types::{Confidence, Query, QueryClass, Record, Speaker, Split, Transcript, split_of};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet};

// ---- the text rule -------------------------------------------------------------------

/// Case-folded, with every run of whitespace made one space, trimmed.
pub fn normalise(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Whether the normalised `term` occurs in the normalised `text` as whole
/// words: the character before the match and the one after it (when there
/// are any) are neither letters nor digits.
pub fn term_appears(text: &str, term: &str) -> bool {
    if term.is_empty() {
        return false;
    }
    let mut from = 0;
    while let Some(found) = text[from..].find(term) {
        let start = from + found;
        let end = start + term.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        if !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric) {
            return true;
        }
        // Try again from the next character.
        from = start + text[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// Whether every one of `terms` appears in `text` (true for no terms).
pub fn all_terms_appear(text: &str, terms: &[String]) -> bool {
    terms.iter().all(|term| term_appears(text, term))
}

// ---- planted truth ---------------------------------------------------------------------

/// A record's planted truth, as the gate scores against it.
#[derive(Clone, Debug)]
pub struct TruthRow {
    pub id: String,
    pub written_at: i64,
    pub thread: Option<String>,
    pub correct: bool,
    /// Normalised.
    pub key_terms: Vec<String>,
}

/// Every record with planted truth, and per incident its wrong statements.
#[derive(Debug)]
pub struct Truth {
    rows: Vec<TruthRow>,
    by_id: HashMap<String, usize>,
    wrong_by_thread: HashMap<String, Vec<usize>>,
    /// Wrong statements without key terms: they can never be recognised and are counted, not used.
    pub wrong_without_terms: usize,
    /// True records without key terms: any extraction of a call that states one counts as stating it.
    pub true_without_terms: usize,
}

impl Truth {
    /// Records without `truth` are skipped (standing documentation). A key
    /// term that is empty once normalised is a contract error.
    pub fn new(records: &[Record]) -> Result<Self, EvalError> {
        let mut truth = Self {
            rows: Vec::new(),
            by_id: HashMap::new(),
            wrong_by_thread: HashMap::new(),
            wrong_without_terms: 0,
            true_without_terms: 0,
        };
        for record in records {
            let Some(planted) = &record.truth else {
                continue;
            };
            let key_terms: Vec<String> = planted.key_terms.iter().map(|t| normalise(t)).collect();
            if key_terms.iter().any(String::is_empty) {
                return Err(EvalError::invalid(format!(
                    "records.jsonl: record {:?} has an empty key term",
                    record.id
                )));
            }
            let index = truth.rows.len();
            if truth.by_id.insert(record.id.clone(), index).is_some() {
                return Err(EvalError::invalid(format!(
                    "records.jsonl: record id {:?} appears twice",
                    record.id
                )));
            }
            match (planted.correct, key_terms.is_empty()) {
                (false, true) => truth.wrong_without_terms += 1,
                (false, false) => {
                    if let Some(thread) = &record.thread {
                        truth
                            .wrong_by_thread
                            .entry(thread.clone())
                            .or_default()
                            .push(index);
                    }
                }
                (true, true) => truth.true_without_terms += 1,
                (true, false) => {}
            }
            truth.rows.push(TruthRow {
                id: record.id.clone(),
                written_at: record.written_at,
                thread: record.thread.clone(),
                correct: planted.correct,
                key_terms,
            });
        }
        Ok(truth)
    }

    pub fn row(&self, index: usize) -> &TruthRow {
        &self.rows[index]
    }

    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.by_id.get(id).copied()
    }

    /// The recognisable wrong statements of an incident.
    pub fn wrong_of(&self, thread: &str) -> &[usize] {
        self.wrong_by_thread.get(thread).map_or(&[], Vec::as_slice)
    }
}

/// What a text states by the key-term rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stated {
    /// A true record: all its key terms appear, and no wrong statement's do.
    True(usize),
    /// A planted wrong statement: all of its key terms appear.
    Wrong(usize),
    Nothing,
}

impl Stated {
    pub fn record(self) -> Option<usize> {
        match self {
            Stated::True(r) | Stated::Wrong(r) => Some(r),
            Stated::Nothing => None,
        }
    }
}

/// What a normalised text states, judged against the true records its
/// call(s) stated and the wrong statements of its incident(s). A wrong
/// statement wins over a true one (the text is then not correct); of several
/// true records the newest is named, of several wrong ones the first given.
pub fn stated_by(text: &str, true_records: &[usize], wrong_records: &[usize], truth: &Truth) -> Stated {
    if let Some(wrong) = wrong_records
        .iter()
        .find(|w| all_terms_appear(text, &truth.row(**w).key_terms))
    {
        return Stated::Wrong(*wrong);
    }
    true_records
        .iter()
        .filter(|r| all_terms_appear(text, &truth.row(**r).key_terms))
        .max_by(|a, b| {
            let (ra, rb) = (truth.row(**a), truth.row(**b));
            ra.written_at.cmp(&rb.written_at).then_with(|| rb.id.cmp(&ra.id))
        })
        .map_or(Stated::Nothing, |r| Stated::True(*r))
}

/// The split of an incident: [`cb_types::split_of`] applied to its thread, so
/// the rule has one owner.
pub fn split_of_thread(thread: &str) -> Split {
    split_of(&Query {
        id: thread.to_string(),
        text: String::new(),
        asked_at: 0,
        class: QueryClass::Changing,
        gold: Vec::new(),
        thread: Some(thread.to_string()),
        harmful: Vec::new(),
    })
}

/// The moment a call ends: its start plus the end of its last turn, seconds.
pub fn call_end(transcript: &Transcript) -> f64 {
    transcript.started_at as f64
        + transcript
            .turns
            .iter()
            .map(|t| f64::from(t.ends_at_s))
            .fold(0.0, f64::max)
}

/// When a call first states a record: the end of the agent turn by which
/// every key term of the record has appeared in the agent's words so far.
/// `None` when they never all appear (the words were altered); the caller
/// then uses the end of the call.
pub fn statement_moment(transcript: &Transcript, key_terms: &[String]) -> Option<f64> {
    let mut said = String::new();
    for turn in transcript.turns.iter().filter(|t| t.who == Speaker::Agent) {
        if !said.is_empty() {
            said.push(' ');
        }
        said.push_str(&normalise(&turn.text));
        if all_terms_appear(&said, key_terms) {
            return Some(transcript.started_at as f64 + f64::from(turn.ends_at_s));
        }
    }
    None
}

// ---- the extractions of one split ----------------------------------------------------

/// What the gate knows about one extraction.
#[derive(Clone, Debug)]
pub struct Item {
    pub transcript: String,
    pub agent: String,
    pub thread: String,
    /// When the extraction is available: the end of the call plus the model's time, seconds.
    pub arrives_at: f64,
    pub confidence: Confidence,
    /// Normalised.
    pub problem: String,
    /// Normalised.
    pub solution: String,
    /// The true records its call states.
    pub true_states: Vec<usize>,
    /// What the extraction states by the key-term rule.
    pub stated: Stated,
}

impl Item {
    pub fn correct(&self) -> bool {
        matches!(self.stated, Stated::True(_))
    }
}

/// Pairwise likeness (cosine) of the items of one split, kept once per pair.
#[derive(Debug)]
pub struct Likeness {
    items: usize,
    upper: Vec<f32>,
}

impl Likeness {
    /// Refused above this many items: the pairs would not fit in memory comfortably.
    pub const MAX_ITEMS: usize = 20_000;

    /// `rows[i]` is item i's row in `vectors`. Computed on up to `threads` threads.
    pub fn compute(vectors: &Vectors, rows: &[usize], threads: usize) -> Result<Self, EvalError> {
        let items = rows.len();
        if items > Self::MAX_ITEMS {
            return Err(EvalError::invalid(format!(
                "gate: {items} extractions in one split; at most {} are compared pairwise",
                Self::MAX_ITEMS
            )));
        }
        let mut upper = vec![0.0f32; items * items.saturating_sub(1) / 2];
        let mut row_slices: Vec<(usize, &mut [f32])> = Vec::with_capacity(items);
        let mut rest: &mut [f32] = &mut upper;
        for i in 0..items {
            let (this, after) = rest.split_at_mut(items - i - 1);
            row_slices.push((i, this));
            rest = after;
        }
        let threads = threads.clamp(1, items.max(1));
        let mut buckets: Vec<Vec<(usize, &mut [f32])>> = (0..threads).map(|_| Vec::new()).collect();
        for (n, row) in row_slices.into_iter().enumerate() {
            buckets[n % threads].push(row);
        }
        std::thread::scope(|scope| {
            for bucket in buckets {
                scope.spawn(move || {
                    for (i, slice) in bucket {
                        for (offset, cell) in slice.iter_mut().enumerate() {
                            *cell = vectors.cosine(rows[i], rows[i + 1 + offset]) as f32;
                        }
                    }
                });
            }
        });
        Ok(Self { items, upper })
    }

    /// The likeness of two different items.
    pub fn between(&self, a: usize, b: usize) -> f32 {
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        self.upper[i * (2 * self.items - i - 1) / 2 + (j - i - 1)]
    }
}

// ---- the gate --------------------------------------------------------------------------

/// What "agree" means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Agreement {
    /// Every pair of the group is more alike than the bar (the paper).
    Pairs,
    /// The newcomer's average likeness to a pending group is above the bar.
    Average,
}

/// What "independent" means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Independence {
    /// Different calls (the paper).
    Calls,
    /// Different agents: two calls by one agent count once.
    Agents,
}

/// One setting of the gate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateRule {
    /// How many independent conversations must agree (1 = every extraction is promoted).
    pub conversations: usize,
    pub agreement: Agreement,
    pub independence: Independence,
    /// Extractions whose confidence (the product of its three parts) is under this are dropped.
    pub confidence_cut: Option<f64>,
    /// Likeness must be above this (strictly, as in the paper).
    pub bar: f64,
}

/// The most conversations a rule may ask for.
pub const MAX_CONVERSATIONS: usize = 5;

/// A promoted group: its members (item indices, by arrival) and the moment it was promoted.
#[derive(Clone, Debug, PartialEq)]
pub struct Promoted {
    pub members: Vec<usize>,
    pub at: f64,
}

fn independent(rule: &GateRule, items: &[Item], a: usize, b: usize) -> bool {
    match rule.independence {
        Independence::Calls => items[a].transcript != items[b].transcript,
        Independence::Agents => items[a].agent != items[b].agent,
    }
}

fn passes_cut(rule: &GateRule, item: &Item) -> bool {
    rule.confidence_cut
        .is_none_or(|cut| f64::from(item.confidence.product()) >= cut)
}

/// The first set of `need` candidates (in the given order) that agree and
/// are independent pair by pair.
fn first_clique(
    candidates: &[usize],
    need: usize,
    fits: &dyn Fn(usize, usize) -> bool,
    chosen: &mut Vec<usize>,
    from: usize,
) -> bool {
    if chosen.len() == need {
        return true;
    }
    for (offset, candidate) in candidates[from..].iter().enumerate() {
        if chosen.iter().all(|c| fits(*c, *candidate)) {
            chosen.push(*candidate);
            if first_clique(candidates, need, fits, chosen, from + offset + 1) {
                return true;
            }
            chosen.pop();
        }
    }
    false
}

/// Replay `order` (item indices by arrival) through the gate and return every promoted group.
///
/// - [`Agreement::Pairs`]: the paper's Algorithm 2. A newcomer looks among
///   the pending extractions for `conversations - 1` that, with it, agree and
///   are independent pair by pair; the most alike first, earlier ones first
///   among equals. Found: all are promoted and leave the pending pool.
///   Otherwise the newcomer waits.
/// - [`Agreement::Average`]: pending extractions form groups. A newcomer
///   joins the pending group to which its average likeness is highest, when
///   that is above the bar (the earlier group among equals), or else starts
///   a group. A group is promoted when it holds `conversations` independent
///   members.
pub fn replay(rule: &GateRule, items: &[Item], order: &[usize], likeness: &Likeness) -> Vec<Promoted> {
    let bar = rule.bar as f32;
    let mut promoted = Vec::new();
    match rule.agreement {
        Agreement::Pairs => {
            let mut pending: Vec<usize> = Vec::new();
            let fits = |a: usize, b: usize| likeness.between(a, b) > bar && independent(rule, items, a, b);
            for &newcomer in order {
                if !passes_cut(rule, &items[newcomer]) {
                    continue;
                }
                let need = rule.conversations.saturating_sub(1);
                let mut candidates: Vec<usize> =
                    pending.iter().copied().filter(|p| fits(newcomer, *p)).collect();
                candidates.sort_by(|a, b| {
                    likeness
                        .between(newcomer, *b)
                        .total_cmp(&likeness.between(newcomer, *a))
                });
                let mut chosen = Vec::with_capacity(need);
                if first_clique(&candidates, need, &fits, &mut chosen, 0) {
                    pending.retain(|p| !chosen.contains(p));
                    chosen.push(newcomer);
                    chosen.sort_by(|a, b| items[*a].arrives_at.total_cmp(&items[*b].arrives_at));
                    promoted.push(Promoted {
                        members: chosen,
                        at: items[newcomer].arrives_at,
                    });
                } else {
                    pending.push(newcomer);
                }
            }
        }
        Agreement::Average => {
            let mut groups: Vec<Vec<usize>> = Vec::new();
            let distinct = |members: &[usize]| -> usize {
                match rule.independence {
                    Independence::Calls => members
                        .iter()
                        .map(|m| &items[*m].transcript)
                        .collect::<HashSet<_>>()
                        .len(),
                    Independence::Agents => members
                        .iter()
                        .map(|m| &items[*m].agent)
                        .collect::<HashSet<_>>()
                        .len(),
                }
            };
            for &newcomer in order {
                if !passes_cut(rule, &items[newcomer]) {
                    continue;
                }
                let mut best: Option<(usize, f64)> = None;
                for (g, members) in groups.iter().enumerate() {
                    let average = members
                        .iter()
                        .map(|m| f64::from(likeness.between(newcomer, *m)))
                        .sum::<f64>()
                        / members.len() as f64;
                    if average > f64::from(bar) && best.is_none_or(|(_, top)| average > top) {
                        best = Some((g, average));
                    }
                }
                let g = match best {
                    Some((g, _)) => {
                        groups[g].push(newcomer);
                        g
                    }
                    None => {
                        groups.push(vec![newcomer]);
                        groups.len() - 1
                    }
                };
                if distinct(&groups[g]) >= rule.conversations {
                    promoted.push(Promoted {
                        members: groups.remove(g),
                        at: items[newcomer].arrives_at,
                    });
                }
            }
        }
    }
    promoted
}

// ---- what a replay achieved ------------------------------------------------------------

/// The measures of DESIGN.md section 6 for one rule on one split.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GateOutcome {
    /// Extractions above the confidence cut.
    pub considered: usize,
    /// Promoted records (groups).
    pub promoted: usize,
    /// Promoted records whose merged text states a true record and whose members do not mix records.
    pub correct: usize,
    /// `correct / promoted`.
    pub precision: Option<f64>,
    /// Promoted groups whose members state two or more different records.
    pub false_merges: usize,
    /// True records the covered calls state.
    pub states: usize,
    /// Of those, the ones some correct promoted record states.
    pub states_validated: usize,
    pub share_validated: Option<f64>,
    /// True records that two or more of the covered calls state, and of those the ones
    /// validated: the denominator of claim 3's second pass line. A record stated in one call can never be confirmed by a second.
    #[serde(default)]
    pub states_in_two_calls: usize,
    #[serde(default)]
    pub states_in_two_calls_validated: usize,
    #[serde(default)]
    pub share_validated_in_two_calls: Option<f64>,
    /// From a state's first statement to the promotion of the first correct record stating it, seconds.
    pub wait_median_s: Option<f64>,
    pub wait_p90_s: Option<f64>,
}

/// The paper's merge: the problem of the member most confident of its problem,
/// the solution of the member most confident of its solution (the earlier among equals).
pub fn merged_parts<'a>(members: &[usize], items: &'a [Item]) -> (&'a str, &'a str) {
    let best = |part: fn(&Confidence) -> f32| -> usize {
        let mut best = members[0];
        for m in &members[1..] {
            if part(&items[*m].confidence) > part(&items[best].confidence) {
                best = *m;
            }
        }
        best
    };
    let problem = best(|c| c.problem);
    let solution = best(|c| c.solution);
    (&items[problem].problem, &items[solution].solution)
}

/// [`merged_parts`] as one text: the problem, a space, the solution.
pub fn merged_text(members: &[usize], items: &[Item]) -> String {
    let (problem, solution) = merged_parts(members, items);
    format!("{problem} {solution}").trim().to_string()
}

/// The key-term verdict on one promoted group: what its merged text states, judged
/// against the true records its members' calls state and the wrong statements of their
/// incidents, and whether its members state two or more different records (a false
/// merge). The group is correct when it states a true record and does not mix.
pub fn judge_group(members: &[usize], items: &[Item], truth: &Truth) -> (Stated, bool) {
    let named: HashSet<usize> = members.iter().filter_map(|m| items[*m].stated.record()).collect();
    let mut true_records: Vec<usize> = members
        .iter()
        .flat_map(|m| items[*m].true_states.iter().copied())
        .collect();
    true_records.sort_unstable();
    true_records.dedup();
    let threads: BTreeSet<&str> = members.iter().map(|m| items[*m].thread.as_str()).collect();
    let wrong: Vec<usize> = threads
        .iter()
        .flat_map(|t| truth.wrong_of(t).iter().copied())
        .collect();
    let merged = normalise(&merged_text(members, items));
    (stated_by(&merged, &true_records, &wrong, truth), named.len() >= 2)
}

/// Whether [`judge_group`] finds the group correct.
pub fn group_is_correct(members: &[usize], items: &[Item], truth: &Truth) -> bool {
    matches!(judge_group(members, items, truth), (Stated::True(_), false))
}

/// Score a replay. `first_statement[r]` is when record `r` was first stated
/// in a covered call; `states` lists the true records covered calls state, and
/// `stated_twice` those that two or more covered calls state.
pub fn score(
    rule: &GateRule,
    items: &[Item],
    promoted: &[Promoted],
    truth: &Truth,
    (states, stated_twice): (&BTreeSet<usize>, &BTreeSet<usize>),
    first_statement: &HashMap<usize, f64>,
) -> GateOutcome {
    let considered = items.iter().filter(|i| passes_cut(rule, i)).count();
    let mut correct = 0;
    let mut false_merges = 0;
    let mut validated_at: HashMap<usize, f64> = HashMap::new();
    for group in promoted {
        let (stated, mixes) = judge_group(&group.members, items, truth);
        if mixes {
            false_merges += 1;
        }
        if let (Stated::True(record), false) = (stated, mixes) {
            correct += 1;
            validated_at.entry(record).or_insert(group.at);
        }
    }
    let waits = sorted(
        validated_at
            .iter()
            .filter(|(record, _)| states.contains(*record))
            .filter_map(|(record, at)| first_statement.get(record).map(|first| at - first))
            .collect(),
    );
    let states_validated = validated_at.keys().filter(|r| states.contains(*r)).count();
    let twice_validated = validated_at.keys().filter(|r| stated_twice.contains(*r)).count();
    let share = |part: usize, whole: usize| (whole > 0).then(|| part as f64 / whole as f64);
    GateOutcome {
        considered,
        promoted: promoted.len(),
        correct,
        precision: share(correct, promoted.len()),
        false_merges,
        states: states.len(),
        states_validated,
        share_validated: share(states_validated, states.len()),
        states_in_two_calls: stated_twice.len(),
        states_in_two_calls_validated: twice_validated,
        share_validated_in_two_calls: share(twice_validated, stated_twice.len()),
        wait_median_s: median(&waits),
        wait_p90_s: nearest_rank(&waits, 900),
    }
}

/// A language-model judge's verdict on one extraction (`--judge` file, one per line).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeVerdict {
    pub transcript: String,
    pub correct: bool,
}

/// How far the key-term rule and the judge agree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct JudgeAgreement {
    /// Extractions both judged.
    pub judged: usize,
    pub both_correct: usize,
    pub both_wrong: usize,
    pub only_rule_correct: usize,
    pub only_judge_correct: usize,
    /// The share on which they agree.
    pub agreement: Option<f64>,
    /// Cohen's kappa: `(p_o - p_e) / (1 - p_e)`; `None` when `p_e` is 1.
    pub kappa: Option<f64>,
}

/// Agreement of two verdicts per extraction: (rule, judge).
pub fn agreement_of(pairs: &[(bool, bool)]) -> JudgeAgreement {
    let count = |rule: bool, judge: bool| pairs.iter().filter(|p| **p == (rule, judge)).count();
    let (both_correct, both_wrong) = (count(true, true), count(false, false));
    let (only_rule_correct, only_judge_correct) = (count(true, false), count(false, true));
    let n = pairs.len() as f64;
    let (agreement, kappa) = if pairs.is_empty() {
        (None, None)
    } else {
        let observed = (both_correct + both_wrong) as f64 / n;
        let rule_yes = (both_correct + only_rule_correct) as f64 / n;
        let judge_yes = (both_correct + only_judge_correct) as f64 / n;
        let expected = rule_yes * judge_yes + (1.0 - rule_yes) * (1.0 - judge_yes);
        let kappa = (expected < 1.0).then(|| (observed - expected) / (1.0 - expected));
        (Some(observed), kappa)
    };
    JudgeAgreement {
        judged: pairs.len(),
        both_correct,
        both_wrong,
        only_rule_correct,
        only_judge_correct,
        agreement,
        kappa,
    }
}

// ---- the error model's inputs and the gain's interval (the supplement) -----------------

/// Pairs of extractions of one incident from two different calls, by what the two state,
/// and how many of each are more alike than a bar (strictly, as the gate asks).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairCounts {
    /// Both correct, stating the same true record.
    pub same_state: usize,
    pub same_state_matched: usize,
    /// Both correct, stating different true records: a merge of the two is false.
    pub different_states: usize,
    pub different_states_matched: usize,
    /// Neither correct.
    pub wrong_wrong: usize,
    pub wrong_wrong_matched: usize,
    /// One correct, the other not.
    pub correct_wrong: usize,
    pub correct_wrong_matched: usize,
}

/// Count every pair of `items` of one incident from two different calls, once.
pub fn pair_counts(items: &[Item], likeness: &Likeness, bar: f64) -> PairCounts {
    let bar = bar as f32;
    let mut by_thread: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, item) in items.iter().enumerate() {
        by_thread.entry(item.thread.as_str()).or_default().push(i);
    }
    let mut c = PairCounts::default();
    for members in by_thread.values() {
        for (n, &a) in members.iter().enumerate() {
            for &b in &members[n + 1..] {
                if items[a].transcript == items[b].transcript {
                    continue;
                }
                let (pairs, matched) = match (items[a].correct(), items[b].correct()) {
                    (true, true) if items[a].stated == items[b].stated => {
                        (&mut c.same_state, &mut c.same_state_matched)
                    }
                    (true, true) => (&mut c.different_states, &mut c.different_states_matched),
                    (false, false) => (&mut c.wrong_wrong, &mut c.wrong_wrong_matched),
                    _ => (&mut c.correct_wrong, &mut c.correct_wrong_matched),
                };
                *pairs += 1;
                *matched += usize::from(likeness.between(a, b) > bar);
            }
        }
    }
    c
}

/// The supplement's Equation (p2): the precision of records validated by two
/// conversations that err independently, from the precision `p` of one extraction and the
/// match rates `m_c` (both correct, one state), `m_w` (both wrong) and `m_x` (one of each).
pub fn predicted_precision_two(p: f64, m_c: f64, m_w: f64, m_x: f64) -> Option<f64> {
    let correct = p * p * m_c;
    let all = correct + (1.0 - p).powi(2) * m_w + 2.0 * p * (1.0 - p) * m_x;
    (all > 0.0).then(|| correct / all)
}

/// One incident's promoted groups, and the correct ones, under the rule of one
/// conversation and under the rule of two.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IncidentTally {
    pub one_promoted: u32,
    pub one_correct: u32,
    pub two_promoted: u32,
    pub two_correct: u32,
}

/// The 95% percentile interval of precision(two) − precision(one) over `draws` bootstrap
/// resamples of the incidents (as many as there are, with replacement), drawn from the
/// stream `bootstrap/gate/precision-gain`. A resample in which either rule promoted
/// nothing has no precision and is left out. `None` without incidents.
pub fn precision_gain_interval(tallies: &[IncidentTally], seed: u64, draws: usize) -> Option<Interval> {
    if tallies.is_empty() {
        return None;
    }
    let n = tallies.len() as u64;
    let mut rng = stream(seed, "bootstrap/gate/precision-gain");
    let mut gains = Vec::with_capacity(draws);
    for _ in 0..draws {
        let mut sum = IncidentTally::default();
        for _ in 0..n {
            let t = &tallies[uniform_below(&mut rng, n) as usize];
            sum.one_promoted += t.one_promoted;
            sum.one_correct += t.one_correct;
            sum.two_promoted += t.two_promoted;
            sum.two_correct += t.two_correct;
        }
        if sum.one_promoted > 0 && sum.two_promoted > 0 {
            gains.push(
                f64::from(sum.two_correct) / f64::from(sum.two_promoted)
                    - f64::from(sum.one_correct) / f64::from(sum.one_promoted),
            );
        }
    }
    percentile_interval(&gains)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{record, truth};
    use crate::vectors::VectorIndex;
    use cb_types::Turn;

    #[test]
    fn a_text_is_case_folded_and_its_whitespace_collapsed() {
        assert_eq!(
            normalise("  The Fibre\tCut\n near  EXCHANGE. "),
            "the fibre cut near exchange."
        );
        assert_eq!(normalise("ÉTAT Über"), "état über");
        assert_eq!(normalise(""), "");
    }

    #[test]
    fn a_key_term_must_appear_as_whole_words() {
        let text = normalise("We found a fibre cut near the exchange; the fix is due at 4:30 pm.");
        assert!(term_appears(&text, &normalise("Fibre  Cut")));
        assert!(term_appears(&text, "exchange"));
        assert!(term_appears(&text, "4:30 pm"));
        assert!(term_appears(&text, "we"));
        // "fix" is there as a word, but "fixed" is not; "cut near the" is, "ut near" is not.
        assert!(term_appears(&text, "fix"));
        assert!(!term_appears(&text, "fixed"));
        assert!(!term_appears(&text, "ut near"));
        // A later occurrence counts when an earlier one is inside a word.
        assert!(term_appears("prefix fix", "fix"));
        assert!(!term_appears("prefix suffix", "fix"));
        assert!(!term_appears(&text, ""));
        assert!(all_terms_appear(&text, &[]));
        assert!(all_terms_appear(
            &text,
            &["fibre cut".to_string(), "exchange".to_string()]
        ));
        assert!(!all_terms_appear(
            &text,
            &["fibre cut".to_string(), "router".to_string()]
        ));
    }

    fn truth_table() -> Truth {
        let mut records = vec![
            Record {
                truth: truth(true, &["no cause yet"], Some(200)),
                ..record("i1.s1", 100)
            },
            Record {
                truth: truth(true, &["fibre cut"], Some(300)),
                ..record("i1.s2", 200)
            },
            Record {
                truth: truth(true, &["fibre cut", "restart the router"], None),
                ..record("i1.s3", 300)
            },
            Record {
                truth: truth(false, &["your own router", "is faulty"], None),
                ..record("i1.d1", 150)
            },
            Record {
                truth: truth(false, &[], None),
                ..record("i1.d2", 160)
            },
            record("doc", 10),
        ];
        for r in &mut records {
            if r.id.starts_with("i1") {
                r.thread = Some("i1".to_string());
            }
        }
        Truth::new(&records).unwrap()
    }

    #[test]
    fn an_extraction_states_the_newest_true_state_whose_terms_all_appear() {
        let truth = truth_table();
        let at = |id: &str| truth.index_of(id).unwrap();
        let states = [at("i1.s2"), at("i1.s3")];
        let wrong = truth.wrong_of("i1").to_vec();
        assert_eq!(wrong, [at("i1.d1")]);
        assert_eq!(truth.wrong_without_terms, 1);
        // s3's terms all appear, and so do s2's: the newest, s3, is named.
        let text = normalise("Fibre cut in the north; restart the router to get back online.");
        assert_eq!(
            stated_by(&text, &states, &wrong, &truth),
            Stated::True(at("i1.s3"))
        );
        // Only s2's terms.
        assert_eq!(
            stated_by("a fibre cut", &states, &wrong, &truth),
            Stated::True(at("i1.s2"))
        );
        // A true state of the incident that this call did not state does not count.
        assert_eq!(
            stated_by("no cause yet", &states, &wrong, &truth),
            Stated::Nothing
        );
        // A wrong statement in full makes the text wrong, even beside a true state.
        let both = normalise("A fibre cut. Also your own router is faulty.");
        assert_eq!(
            stated_by(&both, &states, &wrong, &truth),
            Stated::Wrong(at("i1.d1"))
        );
        // Part of a wrong statement's terms is not the wrong statement.
        let part = normalise("A fibre cut; your own router is fine.");
        assert_eq!(
            stated_by(&part, &states, &wrong, &truth),
            Stated::True(at("i1.s2"))
        );
        assert_eq!(
            stated_by("nothing useful", &states, &wrong, &truth),
            Stated::Nothing
        );
    }

    #[test]
    fn an_empty_key_term_is_refused_and_records_without_truth_are_skipped() {
        let bad = Record {
            truth: truth(true, &["  "], None),
            ..record("x", 1)
        };
        assert!(Truth::new(&[bad]).is_err());
        let table = truth_table();
        assert!(table.index_of("doc").is_none());
    }

    #[test]
    fn the_split_of_an_incident_follows_the_question_rule() {
        // inc-0006 hashes to a multiple of 5 (validation); inc-0001 does not (see aggregate.rs).
        assert_eq!(split_of_thread("inc-0006"), Split::Validation);
        assert_eq!(split_of_thread("inc-0001"), Split::Test);
    }

    fn call(turns: &[(Speaker, &str, f32)]) -> Transcript {
        Transcript {
            id: "t".to_string(),
            thread: "i1".to_string(),
            started_at: 1_000,
            agent: "a".to_string(),
            persona: String::new(),
            turns: turns
                .iter()
                .map(|(who, text, ends)| Turn {
                    who: *who,
                    text: text.to_string(),
                    ends_at_s: *ends,
                })
                .collect(),
            states: Vec::new(),
            asr_error_rate: 0.0,
        }
    }

    #[test]
    fn a_state_is_stated_at_the_end_of_the_agent_turn_that_completes_its_terms() {
        let transcript = call(&[
            (
                Speaker::Customer,
                "Is there a fibre cut? Restart the router?",
                10.0,
            ),
            (Speaker::Agent, "Let me look.", 20.0),
            (Speaker::Agent, "There is a fibre cut.", 35.5),
            (Speaker::Customer, "Okay.", 40.0),
            (Speaker::Agent, "Please restart the router.", 52.0),
        ]);
        let terms = |t: &[&str]| -> Vec<String> { t.iter().map(|s| normalise(s)).collect() };
        // The customer's words do not count; the terms are complete after the agent's third turn.
        assert_eq!(
            statement_moment(&transcript, &terms(&["fibre cut", "restart the router"])),
            Some(1_052.0)
        );
        assert_eq!(
            statement_moment(&transcript, &terms(&["fibre cut"])),
            Some(1_035.5)
        );
        // No terms: the end of the agent's first turn.
        assert_eq!(statement_moment(&transcript, &[]), Some(1_020.0));
        assert_eq!(statement_moment(&transcript, &terms(&["power cut"])), None);
        assert_eq!(call_end(&transcript), 1_052.0);
    }

    /// Items with given agents, arrival times, vectors and labels.
    fn items(rows: &[(&str, f64, [f32; 2], Stated, f32)]) -> (Vec<Item>, Likeness) {
        let list: Vec<Item> = rows
            .iter()
            .enumerate()
            .map(|(i, (agent, at, _, stated, confidence))| Item {
                transcript: format!("t{i}"),
                agent: agent.to_string(),
                thread: "i1".to_string(),
                arrives_at: *at,
                confidence: Confidence {
                    problem: *confidence,
                    solution: 1.0,
                    time: 1.0,
                },
                problem: format!("problem {i}"),
                solution: format!("solution {i}"),
                true_states: Vec::new(),
                stated: *stated,
            })
            .collect();
        let index = VectorIndex {
            dim: 2,
            ids: (0..rows.len()).map(|i| format!("t{i}")).collect(),
        };
        let values: Vec<f32> = rows.iter().flat_map(|r| r.2).collect();
        let vectors = Vectors::new(index, values).unwrap();
        let rows_of: Vec<usize> = (0..rows.len()).collect();
        let likeness = Likeness::compute(&vectors, &rows_of, 3).unwrap();
        (list, likeness)
    }

    fn rule(conversations: usize, agreement: Agreement, independence: Independence) -> GateRule {
        GateRule {
            conversations,
            agreement,
            independence,
            confidence_cut: None,
            bar: 0.92,
        }
    }

    fn groups(promoted: &[Promoted]) -> Vec<Vec<usize>> {
        promoted.iter().map(|p| p.members.clone()).collect()
    }

    #[test]
    fn likeness_is_kept_once_per_pair_and_read_either_way() {
        let (_, likeness) = items(&[
            ("a", 1.0, [1.0, 0.0], Stated::Nothing, 1.0),
            ("b", 2.0, [0.0, 1.0], Stated::Nothing, 1.0),
            ("c", 3.0, [1.0, 1.0], Stated::Nothing, 1.0),
        ]);
        assert_eq!(likeness.between(0, 1), 0.0);
        assert_eq!(likeness.between(1, 0), 0.0);
        // cos((1,0), (1,1)) = 1/sqrt 2
        assert_eq!(likeness.between(0, 2), std::f32::consts::FRAC_1_SQRT_2);
        assert_eq!(likeness.between(2, 1), likeness.between(1, 2));
    }

    #[test]
    fn with_two_conversations_the_second_alike_extraction_promotes_both() {
        // 0 and 2 say the same thing (cos 0.995); 1 says something else; 3 is like 0 and 2.
        let (list, likeness) = items(&[
            ("a", 10.0, [1.0, 0.0], Stated::Nothing, 1.0),
            ("b", 20.0, [0.0, 1.0], Stated::Nothing, 1.0),
            ("c", 30.0, [1.0, 0.1], Stated::Nothing, 1.0),
            ("d", 40.0, [1.0, 0.05], Stated::Nothing, 1.0),
        ]);
        let order = [0, 1, 2, 3];
        let promoted = replay(
            &rule(2, Agreement::Pairs, Independence::Calls),
            &list,
            &order,
            &likeness,
        );
        // 2 arrives and finds 0 pending: both promoted at 30. 3 finds only 1 pending (unlike): it waits.
        assert_eq!(groups(&promoted), [vec![0, 2]]);
        assert_eq!(promoted[0].at, 30.0);
        // With one conversation every extraction is promoted when it arrives.
        let single = replay(
            &rule(1, Agreement::Pairs, Independence::Calls),
            &list,
            &order,
            &likeness,
        );
        assert_eq!(groups(&single), [vec![0], vec![1], vec![2], vec![3]]);
        // With three, 0, 2 and 3 must all be alike pair by pair: they are, and 3 completes them.
        let three = replay(
            &rule(3, Agreement::Pairs, Independence::Calls),
            &list,
            &order,
            &likeness,
        );
        assert_eq!(groups(&three), [vec![0, 2, 3]]);
        assert_eq!(three[0].at, 40.0);
    }

    #[test]
    fn the_bar_is_strict_and_the_most_alike_pending_extraction_is_taken() {
        let (list, likeness) = items(&[
            ("a", 10.0, [1.0, 0.0], Stated::Nothing, 1.0),
            ("b", 20.0, [1.0, 0.11], Stated::Nothing, 1.0),
            ("c", 30.0, [1.0, 0.1], Stated::Nothing, 1.0),
        ]);
        let promoted = replay(
            &rule(2, Agreement::Pairs, Independence::Calls),
            &list,
            &[0, 1, 2],
            &likeness,
        );
        // 1 arrives first and is alike to 0 (cos 0.9940): they are promoted together; 2 then waits.
        assert_eq!(groups(&promoted), [vec![0, 1]]);
        let mut strict = rule(2, Agreement::Pairs, Independence::Calls);
        strict.bar = f64::from(likeness.between(0, 1));
        // A bar equal to the likeness is not passed: 1 cannot join 0, and 2 then picks 1 (more alike).
        assert_eq!(
            groups(&replay(&strict, &list, &[0, 1, 2], &likeness)),
            [vec![1, 2]]
        );
    }

    #[test]
    fn calls_by_one_agent_are_not_independent_when_agents_must_differ() {
        let (list, likeness) = items(&[
            ("a", 10.0, [1.0, 0.0], Stated::Nothing, 1.0),
            ("a", 20.0, [1.0, 0.01], Stated::Nothing, 1.0),
            ("b", 30.0, [1.0, 0.02], Stated::Nothing, 1.0),
        ]);
        let order = [0, 1, 2];
        assert_eq!(
            groups(&replay(
                &rule(2, Agreement::Pairs, Independence::Calls),
                &list,
                &order,
                &likeness
            )),
            [vec![0, 1]]
        );
        // Agents must differ: 1 cannot pair with 0 (both agent a). 2 (agent b) is alike to both and
        // takes the more alike, 1 (cos 0.99995 against 0.9998).
        assert_eq!(
            groups(&replay(
                &rule(2, Agreement::Pairs, Independence::Agents),
                &list,
                &order,
                &likeness
            )),
            [vec![1, 2]]
        );
        // Average: agent a's two calls form a group that waits; b joins it and makes two agents.
        assert_eq!(
            groups(&replay(
                &rule(2, Agreement::Average, Independence::Agents),
                &list,
                &order,
                &likeness
            )),
            [vec![0, 1, 2]]
        );
    }

    #[test]
    fn the_average_rule_joins_the_group_the_newcomer_is_most_like_on_average() {
        // 0 and 1 are not alike (cos 0.8944 < 0.92), so 1 starts its own group. 2 is alike to both
        // groups: its average likeness to {0} is cos(2,0) = 0.9701 and to {1} cos(2,1) = 0.9762, so it
        // joins {1}.
        let (list, likeness) = items(&[
            ("a", 10.0, [1.0, 0.0], Stated::Nothing, 1.0),
            ("b", 20.0, [1.0, 0.5], Stated::Nothing, 1.0),
            ("c", 30.0, [1.0, 0.25], Stated::Nothing, 1.0),
        ]);
        let promoted = replay(
            &rule(2, Agreement::Average, Independence::Calls),
            &list,
            &[0, 1, 2],
            &likeness,
        );
        assert_eq!(groups(&promoted), [vec![1, 2]]);
    }

    #[test]
    fn the_confidence_cut_drops_extractions_below_it() {
        let (list, likeness) = items(&[
            ("a", 10.0, [1.0, 0.0], Stated::Nothing, 0.84),
            ("b", 20.0, [1.0, 0.01], Stated::Nothing, 0.85),
            ("c", 30.0, [1.0, 0.02], Stated::Nothing, 0.90),
        ]);
        let mut cut = rule(2, Agreement::Pairs, Independence::Calls);
        cut.confidence_cut = Some(0.85);
        // 0 (0.84) is dropped; 1 (0.85, not below the cut) waits; 2 joins it.
        assert_eq!(groups(&replay(&cut, &list, &[0, 1, 2], &likeness)), [vec![1, 2]]);
    }

    #[test]
    fn precision_false_merges_share_and_wait_follow_their_definitions() {
        let truth = truth_table();
        let at = |id: &str| truth.index_of(id).unwrap();
        let (mut list, likeness) = items(&[
            ("a", 1_100.0, [1.0, 0.0], Stated::True(at("i1.s2")), 1.0),
            ("b", 1_400.0, [1.0, 0.01], Stated::True(at("i1.s2")), 0.9),
            ("c", 2_000.0, [1.0, 0.02], Stated::True(at("i1.s3")), 1.0),
            ("d", 2_500.0, [0.0, 1.0], Stated::Wrong(at("i1.d1")), 1.0),
            ("e", 2_600.0, [0.0, 1.0], Stated::Wrong(at("i1.d1")), 1.0),
        ]);
        list[0].problem = "internet down".to_string();
        list[0].solution = "a fibre cut".to_string();
        list[1].solution = "a fibre cut in town".to_string();
        list[2].solution = "fibre cut; restart the router".to_string();
        list[3].solution = "your own router is faulty".to_string();
        list[4].solution = "your own router is faulty".to_string();
        for item in &mut list[..3] {
            item.true_states = vec![at("i1.s2"), at("i1.s3")];
        }
        let order = [0, 1, 2, 3, 4];
        let pairs = rule(2, Agreement::Pairs, Independence::Calls);
        let promoted = replay(&pairs, &list, &order, &likeness);
        // {0, 1} at 1 400: both state s2 and the merged text states s2: correct.
        // {3, 4} at 2 600: both state the wrong statement d1: not correct, not a false merge.
        // 2 waits (its only alike pending partner was taken).
        assert_eq!(groups(&promoted), [vec![0, 1], vec![3, 4]]);
        let states: BTreeSet<usize> = [at("i1.s2"), at("i1.s3")].into_iter().collect();
        let first: HashMap<usize, f64> = [(at("i1.s2"), 1_050.0), (at("i1.s3"), 1_900.0)]
            .into_iter()
            .collect();
        // Calls 0, 1 and 2 each state s2 and s3: both are stated in two or more calls.
        let outcome = score(&pairs, &list, &promoted, &truth, (&states, &states), &first);
        assert_eq!(
            (outcome.promoted, outcome.correct, outcome.false_merges),
            (2, 1, 0)
        );
        assert_eq!(outcome.precision, Some(0.5));
        // s2 validated at 1 400, first stated at 1 050: a wait of 350 s. s3 never validated.
        assert_eq!((outcome.states, outcome.states_validated), (2, 1));
        assert_eq!(outcome.share_validated, Some(0.5));
        assert_eq!(
            (outcome.states_in_two_calls, outcome.share_validated_in_two_calls),
            (2, Some(0.5))
        );
        // Counted over a record stated twice only (s3): it was never validated.
        let only_s3: BTreeSet<usize> = [at("i1.s3")].into_iter().collect();
        let outcome = score(&pairs, &list, &promoted, &truth, (&states, &only_s3), &first);
        assert_eq!(outcome.share_validated_in_two_calls, Some(0.0));
        let outcome = score(
            &pairs,
            &list,
            &promoted,
            &truth,
            (&states, &BTreeSet::new()),
            &first,
        );
        assert_eq!(outcome.share_validated_in_two_calls, None);
        assert_eq!(
            (outcome.wait_median_s, outcome.wait_p90_s),
            (Some(350.0), Some(350.0))
        );
        // Three conversations: 0, 1 and 2 are alike; their members state s2, s2 and s3: a false
        // merge, and not correct although the merged text states a true record.
        let three = rule(3, Agreement::Pairs, Independence::Calls);
        let promoted = replay(&three, &list, &order, &likeness);
        assert_eq!(groups(&promoted), [vec![0, 1, 2]]);
        let outcome = score(&three, &list, &promoted, &truth, (&states, &states), &first);
        assert_eq!(
            (outcome.promoted, outcome.correct, outcome.false_merges),
            (1, 0, 1)
        );
        assert_eq!(outcome.precision, Some(0.0));
        assert_eq!(outcome.share_validated, Some(0.0));
        assert_eq!(outcome.wait_median_s, None);
        // With one conversation every extraction is its own record: 3 of 5 correct.
        let single = rule(1, Agreement::Pairs, Independence::Calls);
        let outcome = score(
            &single,
            &list,
            &replay(&single, &list, &order, &likeness),
            &truth,
            (&states, &states),
            &first,
        );
        assert_eq!((outcome.promoted, outcome.correct), (5, 3));
        // s2 validated at 1 100 (wait 50 s), s3 at 2 000 (wait 100 s): median 75, P90 100.
        assert_eq!(
            (outcome.wait_median_s, outcome.wait_p90_s),
            (Some(75.0), Some(100.0))
        );
    }

    #[test]
    fn the_merge_takes_the_most_confident_problem_and_solution() {
        let (mut list, _) = items(&[
            ("a", 1.0, [1.0, 0.0], Stated::Nothing, 0.5),
            ("b", 2.0, [1.0, 0.0], Stated::Nothing, 0.9),
        ]);
        list[0].confidence.solution = 0.95;
        list[1].confidence.solution = 0.9;
        // Problem from 1 (0.9 > 0.5), solution from 0 (0.95 > 0.9).
        assert_eq!(merged_text(&[0, 1], &list), "problem 1 solution 0");
        // Equal confidence: the earlier member.
        list[1].confidence.problem = 0.5;
        assert_eq!(merged_text(&[0, 1], &list), "problem 0 solution 0");
    }

    #[test]
    fn agreement_and_kappa_of_a_worked_example() {
        // 10 judged: both correct 6, both wrong 2, only the rule says correct 1, only the judge 1.
        // p_o = 8/10 = 0.8. Rule says correct 7/10, judge 7/10.
        // p_e = 0.7 * 0.7 + 0.3 * 0.3 = 0.58. kappa = (0.8 - 0.58) / (1 - 0.58) = 0.22 / 0.42 = 0.5238.
        let mut pairs = vec![(true, true); 6];
        pairs.extend([(false, false); 2]);
        pairs.push((true, false));
        pairs.push((false, true));
        let a = agreement_of(&pairs);
        assert_eq!(
            (
                a.judged,
                a.both_correct,
                a.both_wrong,
                a.only_rule_correct,
                a.only_judge_correct
            ),
            (10, 6, 2, 1, 1)
        );
        assert_eq!(a.agreement, Some(0.8));
        assert!((a.kappa.unwrap() - 0.22 / 0.42).abs() < 1e-12);
        // Everyone says correct: agreement 1, kappa undefined.
        let all = agreement_of(&[(true, true); 3]);
        assert_eq!((all.agreement, all.kappa), (Some(1.0), None));
        assert_eq!(agreement_of(&[]).agreement, None);
    }

    #[test]
    fn pairs_are_counted_within_an_incident_by_what_the_two_state() {
        let (mut list, likeness) = items(&[
            ("a", 1.0, [1.0, 0.0], Stated::True(1), 1.0),
            ("a", 2.0, [1.0, 0.0], Stated::True(1), 1.0),
            ("a", 3.0, [0.0, 1.0], Stated::True(2), 1.0),
            ("a", 4.0, [1.0, 0.0], Stated::Wrong(5), 1.0),
            ("a", 5.0, [1.0, 0.0], Stated::Nothing, 1.0),
            ("a", 6.0, [0.6, 0.8], Stated::Nothing, 1.0),
        ]);
        // Item 4 belongs to another incident: it is in no pair.
        list[4].thread = "i2".to_string();
        let counts = pair_counts(&list, &likeness, 0.92);
        assert_eq!(
            counts,
            PairCounts {
                same_state: 1,
                same_state_matched: 1,
                different_states: 2,
                different_states_matched: 0,
                // 0-3 and 1-3 alike; 2-3 and the three pairs with item 5 not.
                correct_wrong: 6,
                correct_wrong_matched: 2,
                wrong_wrong: 1,
                wrong_wrong_matched: 0,
            }
        );
        // The bar is strict: at 1.0 not even identical vectors match.
        assert_eq!(pair_counts(&list, &likeness, 1.0).same_state_matched, 0);
    }

    #[test]
    fn the_error_model_follows_its_equation() {
        // p = 0.5: correct 0.25 * 0.9, wrong 0.25 * 0.1, mixed 2 * 0.25 * 0.1; 0.225 / 0.3.
        let p2 = predicted_precision_two(0.5, 0.9, 0.1, 0.1).unwrap();
        assert!((p2 - 0.75).abs() < 1e-12, "{p2}");
        assert_eq!(predicted_precision_two(0.5, 0.0, 0.0, 0.0), None);
    }

    #[test]
    fn the_gain_interval_resamples_incidents() {
        assert_eq!(precision_gain_interval(&[], 1, 100), None);
        // Every incident: one of two right alone, both right with two conversations, so every
        // resample gains exactly 50 points.
        let even = vec![
            IncidentTally {
                one_promoted: 2,
                one_correct: 1,
                two_promoted: 1,
                two_correct: 1,
            };
            40
        ];
        let interval = precision_gain_interval(&even, 7, 1_000).unwrap();
        assert!((interval.lo - 0.5).abs() < 1e-12 && (interval.hi - 0.5).abs() < 1e-12);
        // Incidents that differ: the interval holds the observed gain, and one seed gives one interval.
        let mixed: Vec<IncidentTally> = (0..60u32)
            .map(|i| IncidentTally {
                one_promoted: 4,
                one_correct: i % 4,
                two_promoted: 2,
                two_correct: u32::from(i % 3 != 0) + 1,
            })
            .collect();
        let sum = |f: fn(&IncidentTally) -> u32| f64::from(mixed.iter().map(f).sum::<u32>());
        let observed = sum(|t| t.two_correct) / sum(|t| t.two_promoted)
            - sum(|t| t.one_correct) / sum(|t| t.one_promoted);
        let interval = precision_gain_interval(&mixed, 7, 2_000).unwrap();
        assert!(
            interval.lo < observed && observed < interval.hi,
            "{interval:?} {observed}"
        );
        assert_eq!(precision_gain_interval(&mixed, 7, 2_000), Some(interval));
    }
}
