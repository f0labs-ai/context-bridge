//! The shape every public source is read into, and the one set of rules that
//! turns it into records and questions (docs/PUBLIC_DATA.md).
//!
//! A loader hands over [`Incident`]s: dated updates, oldest first, each with
//! what the incident affected at that point and whether the incident was still
//! open. [`build`] keeps the complete ones, writes one record per update, and
//! asks questions at moments drawn from a seeded generator. Which incidents a
//! question is about is decided in one place, `Threads::fitting`: the
//! incidents in progress at that moment that the question's wording fits.

use super::text::squash;
use super::time;
use super::wording::{self, Intent};
use cb_types::{Gold, Phase, Query, QueryClass, Record, RecordTruth, Tier, UnixSeconds, fnv1a64, ids};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use thiserror::Error;

/// What tells one source's records from another's.
#[derive(Clone, Debug)]
pub struct SourceSpec {
    /// The data set's name ("github").
    pub name: &'static str,
    /// `Record.source` of an incident update ("github-status").
    pub record_source: &'static str,
    /// First part of every id ("gh").
    pub id_prefix: &'static str,
    /// How finely the source times an update, in seconds: an update stamped `t`
    /// was posted somewhere in `[t, t + precision)`. No question is asked
    /// inside that span of an update it depends on.
    pub time_precision_s: i64,
    /// Of the questions about one incident that names a component, the share
    /// worded with the service's name instead ("Is something wrong with GitHub
    /// right now?"), drawn per question from its own generator. A question
    /// about an incident that names nothing always names the service.
    pub service_question_share: f64,
}

impl SourceSpec {
    /// An update's time as its record's text prints it, as finely as the source times it.
    pub fn printed_time(&self, t: UnixSeconds) -> String {
        time::printed_utc(t, self.time_precision_s >= 60)
    }
}

/// Whether an incident is still going on as of an update.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Life {
    /// In progress.
    Open,
    /// The source gives mixed signs: it may be over.
    Unclear,
    /// Over.
    Over,
}

/// One update of an incident, as the source gave it.
#[derive(Clone, Debug, PartialEq)]
pub struct Update {
    /// When it was posted.
    pub at: UnixSeconds,
    /// When its text last changed, when the source says that was after `at`.
    pub edited_at: Option<UnixSeconds>,
    pub phase: Phase,
    pub life: Life,
    /// The incident's name when this update was posted, as far as the source tells.
    pub title: String,
    /// The status word shown with the update ("Investigating"), if any.
    pub label: Option<String>,
    /// Plain text.
    pub body: String,
    /// What the incident affected as of this update, each the way a caller would
    /// name it ("GitHub Actions"). Empty when the source names nothing: the
    /// incident may then affect anything of its service.
    pub components: Vec<String>,
    /// Whether the update states a way around the problem.
    pub workaround: bool,
}

/// One incident: the source's own thread of updates.
#[derive(Clone, Debug, PartialEq)]
pub struct Incident {
    /// The source's id of the incident, in id characters (`a-z`, `0-9`, `-`, `_`).
    pub code: String,
    /// The service it belongs to, the way a caller would name it ("GitHub").
    pub service: String,
    /// Oldest first.
    pub updates: Vec<Update>,
}

/// The numbers of docs/PUBLIC_DATA.md.
#[derive(Clone, Debug)]
pub struct Rules {
    /// An incident is kept when it has at least this many updates.
    pub min_updates: usize,
    /// A gap between two updates gets a question when it lasts at least this long.
    pub min_gap_s: i64,
    /// A post-incident question is asked this long after the last update (from, to).
    pub post_incident_after_s: (i64, i64),
    /// A workaround question is asked within this long of the update that first states one.
    pub workaround_within_s: i64,
    /// An edit within this long of posting counts as part of the post.
    pub edit_grace_s: i64,
    /// How many moments are tried before a question is given up.
    pub draws: u32,
    /// An incident the source never closes counts as in progress until this
    /// long after its last update: the source does not say when it ended.
    pub unclosed_open_for_s: i64,
}

impl Default for Rules {
    fn default() -> Self {
        Self {
            min_updates: 3,
            min_gap_s: 120,
            post_incident_after_s: (3_600, 172_800),
            workaround_within_s: 600,
            edit_grace_s: 120,
            draws: 8,
            unclosed_open_for_s: 172_800,
        }
    }
}

#[derive(Debug, Error)]
pub enum TimelineError {
    #[error("incident {code:?}: its updates are not in time order")]
    OutOfOrder { code: String },
    #[error("incident {code:?}: it is open again after an update that said it was over")]
    ReopensAfterOver { code: String },
    #[error("two incidents share the code {code:?}")]
    DuplicateCode { code: String },
    #[error(transparent)]
    Invalid(#[from] InvalidDataSet),
}

/// A broken rule of the data set contract, found before anything is written.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum InvalidDataSet {
    #[error("{0}")]
    Id(String),
    #[error("record id {0:?} is used twice")]
    SameRecordId(String),
    #[error("question id {0:?} is used twice")]
    SameQueryId(String),
    #[error(
        "records {first:?} and {second:?} have the same text; the search index would keep only one of them"
    )]
    SameText { first: String, second: String },
    #[error("question {query:?} grades record {record:?}, which does not exist")]
    UnknownGold { query: String, record: String },
    #[error("question {query:?} is answered by record {record:?}, written after the question was asked")]
    AnswerFromTheFuture { query: String, record: String },
    #[error("question {query:?} has no record of grade 3")]
    NoAnswer { query: String },
}

/// Why an incident was not kept.
pub const LEFT_OUT_FEW_UPDATES: &str = "fewer updates than the rule asks for";
pub const LEFT_OUT_NOT_CLOSED: &str = "not closed in the source";
/// Why a question was not asked.
pub const NOT_ASKED_SHARED: &str = "changing: another incident in progress fits every wording";
pub const NOT_ASKED_EDITED: &str = "changing: the current update was rewritten after the next one";
pub const NOT_ASKED_POST: &str = "post_incident: another incident fits every wording";
pub const NOT_ASKED_WORKAROUND: &str = "workaround: no clear moment";
pub const NOT_ASKED_AMBIGUOUS: &str = "ambiguous: no clear moment";
pub const NOT_ASKED_WORDING: &str = "every phrasing would copy the answer";
pub const NOT_ASKED_HORIZON: &str = "post_incident: the source ends less than an hour after the incident";

/// What a conversion did, for the report and `stats.json`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Stats {
    pub incidents_read: u64,
    pub incidents_kept: u64,
    pub incidents_left_out: BTreeMap<String, u64>,
    /// Updates of the kept incidents, as the source has them.
    pub updates: u64,
    /// Records written for them.
    pub records: u64,
    /// Closing updates that were rewritten later and so became two records.
    pub closing_rewrites: u64,
    /// Updates that repeat the one before them word for word at the same printed
    /// time (the page posted it twice): no record of their own.
    pub repeats_merged: u64,
    /// Updates whose text without its time (`"<name>. <Status>: <body>"`) is the
    /// same as an earlier update's in the data set, and the number of such texts.
    pub identical_without_time: u64,
    pub identical_without_time_texts: u64,
    /// Questions by class.
    pub questions: BTreeMap<String, u64>,
    /// Of the changing questions, those asking for a workaround.
    pub workaround_questions: u64,
    /// Questions by class that name the service as a whole rather than a component.
    pub questions_naming_the_service: BTreeMap<String, u64>,
    pub questions_not_asked: BTreeMap<String, u64>,
    pub first_update_at: Option<UnixSeconds>,
    pub last_update_at: Option<UnixSeconds>,
    pub updates_per_incident_median: f64,
    pub updates_per_incident_max: u64,
    /// Median time between two consecutive updates of an incident, seconds.
    pub gap_median_s: f64,
}

pub struct Built {
    pub records: Vec<Record>,
    pub queries: Vec<Query>,
    pub stats: Stats,
}

// ---- threads -----------------------------------------------------------------------

/// One record-to-be of a kept incident.
#[derive(Clone, Debug)]
struct Step {
    at: UnixSeconds,
    /// From this moment the step's text is known to be what the source showed.
    ready_at: UnixSeconds,
    phase: Phase,
    life: Life,
    title: String,
    label: Option<String>,
    body: String,
    components: Vec<String>,
    workaround: bool,
}

#[derive(Clone, Debug)]
struct Thread {
    /// `<prefix>-<code>`: `Record.thread`.
    id: String,
    code: String,
    service: String,
    /// When the source's last update of the incident was posted.
    last_posted_at: UnixSeconds,
    steps: Vec<Step>,
    /// A kept incident has records and questions. One that was left out has
    /// neither, but while it is in progress no question is asked that its
    /// wording would fit too.
    kept: bool,
}

/// What a question names.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Topic {
    /// A component of the service ("GitHub Actions").
    Component { service: String, name: String },
    /// The service as a whole ("GitHub"), when nothing more specific can be named.
    Service(String),
}

impl Topic {
    fn name(&self) -> &str {
        match self {
            Topic::Component { name, .. } | Topic::Service(name) => name,
        }
    }
}

impl Thread {
    fn start(&self) -> UnixSeconds {
        self.steps.first().map_or(i64::MAX, |s| s.at)
    }

    fn record_id(&self, step: usize) -> String {
        format!("{}.{}", self.id, step + 1)
    }

    /// The step that is current at `t`: the last one posted at or before it.
    fn current(&self, t: UnixSeconds) -> Option<usize> {
        self.steps.partition_point(|s| s.at <= t).checked_sub(1)
    }

    /// The first moment from which the incident is certainly over.
    fn over_at(&self) -> Option<UnixSeconds> {
        self.steps.iter().find(|s| s.life == Life::Over).map(|s| s.at)
    }

    /// Whether step `step` fits a question naming `topic`. A step that names no
    /// component may concern any component of its service.
    fn fits(&self, step: usize, topic: &Topic) -> bool {
        match topic {
            Topic::Service(service) => self.service == *service,
            Topic::Component { service, name } => {
                let components = &self.steps[step].components;
                self.service == *service && (components.is_empty() || components.contains(name))
            }
        }
    }

    /// Whether step `step` names `topic` in so many words.
    fn names(&self, step: usize, topic: &Topic) -> bool {
        match topic {
            Topic::Service(service) => self.service == *service,
            Topic::Component { service, name } => {
                self.service == *service && self.steps[step].components.contains(name)
            }
        }
    }

    /// Whether any step of the incident may concern `topic`.
    fn may_concern(&self, topic: &Topic) -> bool {
        (0..self.steps.len()).any(|step| self.fits(step, topic))
    }

    /// Grade 3 for step `current`, grade 1 for the steps before it.
    fn gold(&self, current: usize) -> Vec<Gold> {
        let mut gold = vec![Gold {
            record: self.record_id(current),
            grade: 3,
        }];
        gold.extend((0..current).map(|step| Gold {
            record: self.record_id(step),
            grade: 1,
        }));
        gold
    }

    /// The components the given steps name, as topics, in a fixed order.
    fn component_topics(&self, steps: impl Iterator<Item = usize>) -> Vec<Topic> {
        let names: BTreeSet<&String> = steps.flat_map(|s| self.steps[s].components.iter()).collect();
        names
            .into_iter()
            .map(|name| Topic::Component {
                service: self.service.clone(),
                name: name.clone(),
            })
            .collect()
    }
}

struct Threads {
    /// Sorted by start, then code.
    all: Vec<Thread>,
    /// See [`Rules::unclosed_open_for_s`].
    unclosed_open_for_s: i64,
}

impl Threads {
    /// The current step of `thread` at `t`, if the incident may be in progress then.
    fn in_progress(&self, thread: &Thread, t: UnixSeconds) -> Option<usize> {
        let current = thread.current(t)?;
        if thread.steps[current].life == Life::Over {
            return None;
        }
        let never_closed = thread.over_at().is_none();
        if never_closed && t > thread.last_posted_at + self.unclosed_open_for_s {
            return None;
        }
        Some(current)
    }

    /// The incidents that may be in progress at `t` and that a question naming
    /// `topic` fits, each with its current step. This is the one place that
    /// decides which incidents a question is about.
    fn fitting(&self, topic: &Topic, t: UnixSeconds) -> Vec<(usize, usize)> {
        let mut fits = Vec::new();
        for (index, thread) in self.all.iter().enumerate() {
            if thread.start() > t {
                break;
            }
            let Some(current) = self.in_progress(thread, t) else {
                continue;
            };
            if thread.fits(current, topic) {
                fits.push((index, current));
            }
        }
        fits
    }

    /// Whether, at `t`, every fitting incident is certainly in progress and its
    /// current step's text is known.
    fn clear(&self, fits: &[(usize, usize)], t: UnixSeconds) -> bool {
        fits.iter().all(|&(thread, step)| {
            let step = &self.all[thread].steps[step];
            step.life == Life::Open && t >= step.ready_at
        })
    }

    /// A topic that fits incident `thread` and no other at `t`: one of its
    /// components, or the service as a whole when it names none, when none fits
    /// it alone, or when `service_first` asks for the service and it fits.
    fn topic_for_one(
        &self,
        thread: usize,
        step: usize,
        t: UnixSeconds,
        service_first: bool,
        rng: &mut ChaCha20Rng,
    ) -> Option<Topic> {
        let this = &self.all[thread];
        let only_this = |topic: &Topic| {
            let fits = self.fitting(topic, t);
            fits == [(thread, step)] && self.clear(&fits, t)
        };
        let service = Topic::Service(this.service.clone());
        if service_first && only_this(&service) {
            return Some(service);
        }
        let usable: Vec<Topic> = this
            .component_topics(std::iter::once(step))
            .into_iter()
            .filter(|topic| only_this(topic))
            .collect();
        if !usable.is_empty() {
            return Some(usable[rng.random_range(0..usable.len())].clone());
        }
        only_this(&service).then_some(service)
    }
}

/// Whether a question is to be worded with the service's name, drawn from the
/// question's own generator. Nothing is drawn when the share is zero.
fn service_first(share: f64, rng: &mut ChaCha20Rng) -> bool {
    share > 0.0 && rng.random_range(0.0..1.0) < share
}

/// The generator of one question: a stream of its own, so that a question's
/// moment and wording do not depend on which other questions exist.
fn rng_for(seed: u64, question_id: &str) -> ChaCha20Rng {
    let mut key = seed.to_le_bytes().to_vec();
    key.push(0);
    key.extend_from_slice(question_id.as_bytes());
    ChaCha20Rng::seed_from_u64(fnv1a64(&key))
}

fn median(values: &mut [i64]) -> f64 {
    values.sort_unstable();
    match values.len() {
        0 => 0.0,
        n if n % 2 == 1 => values[n / 2] as f64,
        n => (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0,
    }
}

fn bump(counts: &mut BTreeMap<String, u64>, key: &str) {
    *counts.entry(key.to_string()).or_insert(0) += 1;
}

fn class_name(class: QueryClass) -> &'static str {
    match class {
        QueryClass::Changing => "changing",
        QueryClass::PostIncident => "post_incident",
        QueryClass::Ambiguous => "ambiguous",
        QueryClass::Ordinary => "ordinary",
        QueryClass::OrdinaryNear => "ordinary_near",
    }
}

/// `"<incident name>. Update <time> UTC, <Status>: <body>"`, leaving out what is
/// absent. The time makes the text of every update its own: status pages
/// repeat updates word for word ("We are continuing to investigate"), and the
/// search index keeps one passage per text.
fn record_text(spec: &SourceSpec, step: &Step) -> String {
    let title = step.title.trim().trim_end_matches('.');
    let time = spec.printed_time(step.at);
    match (&step.label, step.body.is_empty()) {
        (Some(label), false) => format!("{title}. Update {time}, {label}: {}", step.body),
        (Some(label), true) => format!("{title}. Update {time}, {label}."),
        (None, false) => format!("{title}. Update {time}: {}", step.body),
        (None, true) => format!("{title}. Update {time}."),
    }
}

/// An update's text as docs/PUBLIC_DATA.md first wrote it, without its time.
fn text_without_time(update: &Update) -> String {
    let title = squash(update.title.trim().trim_end_matches('.'));
    let body = squash(&update.body);
    match &update.label {
        Some(label) => format!("{title}. {label}: {body}"),
        None => format!("{title}. {body}"),
    }
}

/// The kept incident as a thread of steps.
///
/// A closing update that was rewritten well after it was posted becomes two
/// steps: at its own time only what is certain (the incident's name and
/// status), and its text from the moment the text took its final form. An
/// update that repeats the step before it word for word at the same printed
/// time adds nothing and is folded into it.
fn thread_of(spec: &SourceSpec, rules: &Rules, incident: &Incident, kept: bool, stats: &mut Stats) -> Thread {
    let p = spec.time_precision_s;
    let last = incident.updates.len().saturating_sub(1);
    let mut steps: Vec<Step> = Vec::with_capacity(incident.updates.len() + 1);
    for (i, update) in incident.updates.iter().enumerate() {
        let rewritten_at = update.edited_at.filter(|e| e - update.at > rules.edit_grace_s);
        let step = Step {
            at: update.at,
            ready_at: update.at + p,
            phase: update.phase,
            life: update.life,
            title: squash(&update.title),
            label: update.label.clone(),
            body: update.body.clone(),
            components: update.components.clone(),
            workaround: update.workaround,
        };
        let new_steps = match rewritten_at {
            Some(rewritten_at) if i == last => {
                stats.closing_rewrites += 1;
                let bare = Step {
                    body: String::new(),
                    workaround: false,
                    ..step.clone()
                };
                vec![
                    bare,
                    Step {
                        at: rewritten_at,
                        ready_at: rewritten_at + p,
                        ..step
                    },
                ]
            }
            // Rewritten in the middle of the timeline: the record stays where it was
            // posted, and it answers no question before its text was final.
            Some(rewritten_at) => vec![Step {
                ready_at: rewritten_at + p,
                ..step
            }],
            None => vec![step],
        };
        for step in new_steps {
            if let Some(previous) = steps.last_mut()
                && record_text(spec, previous) == record_text(spec, &step)
            {
                previous.ready_at = previous.ready_at.max(step.ready_at);
                previous.components = step.components;
                stats.repeats_merged += 1;
                continue;
            }
            steps.push(step);
        }
    }
    Thread {
        id: format!("{}-{}", spec.id_prefix, incident.code),
        code: incident.code.clone(),
        service: incident.service.clone(),
        last_posted_at: incident.updates.last().map_or(0, |u| u.at),
        steps,
        kept,
    }
}

fn records_of(spec: &SourceSpec, thread: &Thread) -> Vec<Record> {
    thread
        .steps
        .iter()
        .enumerate()
        .map(|(i, step)| Record {
            id: thread.record_id(i),
            text: record_text(spec, step),
            written_at: step.at,
            valid_until: None,
            thread: Some(thread.id.clone()),
            tier: Tier::Changing,
            title: Some(step.title.clone()),
            source: spec.record_source.to_string(),
            truth: Some(RecordTruth {
                correct: true,
                phase: step.phase,
                key_terms: Vec::new(),
                superseded_at: thread.steps.get(i + 1).map(|next| next.at),
            }),
        })
        .collect()
}

// ---- questions ---------------------------------------------------------------------

struct Asker<'a> {
    spec: &'a SourceSpec,
    rules: &'a Rules,
    threads: &'a Threads,
    seed: u64,
    /// No question is asked after this moment: the source says nothing about later.
    horizon: UnixSeconds,
    queries: Vec<Query>,
    stats: &'a mut Stats,
}

impl Asker<'_> {
    fn push(&mut self, query: Query, workaround: bool, topic: &Topic) {
        bump(&mut self.stats.questions, class_name(query.class));
        if matches!(topic, Topic::Service(_)) {
            bump(
                &mut self.stats.questions_naming_the_service,
                class_name(query.class),
            );
        }
        if workaround {
            self.stats.workaround_questions += 1;
        }
        self.queries.push(query);
    }

    fn not_asked(&mut self, why: &str) {
        bump(&mut self.stats.questions_not_asked, why);
    }

    /// A question about incident `thread` alone, asked while step `step` is
    /// current, at a moment in `[lo, hi]`.
    #[allow(clippy::too_many_arguments)]
    fn about_one(
        &mut self,
        id: String,
        thread: usize,
        step: usize,
        (lo, hi): (UnixSeconds, UnixSeconds),
        intents: &[Intent],
        nobody_else: &str,
        workaround: bool,
    ) {
        let threads = self.threads;
        let this = &threads.all[thread];
        let mut rng = rng_for(self.seed, &id);
        let by_service = service_first(self.spec.service_question_share, &mut rng);
        let found = (0..self.rules.draws).find_map(|_| {
            let asked_at = rng.random_range(lo..=hi);
            threads
                .topic_for_one(thread, step, asked_at, by_service, &mut rng)
                .map(|topic| (asked_at, topic))
        });
        let Some((asked_at, topic)) = found else {
            self.not_asked(nobody_else);
            return;
        };
        let intent = intents[rng.random_range(0..intents.len())];
        let Some(text) = wording::ask(intent, topic.name(), &this.steps[step].body, &mut rng) else {
            self.not_asked(NOT_ASKED_WORDING);
            return;
        };
        self.push(
            Query {
                id,
                text,
                asked_at,
                class: QueryClass::Changing,
                gold: this.gold(step),
                thread: Some(this.id.clone()),
                harmful: Vec::new(),
            },
            workaround,
            &topic,
        );
    }

    /// One question for every gap of at least the rule's length while the incident is open.
    fn changing(&mut self, thread: usize) {
        let threads = self.threads;
        let this = &threads.all[thread];
        for step in 0..this.steps.len().saturating_sub(1) {
            let (now, next) = (&this.steps[step], &this.steps[step + 1]);
            if now.life != Life::Open || next.at - now.at < self.rules.min_gap_s {
                continue;
            }
            let window = (now.ready_at, next.at - 1);
            if window.0 > window.1 {
                self.not_asked(NOT_ASKED_EDITED);
                continue;
            }
            let id = format!("q-{}.c{}", this.id, step + 1);
            let intents = [Intent::Now, Intent::Status, Intent::Fixed];
            self.about_one(id, thread, step, window, &intents, NOT_ASKED_SHARED, false);
        }
    }

    /// One question shortly after the first update that states a workaround.
    fn workaround(&mut self, thread: usize) {
        let threads = self.threads;
        let this = &threads.all[thread];
        let Some(step) = this.steps.iter().position(|s| s.workaround) else {
            return;
        };
        let (now, Some(next)) = (&this.steps[step], this.steps.get(step + 1)) else {
            return;
        };
        if now.life != Life::Open {
            return;
        }
        let window = (
            now.ready_at,
            (next.at - 1).min(now.at + self.rules.workaround_within_s),
        );
        if window.0 > window.1 {
            self.not_asked(NOT_ASKED_WORKAROUND);
            return;
        }
        let id = format!("q-{}.w{}", this.id, step + 1);
        self.about_one(
            id,
            thread,
            step,
            window,
            &[Intent::Workaround],
            NOT_ASKED_WORKAROUND,
            true,
        );
    }

    /// One question 1 to 48 hours after the incident's last update, naming
    /// something that no other incident may concern from the moment it ended
    /// to the moment the question is asked.
    fn post_incident(&mut self, thread: usize) {
        let threads = self.threads;
        let this = &threads.all[thread];
        let Some(over_at) = this.over_at() else {
            return;
        };
        let (from, to) = self.rules.post_incident_after_s;
        let (lo, hi) = (
            this.last_posted_at + from,
            (this.last_posted_at + to).min(self.horizon),
        );
        if lo > hi {
            self.not_asked(NOT_ASKED_HORIZON);
            return;
        }

        // The last moment the topic is still this incident's alone, if it is at `lo`.
        let quiet_until = |topic: &Topic| -> Option<UnixSeconds> {
            let mut until = hi;
            for (other_index, other) in threads.all.iter().enumerate() {
                if other.start() > hi {
                    break;
                }
                if other_index == thread || !other.may_concern(topic) {
                    continue;
                }
                if threads.in_progress(other, over_at).is_some() {
                    return None;
                }
                if let Some(next) = other.steps.iter().find(|s| s.at > over_at) {
                    until = until.min(next.at - 1);
                }
            }
            (until >= lo).then_some(until)
        };

        let id = format!("q-{}.post", this.id);
        let mut rng = rng_for(self.seed, &id);
        let by_service = service_first(self.spec.service_question_share, &mut rng);
        let usable: Vec<(Topic, UnixSeconds)> = this
            .component_topics(0..this.steps.len())
            .into_iter()
            .filter_map(|topic| quiet_until(&topic).map(|until| (topic, until)))
            .collect();
        let service = Topic::Service(this.service.clone());
        let by_service_until = quiet_until(&service);
        let chosen = match (by_service, by_service_until) {
            (true, Some(until)) => Some((service, until)),
            _ if !usable.is_empty() => Some(usable[rng.random_range(0..usable.len())].clone()),
            (_, until) => until.map(|until| (service, until)),
        };
        let Some((topic, until)) = chosen else {
            self.not_asked(NOT_ASKED_POST);
            return;
        };
        // A moment at which the current update's text is known.
        let found = (0..self.rules.draws).find_map(|_| {
            let asked_at = rng.random_range(lo..=until);
            let current = this.current(asked_at)?;
            (asked_at >= this.steps[current].ready_at).then_some((asked_at, current))
        });
        let Some((asked_at, current)) = found else {
            self.not_asked(NOT_ASKED_POST);
            return;
        };
        let intents = [Intent::Now, Intent::Status, Intent::Fixed];
        let intent = intents[rng.random_range(0..intents.len())];
        let Some(text) = wording::ask(intent, topic.name(), &this.steps[current].body, &mut rng) else {
            self.not_asked(NOT_ASKED_WORDING);
            return;
        };
        self.push(
            Query {
                id,
                text,
                asked_at,
                class: QueryClass::PostIncident,
                gold: this.gold(current),
                thread: Some(this.id.clone()),
                harmful: Vec::new(),
            },
            false,
            &topic,
        );
    }

    /// One question for a pair of incidents that are open at the same time and
    /// name the same component. It names only that component, so it fits both.
    /// Two incidents that name no component share their service.
    fn ambiguous(&mut self, a: usize, b: usize) {
        let threads = self.threads;
        let (first, second) = (&threads.all[a], &threads.all[b]);
        if first.service != second.service {
            return;
        }
        // Every stretch of time in which both are open and name a topic in common.
        let mut windows: Vec<(Topic, UnixSeconds, UnixSeconds)> = Vec::new();
        for i in 0..first.steps.len().saturating_sub(1) {
            let step_a = &first.steps[i];
            if step_a.life != Life::Open {
                continue;
            }
            for j in 0..second.steps.len().saturating_sub(1) {
                let step_b = &second.steps[j];
                if step_b.life != Life::Open {
                    continue;
                }
                let lo = step_a.ready_at.max(step_b.ready_at);
                let hi = first.steps[i + 1].at.min(second.steps[j + 1].at) - 1;
                if lo > hi {
                    continue;
                }
                for name in step_a.components.iter().filter(|c| step_b.components.contains(c)) {
                    let topic = Topic::Component {
                        service: first.service.clone(),
                        name: name.clone(),
                    };
                    windows.push((topic, lo, hi));
                }
                if step_a.components.is_empty() && step_b.components.is_empty() {
                    windows.push((Topic::Service(first.service.clone()), lo, hi));
                }
            }
        }
        if windows.is_empty() {
            return;
        }
        let id = format!("q-{}-amb-{}-{}", self.spec.id_prefix, first.code, second.code);
        let mut rng = rng_for(self.seed, &id);
        let total: i64 = windows.iter().map(|(_, lo, hi)| hi - lo + 1).sum();
        let found = (0..self.rules.draws).find_map(|_| {
            // A moment drawn evenly over all the shared time.
            let mut offset = rng.random_range(0..total);
            let (topic, asked_at) = windows.iter().find_map(|(topic, lo, hi)| {
                let length = hi - lo + 1;
                if offset < length {
                    Some((topic, lo + offset))
                } else {
                    offset -= length;
                    None
                }
            })?;
            let fits = threads.fitting(topic, asked_at);
            let both = fits.iter().any(|f| f.0 == a) && fits.iter().any(|f| f.0 == b);
            // Every incident it fits must name the topic: one that names nothing
            // might or might not be about it, and could not be graded.
            let named = fits
                .iter()
                .all(|&(t, s)| threads.all[t].kept && threads.all[t].names(s, topic));
            (both && named && threads.clear(&fits, asked_at)).then_some((topic.clone(), asked_at, fits))
        });
        let Some((topic, asked_at, fits)) = found else {
            self.not_asked(NOT_ASKED_AMBIGUOUS);
            return;
        };
        let intents = [Intent::Now, Intent::Status];
        let intent = intents[rng.random_range(0..intents.len())];
        let answers: Vec<&str> = fits
            .iter()
            .map(|&(t, s)| threads.all[t].steps[s].body.as_str())
            .collect();
        let Some(text) = wording::ask(intent, topic.name(), &answers.join(" "), &mut rng) else {
            self.not_asked(NOT_ASKED_WORDING);
            return;
        };
        // Grade 3 for the current update of every incident it fits, then the earlier ones.
        let mut gold: Vec<Gold> = Vec::new();
        for &(thread, step) in &fits {
            gold.extend(threads.all[thread].gold(step));
        }
        gold.sort_by_key(|g| std::cmp::Reverse(g.grade));
        self.push(
            Query {
                id,
                text,
                asked_at,
                class: QueryClass::Ambiguous,
                gold,
                thread: None,
                harmful: Vec::new(),
            },
            false,
            &topic,
        );
    }
}

// ---- the data set contract ---------------------------------------------------------

/// Check the rules every data set keeps before it is written: ids are valid and
/// used once, no two records have the same text (after collapsing whitespace),
/// every graded record exists, and a question's answer was written by the time
/// it was asked.
pub fn validate(records: &[Record], queries: &[Query]) -> Result<(), InvalidDataSet> {
    let mut written_at: HashMap<&str, UnixSeconds> = HashMap::with_capacity(records.len());
    let mut texts: HashMap<String, &str> = HashMap::with_capacity(records.len());
    for record in records {
        ids::check(&record.id).map_err(|e| InvalidDataSet::Id(e.to_string()))?;
        if written_at.insert(record.id.as_str(), record.written_at).is_some() {
            return Err(InvalidDataSet::SameRecordId(record.id.clone()));
        }
        if let Some(first) = texts.insert(squash(&record.text), record.id.as_str()) {
            return Err(InvalidDataSet::SameText {
                first: first.to_string(),
                second: record.id.clone(),
            });
        }
    }
    let mut query_ids = BTreeSet::new();
    for query in queries {
        if !query_ids.insert(query.id.as_str()) {
            return Err(InvalidDataSet::SameQueryId(query.id.clone()));
        }
        for gold in &query.gold {
            let Some(at) = written_at.get(gold.record.as_str()) else {
                return Err(InvalidDataSet::UnknownGold {
                    query: query.id.clone(),
                    record: gold.record.clone(),
                });
            };
            if gold.grade == 3 && *at > query.asked_at {
                return Err(InvalidDataSet::AnswerFromTheFuture {
                    query: query.id.clone(),
                    record: gold.record.clone(),
                });
            }
        }
        if !query.gold.iter().any(|g| g.grade == 3) {
            return Err(InvalidDataSet::NoAnswer {
                query: query.id.clone(),
            });
        }
    }
    Ok(())
}

/// Records and questions for the incidents of one source. Every incident is
/// read; the complete ones are kept. `horizon` is where the source ends (when it
/// was fetched or collected): no question is asked after it.
pub fn build(
    spec: &SourceSpec,
    rules: &Rules,
    mut incidents: Vec<Incident>,
    horizon: UnixSeconds,
    seed: u64,
) -> Result<Built, TimelineError> {
    let mut stats = Stats {
        incidents_read: incidents.len() as u64,
        ..Stats::default()
    };
    incidents.sort_by(|a, b| {
        let start = |i: &Incident| i.updates.first().map_or(i64::MAX, |u| u.at);
        (start(a), &a.code).cmp(&(start(b), &b.code))
    });

    // Keep the complete ones.
    let mut codes = BTreeSet::new();
    let mut is_kept: Vec<bool> = Vec::with_capacity(incidents.len());
    for incident in &incidents {
        if !codes.insert(incident.code.as_str()) {
            return Err(TimelineError::DuplicateCode {
                code: incident.code.clone(),
            });
        }
        if incident.updates.windows(2).any(|w| w[1].at < w[0].at) {
            return Err(TimelineError::OutOfOrder {
                code: incident.code.clone(),
            });
        }
        if incident.updates.windows(2).any(|w| w[1].life < w[0].life) {
            return Err(TimelineError::ReopensAfterOver {
                code: incident.code.clone(),
            });
        }
        let kept = if incident.updates.len() < rules.min_updates {
            bump(&mut stats.incidents_left_out, LEFT_OUT_FEW_UPDATES);
            false
        } else if incident.updates.last().is_none_or(|u| u.life != Life::Over) {
            bump(&mut stats.incidents_left_out, LEFT_OUT_NOT_CLOSED);
            false
        } else {
            true
        };
        is_kept.push(kept);
    }
    let kept: Vec<&Incident> = incidents
        .iter()
        .zip(&is_kept)
        .filter(|(_, k)| **k)
        .map(|(i, _)| i)
        .collect();

    // What the source has for them.
    stats.incidents_kept = kept.len() as u64;
    stats.updates = kept.iter().map(|i| i.updates.len() as u64).sum();
    stats.first_update_at = kept.iter().filter_map(|i| i.updates.first()).map(|u| u.at).min();
    stats.last_update_at = kept.iter().filter_map(|i| i.updates.last()).map(|u| u.at).max();
    let mut counts: Vec<i64> = kept.iter().map(|i| i.updates.len() as i64).collect();
    stats.updates_per_incident_max = counts.iter().copied().max().unwrap_or(0) as u64;
    stats.updates_per_incident_median = median(&mut counts);
    let mut gaps: Vec<i64> = kept
        .iter()
        .flat_map(|i| i.updates.windows(2).map(|w| w[1].at - w[0].at))
        .collect();
    stats.gap_median_s = median(&mut gaps);
    let mut without_time: HashMap<String, u64> = HashMap::new();
    for update in kept.iter().flat_map(|i| i.updates.iter()) {
        *without_time.entry(text_without_time(update)).or_insert(0) += 1;
    }
    for n in without_time.values().filter(|n| **n > 1) {
        stats.identical_without_time += n - 1;
        stats.identical_without_time_texts += 1;
    }

    // Threads, and the records of the kept ones.
    let mut scratch = Stats::default();
    let mut all: Vec<Thread> = Vec::with_capacity(incidents.len());
    for (incident, kept) in incidents.iter().zip(&is_kept) {
        let tally = if *kept { &mut stats } else { &mut scratch };
        all.push(thread_of(spec, rules, incident, *kept, tally));
    }
    all.sort_by(|a, b| (a.start(), &a.code).cmp(&(b.start(), &b.code)));
    let threads = Threads {
        all,
        unclosed_open_for_s: rules.unclosed_open_for_s,
    };
    let records: Vec<Record> = threads
        .all
        .iter()
        .filter(|t| t.kept)
        .flat_map(|t| records_of(spec, t))
        .collect();
    stats.records = records.len() as u64;

    // Questions, about kept incidents only.
    let mut asker = Asker {
        spec,
        rules,
        threads: &threads,
        seed,
        horizon,
        queries: Vec::new(),
        stats: &mut stats,
    };
    for index in (0..threads.all.len()).filter(|&i| threads.all[i].kept) {
        asker.changing(index);
        asker.workaround(index);
        asker.post_incident(index);
    }
    for a in (0..threads.all.len()).filter(|&i| threads.all[i].kept) {
        // Incidents are sorted by start: a later one overlaps `a` only if it
        // starts before `a` is over.
        let open_until = threads.all[a].over_at().unwrap_or(i64::MAX);
        for b in a + 1..threads.all.len() {
            if threads.all[b].start() >= open_until {
                break;
            }
            if threads.all[b].kept {
                asker.ambiguous(a, b);
            }
        }
    }
    let mut queries = asker.queries;
    queries.retain(|q| q.asked_at <= horizon);
    queries.sort_by(|a, b| (a.asked_at, &a.id).cmp(&(b.asked_at, &b.id)));

    validate(&records, &queries)?;
    Ok(Built {
        records,
        queries,
        stats,
    })
}

/// The latest update of any incident: where a collected source ends.
pub fn latest_update(incidents: &[Incident]) -> UnixSeconds {
    incidents
        .iter()
        .flat_map(|i| i.updates.iter().map(|u| u.at))
        .max()
        .unwrap_or(i64::MIN)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2024-01-01T00:00:00Z
    const T0: UnixSeconds = 1_704_067_200;
    const HOUR: i64 = 3_600;
    const FAR: UnixSeconds = T0 + 3_650 * 24 * HOUR;
    const SPEC: SourceSpec = SourceSpec {
        name: "test",
        record_source: "test-status",
        id_prefix: "t",
        time_precision_s: 1,
        service_question_share: 0.0,
    };
    const ACTIONS: &str = "GitHub Actions";
    const PAGES: &str = "GitHub Pages";

    fn update(at: UnixSeconds, label: &str, body: &str, components: &[&str]) -> Update {
        let (phase, life) = match label {
            "Investigating" => (Phase::Investigating, Life::Open),
            "Identified" => (Phase::RootCause, Life::Open),
            "Monitoring" => (Phase::FixInProgress, Life::Open),
            _ => (Phase::Resolved, Life::Over),
        };
        Update {
            at,
            edited_at: None,
            phase,
            life,
            title: "Incident with Actions".into(),
            label: Some(label.into()),
            body: body.into(),
            components: components.iter().map(|c| c.to_string()).collect(),
            workaround: false,
        }
    }

    fn incident(code: &str, updates: Vec<Update>) -> Incident {
        Incident {
            code: code.into(),
            service: "GitHub".into(),
            updates,
        }
    }

    /// Four updates: gaps of 300 s, 119 s, 181 s; resolved 100 s later.
    fn four_updates(code: &str, start: UnixSeconds, components: &[&str]) -> Incident {
        incident(
            code,
            vec![
                update(
                    start,
                    "Investigating",
                    "We are investigating delayed workflow runs.",
                    components,
                ),
                update(
                    start + 300,
                    "Identified",
                    "A configuration change caused the delays.",
                    components,
                ),
                update(
                    start + 419,
                    "Investigating",
                    "Runs are still queued in one region.",
                    components,
                ),
                update(
                    start + 600,
                    "Monitoring",
                    "The change was rolled back; queues are draining.",
                    components,
                ),
                update(
                    start + 700,
                    "Resolved",
                    "This incident has been resolved.",
                    components,
                ),
            ],
        )
    }

    fn by_id<'a>(built: &'a Built, id: &str) -> Option<&'a Query> {
        built.queries.iter().find(|q| q.id == id)
    }

    fn grade3(query: &Query) -> Vec<&str> {
        query
            .gold
            .iter()
            .filter(|g| g.grade == 3)
            .map(|g| g.record.as_str())
            .collect()
    }

    #[test]
    fn records_follow_the_updates_oldest_first_and_each_is_superseded_by_the_next() {
        let built = build(
            &SPEC,
            &Rules::default(),
            vec![four_updates("a1", T0, &[ACTIONS])],
            FAR,
            42,
        )
        .unwrap();
        let ids: Vec<&str> = built.records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["t-a1.1", "t-a1.2", "t-a1.3", "t-a1.4", "t-a1.5"]);
        let written: Vec<i64> = built.records.iter().map(|r| r.written_at - T0).collect();
        assert_eq!(written, [0, 300, 419, 600, 700]);
        let superseded: Vec<Option<i64>> = built
            .records
            .iter()
            .map(|r| r.truth.as_ref().and_then(|t| t.superseded_at).map(|s| s - T0))
            .collect();
        assert_eq!(superseded, [Some(300), Some(419), Some(600), Some(700), None]);
        let phases: Vec<Phase> = built
            .records
            .iter()
            .filter_map(|r| r.truth.as_ref())
            .map(|t| t.phase)
            .collect();
        assert_eq!(
            phases,
            [
                Phase::Investigating,
                Phase::RootCause,
                Phase::Investigating,
                Phase::FixInProgress,
                Phase::Resolved
            ]
        );
        let first = &built.records[0];
        assert_eq!(
            first.text,
            "Incident with Actions. Update 2024-01-01 00:00:00 UTC, Investigating: We are investigating delayed \
             workflow runs."
        );
        assert_eq!(first.thread.as_deref(), Some("t-a1"));
        assert_eq!(first.tier, Tier::Changing);
        assert_eq!(first.valid_until, None);
    }

    #[test]
    fn a_changing_question_falls_in_every_gap_of_two_minutes_and_its_answer_is_the_update_current_then() {
        let built = build(
            &SPEC,
            &Rules::default(),
            vec![four_updates("a1", T0, &[ACTIONS])],
            FAR,
            42,
        )
        .unwrap();
        let changing: Vec<&str> = built
            .queries
            .iter()
            .filter(|q| q.class == QueryClass::Changing)
            .map(|q| q.id.as_str())
            .collect();
        // 300 s and 181 s qualify; 119 s does not; the last gap leads to the resolution.
        assert_eq!(changing, ["q-t-a1.c1", "q-t-a1.c3"]);
        for (id, step, from, to) in [("q-t-a1.c1", 1, 1, 299), ("q-t-a1.c3", 3, 420, 599)] {
            let q = by_id(&built, id).unwrap();
            assert!(
                (T0 + from..=T0 + to).contains(&q.asked_at),
                "{id} asked at +{}",
                q.asked_at - T0
            );
            assert_eq!(grade3(q), [format!("t-a1.{step}").as_str()]);
            let earlier: Vec<String> = (1..step).map(|n| format!("t-a1.{n}")).collect();
            let ones: Vec<&str> = q
                .gold
                .iter()
                .filter(|g| g.grade == 1)
                .map(|g| g.record.as_str())
                .collect();
            assert_eq!(ones, earlier.iter().map(String::as_str).collect::<Vec<_>>());
            assert_eq!(q.thread.as_deref(), Some("t-a1"));
            assert!(q.text.contains(ACTIONS), "{}", q.text);
        }
    }

    #[test]
    fn a_post_incident_question_comes_one_to_48_hours_later_unless_the_source_ends_first() {
        let built = build(
            &SPEC,
            &Rules::default(),
            vec![four_updates("a1", T0, &[ACTIONS])],
            FAR,
            42,
        )
        .unwrap();
        let q = by_id(&built, "q-t-a1.post").unwrap();
        assert_eq!(q.class, QueryClass::PostIncident);
        assert!((T0 + 700 + HOUR..=T0 + 700 + 48 * HOUR).contains(&q.asked_at));
        assert_eq!(grade3(q), ["t-a1.5"]);
        assert_eq!(q.gold.len(), 5);

        let built = build(
            &SPEC,
            &Rules::default(),
            vec![four_updates("a1", T0, &[ACTIONS])],
            T0 + 1_800,
            42,
        )
        .unwrap();
        assert!(by_id(&built, "q-t-a1.post").is_none());
        assert_eq!(built.stats.questions_not_asked.get(NOT_ASKED_HORIZON), Some(&1));
    }

    #[test]
    fn two_open_incidents_naming_one_component_give_an_ambiguous_question() {
        let a = incident(
            "a",
            vec![
                update(T0, "Investigating", "Workflow runs are delayed.", &[ACTIONS]),
                update(
                    T0 + 2 * HOUR,
                    "Investigating",
                    "Delays continue for larger runners.",
                    &[ACTIONS],
                ),
                update(
                    T0 + 3 * HOUR,
                    "Resolved",
                    "This incident has been resolved.",
                    &[ACTIONS],
                ),
            ],
        );
        let b = incident(
            "b",
            vec![
                update(
                    T0 + HOUR,
                    "Investigating",
                    "Pages builds fail to start.",
                    &[ACTIONS, PAGES],
                ),
                update(
                    T0 + 2 * HOUR + 600,
                    "Monitoring",
                    "Builds are recovering.",
                    &[ACTIONS, PAGES],
                ),
                update(
                    T0 + 4 * HOUR,
                    "Resolved",
                    "This incident has been resolved.",
                    &[ACTIONS, PAGES],
                ),
            ],
        );
        let built = build(&SPEC, &Rules::default(), vec![a, b], FAR, 42).unwrap();
        let q = by_id(&built, "q-t-amb-a-b").unwrap();
        assert_eq!(q.class, QueryClass::Ambiguous);
        assert_eq!(q.thread, None);
        assert!(q.text.contains(ACTIONS), "{}", q.text);
        assert!((T0 + HOUR + 1..T0 + 3 * HOUR).contains(&q.asked_at));
        let a_current = if q.asked_at < T0 + 2 * HOUR {
            "t-a.1"
        } else {
            "t-a.2"
        };
        let b_current = if q.asked_at < T0 + 2 * HOUR + 600 {
            "t-b.1"
        } else {
            "t-b.2"
        };
        let mut answers = grade3(q);
        answers.sort_unstable();
        assert_eq!(answers, [a_current, b_current]);
        // While both are open, a question about one alone may not name Actions.
        for q in built.queries.iter().filter(|q| q.class == QueryClass::Changing) {
            let both_open = (T0 + HOUR..T0 + 3 * HOUR).contains(&q.asked_at);
            assert!(
                !(both_open && q.text.contains(ACTIONS)),
                "{} at +{}",
                q.text,
                q.asked_at - T0
            );
        }
    }

    #[test]
    fn an_incident_that_is_not_kept_still_blocks_questions_while_it_is_in_progress() {
        let kept = four_updates("k", T0, &[ACTIONS]);
        let short = incident(
            "s",
            vec![
                update(
                    T0 - 50,
                    "Investigating",
                    "Actions jobs fail to start.",
                    &[ACTIONS],
                ),
                update(
                    T0 + 1_000,
                    "Resolved",
                    "This incident has been resolved.",
                    &[ACTIONS],
                ),
            ],
        );
        let built = build(&SPEC, &Rules::default(), vec![kept, short], FAR, 42).unwrap();
        assert_eq!(built.stats.incidents_kept, 1);
        assert!(built.records.iter().all(|r| r.thread.as_deref() == Some("t-k")));
        assert!(built.queries.iter().all(|q| q.class != QueryClass::Changing));
        assert_eq!(built.stats.questions_not_asked.get(NOT_ASKED_SHARED), Some(&2));
    }

    #[test]
    fn an_incident_that_names_nothing_may_be_about_any_component_of_its_service() {
        let named = four_updates("n", T0, &[ACTIONS]);
        let mut unscoped = four_updates("u", T0 - 100, &[]);
        unscoped.updates[4].at = T0 + 1_000;
        let mut elsewhere = four_updates("e", T0, &["the OpenAI API"]);
        elsewhere.service = "OpenAI".into();
        for update in &mut elsewhere.updates {
            update.title = "Elevated API errors".into();
        }
        let built = build(
            &SPEC,
            &Rules::default(),
            vec![named, unscoped, elsewhere],
            FAR,
            42,
        )
        .unwrap();
        let changing: Vec<&Query> = built
            .queries
            .iter()
            .filter(|q| q.class == QueryClass::Changing)
            .collect();
        // While both GitHub incidents are in progress (T0 to T0+700) nothing is asked
        // about either: the one that names nothing might be about Actions too.
        for q in &changing {
            if q.thread.as_deref() != Some("t-e") {
                assert!(
                    q.asked_at < T0 || q.asked_at > T0 + 700,
                    "{} at +{}",
                    q.id,
                    q.asked_at - T0
                );
            }
        }
        assert!(changing.iter().all(|q| q.thread.as_deref() != Some("t-n")));
        // OpenAI's incident is not touched.
        let openai: Vec<&str> = changing
            .iter()
            .filter(|q| q.thread.as_deref() == Some("t-e"))
            .map(|q| q.id.as_str())
            .collect();
        assert_eq!(openai, ["q-t-e.c1", "q-t-e.c3"]);
    }

    #[test]
    fn an_incident_the_source_never_closes_blocks_for_48_hours_after_its_last_update() {
        let open = incident(
            "o",
            vec![update(T0, "Investigating", "Actions runs are slow.", &[ACTIONS])],
        );
        let later = four_updates("l", T0 + 47 * HOUR, &[ACTIONS]);
        let built = build(&SPEC, &Rules::default(), vec![open.clone(), later], FAR, 42).unwrap();
        assert!(built.queries.iter().all(|q| q.class != QueryClass::Changing));
        let after = four_updates("l", T0 + 49 * HOUR, &[ACTIONS]);
        let built = build(&SPEC, &Rules::default(), vec![open, after], FAR, 42).unwrap();
        assert_eq!(built.stats.questions.get("changing"), Some(&2));
    }

    #[test]
    fn repeated_updates_get_texts_of_their_own_and_a_repeat_in_the_same_second_is_one_record() {
        let mut updates = vec![
            update(
                T0,
                "Investigating",
                "We are continuing to investigate this issue.",
                &[ACTIONS],
            ),
            update(
                T0 + 600,
                "Investigating",
                "We are continuing to investigate this issue.",
                &[ACTIONS],
            ),
            update(
                T0 + 600,
                "Investigating",
                "We are continuing to investigate this issue.",
                &[ACTIONS],
            ),
            update(
                T0 + 1_200,
                "Resolved",
                "This incident has been resolved.",
                &[ACTIONS],
            ),
        ];
        updates[2].components.push(PAGES.into());
        let built = build(&SPEC, &Rules::default(), vec![incident("r", updates)], FAR, 42).unwrap();
        assert_eq!(built.records.len(), 3);
        assert_eq!(built.stats.repeats_merged, 1);
        assert_eq!(built.stats.identical_without_time, 2);
        assert_eq!(built.stats.identical_without_time_texts, 1);
        assert_ne!(built.records[0].text, built.records[1].text);
        assert!(built.records[1].text.contains("Update 2024-01-01 00:10:00 UTC"));
        assert!(validate(&built.records, &built.queries).is_ok());
    }

    #[test]
    fn a_closing_update_rewritten_later_is_bare_until_its_new_text_appears() {
        let mut incident = four_updates("w", T0, &[ACTIONS]);
        let summary = "Between 00:00 and 00:11 UTC, workflow runs were delayed by a configuration change.";
        incident.updates[4].body = summary.into();
        incident.updates[4].edited_at = Some(T0 + 700 + 20 * HOUR);
        let built = build(&SPEC, &Rules::default(), vec![incident], FAR, 42).unwrap();
        assert_eq!(built.stats.closing_rewrites, 1);
        let bare = &built.records[4];
        let full = &built.records[5];
        assert_eq!(
            bare.text,
            "Incident with Actions. Update 2024-01-01 00:11:40 UTC, Resolved."
        );
        assert_eq!(bare.written_at, T0 + 700);
        assert_eq!(
            bare.truth.as_ref().and_then(|t| t.superseded_at),
            Some(T0 + 700 + 20 * HOUR)
        );
        assert_eq!(full.written_at, T0 + 700 + 20 * HOUR);
        assert!(full.text.ends_with(summary));
        let post = by_id(&built, "q-t-w.post").unwrap();
        let expected = if post.asked_at < full.written_at {
            "t-w.5"
        } else {
            "t-w.6"
        };
        assert_eq!(grade3(post), [expected]);
    }

    #[test]
    fn the_contract_check_refuses_shared_texts_and_unanswerable_questions() {
        let record = |id: &str, text: &str, at: i64| Record {
            id: id.into(),
            text: text.into(),
            written_at: at,
            valid_until: None,
            thread: None,
            tier: Tier::Changing,
            title: None,
            source: "test".into(),
            truth: None,
        };
        let query = |gold: &[(&str, u8)], asked_at: i64| Query {
            id: "q-1".into(),
            text: "Is it fixed?".into(),
            asked_at,
            class: QueryClass::Changing,
            gold: gold
                .iter()
                .map(|(r, g)| Gold {
                    record: (*r).into(),
                    grade: *g,
                })
                .collect(),
            thread: None,
            harmful: vec![],
        };
        let two = [
            record("r-1", "Fixed  now.", 10),
            record("r-2", "Fixed now.\n", 20),
        ];
        assert_eq!(
            validate(&two, &[]),
            Err(InvalidDataSet::SameText {
                first: "r-1".into(),
                second: "r-2".into()
            })
        );
        let ok = [record("r-1", "Investigating.", 10), record("r-2", "Fixed.", 20)];
        assert_eq!(validate(&ok, &[query(&[("r-2", 3), ("r-1", 1)], 25)]), Ok(()));
        assert!(matches!(
            validate(&ok, &[query(&[("r-3", 3)], 25)]),
            Err(InvalidDataSet::UnknownGold { .. })
        ));
        assert!(matches!(
            validate(&ok, &[query(&[("r-2", 3)], 15)]),
            Err(InvalidDataSet::AnswerFromTheFuture { .. })
        ));
        assert!(matches!(
            validate(&ok, &[query(&[("r-1", 1)], 25)]),
            Err(InvalidDataSet::NoAnswer { .. })
        ));
        let same_id = [record("r-1", "a", 1), record("r-1", "b", 2)];
        assert_eq!(
            validate(&same_id, &[]),
            Err(InvalidDataSet::SameRecordId("r-1".into()))
        );
        assert!(matches!(
            validate(&[record("R 1", "a", 1)], &[]),
            Err(InvalidDataSet::Id(_))
        ));
    }

    #[test]
    fn the_seed_fixes_every_moment_and_wording() {
        let incidents = || {
            vec![
                four_updates("a", T0, &[ACTIONS]),
                four_updates("b", T0 + 10 * HOUR, &[PAGES]),
                four_updates("c", T0 + 100 * HOUR, &[ACTIONS, PAGES]),
            ]
        };
        let once = build(&SPEC, &Rules::default(), incidents(), FAR, 42).unwrap();
        let again = build(&SPEC, &Rules::default(), incidents(), FAR, 42).unwrap();
        assert_eq!(once.queries, again.queries);
        assert_eq!(once.records, again.records);
        let other = build(&SPEC, &Rules::default(), incidents(), FAR, 43).unwrap();
        assert_eq!(other.queries.len(), once.queries.len());
        assert_ne!(other.queries, once.queries);
        assert_eq!(other.records, once.records);
    }

    #[test]
    fn a_share_of_the_questions_names_the_service_and_the_rest_name_the_component() {
        let incidents = || -> Vec<Incident> {
            (0..80)
                .map(|i| four_updates(&format!("i{i}"), T0 + i * 10 * HOUR, &[ACTIONS]))
                .collect()
        };
        let with_share = SourceSpec {
            service_question_share: 0.25,
            ..SPEC
        };
        let built = build(&with_share, &Rules::default(), incidents(), FAR, 42).unwrap();
        let changing: Vec<&Query> = built
            .queries
            .iter()
            .filter(|q| q.class == QueryClass::Changing)
            .collect();
        assert_eq!(changing.len(), 160);
        let by_service = changing.iter().filter(|q| !q.text.contains(ACTIONS)).count();
        assert!((25..=55).contains(&by_service), "{by_service} of 160");
        assert_eq!(
            built.stats.questions_naming_the_service.get("changing"),
            Some(&(by_service as u64))
        );
        for q in changing.iter().filter(|q| !q.text.contains(ACTIONS)) {
            assert!(q.text.contains("GitHub"), "{}", q.text);
        }
        let post_by_service = built
            .stats
            .questions_naming_the_service
            .get("post_incident")
            .copied()
            .unwrap_or(0);
        assert!((5..=35).contains(&post_by_service), "{post_by_service} of 80");

        // With no share, every question names the component.
        let built = build(&SPEC, &Rules::default(), incidents(), FAR, 42).unwrap();
        assert!(built.queries.iter().all(|q| q.text.contains(ACTIONS)));
        assert!(built.stats.questions_naming_the_service.is_empty());

        // An incident that names nothing is always asked about as the service.
        let built = build(
            &with_share,
            &Rules::default(),
            vec![four_updates("u", T0, &[])],
            FAR,
            42,
        )
        .unwrap();
        assert!(!built.queries.is_empty());
        assert!(built.queries.iter().all(|q| q.text.contains("GitHub")));
        assert_eq!(built.stats.questions_naming_the_service.get("changing"), Some(&2));
    }

    #[test]
    fn broken_timelines_are_refused() {
        let mut backwards = four_updates("x", T0, &[ACTIONS]);
        backwards.updates.swap(1, 2);
        assert!(matches!(
            build(&SPEC, &Rules::default(), vec![backwards], FAR, 42),
            Err(TimelineError::OutOfOrder { .. })
        ));
        let mut reopened = four_updates("y", T0, &[ACTIONS]);
        reopened.updates[1].life = Life::Over;
        assert!(matches!(
            build(&SPEC, &Rules::default(), vec![reopened], FAR, 42),
            Err(TimelineError::ReopensAfterOver { .. })
        ));
        let twice = vec![
            four_updates("z", T0, &[ACTIONS]),
            four_updates("z", T0 + HOUR, &[ACTIONS]),
        ];
        assert!(matches!(
            build(&SPEC, &Rules::default(), twice, FAR, 42),
            Err(TimelineError::DuplicateCode { .. })
        ));
    }
}
