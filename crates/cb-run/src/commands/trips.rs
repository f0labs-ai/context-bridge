//! `cb-run trips`: how long a new record takes to become searchable.
//!
//! One record at a time. The clock is noted, the upload starts, and from that same moment
//! the live search is asked the record's question at a fixed pace until an answer holds
//! the record. Every mark of a trip is read from one monotonic clock:
//!
//! - `written_ms`: the upload call returned;
//! - `first_hit_ms`: the first answer that holds the record ARRIVED (the moment a search
//!   was sent plus that search's own duration; when it was sent decides nothing).
//!
//! Searches overlap when an answer takes longer than the pace; how many may be under way
//! at once is bounded, and a search that is not sent because of the bound is counted.

use super::{
    Shared, about_knowledge_service, ask_what_the_service_is, belongs_to_records, read_queries, read_records,
};
use crate::error::{FailureKind, RunError};
use crate::http::millis;
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::knowledge::{
    Document, KnowledgeService, Label, SearchAnswer, SearchRequest, Tenant, Timed, Uploaded, features,
};
use crate::retry::with_retries;
use crate::run::{Report, Session};
use cb_types::{PipelineTrip, Query, Record};
use futures::FutureExt;
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::time::MissedTickBehavior;

pub const TRIPS_FILE: &str = "trips.jsonl";
const COMMAND: &str = "trips";
/// Passages a call asks the search for (docs/RUN_TOOLS.md).
pub const PASSAGES: u32 = 4;
/// The relevance cut-off a call sends (docs/RUN_TOOLS.md).
pub const CUT_OFF: f64 = 0.12;
/// The id the failure "fewer records with a question than asked for" is filed under.
const ASKED_COUNT: &str = "n";

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct TripsArgs {
    /// The data set directory; its records.jsonl and queries.jsonl are read.
    #[arg(long)]
    pub data: PathBuf,
    /// The tenant of the knowledge service the records are uploaded under.
    #[arg(long)]
    pub tenant: String,
    /// The collection label the documents are filed under and searched in.
    #[arg(long)]
    pub collection: String,
    /// The run directory; trips.jsonl is appended to.
    #[arg(long)]
    pub run: PathBuf,
    /// How many records to time: the first N of the data set that have a question.
    /// Without it, all of them.
    #[arg(long)]
    pub n: Option<usize>,
    /// The pace of the searches, in milliseconds.
    #[arg(long, default_value_t = 25)]
    pub poll_ms: u64,
    /// How long a record is waited for before its trip counts as a failure.
    #[arg(long, default_value_t = 30_000)]
    pub give_up_ms: u64,
    /// Searches that may be under way at once while a record is awaited.
    #[arg(long, default_value_t = 8)]
    pub max_polls_in_flight: usize,
}

/// One line of `trips.jsonl`: a `cb_types::PipelineTrip` (`record`, `written_ms`,
/// `first_hit_ms`) with what the service and the poller add. A reader that knows only
/// `PipelineTrip` reads the line as one.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TripLine {
    #[serde(flatten)]
    pub trip: PipelineTrip,
    /// Where the upload's time went, as the service measured it.
    pub timings_ms: BTreeMap<String, f64>,
    /// Passages the service indexed for the record.
    pub passages_indexed: u64,
    /// Searches sent while the record was awaited.
    pub polls: u32,
    /// Searches that got no usable answer; they decide nothing.
    pub polls_failed: u32,
    /// Searches not sent because too many were under way.
    pub polls_held_back: u32,
    /// The service's `mode` for the answer that first held the record (`reranked`
    /// unless the service was degraded).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_hit_mode: Option<String>,
}

/// How a record is awaited.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pace {
    pub poll_every: Duration,
    pub give_up_after: Duration,
    pub max_in_flight: usize,
}

/// What one trip measured.
#[derive(Debug)]
pub struct Measured {
    pub written: Duration,
    pub first_hit: Duration,
    pub first_hit_mode: Option<String>,
    pub uploaded: Uploaded,
    pub polls: u32,
    pub polls_failed: u32,
    pub polls_held_back: u32,
}

/// Why a trip could not be measured.
#[derive(Debug)]
pub struct NotMeasured {
    pub kind: FailureKind,
    pub error: RunError,
}

/// One search, sent now. Answers the moment its answer ARRIVED, read from the trip's
/// clock as soon as the whole answer is in, and what came back.
async fn one_poll<K: KnowledgeService>(
    kb: &K,
    tenant: &Tenant,
    search: &SearchRequest,
    clock: Instant,
) -> (Duration, Result<Timed<SearchAnswer>, RunError>) {
    let result = kb.search(tenant, search).await;
    (clock.elapsed(), result)
}

/// When a search's answer arrived, and its mode, if it holds the record.
fn hit_in(
    record_id: &str,
    arrived: Duration,
    result: &Result<Timed<SearchAnswer>, RunError>,
) -> Option<(Duration, Option<String>)> {
    match result {
        Ok(answer) if answer.value.contains_record(record_id) => Some((arrived, answer.value.mode.clone())),
        _ => None,
    }
}

/// Times one record: the upload and the searches start at the same instant and are read
/// from one clock. The upload is tried once (a retry would be a different trip), and so
/// is each search.
pub async fn time_one_trip<K: KnowledgeService>(
    kb: &K,
    tenant: &Tenant,
    document: &Document<'_>,
    search: &SearchRequest,
    pace: Pace,
) -> Result<Measured, NotMeasured> {
    let record_id = document.record_id;
    let clock = Instant::now();
    // The upload's return is read from the same clock the moment the whole answer is in.
    let upload = async {
        let result = kb.upload(tenant, document).await;
        (clock.elapsed(), result)
    };
    tokio::pin!(upload);
    let mut ticks = tokio::time::interval(pace.poll_every);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut in_flight = FuturesUnordered::new();
    let mut written: Option<(Duration, Uploaded)> = None;
    let mut first_hit: Option<(Duration, Option<String>)> = None;
    let (mut polls, mut polls_failed, mut polls_held_back) = (0u32, 0u32, 0u32);

    loop {
        if let (Some((written, uploaded)), Some((first_hit, first_hit_mode))) = (&written, &first_hit) {
            return Ok(Measured {
                written: *written,
                first_hit: *first_hit,
                first_hit_mode: first_hit_mode.clone(),
                uploaded: uploaded.clone(),
                polls,
                polls_failed,
                polls_held_back,
            });
        }
        let left = pace.give_up_after.saturating_sub(clock.elapsed());
        if left.is_zero() {
            let reason = match (&written, &first_hit) {
                (None, _) => format!(
                    "the upload did not return within {} ms",
                    pace.give_up_after.as_millis()
                ),
                (Some((written, _)), _) => format!(
                    "not found by the search within {} ms ({polls} searches, {polls_failed} without an answer; \
                     the upload returned after {:.1} ms)",
                    pace.give_up_after.as_millis(),
                    millis(*written)
                ),
            };
            return Err(NotMeasured {
                kind: FailureKind::NeverFound,
                error: RunError::item(record_id, reason),
            });
        }
        tokio::select! {
            biased;
            Some((arrived, result)) = in_flight.next(), if !in_flight.is_empty() => {
                let result: Result<Timed<SearchAnswer>, RunError> = result;
                if result.is_err() {
                    polls_failed += 1;
                }
                if let Some(hit) = hit_in(record_id, arrived, &result) {
                    let mut earliest = hit;
                    // Answers that have arrived as well and wait to be looked at: the
                    // earliest arrival counts, not the order they are looked at in.
                    while let Some(Some((arrived, result))) = in_flight.next().now_or_never() {
                        if result.is_err() {
                            polls_failed += 1;
                        }
                        if let Some(other) = hit_in(record_id, arrived, &result)
                            && other.0 < earliest.0
                        {
                            earliest = other;
                        }
                    }
                    first_hit = Some(earliest);
                    // Found: the searches still under way are dropped, and no more are sent.
                    in_flight.clear();
                }
            }
            (returned, result) = &mut upload, if written.is_none() => {
                match result {
                    Ok(uploaded) => written = Some((returned, uploaded.value)),
                    Err(error) => {
                        return Err(NotMeasured { kind: error.kind(), error });
                    }
                }
            }
            _ = ticks.tick(), if first_hit.is_none() => {
                if in_flight.len() < pace.max_in_flight {
                    polls += 1;
                    in_flight.push(one_poll(kb, tenant, search, clock));
                } else {
                    polls_held_back += 1;
                }
            }
            // Wakes the loop when the waiting time is over, whatever else is under way.
            () = tokio::time::sleep(left) => {}
        }
    }
}

/// The question a record is searched with: the first question of the data set that
/// names the record as its answer (grade 3).
pub fn questions_by_record(queries: &[Query]) -> HashMap<&str, &Query> {
    let mut by_record: HashMap<&str, &Query> = HashMap::new();
    for query in queries {
        for gold in query.gold.iter().filter(|g| g.grade == 3) {
            by_record.entry(gold.record.as_str()).or_insert(query);
        }
    }
    by_record
}

/// Why the live search cannot return a record at this moment, if it cannot: the search
/// is asked as of the service's clock, and leaves out what is not written yet and what
/// has ended.
fn not_visible_now(record: &Record, now: i64) -> Option<String> {
    if record.written_at > now {
        return Some(format!(
            "written_at {} is in the future: the live search would not return the record yet",
            record.written_at
        ));
    }
    match record.valid_until {
        Some(ends) if ends <= now => Some(format!(
            "valid_until {ends} has passed: the live search would never return the record"
        )),
        _ => None,
    }
}

/// Times the trips `trips.jsonl` does not hold yet.
pub async fn trips<K: KnowledgeService>(
    kb: &K,
    args: &TripsArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let tenant = Tenant::parse(&args.tenant)?;
    let label = Label::parse("--collection", &args.collection)?;
    if args.poll_ms == 0 || args.max_polls_in_flight == 0 {
        return Err(RunError::Setting {
            name: "--poll-ms / --max-polls-in-flight",
            reason: "each is at least 1".into(),
        });
    }
    let records = read_records(&args.data)?;
    let queries = read_queries(&args.data)?;
    let mut session = Session::begin(&args.run, COMMAND, args, about_knowledge_service(kb, shared))?;
    match work(
        kb,
        args,
        shared,
        (&tenant, &label),
        (&records, &queries),
        &mut session,
    )
    .await
    {
        Ok(done) => session.finish(|id| id == ASKED_COUNT || done.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn work<K: KnowledgeService>(
    kb: &K,
    args: &TripsArgs,
    shared: &Shared,
    (tenant, label): (&Tenant, &Label),
    (records, queries): (&[Record], &[Query]),
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    session.claim(
        TRIPS_FILE,
        belongs_to_records(kb, &args.data, tenant.as_str(), label.as_str())?,
    )?;
    ask_what_the_service_is(
        kb,
        shared,
        session,
        &[features::DOCUMENT_TIMES, features::INGEST_TIMINGS],
    )
    .await?;
    let path = session.file(TRIPS_FILE);
    let existing = read_resumable::<TripLine>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let mut done: HashSet<String> = existing.lines.into_iter().map(|line| line.trip.record).collect();
    let mut out = Appender::open(&path)?;

    let question_of = questions_by_record(queries);
    let with_question: Vec<(&Record, &Query)> = records
        .iter()
        .filter_map(|record| question_of.get(record.id.as_str()).map(|query| (record, *query)))
        .collect();
    session.set(
        "records_without_a_question",
        (records.len() - with_question.len()) as u64,
    );
    let wanted = args.n.unwrap_or(with_question.len());
    if with_question.len() < wanted {
        let error = RunError::item(
            ASKED_COUNT,
            format!(
                "{wanted} records were asked for and {} have a question",
                with_question.len()
            ),
        );
        session.fail_as(ASKED_COUNT, FailureKind::Shortfall, &error, 0)?;
    }
    let pace = Pace {
        poll_every: Duration::from_millis(args.poll_ms),
        give_up_after: Duration::from_millis(args.give_up_ms),
        max_in_flight: args.max_polls_in_flight,
    };

    for (record, query) in with_question.into_iter().take(wanted) {
        session.add("selected", 1);
        if done.contains(&record.id) {
            session.add("already_there", 1);
            continue;
        }
        if let Err(e) = cb_types::ids::check(&record.id) {
            session.fail(&record.id, &RunError::item(&record.id, e.to_string()), 0)?;
            continue;
        }
        if let Some(reason) = not_visible_now(record, chrono::Utc::now().timestamp()) {
            session.fail(&record.id, &RunError::item(&record.id, reason), 0)?;
            continue;
        }
        let search = SearchRequest {
            query: query.text.clone(),
            top_k: PASSAGES,
            score_threshold: CUT_OFF,
            collections: vec![label.as_str().to_string()],
        };
        // A record that the search already returns cannot be timed: its first hit would
        // be the old copy. This search is not part of the trip and may be retried.
        match with_retries(&shared.retry, |_| kb.search(tenant, &search)).await {
            Ok(before) if before.value.value.contains_record(&record.id) => {
                let error = RunError::item(
                    &record.id,
                    "the search returns the record before its upload started; remove it from the index to time it",
                );
                session.fail_as(&record.id, FailureKind::AlreadySearchable, &error, 0)?;
                continue;
            }
            Ok(_) => {}
            Err(gave_up) => {
                session.fail(&record.id, &gave_up.error, gave_up.tries)?;
                continue;
            }
        }
        let document = Document::of(record, label);
        match time_one_trip(kb, tenant, &document, &search, pace).await {
            Ok(measured) => {
                out.append(&[TripLine {
                    trip: PipelineTrip {
                        record: record.id.clone(),
                        // The clock's zero is the hand-over, where extraction would end.
                        extracted_ms: Some(0.0),
                        written_ms: Some(millis(measured.written)),
                        first_hit_ms: Some(millis(measured.first_hit)),
                        ..PipelineTrip::default()
                    },
                    timings_ms: measured.uploaded.timings_ms,
                    passages_indexed: measured.uploaded.passages_indexed,
                    polls: measured.polls,
                    polls_failed: measured.polls_failed,
                    polls_held_back: measured.polls_held_back,
                    first_hit_mode: measured.first_hit_mode,
                }])?;
                done.insert(record.id.clone());
                session.wrote();
            }
            Err(not_measured) => session.fail_as(&record.id, not_measured.kind, &not_measured.error, 1)?,
        }
    }
    out.sync()?;
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::{Gold, QueryClass, Tier};

    fn query(id: &str, gold: &[(&str, u8)]) -> Query {
        Query {
            id: id.into(),
            text: format!("Question {id}?"),
            asked_at: 0,
            class: QueryClass::Changing,
            gold: gold
                .iter()
                .map(|(record, grade)| Gold {
                    record: (*record).into(),
                    grade: *grade,
                })
                .collect(),
            thread: None,
            harmful: vec![],
        }
    }

    #[test]
    fn a_records_question_is_the_first_that_names_it_as_the_answer() {
        let queries = [
            query("q-1", &[("r-2", 3), ("r-1", 1)]),
            query("q-2", &[("r-1", 3)]),
            query("q-3", &[("r-1", 3), ("r-2", 3)]),
        ];
        let by_record = questions_by_record(&queries);
        assert_eq!(by_record["r-1"].id, "q-2");
        assert_eq!(by_record["r-2"].id, "q-1");
        assert!(!by_record.contains_key("r-3"));
    }

    #[test]
    fn a_trip_line_reads_as_a_pipeline_trip() {
        let line = TripLine {
            trip: PipelineTrip {
                record: "trip-0001".into(),
                written_ms: Some(31.5),
                first_hit_ms: Some(48.25),
                ..PipelineTrip::default()
            },
            timings_ms: [("searchable".to_string(), 21.3)].into(),
            passages_indexed: 1,
            polls: 3,
            polls_failed: 0,
            polls_held_back: 0,
            first_hit_mode: Some("reranked".into()),
        };
        let text = serde_json::to_string(&line).unwrap();
        let trip: PipelineTrip = serde_json::from_str(&text).unwrap();
        assert_eq!(trip, line.trip);
        assert!(text.contains("\"timings_ms\":{\"searchable\":21.3}"), "{text}");
        assert!(
            !text.contains("transcript_ms"),
            "a mark that was not taken is absent: {text}"
        );
        let back: TripLine = serde_json::from_str(&text).unwrap();
        assert_eq!(back, line);
    }

    #[test]
    fn a_record_the_live_search_cannot_return_now_is_named_before_any_waiting() {
        let record = |written_at, valid_until| Record {
            id: "r".into(),
            text: String::new(),
            written_at,
            valid_until,
            thread: None,
            tier: Tier::Changing,
            title: None,
            source: "test".into(),
            truth: None,
        };
        assert!(not_visible_now(&record(100, None), 200).is_none());
        assert!(not_visible_now(&record(100, Some(300)), 200).is_none());
        assert!(
            not_visible_now(&record(100, Some(200)), 200)
                .unwrap()
                .contains("valid_until")
        );
        assert!(
            not_visible_now(&record(201, None), 200)
                .unwrap()
                .contains("future")
        );
    }
}
