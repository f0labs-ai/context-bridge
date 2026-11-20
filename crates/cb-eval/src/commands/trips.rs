//! `cb-eval trips`: how long a record takes from speech to search
//! (DESIGN.md section 7).
//!
//! Every mark of a [`PipelineTrip`] is milliseconds after the end of the
//! utterance that stated the record. A percentile of a path is taken over
//! whole trips through that path; a stage is summarised by the median of its
//! per-record times. Stage percentiles are never added up.

use crate::error::EvalError;
use crate::files::{TRIPS, made_by, read_jsonl_file, sha256_of_file, write_json_file, write_text_file};
use crate::latex::{Numbers, Table, csv_line, csv_number, escape, is_plain_name};
use crate::stats::{median, nearest_rank, sorted};
use cb_types::PipelineTrip;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Where the outputs go, under the output directory.
pub const TRIPS_DIR: &str = "trips";
/// Claim 1: the P95 of hand-off to first hit is at or under 2.0 s …
pub const PASS_LINE_MS: f64 = 2_000.0;
/// … over at least 1,000 records.
pub const PASS_LINE_RECORDS: usize = 1_000;
/// The percentiles reported for a path, in thousandths.
pub const PERCENTILES: [u32; 4] = [500, 900, 950, 990];

/// A moment in a record's trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mark {
    /// The end of the utterance that stated the record: 0 by definition.
    Utterance,
    Transcript,
    Masked,
    Extracted,
    Written,
    FirstHit,
}

impl Mark {
    /// In the order a trip passes them.
    pub const ALL: [Mark; 6] = [
        Mark::Utterance,
        Mark::Transcript,
        Mark::Masked,
        Mark::Extracted,
        Mark::Written,
        Mark::FirstHit,
    ];

    /// The mark's time in a trip, milliseconds; `None` when the run did not take it.
    ///
    /// A trip timed from the hand-over (`cb-run trips`) begins where extraction ends: it
    /// has no speech, masking or extraction marks, and its clock's zero is the hand-over,
    /// so its `extracted` mark is 0 when the line leaves it out.
    pub fn of(self, trip: &PipelineTrip) -> Option<f64> {
        let begins_at_hand_over = trip.transcript_ms.is_none() && trip.masked_ms.is_none();
        match self {
            Mark::Utterance => Some(0.0),
            Mark::Transcript => trip.transcript_ms,
            Mark::Masked => trip.masked_ms,
            Mark::Extracted => trip.extracted_ms.or(begins_at_hand_over.then_some(0.0)),
            Mark::Written => trip.written_ms,
            Mark::FirstHit => trip.first_hit_ms,
        }
    }
}

/// A stretch of the pipeline measured over whole trips.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stretch {
    pub name: &'static str,
    pub from: Mark,
    pub to: Mark,
    /// When set, only trips that carry this mark count (they began with speech).
    pub requires: Option<Mark>,
}

/// The paths reported over whole trips. Claim 1's pass line is on the first.
pub const PATHS: [Stretch; 4] = [
    Stretch {
        name: "handoff_to_first_hit",
        from: Mark::Extracted,
        to: Mark::FirstHit,
        requires: None,
    },
    Stretch {
        name: "handoff_to_written",
        from: Mark::Extracted,
        to: Mark::Written,
        requires: None,
    },
    Stretch {
        name: "utterance_to_first_hit",
        from: Mark::Utterance,
        to: Mark::FirstHit,
        requires: Some(Mark::Transcript),
    },
    Stretch {
        name: "utterance_to_written",
        from: Mark::Utterance,
        to: Mark::Written,
        requires: Some(Mark::Transcript),
    },
];

/// The stages, each from one mark to the next.
pub const STAGES: [Stretch; 5] = [
    Stretch {
        name: "speech_to_text",
        from: Mark::Utterance,
        to: Mark::Transcript,
        requires: Some(Mark::Transcript),
    },
    Stretch {
        name: "masking",
        from: Mark::Transcript,
        to: Mark::Masked,
        requires: None,
    },
    Stretch {
        name: "extraction",
        from: Mark::Masked,
        to: Mark::Extracted,
        requires: None,
    },
    Stretch {
        name: "index_write",
        from: Mark::Extracted,
        to: Mark::Written,
        requires: None,
    },
    // From the hand-over, not from the upload's return: the search often finds the record
    // before the upload call has returned (see `check_trips`).
    Stretch {
        name: "hand_off_to_first_hit",
        from: Mark::Extracted,
        to: Mark::FirstHit,
        requires: None,
    },
];

impl Stretch {
    /// The stretch's time in every trip that has both marks (and the required one).
    pub fn times(&self, trips: &[PipelineTrip]) -> Vec<f64> {
        trips
            .iter()
            .filter(|trip| self.requires.is_none_or(|mark| mark.of(trip).is_some()))
            .filter_map(|trip| Some(self.to.of(trip)? - self.from.of(trip)?))
            .collect()
    }
}

/// The smallest number of trips for which a percentile is reported: at least
/// one trip must lie above it, so `n >= 1 / (1 - q)` (P99 needs 100 trips).
pub fn trips_needed(per_mille: u32) -> usize {
    1000usize.div_ceil(1000 - per_mille.min(999) as usize)
}

/// Percentiles of one path over whole trips.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathSummary {
    pub path: String,
    pub from: Mark,
    pub to: Mark,
    /// Records counted on the path: timed trips and, on a path to the first hit that timed
    /// trips travel, the records the search never returned before the run gave up on them.
    pub trips: usize,
    /// Of `trips`, the records the search never returned (`never_found` failures of the
    /// run): each counts as slower than every timed trip.
    #[serde(default)]
    pub never_found: usize,
    /// Per mille → milliseconds, by nearest rank; a percentile with too few trips is left out.
    pub percentiles_ms: BTreeMap<u32, f64>,
    /// Percentiles (per mille) that fall among the records never found: beyond the give-up time.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub percentiles_beyond_give_up: Vec<u32>,
    /// The slowest timed trip; absent when a record was never found.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_ms: Option<f64>,
}

/// The median of one stage's per-record times.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageSummary {
    pub stage: String,
    pub from: Mark,
    pub to: Mark,
    pub records: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub median_ms: Option<f64>,
}

/// Claim 1's pass line, as measured.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PassLine {
    pub path: String,
    pub per_mille: u32,
    pub limit_ms: f64,
    pub records_required: usize,
    pub records: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_ms: Option<f64>,
    pub met: bool,
}

/// `trips/summary.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TripsSummary {
    pub name: String,
    pub made_by: String,
    /// File name → sha256.
    pub inputs: BTreeMap<String, String>,
    pub trips: usize,
    pub paths: Vec<PathSummary>,
    pub stages: Vec<StageSummary>,
    pub pass_line: PassLine,
}

/// Refused: a record with two trips, a mark that is not a number or is
/// negative, and marks out of the pipeline's order.
pub fn check_trips(trips: &[PipelineTrip]) -> Result<(), EvalError> {
    let mut seen = HashSet::with_capacity(trips.len());
    for trip in trips {
        let wrong = |what: &str| EvalError::invalid(format!("trips.jsonl: record {:?}: {what}", trip.record));
        if !seen.insert(trip.record.as_str()) {
            return Err(wrong("it has two trips"));
        }
        let mut last: Option<(Mark, f64)> = None;
        for mark in Mark::ALL {
            let Some(at) = mark.of(trip) else {
                continue;
            };
            if !at.is_finite() || at < 0.0 {
                return Err(wrong(&format!(
                    "mark {mark:?} is {at}, not a time after the utterance"
                )));
            }
            if let Some((before, earlier)) = last
                && at < earlier
            {
                return Err(wrong(&format!(
                    "mark {mark:?} ({at} ms) comes before mark {before:?} ({earlier} ms)"
                )));
            }
            // The upload's return and the first hit both follow the hand-over, in either
            // order: the record becomes searchable inside the upload call, and the search
            // can find it before the call has returned. So the first hit is checked
            // against the marks before `written`, not against `written`.
            if mark != Mark::Written {
                last = Some((mark, at));
            }
        }
    }
    Ok(())
}

/// Every percentile and median of a set of trips.
pub fn summarise(
    name: &str,
    trips: &[PipelineTrip],
    never_found: usize,
    inputs: BTreeMap<String, String>,
) -> Result<TripsSummary, EvalError> {
    check_trips(trips)?;
    let paths: Vec<PathSummary> = PATHS
        .iter()
        .map(|path| {
            // A record the search never returned counts, slower than every timed trip, on
            // each path to the first hit that the run's timed trips travel: a run's records
            // carry the same marks. It was written, so a path to `written` does not count it.
            let mut times = path.times(trips);
            let lost = if path.to == Mark::FirstHit && !times.is_empty() {
                never_found
            } else {
                0
            };
            times.extend(std::iter::repeat_n(f64::INFINITY, lost));
            let times = sorted(times);
            let ranked: Vec<(u32, f64)> = PERCENTILES
                .iter()
                .filter(|q| times.len() >= trips_needed(**q))
                .filter_map(|q| Some((*q, nearest_rank(&times, *q)?)))
                .collect();
            PathSummary {
                path: path.name.to_string(),
                from: path.from,
                to: path.to,
                trips: times.len(),
                never_found: lost,
                percentiles_ms: ranked.iter().filter(|(_, v)| v.is_finite()).copied().collect(),
                percentiles_beyond_give_up: ranked
                    .iter()
                    .filter(|(_, v)| !v.is_finite())
                    .map(|(q, _)| *q)
                    .collect(),
                max_ms: times.last().copied().filter(|v| v.is_finite()),
            }
        })
        .collect();
    let stages = STAGES
        .iter()
        .map(|stage| {
            let times = sorted(stage.times(trips));
            StageSummary {
                stage: stage.name.to_string(),
                from: stage.from,
                to: stage.to,
                records: times.len(),
                median_ms: median(&times),
            }
        })
        .collect();
    let handoff = &paths[0];
    let value_ms = handoff.percentiles_ms.get(&950).copied();
    let pass_line = PassLine {
        path: handoff.path.clone(),
        per_mille: 950,
        limit_ms: PASS_LINE_MS,
        records_required: PASS_LINE_RECORDS,
        records: handoff.trips,
        value_ms,
        met: handoff.trips >= PASS_LINE_RECORDS && value_ms.is_some_and(|v| v <= PASS_LINE_MS),
    };
    Ok(TripsSummary {
        name: name.to_string(),
        made_by: made_by(),
        inputs,
        trips: trips.len(),
        paths,
        stages,
        pass_line,
    })
}

#[derive(Clone, Debug)]
pub struct TripsArgs {
    pub run: PathBuf,
    pub out: PathBuf,
    pub name: String,
}

/// What `trips` did.
#[derive(Debug)]
pub struct TripsRun {
    pub summary: TripsSummary,
    pub files: Vec<PathBuf>,
}

/// Read `trips.jsonl` and write `trips/summary.json`, `paths.csv`,
/// `stages.csv`, `table.tex` and `numbers.tex`.
pub fn run(args: &TripsArgs) -> Result<TripsRun, EvalError> {
    if !is_plain_name(&args.name) {
        return Err(EvalError::invalid(format!(
            "trips: name {:?} may hold letters, digits, '-', '_' and '.' only",
            args.name
        )));
    }
    let path = args.run.join(TRIPS);
    let trips: Vec<PipelineTrip> = read_jsonl_file(&path)?;
    let mut inputs: BTreeMap<String, String> = [(TRIPS.to_string(), sha256_of_file(&path)?)]
        .into_iter()
        .collect();
    // The records the search never returned before the run gave up on them.
    #[derive(Deserialize)]
    struct Failure {
        kind: String,
    }
    let failures = args.run.join("failures.jsonl");
    let never_found = if failures.exists() {
        inputs.insert("failures.jsonl".to_string(), sha256_of_file(&failures)?);
        read_jsonl_file::<Failure>(&failures)?
            .iter()
            .filter(|f| f.kind == "never_found")
            .count()
    } else {
        0
    };
    let summary = summarise(&args.name, &trips, never_found, inputs)?;
    let dir = args.out.join(TRIPS_DIR);
    let mut files = Vec::new();
    let mut put = |name: &str, write: &dyn Fn(&Path) -> Result<(), EvalError>| -> Result<(), EvalError> {
        let path = dir.join(name);
        write(&path)?;
        files.push(path);
        Ok(())
    };
    put("summary.json", &|p| write_json_file(p, &summary))?;
    put("paths.csv", &|p| write_text_file(p, &paths_csv(&summary)))?;
    put("stages.csv", &|p| write_text_file(p, &stages_csv(&summary)))?;
    put("table.tex", &|p| write_text_file(p, &table(&summary).render()))?;
    let numbers = numbers_of(&summary)?;
    put("numbers.tex", &|p| write_text_file(p, &numbers.render()))?;
    Ok(TripsRun { summary, files })
}

fn optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, csv_number)
}

fn paths_csv(summary: &TripsSummary) -> String {
    let mut text = csv_line(&[
        "name", "path", "trips", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms",
    ]);
    for path in &summary.paths {
        let mut row = vec![summary.name.clone(), path.path.clone(), path.trips.to_string()];
        row.extend(
            PERCENTILES
                .iter()
                .map(|q| optional(path.percentiles_ms.get(q).copied())),
        );
        row.push(optional(path.max_ms));
        text.push_str(&csv_line(&row));
    }
    text
}

fn stages_csv(summary: &TripsSummary) -> String {
    let mut text = csv_line(&["name", "stage", "records", "median_ms"]);
    for stage in &summary.stages {
        text.push_str(&csv_line(&[
            summary.name.clone(),
            stage.stage.clone(),
            stage.records.to_string(),
            optional(stage.median_ms),
        ]));
    }
    text
}

/// Whole milliseconds, or `--`.
fn ms(value: Option<f64>) -> String {
    value.map_or_else(|| "--".to_string(), |v| format!("{v:.0}"))
}

fn table(summary: &TripsSummary) -> Table {
    let mut rows: Vec<Vec<String>> = summary
        .paths
        .iter()
        .map(|p| {
            let mut row = vec![escape(&p.path), p.trips.to_string()];
            row.extend(PERCENTILES.iter().map(|q| ms(p.percentiles_ms.get(q).copied())));
            row
        })
        .collect();
    rows.extend(summary.stages.iter().map(|s| {
        vec![
            format!("{} (stage)", escape(&s.stage)),
            s.records.to_string(),
            format!(
                "{} (median)",
                s.median_ms
                    .map_or_else(|| "--".to_string(), |v| format!("{v:.1}"))
            ),
            String::new(),
            String::new(),
            String::new(),
        ]
    }));
    Table {
        comments: vec![
            format!(
                "Written by {} (cb-eval trips). Do not edit; run trips again.",
                summary.made_by
            ),
            format!(
                "{}: percentiles over whole trips (nearest rank; one is left out when fewer than 1/(1-q) trips \
                 have the path); stages are medians of per-record times and are never added up.",
                summary.name
            ),
        ],
        preamble: Vec::new(),
        columns: "lrrrrr".to_string(),
        header: ["Path", "Trips", "P50 (ms)", "P90 (ms)", "P95 (ms)", "P99 (ms)"]
            .map(String::from)
            .to_vec(),
        rows,
    }
}

/// Keys: `<name>/trips/<path>/trips`, `/p50-ms`, `/p90-ms`, `/p95-ms`,
/// `/p99-ms`, `/max-ms` and `/p95-s`; `<name>/trips/stage/<stage>/median-ms`
/// and `/records`; `<name>/trips/pass-line/met` (yes or no) and `/p95-s`.
pub fn numbers_of(summary: &TripsSummary) -> Result<Numbers, EvalError> {
    let name = &summary.name;
    let mut numbers = Numbers::new(vec![
        format!(
            "Written by {} (cb-eval trips). Do not edit; run trips again.",
            summary.made_by
        ),
        format!("{name}: {} trips.", summary.trips),
    ]);
    numbers.push(format!("{name}/trips/trips"), summary.trips.to_string())?;
    for path in &summary.paths {
        let key = format!("{name}/trips/{}", path.path);
        numbers.push(format!("{key}/trips"), path.trips.to_string())?;
        for q in PERCENTILES {
            if let Some(value) = path.percentiles_ms.get(&q) {
                numbers.push(format!("{key}/p{}-ms", q / 10), format!("{value:.0}"))?;
            }
        }
        if let Some(p95) = path.percentiles_ms.get(&950) {
            numbers.push(format!("{key}/p95-s"), format!("{:.2}", p95 / 1000.0))?;
        }
        if let Some(max) = path.max_ms {
            numbers.push(format!("{key}/max-ms"), format!("{max:.0}"))?;
        }
    }
    for stage in &summary.stages {
        let key = format!("{name}/trips/stage/{}", stage.stage);
        numbers.push(format!("{key}/records"), stage.records.to_string())?;
        if let Some(median) = stage.median_ms {
            numbers.push(format!("{key}/median-ms"), format!("{median:.1}"))?;
        }
    }
    numbers.push(
        format!("{name}/trips/pass-line/met"),
        if summary.pass_line.met { "yes" } else { "no" }.to_string(),
    )?;
    Ok(numbers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{read_json_file, test_dir, write_jsonl_file};
    use std::fs;

    fn trip(record: &str, marks: [Option<f64>; 5]) -> PipelineTrip {
        PipelineTrip {
            record: record.to_string(),
            transcript_ms: marks[0],
            masked_ms: marks[1],
            extracted_ms: marks[2],
            written_ms: marks[3],
            first_hit_ms: marks[4],
        }
    }

    #[test]
    fn a_percentile_needs_one_trip_above_it() {
        assert_eq!(trips_needed(500), 2);
        assert_eq!(trips_needed(900), 10);
        assert_eq!(trips_needed(950), 20);
        assert_eq!(trips_needed(990), 100);
    }

    #[test]
    fn paths_are_whole_trips_and_stages_are_medians_of_their_own_times() {
        // Three trips from speech. Stage times: speech 100, 200, 300; masking 10, 10, 50;
        // extraction 500, 400, 900; index write 20, 30, 10; hand-off to first hit 25, 35, 50.
        let trips = vec![
            trip(
                "a",
                [Some(100.0), Some(110.0), Some(610.0), Some(630.0), Some(635.0)],
            ),
            trip(
                "b",
                [Some(200.0), Some(210.0), Some(610.0), Some(640.0), Some(645.0)],
            ),
            trip(
                "c",
                [
                    Some(300.0),
                    Some(350.0),
                    Some(1_250.0),
                    Some(1_260.0),
                    Some(1_300.0),
                ],
            ),
        ];
        let summary = summarise("t", &trips, 0, BTreeMap::new()).unwrap();
        let median_of = |stage: &str| {
            summary
                .stages
                .iter()
                .find(|s| s.stage == stage)
                .unwrap()
                .median_ms
        };
        assert_eq!(median_of("speech_to_text"), Some(200.0));
        assert_eq!(median_of("masking"), Some(10.0));
        assert_eq!(median_of("extraction"), Some(500.0));
        assert_eq!(median_of("index_write"), Some(20.0));
        assert_eq!(median_of("hand_off_to_first_hit"), Some(35.0));
        // The whole trips from the utterance: 635, 645, 1 300. Its median is 645, while the
        // medians of speech, masking, extraction and hand-off to first hit add up to
        // 200 + 10 + 500 + 35 = 745: stage medians do not add up to a trip.
        let whole = summary
            .paths
            .iter()
            .find(|p| p.path == "utterance_to_first_hit")
            .unwrap();
        assert_eq!(whole.trips, 3);
        assert_eq!(whole.percentiles_ms.get(&500), Some(&645.0));
        // Three trips are too few for P90 and above.
        assert_eq!(whole.percentiles_ms.len(), 1);
        assert_eq!(whole.max_ms, Some(1_300.0));
        // Hand-off to first hit: 635 - 610 = 25, 645 - 610 = 35, 1 300 - 1 250 = 50.
        let handoff = &summary.paths[0];
        assert_eq!(handoff.path, "handoff_to_first_hit");
        assert_eq!(handoff.percentiles_ms.get(&500), Some(&35.0));
        assert!(!summary.pass_line.met);
        assert_eq!(summary.pass_line.value_ms, None);
    }

    #[test]
    fn the_search_may_find_a_record_before_its_upload_returns() {
        // Found at 68 ms, the upload call returned at 113 ms: both after the hand-over.
        let early = vec![
            trip("early", [None, None, Some(0.0), Some(113.0), Some(68.0)]),
            trip("late", [None, None, Some(0.0), Some(90.0), Some(120.0)]),
        ];
        let summary = summarise("t", &early, 0, BTreeMap::new()).unwrap();
        assert_eq!(summary.paths[0].max_ms, Some(120.0));
        // A first hit before the hand-over is still refused.
        let before = vec![trip("bad", [None, None, Some(50.0), Some(113.0), Some(40.0)])];
        let error = summarise("t", &before, 0, BTreeMap::new())
            .unwrap_err()
            .to_string();
        assert!(error.contains("comes before mark Extracted"), "{error}");
    }

    #[test]
    fn trips_without_speech_count_from_the_hand_off_only() {
        // A record handed straight to the knowledge service: no speech marks, hand-off at 0.
        let trips = vec![
            trip("direct", [None, None, Some(0.0), Some(800.0), Some(900.0)]),
            trip(
                "spoken",
                [Some(100.0), Some(110.0), Some(600.0), Some(700.0), Some(750.0)],
            ),
        ];
        let summary = summarise("t", &trips, 0, BTreeMap::new()).unwrap();
        let path = |name: &str| summary.paths.iter().find(|p| p.path == name).unwrap();
        assert_eq!(path("handoff_to_first_hit").trips, 2);
        // Only the spoken trip began at an utterance.
        assert_eq!(path("utterance_to_first_hit").trips, 1);
        let stage = |name: &str| summary.stages.iter().find(|s| s.stage == name).unwrap();
        assert_eq!(stage("speech_to_text").records, 1);
        assert_eq!(stage("index_write").records, 2);
        // index write: 800 and 100: median 450.
        assert_eq!(stage("index_write").median_ms, Some(450.0));
        // A line written by `cb-run trips` leaves the hand-over mark out: its zero is the
        // hand-over, so it counts the same as the direct trip above.
        let from_hand_over = vec![
            trip("timed-1", [None, None, None, Some(800.0), Some(900.0)]),
            trip("timed-2", [None, None, None, Some(850.0), Some(1_000.0)]),
        ];
        let summary = summarise("t", &from_hand_over, 0, BTreeMap::new()).unwrap();
        assert_eq!(summary.paths[0].trips, 2);
        assert_eq!(summary.paths[0].max_ms, Some(1_000.0));
    }

    #[test]
    fn a_record_never_found_counts_as_slower_than_every_timed_trip() {
        let timed = |n: usize| -> Vec<PipelineTrip> {
            (0..n)
                .map(|i| {
                    trip(
                        &format!("r{i}"),
                        [None, None, Some(0.0), Some(100.0), Some(500.0)],
                    )
                })
                .collect()
        };
        // 975 timed at 500 ms and 25 never found: P95 is the 950th of 1 000, a timed trip;
        // P99 falls among the records never found.
        let summary = summarise("t", &timed(975), 25, BTreeMap::new()).unwrap();
        let handoff = &summary.paths[0];
        assert_eq!((handoff.trips, handoff.never_found), (1_000, 25));
        assert_eq!(handoff.percentiles_ms.get(&950), Some(&500.0));
        assert_eq!(handoff.percentiles_beyond_give_up, vec![990]);
        assert_eq!(handoff.max_ms, None);
        assert!(summary.pass_line.met && summary.pass_line.records == 1_000);
        // 60 never found of 1 000: P95 itself is beyond the give-up time, and the line is not met.
        let summary = summarise("t", &timed(940), 60, BTreeMap::new()).unwrap();
        assert_eq!(summary.pass_line.value_ms, None);
        assert!(!summary.pass_line.met);
        // No timed trip begins with speech here, so that path counts none; nor does the path
        // to `written`.
        assert_eq!(summary.paths[2].never_found, 0);
        assert_eq!(
            (summary.paths[1].path.as_str(), summary.paths[1].never_found),
            ("handoff_to_written", 0)
        );
        // Trips put together from speech: the records never found count from the utterance too.
        let from_speech: Vec<PipelineTrip> = (0..19)
            .map(|i| {
                trip(
                    &format!("s{i}"),
                    [
                        Some(100.0),
                        Some(101.0),
                        Some(1_100.0),
                        Some(1_200.0),
                        Some(1_150.0),
                    ],
                )
            })
            .collect();
        let summary = summarise("t", &from_speech, 1, BTreeMap::new()).unwrap();
        let utterance = summary
            .paths
            .iter()
            .find(|p| p.path == "utterance_to_first_hit")
            .unwrap();
        assert_eq!((utterance.trips, utterance.never_found), (20, 1));
        assert_eq!(utterance.percentiles_ms.get(&950), Some(&1_150.0));
    }

    #[test]
    fn the_pass_line_needs_a_thousand_trips_and_a_p95_at_or_under_two_seconds() {
        // 1 000 trips with hand-off to first hit of 1, 2, …, 1 000 ms × 2: P95 is the 950th, 1 900 ms.
        let trips: Vec<PipelineTrip> = (1..=1000)
            .map(|i| {
                trip(
                    &format!("r{i}"),
                    [None, None, Some(0.0), None, Some(2.0 * f64::from(i))],
                )
            })
            .collect();
        let summary = summarise("t", &trips, 0, BTreeMap::new()).unwrap();
        assert_eq!(summary.pass_line.value_ms, Some(1_900.0));
        assert!(summary.pass_line.met);
        assert_eq!(summary.paths[0].percentiles_ms.get(&990), Some(&1_980.0));
        // 999 trips are not enough, whatever their times.
        let summary = summarise("t", &trips[..999], 0, BTreeMap::new()).unwrap();
        assert!(!summary.pass_line.met);
        // Slower trips miss the line: P95 of 1 000 trips of 2.2 ms × i is 2 090 ms.
        let slow: Vec<PipelineTrip> = (1..=1000)
            .map(|i| {
                trip(
                    &format!("r{i}"),
                    [None, None, Some(0.0), None, Some(2.2 * f64::from(i))],
                )
            })
            .collect();
        assert!(!summarise("t", &slow, 0, BTreeMap::new()).unwrap().pass_line.met);
    }

    #[test]
    fn trips_that_break_the_contract_are_refused() {
        let refusal = |trips: Vec<PipelineTrip>| {
            summarise("t", &trips, 0, BTreeMap::new())
                .unwrap_err()
                .to_string()
        };
        assert!(refusal(vec![trip("a", [None; 5]), trip("a", [None; 5])]).contains("two trips"));
        assert!(refusal(vec![trip("a", [Some(-1.0), None, None, None, None])]).contains("not a time"));
        assert!(refusal(vec![trip("a", [Some(f64::NAN), None, None, None, None])]).contains("not a time"));
        // Written before extracted.
        assert!(
            refusal(vec![trip("a", [None, None, Some(500.0), Some(400.0), None])]).contains("comes before")
        );
    }

    #[test]
    fn trips_writes_its_files_and_numbers() {
        let dir = test_dir("trips");
        let trips: Vec<PipelineTrip> = (1..=40)
            .map(|i| {
                let t = f64::from(i);
                trip(
                    &format!("r{i}"),
                    [
                        Some(t),
                        Some(2.0 * t),
                        Some(10.0 * t),
                        Some(11.0 * t),
                        Some(12.0 * t),
                    ],
                )
            })
            .collect();
        write_jsonl_file(&dir.join(TRIPS), &trips).unwrap();
        let done = run(&TripsArgs {
            run: dir.clone(),
            out: dir.clone(),
            name: "tiny".to_string(),
        })
        .unwrap();
        assert_eq!(done.files.len(), 5);
        let back: TripsSummary = read_json_file(&dir.join("trips/summary.json")).unwrap();
        assert_eq!(back, done.summary);
        // Hand-off to first hit is 2 * i ms: of 40 trips, P95 is the 38th: 76 ms.
        let numbers = fs::read_to_string(dir.join("trips/numbers.tex")).unwrap();
        assert!(numbers.contains("\\csname cbnum/tiny/trips/handoff_to_first_hit/p95-ms\\endcsname{76}"));
        assert!(numbers.contains("\\csname cbnum/tiny/trips/handoff_to_first_hit/p95-s\\endcsname{0.08}"));
        // Forty trips are too few for P99.
        assert!(!numbers.contains("handoff_to_first_hit/p99-ms"));
        // Extraction takes 8 * i ms: the median of 8, 16, …, 320 is (160 + 168) / 2 = 164.
        assert!(numbers.contains("\\csname cbnum/tiny/trips/stage/extraction/median-ms\\endcsname{164.0}"));
        assert!(numbers.contains("\\csname cbnum/tiny/trips/pass-line/met\\endcsname{no}"));
        let paths = fs::read_to_string(dir.join("trips/paths.csv")).unwrap();
        assert!(
            paths.contains("tiny,handoff_to_first_hit,40,40,72,76,,80\n"),
            "{paths}"
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
