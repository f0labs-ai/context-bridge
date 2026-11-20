//! `cb-run trips` against the stand-in: an upload and the searches that wait for it.

use crate::fake::{Action, Fake, Route};
use crate::support::*;
use cb_run::commands::trips::{TRIPS_FILE, TripsArgs, trips};
use serde_json::json;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// An hour ago: a moment the live search, asked as of now, may return.
fn an_hour_ago() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
        - 3_600
}

/// Records each with a question that names it (the stand-in's search finds a document
/// by the record id in the question).
fn data_set(dir: &Path, ids: &[&str]) {
    let written_at = an_hour_ago();
    let records: Vec<_> = ids
        .iter()
        .map(|id| record(id, &format!("A new fact about {id}."), written_at))
        .collect();
    let queries: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(n, id)| {
            query(
                &format!("q-{n}"),
                &format!("What is new about {id}?"),
                written_at + 60,
                &[id],
            )
        })
        .collect();
    write_data_set(dir, &records, &queries);
}

fn args(data: &Path, run: &Path, poll_ms: u64, give_up_ms: u64, max_polls: usize) -> TripsArgs {
    TripsArgs {
        data: data.to_path_buf(),
        tenant: TENANT.into(),
        collection: LABEL.into(),
        run: run.to_path_buf(),
        n: None,
        poll_ms,
        give_up_ms,
        max_polls_in_flight: max_polls,
    }
}

#[tokio::test]
async fn each_trip_is_timed_from_one_clock_and_written_in_order() {
    let fake = Fake::start().await;
    fake.set(&fake.state.upload_takes, Duration::from_millis(30));
    fake.set(&fake.state.searchable_after, Duration::from_millis(60));
    fake.set(&fake.state.search_takes, Duration::from_millis(5));
    let dir = Scratch::new("trips-happy");
    let (data, run) = (dir.join("data"), dir.join("run"));
    data_set(&data, &["trip-0001", "trip-0002", "trip-0003"]);

    let report = trips(&kb(&fake), &args(&data, &run, 10, 5_000, 4), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let written = lines(&run.join(TRIPS_FILE));
    let ids: Vec<&str> = written.iter().map(|l| l["record"].as_str().unwrap()).collect();
    assert_eq!(ids, ["trip-0001", "trip-0002", "trip-0003"]);
    for line in &written {
        let (written_ms, first_hit_ms) = (
            line["written_ms"].as_f64().unwrap(),
            line["first_hit_ms"].as_f64().unwrap(),
        );
        assert!(written_ms >= 30.0, "{line}");
        // Searchable 60 ms after the upload started, and that answer took 5 ms to arrive.
        assert!((65.0..2_000.0).contains(&first_hit_ms), "{line}");
        assert!(line["polls"].as_u64().unwrap() >= 2, "{line}");
        assert_eq!(line["timings_ms"]["searchable"], 21.3);
        assert_eq!(line["passages_indexed"], 1);
        assert_eq!(line["first_hit_mode"], "reranked");
        assert!(
            line.get("transcript_ms").is_none(),
            "a mark that was not taken is absent"
        );
    }
    // The searches start at the same instant as the upload.
    let ingest = fake
        .seen(Route::Ingest)
        .into_iter()
        .find(|s| s.key.contains("trip-0001"))
        .unwrap();
    let searches: Vec<_> = fake
        .seen(Route::Search)
        .into_iter()
        .filter(|s| s.key.contains("trip-0001"))
        .collect();
    // The first search is the check made before the trip; the second is the first poll.
    let first_poll = &searches[1];
    let apart = if first_poll.at > ingest.at {
        first_poll.at - ingest.at
    } else {
        ingest.at - first_poll.at
    };
    assert!(apart < Duration::from_millis(25), "{apart:?}");
    // A search is asked as a call asks it.
    assert_eq!(first_poll.body["top_k"], 4);
    assert_eq!(first_poll.body["score_threshold"], 0.12);
    assert_eq!(first_poll.body["collections"], json!(["cb"]));
    assert_no_secret_in(&run);
}

#[tokio::test]
async fn a_hit_counts_when_its_answer_arrived_and_may_come_before_the_upload_returns() {
    let fake = Fake::start().await;
    fake.set(&fake.state.upload_takes, Duration::from_millis(150));
    let dir = Scratch::new("trips-arrival");
    let (data, run) = (dir.join("data"), dir.join("run"));
    data_set(&data, &["trip-0001"]);
    // The check before the trip answers at once; the first poll takes 400 ms to answer;
    // the polls after it answer at once.
    fake.rule(Route::Search, "trip-0001", 1, Action::Slow(Duration::ZERO));
    fake.rule(
        Route::Search,
        "trip-0001",
        1,
        Action::Slow(Duration::from_millis(400)),
    );

    let report = trips(&kb(&fake), &args(&data, &run, 10, 5_000, 8), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let line = &lines(&run.join(TRIPS_FILE))[0];
    let (written_ms, first_hit_ms) = (
        line["written_ms"].as_f64().unwrap(),
        line["first_hit_ms"].as_f64().unwrap(),
    );
    assert!(
        first_hit_ms < 300.0,
        "a later poll's quick answer counts, not the slow first one: {line}"
    );
    assert!(written_ms >= 150.0, "{line}");
    assert!(
        first_hit_ms < written_ms,
        "found before the upload returned, and kept: {line}"
    );
}

#[tokio::test]
async fn polls_overlap_but_never_more_than_the_bound() {
    let fake = Fake::start().await;
    fake.set(&fake.state.searchable_after, Duration::from_millis(300));
    fake.set(&fake.state.search_takes, Duration::from_millis(80));
    let dir = Scratch::new("trips-bound");
    let (data, run) = (dir.join("data"), dir.join("run"));
    data_set(&data, &["trip-0001"]);

    let report = trips(&kb(&fake), &args(&data, &run, 5, 5_000, 3), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let most = fake.state.most_searches_at_once.load(Ordering::SeqCst);
    assert!((2..=3).contains(&most), "{most} searches at once");
    let line = &lines(&run.join(TRIPS_FILE))[0];
    assert!(line["polls_held_back"].as_u64().unwrap() > 0, "{line}");
}

#[tokio::test]
async fn a_record_never_found_is_a_failure_and_a_later_run_times_it() {
    let fake = Fake::start().await;
    fake.set(&fake.state.searchable_after, Duration::from_secs(60));
    let dir = Scratch::new("trips-never");
    let (data, run) = (dir.join("data"), dir.join("run"));
    data_set(&data, &["trip-0001"]);

    let report = trips(&kb(&fake), &args(&data, &run, 10, 300, 4), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("trip-0001".to_string(), "never_found".to_string(), 1)]
    );
    assert!(lines(&run.join(TRIPS_FILE)).is_empty());

    fake.set(&fake.state.searchable_after, Duration::ZERO);
    // The index still holds the document from the first try: it cannot be timed again
    // until it is removed, and the run says so instead of timing a stale copy.
    let report = trips(&kb(&fake), &args(&data, &run, 10, 300, 4), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("trip-0001".to_string(), "already_searchable".to_string(), 0)]
    );

    fake.state.documents.lock().unwrap().clear();
    let report = trips(&kb(&fake), &args(&data, &run, 10, 2_000, 4), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(lines(&run.join(TRIPS_FILE)).len(), 1);
    assert!(failures(&run).is_empty());
}

#[tokio::test]
async fn a_failed_upload_is_tried_once_and_timed_on_the_next_run() {
    let fake = Fake::start().await;
    let dir = Scratch::new("trips-upload-failure");
    let (data, run) = (dir.join("data"), dir.join("run"));
    data_set(&data, &["trip-0001", "trip-0002"]);
    fake.rule(Route::Ingest, "trip-0001", 1, Action::Status(503));

    let report = trips(&kb(&fake), &args(&data, &run, 10, 2_000, 4), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("trip-0001".to_string(), "http_status".to_string(), 1)]
    );
    assert_eq!(
        fake.count(Route::Ingest, "trip-0001"),
        1,
        "a retried upload would be another trip"
    );
    assert_eq!(lines(&run.join(TRIPS_FILE)).len(), 1);

    let report = trips(&kb(&fake), &args(&data, &run, 10, 2_000, 4), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let ids: Vec<String> = lines(&run.join(TRIPS_FILE))
        .iter()
        .map(|l| l["record"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, ["trip-0002", "trip-0001"]);
}

#[tokio::test]
async fn asking_for_more_trips_than_there_are_is_a_failure_not_a_short_run() {
    let fake = Fake::start().await;
    let dir = Scratch::new("trips-shortfall");
    let (data, run) = (dir.join("data"), dir.join("run"));
    data_set(&data, &["trip-0001"]);
    let mut asked = args(&data, &run, 10, 2_000, 4);
    asked.n = Some(1_000);
    let report = trips(&kb(&fake), &asked, &shared()).await.unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("n".to_string(), "shortfall".to_string(), 0)]
    );
    assert_eq!(
        lines(&run.join(TRIPS_FILE)).len(),
        1,
        "what exists is timed all the same"
    );
}
