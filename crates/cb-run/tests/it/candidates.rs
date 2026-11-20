//! `cb-run candidates` against the stand-in.

use crate::fake::{Action, Fake, Route, point_id};
use crate::support::*;
use cb_run::commands::candidates::{CANDIDATES_FILE, CandidatesArgs, candidates};
use cb_run::commands::upload::{UploadArgs, upload};
use cb_types::{Query, Record};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::Path;

/// 2026-06-01T00:00:00Z, on the hour.
const T0: i64 = 1_780_272_000;
const HELP_TEXT: &str = "Help text without a date.";

fn records() -> Vec<Record> {
    let mut first = changing(
        "inc-1.s1",
        "inc-1",
        "Northgate outage",
        "Home internet is down in Northgate.",
        T0 + 600,
    );
    first.valid_until = Some(T0 + 2_400);
    vec![
        first,
        changing(
            "inc-1.s2",
            "inc-1",
            "Northgate outage",
            "The cause is a cut fibre.\n\nA crew is on site.",
            T0 + 2_400,
        ),
        changing(
            "inc-1.s3",
            "inc-1",
            "Northgate outage",
            "Service is back.",
            T0 + 7_200,
        ),
        record("doc-1.1", "How to restart the router.", T0 - 100 * 86_400),
    ]
}

fn queries() -> Vec<Query> {
    vec![
        query(
            "q-1",
            "Is home internet down in Northgate?",
            T0 + 3_300,
            &["inc-1.s2"],
        ),
        query("q-2", "Is the fibre fixed yet?", T0 + 7_300, &["inc-1.s3"]),
        query("q-3", "Why is my internet out?", T0 + 900, &["inc-1.s1"]),
    ]
}

/// A data set uploaded through `upload` into the stand-in, plus one document of the same
/// label that is not a record (a help page with no date).
async fn loaded(fake: &Fake, dir: &Scratch) -> (std::path::PathBuf, std::path::PathBuf) {
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &records(), &queries());
    let uploaded = upload(
        &kb(fake),
        &UploadArgs {
            data: data.clone(),
            tenant: TENANT.into(),
            collection: LABEL.into(),
            run: run.clone(),
            concurrency: 2,
            only: None,
        },
        &shared(),
    )
    .await
    .unwrap();
    assert!(uploaded.is_complete());
    fake.add_document(TENANT, LABEL, "https://docs.example.com/a", HELP_TEXT, json!({}));
    (data, run)
}

fn args(data: &Path, run: &Path, concurrency: usize) -> CandidatesArgs {
    CandidatesArgs {
        data: data.to_path_buf(),
        tenant: TENANT.into(),
        collection: LABEL.into(),
        run: run.to_path_buf(),
        concurrency,
    }
}

fn records_of(list: &Value) -> Vec<String> {
    list.as_array()
        .unwrap()
        .iter()
        .map(|s| s["record"].as_str().unwrap().to_string())
        .collect()
}

const VIEWS: [&str; 7] = [
    "now",
    "batch_300",
    "batch_1800",
    "batch_3600",
    "window_3600",
    "window_14400",
    "window_86400",
];

#[tokio::test]
async fn seven_lines_per_question_in_question_order_with_points_joined_to_records() {
    let fake = Fake::start().await;
    let dir = Scratch::new("candidates-happy");
    let (data, run) = loaded(&fake, &dir).await;
    let report = candidates(&kb(&fake), &args(&data, &run, 1), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));

    let written = lines(&run.join(CANDIDATES_FILE));
    assert_eq!(written.len(), 21);
    let keys: Vec<(String, String)> = written
        .iter()
        .map(|l| {
            (
                l["query"].as_str().unwrap().to_string(),
                l["view"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    let expected: Vec<(String, String)> = ["q-1", "q-2", "q-3"]
        .iter()
        .flat_map(|q| VIEWS.iter().map(move |v| (q.to_string(), v.to_string())))
        .collect();
    assert_eq!(
        keys, expected,
        "question order, then the views in the design's order"
    );

    let help = point_id(TENANT, LABEL, HELP_TEXT);
    let line = |q: &str, v: &str| {
        written
            .iter()
            .find(|l| l["query"] == q && l["view"] == v)
            .unwrap()
            .clone()
    };
    // q-1, asked at 00:55. `now`: everything written by then, the second update once (its
    // best position, though it has two passages), the undated page under its point id.
    let now = line("q-1", "now");
    assert_eq!(now["as_of"], T0 + 3_300);
    assert_eq!(
        records_of(&now["dense"]),
        ["doc-1.1", "inc-1.s1", "inc-1.s2", help.as_str()]
    );
    // The list by keywords is the stand-in's reverse: the second update's second passage
    // comes first, and the record keeps that position and that score.
    assert_eq!(
        records_of(&now["sparse"]),
        ["inc-1.s2", help.as_str(), "inc-1.s1", "doc-1.1"]
    );
    assert_eq!(now["sparse"][0]["score"], 12.5);
    // Relevance: each record's best passage, best first.
    assert_eq!(
        records_of(&now["rerank"]),
        ["inc-1.s2", help.as_str(), "inc-1.s1", "doc-1.1"]
    );
    assert_eq!(
        now["timings_ms"],
        json!({"embed": 7.1, "rerank": 31.0, "search": 9.4})
    );
    // A batch index refreshed at 00:30 had only the first update; one refreshed at 00:00 none.
    assert_eq!(
        records_of(&line("q-1", "batch_1800")["dense"]),
        ["doc-1.1", "inc-1.s1", help.as_str()]
    );
    assert_eq!(
        records_of(&line("q-1", "batch_3600")["dense"]),
        ["doc-1.1", help.as_str()]
    );
    // A window sees only what was written in it, never an undated page.
    assert_eq!(
        records_of(&line("q-1", "window_3600")["dense"]),
        ["inc-1.s1", "inc-1.s2"]
    );
    for view in &VIEWS[1..] {
        let other = line("q-1", view);
        assert_eq!(other["rerank"], json!([]), "{view}");
        assert!(
            other.get("timings_ms").is_none(),
            "{view}: one request, one set of timings"
        );
    }
    // q-3, asked at 00:15: the second update was not written yet.
    assert_eq!(
        records_of(&line("q-3", "now")["dense"]),
        ["doc-1.1", "inc-1.s1", help.as_str()]
    );

    // The request is the contract's.
    let sent = &fake.seen(Route::Candidates)[0].body;
    assert_eq!(sent["query"], "Is home internet down in Northgate?");
    assert_eq!(
        (
            sent["as_of"].as_i64(),
            sent["n"].as_u64(),
            sent["exact"].as_bool()
        ),
        (Some(T0 + 3_300), Some(100), Some(true))
    );
    assert_eq!(sent["collections"], json!(["cb"]));
    assert_eq!(sent["rerank"], json!({"view": "now", "per_list": 64}));
    let names: Vec<&str> = sent["views"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, VIEWS);
    assert_eq!(sent["views"][3], json!({"name": "batch_3600", "index_as_of": T0}));
    assert_eq!(
        sent["views"][4],
        json!({"name": "window_3600", "written_from": T0 - 300})
    );

    let entry = last_entry(&run);
    assert_eq!(entry["counts"]["records_uploaded_in_this_run"], 4);
    assert_eq!(entry["counts"]["records_in_data_set"], 4);
    assert_eq!(entry["counts"]["lines_written"], 21);
    assert_no_secret_in(&run);
}

#[tokio::test]
async fn with_concurrency_a_resume_loses_and_repeats_nothing() {
    let fake = Fake::start().await;
    let dir = Scratch::new("candidates-resume");
    let (data, run) = loaded(&fake, &dir).await;
    fake.forget_requests();
    fake.rule(
        Route::Candidates,
        "Is the fibre fixed yet?",
        3,
        Action::Status(503),
    );
    fake.rule(
        Route::Candidates,
        "Why is my internet out?",
        1,
        Action::WrongShape,
    );

    let report = candidates(&kb(&fake), &args(&data, &run, 3), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("q-2".to_string(), "http_status".to_string(), 3)]
    );
    let first: Vec<Value> = lines(&run.join(CANDIDATES_FILE));
    assert_eq!(first.len(), 14);
    assert_eq!(
        fake.count(Route::Candidates, "Why is my internet out?"),
        2,
        "a malformed answer is tried again"
    );

    fake.clear_rules();
    fake.forget_requests();
    let report = candidates(&kb(&fake), &args(&data, &run, 3), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(
        fake.seen(Route::Candidates).len(),
        1,
        "only the missing question is asked"
    );
    let all = lines(&run.join(CANDIDATES_FILE));
    assert_eq!(all.len(), 21);
    let keys: HashSet<(String, String)> = all
        .iter()
        .map(|l| {
            (
                l["query"].as_str().unwrap().to_string(),
                l["view"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(keys.len(), 21, "no (question, view) twice");
    assert!(failures(&run).is_empty());
}

#[tokio::test]
async fn a_question_cut_off_mid_way_gets_only_its_missing_views() {
    let fake = Fake::start().await;
    let dir = Scratch::new("candidates-partial");
    let (data, run) = loaded(&fake, &dir).await;
    candidates(&kb(&fake), &args(&data, &run, 1), &shared())
        .await
        .unwrap();
    // Keep q-1's first three lines and a piece of its fourth, as a crash would.
    let path = run.join(CANDIDATES_FILE);
    let text = std::fs::read_to_string(&path).unwrap();
    let kept: Vec<&str> = text.lines().take(4).collect();
    std::fs::write(&path, format!("{}\n{}", kept[..3].join("\n"), &kept[3][..20])).unwrap();
    fake.forget_requests();

    let report = candidates(&kb(&fake), &args(&data, &run, 1), &shared())
        .await
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(
        fake.seen(Route::Candidates).len(),
        3,
        "q-1 for its four missing views, q-2, q-3"
    );
    let all = lines(&path);
    assert_eq!(all.len(), 21);
    let q1: Vec<&str> = all
        .iter()
        .filter(|l| l["query"] == "q-1")
        .map(|l| l["view"].as_str().unwrap())
        .collect();
    assert_eq!(q1, VIEWS, "the three kept, then the four missing, each once");
}

#[tokio::test]
async fn an_answer_without_every_view_is_a_failure_that_is_not_tried_again() {
    let fake = Fake::start().await;
    let dir = Scratch::new("candidates-missing-view");
    let (data, run) = loaded(&fake, &dir).await;
    let short = json!({
        "views": {"now": {"dense": [], "sparse": []}},
        "rerank": [], "passages": {}, "timings_ms": {"embed": 1.0, "search": 1.0, "rerank": 0.0},
    });
    fake.rule(
        Route::Candidates,
        "Is home internet down",
        3,
        Action::Body(short.to_string()),
    );
    let report = candidates(&kb(&fake), &args(&data, &run, 1), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("q-1".to_string(), "contract".to_string(), 1)]
    );
    assert_eq!(fake.count(Route::Candidates, "Is home internet down"), 1);
    let failure = &failures(&run)[0];
    assert!(
        failure["message"]
            .as_str()
            .unwrap()
            .contains("did not answer view batch_300")
    );
    assert!(
        lines(&run.join(CANDIDATES_FILE))
            .iter()
            .all(|l| l["query"] != "q-1"),
        "no partial question"
    );
}

#[tokio::test]
async fn an_index_that_holds_other_times_than_the_data_set_is_a_failure() {
    let fake = Fake::start().await;
    let dir = Scratch::new("candidates-times");
    let (data, run) = loaded(&fake, &dir).await;
    // The same document, filed again a minute later than the data set says.
    fake.add_document(
        TENANT,
        LABEL,
        "cb/inc-1.s1.md",
        "Home internet is down in Northgate.",
        json!({"written_at": T0 + 660, "valid_until": T0 + 2_400, "document_ref": "inc-1.s1"}),
    );
    let report = candidates(&kb(&fake), &args(&data, &run, 1), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    let message = failures(&run)[0]["message"].as_str().unwrap().to_string();
    assert!(
        message.contains("record inc-1.s1") && message.contains("written_at"),
        "{message}"
    );
}

#[tokio::test]
async fn a_build_without_the_candidates_route_is_refused_before_any_question() {
    let fake = Fake::start().await;
    let dir = Scratch::new("candidates-features");
    let (data, run) = loaded(&fake, &dir).await;
    fake.set_features(&["document_times", "search_as_of", "ingest_timings"]);
    fake.forget_requests();
    let error = candidates(&kb(&fake), &args(&data, &run, 1), &shared())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("eval_candidates"), "{error}");
    assert!(fake.seen(Route::Candidates).is_empty());
}
