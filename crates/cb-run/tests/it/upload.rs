//! `cb-run upload` against the stand-in.

use crate::fake::{Action, Fake, Route};
use crate::support::*;
use cb_run::commands::upload::{UPLOADS_FILE, UploadArgs, upload};
use cb_run::error::RunError;
use cb_types::Record;
use serde_json::json;
use std::path::Path;
use std::time::Duration;

fn args(data: &Path, run: &Path, tenant: &str, concurrency: usize) -> UploadArgs {
    UploadArgs {
        data: data.to_path_buf(),
        tenant: tenant.into(),
        collection: LABEL.into(),
        run: run.to_path_buf(),
        concurrency,
        only: None,
    }
}

fn ids_of(path: &Path) -> Vec<String> {
    lines(path)
        .iter()
        .map(|l| l["record"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn every_record_becomes_one_document_with_its_times_and_the_services_answer_is_kept() {
    let fake = Fake::start().await;
    let dir = Scratch::new("upload-happy");
    let (data, run) = (dir.join("data"), dir.join("run"));
    let mut first = changing(
        "inc-0001.s1",
        "inc-0001",
        "Northgate outage",
        "Home internet is down in Northgate.",
        1_780_272_000,
    );
    first.valid_until = Some(1_780_279_200);
    let second = changing(
        "inc-0001.s2",
        "inc-0001",
        "Northgate outage",
        "The cause is a cut fibre.\n\nA crew is on site.",
        1_780_273_000,
    );
    let standing = record(
        "doc-012.3",
        "Restart the router to apply a new plan.",
        1_700_000_000,
    );
    write_data_set(&data, &[first, second, standing], &[]);

    let report = upload(&kb(&fake), &args(&data, &run, TENANT, 2), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));

    let written = lines(&run.join(UPLOADS_FILE));
    assert_eq!(
        ids_of(&run.join(UPLOADS_FILE)),
        ["inc-0001.s1", "inc-0001.s2", "doc-012.3"]
    );
    assert_eq!(written[0]["passages_indexed"], 1);
    assert_eq!(written[1]["passages_indexed"], 2);
    assert_eq!(written[0]["timings_ms"]["searchable"], 21.3);
    assert_eq!(written[0]["timings_ms"]["total"], 25.3);
    assert!(written[0]["wall_ms"].as_f64().unwrap() > 0.0);
    assert_eq!(written[0]["tries"], 1);

    // What the service received: its own multipart field names, the record's own path,
    // its times, its title, its id as the document's reference; and the token.
    let ingests = fake.seen(Route::Ingest);
    assert_eq!(ingests.len(), 3);
    let sent = ingests
        .iter()
        .find(|s| s.body["path"] == "cb/inc-0001.s1.md")
        .unwrap();
    assert_eq!(sent.body["collection"], "cb");
    assert_eq!(sent.body["fields"], json!(["collection", "attributes", "files"]));
    assert_eq!(sent.body["text"], "Home internet is down in Northgate.");
    assert_eq!(
        sent.body["attributes"],
        json!({"written_at": 1_780_272_000, "valid_until": 1_780_279_200, "title": "Northgate outage", "document_ref": "inc-0001.s1"})
    );
    assert_eq!(
        sent.authorization.as_deref(),
        Some(format!("Bearer {KB_TOKEN}").as_str())
    );
    let doc = ingests
        .iter()
        .find(|s| s.body["path"] == "cb/doc-012.3.md")
        .unwrap();
    assert_eq!(
        doc.body["attributes"],
        json!({"written_at": 1_700_000_000, "document_ref": "doc-012.3"})
    );

    // Every uploaded record was looked for in the index afterwards.
    assert_eq!(fake.seen(Route::PassageCount).len(), 3);

    let entry = last_entry(&run);
    assert_eq!(
        (entry["command"].as_str(), entry["ending"].as_str()),
        (Some("upload"), Some("complete"))
    );
    assert_eq!(entry["counts"]["written"], 3);
    assert_eq!(entry["counts"]["checked_in_index"], 3);
    assert_eq!(entry["machine"]["label"], "test machine");
    let features = entry["knowledge_service"]["info"]["features"].as_array().unwrap();
    assert!(features.iter().any(|f| f == "ingest_timings"));
    assert_eq!(std::fs::read_to_string(run.join("failures.jsonl")).unwrap(), "");
    assert_no_secret_in(&run);
}

#[tokio::test]
async fn too_many_requests_and_server_failures_are_tried_again_a_refusal_is_not_and_a_resume_does_the_rest() {
    let fake = Fake::start().await;
    let dir = Scratch::new("upload-retry");
    let (data, run) = (dir.join("data"), dir.join("run"));
    let records: Vec<Record> = ["r-a", "r-b", "r-c", "r-d", "r-e"]
        .iter()
        .map(|id| record(id, &format!("The text of {id}."), 1_780_272_000))
        .collect();
    write_data_set(&data, &records, &[]);
    fake.rule(Route::Ingest, "cb/r-a.md", 1, Action::StatusRetryAfter(429, 0));
    fake.rule(Route::Ingest, "cb/r-b.md", 3, Action::Status(503));
    fake.rule(Route::Ingest, "cb/r-c.md", 1, Action::Status(400));
    fake.rule(Route::Ingest, "cb/r-d.md", 1, Action::Garbage);
    fake.rule(Route::Ingest, "cb/r-e.md", 1, Action::WrongShape);

    let report = upload(&kb(&fake), &args(&data, &run, TENANT, 3), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(report.remaining.own, 2);
    let mut failed = failure_summary(&run);
    failed.sort();
    assert_eq!(
        failed,
        vec![
            ("r-b".to_string(), "http_status".to_string(), 3),
            ("r-c".to_string(), "http_status".to_string(), 1),
        ]
    );
    let b = failures(&run).into_iter().find(|f| f["id"] == "r-b").unwrap();
    assert_eq!(b["status"], 503);
    assert!(b["raw"].as_str().unwrap().contains("the stand-in answers 503"));
    assert!(b["url"].as_str().unwrap().ends_with("/v1/tenants/t-proof/ingest"));
    assert_eq!(
        ["r-a", "r-b", "r-c", "r-d", "r-e"].map(|id| fake.count(Route::Ingest, &format!("cb/{id}.md"))),
        [2, 3, 1, 2, 2]
    );
    assert_eq!(ids_of(&run.join(UPLOADS_FILE)), ["r-a", "r-d", "r-e"]);
    assert_eq!(lines(&run.join(UPLOADS_FILE))[0]["tries"], 2);
    assert_eq!(last_entry(&run)["ending"], "incomplete");

    // The service is well again: only what is missing is uploaded.
    fake.clear_rules();
    fake.forget_requests();
    let report = upload(&kb(&fake), &args(&data, &run, TENANT, 3), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let again: Vec<String> = fake
        .seen(Route::Ingest)
        .iter()
        .map(|s| s.body["path"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(again.len(), 2);
    assert!(again.contains(&"cb/r-b.md".to_string()) && again.contains(&"cb/r-c.md".to_string()));
    let mut all = ids_of(&run.join(UPLOADS_FILE));
    all.sort();
    assert_eq!(all, ["r-a", "r-b", "r-c", "r-d", "r-e"]);
    assert!(failures(&run).is_empty());
    let entries = manifest_entries(&run);
    assert_eq!(entries.len(), 2);
    assert_eq!(
        (entries[0]["ending"].as_str(), entries[1]["ending"].as_str()),
        (Some("incomplete"), Some("complete"))
    );
}

#[tokio::test]
async fn an_answer_that_comes_too_late_is_a_timeout_and_is_tried_again() {
    let fake = Fake::start().await;
    let dir = Scratch::new("upload-timeout");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &[record("r-slow", "Slow to answer.", 1_780_272_000)], &[]);
    fake.rule(
        Route::Ingest,
        "cb/r-slow.md",
        1,
        Action::Slow(Duration::from_millis(1_500)),
    );
    let service = kb_with_timeout(&fake, Duration::from_millis(400));
    let report = upload(&service, &args(&data, &run, TENANT, 1), &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(lines(&run.join(UPLOADS_FILE))[0]["tries"], 2);
    // The wall time is the successful try's, not the time spent waiting on the first.
    assert!(lines(&run.join(UPLOADS_FILE))[0]["wall_ms"].as_f64().unwrap() < 400.0);
}

#[tokio::test]
async fn passages_another_record_took_over_make_a_failure_and_an_identical_text_is_not_uploaded() {
    let fake = Fake::start().await;
    let dir = Scratch::new("upload-takeover");
    let (data, run) = (dir.join("data"), dir.join("run"));
    let shared_sentence = "We are continuing to investigate.";
    let records = vec![
        record(
            "r-1",
            &format!("Home internet is down.\n\n{shared_sentence}"),
            1_780_272_000,
        ),
        record(
            "r-2",
            &format!("Mobile data is slow.\n\n{shared_sentence}"),
            1_780_272_100,
        ),
        record(
            "r-3",
            &format!("Home internet is down.\n\n{shared_sentence}"),
            1_780_272_200,
        ),
        record("r-4", "TV is fine.", 1_780_272_300),
    ];
    write_data_set(&data, &records, &[]);

    let report = upload(&kb(&fake), &args(&data, &run, TENANT, 1), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    let mut failed = failure_summary(&run);
    failed.sort();
    assert_eq!(
        failed,
        vec![
            ("r-1".to_string(), "not_whole_in_index".to_string(), 1),
            ("r-3".to_string(), "duplicate_text".to_string(), 0),
        ]
    );
    // The record with the very same text as an earlier one was never sent.
    assert_eq!(fake.count(Route::Ingest, "cb/r-3.md"), 0);
    let r1 = failures(&run).into_iter().find(|f| f["id"] == "r-1").unwrap();
    let message = r1["message"].as_str().unwrap();
    assert!(
        message.contains("indexed 2") && message.contains("holds 1"),
        "{message}"
    );
    assert_eq!(ids_of(&run.join(UPLOADS_FILE)), ["r-1", "r-2", "r-4"]);

    // A later run cannot pass for complete while the index still lacks them.
    let report = upload(&kb(&fake), &args(&data, &run, TENANT, 1), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(report.remaining.own, 2);
    assert_eq!(
        fake.count(Route::Ingest, "cb/r-1.md"),
        1,
        "an uploaded record is not uploaded again"
    );
}

#[tokio::test]
async fn a_service_without_the_experiment_features_is_refused_before_any_upload() {
    let fake = Fake::start().await;
    fake.set_features(&["document_titles", "removal_proof"]);
    let dir = Scratch::new("upload-features");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &[record("r-1", "Text.", 1_780_272_000)], &[]);
    let error = upload(&kb(&fake), &args(&data, &run, TENANT, 1), &shared())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("document_times"), "{error}");
    assert!(fake.seen(Route::Ingest).is_empty());
    assert_eq!(last_entry(&run)["ending"], "stopped");
}

#[tokio::test]
async fn a_run_directory_belongs_to_one_tenant() {
    let fake = Fake::start().await;
    let dir = Scratch::new("upload-claim");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &[record("r-1", "Text.", 1_780_272_000)], &[]);
    assert!(
        upload(&kb(&fake), &args(&data, &run, "t-a", 1), &shared())
            .await
            .unwrap()
            .is_complete()
    );
    fake.forget_requests();
    let error = upload(&kb(&fake), &args(&data, &run, "t-b", 1), &shared())
        .await
        .unwrap_err();
    assert!(matches!(error, RunError::File { .. }), "{error}");
    assert!(
        error.to_string().contains("tenant = t-a") && error.to_string().contains("tenant = t-b"),
        "{error}"
    );
    assert!(
        fake.seen(Route::Ingest).is_empty(),
        "nothing was skipped as done for another tenant"
    );
}

#[tokio::test]
async fn a_line_cut_off_by_a_crash_is_removed_and_its_record_uploaded_again() {
    let fake = Fake::start().await;
    let dir = Scratch::new("upload-crash");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(
        &data,
        &[
            record("r-1", "One.", 1_780_272_000),
            record("r-2", "Two.", 1_780_272_000),
        ],
        &[],
    );
    upload(&kb(&fake), &args(&data, &run, TENANT, 1), &shared())
        .await
        .unwrap();
    // A crash in the middle of the second line.
    let path = run.join(UPLOADS_FILE);
    let text = std::fs::read_to_string(&path).unwrap();
    let cut = text.len() - 25;
    std::fs::write(&path, &text[..cut]).unwrap();
    fake.forget_requests();

    let report = upload(&kb(&fake), &args(&data, &run, TENANT, 1), &shared())
        .await
        .unwrap();
    assert!(report.is_complete());
    let sent: Vec<_> = fake
        .seen(Route::Ingest)
        .iter()
        .map(|s| s.body["path"].clone())
        .collect();
    assert_eq!(sent, [json!("cb/r-2.md")]);
    assert_eq!(ids_of(&path), ["r-1", "r-2"]);
    assert!(
        last_entry(&run)["counts"]["cut_off_bytes_removed"]
            .as_u64()
            .unwrap()
            > 0
    );
}
