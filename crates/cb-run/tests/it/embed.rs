//! `cb-run embed` against the stand-in's `/v1/embed`.

use crate::fake::{Action, Fake, Route, vector_of};
use crate::support::*;
use cb_run::commands::embed::{EmbedArgs, embed};
use cb_run::vector_file::VectorTable;
use serde_json::json;
use std::path::Path;

fn texts(file: &Path, count: usize, long_one: Option<usize>) -> Vec<(String, String)> {
    let all: Vec<(String, String)> = (0..count)
        .map(|i| {
            let text = match long_one {
                Some(n) if n == i => "x".repeat(1_001),
                _ => format!("problem and solution number {i:02}"),
            };
            (format!("x-{i:02}"), text)
        })
        .collect();
    let lines: Vec<serde_json::Value> = all
        .iter()
        .map(|(id, text)| json!({"id": id, "text": text}))
        .collect();
    write_lines(file, &lines);
    all
}

fn args(file: &Path, run: &Path) -> EmbedArgs {
    EmbedArgs {
        texts: file.to_path_buf(),
        run: run.to_path_buf(),
        name: "extractions".into(),
    }
}

#[tokio::test]
async fn texts_are_embedded_in_batches_of_64_and_written_in_their_order() {
    let fake = Fake::start().await;
    let dir = Scratch::new("embed-happy");
    let (file, run) = (dir.join("texts.jsonl"), dir.join("run"));
    let all = texts(&file, 70, None);

    let report = embed(&kb(&fake), &args(&file, &run), &shared()).await.unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let table = VectorTable::read(&run, "extractions").unwrap().unwrap();
    assert_eq!(table.dim(), 4);
    let ids: Vec<&str> = all.iter().map(|(id, _)| id.as_str()).collect();
    assert_eq!(table.ids(), ids.as_slice());
    for (i, (_, text)) in all.iter().enumerate() {
        let expected = vector_of(text);
        assert!(
            table
                .row(i)
                .unwrap()
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "row {i}"
        );
    }
    let batches: Vec<usize> = fake
        .seen(Route::Embed)
        .iter()
        .map(|s| s.body["texts"].as_array().unwrap().len())
        .collect();
    assert_eq!(batches, [64, 6]);
    assert!(fake.seen(Route::Embed).iter().all(|s| s.body["kind"] == "query"));
    let entry = last_entry(&run);
    assert_eq!(entry["knowledge_service"]["embedding_model"], "fake-e5#query");
    assert_eq!(entry["counts"]["rows"], 70);
}

#[tokio::test]
async fn a_text_the_service_refuses_costs_only_its_own_vector() {
    let fake = Fake::start().await;
    let dir = Scratch::new("embed-refused");
    let (file, run) = (dir.join("texts.jsonl"), dir.join("run"));
    texts(&file, 10, Some(3));

    let report = embed(&kb(&fake), &args(&file, &run), &shared()).await.unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("x-03".to_string(), "http_status".to_string(), 1)]
    );
    let table = VectorTable::read(&run, "extractions").unwrap().unwrap();
    assert_eq!(table.len(), 9);
    assert!(!table.contains("x-03"));
    // The batch of ten was refused, then each text went alone.
    assert_eq!(fake.seen(Route::Embed).len(), 11);
}

#[tokio::test]
async fn a_batch_that_failed_is_embedded_on_the_next_run_and_nothing_twice() {
    let fake = Fake::start().await;
    let dir = Scratch::new("embed-resume");
    let (file, run) = (dir.join("texts.jsonl"), dir.join("run"));
    texts(&file, 70, None);
    fake.rule(Route::Embed, "number 65", 3, Action::Status(503));

    let report = embed(&kb(&fake), &args(&file, &run), &shared()).await.unwrap();
    assert!(!report.is_complete());
    assert_eq!(failures(&run).len(), 6, "every text of the failed batch is named");
    assert_eq!(VectorTable::read(&run, "extractions").unwrap().unwrap().len(), 64);

    fake.clear_rules();
    fake.forget_requests();
    let report = embed(&kb(&fake), &args(&file, &run), &shared()).await.unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let batches: Vec<usize> = fake
        .seen(Route::Embed)
        .iter()
        .map(|s| s.body["texts"].as_array().unwrap().len())
        .collect();
    assert_eq!(batches, [6]);
    let table = VectorTable::read(&run, "extractions").unwrap().unwrap();
    assert_eq!(table.len(), 70);
    assert_eq!(table.ids()[64], "x-64");
    assert!(failures(&run).is_empty());
}
