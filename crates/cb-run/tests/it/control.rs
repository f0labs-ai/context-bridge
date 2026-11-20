//! `cb-run control export` and `control import` against two stand-in indexes: the
//! product's (read only) and the experiment's.

use crate::fake::{Action, Fake, Route};
use crate::support::*;
use cb_run::commands::control::{ExportArgs, ImportArgs, export, import};
use cb_run::error::RunError;
use cb_types::{Query, Record};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::atomic::Ordering;

const SOURCE: &str = "product_kb";
const EXPERIMENT: &str = "cb_exp";
const PRODUCT_TENANT: &str = "prod-tenant";
const WINDOW_START: i64 = 1_780_272_000;

fn product_points(fake: &Fake) {
    let point = |id: &str, tenant: &str, text: &str, parent: &str, dense: [f32; 4]| {
        fake.add_point(
            SOURCE,
            id,
            json!({"tenant_id": tenant, "collection": "website", "text": text, "parent_id": parent,
                   "heading": "Getting started", "kind": "step", "seq": 0, "source_path": "https://docs.example.com/start",
                   "doc_status": "UNKNOWN", "content_hash": format!("hash-{id}")}),
            json!({"dense": dense, "sparse": {"indices": [3, 8], "values": [0.5, 1.25]}}),
        );
    };
    point(
        "p-a",
        PRODUCT_TENANT,
        "Open Settings, then Workspace.",
        "sec-1",
        [0.1, 0.2, 0.3, 0.4],
    );
    point(
        "p-b",
        PRODUCT_TENANT,
        "Choose a plan for the workspace.",
        "sec-1",
        [0.5, 0.6, 0.7, 0.8],
    );
    point(
        "p-c",
        PRODUCT_TENANT,
        "Pay by card or by invoice.",
        "sec-2",
        [0.9, 1.0, 1.1, 1.2],
    );
    point(
        "p-x",
        "someone-else",
        "Not this tenant's.",
        "sec-9",
        [9.0, 9.0, 9.0, 9.0],
    );
}

fn held_out(path: &Path, questions: Value) {
    std::fs::write(
        path,
        json!({"site": "https://docs.example.com", "tenant": PRODUCT_TENANT, "questions": questions})
            .to_string(),
    )
    .unwrap();
}

fn export_args(questions: &Path, out: &Path, run: &Path) -> ExportArgs {
    ExportArgs {
        from_collection: SOURCE.into(),
        from_tenant: PRODUCT_TENANT.into(),
        questions: questions.to_path_buf(),
        window_start: WINDOW_START,
        out: out.to_path_buf(),
        seed: 42,
        leave_out_label: vec![],
        run: Some(run.to_path_buf()),
    }
}

fn import_args(data: &Path, collection: &str, run: &Path) -> ImportArgs {
    ImportArgs {
        data: data.to_path_buf(),
        tenant: "t-exp".into(),
        collection: collection.into(),
        label: "cb".into(),
        run: Some(run.to_path_buf()),
    }
}

/// A control data set exported from the product's stand-in index.
async fn exported(source: &Fake, dir: &Scratch) -> std::path::PathBuf {
    product_points(source);
    let questions = dir.join("helpq.json");
    held_out(
        &questions,
        json!([
            {"qid": "1", "style": "short", "question": "How do I open settings?", "gold_id": "p-a", "gold_parent": "sec-1"},
            {"qid": "2", "style": "long", "question": "Can I pay with a card?", "gold_id": "p-c", "gold_parent": "sec-2"},
        ]),
    );
    let out = dir.join("helpcentre");
    let report = export(
        &index(source),
        &export_args(&questions, &out, &dir.join("run-export")),
        &shared(),
    )
    .await
    .unwrap();
    assert!(
        report.is_complete(),
        "{:?}",
        failure_summary(&dir.join("run-export"))
    );
    out
}

#[tokio::test]
async fn export_reads_one_tenant_only_and_writes_a_whole_data_set() {
    let source = Fake::start().await;
    source.state.page_cap.store(2, Ordering::Relaxed);
    let dir = Scratch::new("control-export");
    let out = exported(&source, &dir).await;

    let records: Vec<Record> = lines(&out.join("records.jsonl"))
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
    let ids: Vec<&str> = records.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, ["p-a", "p-b", "p-c"]);
    for record in &records {
        let age = WINDOW_START - record.written_at;
        assert!((30 * 86_400..=700 * 86_400).contains(&age), "{record:?}");
        assert_eq!(record.tier, cb_types::Tier::Standing);
        assert_eq!(record.title.as_deref(), Some("Getting started"));
    }
    assert_eq!(records[0].text, "Open Settings, then Workspace.");
    let queries: Vec<Query> = lines(&out.join("queries.jsonl"))
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect();
    assert_eq!(queries.len(), 2);
    assert_eq!(queries[0].id, "hq-1");
    let gold: Vec<(&str, u8)> = queries[0]
        .gold
        .iter()
        .map(|g| (g.record.as_str(), g.grade))
        .collect();
    assert_eq!(
        gold,
        [("p-a", 3), ("p-b", 1)],
        "its passage, and the passage of the same section"
    );
    assert_eq!(queries[1].gold.len(), 1);
    let points = lines(&out.join("points.jsonl"));
    assert_eq!(points.len(), 3);
    assert_eq!(points[0]["id"], "p-a");
    assert_eq!(points[0]["dense"], json!([0.1, 0.2, 0.3, 0.4]));
    assert_eq!(
        points[0]["sparse"],
        json!({"indices": [3, 8], "values": [0.5, 1.25]})
    );
    assert_eq!(
        points[0]["payload"]["content_hash"], "hash-p-a",
        "the product's payload is kept whole"
    );
    let origin: Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("export.json")).unwrap()).unwrap();
    assert_eq!(
        (origin["collection"].as_str(), origin["tenant"].as_str()),
        (Some(SOURCE), Some(PRODUCT_TENANT))
    );
    let manifest: cb_types::Manifest =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    for (file, hash) in &manifest.sha256 {
        assert_eq!(
            hash,
            &cb_run::atomic::sha256_of_file(&out.join(file)).unwrap(),
            "{file}"
        );
    }
    assert_eq!(manifest.counts.get("points"), Some(&3));
    // Read only: counts and scrolls of the one tenant, nothing else.
    for seen in source.all_seen() {
        assert!(
            matches!(seen.route, Route::Count | Route::Scroll),
            "{:?}",
            seen.route
        );
        assert_eq!(seen.body["filter"]["must"][0]["match"]["value"], PRODUCT_TENANT);
    }
}

#[tokio::test]
async fn an_export_that_cannot_read_the_index_whole_writes_no_data_set() {
    let source = Fake::start().await;
    product_points(&source);
    source.always(Route::Scroll, "", Action::Status(503));
    let dir = Scratch::new("control-export-failure");
    let questions = dir.join("helpq.json");
    held_out(
        &questions,
        json!([{"qid": "1", "question": "How do I open settings?", "gold_id": "p-a"}]),
    );
    let out = dir.join("helpcentre");
    let run = dir.join("run-export");
    let report = export(&index(&source), &export_args(&questions, &out, &run), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![(String::new(), "http_status".to_string(), 3)]
    );
    assert!(!out.join("records.jsonl").exists() && !out.join("manifest.json").exists());
}

#[tokio::test]
async fn a_held_out_question_whose_passage_is_gone_is_named_not_dropped() {
    let source = Fake::start().await;
    product_points(&source);
    let dir = Scratch::new("control-export-gone");
    let questions = dir.join("helpq.json");
    held_out(
        &questions,
        json!([
            {"qid": "1", "question": "How do I open settings?", "gold_id": "p-a"},
            {"qid": "3", "question": "Where did it go?", "gold_id": "p-gone"},
        ]),
    );
    let (out, run) = (dir.join("helpcentre"), dir.join("run-export"));
    let report = export(&index(&source), &export_args(&questions, &out, &run), &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("hq-3".to_string(), "input".to_string(), 1)]
    );
    assert_eq!(lines(&out.join("queries.jsonl")).len(), 1);
}

#[tokio::test]
async fn import_writes_into_the_experiments_index_only_with_the_moment_each_passage_was_written() {
    let source = Fake::start().await;
    let experiment = Fake::start().await;
    experiment.create_collection(EXPERIMENT);
    let dir = Scratch::new("control-import");
    let data = exported(&source, &dir).await;
    source.forget_requests();
    let run = dir.join("run-import");

    let report = import(
        &index(&experiment),
        &import_args(&data, EXPERIMENT, &run),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let written_at: Vec<(String, i64)> = lines(&data.join("records.jsonl"))
        .iter()
        .map(|r| {
            (
                r["id"].as_str().unwrap().to_string(),
                r["written_at"].as_i64().unwrap(),
            )
        })
        .collect();
    let points = experiment.points(EXPERIMENT);
    assert_eq!(points.len(), 3);
    for point in &points {
        let id = point.id.as_str().unwrap();
        let expected = written_at.iter().find(|(r, _)| r == id).unwrap().1;
        assert_eq!(point.payload["tenant_id"], "t-exp");
        assert_eq!(point.payload["collection"], "cb");
        assert_eq!(point.payload["written_at"], expected);
        assert_eq!(point.payload["kind"], "step", "the product's enrichment is kept");
        assert_eq!(point.payload["source_path"], "https://docs.example.com/start");
        assert_eq!(
            point.vector["sparse"],
            json!({"indices": [3, 8], "values": [0.5, 1.25]})
        );
    }
    assert!(
        source.all_seen().is_empty(),
        "the product's index is never touched by the import"
    );
    assert_eq!(lines(&run.join("imported.jsonl")).len(), 3);
}

#[tokio::test]
async fn import_refuses_the_index_the_data_set_came_from_by_address_or_by_collection() {
    let source = Fake::start().await;
    let experiment = Fake::start().await;
    experiment.create_collection(SOURCE);
    source.create_collection(EXPERIMENT);
    let dir = Scratch::new("control-import-refused");
    let data = exported(&source, &dir).await;
    source.forget_requests();

    // The same server, under another collection name.
    let error = import(
        &index(&source),
        &import_args(&data, EXPERIMENT, &dir.join("r1")),
        &shared(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, RunError::Refused { .. }), "{error}");
    // Another server, under the source's collection name.
    let error = import(
        &index(&experiment),
        &import_args(&data, SOURCE, &dir.join("r2")),
        &shared(),
    )
    .await
    .unwrap_err();
    assert!(matches!(error, RunError::Refused { .. }), "{error}");
    assert!(source.seen(Route::Upsert).is_empty() && experiment.seen(Route::Upsert).is_empty());
}

#[tokio::test]
async fn an_import_that_failed_writes_the_rest_on_the_next_run_and_a_changed_file_is_refused() {
    let source = Fake::start().await;
    let experiment = Fake::start().await;
    experiment.create_collection(EXPERIMENT);
    let dir = Scratch::new("control-import-resume");
    let data = exported(&source, &dir).await;
    let run = dir.join("run-import");
    experiment.rule(Route::Upsert, EXPERIMENT, 3, Action::Status(503));

    let report = import(
        &index(&experiment),
        &import_args(&data, EXPERIMENT, &run),
        &shared(),
    )
    .await
    .unwrap();
    assert!(!report.is_complete());
    assert_eq!(failures(&run).len(), 3, "each point of the failed batch is named");
    assert!(experiment.points(EXPERIMENT).is_empty());

    experiment.clear_rules();
    let report = import(
        &index(&experiment),
        &import_args(&data, EXPERIMENT, &run),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(experiment.points(EXPERIMENT).len(), 3);

    // A points file that is not the one the manifest describes is not imported.
    let points = data.join("points.jsonl");
    let mut text = std::fs::read_to_string(&points).unwrap();
    text.push('\n');
    std::fs::write(&points, text).unwrap();
    let error = import(
        &index(&experiment),
        &import_args(&data, EXPERIMENT, &dir.join("r3")),
        &shared(),
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().contains("not the one the manifest describes"),
        "{error}"
    );
}
