//! `cb-run vectors` against the stand-in's search index.

use crate::fake::{Action, Fake, Route};
use crate::support::*;
use cb_run::commands::vectors::{VectorsArgs, vectors};
use cb_run::vector_file::VectorTable;
use serde_json::json;
use std::path::Path;
use std::sync::atomic::Ordering;

const INDEX: &str = "kb_index";

fn point(fake: &Fake, id: &str, tenant: &str, path: &str, seq: i64, dense: Option<[f32; 4]>) {
    let vector = match dense {
        Some(dense) => json!({"dense": dense, "sparse": {"indices": [1, 2], "values": [0.5, 0.5]}}),
        None => json!({"sparse": {"indices": [1], "values": [1.0]}}),
    };
    fake.add_point(
        INDEX,
        id,
        json!({"tenant_id": tenant, "collection": "cb", "source_path": path, "seq": seq, "text": "never read"}),
        vector,
    );
}

fn tenant_points(fake: &Fake) {
    point(
        fake,
        "p-01",
        TENANT,
        "cb/inc-1.s1.md",
        1,
        Some([1.0, 1.0, 1.0, 1.0]),
    );
    point(
        fake,
        "p-02",
        TENANT,
        "cb/inc-1.s1.md",
        0,
        Some([0.0, 0.0, 0.0, 1.0]),
    );
    point(
        fake,
        "p-03",
        TENANT,
        "cb/doc-1.1.md",
        0,
        Some([2.0, 2.0, 2.0, 2.0]),
    );
    point(
        fake,
        "p-04",
        TENANT,
        "https://docs.example.com/a",
        0,
        Some([3.0, 3.0, 3.0, 3.0]),
    );
    point(
        fake,
        "p-05",
        "another-tenant",
        "cb/inc-9.s1.md",
        0,
        Some([9.0, 9.0, 9.0, 9.0]),
    );
}

fn args(run: &Path) -> VectorsArgs {
    VectorsArgs {
        collection: INDEX.into(),
        tenant: TENANT.into(),
        run: run.to_path_buf(),
    }
}

#[tokio::test]
async fn one_vector_per_record_its_first_passage_read_page_by_page() {
    let fake = Fake::start().await;
    tenant_points(&fake);
    fake.state.page_cap.store(2, Ordering::Relaxed);
    let dir = Scratch::new("vectors-happy");
    let run = dir.join("run");

    let report = vectors(&index(&fake), &args(&run), &shared()).await.unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let index_file: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("vectors.json")).unwrap()).unwrap();
    assert_eq!(
        index_file,
        json!({"dim": 4, "ids": ["doc-1.1", "inc-1.s1", "p-04"]})
    );
    let table = VectorTable::read(&run, "vectors").unwrap().unwrap();
    assert_eq!(table.row(0).unwrap(), &[2.0, 2.0, 2.0, 2.0]);
    assert_eq!(
        table.row(1).unwrap(),
        &[0.0, 0.0, 0.0, 1.0],
        "the record's first passage"
    );
    assert_eq!(
        table.row(2).unwrap(),
        &[3.0, 3.0, 3.0, 3.0],
        "another document keeps its point id"
    );
    let bytes = std::fs::read(run.join("vectors.f32")).unwrap();
    assert_eq!(bytes.len(), 3 * 4 * 4);
    assert_eq!(&bytes[16..20], &0.0f32.to_le_bytes());
    assert_eq!(&bytes[28..32], &1.0f32.to_le_bytes());

    // Read only: one count, scrolls of the tenant's points, no passage text asked for.
    let scrolls = fake.seen(Route::Scroll);
    assert_eq!(scrolls.len(), 2, "four points in pages of two");
    for scroll in &scrolls {
        assert_eq!(
            scroll.body["filter"],
            json!({"must": [{"key": "tenant_id", "match": {"value": TENANT}}]})
        );
        assert_eq!(scroll.body["with_payload"], json!(["source_path", "seq"]));
        assert_eq!(scroll.body["with_vector"], json!(["dense"]));
    }
    assert_eq!(fake.seen(Route::Count).len(), 1);
    assert!(fake.seen(Route::Upsert).is_empty());
    let entry = last_entry(&run);
    assert_eq!(
        (
            entry["counts"]["passages_read"].as_u64(),
            entry["counts"]["written"].as_u64()
        ),
        (Some(4), Some(3))
    );
    assert_eq!(entry["index"]["collection"], INDEX);
}

#[tokio::test]
async fn an_index_that_cannot_be_read_whole_writes_nothing_and_a_later_run_writes_all() {
    let fake = Fake::start().await;
    tenant_points(&fake);
    fake.state.page_cap.store(2, Ordering::Relaxed);
    let dir = Scratch::new("vectors-failure");
    let run = dir.join("run");
    // The second page fails three times.
    fake.rule(Route::Scroll, "\"offset\"", 3, Action::Status(503));

    let report = vectors(&index(&fake), &args(&run), &shared()).await.unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![(String::new(), "http_status".to_string(), 3)]
    );
    assert!(!run.join("vectors.json").exists() && !run.join("vectors.f32").exists());

    fake.clear_rules();
    // A malformed page on the way is tried again.
    fake.rule(Route::Scroll, "\"offset\"", 1, Action::Garbage);
    let report = vectors(&index(&fake), &args(&run), &shared()).await.unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(VectorTable::read(&run, "vectors").unwrap().unwrap().len(), 3);
    assert!(failures(&run).is_empty());
}

#[tokio::test]
async fn a_read_that_returns_fewer_points_than_the_index_counts_is_a_failure() {
    let fake = Fake::start().await;
    tenant_points(&fake);
    let dir = Scratch::new("vectors-count");
    let run = dir.join("run");
    fake.rule(
        Route::Count,
        "",
        1,
        Action::Body(json!({"result": {"count": 99}, "status": "ok", "time": 0.0}).to_string()),
    );
    let report = vectors(&index(&fake), &args(&run), &shared()).await.unwrap();
    assert!(!report.is_complete());
    let failure = &failures(&run)[0];
    assert_eq!(failure["kind"], "contract");
    assert!(
        failure["message"].as_str().unwrap().contains("counts 99"),
        "{failure}"
    );
    assert!(!run.join("vectors.json").exists());
}

#[tokio::test]
async fn a_passage_without_a_vector_is_named_and_the_others_are_written() {
    let fake = Fake::start().await;
    tenant_points(&fake);
    point(&fake, "p-06", TENANT, "cb/inc-1.s2.md", 0, None);
    let dir = Scratch::new("vectors-no-vector");
    let run = dir.join("run");
    let report = vectors(&index(&fake), &args(&run), &shared()).await.unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("p-06".to_string(), "contract".to_string(), 1)]
    );
    let table = VectorTable::read(&run, "vectors").unwrap().unwrap();
    assert_eq!(table.ids(), ["doc-1.1", "inc-1.s1", "p-04"]);
}
