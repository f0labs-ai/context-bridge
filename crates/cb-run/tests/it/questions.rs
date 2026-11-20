//! `cb-run questions` against the stand-in's language model.

use crate::fake::{Fake, Route};
use crate::support::*;
use cb_run::commands::questions::{QUESTIONS_FILE, QuestionsArgs, questions};
use cb_types::{Manifest, Query, QueryClass, Record};
use serde_json::Value;
use std::path::Path;
use std::time::Duration;

const T0: i64 = 1_780_272_000;

fn records() -> Vec<Record> {
    vec![
        changing(
            "gh-a.1",
            "gh-a",
            "Incident with Actions",
            "Actions runs are delayed.",
            T0,
        ),
        changing(
            "gh-a.2",
            "gh-a",
            "Incident with Actions",
            "Actions runs are back to normal.",
            T0 + 3_600,
        ),
        changing(
            "gh-b.1",
            "gh-b",
            "Billing portal slow",
            "The billing portal is slow.",
            T0 + 10_000,
        ),
        changing(
            "gh-b.2",
            "gh-b",
            "Billing portal slow",
            "The billing portal is fast again.",
            T0 + 20_000,
        ),
        Record {
            title: Some("About GitHub Actions".into()),
            ..record(
                "ghdoc-actions.1",
                "GitHub Actions runs your workflows.",
                T0 - 1_000 * 86_400,
            )
        },
        Record {
            title: Some("Paying a bill".into()),
            ..record(
                "ghdoc-billing.1",
                "Pay your bill on the billing page.",
                T0 - 1_000 * 86_400,
            )
        },
        Record {
            title: Some("Your profile".into()),
            ..record(
                "ghdoc-profile.1",
                "Change your profile picture in settings.",
                T0 - 1_000 * 86_400,
            )
        },
        Record {
            title: Some("Actions runners".into()),
            ..record(
                "ghdoc-actions.2",
                "Self-hosted runners run on your machines.",
                T0 - 1_000 * 86_400,
            )
        },
    ]
}

fn args(data: &Path, run: &Path, sample: usize, per_section: usize, seed: u64) -> QuestionsArgs {
    QuestionsArgs {
        data: data.to_path_buf(),
        model: "qwen3-8b-instruct".into(),
        per_section,
        sample,
        seed,
        run: Some(run.to_path_buf()),
        prompt: None,
        concurrency: 2,
        product_words: None,
        max_tokens: None,
        extra_body: None,
    }
}

/// The stand-in writes as many questions as the message asks for.
fn answers(fake: &Fake) {
    fake.chat_answers(|message| {
        let count = if message.contains("Write 2 question(s)") {
            2
        } else {
            1
        };
        let questions: Vec<String> = (1..=count).map(|n| format!("Question {n} about this?")).collect();
        serde_json::json!({ "questions": questions }).to_string()
    });
}

fn queries_of(data: &Path) -> Vec<Query> {
    lines(&data.join("queries.jsonl"))
        .into_iter()
        .map(|v| serde_json::from_value(v).unwrap())
        .collect()
}

fn manifest_of(data: &Path) -> Manifest {
    serde_json::from_str(&std::fs::read_to_string(data.join("manifest.json")).unwrap()).unwrap()
}

#[tokio::test]
async fn questions_join_the_data_set_inside_an_incident_and_the_manifest_follows() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("questions-happy");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(
        &data,
        &records(),
        &[query("q-existing", "Is Actions down?", T0 + 60, &["gh-a.1"])],
    );

    let report = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &run, 3, 2, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let all = queries_of(&data);
    assert_eq!(all.len(), 7);
    assert_eq!(
        all[0].id, "q-existing",
        "the data set's own questions stay first and untouched"
    );
    let by_id = |id: &str| records().into_iter().find(|r| r.id == id).unwrap();
    for question in &all[1..] {
        assert_eq!(question.gold.len(), 1);
        assert_eq!(question.gold[0].grade, 3);
        let gold = by_id(&question.gold[0].record);
        assert!(gold.id.starts_with("ghdoc-"));
        assert!(question.asked_at >= gold.written_at);
        let inside_a = (T0..=T0 + 3_600).contains(&question.asked_at);
        let inside_b = (T0 + 10_000..=T0 + 20_000).contains(&question.asked_at);
        assert!(inside_a || inside_b, "{question:?}");
        assert!(matches!(
            question.class,
            QueryClass::Ordinary | QueryClass::OrdinaryNear
        ));
        assert!(question.id.starts_with("oq-") && cb_types::ids::check(&question.id).is_ok());
    }
    // The manifest says how many questions there are, hashes the file as it is now and
    // says where the new ones came from.
    let manifest = manifest_of(&data);
    assert_eq!(manifest.counts.get("queries"), Some(&7));
    assert_eq!(
        manifest.sha256.get("queries.jsonl"),
        Some(&cb_run::atomic::sha256_of_file(&data.join("queries.jsonl")).unwrap())
    );
    assert!(
        manifest
            .sources
            .iter()
            .any(|s| s.contains("seed 42") && s.contains("questions.v1")),
        "{:?}",
        manifest.sources
    );
    // Each record's passage and title went to the model, with the count asked for.
    let sent: Vec<String> = fake.seen(Route::Chat).iter().map(|c| c.key.clone()).collect();
    assert_eq!(sent.len(), 3);
    assert!(
        sent.iter()
            .all(|m| m.contains("Write 2 question(s)") && m.contains("Passage title:"))
    );
    assert_eq!(lines(&run.join(QUESTIONS_FILE)).len(), 6);

    // The same run again asks nothing and adds nothing.
    fake.forget_requests();
    let report = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &run, 3, 2, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete());
    assert!(fake.seen(Route::Chat).is_empty());
    assert_eq!(queries_of(&data).len(), 7);

    // A larger sample adds the new record's questions only.
    let report = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &run, 4, 2, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete());
    assert_eq!(fake.seen(Route::Chat).len(), 1);
    assert_eq!(queries_of(&data).len(), 9);
    assert_eq!(manifest_of(&data).counts.get("queries"), Some(&9));
}

#[tokio::test]
async fn an_answer_that_is_not_the_asked_json_is_a_failure_and_a_resume_adds_the_rest() {
    let fake = Fake::start().await;
    fake.chat_answers(|message| {
        if message.contains("Your profile") {
            "Sure! Here are some questions: what is a profile?".to_string()
        } else {
            serde_json::json!({"questions": ["What does this do?"]}).to_string()
        }
    });
    let dir = Scratch::new("questions-resume");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &records(), &[]);

    let report = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &run, 4, 1, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("ghdoc-profile.1".to_string(), "answer_not_asked".to_string(), 3)]
    );
    assert_eq!(
        failures(&run)[0]["raw"],
        "Sure! Here are some questions: what is a profile?"
    );
    assert_eq!(queries_of(&data).len(), 3);
    assert_eq!(manifest_of(&data).counts.get("queries"), Some(&3));

    answers(&fake);
    fake.forget_requests();
    let report = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &run, 4, 1, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(fake.seen(Route::Chat).len(), 1);
    let all = queries_of(&data);
    assert_eq!(all.len(), 4);
    assert!(all.iter().any(|q| q.gold[0].record == "ghdoc-profile.1"));
    assert!(failures(&run).is_empty());
}

#[tokio::test]
async fn a_sample_larger_than_the_data_set_is_a_failure_not_a_smaller_sample() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("questions-shortfall");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &records(), &[]);
    let report = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &run, 10, 1, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("sample".to_string(), "shortfall".to_string(), 0)]
    );
    assert_eq!(
        queries_of(&data).len(),
        4,
        "every standing record got its question all the same"
    );
}

#[tokio::test]
async fn questions_planned_with_another_seed_are_never_mixed_into_the_data_set() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("questions-seed");
    let data = dir.join("data");
    write_data_set(&data, &records(), &[]);
    let first = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &dir.join("run-a"), 3, 1, 42),
        &shared(),
    )
    .await
    .unwrap();
    assert!(first.is_complete());
    let before = std::fs::read(data.join("queries.jsonl")).unwrap();
    fake.forget_requests();

    let error = questions(
        &model(&fake, Duration::from_secs(5)),
        &args(&data, &dir.join("run-b"), 3, 1, 7),
        &shared(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("another seed"), "{error}");
    assert!(fake.seen(Route::Chat).is_empty());
    assert_eq!(
        std::fs::read(data.join("queries.jsonl")).unwrap(),
        before,
        "the data set is untouched"
    );
}

#[tokio::test]
async fn the_extra_body_reaches_the_endpoint_and_the_manifest() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("questions-extra");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(&data, &records(), &[]);
    let mut asked = args(&data, &run, 1, 1, 42);
    asked.extra_body = Some(r#"{"chat_template_kwargs": {"enable_thinking": false}}"#.parse().unwrap());
    let report = questions(&model(&fake, Duration::from_secs(5)), &asked, &shared())
        .await
        .unwrap();
    assert!(report.is_complete());
    let sent = &fake.seen(Route::Chat)[0].body;
    assert_eq!(
        sent["chat_template_kwargs"],
        serde_json::json!({"enable_thinking": false})
    );
    let entry: Value = last_entry(&run);
    assert_eq!(
        entry["model"]["extra_body"],
        serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}})
    );
    assert_eq!(
        entry["flags"]["extra_body"],
        serde_json::json!({"chat_template_kwargs": {"enable_thinking": false}})
    );
    assert_eq!(entry["counts"]["tokens_in"], 321);
}
