//! `cb-run extract` against the stand-in's language model.

use crate::fake::{Action, Fake, Route};
use crate::support::*;
use cb_run::commands::extract::{EXTRACTIONS_FILE, ExtractArgs, extract};
use cb_run::model::ExtraBody;
use cb_types::{Speaker, Transcript, Turn};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;

const T0: i64 = 1_780_272_000;

fn turn(who: Speaker, text: &str) -> Turn {
    Turn {
        who,
        text: text.into(),
        ends_at_s: 1.0,
    }
}

/// A call whose agent turn carries `mark`, so the stand-in can tell the calls apart.
fn call(id: &str, mark: &str, started_at: i64) -> Transcript {
    Transcript {
        id: id.into(),
        thread: "inc-0001".into(),
        started_at,
        agent: "agent-7".into(),
        persona: "brief".into(),
        turns: vec![
            turn(
                Speaker::Customer,
                "Hi, this is Ada Lovelace. My internet is down.",
            ),
            turn(
                Speaker::Agent,
                &format!("There is an outage in Northgate ({mark})."),
            ),
        ],
        states: vec![],
        asr_error_rate: 0.0,
    }
}

/// The repository's own extraction prompt.
fn prompt() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../prompts/extract.v1.md")
}

fn args(transcripts: &Path, run: &Path, concurrency: usize) -> ExtractArgs {
    ExtractArgs {
        transcripts: transcripts.to_path_buf(),
        model: "qwen3-8b-instruct".into(),
        prompt: prompt(),
        run: run.to_path_buf(),
        concurrency,
        limit: None,
        masked_with: None,
        max_tokens: None,
        extra_body: None,
    }
}

const GOOD: &str = r#"{"found": true, "problem": "Home internet is down in Northgate.", "solution": "A cut fibre; back by 02:00.", "t_start": "2026-06-01T00:05:00Z", "t_eta": "2026-06-01T02:00:00+01:00", "confidence": {"problem": 1.0, "solution": 0.5, "time": 0.5}}"#;

/// What the stand-in's model answers each call: good and fenced, nothing found, or good
/// with words around it.
fn answers(fake: &Fake) {
    fake.chat_answers(|message| {
        if message.contains("MARK-1") {
            format!("```json\n{GOOD}\n```")
        } else if message.contains("MARK-2") {
            "{\"found\": false}".to_string()
        } else if message.contains("MARK-3") {
            format!("Here is what the agent said:\n{GOOD}\nI hope this helps.")
        } else {
            "I am sorry, I cannot help with that.".to_string()
        }
    });
}

#[tokio::test]
async fn each_transcript_is_read_once_and_its_answer_kept_in_the_contracts_shape() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("extract-happy");
    let (file, run) = (dir.join("transcripts.jsonl"), dir.join("run"));
    write_lines(
        &file,
        &[
            call("c-1", "MARK-1", T0),
            call("c-2", "MARK-2", T0 + 60),
            call("c-3", "MARK-3", T0 + 120),
        ],
    );
    let mut asked = args(&file, &run, 2);
    asked.extra_body = Some(
        r#"{"reasoning":{"enabled":false},"chat_template_kwargs":{"enable_thinking":false}}"#
            .parse::<ExtraBody>()
            .unwrap(),
    );

    let report = extract(&model(&fake, Duration::from_secs(5)), &asked, &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let written = lines(&run.join(EXTRACTIONS_FILE));
    let ids: Vec<&str> = written
        .iter()
        .map(|l| l["transcript"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["c-1", "c-2", "c-3"]);
    let first = &written[0];
    assert_eq!(first["problem"], "Home internet is down in Northgate.");
    assert_eq!(first["t_start"], T0 + 300);
    assert_eq!(first["t_eta"], T0 + 3_600, "02:00 at UTC+1 is 01:00 UTC");
    assert_eq!(
        first["confidence"],
        json!({"problem": 1.0, "solution": 0.5, "time": 0.5})
    );
    assert_eq!(first["model"], "qwen3-8b-instruct");
    assert_eq!(first["prompt_version"], "extract.v1");
    assert_eq!(
        (first["tokens_in"].as_u64(), first["tokens_out"].as_u64()),
        (Some(321), Some(45))
    );
    assert!(first["latency_ms"].as_f64().unwrap() > 0.0);
    // Nothing found is written all the same, with empty fields and no confidence.
    assert_eq!(written[1]["problem"], "");
    assert_eq!(written[1]["solution"], "");
    assert_eq!(
        written[1]["confidence"],
        json!({"problem": 0.0, "solution": 0.0, "time": 0.0})
    );
    assert!(written[1].get("t_start").is_none());
    assert_eq!(
        written[2]["solution"], written[0]["solution"],
        "words around the object are put up with"
    );

    // What the endpoint was sent.
    let calls = fake.seen(Route::Chat);
    assert_eq!(calls.len(), 3);
    let sent = calls.iter().find(|c| c.key.contains("MARK-1")).unwrap();
    assert_eq!(sent.body["model"], "qwen3-8b-instruct");
    assert_eq!(sent.body["temperature"], 0);
    assert_eq!(sent.body["response_format"], json!({"type": "json_object"}));
    assert_eq!(
        sent.body["chat_template_kwargs"],
        json!({"enable_thinking": false})
    );
    assert_eq!(sent.body["reasoning"], json!({"enabled": false}));
    assert_eq!(
        sent.authorization.as_deref(),
        Some(format!("Bearer {MODEL_KEY}").as_str())
    );
    let message = sent.body["messages"][0]["content"].as_str().unwrap();
    assert!(
        message.contains("The call started at 2026-06-01T00:00:00Z (UTC)."),
        "{message}"
    );
    assert!(!message.contains("{{started_at}}"));
    assert!(message.ends_with(
        "Customer: Hi, this is Ada Lovelace. My internet is down.\nAgent: There is an outage in Northgate (MARK-1).\n"
    ));

    let entry = last_entry(&run);
    let model = &entry["model"];
    assert_eq!(model["name"], "qwen3-8b-instruct");
    assert_eq!(model["reported_names"], json!(["fake-model-2026"]));
    assert_eq!(model["prompt_version"], "extract.v1");
    assert_eq!(model["prompt_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(model["json_mode"], true);
    assert_eq!(
        model["extra_body"],
        json!({"chat_template_kwargs": {"enable_thinking": false}, "reasoning": {"enabled": false}})
    );
    assert_eq!(entry["counts"]["tokens_in"], 963);
    assert_eq!(entry["counts"]["nothing_found"], 1);
    assert_no_secret_in(&run);
}

#[tokio::test]
async fn an_answer_that_is_not_the_asked_json_is_a_failure_that_keeps_it_and_a_resume_reads_only_that_call() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("extract-malformed");
    let (file, run) = (dir.join("transcripts.jsonl"), dir.join("run"));
    write_lines(&file, &[call("c-1", "MARK-1", T0), call("c-4", "MARK-4", T0)]);

    let report = extract(
        &model(&fake, Duration::from_secs(5)),
        &args(&file, &run, 2),
        &shared(),
    )
    .await
    .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("c-4".to_string(), "answer_not_asked".to_string(), 3)]
    );
    let failure = &failures(&run)[0];
    assert_eq!(failure["raw"], "I am sorry, I cannot help with that.");
    assert!(failure["message"].as_str().unwrap().contains("no JSON object"));
    assert_eq!(fake.count(Route::Chat, "MARK-4"), 3);

    fake.chat_answers(|_| GOOD.to_string());
    fake.forget_requests();
    let report = extract(
        &model(&fake, Duration::from_secs(5)),
        &args(&file, &run, 2),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete());
    assert_eq!(
        fake.seen(Route::Chat).len(),
        1,
        "only the call that failed is read again"
    );
    assert_eq!(lines(&run.join(EXTRACTIONS_FILE)).len(), 2);
    assert!(failures(&run).is_empty());
}

#[tokio::test]
async fn an_endpoint_that_refuses_response_format_is_asked_without_it_from_then_on() {
    let fake = Fake::start().await;
    answers(&fake);
    fake.state.refuses_response_format.store(true, Ordering::Relaxed);
    let dir = Scratch::new("extract-format");
    let (file, run) = (dir.join("transcripts.jsonl"), dir.join("run"));
    write_lines(&file, &[call("c-1", "MARK-1", T0), call("c-3", "MARK-3", T0)]);

    let report = extract(
        &model(&fake, Duration::from_secs(5)),
        &args(&file, &run, 1),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let calls = fake.seen(Route::Chat);
    let with_format: Vec<bool> = calls
        .iter()
        .map(|c| c.body.get("response_format").is_some())
        .collect();
    assert_eq!(with_format, [true, false, false]);
    assert_eq!(last_entry(&run)["model"]["json_mode"], false);
}

#[tokio::test]
async fn the_latency_is_the_successful_calls_and_a_thinking_models_reasoning_is_never_read() {
    let fake = Fake::start().await;
    answers(&fake);
    // A first try that fails slowly, then a quick answer.
    fake.rule(
        Route::Chat,
        "MARK-1",
        1,
        Action::SlowStatus(Duration::from_millis(400), 503),
    );
    // Beside the answer, a reasoning that holds another object.
    *fake.state.reasoning.lock().unwrap() =
        Some(r#"{"found": true, "problem": "a draft", "solution": "a draft", "confidence": {"problem": 1, "solution": 1, "time": 1}}"#.into());
    fake.state.reports_usage.store(false, Ordering::Relaxed);
    let dir = Scratch::new("extract-latency");
    let (file, run) = (dir.join("transcripts.jsonl"), dir.join("run"));
    write_lines(&file, &[call("c-1", "MARK-1", T0), call("c-2", "MARK-2", T0)]);

    let report = extract(
        &model(&fake, Duration::from_secs(5)),
        &args(&file, &run, 1),
        &shared(),
    )
    .await
    .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    let written = lines(&run.join(EXTRACTIONS_FILE));
    assert_eq!(fake.count(Route::Chat, "MARK-1"), 2);
    assert!(
        written[0]["latency_ms"].as_f64().unwrap() < 300.0,
        "{}",
        written[0]["latency_ms"]
    );
    assert_eq!(written[0]["problem"], "Home internet is down in Northgate.");
    assert_eq!(
        written[1]["problem"], "",
        "the content said nothing was found; the reasoning is not read"
    );
    // No usage reported: zero, and the answers are counted.
    assert_eq!(
        (
            written[0]["tokens_in"].as_u64(),
            written[0]["tokens_out"].as_u64()
        ),
        (Some(0), Some(0))
    );
    assert_eq!(last_entry(&run)["counts"]["answers_without_token_counts"], 2);
}

#[tokio::test]
async fn masked_text_is_what_the_model_reads_and_a_call_missing_from_it_is_a_failure() {
    let fake = Fake::start().await;
    answers(&fake);
    let dir = Scratch::new("extract-masked");
    let (file, masked, run) = (
        dir.join("transcripts.jsonl"),
        dir.join("masked.jsonl"),
        dir.join("run"),
    );
    write_lines(&file, &[call("c-1", "MARK-1", T0), call("c-2", "MARK-2", T0)]);
    let mut hidden = call("c-1", "MARK-1", T0);
    hidden.turns[0].text = "Hi, this is [NAME]. My internet is down.".into();
    write_lines(&masked, &[hidden]);
    let mut asked = args(&file, &run, 1);
    asked.masked_with = Some(masked);

    let report = extract(&model(&fake, Duration::from_secs(5)), &asked, &shared())
        .await
        .unwrap();
    assert!(!report.is_complete());
    assert_eq!(
        failure_summary(&run),
        vec![("c-2".to_string(), "input".to_string(), 0)]
    );
    let calls = fake.seen(Route::Chat);
    assert_eq!(calls.len(), 1, "the unmasked text of c-2 is never sent");
    let message = calls[0].body["messages"][0]["content"].as_str().unwrap();
    assert!(
        message.contains("[NAME]") && !message.contains("Ada Lovelace"),
        "{message}"
    );
}

#[tokio::test]
async fn an_answer_that_comes_too_late_is_tried_again_and_a_limit_reads_the_first_calls_only() {
    let fake = Fake::start().await;
    answers(&fake);
    fake.rule(
        Route::Chat,
        "MARK-1",
        1,
        Action::Slow(Duration::from_millis(1_500)),
    );
    let dir = Scratch::new("extract-timeout");
    let (file, run) = (dir.join("transcripts.jsonl"), dir.join("run"));
    write_lines(
        &file,
        &[
            call("c-1", "MARK-1", T0),
            call("c-2", "MARK-2", T0),
            call("c-3", "MARK-3", T0),
        ],
    );
    let mut asked = args(&file, &run, 1);
    asked.limit = Some(2);
    let report = extract(&model(&fake, Duration::from_millis(400)), &asked, &shared())
        .await
        .unwrap();
    assert!(report.is_complete(), "{:?}", failure_summary(&run));
    assert_eq!(fake.count(Route::Chat, "MARK-1"), 2);
    assert_eq!(fake.count(Route::Chat, "MARK-3"), 0);
    assert_eq!(lines(&run.join(EXTRACTIONS_FILE)).len(), 2);
}
