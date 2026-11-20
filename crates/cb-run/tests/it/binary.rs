//! The `cb-run` binary itself: its exit status and what it prints.

use crate::fake::{Action, Fake, Route};
use crate::support::*;
use std::path::Path;
use std::process::Output;

const TEXT_ONE: &str = "Record text one, never logged.";
const TEXT_TWO: &str = "Record text two, never logged.";

/// Runs the binary with a clean environment: only the variables given (no proxy, no
/// token from the developer's shell).
async fn run_binary(env: Vec<(&'static str, String)>, args: Vec<String>) -> Output {
    tokio::task::spawn_blocking(move || {
        std::process::Command::new(env!("CARGO_BIN_EXE_cb-run"))
            .env_clear()
            .envs(env)
            .args(args)
            .output()
            .expect("the binary runs")
    })
    .await
    .expect("the binary's task")
}

fn upload_line(data: &Path, run: &Path) -> Vec<String> {
    [
        "upload",
        "--data",
        &data.display().to_string(),
        "--tenant",
        TENANT,
        "--collection",
        LABEL,
        "--run",
        &run.display().to_string(),
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_exit_status_says_whether_the_run_is_complete_and_nothing_secret_is_printed() {
    let fake = Fake::start().await;
    let dir = Scratch::new("binary");
    let (data, run) = (dir.join("data"), dir.join("run"));
    write_data_set(
        &data,
        &[
            record("r-1", TEXT_ONE, 1_780_272_000),
            record("r-2", TEXT_TWO, 1_780_272_000),
        ],
        &[],
    );
    let env = vec![
        ("CB_KB_URL", fake.url.clone()),
        ("CB_KB_TOKEN", KB_TOKEN.to_string()),
        ("CB_RETRY_BASE_MS", "1".to_string()),
        ("CB_LOG_FORMAT", "json".to_string()),
        ("RUST_LOG", "debug".to_string()),
    ];
    fake.always(Route::Ingest, "cb/r-2.md", Action::Status(400));

    // A failure remains: exit 2, and the run is said not to be complete.
    let output = run_binary(env.clone(), upload_line(&data, &run)).await;
    let (stdout, stderr) = (
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(
        stderr.contains("not complete") && stderr.contains("failures.jsonl"),
        "{stderr}"
    );
    assert!(
        stderr
            .lines()
            .any(|l| l.contains("\"item_failed\"") && l.contains("r-2")),
        "one structured line per failure: {stderr}"
    );
    assert!(
        stderr.lines().any(|l| l.contains("\"command_ended\"")),
        "{stderr}"
    );
    for printed in [&stdout, &stderr] {
        assert!(!printed.contains(KB_TOKEN), "a token was printed");
        assert!(
            !printed.contains(TEXT_ONE) && !printed.contains(TEXT_TWO),
            "a record's text was printed"
        );
    }
    assert!(
        fake.seen(Route::Ingest)
            .iter()
            .all(|s| s.authorization.as_deref() == Some(&*format!("Bearer {KB_TOKEN}")))
    );

    // Nothing remains: exit 0.
    fake.clear_rules();
    let output = run_binary(env.clone(), upload_line(&data, &run)).await;
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_no_secret_in(&run);

    // The command could not run: exit 1, and it says which setting is missing.
    let without_address: Vec<_> = env.into_iter().filter(|(name, _)| *name != "CB_KB_URL").collect();
    let output = run_binary(without_address, upload_line(&data, &dir.join("run-2"))).await;
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("CB_KB_URL"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_token_given_as_a_flag_is_not_shown_in_the_help_or_an_error() {
    let output = run_binary(
        vec![],
        [
            "--kb-token",
            "flag-token-never-shown",
            "vectors",
            "--collection",
            "c",
            "--tenant",
            "t",
            "--run",
            "/nonexistent/run",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
    )
    .await;
    assert_eq!(output.status.code(), Some(1), "no index address");
    let printed = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!printed.contains("flag-token-never-shown"), "{printed}");
    let help = run_binary(
        vec![("CB_KB_TOKEN", "env-token-never-shown".to_string())],
        vec!["--help".to_string()],
    )
    .await;
    let printed = String::from_utf8_lossy(&help.stdout);
    assert!(
        printed.contains("CB_KB_TOKEN") && !printed.contains("env-token-never-shown"),
        "{printed}"
    );
}
