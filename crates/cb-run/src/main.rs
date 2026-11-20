//! `cb-run` — drives the services under test (docs/RUN_TOOLS.md).
//!
//! Exit status: 0 when the run directory holds no failure; 2 when failures remain in
//! `<run>/failures.jsonl`; 1 when the command could not run at all.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use anyhow::Context;
use cb_run::cli::{Cli, LogFormat};
use cb_run::run::{FAILURES_FILE, Report};
use clap::Parser;
use std::process::ExitCode;
use tracing_subscriber::EnvFilter;

/// Log lines go to standard error, `info` and above unless `RUST_LOG` says otherwise.
fn start_logging(format: LogFormat) {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr);
    match format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
}

fn run(cli: &Cli) -> anyhow::Result<Report> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?;
    let report = runtime.block_on(cb_run::cli::run(cli))?;
    Ok(report)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    start_logging(cli.services.log_format);
    match run(&cli) {
        Ok(report) if report.is_complete() => ExitCode::SUCCESS,
        Ok(report) => {
            eprintln!(
                "cb-run {}: not complete: {} failure(s) of this command and {} of other commands remain in {}",
                report.command,
                report.remaining.own,
                report.remaining.others,
                report.run_dir.join(FAILURES_FILE).display()
            );
            ExitCode::from(2)
        }
        Err(error) => {
            eprintln!("cb-run: {error:#}");
            ExitCode::FAILURE
        }
    }
}
