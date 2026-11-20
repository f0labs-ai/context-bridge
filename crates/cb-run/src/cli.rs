//! The command line: where the services are, and which command to run.
//!
//! Addresses and tokens come from flags or the environment (`CB_KB_URL`, `CB_KB_TOKEN`,
//! `CB_QDRANT_URL`, `CB_MODEL_URL`, `CB_MODEL_KEY`). Nothing is built in. A token is held
//! as a [`Secret`]: it is sent to its service and shown nowhere, not in `--help`, not in
//! a log line, not in a manifest.

use crate::commands::Shared;
use crate::commands::candidates::{CandidatesArgs, candidates};
use crate::commands::control::{ExportArgs, ImportArgs, export, import};
use crate::commands::embed::{EmbedArgs, embed};
use crate::commands::extract::{ExtractArgs, extract};
use crate::commands::judge::{JudgeArgs, judge};
use crate::commands::newest::{NewestArgs, newest};
use crate::commands::questions::{QuestionsArgs, questions};
use crate::commands::trips::{TripsArgs, trips};
use crate::commands::upload::{UploadArgs, upload};
use crate::commands::vectors::{VectorsArgs, vectors};
use crate::error::RunError;
use crate::http::{Http, HttpSettings, parse_address};
use crate::index::HttpSearchIndex;
use crate::knowledge::HttpKnowledgeService;
use crate::model::HttpModelEndpoint;
use crate::retry::RetryPolicy;
use crate::run::Report;
use crate::secret::Secret;
use clap::{Parser, Subcommand, ValueEnum};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(
    name = "cb-run",
    version,
    about = "Drives the services under test and writes run files (docs/RUN_TOOLS.md)",
    after_help = "Exit status: 0 when the run directory holds no failure; 2 when failures remain in \
                  <run>/failures.jsonl; 1 when the command could not run."
)]
pub struct Cli {
    #[command(flatten)]
    pub services: ServiceArgs,
    #[command(subcommand)]
    pub command: Command,
}

/// How log lines are written (to standard error).
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogFormat {
    /// `key=value` lines for a person.
    Text,
    /// One JSON object per line for a machine.
    Json,
}

/// Where the services are and how they are called. The same for every command.
#[derive(clap::Args, Debug, Clone)]
pub struct ServiceArgs {
    /// The knowledge service's address.
    #[arg(long, env = "CB_KB_URL", hide_env_values = true, global = true)]
    pub kb_url: Option<String>,
    /// The knowledge service's internal token. Use the environment variable: a flag is
    /// visible to everyone who can list processes.
    #[arg(long, env = "CB_KB_TOKEN", hide_env_values = true, global = true)]
    pub kb_token: Option<Secret>,
    /// The search index's address.
    #[arg(long, env = "CB_QDRANT_URL", hide_env_values = true, global = true)]
    pub qdrant_url: Option<String>,
    /// The language-model endpoint's address (OpenAI-compatible; `…/v1`).
    #[arg(long, env = "CB_MODEL_URL", hide_env_values = true, global = true)]
    pub model_url: Option<String>,
    /// The language-model endpoint's key. Use the environment variable.
    #[arg(long, env = "CB_MODEL_KEY", hide_env_values = true, global = true)]
    pub model_key: Option<Secret>,
    /// How long a request may take to connect, in milliseconds.
    #[arg(long, env = "CB_CONNECT_TIMEOUT_MS", default_value_t = 5_000, global = true)]
    pub connect_timeout_ms: u64,
    /// How long a whole request may take, in milliseconds.
    #[arg(long, env = "CB_TIMEOUT_MS", default_value_t = 120_000, global = true)]
    pub timeout_ms: u64,
    /// The wait before a second try, in milliseconds; it doubles before a third.
    #[arg(long, env = "CB_RETRY_BASE_MS", default_value_t = 500, global = true)]
    pub retry_base_ms: u64,
    /// What to call this machine in the manifest (its hardware, for timing claims).
    #[arg(long, env = "CB_MACHINE", global = true)]
    pub machine: Option<String>,
    #[arg(long, env = "CB_LOG_FORMAT", value_enum, default_value_t = LogFormat::Text, global = true)]
    pub log_format: LogFormat,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Upload a data set's records to the knowledge service.
    Upload(UploadArgs),
    /// Collect the candidate lists behind a search, for every question and view.
    Candidates(CandidatesArgs),
    /// Collect the newest records each question could be shown, with their relevance.
    Newest(NewestArgs),
    /// Read one vector per record from the search index.
    Vectors(VectorsArgs),
    /// Read transcripts with a language model.
    Extract(ExtractArgs),
    /// A language model's verdict on each extraction of the gate's judge sample.
    Judge(JudgeArgs),
    /// Embed texts through the knowledge service.
    Embed(EmbedArgs),
    /// Time how long a new record takes to become searchable.
    Trips(TripsArgs),
    /// Write ordinary questions for a data set's standing records with a language model.
    Questions(QuestionsArgs),
    /// The control data: passages of the product's own index.
    #[command(subcommand)]
    Control(ControlCommand),
}

#[derive(Subcommand, Debug)]
pub enum ControlCommand {
    /// Read one tenant's passages from a search index and write them as a data set.
    Export(ExportArgs),
    /// Write a control data set's passages into the experiment's collection.
    Import(ImportArgs),
}

fn required<'a>(value: Option<&'a String>, name: &'static str, flag: &str) -> Result<&'a str, RunError> {
    value
        .map(String::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| RunError::Setting {
            name,
            reason: format!("not set (or give {flag})"),
        })
}

impl ServiceArgs {
    fn http(&self) -> Result<Http, RunError> {
        Http::new(&HttpSettings {
            connect_timeout: Duration::from_millis(self.connect_timeout_ms),
            total_timeout: Duration::from_millis(self.timeout_ms),
            proxy_from_env: true,
        })
    }

    /// What every command is given besides its own flags.
    pub fn shared(&self) -> Shared {
        Shared {
            retry: RetryPolicy::standard(Duration::from_millis(self.retry_base_ms)),
            machine_label: self.machine.clone().filter(|label| !label.trim().is_empty()),
        }
    }

    /// The knowledge service these flags name.
    pub fn knowledge_service(&self) -> Result<HttpKnowledgeService, RunError> {
        let base = parse_address(
            "CB_KB_URL",
            required(self.kb_url.as_ref(), "CB_KB_URL", "--kb-url")?,
        )?;
        Ok(HttpKnowledgeService::new(
            self.http()?,
            base,
            self.kb_token.clone(),
        ))
    }

    /// The search index these flags name.
    pub fn search_index(&self) -> Result<HttpSearchIndex, RunError> {
        let base = parse_address(
            "CB_QDRANT_URL",
            required(self.qdrant_url.as_ref(), "CB_QDRANT_URL", "--qdrant-url")?,
        )?;
        Ok(HttpSearchIndex::new(self.http()?, base))
    }

    /// The language-model endpoint these flags name.
    pub fn model_endpoint(&self) -> Result<HttpModelEndpoint, RunError> {
        let base = parse_address(
            "CB_MODEL_URL",
            required(self.model_url.as_ref(), "CB_MODEL_URL", "--model-url")?,
        )?;
        HttpModelEndpoint::new(self.http()?, &base, self.model_key.clone())
    }
}

/// Runs the command the line names against the services it names.
pub async fn run(cli: &Cli) -> Result<Report, RunError> {
    let services = &cli.services;
    let shared = services.shared();
    match &cli.command {
        Command::Upload(args) => upload(&services.knowledge_service()?, args, &shared).await,
        Command::Candidates(args) => candidates(&services.knowledge_service()?, args, &shared).await,
        Command::Newest(args) => newest(&services.knowledge_service()?, args, &shared).await,
        Command::Vectors(args) => vectors(&services.search_index()?, args, &shared).await,
        Command::Extract(args) => extract(&services.model_endpoint()?, args, &shared).await,
        Command::Judge(args) => judge(&services.model_endpoint()?, args, &shared).await,
        Command::Embed(args) => embed(&services.knowledge_service()?, args, &shared).await,
        Command::Trips(args) => trips(&services.knowledge_service()?, args, &shared).await,
        Command::Questions(args) => questions(&services.model_endpoint()?, args, &shared).await,
        Command::Control(ControlCommand::Export(args)) => {
            export(&services.search_index()?, args, &shared).await
        }
        Command::Control(ControlCommand::Import(args)) => {
            import(&services.search_index()?, args, &shared).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn parse(line: &[&str]) -> Cli {
        Cli::try_parse_from(line).unwrap()
    }

    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn the_documented_invocations_are_understood() {
        let lines: [&[&str]; 10] = [
            &[
                "cb-run",
                "newest",
                "--data",
                "data/github",
                "--tenant",
                "t",
                "--collection",
                "cb",
                "--run",
                "runs/github",
                "--concurrency",
                "8",
            ],
            &[
                "cb-run",
                "upload",
                "--data",
                "data/synth-42",
                "--tenant",
                "t",
                "--collection",
                "cb",
                "--run",
                "runs/synth-42",
            ],
            &[
                "cb-run",
                "candidates",
                "--data",
                "data/synth-42",
                "--tenant",
                "t",
                "--collection",
                "cb",
                "--run",
                "runs/synth-42",
            ],
            &[
                "cb-run",
                "vectors",
                "--collection",
                "cb_index",
                "--tenant",
                "t",
                "--run",
                "runs/synth-42",
            ],
            &[
                "cb-run",
                "extract",
                "--transcripts",
                "data/synth-42/transcripts.jsonl",
                "--model",
                "m",
                "--prompt",
                "prompts/extract.v1.md",
                "--run",
                "runs/x",
            ],
            &[
                "cb-run",
                "embed",
                "--texts",
                "t.jsonl",
                "--run",
                "runs/x",
                "--name",
                "extractions",
            ],
            &[
                "cb-run",
                "trips",
                "--data",
                "data/trips",
                "--tenant",
                "t",
                "--collection",
                "cb",
                "--run",
                "runs/trips",
                "--n",
                "1000",
            ],
            &[
                "cb-run",
                "questions",
                "--data",
                "data/github",
                "--model",
                "m",
                "--per-section",
                "1",
                "--sample",
                "800",
                "--seed",
                "42",
            ],
            &[
                "cb-run",
                "control",
                "export",
                "--from-collection",
                "c",
                "--from-tenant",
                "t",
                "--questions",
                "helpq.json",
                "--window-start",
                "1780272000",
                "--out",
                "data/helpcentre",
            ],
            &[
                "cb-run",
                "control",
                "import",
                "--data",
                "data/helpcentre",
                "--tenant",
                "t",
                "--collection",
                "cb_index",
                "--label",
                "cb",
            ],
        ];
        for line in lines {
            assert!(Cli::try_parse_from(line).is_ok(), "{line:?}");
        }
    }

    #[test]
    fn defaults_are_the_documented_ones() {
        let cli = parse(&[
            "cb-run",
            "upload",
            "--data",
            "d",
            "--tenant",
            "t",
            "--collection",
            "cb",
            "--run",
            "r",
        ]);
        let Command::Upload(args) = &cli.command else {
            panic!("upload")
        };
        assert_eq!((args.concurrency, args.only), (4, None));
        let cli = parse(&[
            "cb-run",
            "extract",
            "--transcripts",
            "t",
            "--model",
            "m",
            "--prompt",
            "p",
            "--run",
            "r",
        ]);
        let Command::Extract(args) = &cli.command else {
            panic!("extract")
        };
        assert_eq!((args.concurrency, args.limit), (8, None));
        let cli = parse(&[
            "cb-run",
            "trips",
            "--data",
            "d",
            "--tenant",
            "t",
            "--collection",
            "cb",
            "--run",
            "r",
        ]);
        let Command::Trips(args) = &cli.command else {
            panic!("trips")
        };
        assert_eq!((args.poll_ms, args.give_up_ms, args.n), (25, 30_000, None));
        assert_eq!(cli.services.shared().retry.max_tries, 3);
    }

    #[test]
    fn an_extra_body_is_one_json_object_that_cannot_set_what_the_tool_sets() {
        let extract = |extra: &str| {
            Cli::try_parse_from([
                "cb-run",
                "extract",
                "--transcripts",
                "t",
                "--model",
                "m",
                "--prompt",
                "p",
                "--run",
                "r",
                "--extra-body",
                extra,
            ])
        };
        let hosted = r#"{"reasoning":{"enabled":false},"chat_template_kwargs":{"enable_thinking":false}}"#;
        let Ok(Cli {
            command: Command::Extract(args),
            ..
        }) = extract(hosted)
        else {
            panic!("the hosted model's extra body is taken")
        };
        assert_eq!(
            args.extra_body.map(|extra| extra.canonical()),
            Some(
                r#"{"chat_template_kwargs":{"enable_thinking":false},"reasoning":{"enabled":false}}"#
                    .to_string()
            )
        );
        for refused in [
            r#"{"messages": []}"#,
            r#"{"temperature": 0.7}"#,
            r#"{"stream": true}"#,
            "[1, 2]",
            "not json",
        ] {
            assert!(extract(refused).is_err(), "{refused}");
        }
        let questions = Cli::try_parse_from([
            "cb-run",
            "questions",
            "--data",
            "d",
            "--model",
            "m",
            "--sample",
            "1",
            "--seed",
            "1",
            "--extra-body",
            r#"{"model": "other"}"#,
        ]);
        assert!(questions.is_err(), "the model is named once, by --model");
    }

    #[test]
    fn a_token_is_never_shown_and_never_recorded() {
        let cli = parse(&[
            "cb-run",
            "--kb-token",
            "kb-very-private",
            "--model-key",
            "sk-very-private",
            "upload",
            "--data",
            "d",
            "--tenant",
            "t",
            "--collection",
            "cb",
            "--run",
            "r",
        ]);
        let shown = format!("{cli:?}");
        assert!(!shown.contains("very-private"), "{shown}");
        assert_eq!(
            cli.services.kb_token.as_ref().map(Secret::expose),
            Some("kb-very-private")
        );
        // What a manifest records of a command is its own flags: no token is among them.
        let Command::Upload(args) = &cli.command else {
            panic!("upload")
        };
        let recorded = serde_json::to_string(args).unwrap();
        assert!(!recorded.contains("very-private"), "{recorded}");
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("CB_KB_TOKEN") && !help.contains("very-private"));
    }

    #[test]
    fn a_service_that_is_not_named_is_asked_for_by_name() {
        let cli = parse(&[
            "cb-run",
            "upload",
            "--data",
            "d",
            "--tenant",
            "t",
            "--collection",
            "cb",
            "--run",
            "r",
        ]);
        let mut services = cli.services.clone();
        services.kb_url = None;
        services.qdrant_url = Some("  ".into());
        services.model_url = Some("localhost:8000".into());
        for error in [
            services.knowledge_service().err(),
            services.search_index().err(),
            services.model_endpoint().err(),
        ] {
            assert!(matches!(error, Some(RunError::Setting { .. })), "{error:?}");
        }
        services.kb_url = Some("http://127.0.0.1:8082".into());
        assert!(services.knowledge_service().is_ok());
    }
}
