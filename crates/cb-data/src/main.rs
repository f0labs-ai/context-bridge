//! `cb-data` — builds the data sets of the ContextBridge proof.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "cb-data", about = "Builds the data sets of the ContextBridge proof")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The generated incident corpus (docs/SYNTH_SPEC.md).
    Synth(cb_data::synth::Args),
    /// Public incident timelines (docs/PUBLIC_DATA.md).
    Public(cb_data::public::Args),
    /// Check that a data set directory keeps the contract.
    Check(cb_data::check::Args),
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Synth(args) => cb_data::synth::run(args),
        Command::Public(args) => cb_data::public::run(args),
        Command::Check(args) => cb_data::check::run(args),
    }
}
