//! Drivers for the services under test: upload, candidate lists, extraction, timing.
//!
//! `cb-run` is the only tool of this repository that talks to a service. It writes down
//! what the services return in the shapes of `cb-types` (DESIGN.md section 3), and the
//! evaluator computes every number from those files alone. So nothing here may lose,
//! reorder or silently skip anything: a run with missing items that looks complete would
//! put a wrong number in the paper. See docs/RUN_TOOLS.md for the commands and
//! `NOTES.md` beside this crate for every decision the specification left open.
//!
//! The layers, from the outside in:
//!
//! - [`cli`]: flags and environment, and the dispatch to a command.
//! - [`commands`]: one module per command.
//! - [`knowledge`], [`index`], [`model`]: each service behind a small trait, with its one
//!   HTTP implementation.
//! - [`retry`], [`http`]: bounded retries with back-off; timeouts on every request.
//! - [`run`]: a command's stay in a run directory (lock, `failures.jsonl`, `manifest.json`).
//! - [`jsonl`], [`atomic`], [`vector_file`]: run files that are appended to or replaced whole.
//! - [`text`], [`seeded`]: pure rules (transcripts, the model's answer, timestamps, seeded draws).
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod atomic;
pub mod cli;
pub mod commands;
pub mod error;
pub mod http;
pub mod index;
pub mod jsonl;
pub mod knowledge;
pub mod model;
pub mod retry;
pub mod run;
pub mod secret;
pub mod seeded;
pub mod text;
pub mod vector_file;

#[cfg(test)]
pub(crate) mod test_support;
