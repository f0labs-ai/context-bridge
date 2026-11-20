//! Ranking rules, the two-conversation rule, measures and statistics of the
//! ContextBridge proof. See DESIGN.md sections 2 to 7.
//!
//! Everything here is computed offline from the files of a data set and a run
//! (`cb_types`), so anyone holding the files can reproduce every number.
//!
//! - [`spec`]: the methods under test as typed values.
//! - [`records`], [`context`], [`vectors`]: what a method may read.
//! - [`methods`]: every ranking method, each a pure function.
//! - [`measures`], [`aggregate`]: the measures of section 5, per question and per group.
//! - [`stats`]: seeded bootstrap intervals and randomization tests over threads.
//! - [`scores`]: the scores `rank` wrote, lined up for statistics.
//! - [`latex`]: numbers, tables and CSV for the paper.
//! - [`gate`]: the two-conversation rule.
//! - [`commands`]: what each command of the `cb-eval` binary does.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod aggregate;
pub mod commands;
pub mod context;
pub mod error;
pub mod files;
pub mod gate;
pub mod latex;
pub mod measures;
pub mod methods;
pub mod parallel;
pub mod records;
pub mod scores;
pub mod spec;
pub mod stats;
pub mod vectors;

#[cfg(test)]
pub(crate) mod testkit;

pub use error::EvalError;
pub use spec::MethodSpec;
