//! Data tools: the generated incident corpus and the loaders for public
//! incident timelines. See DESIGN.md section 2 and docs/SYNTH_SPEC.md.
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod check;
pub mod files;
pub mod public;
pub mod synth;
