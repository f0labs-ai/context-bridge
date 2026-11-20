//! Every command against an in-process stand-in for the services (`fake.rs`): the happy
//! path, and the failure and resume paths, for real over HTTP.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod fake;
mod support;

mod binary;
mod candidates;
mod control;
mod embed;
mod extract;
mod questions;
mod trips;
mod upload;
mod vectors;
