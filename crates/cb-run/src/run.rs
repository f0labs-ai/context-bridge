//! One command's stay in a run directory: the lock, `failures.jsonl` and `manifest.json`.
//!
//! - **The lock.** Only one `cb-run` writes to a run directory at a time, so two resumes
//!   started by mistake can never write the same item twice. The lock is the operating
//!   system's and goes away with the process, so a crash never leaves it behind.
//! - **`failures.jsonl`** holds the failures that REMAIN in the run, one per line, for all
//!   commands. A failure is appended the moment it happens. When a command ends, its
//!   lines for items that are now done are removed; other commands' lines are left as
//!   they are. A command exits non-zero while the file holds any line.
//! - **`manifest.json`** is `{"entries": [...]}`. A command adds its entry when it starts
//!   (no end time yet) and completes it when it ends, so a run that was killed shows as
//!   one that never ended. Entries written by other tools are kept byte for byte.

use crate::atomic::{write_atomically, write_json_atomically};
use crate::error::{FailureKind, RunError};
use crate::jsonl::Appender;
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const FAILURES_FILE: &str = "failures.jsonl";
pub const MANIFEST_FILE: &str = "manifest.json";
const LOCK_FILE: &str = ".cb-run.lock";
const TOOL: &str = "cb-run";

/// The commit this tool was built from (`-dirty` when tracked files differed from it).
pub const GIT_COMMIT: &str = env!("CB_RUN_GIT_COMMIT");
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The current moment as run files carry it: ISO 8601, UTC, milliseconds.
pub fn now_utc() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

// ---- failures ------------------------------------------------------------------------

/// One item a command could not finish.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FailureLine {
    /// The command that failed on the item.
    pub command: String,
    /// The record, question, transcript or point the failure is about; empty when it is
    /// about the command as a whole.
    pub id: String,
    pub kind: FailureKind,
    /// The error in words.
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Tries made before giving up.
    pub tries: u32,
    /// What the service or the model sent back, when something came back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<String>,
    /// When the failure was recorded (UTC).
    pub at: String,
}

/// How many failure lines a run directory holds after a command ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Remaining {
    /// Lines of the command that just ran.
    pub own: usize,
    /// Lines of other commands, or lines this tool did not write.
    pub others: usize,
}

struct FailureLog {
    path: PathBuf,
    command: &'static str,
    /// Lines of other commands, kept exactly as they are.
    others: Vec<String>,
    /// This command's lines from earlier runs.
    earlier: Vec<FailureLine>,
    /// This run's lines.
    fresh: Vec<FailureLine>,
    appender: Appender,
}

impl FailureLog {
    fn open(dir: &Path, command: &'static str) -> Result<Self, RunError> {
        let path = dir.join(FAILURES_FILE);
        let mut others = Vec::new();
        let mut earlier = Vec::new();
        let mut cut_off = false;
        match fs::read(&path) {
            Ok(bytes) => {
                let ends_whole = bytes.is_empty() || bytes.last() == Some(&b'\n');
                let pieces: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
                for (i, piece) in pieces.iter().enumerate() {
                    let text = piece.trim_ascii();
                    if text.is_empty() {
                        continue;
                    }
                    let last_without_newline = !ends_whole && i + 1 == pieces.len();
                    match serde_json::from_slice::<FailureLine>(text) {
                        Ok(line) if line.command == command => earlier.push(line),
                        Ok(_) => others.push(String::from_utf8_lossy(text).into_owned()),
                        // A cut-off last line was never a whole failure record.
                        Err(_) if last_without_newline => cut_off = true,
                        // A whole line this tool cannot read is still someone's record.
                        Err(_) => others.push(String::from_utf8_lossy(text).into_owned()),
                    }
                }
                if cut_off || !ends_whole {
                    Self::rewrite(&path, &others, earlier.iter())?;
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(RunError::io(&path, e)),
        }
        let appender = Appender::open(&path)?;
        Ok(Self {
            path,
            command,
            others,
            earlier,
            fresh: Vec::new(),
            appender,
        })
    }

    fn rewrite<'a>(
        path: &Path,
        others: &[String],
        own: impl Iterator<Item = &'a FailureLine>,
    ) -> Result<(), RunError> {
        let own: Vec<&FailureLine> = own.collect();
        write_atomically(path, |writer| {
            for line in others {
                writer.write_all(line.as_bytes())?;
                writer.write_all(b"\n")?;
            }
            for line in &own {
                serde_json::to_writer(&mut *writer, line).map_err(std::io::Error::other)?;
                writer.write_all(b"\n")?;
            }
            Ok(())
        })
    }

    fn record(&mut self, line: FailureLine) -> Result<(), RunError> {
        self.appender.append(std::slice::from_ref(&line))?;
        self.fresh.push(line);
        Ok(())
    }

    /// Leaves in the file: other commands' lines, this command's earlier lines for items
    /// that are still not done and were not tried again, and this run's lines.
    fn settle(self, done: impl Fn(&str) -> bool) -> Result<Remaining, RunError> {
        let tried_again: HashSet<&str> = self.fresh.iter().map(|l| l.id.as_str()).collect();
        let still_open: Vec<&FailureLine> = self
            .earlier
            .iter()
            .filter(|l| !done(&l.id) && !tried_again.contains(l.id.as_str()))
            .collect();
        let own = still_open.len() + self.fresh.len();
        Self::rewrite(
            &self.path,
            &self.others,
            still_open.into_iter().chain(self.fresh.iter()),
        )?;
        tracing::debug!(
            command = self.command,
            own,
            others = self.others.len(),
            "failures_settled"
        );
        Ok(Remaining {
            own,
            others: self.others.len(),
        })
    }
}

// ---- manifest ------------------------------------------------------------------------

/// The knowledge service a command talked to: where, and what it said about itself.
#[derive(Clone, Debug, Serialize)]
pub struct ServiceRecord {
    pub url: String,
    /// The service's `/v1/info`, exactly as it answered; none when it did not answer.
    pub info: Option<Box<RawValue>>,
    /// The embedding model the service named in its `/v1/embed` answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
}

/// The search index a command read or wrote.
#[derive(Clone, Debug, Serialize)]
pub struct IndexRecord {
    pub url: String,
    pub collection: String,
}

/// The language model a command asked.
#[derive(Clone, Debug, Serialize)]
pub struct ModelRecord {
    pub url: String,
    /// The model as it was asked for.
    pub name: String,
    /// The model names the endpoint put in its answers.
    pub reported_names: Vec<String>,
    /// The prompt's version (its file name without the extension) and the sha256 of its text.
    pub prompt_version: String,
    pub prompt_sha256: String,
    /// Whether answers were asked for with `response_format: json_object` at the end of
    /// the run (false after the endpoint refused the field).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_mode: Option<bool>,
    /// The fields added to every request body for this endpoint (`--extra-body`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<crate::model::ExtraBody>,
}

/// The services a command used, for its manifest entry.
#[derive(Clone, Debug, Default)]
pub struct About {
    pub knowledge_service: Option<ServiceRecord>,
    pub index: Option<IndexRecord>,
    pub model: Option<ModelRecord>,
    /// What the operator calls the machine (`--machine`), for timing claims.
    pub machine_label: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct Machine {
    os: &'static str,
    arch: &'static str,
    cpus: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
}

/// How a command ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ending {
    /// Still running, or killed before it could say more.
    Running,
    /// Everything asked for is in the run files and no failure remains in the run.
    Complete,
    /// The command ran to its end, but failures remain.
    Incomplete,
    /// The command stopped on an error before its end.
    Stopped,
}

#[derive(Clone, Debug, Serialize)]
struct Entry {
    /// Tells this entry from every other one in the file.
    id: String,
    tool: &'static str,
    version: &'static str,
    git_commit: &'static str,
    command: &'static str,
    /// The command's flags as given. Tokens and keys are not flags of this record.
    flags: Box<RawValue>,
    started_at: String,
    ended_at: Option<String>,
    ending: Ending,
    /// The run file this command adds to, when it picks up where an earlier run stopped.
    #[serde(skip_serializing_if = "Option::is_none")]
    output: Option<String>,
    /// What that run file's content belongs to ([`Session::claim`]).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    identity: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    knowledge_service: Option<ServiceRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<IndexRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<ModelRecord>,
    machine: Machine,
    counts: BTreeMap<String, u64>,
}

#[derive(Serialize, Deserialize, Default)]
struct ManifestFile {
    entries: Vec<Box<RawValue>>,
}

#[derive(Deserialize)]
struct EntryId {
    #[serde(default)]
    id: Option<String>,
}

/// The part of an earlier entry that says what a run file belongs to.
#[derive(Deserialize)]
struct EarlierClaim {
    tool: Option<String>,
    command: Option<String>,
    output: Option<String>,
    identity: Option<BTreeMap<String, String>>,
}

fn read_manifest(path: &Path) -> Result<ManifestFile, RunError> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
            RunError::file(
                path,
                format!("not a run manifest ({e}); it is left as it is and nothing was run"),
            )
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(ManifestFile::default()),
        Err(e) => Err(RunError::io(path, e)),
    }
}

/// Puts `entry` into the manifest: in place of the entry with the same id, or at the end.
fn store_entry(path: &Path, entry: &Entry) -> Result<(), RunError> {
    let mut manifest = read_manifest(path)?;
    let raw = serde_json::value::to_raw_value(entry).map_err(|e| RunError::file(path, e.to_string()))?;
    let place = manifest.entries.iter().position(|existing| {
        serde_json::from_str::<EntryId>(existing.get())
            .ok()
            .and_then(|e| e.id)
            .is_some_and(|id| id == entry.id)
    });
    match place {
        Some(i) => manifest.entries[i] = raw,
        None => manifest.entries.push(raw),
    }
    write_json_atomically(path, &manifest)
}

// ---- session -------------------------------------------------------------------------

/// What a command reports when it ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub command: &'static str,
    pub run_dir: PathBuf,
    pub remaining: Remaining,
    pub counts: BTreeMap<String, u64>,
}

impl Report {
    /// Whether the run directory holds no failure at all. Anything else must not pass for
    /// a complete run.
    pub fn is_complete(&self) -> bool {
        self.remaining.own == 0 && self.remaining.others == 0
    }
}

/// How many items make one logged batch.
pub const BATCH_ITEMS: u64 = 100;

/// One command at work in a run directory.
pub struct Session {
    dir: PathBuf,
    /// Held for as long as the command runs.
    _lock: File,
    failures: FailureLog,
    entry: Entry,
    started: Instant,
    batch_started: Instant,
    items_in_batch: u64,
    failures_in_batch: u64,
}

impl Session {
    /// Enters `dir` (created when absent) for `command`: takes the lock, reads the
    /// failures that remain, and adds the command's manifest entry.
    pub fn begin<F: Serialize>(
        dir: &Path,
        command: &'static str,
        flags: &F,
        about: About,
    ) -> Result<Self, RunError> {
        fs::create_dir_all(dir).map_err(|e| RunError::io(dir, e))?;
        let lock_path = dir.join(LOCK_FILE);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|e| RunError::io(&lock_path, e))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(fs::TryLockError::WouldBlock) => {
                return Err(RunError::file(
                    dir,
                    "another cb-run is writing to this run directory; wait for it to end",
                ));
            }
            Err(fs::TryLockError::Error(e)) => return Err(RunError::io(&lock_path, e)),
        }
        let manifest_path = dir.join(MANIFEST_FILE);
        let started_at = now_utc();
        let entry = Entry {
            id: format!("{TOOL}-{command}-{started_at}-{}", std::process::id()),
            tool: TOOL,
            version: VERSION,
            git_commit: GIT_COMMIT,
            command,
            flags: serde_json::value::to_raw_value(flags)
                .map_err(|e| RunError::file(&manifest_path, e.to_string()))?,
            started_at,
            ended_at: None,
            ending: Ending::Running,
            output: None,
            identity: BTreeMap::new(),
            knowledge_service: about.knowledge_service,
            index: about.index,
            model: about.model,
            machine: Machine {
                os: std::env::consts::OS,
                arch: std::env::consts::ARCH,
                cpus: std::thread::available_parallelism().map_or(0, usize::from),
                label: about.machine_label,
            },
            counts: BTreeMap::new(),
        };
        // The manifest first: when it cannot be read, nothing else is touched.
        store_entry(&manifest_path, &entry)?;
        let failures = FailureLog::open(dir, command)?;
        tracing::info!(command, run = %dir.display(), version = VERSION, git_commit = GIT_COMMIT, "command_started");
        let now = Instant::now();
        Ok(Self {
            dir: dir.to_path_buf(),
            _lock: lock,
            failures,
            entry,
            started: now,
            batch_started: now,
            items_in_batch: 0,
            failures_in_batch: 0,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn command(&self) -> &'static str {
        self.entry.command
    }

    /// The path of a run file.
    pub fn file(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    /// Says what the run file `output` belongs to (a data set, a tenant, a collection, a
    /// service, a model), before the command adds to it.
    ///
    /// A command skips what its run file already holds. That is only right while the file
    /// still belongs to the same things: pointed at another tenant or another data set,
    /// the same run directory would skip everything and write nothing, and the run would
    /// look complete. So when the file holds lines and the last command that added to it
    /// said it belongs to something else, this command refuses, and names what differs.
    /// A file that is absent or empty belongs to whoever claims it.
    pub fn claim(&mut self, output: &str, identity: BTreeMap<String, String>) -> Result<(), RunError> {
        let output_path = self.dir.join(output);
        let holds_lines = fs::metadata(&output_path).is_ok_and(|m| m.len() > 0);
        if holds_lines {
            let manifest = read_manifest(&self.dir.join(MANIFEST_FILE))?;
            let last = manifest
                .entries
                .iter()
                .filter_map(|raw| serde_json::from_str::<EarlierClaim>(raw.get()).ok())
                .filter(|earlier| {
                    earlier.tool.as_deref() == Some(TOOL)
                        && earlier.command.as_deref() == Some(self.entry.command)
                        && earlier.output.as_deref() == Some(output)
                })
                .filter_map(|earlier| earlier.identity)
                .next_back();
            if let Some(earlier) = last {
                let differs = identity
                    .iter()
                    .find(|(key, value)| earlier.get(*key) != Some(*value))
                    .map(|(key, value)| (key, earlier.get(key).map_or("nothing", String::as_str), value));
                if let Some((key, before, now)) = differs {
                    return Err(RunError::file(
                        &output_path,
                        format!(
                            "this file was written for {key} = {before} and this run is for {key} = {now}; \
                             a run directory belongs to one of each: use another --run"
                        ),
                    ));
                }
            }
        }
        self.entry.output = Some(output.to_string());
        self.entry.identity = identity;
        store_entry(&self.dir.join(MANIFEST_FILE), &self.entry)
    }

    /// Adds `n` to a count of the manifest entry.
    pub fn add(&mut self, count: &str, n: u64) {
        *self.entry.counts.entry(count.to_string()).or_insert(0) += n;
    }

    /// Sets a count of the manifest entry.
    pub fn set(&mut self, count: &str, n: u64) {
        self.entry.counts.insert(count.to_string(), n);
    }

    /// The model record of the manifest entry, to complete when the run ends.
    pub fn model_mut(&mut self) -> Option<&mut ModelRecord> {
        self.entry.model.as_mut()
    }

    /// The knowledge-service record of the manifest entry, to complete once the service
    /// has said what it is.
    pub fn knowledge_service_mut(&mut self) -> Option<&mut ServiceRecord> {
        self.entry.knowledge_service.as_mut()
    }

    /// One item is in the run files. Counted as `written`, and logged with its batch.
    pub fn wrote(&mut self) {
        self.add("written", 1);
        self.tick(false);
    }

    /// One item failed: written to `failures.jsonl` now, counted as `failed`, logged with
    /// its batch. The log line carries the id, the kind, the status and the tries: never
    /// the error's text, which may quote what a service sent.
    pub fn fail(&mut self, id: &str, error: &RunError, tries: u32) -> Result<(), RunError> {
        self.fail_as(id, error.kind(), error, tries)
    }

    /// As [`Session::fail`], with the kind said by the command.
    pub fn fail_as(
        &mut self,
        id: &str,
        kind: FailureKind,
        error: &RunError,
        tries: u32,
    ) -> Result<(), RunError> {
        tracing::warn!(
            command = self.entry.command,
            id,
            kind = kind.as_str(),
            status = error.status(),
            tries,
            "item_failed"
        );
        self.failures.record(FailureLine {
            command: self.entry.command.to_string(),
            id: id.to_string(),
            kind,
            message: error.to_string(),
            url: error.url().map(str::to_string),
            status: error.status(),
            tries,
            raw: error.raw_answer().map(str::to_string),
            at: now_utc(),
        })?;
        self.add("failed", 1);
        self.tick(true);
        Ok(())
    }

    fn tick(&mut self, failed: bool) {
        self.items_in_batch += 1;
        self.failures_in_batch += u64::from(failed);
        if self.items_in_batch >= BATCH_ITEMS {
            self.log_batch();
        }
    }

    /// One structured line per batch: count, elapsed, failures.
    fn log_batch(&mut self) {
        if self.items_in_batch == 0 {
            return;
        }
        tracing::info!(
            command = self.entry.command,
            count = self.items_in_batch,
            failures = self.failures_in_batch,
            elapsed_ms = self.batch_started.elapsed().as_millis() as u64,
            written_so_far = self.entry.counts.get("written").copied().unwrap_or(0),
            failed_so_far = self.entry.counts.get("failed").copied().unwrap_or(0),
            "batch"
        );
        self.items_in_batch = 0;
        self.failures_in_batch = 0;
        self.batch_started = Instant::now();
    }

    /// Ends the command: settles `failures.jsonl` and completes the manifest entry.
    /// `done` says whether an item is in the run files now; this command's earlier
    /// failures for such items are dropped.
    pub fn finish(mut self, done: impl Fn(&str) -> bool) -> Result<Report, RunError> {
        self.log_batch();
        let remaining = self.failures.settle(done)?;
        self.entry
            .counts
            .insert("failures_remaining".to_string(), remaining.own as u64);
        self.entry.counts.insert(
            "failures_remaining_other_commands".to_string(),
            remaining.others as u64,
        );
        self.entry.ended_at = Some(now_utc());
        self.entry.ending = if remaining.own == 0 && remaining.others == 0 {
            Ending::Complete
        } else {
            Ending::Incomplete
        };
        store_entry(&self.dir.join(MANIFEST_FILE), &self.entry)?;
        tracing::info!(
            command = self.entry.command,
            elapsed_ms = self.started.elapsed().as_millis() as u64,
            written = self.entry.counts.get("written").copied().unwrap_or(0),
            failed = self.entry.counts.get("failed").copied().unwrap_or(0),
            failures_remaining = remaining.own,
            failures_remaining_other_commands = remaining.others,
            "command_ended"
        );
        Ok(Report {
            command: self.entry.command,
            run_dir: self.dir,
            remaining,
            counts: self.entry.counts,
        })
    }

    /// Ends the command after an error that stopped it: the entry says so, and the
    /// failures written so far stay where they are.
    pub fn stop(mut self, error: &RunError) {
        self.entry.ended_at = Some(now_utc());
        self.entry.ending = Ending::Stopped;
        tracing::error!(
            command = self.entry.command,
            kind = error.kind().as_str(),
            "command_stopped"
        );
        if let Err(e) = store_entry(&self.dir.join(MANIFEST_FILE), &self.entry) {
            tracing::error!(
                command = self.entry.command,
                kind = e.kind().as_str(),
                "manifest_not_completed"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ScratchDir;

    #[derive(Serialize)]
    struct Flags {
        data: &'static str,
        concurrency: u32,
    }

    const FLAGS: Flags = Flags {
        data: "data/synth-42",
        concurrency: 4,
    };

    fn timeout() -> RunError {
        RunError::Timeout {
            url: "http://kb/v1/tenants/t/ingest".into(),
            limit_ms: 100,
        }
    }

    fn failure_lines(dir: &Path) -> Vec<FailureLine> {
        fs::read_to_string(dir.join(FAILURES_FILE))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn manifest_entries(dir: &Path) -> Vec<serde_json::Value> {
        let manifest: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST_FILE)).unwrap()).unwrap();
        manifest["entries"].as_array().unwrap().clone()
    }

    #[test]
    fn a_failure_is_on_disk_at_once_and_makes_the_run_incomplete() {
        let dir = ScratchDir::new("run-failure");
        let mut session = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        session.fail("inc-0001.s1", &timeout(), 3).unwrap();
        // Before the command ends: already there.
        let lines = failure_lines(dir.path());
        assert_eq!(lines.len(), 1);
        assert_eq!(
            (lines[0].id.as_str(), lines[0].kind, lines[0].tries),
            ("inc-0001.s1", FailureKind::Timeout, 3)
        );
        let report = session.finish(|_| false).unwrap();
        assert!(!report.is_complete());
        assert_eq!(report.remaining, Remaining { own: 1, others: 0 });
        assert_eq!(report.counts.get("failed"), Some(&1));
        let entries = manifest_entries(dir.path());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["ending"], "incomplete");
        assert_eq!(entries[0]["flags"]["data"], "data/synth-42");
        assert!(entries[0]["ended_at"].is_string());
    }

    #[test]
    fn a_later_run_drops_failures_of_items_now_done_and_keeps_the_rest() {
        let dir = ScratchDir::new("run-resolve");
        let mut first = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        first.fail("a", &timeout(), 3).unwrap();
        first.fail("b", &timeout(), 3).unwrap();
        first.fail("c", &timeout(), 3).unwrap();
        assert_eq!(first.finish(|_| false).unwrap().remaining.own, 3);

        // The second run finishes `a`, fails `b` again, and does not touch `c`.
        let mut second = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        second.fail("b", &RunError::item("b", "still bad"), 1).unwrap();
        let report = second.finish(|id| id == "a").unwrap();
        assert_eq!(report.remaining, Remaining { own: 2, others: 0 });
        let lines = failure_lines(dir.path());
        let ids: Vec<&str> = lines.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, vec!["c", "b"]);
        assert_eq!(
            lines[1].kind,
            FailureKind::Input,
            "the newer failure of b replaced the older"
        );
    }

    #[test]
    fn another_commands_failures_are_kept_and_keep_the_run_incomplete() {
        let dir = ScratchDir::new("run-others");
        let mut upload = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        upload.fail("a", &timeout(), 3).unwrap();
        upload.finish(|_| false).unwrap();
        // A line this tool did not write is somebody's record too.
        let mut file = OpenOptions::new()
            .append(true)
            .open(dir.path().join(FAILURES_FILE))
            .unwrap();
        file.write_all(b"{\"note\":\"written by hand\"}\n").unwrap();
        drop(file);

        let candidates = Session::begin(dir.path(), "candidates", &FLAGS, About::default()).unwrap();
        let report = candidates.finish(|_| true).unwrap();
        assert_eq!(report.remaining, Remaining { own: 0, others: 2 });
        assert!(
            !report.is_complete(),
            "a run with an open upload failure is not complete"
        );
        let text = fs::read_to_string(dir.path().join(FAILURES_FILE)).unwrap();
        assert!(text.contains("\"command\":\"upload\"") && text.contains("written by hand"));
    }

    #[test]
    fn a_clean_run_leaves_an_empty_failures_file_and_a_complete_entry() {
        let dir = ScratchDir::new("run-clean");
        let mut session = Session::begin(dir.path(), "extract", &FLAGS, About::default()).unwrap();
        session.wrote();
        session.wrote();
        let report = session.finish(|_| true).unwrap();
        assert!(report.is_complete());
        assert_eq!(report.counts.get("written"), Some(&2));
        assert_eq!(fs::read_to_string(dir.path().join(FAILURES_FILE)).unwrap(), "");
        assert_eq!(manifest_entries(dir.path())[0]["ending"], "complete");
    }

    #[test]
    fn entries_of_other_tools_are_kept_byte_for_byte_and_each_command_adds_one() {
        let dir = ScratchDir::new("run-manifest");
        let foreign = r#"{"tool":"cb-eval","odd":   [1, 2.50, "x"]}"#;
        fs::write(
            dir.path().join(MANIFEST_FILE),
            format!("{{\"entries\": [{foreign}]}}"),
        )
        .unwrap();
        Session::begin(dir.path(), "upload", &FLAGS, About::default())
            .unwrap()
            .finish(|_| true)
            .unwrap();
        Session::begin(dir.path(), "candidates", &FLAGS, About::default())
            .unwrap()
            .finish(|_| true)
            .unwrap();
        let text = fs::read_to_string(dir.path().join(MANIFEST_FILE)).unwrap();
        assert!(text.contains(foreign), "{text}");
        let entries = manifest_entries(dir.path());
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1]["command"], "upload");
        assert_eq!(entries[2]["command"], "candidates");
        assert_eq!(entries[1]["tool"], "cb-run");
        assert_eq!(entries[1]["git_commit"], GIT_COMMIT);
        assert!(entries[1]["machine"]["os"].is_string());
    }

    #[test]
    fn a_command_that_never_ended_shows_as_running() {
        let dir = ScratchDir::new("run-killed");
        let session = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        let entries = manifest_entries(dir.path());
        assert_eq!(entries[0]["ending"], "running");
        assert!(entries[0]["ended_at"].is_null());
        drop(session);
    }

    #[test]
    fn a_manifest_of_another_shape_is_left_alone_and_nothing_runs() {
        let dir = ScratchDir::new("run-foreign-manifest");
        fs::write(dir.path().join(MANIFEST_FILE), "[1, 2, 3]").unwrap();
        let result = Session::begin(dir.path(), "upload", &FLAGS, About::default());
        assert!(matches!(result, Err(RunError::File { .. })));
        assert_eq!(
            fs::read_to_string(dir.path().join(MANIFEST_FILE)).unwrap(),
            "[1, 2, 3]"
        );
        assert!(!dir.path().join(FAILURES_FILE).exists());
    }

    #[test]
    fn two_commands_cannot_write_to_one_run_directory_at_once() {
        let dir = ScratchDir::new("run-lock");
        let first = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        let second = Session::begin(dir.path(), "upload", &FLAGS, About::default());
        assert!(matches!(second, Err(RunError::File { .. })));
        first.finish(|_| true).unwrap();
        // Once the first has ended, the directory is free again.
        assert!(Session::begin(dir.path(), "upload", &FLAGS, About::default()).is_ok());
    }

    #[test]
    fn a_cut_off_failure_line_is_dropped_when_the_file_is_read_again() {
        let dir = ScratchDir::new("run-torn-failures");
        let mut session = Session::begin(dir.path(), "upload", &FLAGS, About::default()).unwrap();
        session.fail("a", &timeout(), 3).unwrap();
        session.finish(|_| false).unwrap();
        let mut file = OpenOptions::new()
            .append(true)
            .open(dir.path().join(FAILURES_FILE))
            .unwrap();
        file.write_all(b"{\"command\":\"upload\",\"id\":\"b\",\"ki")
            .unwrap();
        drop(file);
        let report = Session::begin(dir.path(), "upload", &FLAGS, About::default())
            .unwrap()
            .finish(|_| false)
            .unwrap();
        assert_eq!(report.remaining.own, 1);
        assert_eq!(failure_lines(dir.path()).len(), 1);
    }

    #[test]
    fn a_stopped_command_says_so_in_its_entry() {
        let dir = ScratchDir::new("run-stopped");
        let session = Session::begin(dir.path(), "vectors", &FLAGS, About::default()).unwrap();
        session.stop(&timeout());
        let entries = manifest_entries(dir.path());
        assert_eq!(entries[0]["ending"], "stopped");
        assert!(entries[0]["ended_at"].is_string());
    }
}
