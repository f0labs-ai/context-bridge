//! Run files that are appended to and picked up again.
//!
//! A line is written whole, in one write, with its newline. So after a crash the only
//! damage a file can have is one cut-off last line. Reading a file back repairs exactly
//! that and nothing else: a broken line anywhere before the end is an error, never skipped.

use crate::error::RunError;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// What a run file already holds.
#[derive(Debug)]
pub struct Existing<T> {
    pub lines: Vec<T>,
    /// Bytes of a cut-off last line that were removed, if a crash left one.
    pub cut_off_bytes: u64,
}

impl<T> Default for Existing<T> {
    fn default() -> Self {
        Self {
            lines: Vec::new(),
            cut_off_bytes: 0,
        }
    }
}

/// What to do with a last line that has no newline and does not parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutOffLine {
    /// Remove it: the file is this tool's own, and the item will be done again.
    Remove,
    /// Refuse to go on: the file is someone else's data.
    Refuse,
}

/// Reads every line of `path` (none when the file does not exist yet), one value per
/// non-empty line, and leaves the file ending in a newline so it can be appended to.
pub fn read_resumable<T: DeserializeOwned>(
    path: &Path,
    cut_off: CutOffLine,
) -> Result<Existing<T>, RunError> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Existing::default()),
        Err(e) => return Err(RunError::io(path, e)),
    };
    let mut reader = BufReader::new(file);
    let mut lines = Vec::new();
    let mut buffer = Vec::new();
    let mut offset: u64 = 0;
    let mut line_number = 0usize;
    let mut cut_at: Option<u64> = None;
    let mut lacks_newline = false;
    loop {
        buffer.clear();
        let read = reader
            .read_until(b'\n', &mut buffer)
            .map_err(|e| RunError::io(path, e))?;
        if read == 0 {
            break;
        }
        line_number += 1;
        let ended = buffer.last() == Some(&b'\n');
        let text = buffer.trim_ascii();
        if !text.is_empty() {
            match serde_json::from_slice::<T>(text) {
                Ok(value) => {
                    lines.push(value);
                    lacks_newline = !ended;
                }
                Err(e) if ended => {
                    return Err(RunError::Line {
                        path: path.to_path_buf(),
                        line: line_number,
                        reason: e.to_string(),
                    });
                }
                Err(e) => match cut_off {
                    CutOffLine::Remove => cut_at = Some(offset),
                    CutOffLine::Refuse => {
                        return Err(RunError::Line {
                            path: path.to_path_buf(),
                            line: line_number,
                            reason: format!("the last line is cut off ({e})"),
                        });
                    }
                },
            }
        }
        offset += read as u64;
    }
    drop(reader);
    let mut cut_off_bytes = 0;
    if let Some(keep) = cut_at {
        cut_off_bytes = offset - keep;
        let file = OpenOptions::new()
            .write(true)
            .open(path)
            .map_err(|e| RunError::io(path, e))?;
        file.set_len(keep).map_err(|e| RunError::io(path, e))?;
        file.sync_all().map_err(|e| RunError::io(path, e))?;
    } else if lacks_newline {
        let mut file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| RunError::io(path, e))?;
        file.write_all(b"\n").map_err(|e| RunError::io(path, e))?;
    }
    Ok(Existing { lines, cut_off_bytes })
}

/// Reads a whole JSON Lines file that must exist and be sound.
pub fn read_all<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, RunError> {
    let file = File::open(path).map_err(|e| RunError::io(path, e))?;
    cb_types::read_jsonl(BufReader::new(file)).map_err(|e| match e {
        cb_types::ContractError::Line { line, source } => RunError::Line {
            path: path.to_path_buf(),
            line,
            reason: source.to_string(),
        },
        cb_types::ContractError::Io(source) => RunError::io(path, source),
        cb_types::ContractError::Invalid(reason) => RunError::file(path, reason),
    })
}

/// An output file that only grows.
#[derive(Debug)]
pub struct Appender {
    path: PathBuf,
    file: File,
}

impl Appender {
    /// Opens `path` for appending, creating it when absent. Call [`read_resumable`] first,
    /// so the file is known to end in a newline.
    pub fn open(path: &Path) -> Result<Self, RunError> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| RunError::io(path, e))?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Appends `items`, one per line, in ONE write: either all of them reach the file or
    /// (after a crash) a cut-off tail that the next read removes.
    pub fn append<T: Serialize>(&mut self, items: &[T]) -> Result<(), RunError> {
        if items.is_empty() {
            return Ok(());
        }
        let mut bytes = Vec::new();
        for item in items {
            serde_json::to_writer(&mut bytes, item).map_err(|e| RunError::file(&self.path, e.to_string()))?;
            bytes.push(b'\n');
        }
        self.file
            .write_all(&bytes)
            .map_err(|e| RunError::io(&self.path, e))
    }

    /// Asks the operating system to put what was appended on disk.
    pub fn sync(&mut self) -> Result<(), RunError> {
        self.file.sync_data().map_err(|e| RunError::io(&self.path, e))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ScratchDir;
    use cb_types::Scored;
    use std::fs;

    fn scored(record: &str) -> Scored {
        Scored {
            record: record.into(),
            score: 0.5,
        }
    }

    #[test]
    fn what_was_appended_is_read_back_in_order() {
        let dir = ScratchDir::new("jsonl-roundtrip");
        let path = dir.path().join("out.jsonl");
        assert!(
            read_resumable::<Scored>(&path, CutOffLine::Remove)
                .unwrap()
                .lines
                .is_empty()
        );
        let mut out = Appender::open(&path).unwrap();
        out.append(&[scored("a"), scored("b")]).unwrap();
        out.append(&[scored("c")]).unwrap();
        out.sync().unwrap();
        let back = read_resumable::<Scored>(&path, CutOffLine::Remove).unwrap();
        assert_eq!(back.lines, vec![scored("a"), scored("b"), scored("c")]);
        assert_eq!(back.cut_off_bytes, 0);
    }

    #[test]
    fn a_cut_off_last_line_is_removed_and_the_rest_kept() {
        let dir = ScratchDir::new("jsonl-torn");
        let path = dir.path().join("out.jsonl");
        let cut_off = "{\"record\":\"b\",\"sco";
        fs::write(&path, format!("{{\"record\":\"a\",\"score\":0.5}}\n{cut_off}")).unwrap();
        let back = read_resumable::<Scored>(&path, CutOffLine::Remove).unwrap();
        assert_eq!(back.lines, vec![scored("a")]);
        assert_eq!(back.cut_off_bytes, cut_off.len() as u64);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "{\"record\":\"a\",\"score\":0.5}\n"
        );
        // The next append starts on a line of its own.
        Appender::open(&path).unwrap().append(&[scored("b")]).unwrap();
        let again = read_resumable::<Scored>(&path, CutOffLine::Remove).unwrap();
        assert_eq!(again.lines, vec![scored("a"), scored("b")]);
    }

    #[test]
    fn a_whole_last_line_without_its_newline_is_kept_and_given_one() {
        let dir = ScratchDir::new("jsonl-no-newline");
        let path = dir.path().join("out.jsonl");
        fs::write(&path, "{\"record\":\"a\",\"score\":0.5}").unwrap();
        let back = read_resumable::<Scored>(&path, CutOffLine::Remove).unwrap();
        assert_eq!(back.lines, vec![scored("a")]);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "{\"record\":\"a\",\"score\":0.5}\n"
        );
    }

    #[test]
    fn a_broken_line_before_the_end_is_an_error_never_skipped() {
        let dir = ScratchDir::new("jsonl-broken");
        let path = dir.path().join("out.jsonl");
        fs::write(
            &path,
            "{\"record\":\"a\",\"score\":0.5}\nnot json\n{\"record\":\"c\",\"score\":0.5}\n",
        )
        .unwrap();
        let error = read_resumable::<Scored>(&path, CutOffLine::Remove).unwrap_err();
        assert!(matches!(error, RunError::Line { line: 2, .. }), "{error}");
        // Nothing was changed.
        assert!(fs::read_to_string(&path).unwrap().contains("not json"));
    }

    #[test]
    fn someone_elses_file_with_a_cut_off_line_is_refused_untouched() {
        let dir = ScratchDir::new("jsonl-refuse");
        let path = dir.path().join("queries.jsonl");
        let content = "{\"record\":\"a\",\"score\":0.5}\n{\"record\":\"b\"";
        fs::write(&path, content).unwrap();
        let error = read_resumable::<Scored>(&path, CutOffLine::Refuse).unwrap_err();
        assert!(matches!(error, RunError::Line { line: 2, .. }), "{error}");
        assert_eq!(fs::read_to_string(&path).unwrap(), content);
    }

    #[test]
    fn blank_lines_are_passed_over() {
        let dir = ScratchDir::new("jsonl-blank");
        let path = dir.path().join("out.jsonl");
        fs::write(&path, "\n{\"record\":\"a\",\"score\":0.5}\n\n").unwrap();
        let back = read_resumable::<Scored>(&path, CutOffLine::Remove).unwrap();
        assert_eq!(back.lines, vec![scored("a")]);
    }
}
