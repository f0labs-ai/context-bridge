//! Reading the files of a data set and a run, and writing this crate's own.
//!
//! Every file this crate writes is written beside its final name and moved
//! into place when complete, so a file that exists is whole.

use crate::error::EvalError;
use crate::vectors::{VectorIndex, Vectors};
use cb_types::{Candidates, read_jsonl, write_jsonl};
use serde::Serialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const RECORDS: &str = "records.jsonl";
pub const QUERIES: &str = "queries.jsonl";
pub const TRANSCRIPTS: &str = "transcripts.jsonl";
pub const CANDIDATES: &str = "candidates.jsonl";
/// Each question's newest records and their relevance (`cb-run newest`).
pub const NEWEST: &str = "newest.jsonl";
/// Passage vectors of the records: `vectors.f32` and `vectors.json`.
pub const VECTORS: &str = "vectors";
pub const EXTRACTIONS: &str = "extractions.jsonl";
/// One vector per line of `extractions.jsonl`: `extraction_vectors.f32` and `.json`.
pub const EXTRACTION_VECTORS: &str = "extraction_vectors";
pub const TRIPS: &str = "trips.jsonl";
pub const RANKINGS_DIR: &str = "rankings";
pub const MEASURES_DIR: &str = "measures";
pub const TABLES_DIR: &str = "tables";
pub const FIGURES_DIR: &str = "figures";
pub const TUNED: &str = "tuned.json";
pub const SUMMARY: &str = "summary.json";
pub const NUMBERS: &str = "numbers.tex";

/// This tool and its version, as recorded in the files it writes.
pub fn made_by() -> String {
    format!("cb-eval {}", env!("CARGO_PKG_VERSION"))
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> EvalError + '_ {
    move |source| EvalError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Read one value per non-empty line.
pub fn read_jsonl_file<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>, EvalError> {
    let file = File::open(path).map_err(io(path))?;
    read_jsonl(BufReader::new(file)).map_err(|source| EvalError::Contract {
        path: path.to_path_buf(),
        source,
    })
}

pub fn read_json_file<T: DeserializeOwned>(path: &Path) -> Result<T, EvalError> {
    let file = File::open(path).map_err(io(path))?;
    serde_json::from_reader(BufReader::new(file)).map_err(|source| EvalError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// The name a file is written under before it is moved into place.
fn unfinished(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    path.with_file_name(name)
}

/// Write a file through `fill`, creating its directory, and move it into place when complete.
pub fn write_file(
    path: &Path,
    fill: impl FnOnce(&mut BufWriter<File>) -> Result<(), EvalError>,
) -> Result<(), EvalError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(io(parent))?;
    }
    let part = unfinished(path);
    let mut writer = BufWriter::new(File::create(&part).map_err(io(&part))?);
    fill(&mut writer)?;
    writer.flush().map_err(io(&part))?;
    drop(writer);
    fs::rename(&part, path).map_err(io(path))
}

pub fn write_jsonl_file<T: Serialize>(path: &Path, items: &[T]) -> Result<(), EvalError> {
    write_file(path, |writer| {
        write_jsonl(writer, items).map_err(|source| EvalError::Contract {
            path: path.to_path_buf(),
            source,
        })
    })
}

/// Pretty JSON with a final newline.
pub fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<(), EvalError> {
    write_file(path, |writer| {
        serde_json::to_writer_pretty(&mut *writer, value).map_err(|source| EvalError::Json {
            path: path.to_path_buf(),
            source,
        })?;
        writer.write_all(b"\n").map_err(io(path))
    })
}

pub fn write_text_file(path: &Path, text: &str) -> Result<(), EvalError> {
    write_file(path, |writer| writer.write_all(text.as_bytes()).map_err(io(path)))
}

/// The SHA-256 of a file, in hex.
pub fn sha256_of_file(path: &Path) -> Result<String, EvalError> {
    let mut hasher = Sha256::new();
    let mut file = BufReader::new(File::open(path).map_err(io(path))?);
    std::io::copy(&mut file, &mut hasher).map_err(io(path))?;
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// Read `<stem>.json` and `<stem>.f32`.
pub fn read_vectors(stem: &Path) -> Result<Vectors, EvalError> {
    let index: VectorIndex = read_json_file(&stem.with_extension("json"))?;
    let values = stem.with_extension("f32");
    let mut bytes = Vec::new();
    File::open(&values)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(io(&values))?;
    Vectors::from_le_bytes(index, &bytes)
        .map_err(|error| EvalError::invalid(format!("{}: {error}", values.display())))
}

/// Write `<stem>.json` and `<stem>.f32` (used by tests and by anyone preparing a run by hand).
pub fn write_vectors(stem: &Path, index: &VectorIndex, values: &[f32]) -> Result<(), EvalError> {
    write_json_file(&stem.with_extension("json"), index)?;
    let path = stem.with_extension("f32");
    write_file(&path, |writer| {
        for value in values {
            writer.write_all(&value.to_le_bytes()).map_err(io(&path))?;
        }
        Ok(())
    })
}

/// `candidates.jsonl` opened for reading one question at a time.
///
/// The file holds a line per question and view and can be far larger than
/// memory should be asked to hold, so opening it only notes where each
/// question's lines are; [`CandidateFile::lines_of`] reads them when asked.
#[derive(Debug)]
pub struct CandidateFile {
    path: PathBuf,
    file: File,
    /// For each question, the offset and length of each of its lines, in file order.
    lines: HashMap<String, Vec<(u64, usize)>>,
}

impl CandidateFile {
    pub fn open(path: &Path) -> Result<Self, EvalError> {
        #[derive(serde::Deserialize)]
        struct QueryOf {
            query: String,
        }
        let mut reader = BufReader::new(File::open(path).map_err(io(path))?);
        let mut lines: HashMap<String, Vec<(u64, usize)>> = HashMap::new();
        let mut line = Vec::new();
        let mut offset = 0u64;
        let mut number = 0usize;
        loop {
            line.clear();
            let read = reader.read_until(b'\n', &mut line).map_err(io(path))?;
            if read == 0 {
                break;
            }
            number += 1;
            if !line.iter().all(u8::is_ascii_whitespace) {
                let QueryOf { query } =
                    serde_json::from_slice(&line).map_err(|source| EvalError::Contract {
                        path: path.to_path_buf(),
                        source: cb_types::ContractError::Line { line: number, source },
                    })?;
                lines.entry(query).or_default().push((offset, read));
            }
            offset += read as u64;
        }
        Ok(Self {
            path: path.to_path_buf(),
            file: File::open(path).map_err(io(path))?,
            lines,
        })
    }

    /// The questions the file has lines for, in no particular order.
    pub fn questions(&self) -> impl Iterator<Item = &str> {
        self.lines.keys().map(String::as_str)
    }

    /// Every line of one question, in file order. Empty when the file has none.
    pub fn lines_of(&mut self, query: &str) -> Result<Vec<Candidates>, EvalError> {
        let Some(places) = self.lines.get(query) else {
            return Ok(Vec::new());
        };
        let mut out = Vec::with_capacity(places.len());
        let mut buffer = Vec::new();
        for (offset, length) in places {
            buffer.resize(*length, 0);
            self.file.seek(SeekFrom::Start(*offset)).map_err(io(&self.path))?;
            self.file.read_exact(&mut buffer).map_err(io(&self.path))?;
            out.push(serde_json::from_slice(&buffer).map_err(|source| EvalError::Json {
                path: self.path.clone(),
                source,
            })?);
        }
        Ok(out)
    }
}

/// A file of lines that is added to a little at a time without staying open,
/// so hundreds can be written side by side. Moved into place by [`LineFile::finish`].
#[derive(Debug)]
pub struct LineFile {
    path: PathBuf,
    pending: Vec<u8>,
    started: bool,
}

impl LineFile {
    /// How much is held before it is written out.
    const HOLD: usize = 256 * 1024;

    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            pending: Vec::new(),
            started: false,
        }
    }

    pub fn push<T: Serialize>(&mut self, item: &T) -> Result<(), EvalError> {
        serde_json::to_writer(&mut self.pending, item).map_err(|source| EvalError::Json {
            path: self.path.clone(),
            source,
        })?;
        self.pending.push(b'\n');
        if self.pending.len() >= Self::HOLD {
            self.write_out()?;
        }
        Ok(())
    }

    fn write_out(&mut self) -> Result<(), EvalError> {
        let part = unfinished(&self.path);
        let mut options = fs::OpenOptions::new();
        if self.started {
            options.append(true);
        } else {
            if let Some(parent) = self.path.parent() {
                fs::create_dir_all(parent).map_err(io(parent))?;
            }
            options.write(true).create(true).truncate(true);
        }
        options
            .open(&part)
            .and_then(|mut file| file.write_all(&self.pending))
            .map_err(io(&part))?;
        self.started = true;
        self.pending.clear();
        Ok(())
    }

    pub fn finish(mut self) -> Result<(), EvalError> {
        self.write_out()?;
        fs::rename(unfinished(&self.path), &self.path).map_err(io(&self.path))
    }
}

#[cfg(test)]
pub(crate) fn test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("cb-eval-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{candidates, scored};
    use cb_types::Scored;

    #[test]
    fn files_read_back_as_written_and_leave_nothing_unfinished() {
        let dir = test_dir("files");
        let items = scored(&[("a", 0.5), ("b", 0.25)]);
        let lines = dir.join("deep/er/items.jsonl");
        write_jsonl_file(&lines, &items).unwrap();
        assert_eq!(read_jsonl_file::<Scored>(&lines).unwrap(), items);
        let json = dir.join("value.json");
        write_json_file(&json, &items).unwrap();
        assert_eq!(read_json_file::<Vec<Scored>>(&json).unwrap(), items);
        assert!(fs::read_to_string(&json).unwrap().ends_with("]\n"));
        write_text_file(&dir.join("t.txt"), "abc").unwrap();
        // sha256("abc")
        assert_eq!(
            sha256_of_file(&dir.join("t.txt")).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let unfinished: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".part"))
            .collect();
        assert!(unfinished.is_empty());
        assert!(matches!(
            read_json_file::<Vec<Scored>>(&dir.join("absent.json")),
            Err(EvalError::Io { .. })
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn vectors_read_back_as_written() {
        let dir = test_dir("vectors");
        let index = VectorIndex {
            dim: 2,
            ids: vec!["a".into(), "b".into()],
        };
        write_vectors(&dir.join("vectors"), &index, &[1.0, 0.0, 0.5, -2.0]).unwrap();
        assert_eq!(fs::metadata(dir.join("vectors.f32")).unwrap().len(), 16);
        let vectors = read_vectors(&dir.join("vectors")).unwrap();
        assert_eq!((vectors.dim(), vectors.len()), (2, 2));
        assert_eq!(vectors.row(1), [0.5, -2.0]);
        fs::write(dir.join("vectors.f32"), [0u8; 12]).unwrap();
        assert!(read_vectors(&dir.join("vectors")).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn candidate_lines_are_found_by_question_wherever_they_are_in_the_file() {
        let dir = test_dir("candidates");
        let path = dir.join(CANDIDATES);
        let none = scored(&[]);
        let lines = vec![
            candidates("q1", "now", 10, &scored(&[("a", 0.5)]), &none, &none),
            candidates("q2", "now", 20, &none, &none, &none),
            candidates("q1", "batch_300", 10, &none, &scored(&[("b", 2.0)]), &none),
        ];
        write_jsonl_file(&path, &lines).unwrap();
        // A blank line in the middle is skipped, as cb_types::read_jsonl skips it.
        let text = fs::read_to_string(&path).unwrap().replacen('\n', "\n\n", 1);
        fs::write(&path, text).unwrap();
        let mut file = CandidateFile::open(&path).unwrap();
        let mut questions: Vec<&str> = file.questions().collect();
        questions.sort();
        assert_eq!(questions, ["q1", "q2"]);
        assert_eq!(file.lines_of("q1").unwrap(), [lines[0].clone(), lines[2].clone()]);
        assert_eq!(file.lines_of("q2").unwrap(), [lines[1].clone()]);
        assert!(file.lines_of("q3").unwrap().is_empty());
        fs::write(&path, "{\"query\":\"q1\"}\nnot json\n").unwrap();
        assert!(matches!(
            CandidateFile::open(&path),
            Err(EvalError::Contract {
                source: cb_types::ContractError::Line { line: 2, .. },
                ..
            })
        ));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_line_file_holds_every_line_in_order_however_often_it_was_written_out() {
        let dir = test_dir("lines");
        let path = dir.join("out/many.jsonl");
        let mut file = LineFile::new(path.clone());
        let item = |i: usize| Scored {
            record: format!("record-{i:06}-{}", "x".repeat(40)),
            score: i as f64,
        };
        // About 70 bytes a line: 10 000 lines cross the 256 KiB hold twice.
        for i in 0..10_000 {
            file.push(&item(i)).unwrap();
        }
        assert!(!path.exists());
        file.finish().unwrap();
        let back: Vec<Scored> = read_jsonl_file(&path).unwrap();
        assert_eq!(back.len(), 10_000);
        assert!(back.iter().enumerate().all(|(i, s)| *s == item(i)));
        // Writing the same path again replaces the file.
        let mut again = LineFile::new(path.clone());
        again.push(&item(7)).unwrap();
        again.finish().unwrap();
        assert_eq!(read_jsonl_file::<Scored>(&path).unwrap(), [item(7)]);
        // A file with no lines still exists, empty.
        let empty = dir.join("out/empty.jsonl");
        LineFile::new(empty.clone()).finish().unwrap();
        assert_eq!(fs::metadata(&empty).unwrap().len(), 0);
        fs::remove_dir_all(&dir).unwrap();
    }
}
