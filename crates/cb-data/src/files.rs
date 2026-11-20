//! Writing and reading a data set directory: the JSON Lines files and their manifest.

use cb_types::{Manifest, Query, Record, Transcript, read_jsonl, write_jsonl};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

pub const RECORDS: &str = "records.jsonl";
pub const QUERIES: &str = "queries.jsonl";
pub const TRANSCRIPTS: &str = "transcripts.jsonl";
pub const MANIFEST: &str = "manifest.json";

/// What a data set holds in memory.
#[derive(Debug, Default)]
pub struct DataSet {
    pub records: Vec<Record>,
    pub queries: Vec<Query>,
    pub transcripts: Vec<Transcript>,
}

/// What the manifest says beyond counts and hashes.
#[derive(Debug, Clone)]
pub struct Provenance {
    pub name: String,
    pub made_by: String,
    pub seed: Option<u64>,
    pub sources: Vec<String>,
}

fn sha256_of(path: &Path) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    let mut file = BufReader::new(File::open(path)?);
    std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn write_file<T: serde::Serialize>(dir: &Path, name: &str, items: &[T]) -> anyhow::Result<()> {
    let mut w = BufWriter::new(File::create(dir.join(name))?);
    write_jsonl(&mut w, items)?;
    w.flush()?;
    Ok(())
}

/// Write `data` under `dir` (created if absent) with a manifest of counts and hashes.
/// `extra_files` are files already written into `dir` (a second transcript file, say)
/// that the manifest should hash too.
pub fn write_data_set(
    dir: &Path,
    data: &DataSet,
    provenance: &Provenance,
    extra_files: &[&str],
) -> anyhow::Result<Manifest> {
    std::fs::create_dir_all(dir)?;
    write_file(dir, RECORDS, &data.records)?;
    write_file(dir, QUERIES, &data.queries)?;
    let mut names = vec![RECORDS, QUERIES];
    if !data.transcripts.is_empty() {
        write_file(dir, TRANSCRIPTS, &data.transcripts)?;
        names.push(TRANSCRIPTS);
    }
    names.extend_from_slice(extra_files);
    let mut sha256 = BTreeMap::new();
    for name in names {
        sha256.insert(name.to_string(), sha256_of(&dir.join(name))?);
    }
    let mut counts = BTreeMap::new();
    counts.insert("records".to_string(), data.records.len() as u64);
    counts.insert("queries".to_string(), data.queries.len() as u64);
    counts.insert("transcripts".to_string(), data.transcripts.len() as u64);
    let manifest = Manifest {
        name: provenance.name.clone(),
        made_by: provenance.made_by.clone(),
        seed: provenance.seed,
        counts,
        sha256,
        sources: provenance.sources.clone(),
    };
    let mut w = BufWriter::new(File::create(dir.join(MANIFEST))?);
    serde_json::to_writer_pretty(&mut w, &manifest)?;
    w.write_all(b"\n")?;
    w.flush()?;
    Ok(manifest)
}

/// Read a data set directory. Transcripts are optional.
pub fn read_data_set(dir: &Path) -> anyhow::Result<DataSet> {
    let read =
        |name: &str| -> anyhow::Result<BufReader<File>> { Ok(BufReader::new(File::open(dir.join(name))?)) };
    let transcripts = if dir.join(TRANSCRIPTS).exists() {
        read_jsonl(read(TRANSCRIPTS)?)?
    } else {
        Vec::new()
    };
    Ok(DataSet {
        records: read_jsonl(read(RECORDS)?)?,
        queries: read_jsonl(read(QUERIES)?)?,
        transcripts,
    })
}

/// The manifest of a data set directory.
pub fn read_manifest(dir: &Path) -> anyhow::Result<Manifest> {
    Ok(serde_json::from_reader(BufReader::new(File::open(
        dir.join(MANIFEST),
    )?))?)
}

/// Whether every file the manifest names still has the hash it recorded.
pub fn hashes_match(dir: &Path, manifest: &Manifest) -> anyhow::Result<bool> {
    for (name, expected) in &manifest.sha256 {
        if &sha256_of(&dir.join(name))? != expected {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::{Gold, QueryClass, Tier};

    #[test]
    fn a_data_set_reads_back_as_written_and_its_hashes_hold() {
        let dir = std::env::temp_dir().join(format!("cb-data-files-test-{}", std::process::id()));
        let data = DataSet {
            records: vec![Record {
                id: "r-1".into(),
                text: "Fixed.".into(),
                written_at: 10,
                valid_until: None,
                thread: Some("inc-1".into()),
                tier: Tier::Changing,
                title: None,
                source: "test".into(),
                truth: None,
            }],
            queries: vec![Query {
                id: "q-1".into(),
                text: "Is it fixed?".into(),
                asked_at: 20,
                class: QueryClass::Changing,
                gold: vec![Gold {
                    record: "r-1".into(),
                    grade: 3,
                }],
                thread: Some("inc-1".into()),
                harmful: vec![],
            }],
            transcripts: vec![],
        };
        let provenance = Provenance {
            name: "t".into(),
            made_by: "test".into(),
            seed: Some(1),
            sources: vec![],
        };
        let manifest = write_data_set(&dir, &data, &provenance, &[]).unwrap();
        assert_eq!(manifest.counts.get("records"), Some(&1));
        assert!(!dir.join(TRANSCRIPTS).exists());
        let back = read_data_set(&dir).unwrap();
        assert_eq!(back.records, data.records);
        assert_eq!(back.queries, data.queries);
        assert!(hashes_match(&dir, &read_manifest(&dir).unwrap()).unwrap());
        std::fs::write(dir.join(RECORDS), b"").unwrap();
        assert!(!hashes_match(&dir, &manifest).unwrap());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
