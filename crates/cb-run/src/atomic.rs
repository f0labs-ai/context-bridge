//! Whole files written so that a reader sees the old content or the new, never a part.

use crate::error::RunError;
use sha2::{Digest, Sha256};
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

/// The name a file is written under before it takes its own: beside it, so the rename
/// stays on one file system.
fn temporary_beside(path: &Path) -> PathBuf {
    let mut name = OsString::from(".");
    name.push(path.file_name().unwrap_or_default());
    name.push(format!(".tmp-{}", std::process::id()));
    path.with_file_name(name)
}

/// Writes `path` through a temporary file beside it: filled, flushed to disk, then renamed
/// over the old file. If filling fails, the old file is untouched and the temporary one removed.
pub fn write_atomically(
    path: &Path,
    fill: impl FnOnce(&mut BufWriter<File>) -> std::io::Result<()>,
) -> Result<(), RunError> {
    let temporary = temporary_beside(path);
    let written = (|| {
        let mut writer = BufWriter::new(File::create(&temporary)?);
        fill(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written.map_err(|source| RunError::io(path, source))
}

/// Writes one JSON value, indented, with a final newline, atomically.
pub fn write_json_atomically<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), RunError> {
    write_atomically(path, |writer| {
        serde_json::to_writer_pretty(&mut *writer, value).map_err(std::io::Error::other)?;
        writer.write_all(b"\n")
    })
}

/// The sha256 of a file, as lower-case hex.
pub fn sha256_of_file(path: &Path) -> Result<String, RunError> {
    let mut hasher = Sha256::new();
    let mut file = BufReader::new(File::open(path).map_err(|e| RunError::io(path, e))?);
    std::io::copy(&mut file, &mut hasher).map_err(|e| RunError::io(path, e))?;
    Ok(hex(&hasher.finalize()))
}

/// The sha256 of some bytes, as lower-case hex.
pub fn sha256_of_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ScratchDir;

    #[test]
    fn a_failed_write_leaves_the_old_file_and_no_temporary_one() {
        let dir = ScratchDir::new("atomic-failed");
        let path = dir.path().join("vectors.json");
        fs::write(&path, b"old").unwrap();
        let result = write_atomically(&path, |writer| {
            writer.write_all(b"half of the new")?;
            Err(std::io::Error::other("disk full"))
        });
        assert!(matches!(result, Err(RunError::Io { .. })));
        assert_eq!(fs::read(&path).unwrap(), b"old");
        let left: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .collect();
        assert_eq!(left, vec![OsString::from("vectors.json")]);
    }

    #[test]
    fn a_finished_write_replaces_the_file() {
        let dir = ScratchDir::new("atomic-finished");
        let path = dir.path().join("manifest.json");
        fs::write(&path, b"old").unwrap();
        write_json_atomically(&path, &serde_json::json!({"entries": []})).unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "{\n  \"entries\": []\n}\n");
    }

    #[test]
    fn sha256_matches_a_known_value() {
        assert_eq!(
            sha256_of_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let dir = ScratchDir::new("atomic-sha");
        let path = dir.path().join("f");
        fs::write(&path, b"abc").unwrap();
        assert_eq!(sha256_of_file(&path).unwrap(), sha256_of_bytes(b"abc"));
    }
}
