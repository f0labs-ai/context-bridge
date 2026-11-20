//! Archives kept on Zenodo: one pinned file per record.
//!
//! A Zenodo record is fixed once published, so the loader names the exact file
//! and its SHA-256. The download step makes one request for it; the convert
//! step checks the hash again before reading, so a damaged or swapped archive
//! is never converted. Zenodo asks crawlers to wait ten seconds between
//! requests and to stay out of its API except for record files
//! (`robots.txt`); the file URL below is one of those.

use super::fetch::{FetchError, Fetcher, Outcome, Politeness};
use sha2::{Digest, Sha256};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

/// One file of a Zenodo record.
#[derive(Clone, Copy, Debug)]
pub struct ZenodoFile {
    /// The record's DOI.
    pub doi: &'static str,
    /// The record's page.
    pub page: &'static str,
    pub title: &'static str,
    pub author: &'static str,
    pub published: &'static str,
    /// Where the file is downloaded from.
    pub url: &'static str,
    /// The name it is kept under in `--raw`.
    pub file_name: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// The licence line of both records, as their Zenodo pages print it.
pub const LICENCE_NAME: &str = "Creative Commons Attribution 4.0 International";
pub const LICENCE_SENTENCE: &str = "The Creative Commons Attribution license allows re-distribution and re-use \
                                    of a licensed work on the condition that the creator is appropriately credited.";
pub const LICENCE_URL: &str = "https://creativecommons.org/licenses/by/4.0/legalcode";

#[derive(Debug, Error)]
pub enum ZenodoError {
    #[error(transparent)]
    Fetch(#[from] FetchError),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error(
        "{path}: this is not the file of {doi}: it has {found_bytes} bytes and SHA-256 {found_sha256}, \
         the record's file has {bytes} bytes and SHA-256 {sha256}"
    )]
    NotTheFile {
        path: PathBuf,
        doi: &'static str,
        found_bytes: u64,
        found_sha256: String,
        bytes: u64,
        sha256: &'static str,
    },
    #[error("{path}: not a zip archive: {source}")]
    Zip {
        path: PathBuf,
        #[source]
        source: zip::result::ZipError,
    },
    #[error("{archive}: no member named {member:?}")]
    NoMember { archive: PathBuf, member: String },
    #[error("{archive}: member {member:?}: {source}")]
    Member {
        archive: PathBuf,
        member: String,
        #[source]
        source: std::io::Error,
    },
}

impl ZenodoFile {
    pub fn path(&self, raw: &Path) -> PathBuf {
        raw.join(self.file_name)
    }

    pub fn allowed_urls(&self) -> Vec<String> {
        vec![self.url.to_string()]
    }

    /// The record, its author and its licence, for a manifest.
    pub fn sources(&self) -> Vec<String> {
        vec![
            format!(
                "{}. \"{}\". Zenodo, {}. https://doi.org/{} (file {}, {} bytes, sha256 {}).",
                self.author, self.title, self.published, self.doi, self.file_name, self.bytes, self.sha256
            ),
            format!(
                "Licence, as stated on {}: \"{LICENCE_NAME}\" - \"{LICENCE_SENTENCE}\" ({LICENCE_URL}). \
                 The licence is the compiler's; the incident texts were written by the service providers \
                 on their public status pages.",
                self.page
            ),
        ]
    }

    /// The archive's bytes, after checking that they are the record's file.
    pub fn read(&self, raw: &Path) -> Result<Vec<u8>, ZenodoError> {
        let path = self.path(raw);
        let bytes = std::fs::read(&path).map_err(|source| ZenodoError::Io {
            path: path.clone(),
            source,
        })?;
        let found_sha256: String = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if bytes.len() as u64 != self.bytes || found_sha256 != self.sha256 {
            return Err(ZenodoError::NotTheFile {
                path,
                doi: self.doi,
                found_bytes: bytes.len() as u64,
                found_sha256,
                bytes: self.bytes,
                sha256: self.sha256,
            });
        }
        Ok(bytes)
    }

    /// The archive, opened, after checking that it is the record's file.
    pub fn open(&self, raw: &Path) -> Result<Archive, ZenodoError> {
        Archive::from_bytes(self.path(raw), self.read(raw)?)
    }

    /// Download the file unless it is already there, then check it.
    pub async fn download(&self, raw: &Path, fetcher: &mut Fetcher) -> Result<Outcome, ZenodoError> {
        let outcome = fetcher.get_to_file(self.url, &self.path(raw)).await?;
        self.read(raw)?;
        Ok(outcome)
    }
}

/// Zenodo's `robots.txt` asks for ten seconds between requests.
pub fn politeness() -> Politeness {
    Politeness {
        min_interval: Duration::from_secs(10),
        timeout: Duration::from_secs(300),
        ..Politeness::default()
    }
}

/// An opened archive.
pub struct Archive {
    path: PathBuf,
    zip: zip::ZipArchive<Cursor<Vec<u8>>>,
}

impl Archive {
    /// A zip archive held in memory; `path` names it in errors.
    pub fn from_bytes(path: PathBuf, bytes: Vec<u8>) -> Result<Self, ZenodoError> {
        let zip = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|source| ZenodoError::Zip {
            path: path.clone(),
            source,
        })?;
        Ok(Self { path, zip })
    }

    /// The names of the files in the archive, sorted.
    pub fn names(&self) -> Result<Vec<String>, ZenodoError> {
        let mut names = Vec::new();
        for name in self.zip.file_names() {
            let name = name.map_err(|source| ZenodoError::Zip {
                path: self.path.clone(),
                source,
            })?;
            names.push(name.into_owned());
        }
        names.sort();
        Ok(names)
    }

    /// One member, whole.
    pub fn member(&mut self, name: &str) -> Result<Vec<u8>, ZenodoError> {
        let mut file = self.zip.by_name(name).map_err(|_| ZenodoError::NoMember {
            archive: self.path.clone(),
            member: name.to_string(),
        })?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|source| ZenodoError::Member {
                archive: self.path.clone(),
                member: name.to_string(),
                source,
            })?;
        Ok(bytes)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    //! Small archives for tests.
    use std::io::{Cursor, Write};
    use zip::write::SimpleFileOptions;

    /// A zip holding the given members.
    pub fn zip_of(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, bytes) in members {
            writer.start_file(*name, SimpleFileOptions::default()).unwrap();
            writer.write_all(bytes).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cb-data-zenodo-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn pinned(bytes: &[u8]) -> ZenodoFile {
        let sha256: String = Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect();
        ZenodoFile {
            doi: "10.5281/zenodo.0",
            page: "https://zenodo.org/records/0",
            title: "Test",
            author: "Nobody",
            published: "2024-01-01",
            url: "https://zenodo.org/api/records/0/files/data.zip/content",
            file_name: "data.zip",
            bytes: bytes.len() as u64,
            sha256: Box::leak(sha256.into_boxed_str()),
        }
    }

    #[test]
    fn an_archive_is_read_only_when_it_is_the_pinned_file() {
        let bytes = testing::zip_of(&[("a/one.csv", b"x,y\n1,2\n"), ("b.txt", b"hello")]);
        let file = pinned(&bytes);
        let dir = temp_dir("pinned");
        std::fs::write(file.path(&dir), &bytes).unwrap();

        let mut archive = file.open(&dir).unwrap();
        assert_eq!(archive.names().unwrap(), ["a/one.csv", "b.txt"]);
        assert_eq!(archive.member("a/one.csv").unwrap(), b"x,y\n1,2\n");
        assert!(matches!(
            archive.member("missing"),
            Err(ZenodoError::NoMember { .. })
        ));

        // One byte different: refused, and the error says what was found.
        let mut damaged = bytes.clone();
        let last = damaged.len() - 1;
        damaged[last] ^= 1;
        std::fs::write(file.path(&dir), &damaged).unwrap();
        let error = file.open(&dir).err().unwrap();
        assert!(matches!(error, ZenodoError::NotTheFile { .. }), "{error}");
        assert!(error.to_string().contains("10.5281/zenodo.0"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_manifest_lines_name_the_record_and_quote_its_licence() {
        let file = pinned(b"x");
        let lines = file.sources();
        assert!(lines[0].contains("https://doi.org/10.5281/zenodo.0"));
        assert!(lines[1].contains("\"Creative Commons Attribution 4.0 International\""));
        assert!(lines[1].contains("on the condition that the creator is appropriately credited."));
        assert_eq!(file.allowed_urls(), [file.url]);
        assert_eq!(politeness().min_interval, Duration::from_secs(10));
    }
}
