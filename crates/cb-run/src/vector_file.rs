//! Vectors on disk: a file of little-endian f32 rows and a JSON index that names each row.
//!
//! `<name>.f32` holds `ids.len()` rows of `dim` values, row after row. `<name>.json` is
//! `{"dim": 1024, "ids": [...]}`: row `i` belongs to `ids[i]`. Both are written through a
//! temporary file and a rename, the rows first and the index last, so an index never names
//! a row that is not on disk.

use crate::atomic::{write_atomically, write_json_atomically};
use crate::error::RunError;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};

/// The JSON index of a vector file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorIndex {
    pub dim: usize,
    pub ids: Vec<String>,
}

/// Vectors of one width, each under an id, in row order.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTable {
    dim: usize,
    ids: Vec<String>,
    seen: HashSet<String>,
    values: Vec<f32>,
}

/// Where a table called `name` lives in `dir`.
pub fn rows_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.f32"))
}

pub fn index_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.json"))
}

impl VectorTable {
    /// An empty table. The width is fixed by the first row.
    pub fn new() -> Self {
        Self {
            dim: 0,
            ids: Vec::new(),
            seen: HashSet::new(),
            values: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn ids(&self) -> &[String] {
        &self.ids
    }

    pub fn contains(&self, id: &str) -> bool {
        self.seen.contains(id)
    }

    /// Row `i`, when there is one.
    pub fn row(&self, i: usize) -> Option<&[f32]> {
        let start = i.checked_mul(self.dim)?;
        self.values.get(start..start + self.dim)
    }

    /// Adds a row. A row of another width, an empty row or an id that is already there is
    /// refused: each would make the file say something false.
    pub fn push(&mut self, id: &str, row: &[f32]) -> Result<(), RunError> {
        if row.is_empty() {
            return Err(RunError::item(id, "a vector has at least one value"));
        }
        if self.ids.is_empty() {
            self.dim = row.len();
        } else if row.len() != self.dim {
            return Err(RunError::item(
                id,
                format!("a vector of {} values among vectors of {}", row.len(), self.dim),
            ));
        }
        if !self.seen.insert(id.to_string()) {
            return Err(RunError::item(id, "this id already has a vector"));
        }
        self.ids.push(id.to_string());
        self.values.extend_from_slice(row);
        Ok(())
    }

    /// Writes `<name>.f32` and `<name>.json` into `dir`, each atomically.
    pub fn write(&self, dir: &Path, name: &str) -> Result<(), RunError> {
        write_atomically(&rows_path(dir, name), |writer| {
            for value in &self.values {
                writer.write_all(&value.to_le_bytes())?;
            }
            Ok(())
        })?;
        write_json_atomically(
            &index_path(dir, name),
            &VectorIndex {
                dim: self.dim,
                ids: self.ids.clone(),
            },
        )
    }

    /// Reads the table called `name` from `dir`; none when it was never written.
    ///
    /// The rows file may hold more rows than the index names (a write that was stopped
    /// between its two renames): the index decides, the extra rows are ignored. Fewer
    /// rows than the index names is an error.
    pub fn read(dir: &Path, name: &str) -> Result<Option<Self>, RunError> {
        let index_file = index_path(dir, name);
        let rows_file = rows_path(dir, name);
        let index: VectorIndex = match File::open(&index_file) {
            Ok(file) => serde_json::from_reader(BufReader::new(file))
                .map_err(|e| RunError::file(&index_file, format!("not a vector index: {e}")))?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(RunError::io(&index_file, e)),
        };
        let wanted = index
            .ids
            .len()
            .checked_mul(index.dim)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| RunError::file(&index_file, "the index is too large to be real"))?;
        let mut bytes = Vec::with_capacity(wanted);
        if wanted > 0 {
            let file = File::open(&rows_file).map_err(|e| RunError::io(&rows_file, e))?;
            file.take(wanted as u64)
                .read_to_end(&mut bytes)
                .map_err(|e| RunError::io(&rows_file, e))?;
        }
        if bytes.len() != wanted {
            return Err(RunError::file(
                &rows_file,
                format!(
                    "{} bytes, but the index names {} rows of {} values ({wanted} bytes)",
                    bytes.len(),
                    index.ids.len(),
                    index.dim
                ),
            ));
        }
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let seen: HashSet<String> = index.ids.iter().cloned().collect();
        if seen.len() != index.ids.len() {
            return Err(RunError::file(&index_file, "an id is listed twice"));
        }
        Ok(Some(Self {
            dim: index.dim,
            ids: index.ids,
            seen,
            values,
        }))
    }
}

impl Default for VectorTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::ScratchDir;
    use std::fs;

    #[test]
    fn a_table_reads_back_exactly_as_written() {
        let dir = ScratchDir::new("vectors-roundtrip");
        let mut table = VectorTable::new();
        table
            .push("inc-0001.s1", &[0.1, -2.5, f32::MIN_POSITIVE])
            .unwrap();
        table.push("doc-012.3", &[1.0e-7, 3.4e38, -0.0]).unwrap();
        table.write(dir.path(), "vectors").unwrap();

        // Little-endian f32, row after row.
        let bytes = fs::read(dir.path().join("vectors.f32")).unwrap();
        assert_eq!(bytes.len(), 2 * 3 * 4);
        assert_eq!(&bytes[..4], &0.1f32.to_le_bytes());
        assert_eq!(&bytes[12..16], &1.0e-7f32.to_le_bytes());
        let index: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(dir.path().join("vectors.json")).unwrap()).unwrap();
        assert_eq!(
            index,
            serde_json::json!({"dim": 3, "ids": ["inc-0001.s1", "doc-012.3"]})
        );

        let back = VectorTable::read(dir.path(), "vectors").unwrap().unwrap();
        assert_eq!(back.dim(), 3);
        assert_eq!(back.ids(), table.ids());
        for i in 0..2 {
            let (a, b) = (back.row(i).unwrap(), table.row(i).unwrap());
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "row {i}"
            );
        }
        assert!(back.row(2).is_none());
    }

    #[test]
    fn a_table_that_was_never_written_is_none() {
        let dir = ScratchDir::new("vectors-none");
        assert!(VectorTable::read(dir.path(), "vectors").unwrap().is_none());
    }

    #[test]
    fn rows_of_another_width_and_repeated_ids_are_refused() {
        let mut table = VectorTable::new();
        table.push("a", &[1.0, 2.0]).unwrap();
        assert!(table.push("b", &[1.0]).is_err());
        assert!(table.push("a", &[3.0, 4.0]).is_err());
        assert!(table.push("c", &[]).is_err());
        assert_eq!(table.len(), 1);
        assert!(table.contains("a") && !table.contains("b"));
    }

    #[test]
    fn fewer_rows_than_the_index_names_is_an_error_and_extra_rows_are_ignored() {
        let dir = ScratchDir::new("vectors-mismatch");
        let mut table = VectorTable::new();
        table.push("a", &[1.0, 2.0]).unwrap();
        table.push("b", &[3.0, 4.0]).unwrap();
        table.write(dir.path(), "v").unwrap();
        let rows = dir.path().join("v.f32");
        let whole = fs::read(&rows).unwrap();

        fs::write(&rows, &whole[..12]).unwrap();
        assert!(matches!(
            VectorTable::read(dir.path(), "v"),
            Err(RunError::File { .. })
        ));

        let mut longer = whole.clone();
        longer.extend_from_slice(&[0u8; 8]);
        fs::write(&rows, &longer).unwrap();
        let back = VectorTable::read(dir.path(), "v").unwrap().unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back.row(1).unwrap(), &[3.0, 4.0]);
    }

    #[test]
    fn an_empty_table_is_written_and_read() {
        let dir = ScratchDir::new("vectors-empty");
        VectorTable::new().write(dir.path(), "v").unwrap();
        let back = VectorTable::read(dir.path(), "v").unwrap().unwrap();
        assert!(back.is_empty());
        assert_eq!(fs::read(dir.path().join("v.f32")).unwrap().len(), 0);
    }
}
