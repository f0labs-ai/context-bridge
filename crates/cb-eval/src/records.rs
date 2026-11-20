//! The record table: everything a ranking method may know about a record.
//!
//! A method sees a record's id, when it was written, its stated end, its tier
//! and (keyed methods only) its thread. `Record.truth` is not stored here at
//! all, so no method can read it.

use crate::error::EvalError;
use cb_types::{Record, Tier, UnixSeconds};
use std::cmp::Ordering;
use std::collections::HashMap;

/// A record's row in the table. Cheap to copy, hash and compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RecordIdx(u32);

impl RecordIdx {
    pub(crate) fn row(self) -> usize {
        self.0 as usize
    }
}

/// id → written_at, valid_until, tier, thread, for every record of a data set.
#[derive(Debug)]
pub struct RecordTable {
    ids: Vec<String>,
    written_at: Vec<UnixSeconds>,
    valid_until: Vec<Option<UnixSeconds>>,
    tier: Vec<Tier>,
    thread: Vec<Option<u32>>,
    by_id: HashMap<String, u32>,
    /// Every record ordered newest first, then by id: the tie-break order of DESIGN.md section 4.
    newest_first: Vec<RecordIdx>,
    /// Each record's position in `newest_first`.
    recency_rank: Vec<u32>,
    /// Per thread, its records newest first, then by id.
    thread_members: Vec<Vec<RecordIdx>>,
}

impl RecordTable {
    /// Build the table. Two records with one id are a contract error.
    pub fn from_records(records: &[Record]) -> Result<Self, EvalError> {
        let count = u32::try_from(records.len())
            .map_err(|_| EvalError::invalid("records.jsonl holds more than 4 billion records"))?;
        let mut by_id = HashMap::with_capacity(records.len());
        let mut thread_ids: HashMap<&str, u32> = HashMap::new();
        let mut thread = Vec::with_capacity(records.len());
        for (row, record) in (0..count).zip(records) {
            if by_id.insert(record.id.clone(), row).is_some() {
                return Err(EvalError::invalid(format!(
                    "records.jsonl: record id {:?} appears twice",
                    record.id
                )));
            }
            thread.push(record.thread.as_deref().map(|name| {
                let next = thread_ids.len() as u32;
                *thread_ids.entry(name).or_insert(next)
            }));
        }
        let mut newest_first: Vec<RecordIdx> = (0..count).map(RecordIdx).collect();
        newest_first.sort_by(|a, b| {
            let (ra, rb) = (&records[a.row()], &records[b.row()]);
            rb.written_at.cmp(&ra.written_at).then_with(|| ra.id.cmp(&rb.id))
        });
        let mut recency_rank = vec![0u32; records.len()];
        let mut thread_members = vec![Vec::new(); thread_ids.len()];
        for (position, record) in (0..count).zip(&newest_first) {
            recency_rank[record.row()] = position;
            if let Some(t) = thread[record.row()] {
                thread_members[t as usize].push(*record);
            }
        }
        Ok(Self {
            ids: records.iter().map(|r| r.id.clone()).collect(),
            written_at: records.iter().map(|r| r.written_at).collect(),
            valid_until: records.iter().map(|r| r.valid_until).collect(),
            tier: records.iter().map(|r| r.tier).collect(),
            thread,
            by_id,
            newest_first,
            recency_rank,
            thread_members,
        })
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn index_of(&self, id: &str) -> Option<RecordIdx> {
        self.by_id.get(id).copied().map(RecordIdx)
    }

    pub fn id(&self, record: RecordIdx) -> &str {
        &self.ids[record.row()]
    }

    pub fn written_at(&self, record: RecordIdx) -> UnixSeconds {
        self.written_at[record.row()]
    }

    pub fn valid_until(&self, record: RecordIdx) -> Option<UnixSeconds> {
        self.valid_until[record.row()]
    }

    pub fn tier(&self, record: RecordIdx) -> Tier {
        self.tier[record.row()]
    }

    /// The tie-break of every method: the newer record first, then the smaller id.
    pub fn cmp_newer_then_id(&self, a: RecordIdx, b: RecordIdx) -> Ordering {
        self.recency_rank[a.row()].cmp(&self.recency_rank[b.row()])
    }

    /// The `n` newest records with `written_at <= asked_at`, newest first, then by id.
    pub fn newest_visible(&self, asked_at: UnixSeconds, n: usize) -> &[RecordIdx] {
        let start = self
            .newest_first
            .partition_point(|r| self.written_at[r.row()] > asked_at);
        let end = start.saturating_add(n).min(self.newest_first.len());
        &self.newest_first[start..end]
    }

    /// **Keyed methods only.** The newest record visible at `asked_at` in the
    /// thread of `record` (ties by id); `None` when `record` has no thread.
    pub fn keyed_newest_visible_in_thread_of(
        &self,
        record: RecordIdx,
        asked_at: UnixSeconds,
    ) -> Option<RecordIdx> {
        let members = &self.thread_members[self.thread[record.row()]? as usize];
        members
            .iter()
            .copied()
            .find(|m| self.written_at[m.row()] <= asked_at)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{record, threaded};

    fn ids<'a>(table: &'a RecordTable, records: &[RecordIdx]) -> Vec<&'a str> {
        records.iter().map(|r| table.id(*r)).collect()
    }

    #[test]
    fn newest_visible_is_newest_first_then_by_id_and_never_sees_the_future() {
        let table = RecordTable::from_records(&[
            record("b", 100),
            record("a", 100),
            record("c", 300),
            record("d", 200),
            record("e", 301),
        ])
        .unwrap();
        // Asked at 300: "e" (301) is not written yet. Newest first: c(300), d(200), then
        // a and b share 100 and go by id.
        assert_eq!(ids(&table, table.newest_visible(300, 20)), ["c", "d", "a", "b"]);
        assert_eq!(ids(&table, table.newest_visible(300, 2)), ["c", "d"]);
        assert_eq!(ids(&table, table.newest_visible(99, 20)), Vec::<&str>::new());
        assert_eq!(ids(&table, table.newest_visible(i64::MAX, 1)), ["e"]);
    }

    #[test]
    fn the_tie_break_puts_the_newer_record_first_then_the_smaller_id() {
        let table =
            RecordTable::from_records(&[record("b", 100), record("a", 100), record("c", 300)]).unwrap();
        let at = |id: &str| table.index_of(id).unwrap();
        assert_eq!(table.cmp_newer_then_id(at("c"), at("a")), Ordering::Less);
        assert_eq!(table.cmp_newer_then_id(at("a"), at("b")), Ordering::Less);
        assert_eq!(table.cmp_newer_then_id(at("b"), at("a")), Ordering::Greater);
        assert_eq!(table.cmp_newer_then_id(at("a"), at("a")), Ordering::Equal);
    }

    #[test]
    fn the_newest_visible_record_of_a_thread_moves_with_the_moment_asked() {
        let table = RecordTable::from_records(&[
            threaded("i1.s1", 100, "i1"),
            threaded("i1.s2", 200, "i1"),
            threaded("i2.s1", 150, "i2"),
            record("doc", 10),
        ])
        .unwrap();
        let at = |id: &str| table.index_of(id).unwrap();
        let newest = |id: &str, t| {
            table
                .keyed_newest_visible_in_thread_of(at(id), t)
                .map(|r| table.id(r))
        };
        assert_eq!(newest("i1.s1", 150), Some("i1.s1"));
        assert_eq!(newest("i1.s1", 200), Some("i1.s2"));
        assert_eq!(newest("i1.s2", 99), None);
        assert_eq!(newest("i2.s1", 500), Some("i2.s1"));
        assert_eq!(newest("doc", 500), None);
    }

    #[test]
    fn a_repeated_id_is_refused() {
        let twice = RecordTable::from_records(&[record("a", 1), record("a", 2)]);
        assert!(matches!(twice, Err(EvalError::Invalid(m)) if m.contains("appears twice")));
    }
}
