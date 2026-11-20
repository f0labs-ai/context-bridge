//! `cb-run vectors`: one vector per record, read straight from the search index, written
//! as `<run>/vectors.f32` + `<run>/vectors.json`.

use super::{Shared, read_every_point};
use crate::error::{FailureKind, RunError};
use crate::index::{DENSE, IndexReader, PassagePlace, PayloadWant, Point, PointId, Want};
use crate::run::{About, IndexRecord, Report, Session};
use crate::vector_file::VectorTable;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The name of the vector file pair in a run directory.
pub const VECTORS_NAME: &str = "vectors";
const COMMAND: &str = "vectors";

/// What a scroll asks for: where each passage sits, and its vector by meaning. Never a
/// passage's text.
const WANT: Want = Want {
    payload: PayloadWant::Fields(&["source_path", "seq"]),
    vectors: &[DENSE],
    page: 256,
};

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct VectorsArgs {
    /// The index's collection (the search index's own name for it, as the knowledge
    /// service's `/v1/info` gives it under `collection`), not a collection label.
    #[arg(long)]
    pub collection: String,
    /// The tenant whose passages are read.
    #[arg(long)]
    pub tenant: String,
    /// The run directory; vectors.f32 and vectors.json are written.
    #[arg(long)]
    pub run: PathBuf,
}

/// The first passage seen so far for one record.
#[derive(Debug)]
struct FirstPassage {
    seq: i64,
    point: String,
    vector: Vec<f32>,
}

/// The record a passage belongs to: the record its document was uploaded for, or — for
/// a passage of any other document — the point itself. The same join as `candidates`.
pub fn record_of_passage(point: &PointId, place: Option<&PassagePlace>) -> String {
    place
        .and_then(|p| p.source_path.as_deref())
        .and_then(cb_types::ids::record_of)
        .map_or_else(|| point.to_string(), str::to_string)
}

/// Keeps `candidate` when it comes before the passage held for its record: the lower
/// `seq` (a passage without one comes last), and between equals the lower point id, so
/// the choice never depends on the order the index returned them in.
fn keep_first(held: &mut BTreeMap<String, FirstPassage>, record: String, candidate: FirstPassage) {
    match held.get(&record) {
        Some(current)
            if (current.seq, current.point.as_str()) <= (candidate.seq, candidate.point.as_str()) => {}
        _ => {
            held.insert(record, candidate);
        }
    }
}

/// Reads every passage of the tenant and writes one vector per record.
pub async fn vectors<I: IndexReader>(
    index: &I,
    args: &VectorsArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let about = About {
        index: Some(IndexRecord {
            url: index.address(),
            collection: args.collection.clone(),
        }),
        machine_label: shared.machine_label.clone(),
        ..About::default()
    };
    let mut session = Session::begin(&args.run, COMMAND, args, about)?;
    match read_all_passages(index, args, shared, &mut session).await {
        Ok(Some(table)) => {
            if let Err(error) = table.write(session.dir(), VECTORS_NAME) {
                session.stop(&error);
                return Err(error);
            }
            session.set("written", table.len() as u64);
            session.set("dim", table.dim() as u64);
        }
        // The index could not be read whole: the failure is on file and nothing is written.
        Ok(None) => {}
        Err(error) => {
            session.stop(&error);
            return Err(error);
        }
    }
    // The file is rewritten whole on every run, so this run's outcome replaces every
    // earlier failure of this command.
    session.finish(|_| true)
}

/// Every passage of the tenant, reduced to the first passage of each record. `None`
/// when the index could not be read whole; the failure is then on file.
async fn read_all_passages<I: IndexReader>(
    index: &I,
    args: &VectorsArgs,
    shared: &Shared,
    session: &mut Session,
) -> Result<Option<VectorTable>, RunError> {
    let mut first: BTreeMap<String, FirstPassage> = BTreeMap::new();
    let mut without_vector = Vec::new();
    let keep_page = |points: Vec<Point<PassagePlace>>| {
        for point in points {
            let Some(vector) = point.vector.and_then(|v| v.dense).filter(|v| !v.is_empty()) else {
                without_vector.push(point.id.to_string());
                continue;
            };
            let record = record_of_passage(&point.id, point.payload.as_ref());
            let candidate = FirstPassage {
                seq: point.payload.as_ref().and_then(|p| p.seq).unwrap_or(i64::MAX),
                point: point.id.to_string(),
                vector,
            };
            keep_first(&mut first, record, candidate);
        }
    };
    let place = (args.collection.as_str(), args.tenant.as_str());
    match read_every_point(index, place, WANT, &shared.retry, keep_page).await {
        Ok(read) => session.set("passages_read", read),
        Err(gave_up) => {
            session.fail("", &gave_up.error, gave_up.tries)?;
            return Ok(None);
        }
    }
    for point in &without_vector {
        let error = RunError::Contract {
            id: point.clone(),
            url: index.address(),
            reason: "the passage has no vector by meaning in the index".into(),
        };
        session.fail_as(point, FailureKind::Contract, &error, 1)?;
    }
    // Rows in record-id order: the file is the same whatever order the index answered in.
    let mut table = VectorTable::new();
    for (record, passage) in &first {
        table.push(record, &passage.vector)?;
    }
    Ok(Some(table))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn place(source_path: &str, seq: Option<i64>) -> PassagePlace {
        PassagePlace {
            source_path: Some(source_path.into()),
            seq,
        }
    }

    #[test]
    fn a_passage_is_joined_to_its_record_or_stands_for_itself() {
        let uuid = PointId::Text("d7528b29-4637-55f9-ad70-29980bbd48b5".into());
        assert_eq!(
            record_of_passage(&uuid, Some(&place("cb/inc-0001.s1.md", Some(0)))),
            "inc-0001.s1"
        );
        assert_eq!(
            record_of_passage(&uuid, Some(&place("https://docs.example.com/a", Some(2)))),
            "d7528b29-4637-55f9-ad70-29980bbd48b5"
        );
        assert_eq!(record_of_passage(&PointId::Number(7), None), "7");
    }

    #[test]
    fn the_first_passage_of_a_record_is_its_lowest_seq_whatever_the_order_read() {
        let passage = |seq: i64, point: &str, value: f32| FirstPassage {
            seq,
            point: point.into(),
            vector: vec![value],
        };
        let mut held = BTreeMap::new();
        keep_first(&mut held, "r".into(), passage(2, "p-c", 2.0));
        keep_first(&mut held, "r".into(), passage(0, "p-b", 0.0));
        keep_first(&mut held, "r".into(), passage(1, "p-a", 1.0));
        assert_eq!(held["r"].vector, vec![0.0]);
        // The same seq twice: the lower point id, in either order.
        keep_first(&mut held, "s".into(), passage(0, "p-z", 9.0));
        keep_first(&mut held, "s".into(), passage(0, "p-y", 8.0));
        assert_eq!(held["s"].vector, vec![8.0]);
        keep_first(&mut held, "t".into(), passage(0, "p-y", 8.0));
        keep_first(&mut held, "t".into(), passage(0, "p-z", 9.0));
        assert_eq!(held["t"].vector, vec![8.0]);
    }
}
