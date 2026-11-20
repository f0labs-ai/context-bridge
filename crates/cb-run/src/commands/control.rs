//! `cb-run control export` and `cb-run control import`: the control data.
//!
//! `export` reads one tenant's passages out of a search index and writes them as a data
//! set. It is given an [`IndexReader`] and nothing else: it cannot write to the index it
//! reads. `import` writes those passages into the experiment's collection, under the
//! experiment's tenant and collection label, with the moment each was "written" added.
//! It reads where the data set came from (`export.json`) and refuses to write there.

use super::{
    DATA_MANIFEST_FILE, QUERIES_FILE, RECORDS_FILE, Shared, belonging, read_every_point, read_records,
    write_jsonl_atomically,
};
use crate::atomic::{sha256_of_file, write_json_atomically};
use crate::error::RunError;
use crate::index::{
    DENSE, IndexReader, IndexWriter, PassagePayload, PayloadWant, Point, PointId, PointToWrite, SPARSE,
    SparseVector, VectorsToWrite, Want, same_server,
};
use crate::jsonl::{Appender, CutOffLine, read_all, read_resumable};
use crate::knowledge::{Label, Tenant};
use crate::retry::with_retries;
use crate::run::{About, GIT_COMMIT, IndexRecord, Report, Session, VERSION, now_utc};
use crate::seeded::Draws;
use cb_types::{Gold, Manifest, Query, QueryClass, Record, Tier, UnixSeconds};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

pub const POINTS_FILE: &str = "points.jsonl";
/// Where a control data set says which index it was exported from.
pub const ORIGIN_FILE: &str = "export.json";
/// The run file that lists the points an import has written.
pub const IMPORTED_FILE: &str = "imported.jsonl";
pub const DEFAULT_EXPORT_RUN_DIR: &str = "run-control-export";
pub const DEFAULT_IMPORT_RUN_DIR: &str = "run-control-import";
const EXPORT: &str = "control export";
const IMPORT: &str = "control import";

const DAY: UnixSeconds = 86_400;
/// A passage's `written_at` is drawn this many days before the window start, at least…
pub const YOUNGEST_DAYS: i64 = 30;
/// …and at most.
pub const OLDEST_DAYS: i64 = 700;
/// Points written per request by the import.
const IMPORT_BATCH: usize = 64;

/// What the export reads of every point: the whole payload and both vectors.
const EVERYTHING: Want = Want {
    payload: PayloadWant::All,
    vectors: &[DENSE, SPARSE],
    page: 128,
};

// ---- the data set's own files ----------------------------------------------------------

/// One line of `points.jsonl`: a passage as the product holds it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedPoint {
    pub id: PointId,
    pub payload: PassagePayload,
    pub dense: Vec<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse: Option<SparseVector>,
}

/// `export.json`: where a control data set was read from. The import reads it to make
/// sure it never writes there.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportOrigin {
    /// The address of the index (no user, password or query).
    pub index_url: String,
    /// The index's collection the passages were read from.
    pub collection: String,
    pub tenant: String,
    pub exported_at: String,
    pub window_start: UnixSeconds,
    pub seed: u64,
    pub points: u64,
}

/// The held-out questions file: the questions and the passage each was written from.
#[derive(Clone, Debug, Deserialize)]
struct HeldOutFile {
    /// The tenant whose passages the questions were written from.
    tenant: Option<String>,
    questions: Vec<HeldOut>,
}

/// One held-out question.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct HeldOut {
    pub qid: String,
    pub question: String,
    /// The point id of the passage the question was written from.
    pub gold_id: String,
}

fn read_held_out(path: &Path, tenant: &str) -> Result<Vec<HeldOut>, RunError> {
    let bytes = std::fs::read(path).map_err(|e| RunError::io(path, e))?;
    let file: HeldOutFile = serde_json::from_slice(&bytes)
        .map_err(|e| RunError::file(path, format!("not a held-out questions file: {e}")))?;
    if let Some(theirs) = file.tenant.as_deref()
        && theirs != tenant
    {
        return Err(RunError::file(
            path,
            format!("the questions were written from tenant {theirs}, and --from-tenant is {tenant}"),
        ));
    }
    let mut seen = HashSet::new();
    if let Some(twice) = file.questions.iter().find(|q| !seen.insert(q.qid.as_str())) {
        return Err(RunError::file(
            path,
            format!("question id {:?} appears twice", twice.qid),
        ));
    }
    Ok(file.questions)
}

// ---- export ----------------------------------------------------------------------------

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct ExportArgs {
    /// The index's collection the passages are read from (its own name in the search
    /// index, not a collection label).
    #[arg(long)]
    pub from_collection: String,
    /// The tenant whose passages are read.
    #[arg(long)]
    pub from_tenant: String,
    /// The held-out questions file ({"tenant", "questions": [{"qid", "question", "gold_id"}]}).
    #[arg(long)]
    pub questions: PathBuf,
    /// The start of the experiment's window (Unix seconds); every passage is written
    /// 30 to 700 days before it.
    #[arg(long)]
    pub window_start: UnixSeconds,
    /// The data set directory to write.
    #[arg(long)]
    pub out: PathBuf,
    /// The seed of the passages' written_at.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Leave out the passages filed under this collection label in the source. May be
    /// given more than once.
    #[arg(long = "leave-out-label")]
    pub leave_out_label: Vec<String>,
    /// Where failures.jsonl and the record of the run go. Default: <out>/run-control-export.
    #[arg(long)]
    pub run: Option<PathBuf>,
}

/// How a control data set is built from the points read.
#[derive(Clone, Copy, Debug)]
pub struct ExportSettings<'a> {
    /// The data set's name: every record's `source`.
    pub name: &'a str,
    pub window_start: UnixSeconds,
    pub seed: u64,
    pub leave_out_labels: &'a [String],
}

/// A control data set, before it is written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Built {
    pub records: Vec<Record>,
    pub queries: Vec<Query>,
    pub points: Vec<ExportedPoint>,
    /// Points and questions that cannot be part of the data set, each with the reason.
    pub not_exported: Vec<(String, String)>,
    pub left_out_by_label: u64,
}

/// The moment a passage counts as written: drawn 30 to 700 days before the window start
/// from the seed and the passage's own id, so it does not depend on which other
/// passages are exported.
pub fn written_at_for(seed: u64, window_start: UnixSeconds, point_id: &str) -> UnixSeconds {
    Draws::keyed(seed, point_id).between(
        window_start.saturating_sub(OLDEST_DAYS * DAY),
        window_start.saturating_sub(YOUNGEST_DAYS * DAY),
    )
}

/// Builds the data set: one standing record per passage (the point id as the record id),
/// one ordinary question per held-out question (grade 3 for the passage it was written
/// from, grade 1 for the passages that share that passage's parent section), and the
/// points themselves. Passages are taken in id order, so the files are the same whatever
/// order the index answered in.
pub fn build_control_set(
    mut points: Vec<Point<PassagePayload>>,
    held_out: &[HeldOut],
    settings: ExportSettings<'_>,
) -> Built {
    points.sort_by(|a, b| a.id.cmp(&b.id));
    let mut built = Built::default();
    let mut parent_of: HashMap<String, String> = HashMap::new();
    let mut children_of: HashMap<String, Vec<String>> = HashMap::new();
    for point in points {
        let id = point.id.to_string();
        let Some(payload) = point.payload else {
            built.not_exported.push((id, "the passage has no payload".into()));
            continue;
        };
        if payload
            .collection
            .as_deref()
            .is_some_and(|label| settings.leave_out_labels.iter().any(|left_out| left_out == label))
        {
            built.left_out_by_label += 1;
            continue;
        }
        let Some(text) = payload.text.clone().filter(|text| !text.trim().is_empty()) else {
            built.not_exported.push((id, "the passage has no text".into()));
            continue;
        };
        let (dense, sparse) = match point.vector {
            Some(vectors) => (vectors.dense.unwrap_or_default(), vectors.sparse),
            None => (Vec::new(), None),
        };
        if dense.is_empty() {
            built
                .not_exported
                .push((id, "the passage has no vector by meaning".into()));
            continue;
        }
        if cb_types::ids::check(&id).is_err() {
            built
                .not_exported
                .push((id, "the point's id cannot be a record id".into()));
            continue;
        }
        if let Some(parent) = payload.parent_id.as_deref().filter(|parent| !parent.is_empty()) {
            parent_of.insert(id.clone(), parent.to_string());
            children_of
                .entry(parent.to_string())
                .or_default()
                .push(id.clone());
        }
        built.records.push(Record {
            id: id.clone(),
            text,
            written_at: written_at_for(settings.seed, settings.window_start, &id),
            valid_until: None,
            thread: None,
            tier: Tier::Standing,
            title: payload
                .title
                .clone()
                .or_else(|| payload.heading.clone())
                .filter(|title| !title.trim().is_empty()),
            source: settings.name.to_string(),
            truth: None,
        });
        built.points.push(ExportedPoint {
            id: point.id,
            payload,
            dense,
            sparse,
        });
    }

    let exported: HashSet<&str> = built.records.iter().map(|r| r.id.as_str()).collect();
    for question in held_out {
        let id = format!("hq-{}", question.qid);
        if cb_types::ids::check(&id).is_err() {
            built
                .not_exported
                .push((id, "the question's id cannot be an id".into()));
            continue;
        }
        if question.question.trim().is_empty() {
            built.not_exported.push((id, "the question has no words".into()));
            continue;
        }
        if !exported.contains(question.gold_id.as_str()) {
            let reason = format!(
                "the passage it was written from ({}) is not among the exported passages",
                question.gold_id
            );
            built.not_exported.push((id, reason));
            continue;
        }
        let mut gold = vec![Gold {
            record: question.gold_id.clone(),
            grade: 3,
        }];
        if let Some(siblings) = parent_of
            .get(&question.gold_id)
            .and_then(|parent| children_of.get(parent))
        {
            gold.extend(
                siblings
                    .iter()
                    .filter(|sibling| **sibling != question.gold_id)
                    .map(|sibling| Gold {
                        record: sibling.clone(),
                        grade: 1,
                    }),
            );
        }
        built.queries.push(Query {
            id,
            text: question.question.clone(),
            // A placeholder: the run that pairs the question with an incident sets the
            // moment. The window start is a moment at which every record is already written.
            asked_at: settings.window_start,
            class: QueryClass::Ordinary,
            gold,
            thread: None,
            harmful: Vec::new(),
        });
    }
    built
}

/// Reads one tenant's passages from a search index and writes them as a data set.
pub async fn export<I: IndexReader>(
    index: &I,
    args: &ExportArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let held_out = read_held_out(&args.questions, &args.from_tenant)?;
    let run = args
        .run
        .clone()
        .unwrap_or_else(|| args.out.join(DEFAULT_EXPORT_RUN_DIR));
    let about = About {
        index: Some(IndexRecord {
            url: index.address(),
            collection: args.from_collection.clone(),
        }),
        machine_label: shared.machine_label.clone(),
        ..About::default()
    };
    let mut session = Session::begin(&run, EXPORT, args, about)?;
    match export_work(index, args, shared, &held_out, &mut session).await {
        // The data set is written whole on every run: this run's outcome replaces
        // every earlier failure of this command.
        Ok(()) => session.finish(|_| true),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn export_work<I: IndexReader>(
    index: &I,
    args: &ExportArgs,
    shared: &Shared,
    held_out: &[HeldOut],
    session: &mut Session,
) -> Result<(), RunError> {
    let mut points: Vec<Point<PassagePayload>> = Vec::new();
    let place = (args.from_collection.as_str(), args.from_tenant.as_str());
    match read_every_point(index, place, EVERYTHING, &shared.retry, |page| {
        points.extend(page)
    })
    .await
    {
        Ok(read) => session.set("passages_read", read),
        Err(gave_up) => {
            // The index could not be read whole: the failure is on file, nothing is written.
            session.fail("", &gave_up.error, gave_up.tries)?;
            return Ok(());
        }
    }
    let name = args
        .out
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("control");
    let built = build_control_set(
        points,
        held_out,
        ExportSettings {
            name,
            window_start: args.window_start,
            seed: args.seed,
            leave_out_labels: &args.leave_out_label,
        },
    );
    for (id, reason) in &built.not_exported {
        session.fail(id, &RunError::item(id, reason.clone()), 1)?;
    }
    session.set("left_out_by_label", built.left_out_by_label);
    session.set("records", built.records.len() as u64);
    session.set("questions", built.queries.len() as u64);

    std::fs::create_dir_all(&args.out).map_err(|e| RunError::io(&args.out, e))?;
    let origin = ExportOrigin {
        index_url: index.address(),
        collection: args.from_collection.clone(),
        tenant: args.from_tenant.clone(),
        exported_at: now_utc(),
        window_start: args.window_start,
        seed: args.seed,
        points: built.points.len() as u64,
    };
    write_jsonl_atomically(&args.out.join(RECORDS_FILE), &built.records)?;
    write_jsonl_atomically(&args.out.join(QUERIES_FILE), &built.queries)?;
    write_jsonl_atomically(&args.out.join(POINTS_FILE), &built.points)?;
    write_json_atomically(&args.out.join(ORIGIN_FILE), &origin)?;

    let mut sha256 = BTreeMap::new();
    for file in [RECORDS_FILE, QUERIES_FILE, POINTS_FILE, ORIGIN_FILE] {
        sha256.insert(file.to_string(), sha256_of_file(&args.out.join(file))?);
    }
    let counts = [
        ("records", built.records.len()),
        ("queries", built.queries.len()),
        ("transcripts", 0),
        ("points", built.points.len()),
    ]
    .into_iter()
    .map(|(what, count)| (what.to_string(), count as u64))
    .collect();
    let questions_file = args.questions.file_name().and_then(|n| n.to_str()).unwrap_or("?");
    // The manifest last: a data set with a manifest is a whole one.
    write_json_atomically(
        &args.out.join(DATA_MANIFEST_FILE),
        &Manifest {
            name: name.to_string(),
            made_by: format!("cb-run {VERSION} ({GIT_COMMIT}) control export"),
            seed: Some(args.seed),
            counts,
            sha256,
            sources: vec![
                format!(
                    "passages of tenant {} in collection {} of the search index at {}, read {}",
                    origin.tenant, origin.collection, origin.index_url, origin.exported_at
                ),
                format!("held-out questions from {questions_file}"),
                format!(
                    "written_at drawn {YOUNGEST_DAYS} to {OLDEST_DAYS} days before {} (seed {}); \
                     asked_at is that same moment until a run pairs each question with an incident",
                    args.window_start, args.seed
                ),
            ],
        },
    )
}

// ---- import ----------------------------------------------------------------------------

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct ImportArgs {
    /// The control data set directory (written by `control export`).
    #[arg(long)]
    pub data: PathBuf,
    /// The experiment's tenant.
    #[arg(long)]
    pub tenant: String,
    /// The experiment's collection in the search index (its own name there), which must
    /// not be the one the data set was exported from.
    #[arg(long)]
    pub collection: String,
    /// The collection label the passages are filed under in the experiment (the label
    /// `upload` and `candidates` use).
    #[arg(long)]
    pub label: String,
    /// Where failures.jsonl and the record of the run go. Default: <data>/run-control-import.
    #[arg(long)]
    pub run: Option<PathBuf>,
}

/// One line of `imported.jsonl`: a point that is in the experiment's index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct ImportedLine {
    id: String,
}

/// Refuses a destination that could be the index the data set was exported from: the
/// same collection name, or the same server. Either one is enough to refuse, so the
/// product's index cannot be written to by naming it in one flag.
pub fn refuse_the_source(
    destination_address: &str,
    destination_collection: &str,
    origin: &ExportOrigin,
) -> Result<(), RunError> {
    let refused = |reason: String| RunError::Refused {
        url: destination_address.to_string(),
        reason,
    };
    if destination_collection == origin.collection {
        return Err(refused(format!(
            "collection {destination_collection} is the one the data set was exported from"
        )));
    }
    let (Ok(destination), Ok(source)) = (Url::parse(destination_address), Url::parse(&origin.index_url))
    else {
        return Err(refused(format!(
            "cannot tell whether this is the index the data set was exported from ({})",
            origin.index_url
        )));
    };
    if same_server(&destination, &source) {
        return Err(refused(format!(
            "this is the index the data set was exported from ({}); the experiment has an index of its own",
            origin.index_url
        )));
    }
    Ok(())
}

fn read_origin(data: &Path) -> Result<ExportOrigin, RunError> {
    let path = data.join(ORIGIN_FILE);
    let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => RunError::file(
            &path,
            "the data set does not say where it was exported from, so nothing is written",
        ),
        _ => RunError::io(&path, e),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| RunError::file(&path, format!("not an export record: {e}")))
}

/// Checks the files an import reads against the hashes in the data set's manifest.
fn check_hashes(data: &Path) -> Result<(), RunError> {
    let manifest_path = data.join(DATA_MANIFEST_FILE);
    let bytes = std::fs::read(&manifest_path).map_err(|e| RunError::io(&manifest_path, e))?;
    let manifest: Manifest = serde_json::from_slice(&bytes)
        .map_err(|e| RunError::file(&manifest_path, format!("not a data set manifest: {e}")))?;
    for file in [RECORDS_FILE, POINTS_FILE, ORIGIN_FILE] {
        let path = data.join(file);
        let expected = manifest
            .sha256
            .get(file)
            .ok_or_else(|| RunError::file(&manifest_path, format!("no hash for {file}")))?;
        if &sha256_of_file(&path)? != expected {
            return Err(RunError::file(
                &path,
                "the file is not the one the manifest describes",
            ));
        }
    }
    Ok(())
}

/// Writes a control data set's passages into the experiment's collection.
pub async fn import<W: IndexWriter>(
    index: &W,
    args: &ImportArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let tenant = Tenant::parse(&args.tenant)?;
    let label = Label::parse("--label", &args.label)?;
    let origin = read_origin(&args.data)?;
    refuse_the_source(&index.address(), &args.collection, &origin)?;
    check_hashes(&args.data)?;
    let records = read_records(&args.data)?;
    let points: Vec<ExportedPoint> = read_all(&args.data.join(POINTS_FILE))?;
    let run = args
        .run
        .clone()
        .unwrap_or_else(|| args.data.join(DEFAULT_IMPORT_RUN_DIR));
    let about = About {
        index: Some(IndexRecord {
            url: index.address(),
            collection: args.collection.clone(),
        }),
        machine_label: shared.machine_label.clone(),
        ..About::default()
    };
    let mut session = Session::begin(&run, IMPORT, args, about)?;
    match import_work(
        index,
        args,
        shared,
        (&tenant, &label),
        (&records, &points),
        &mut session,
    )
    .await
    {
        Ok(done) => session.finish(|id| done.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn import_work<W: IndexWriter>(
    index: &W,
    args: &ImportArgs,
    shared: &Shared,
    (tenant, label): (&Tenant, &Label),
    (records, points): (&[Record], &[ExportedPoint]),
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    session.claim(
        IMPORTED_FILE,
        belonging(&[
            ("index", &index.address()),
            ("collection", &args.collection),
            ("tenant", tenant.as_str()),
            ("label", label.as_str()),
            ("points_sha256", &sha256_of_file(&args.data.join(POINTS_FILE))?),
        ]),
    )?;
    let path = session.file(IMPORTED_FILE);
    let existing = read_resumable::<ImportedLine>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let mut done: HashSet<String> = existing.lines.into_iter().map(|line| line.id).collect();
    let mut out = Appender::open(&path)?;
    let written_at: HashMap<&str, UnixSeconds> =
        records.iter().map(|r| (r.id.as_str(), r.written_at)).collect();

    // Each point with the payload it gets in the experiment: the product's own, under
    // the experiment's tenant and label, with the moment it counts as written.
    let mut todo: Vec<(String, &ExportedPoint, PassagePayload)> = Vec::new();
    for point in points {
        let id = point.id.to_string();
        session.add("selected", 1);
        if done.contains(&id) {
            session.add("already_there", 1);
            continue;
        }
        let Some(written) = written_at.get(id.as_str()) else {
            session.fail(
                &id,
                &RunError::item(&id, "the data set has no record for this point"),
                0,
            )?;
            continue;
        };
        let mut payload = point.payload.clone();
        payload.tenant_id = Some(tenant.as_str().to_string());
        payload.collection = Some(label.as_str().to_string());
        payload.written_at = Some(*written);
        todo.push((id, point, payload));
    }

    for batch in todo.chunks(IMPORT_BATCH) {
        let writes: Vec<PointToWrite<'_>> = batch
            .iter()
            .map(|(_, point, payload)| PointToWrite {
                id: &point.id,
                vector: VectorsToWrite {
                    dense: &point.dense,
                    sparse: point.sparse.as_ref(),
                },
                payload,
            })
            .collect();
        match with_retries(&shared.retry, |_| index.upsert(&args.collection, &writes)).await {
            Ok(_) => {
                let lines: Vec<ImportedLine> = batch
                    .iter()
                    .map(|(id, _, _)| ImportedLine { id: id.clone() })
                    .collect();
                out.append(&lines)?;
                for (id, _, _) in batch {
                    done.insert(id.clone());
                    session.wrote();
                }
            }
            Err(gave_up) => {
                for (id, _, _) in batch {
                    session.fail(id, &gave_up.error, gave_up.tries)?;
                }
            }
        }
    }
    out.sync()?;
    Ok(done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Vectors;

    fn point(id: &str, payload: serde_json::Value, dense: Option<Vec<f32>>) -> Point<PassagePayload> {
        Point {
            id: PointId::Text(id.into()),
            payload: Some(serde_json::from_value(payload).unwrap()),
            vector: Some(Vectors {
                dense,
                sparse: Some(SparseVector {
                    indices: vec![1, 5],
                    values: vec![0.5, 0.25],
                }),
            }),
        }
    }

    fn held_out(qid: &str, gold_id: &str) -> HeldOut {
        HeldOut {
            qid: qid.into(),
            question: format!("Question {qid}?"),
            gold_id: gold_id.into(),
        }
    }

    const WINDOW_START: i64 = 1_780_272_000;

    fn settings(leave_out: &[String]) -> ExportSettings<'_> {
        ExportSettings {
            name: "helpcentre",
            window_start: WINDOW_START,
            seed: 42,
            leave_out_labels: leave_out,
        }
    }

    fn points() -> Vec<Point<PassagePayload>> {
        vec![
            point(
                "p-c",
                serde_json::json!({"tenant_id": "product", "collection": "website", "text": "Third.", "parent_id": "sec-1", "kind": "step"}),
                Some(vec![0.3]),
            ),
            point(
                "p-a",
                serde_json::json!({"tenant_id": "product", "collection": "website", "text": "First.", "parent_id": "sec-1", "heading": "Setting up", "seq": 0}),
                Some(vec![0.1]),
            ),
            point(
                "p-b",
                serde_json::json!({"tenant_id": "product", "collection": "website", "text": "Second.", "parent_id": "sec-2", "title": "Billing", "heading": "Invoices"}),
                Some(vec![0.2]),
            ),
            point(
                "p-atlas",
                serde_json::json!({"tenant_id": "product", "collection": "atlas", "text": "Navigation."}),
                Some(vec![0.4]),
            ),
        ]
    }

    #[test]
    fn every_passage_becomes_a_standing_record_under_its_point_id_in_id_order() {
        let built = build_control_set(points(), &[], settings(&[]));
        let ids: Vec<&str> = built.records.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["p-a", "p-atlas", "p-b", "p-c"]);
        assert!(built.records.iter().all(|r| {
            r.tier == Tier::Standing
                && r.thread.is_none()
                && r.valid_until.is_none()
                && r.source == "helpcentre"
        }));
        assert_eq!(built.records[0].text, "First.");
        // The document's title when the product has one, else the passage's heading.
        assert_eq!(built.records[0].title.as_deref(), Some("Setting up"));
        assert_eq!(built.records[2].title.as_deref(), Some("Billing"));
        assert_eq!(built.records[3].title, None);
        // The points keep their ids, vectors and the product's whole payload.
        assert_eq!(built.points.len(), 4);
        assert_eq!(built.points[0].id, PointId::Text("p-a".into()));
        assert_eq!(built.points[0].dense, vec![0.1]);
        assert_eq!(built.points[0].sparse.as_ref().unwrap().indices, vec![1, 5]);
        assert_eq!(
            built.points[0].payload.rest.get("seq"),
            Some(&serde_json::json!(0))
        );
        assert_eq!(built.points[0].payload.tenant_id.as_deref(), Some("product"));
        assert!(built.not_exported.is_empty());
    }

    #[test]
    fn written_at_lies_30_to_700_days_before_the_window_and_is_fixed_by_seed_and_id() {
        let built = build_control_set(points(), &[], settings(&[]));
        for record in &built.records {
            let age_days = (WINDOW_START - record.written_at) as f64 / 86_400.0;
            assert!(
                (30.0..=700.0).contains(&age_days),
                "{} is {age_days} days old",
                record.id
            );
            assert_eq!(record.written_at, written_at_for(42, WINDOW_START, &record.id));
        }
        // Fewer passages exported: the others keep their moments.
        let fewer = build_control_set(points()[..2].to_vec(), &[], settings(&[]));
        let of = |built: &Built, id: &str| built.records.iter().find(|r| r.id == id).unwrap().written_at;
        assert_eq!(of(&fewer, "p-a"), of(&built, "p-a"));
        assert_ne!(
            written_at_for(42, WINDOW_START, "p-a"),
            written_at_for(43, WINDOW_START, "p-a")
        );
        // The whole range is used.
        let ages: Vec<i64> = (0..400)
            .map(|n| (WINDOW_START - written_at_for(42, WINDOW_START, &format!("p-{n}"))) / 86_400)
            .collect();
        assert!(
            ages.iter().any(|a| *a < 100) && ages.iter().any(|a| *a > 600),
            "{ages:?}"
        );
    }

    #[test]
    fn a_question_gets_its_passage_as_the_answer_and_the_same_section_as_related() {
        let built = build_control_set(
            points(),
            &[
                held_out("1", "p-a"),
                held_out("2", "p-b"),
                held_out("3", "p-gone"),
            ],
            settings(&[]),
        );
        assert_eq!(built.queries.len(), 2);
        let first = &built.queries[0];
        assert_eq!((first.id.as_str(), first.text.as_str()), ("hq-1", "Question 1?"));
        assert_eq!(first.class, QueryClass::Ordinary);
        assert_eq!(first.asked_at, WINDOW_START);
        assert_eq!(
            first.gold,
            vec![
                Gold {
                    record: "p-a".into(),
                    grade: 3
                },
                Gold {
                    record: "p-c".into(),
                    grade: 1
                },
            ]
        );
        // The only passage of its section: the answer and nothing else.
        assert_eq!(
            built.queries[1].gold,
            vec![Gold {
                record: "p-b".into(),
                grade: 3
            }]
        );
        // The gold of every question was written before the question is asked.
        for query in &built.queries {
            let gold = built
                .records
                .iter()
                .find(|r| r.id == query.gold[0].record)
                .unwrap();
            assert!(gold.written_at <= query.asked_at);
        }
        // A question whose passage is gone is named, never dropped in silence.
        assert_eq!(built.not_exported.len(), 1);
        assert_eq!(built.not_exported[0].0, "hq-3");
        assert!(built.not_exported[0].1.contains("p-gone"));
    }

    #[test]
    fn a_label_can_be_left_out_and_what_cannot_be_a_record_is_named() {
        let leave_out = ["atlas".to_string()];
        let mut all = points();
        all.push(point(
            "p-empty",
            serde_json::json!({"collection": "website", "text": "  "}),
            Some(vec![0.5]),
        ));
        all.push(point(
            "p-image",
            serde_json::json!({"collection": "website", "text": "A photo."}),
            None,
        ));
        all.push(point(
            "Not An Id",
            serde_json::json!({"collection": "website", "text": "Words."}),
            Some(vec![0.6]),
        ));
        let built = build_control_set(all, &[held_out("9", "p-atlas")], settings(&leave_out));
        assert_eq!(built.left_out_by_label, 1);
        assert_eq!(built.records.len(), 3);
        assert!(built.records.iter().all(|r| r.id != "p-atlas"));
        let named: HashMap<&str, &str> = built
            .not_exported
            .iter()
            .map(|(id, why)| (id.as_str(), why.as_str()))
            .collect();
        assert!(named["p-empty"].contains("no text"));
        assert!(named["p-image"].contains("no vector"));
        assert!(named["Not An Id"].contains("record id"));
        assert!(named["hq-9"].contains("not among the exported"));
    }

    fn origin() -> ExportOrigin {
        ExportOrigin {
            index_url: "http://127.0.0.1:6339/".into(),
            collection: "kb_source".into(),
            tenant: "0c5a9e21".into(),
            exported_at: "2026-11-19T00:00:00.000Z".into(),
            window_start: WINDOW_START,
            seed: 42,
            points: 4,
        }
    }

    #[test]
    fn the_index_a_data_set_came_from_is_never_written_to() {
        // Another server and another collection: allowed.
        assert!(refuse_the_source("http://127.0.0.1:6340/", "cb_experiment", &origin()).is_ok());
        assert!(refuse_the_source("http://qdrant-exp:6339/", "cb_experiment", &origin()).is_ok());
        // The same collection name, even on another server.
        let error = refuse_the_source("http://qdrant-exp:6333/", "kb_source", &origin()).unwrap_err();
        assert!(matches!(error, RunError::Refused { .. }), "{error}");
        assert!(error.to_string().contains("kb_source"));
        // The same server, even under another collection name, by any of its names.
        for same in [
            "http://127.0.0.1:6339",
            "http://localhost:6339/",
            "http://[::1]:6339/x",
        ] {
            let error = refuse_the_source(same, "cb_experiment", &origin()).unwrap_err();
            assert!(matches!(error, RunError::Refused { .. }), "{same}: {error}");
        }
        // An address that cannot be read is not waved through.
        assert!(refuse_the_source("not an address", "cb_experiment", &origin()).is_err());
    }

    #[test]
    fn a_point_line_round_trips() {
        let built = build_control_set(points(), &[], settings(&[]));
        let line = serde_json::to_string(&built.points[2]).unwrap();
        let back: ExportedPoint = serde_json::from_str(&line).unwrap();
        assert_eq!(back, built.points[2]);
        assert!(line.contains("\"heading\":\"Invoices\""));
    }
}
