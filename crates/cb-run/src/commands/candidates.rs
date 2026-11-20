//! `cb-run candidates`: for each question, the candidate lists behind a search, for the
//! seven views of DESIGN.md section 3.1, written as seven `Candidates` lines.

use super::upload::UPLOADS_FILE;
use super::{
    Shared, about_knowledge_service, ask_what_the_service_is, belongs_to_records, in_order, read_queries,
    read_records,
};
use crate::error::RunError;
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::knowledge::{
    CandidatesAnswer, CandidatesRequest, KnowledgeService, Label, PassageInfo, PointScore, RerankRequest,
    Tenant, ViewRequest, features,
};
use crate::retry::with_retries;
use crate::run::{Report, Session};
use cb_types::{Candidates, Query, Record, Scored, UnixSeconds, views};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

pub const CANDIDATES_FILE: &str = "candidates.jsonl";
const COMMAND: &str = "candidates";

/// Passages asked for per list (docs/RUN_TOOLS.md: `n` 100).
pub const LIST_LENGTH: u32 = 100;
/// Passages of each of the `now` view's two lists that get a relevance score.
pub const RERANK_PER_LIST: u32 = 64;

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct CandidatesArgs {
    /// The data set directory; its queries.jsonl and records.jsonl are read.
    #[arg(long)]
    pub data: PathBuf,
    /// The tenant of the knowledge service that holds the records.
    #[arg(long)]
    pub tenant: String,
    /// The collection label the records were uploaded under.
    #[arg(long)]
    pub collection: String,
    /// The run directory; candidates.jsonl is appended to.
    #[arg(long)]
    pub run: PathBuf,
    /// Questions in flight at once. With 1, lines are written in question order.
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,
}

/// The seven views of DESIGN.md section 3.1 for a question asked at `asked_at`, in the
/// order their lines are written: `now`, the three batch indexes, the three windows.
pub fn views_at(asked_at: UnixSeconds) -> Vec<ViewRequest> {
    let mut all = vec![ViewRequest {
        name: views::NOW.to_string(),
        index_as_of: None,
        written_from: None,
    }];
    for period in views::BATCH_PERIODS {
        all.push(ViewRequest {
            name: views::batch(period),
            // The last moment a batch index refreshed: floor(asked_at / Δ) * Δ.
            index_as_of: Some(views::batch_boundary(asked_at, period)),
            written_from: None,
        });
    }
    for window in views::WINDOWS {
        all.push(ViewRequest {
            name: views::window(window),
            index_as_of: None,
            // The view sees `asked_at - W < written_at <= asked_at`.
            written_from: Some(asked_at.saturating_sub(window)),
        });
    }
    all
}

/// The one request made for a question.
pub fn request_for(query: &Query, collection: &Label) -> CandidatesRequest {
    CandidatesRequest {
        query: query.text.clone(),
        as_of: query.asked_at,
        n: LIST_LENGTH,
        exact: true,
        collections: vec![collection.as_str().to_string()],
        views: views_at(query.asked_at),
        // The newest list is a pass of its own (`cb-run newest`).
        newest: None,
        rerank: RerankRequest {
            view: views::NOW.to_string(),
            per_list: RERANK_PER_LIST,
            newest: 0,
        },
    }
}

/// The record a point belongs to: the record its document was uploaded for
/// (`cb/<record id>.md`), or — for a passage of any other document — the point itself.
pub fn record_of_point(point_id: &str, passage: &PassageInfo) -> String {
    cb_types::ids::record_of(&passage.source_path).map_or_else(|| point_id.to_string(), str::to_string)
}

fn described<'a>(
    point: &PointScore,
    passages: &'a HashMap<String, PassageInfo>,
) -> Result<&'a PassageInfo, String> {
    passages
        .get(&point.id)
        .ok_or_else(|| format!("point {} is in a list and not described in `passages`", point.id))
}

/// A ranked list of points as a ranked list of records: each record once, at the
/// position (and with the score) of its best-placed passage.
pub fn records_by_best_position(
    list: &[PointScore],
    passages: &HashMap<String, PassageInfo>,
) -> Result<Vec<Scored>, String> {
    let mut seen = HashSet::new();
    let mut records = Vec::with_capacity(list.len());
    for point in list {
        let record = record_of_point(&point.id, described(point, passages)?);
        if seen.insert(record.clone()) {
            records.push(Scored {
                record,
                score: point.score,
            });
        }
    }
    Ok(records)
}

/// Relevance scores of points as relevance scores of records: each record once, with the
/// score of its best-scored passage, in the order the service gave. (In a list sorted by
/// score this is the record's best position; it also holds when the list is not sorted.)
pub fn records_by_best_score(
    list: &[PointScore],
    passages: &HashMap<String, PassageInfo>,
) -> Result<Vec<Scored>, String> {
    let mut best: HashMap<String, usize> = HashMap::new();
    let mut records = Vec::with_capacity(list.len());
    for (i, point) in list.iter().enumerate() {
        let record = record_of_point(&point.id, described(point, passages)?);
        match best.get(&record) {
            Some(held) if list[*held].score >= point.score => {}
            _ => {
                best.insert(record.clone(), i);
            }
        }
        records.push(record);
    }
    Ok(records
        .into_iter()
        .enumerate()
        .filter(|(i, record)| best.get(record) == Some(i))
        .map(|(i, record)| Scored {
            record,
            score: list[i].score,
        })
        .collect())
}

/// What the data set says about when a record became known and when it ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordTimes {
    pub written_at: UnixSeconds,
    pub valid_until: Option<UnixSeconds>,
}

pub fn times_of(records: &[Record]) -> HashMap<&str, RecordTimes> {
    records
        .iter()
        .map(|r| {
            (
                r.id.as_str(),
                RecordTimes {
                    written_at: r.written_at,
                    valid_until: r.valid_until,
                },
            )
        })
        .collect()
}

/// Checks that the index holds, for every record of the data set that appears in an
/// answer, the times the data set says. Views are cut by these times inside the service;
/// an index loaded from another build of the data set would give lists that look right
/// and are not.
pub(crate) fn check_times(
    passages: &HashMap<String, PassageInfo>,
    known: &HashMap<&str, RecordTimes>,
) -> Result<(), String> {
    let mut points: Vec<&String> = passages.keys().collect();
    points.sort();
    for point in points {
        let Some(passage) = passages.get(point) else {
            continue;
        };
        let record = record_of_point(point, passage);
        let Some(times) = known.get(record.as_str()) else {
            continue;
        };
        if passage.written_at != Some(times.written_at) {
            return Err(format!(
                "the index holds written_at {:?} for record {record}; the data set says {}",
                passage.written_at, times.written_at
            ));
        }
        if passage.valid_until != times.valid_until {
            return Err(format!(
                "the index holds valid_until {:?} for record {record}; the data set says {:?}",
                passage.valid_until, times.valid_until
            ));
        }
    }
    Ok(())
}

/// The lines of one question, one per asked view, in the order the views were asked.
///
/// Only the view named in `rerank` carries relevance scores. The service reports one set
/// of timings per request; it is written on that same line and nowhere else, so that
/// adding up a file's lines counts each request once.
pub fn lines_for(
    query: &Query,
    request: &CandidatesRequest,
    answer: &CandidatesAnswer,
    known: &HashMap<&str, RecordTimes>,
) -> Result<Vec<Candidates>, String> {
    if let Some(extra) = answer
        .views
        .keys()
        .find(|name| !request.views.iter().any(|asked| &asked.name == *name))
    {
        return Err(format!(
            "the service answered a view that was not asked for: {extra}"
        ));
    }
    check_times(&answer.passages, known)?;
    let mut lines = Vec::with_capacity(request.views.len());
    for view in &request.views {
        let lists = answer
            .views
            .get(&view.name)
            .ok_or_else(|| format!("the service did not answer view {}", view.name))?;
        let scored = view.name == request.rerank.view;
        lines.push(Candidates {
            query: query.id.clone(),
            view: view.name.clone(),
            as_of: query.asked_at,
            dense: records_by_best_position(&lists.dense, &answer.passages)?,
            sparse: records_by_best_position(&lists.sparse, &answer.passages)?,
            rerank: if scored {
                records_by_best_score(&answer.rerank, &answer.passages)?
            } else {
                Vec::new()
            },
            timings_ms: if scored {
                answer.timings_ms.clone()
            } else {
                BTreeMap::new()
            },
        });
    }
    Ok(lines)
}

/// What is read of a line that is already in the file: its key, and the moment it was
/// asked as of.
#[derive(Debug, Deserialize)]
struct LineKey {
    query: String,
    view: String,
    as_of: UnixSeconds,
}

/// The records `uploads.jsonl` holds a line for; none when the file is absent. The file
/// is `upload`'s and is only read here: a cut-off last line is passed over, not removed.
pub(crate) fn records_with_an_upload_line(
    path: &std::path::Path,
) -> Result<Option<HashSet<String>>, RunError> {
    #[derive(Deserialize)]
    struct Uploaded {
        record: String,
    }
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(RunError::io(path, e)),
    };
    Ok(Some(
        text.lines()
            .filter_map(|line| serde_json::from_str::<Uploaded>(line).ok())
            .map(|line| line.record)
            .collect(),
    ))
}

/// Collects the candidate lists of the questions `candidates.jsonl` does not hold yet.
pub async fn candidates<K: KnowledgeService>(
    kb: &K,
    args: &CandidatesArgs,
    shared: &Shared,
) -> Result<Report, RunError> {
    let tenant = Tenant::parse(&args.tenant)?;
    let label = Label::parse("--collection", &args.collection)?;
    let queries = read_queries(&args.data)?;
    let records = read_records(&args.data)?;
    let mut session = Session::begin(&args.run, COMMAND, args, about_knowledge_service(kb, shared))?;
    match work(
        kb,
        args,
        shared,
        (&tenant, &label),
        (&queries, &records),
        &mut session,
    )
    .await
    {
        Ok(have) => session.finish(|id| {
            views_at(0)
                .iter()
                .all(|view| have.contains(&(id.to_string(), view.name.clone())))
        }),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

/// One question's lists: the request, then the answer read strictly.
async fn ask<K: KnowledgeService>(
    kb: &K,
    tenant: &Tenant,
    query: &Query,
    request: &CandidatesRequest,
    known: &HashMap<&str, RecordTimes>,
) -> Result<Vec<Candidates>, RunError> {
    let answer = kb.candidates(tenant, request).await?;
    lines_for(query, request, &answer.value, known).map_err(|reason| RunError::Contract {
        id: query.id.clone(),
        url: kb.address(),
        reason,
    })
}

async fn work<K: KnowledgeService>(
    kb: &K,
    args: &CandidatesArgs,
    shared: &Shared,
    (tenant, label): (&Tenant, &Label),
    (queries, records): (&[Query], &[Record]),
    session: &mut Session,
) -> Result<HashSet<(String, String)>, RunError> {
    session.claim(
        CANDIDATES_FILE,
        belongs_to_records(kb, &args.data, tenant.as_str(), label.as_str())?,
    )?;
    ask_what_the_service_is(
        kb,
        shared,
        session,
        &[features::EVAL_CANDIDATES, features::DOCUMENT_TIMES],
    )
    .await?;
    // Keyword scores use the word statistics of the whole index at the moment of the
    // request (the service's finding 3): lists are only comparable when every record was
    // loaded first. Said in the manifest, from this run directory's own upload file.
    session.set("records_in_data_set", records.len() as u64);
    if let Some(uploaded) = records_with_an_upload_line(&session.file(UPLOADS_FILE))? {
        let here = records.iter().filter(|r| uploaded.contains(&r.id)).count();
        session.set("records_uploaded_in_this_run", here as u64);
    }
    let path = session.file(CANDIDATES_FILE);
    let existing = read_resumable::<LineKey>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let asked_at: HashMap<&str, UnixSeconds> = queries.iter().map(|q| (q.id.as_str(), q.asked_at)).collect();
    let mut have: HashSet<(String, String)> = HashSet::with_capacity(existing.lines.len());
    for line in existing.lines {
        match asked_at.get(line.query.as_str()) {
            // The same id asked at another moment: the lines are another data set's.
            Some(at) if *at != line.as_of => {
                return Err(RunError::file(
                    &path,
                    format!(
                        "question {} is in the file as of {} and the data set asks it at {at}; \
                         the file belongs to another build of the data set: use another --run",
                        line.query, line.as_of
                    ),
                ));
            }
            Some(_) => {}
            None => session.add("lines_of_questions_not_in_the_data_set", 1),
        }
        if !have.insert((line.query.clone(), line.view.clone())) {
            return Err(RunError::file(
                &path,
                format!("question {} has view {} twice", line.query, line.view),
            ));
        }
    }
    let mut out = Appender::open(&path)?;
    let known = times_of(records);

    // A question is asked when any of its seven lines is missing; only the missing lines
    // are then written, so no (question, view) can appear twice.
    let mut todo = Vec::new();
    for query in queries {
        let request = request_for(query, label);
        let missing: Vec<String> = request
            .views
            .iter()
            .filter(|view| !have.contains(&(query.id.clone(), view.name.clone())))
            .map(|view| view.name.clone())
            .collect();
        if missing.is_empty() {
            session.add("already_there", 1);
        } else {
            todo.push((query, request, missing));
        }
    }

    let (retry, known) = (&shared.retry, &known);
    in_order(
        todo,
        args.concurrency,
        |(query, request, missing)| async move {
            let outcome = with_retries(retry, |_| ask(kb, tenant, query, &request, known)).await;
            (query, missing, outcome)
        },
        |(query, missing, outcome)| {
            match outcome {
                Ok(tried) => {
                    let lines: Vec<Candidates> = tried
                        .value
                        .into_iter()
                        .filter(|line| missing.contains(&line.view))
                        .collect();
                    out.append(&lines)?;
                    session.add("lines_written", lines.len() as u64);
                    for line in lines {
                        have.insert((line.query, line.view));
                    }
                    session.wrote();
                }
                Err(gave_up) => session.fail(&query.id, &gave_up.error, gave_up.tries)?,
            }
            Ok(())
        },
    )
    .await?;
    out.sync()?;
    Ok(have)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::QueryClass;

    fn query(asked_at: UnixSeconds) -> Query {
        Query {
            id: "q-1".into(),
            text: "Is home internet down in Northgate?".into(),
            asked_at,
            class: QueryClass::Changing,
            gold: vec![],
            thread: Some("inc-0001".into()),
            harmful: vec![],
        }
    }

    fn passage(source_path: &str, written_at: Option<i64>, valid_until: Option<i64>) -> PassageInfo {
        PassageInfo {
            source_path: source_path.into(),
            kind: Some("prose".into()),
            written_at,
            valid_until,
            chars: Some(100),
        }
    }

    fn point(id: &str, score: f64) -> PointScore {
        PointScore { id: id.into(), score }
    }

    fn scored(record: &str, score: f64) -> Scored {
        Scored {
            record: record.into(),
            score,
        }
    }

    #[test]
    fn the_seven_views_are_built_exactly_as_the_design_says() {
        // 1_780_275_123 = 494_520 * 3600 + 3123: 3123 s past the hour.
        let asked_at = 1_780_275_123;
        let views = views_at(asked_at);
        let shown: Vec<(String, Option<i64>, Option<i64>)> = views
            .iter()
            .map(|v| (v.name.clone(), v.index_as_of, v.written_from))
            .collect();
        assert_eq!(
            shown,
            vec![
                ("now".to_string(), None, None),
                ("batch_300".to_string(), Some(1_780_275_000), None),
                ("batch_1800".to_string(), Some(1_780_273_800), None),
                ("batch_3600".to_string(), Some(1_780_272_000), None),
                ("window_3600".to_string(), None, Some(1_780_271_523)),
                ("window_14400".to_string(), None, Some(1_780_260_723)),
                ("window_86400".to_string(), None, Some(1_780_188_723)),
            ]
        );
        for view in &views {
            if let Some(boundary) = view.index_as_of {
                assert!(boundary <= asked_at);
            }
        }
        // A question asked exactly on a boundary sees that boundary's index.
        assert_eq!(views_at(1_780_272_000)[3].index_as_of, Some(1_780_272_000));
    }

    #[test]
    fn the_request_is_the_one_the_contract_describes() {
        let label = Label::parse("--collection", "cb").unwrap();
        let request = serde_json::to_value(request_for(&query(1_780_275_000), &label)).unwrap();
        assert_eq!(request["query"], "Is home internet down in Northgate?");
        assert_eq!(request["as_of"], 1_780_275_000);
        assert_eq!(request["n"], 100);
        assert_eq!(request["exact"], true);
        assert_eq!(request["collections"], serde_json::json!(["cb"]));
        assert_eq!(
            request["rerank"],
            serde_json::json!({"view": "now", "per_list": 64})
        );
        // No newest list is asked for here: a service without `eval_newest` takes the request.
        assert!(request.get("newest").is_none());
        let views = request["views"].as_array().unwrap();
        assert_eq!(views.len(), 7);
        assert_eq!(views[0], serde_json::json!({"name": "now"}));
        assert_eq!(
            views[1],
            serde_json::json!({"name": "batch_300", "index_as_of": 1_780_275_000})
        );
        assert_eq!(
            views[4],
            serde_json::json!({"name": "window_3600", "written_from": 1_780_271_400})
        );
    }

    #[test]
    fn a_point_is_joined_to_its_record_through_its_document_path() {
        assert_eq!(
            record_of_point("p-1", &passage("cb/inc-0001.s2.md", None, None)),
            "inc-0001.s2"
        );
        // A passage of any other document keeps its point id as its record id.
        assert_eq!(
            record_of_point(
                "d7528b29-4637-55f9-ad70-29980bbd48b5",
                &passage("https://docs.example.com/a", None, None)
            ),
            "d7528b29-4637-55f9-ad70-29980bbd48b5"
        );
        assert_eq!(
            record_of_point("p-3", &passage("cb/Not An Id.md", None, None)),
            "p-3"
        );
    }

    #[test]
    fn a_record_with_several_passages_keeps_its_best_position() {
        let passages: HashMap<String, PassageInfo> = [
            ("p1", "cb/r-b.md"),
            ("p2", "cb/r-a.md"),
            ("p3", "cb/r-b.md"),
            ("p4", "https://docs.example.com/a"),
            ("p5", "cb/r-a.md"),
        ]
        .into_iter()
        .map(|(id, path)| (id.to_string(), passage(path, None, None)))
        .collect();
        let list = [
            point("p1", 0.9),
            point("p2", 0.8),
            point("p3", 0.7),
            point("p4", 0.6),
            point("p5", 0.5),
        ];
        assert_eq!(
            records_by_best_position(&list, &passages).unwrap(),
            vec![scored("r-b", 0.9), scored("r-a", 0.8), scored("p4", 0.6)]
        );
        assert_eq!(records_by_best_position(&[], &passages).unwrap(), vec![]);
        // A point the answer does not describe cannot be joined, and is not guessed.
        let error = records_by_best_position(&[point("p9", 0.1)], &passages).unwrap_err();
        assert!(error.contains("p9"), "{error}");
    }

    #[test]
    fn a_records_relevance_is_its_best_passages_whatever_the_order() {
        let passages: HashMap<String, PassageInfo> = [
            ("p1", "cb/r-a.md"),
            ("p2", "cb/r-b.md"),
            ("p3", "cb/r-a.md"),
            ("p4", "cb/r-b.md"),
        ]
        .into_iter()
        .map(|(id, path)| (id.to_string(), passage(path, None, None)))
        .collect();
        // Not sorted by score: r-a's better passage comes later.
        let list = [
            point("p1", 0.2),
            point("p2", 0.6),
            point("p3", 0.9),
            point("p4", 0.1),
        ];
        assert_eq!(
            records_by_best_score(&list, &passages).unwrap(),
            vec![scored("r-b", 0.6), scored("r-a", 0.9)]
        );
        // Sorted by score: the same as the best position.
        let sorted = [
            point("p3", 0.9),
            point("p2", 0.6),
            point("p1", 0.2),
            point("p4", 0.1),
        ];
        assert_eq!(
            records_by_best_score(&sorted, &passages).unwrap(),
            records_by_best_position(&sorted, &passages).unwrap()
        );
    }

    fn answer() -> CandidatesAnswer {
        let asked = query(1_780_275_000);
        let request = request_for(&asked, &Label::parse("--collection", "cb").unwrap());
        let views = request
            .views
            .iter()
            .map(|view| {
                let lists = if view.name == "now" {
                    crate::knowledge::ViewLists {
                        dense: vec![point("p1", 0.83), point("p2", 0.80)],
                        sparse: vec![point("p2", 11.2)],
                        newest: None,
                    }
                } else {
                    crate::knowledge::ViewLists {
                        dense: vec![],
                        sparse: vec![],
                        newest: None,
                    }
                };
                (view.name.clone(), lists)
            })
            .collect();
        CandidatesAnswer {
            views,
            rerank: vec![point("p2", 0.41), point("p1", 0.30)],
            passages: [
                ("p1".to_string(), passage("cb/inc-0001.s1.md", Some(100), None)),
                (
                    "p2".to_string(),
                    passage("cb/inc-0001.s2.md", Some(200), Some(900)),
                ),
            ]
            .into(),
            timings_ms: [("embed".to_string(), 7.1), ("search".to_string(), 9.4)].into(),
        }
    }

    fn known() -> HashMap<&'static str, RecordTimes> {
        [
            (
                "inc-0001.s1",
                RecordTimes {
                    written_at: 100,
                    valid_until: None,
                },
            ),
            (
                "inc-0001.s2",
                RecordTimes {
                    written_at: 200,
                    valid_until: Some(900),
                },
            ),
        ]
        .into()
    }

    #[test]
    fn one_answer_becomes_seven_lines_in_view_order_and_only_now_is_scored() {
        let asked = query(1_780_275_000);
        let request = request_for(&asked, &Label::parse("--collection", "cb").unwrap());
        let lines = lines_for(&asked, &request, &answer(), &known()).unwrap();
        let names: Vec<&str> = lines.iter().map(|l| l.view.as_str()).collect();
        assert_eq!(
            names,
            [
                "now",
                "batch_300",
                "batch_1800",
                "batch_3600",
                "window_3600",
                "window_14400",
                "window_86400"
            ]
        );
        assert!(lines.iter().all(|l| l.query == "q-1" && l.as_of == 1_780_275_000));
        assert_eq!(
            lines[0].dense,
            vec![scored("inc-0001.s1", 0.83), scored("inc-0001.s2", 0.80)]
        );
        assert_eq!(lines[0].sparse, vec![scored("inc-0001.s2", 11.2)]);
        assert_eq!(
            lines[0].rerank,
            vec![scored("inc-0001.s2", 0.41), scored("inc-0001.s1", 0.30)]
        );
        assert_eq!(lines[0].timings_ms.get("embed"), Some(&7.1));
        for line in &lines[1..] {
            assert!(line.dense.is_empty() && line.sparse.is_empty() && line.rerank.is_empty());
            assert!(line.timings_ms.is_empty(), "one request, one set of timings");
        }
    }

    #[test]
    fn an_answer_with_a_view_missing_or_too_many_is_refused() {
        let asked = query(1_780_275_000);
        let request = request_for(&asked, &Label::parse("--collection", "cb").unwrap());
        let mut missing = answer();
        missing.views.remove("window_86400");
        let error = lines_for(&asked, &request, &missing, &known()).unwrap_err();
        assert!(error.contains("did not answer view window_86400"), "{error}");
        let mut extra = answer();
        extra.views.insert(
            "batch_60".into(),
            crate::knowledge::ViewLists {
                dense: vec![],
                sparse: vec![],
                newest: None,
            },
        );
        let error = lines_for(&asked, &request, &extra, &known()).unwrap_err();
        assert!(error.contains("not asked for: batch_60"), "{error}");
    }

    #[test]
    fn an_index_that_holds_other_times_than_the_data_set_is_refused() {
        let asked = query(1_780_275_000);
        let request = request_for(&asked, &Label::parse("--collection", "cb").unwrap());
        let mut other = known();
        other.insert(
            "inc-0001.s2",
            RecordTimes {
                written_at: 250,
                valid_until: Some(900),
            },
        );
        let error = lines_for(&asked, &request, &answer(), &other).unwrap_err();
        assert!(
            error.contains("written_at Some(200) for record inc-0001.s2"),
            "{error}"
        );
        let mut ends = known();
        ends.insert(
            "inc-0001.s1",
            RecordTimes {
                written_at: 100,
                valid_until: Some(5),
            },
        );
        let error = lines_for(&asked, &request, &answer(), &ends).unwrap_err();
        assert!(
            error.contains("valid_until None for record inc-0001.s1"),
            "{error}"
        );
        // A record the data set does not hold is not judged.
        assert!(lines_for(&asked, &request, &answer(), &HashMap::new()).is_ok());
    }
}
