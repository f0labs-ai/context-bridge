//! `cb-run newest`: for each question, the newest records the index could show when it
//! was asked, each with the relevance model's score, written as one `Newest` line
//! A pass of its own beside `candidates`, whose
//! lines stay as they are: the lists by meaning and by keywords are not asked again.

use super::candidates::{
    RecordTimes, check_times, records_by_best_position, records_by_best_score, records_with_an_upload_line,
    times_of,
};
use super::upload::UPLOADS_FILE;
use super::{
    Shared, about_knowledge_service, ask_what_the_service_is, belongs_to_records, in_order, read_queries,
    read_records,
};
use crate::error::RunError;
use crate::jsonl::{Appender, CutOffLine, read_resumable};
use crate::knowledge::{
    CandidatesAnswer, CandidatesRequest, KnowledgeService, Label, RerankRequest, Tenant, ViewRequest,
    features,
};
use crate::retry::with_retries;
use crate::run::{Report, Session};
use cb_types::{Newest, Query, Record, UnixSeconds, views};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

pub const NEWEST_FILE: &str = "newest.jsonl";
const COMMAND: &str = "newest";

/// Passages of the newest list asked for, and scored, per question.
pub const NEWEST_PASSAGES: u32 = 32;

#[derive(clap::Args, Clone, Debug, Serialize)]
pub struct NewestArgs {
    /// The data set directory; its queries.jsonl and records.jsonl are read.
    #[arg(long)]
    pub data: PathBuf,
    /// The tenant of the knowledge service that holds the records.
    #[arg(long)]
    pub tenant: String,
    /// The collection label the records were uploaded under.
    #[arg(long)]
    pub collection: String,
    /// The run directory; newest.jsonl is appended to.
    #[arg(long)]
    pub run: PathBuf,
    /// Questions in flight at once. With 1, lines are written in question order.
    #[arg(long, default_value_t = 1)]
    pub concurrency: usize,
}

/// The one request made for a question: the `now` view's newest passages, all of them
/// scored. The route always answers the two lists too; one passage each is the least it
/// takes, and they are not read.
pub fn request_for(query: &Query, collection: &Label) -> CandidatesRequest {
    CandidatesRequest {
        query: query.text.clone(),
        as_of: query.asked_at,
        n: 1,
        exact: true,
        collections: vec![collection.as_str().to_string()],
        views: vec![ViewRequest {
            name: views::NOW.to_string(),
            index_as_of: None,
            written_from: None,
        }],
        newest: Some(NEWEST_PASSAGES),
        rerank: RerankRequest {
            view: views::NOW.to_string(),
            per_list: 0,
            newest: NEWEST_PASSAGES,
        },
    }
}

/// The line of one question, read strictly from the service's answer.
///
/// Refused: an answer with another view than `now`, without a newest list, with a newest
/// list that is not newest first, with a passage written after the question, holding
/// other times than the data set, or with a relevance score for a record the newest list
/// does not hold.
pub fn line_for(
    query: &Query,
    answer: &CandidatesAnswer,
    known: &HashMap<&str, RecordTimes>,
) -> Result<Newest, String> {
    if let Some(extra) = answer.views.keys().find(|name| name.as_str() != views::NOW) {
        return Err(format!(
            "the service answered a view that was not asked for: {extra}"
        ));
    }
    let now = answer
        .views
        .get(views::NOW)
        .ok_or_else(|| format!("the service did not answer view {}", views::NOW))?;
    let list = now
        .newest
        .as_deref()
        .ok_or("the service answered no newest list (a build without eval_newest?)")?;
    if list.windows(2).any(|pair| pair[1].score > pair[0].score) {
        return Err("the newest list is not newest first".into());
    }
    if let Some(late) = list.iter().find(|point| point.score > query.asked_at as f64) {
        return Err(format!(
            "point {} of the newest list was written at {}, after the question ({})",
            late.id, late.score, query.asked_at
        ));
    }
    check_times(&answer.passages, known)?;
    let newest = records_by_best_position(list, &answer.passages)?;
    let rerank = records_by_best_score(&answer.rerank, &answer.passages)?;
    let listed: HashSet<&str> = newest.iter().map(|r| r.record.as_str()).collect();
    if let Some(stray) = rerank.iter().find(|r| !listed.contains(r.record.as_str())) {
        return Err(format!(
            "a relevance score for record {}, which the newest list does not hold",
            stray.record
        ));
    }
    Ok(Newest {
        query: query.id.clone(),
        as_of: query.asked_at,
        newest,
        rerank,
        timings_ms: answer.timings_ms.clone(),
    })
}

/// What is read of a line that is already in the file.
#[derive(Debug, Deserialize)]
struct LineKey {
    query: String,
    as_of: UnixSeconds,
}

/// Collects the newest lists of the questions `newest.jsonl` does not hold yet.
pub async fn newest<K: KnowledgeService>(
    kb: &K,
    args: &NewestArgs,
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
        Ok(have) => session.finish(|id| have.contains(id)),
        Err(error) => {
            session.stop(&error);
            Err(error)
        }
    }
}

async fn ask<K: KnowledgeService>(
    kb: &K,
    tenant: &Tenant,
    query: &Query,
    request: &CandidatesRequest,
    known: &HashMap<&str, RecordTimes>,
) -> Result<Newest, RunError> {
    let answer = kb.candidates(tenant, request).await?;
    line_for(query, &answer.value, known).map_err(|reason| RunError::Contract {
        id: query.id.clone(),
        url: kb.address(),
        reason,
    })
}

async fn work<K: KnowledgeService>(
    kb: &K,
    args: &NewestArgs,
    shared: &Shared,
    (tenant, label): (&Tenant, &Label),
    (queries, records): (&[Query], &[Record]),
    session: &mut Session,
) -> Result<HashSet<String>, RunError> {
    session.claim(
        NEWEST_FILE,
        belongs_to_records(kb, &args.data, tenant.as_str(), label.as_str())?,
    )?;
    ask_what_the_service_is(
        kb,
        shared,
        session,
        &[
            features::EVAL_CANDIDATES,
            features::EVAL_NEWEST,
            features::DOCUMENT_TIMES,
        ],
    )
    .await?;
    session.set("records_in_data_set", records.len() as u64);
    session.set("newest_passages_asked", u64::from(NEWEST_PASSAGES));
    if let Some(uploaded) = records_with_an_upload_line(&session.file(UPLOADS_FILE))? {
        let here = records.iter().filter(|r| uploaded.contains(&r.id)).count();
        session.set("records_uploaded_in_this_run", here as u64);
    }
    let path = session.file(NEWEST_FILE);
    let existing = read_resumable::<LineKey>(&path, CutOffLine::Remove)?;
    session.set("cut_off_bytes_removed", existing.cut_off_bytes);
    let asked_at: HashMap<&str, UnixSeconds> = queries.iter().map(|q| (q.id.as_str(), q.asked_at)).collect();
    let mut have: HashSet<String> = HashSet::with_capacity(existing.lines.len());
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
        if !have.insert(line.query.clone()) {
            return Err(RunError::file(
                &path,
                format!("question {} is in the file twice", line.query),
            ));
        }
    }
    let mut out = Appender::open(&path)?;
    let known = times_of(records);

    let mut todo = Vec::new();
    for query in queries {
        if have.contains(&query.id) {
            session.add("already_there", 1);
        } else {
            todo.push((query, request_for(query, label)));
        }
    }

    let (retry, known) = (&shared.retry, &known);
    in_order(
        todo,
        args.concurrency,
        |(query, request)| async move {
            let outcome = with_retries(retry, |_| ask(kb, tenant, query, &request, known)).await;
            (query, outcome)
        },
        |(query, outcome)| {
            match outcome {
                Ok(tried) => {
                    out.append(std::slice::from_ref(&tried.value))?;
                    have.insert(query.id.clone());
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
    use crate::knowledge::{PassageInfo, PointScore, ViewLists};
    use cb_types::{QueryClass, Scored};

    fn query(asked_at: UnixSeconds) -> Query {
        Query {
            id: "q-1".into(),
            text: "Is GitHub down?".into(),
            asked_at,
            class: QueryClass::Changing,
            gold: vec![],
            thread: None,
            harmful: vec![],
        }
    }

    fn point(id: &str, score: f64) -> PointScore {
        PointScore { id: id.into(), score }
    }

    fn passage(source_path: &str, written_at: i64) -> PassageInfo {
        PassageInfo {
            source_path: source_path.into(),
            kind: Some("prose".into()),
            written_at: Some(written_at),
            valid_until: None,
            chars: Some(100),
        }
    }

    fn scored(record: &str, score: f64) -> Scored {
        Scored {
            record: record.into(),
            score,
        }
    }

    /// Three passages: r-b's two (written at 300) and r-a's one (at 200).
    fn answer(newest: Vec<PointScore>, rerank: Vec<PointScore>) -> CandidatesAnswer {
        CandidatesAnswer {
            views: [(
                "now".to_string(),
                ViewLists {
                    dense: vec![],
                    sparse: vec![],
                    newest: Some(newest),
                },
            )]
            .into(),
            rerank,
            passages: [
                ("p1".to_string(), passage("cb/r-b.md", 300)),
                ("p2".to_string(), passage("cb/r-b.md", 300)),
                ("p3".to_string(), passage("cb/r-a.md", 200)),
            ]
            .into(),
            timings_ms: [("rerank".to_string(), 12.5)].into(),
        }
    }

    fn known() -> HashMap<&'static str, RecordTimes> {
        [
            (
                "r-a",
                RecordTimes {
                    written_at: 200,
                    valid_until: None,
                },
            ),
            (
                "r-b",
                RecordTimes {
                    written_at: 300,
                    valid_until: None,
                },
            ),
        ]
        .into()
    }

    #[test]
    fn the_request_asks_for_the_newest_passages_of_now_and_scores_them_alone() {
        let label = Label::parse("--collection", "cb").unwrap();
        let request = serde_json::to_value(request_for(&query(1_000), &label)).unwrap();
        assert_eq!(request["newest"], 32);
        assert_eq!(
            request["rerank"],
            serde_json::json!({"view": "now", "per_list": 0, "newest": 32})
        );
        assert_eq!(request["views"], serde_json::json!([{"name": "now"}]));
        assert_eq!(
            (request["as_of"].as_i64(), request["n"].as_i64()),
            (Some(1_000), Some(1))
        );
    }

    #[test]
    fn a_newest_list_becomes_records_newest_first_with_their_best_relevance() {
        let newest = vec![point("p1", 300.0), point("p2", 300.0), point("p3", 200.0)];
        let rerank = vec![point("p2", 0.6), point("p3", 0.3), point("p1", 0.2)];
        let line = line_for(&query(1_000), &answer(newest, rerank), &known()).unwrap();
        assert_eq!((line.query.as_str(), line.as_of), ("q-1", 1_000));
        assert_eq!(line.newest, vec![scored("r-b", 300.0), scored("r-a", 200.0)]);
        // r-b's better passage gives its relevance.
        assert_eq!(line.rerank, vec![scored("r-b", 0.6), scored("r-a", 0.3)]);
        assert_eq!(line.timings_ms.get("rerank"), Some(&12.5));
    }

    #[test]
    fn an_answer_that_breaks_the_contract_is_refused() {
        let refused = |answer: CandidatesAnswer, asked_at: UnixSeconds| {
            line_for(&query(asked_at), &answer, &known()).unwrap_err()
        };
        let good = || vec![point("p1", 300.0), point("p3", 200.0)];
        // Not newest first.
        let error = refused(
            answer(vec![point("p3", 200.0), point("p1", 300.0)], vec![]),
            1_000,
        );
        assert!(error.contains("not newest first"), "{error}");
        // A passage written after the question: the view was not "as of" it.
        let error = refused(answer(good(), vec![]), 250);
        assert!(error.contains("after the question"), "{error}");
        // A relevance score for a record outside the list.
        let mut stray = answer(vec![point("p3", 200.0)], vec![point("p1", 0.5)]);
        stray.passages.insert("p1".into(), passage("cb/r-b.md", 300));
        let error = refused(stray, 1_000);
        assert!(error.contains("record r-b"), "{error}");
        // No newest list: a service without the feature.
        let mut bare = answer(good(), vec![]);
        if let Some(now) = bare.views.get_mut("now") {
            now.newest = None;
        }
        let error = refused(bare, 1_000);
        assert!(error.contains("no newest list"), "{error}");
        // Another view than now.
        let mut extra = answer(good(), vec![]);
        extra.views.insert(
            "batch_300".into(),
            ViewLists {
                dense: vec![],
                sparse: vec![],
                newest: None,
            },
        );
        let error = refused(extra, 1_000);
        assert!(error.contains("batch_300"), "{error}");
        // Other times than the data set.
        let mut moved = known();
        moved.insert(
            "r-a",
            RecordTimes {
                written_at: 250,
                valid_until: None,
            },
        );
        let error = line_for(&query(1_000), &answer(good(), vec![]), &moved).unwrap_err();
        assert!(error.contains("record r-a"), "{error}");
    }
}
