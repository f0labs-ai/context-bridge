//! A tiny data set and run, taken through the binary: `rank`, then `tune`,
//! then `report`, with the numbers checked against values worked out by hand.
//!
//! Three incidents, each with three states (s1, s2, s3) and one planted wrong
//! statement (d1), and three help sections. Thread inc-a is in the validation
//! split, inc-b and inc-c in the test split; the ordinary questions q-ord-1 and
//! q-ord-2 are test questions and q-ord-6 a validation question (FNV-1a of the
//! name modulo 5). Set CB_EVAL_E2E_KEEP=1 to keep the files for a look.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use cb_eval::context::View;
use cb_eval::files::write_vectors;
use cb_eval::vectors::VectorIndex;
use cb_types::{
    Candidates, Confidence, Extraction, Gold, Phase, PipelineTrip, Query, QueryClass, Ranking, Record,
    RecordTruth, Scored, Speaker, Split, Tier, Transcript, Turn, split_of, write_jsonl,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// A moment divisible by 3 600, so the batch views fall on whole hours.
const T0: i64 = 1_779_998_400;
const INCIDENTS: [(&str, i64); 3] = [("inc-a", T0), ("inc-b", T0 + 10_800), ("inc-c", T0 + 21_600)];
const ORDINARY_ASKED: i64 = T0 + 25_000;
const DOCS_WRITTEN: i64 = T0 - 100_000;

fn record(id: &str, written_at: i64, thread: Option<&str>, truth: Option<RecordTruth>) -> Record {
    Record {
        id: id.to_string(),
        text: format!("text of {id}"),
        written_at,
        valid_until: None,
        thread: thread.map(str::to_string),
        tier: if thread.is_some() {
            Tier::Changing
        } else {
            Tier::Standing
        },
        title: None,
        source: "end-to-end".to_string(),
        truth,
    }
}

fn truth(correct: bool, superseded_at: Option<i64>) -> Option<RecordTruth> {
    Some(RecordTruth {
        correct,
        phase: Phase::Investigating,
        key_terms: Vec::new(),
        superseded_at,
    })
}

fn records() -> Vec<Record> {
    let mut out = Vec::new();
    for (thread, start) in INCIDENTS {
        let id = |state: &str| format!("{thread}.{state}");
        out.push(record(
            &id("s1"),
            start + 100,
            Some(thread),
            truth(true, Some(start + 1_000)),
        ));
        out.push(record(&id("d1"), start + 500, Some(thread), truth(false, None)));
        out.push(record(
            &id("s2"),
            start + 1_000,
            Some(thread),
            truth(true, Some(start + 3_000)),
        ));
        out.push(record(&id("s3"), start + 3_000, Some(thread), truth(true, None)));
    }
    for doc in ["doc-1.1", "doc-1.2", "doc-2.1"] {
        out.push(record(doc, DOCS_WRITTEN, None, None));
    }
    out
}

fn query(
    id: &str,
    asked_at: i64,
    class: QueryClass,
    gold: &[(&str, u8)],
    thread: Option<&str>,
    harmful: &[String],
) -> Query {
    Query {
        id: id.to_string(),
        text: format!("question {id}"),
        asked_at,
        class,
        gold: gold
            .iter()
            .map(|(record, grade)| Gold {
                record: record.to_string(),
                grade: *grade,
            })
            .collect(),
        thread: thread.map(str::to_string),
        harmful: harmful.to_vec(),
    }
}

fn scored(pairs: &[(String, f64)]) -> Vec<Scored> {
    pairs
        .iter()
        .map(|(record, score)| Scored {
            record: record.clone(),
            score: *score,
        })
        .collect()
}

/// The questions with their `now` lists: (question, dense, sparse, relevance scores).
type Asked = (Query, Vec<(String, f64)>, Vec<(String, f64)>, Vec<(String, f64)>);

fn questions() -> Vec<Asked> {
    let mut out = Vec::new();
    for (thread, start) in INCIDENTS {
        let id = |state: &str| format!("{thread}.{state}");
        let harmful = vec![id("d1")];
        // q1, asked while s1 is the latest state: the wrong statement d1 leads the dense list.
        out.push((
            query(
                &format!("{thread}-q1"),
                start + 800,
                QueryClass::Changing,
                &[(&id("s1"), 3)],
                Some(thread),
                &harmful,
            ),
            vec![(id("d1"), 0.82), (id("s1"), 0.80), ("doc-1.1".to_string(), 0.40)],
            vec![(id("s1"), 9.0), (id("d1"), 7.0)],
            vec![(id("s1"), 0.70), (id("d1"), 0.55), ("doc-1.1".to_string(), 0.05)],
        ));
        // q2, asked while s3 is the latest state: the oldest state leads the dense list.
        out.push((
            query(
                &format!("{thread}-q2"),
                start + 4_000,
                QueryClass::Changing,
                &[(&id("s3"), 3), (&id("s2"), 1), (&id("s1"), 1)],
                Some(thread),
                &harmful,
            ),
            vec![
                (id("s1"), 0.85),
                (id("s2"), 0.83),
                (id("s3"), 0.81),
                (id("d1"), 0.75),
                ("doc-1.1".to_string(), 0.40),
            ],
            vec![(id("s2"), 8.0), (id("s3"), 7.5), (id("s1"), 6.0)],
            vec![
                (id("s1"), 0.62),
                (id("s2"), 0.64),
                (id("s3"), 0.60),
                (id("d1"), 0.30),
                ("doc-1.1".to_string(), 0.05),
            ],
        ));
    }
    let named = |pairs: &[(&str, f64)]| -> Vec<(String, f64)> {
        pairs.iter().map(|(id, s)| (id.to_string(), *s)).collect()
    };
    out.push((
        query(
            "q-ord-1",
            ORDINARY_ASKED,
            QueryClass::Ordinary,
            &[("doc-1.1", 3), ("doc-1.2", 1)],
            None,
            &[],
        ),
        named(&[("doc-1.1", 0.90), ("doc-1.2", 0.85), ("inc-c.s3", 0.50)]),
        named(&[("doc-1.2", 6.0), ("doc-1.1", 5.5)]),
        named(&[("doc-1.1", 0.80), ("doc-1.2", 0.75), ("inc-c.s3", 0.20)]),
    ));
    out.push((
        query(
            "q-ord-2",
            ORDINARY_ASKED,
            QueryClass::Ordinary,
            &[("doc-2.1", 3)],
            None,
            &[],
        ),
        named(&[("doc-2.1", 0.70), ("inc-b.s3", 0.69)]),
        named(&[("inc-b.s3", 4.0), ("doc-2.1", 3.0)]),
        named(&[("doc-2.1", 0.50), ("inc-b.s3", 0.55)]),
    ));
    out.push((
        query(
            "q-ord-6",
            ORDINARY_ASKED,
            QueryClass::Ordinary,
            &[("doc-1.2", 3)],
            None,
            &[],
        ),
        named(&[("doc-1.2", 0.90)]),
        Vec::new(),
        named(&[("doc-1.2", 0.60)]),
    ));
    out
}

/// Write the data set, the run's candidate lists (views now and batch_3600) and passage vectors.
fn build(root: &Path) -> (PathBuf, PathBuf) {
    let (data, run) = (root.join("data"), root.join("run"));
    fs::create_dir_all(&data).unwrap();
    fs::create_dir_all(&run).unwrap();
    let records = records();
    let written: BTreeMap<String, i64> = records.iter().map(|r| (r.id.clone(), r.written_at)).collect();
    let asked = questions();
    let mut lines = Vec::new();
    for (question, dense, sparse, rerank) in &asked {
        lines.push(Candidates {
            query: question.id.clone(),
            view: "now".to_string(),
            as_of: question.asked_at,
            dense: scored(dense),
            sparse: scored(sparse),
            rerank: scored(rerank),
            timings_ms: BTreeMap::new(),
        });
        // An index refreshed on the hour holds only what was written by the last full hour.
        let batch = View::Batch(3_600);
        let seen = |list: &[(String, f64)]| -> Vec<(String, f64)> {
            list.iter()
                .filter(|(id, _)| batch.sees(written[id], question.asked_at))
                .cloned()
                .collect()
        };
        lines.push(Candidates {
            query: question.id.clone(),
            view: batch.name(),
            as_of: question.asked_at,
            dense: scored(&seen(dense)),
            sparse: scored(&seen(sparse)),
            rerank: Vec::new(),
            timings_ms: BTreeMap::new(),
        });
    }
    let queries: Vec<Query> = asked.into_iter().map(|(q, ..)| q).collect();
    let write = |path: PathBuf, f: &dyn Fn(&mut Vec<u8>)| {
        let mut bytes = Vec::new();
        f(&mut bytes);
        fs::write(path, bytes).unwrap();
    };
    write(data.join("records.jsonl"), &|b| write_jsonl(b, &records).unwrap());
    write(data.join("queries.jsonl"), &|b| write_jsonl(b, &queries).unwrap());
    write(run.join("candidates.jsonl"), &|b| write_jsonl(b, &lines).unwrap());
    // The three states of an incident say nearly the same thing (cosines above 0.99); the wrong
    // statement and the help sections point elsewhere.
    let mut ids = Vec::new();
    let mut values = Vec::new();
    for record in &records {
        let vector: [f32; 3] = match record.id.rsplit('.').next() {
            Some("s1") => [1.0, 0.0, 0.1],
            Some("s2") => [1.0, 0.05, 0.1],
            Some("s3") => [1.0, 0.1, 0.1],
            Some("d1") => [0.5, 1.0, 0.0],
            _ if record.id == "doc-2.1" => [0.0, 1.0, 0.2],
            _ if record.id == "doc-1.2" => [0.0, 0.1, 1.0],
            _ => [0.0, 0.0, 1.0],
        };
        ids.push(record.id.clone());
        values.extend(vector);
    }
    write_vectors(&run.join("vectors"), &VectorIndex { dim: 3, ids }, &values).unwrap();
    (data, run)
}

fn cb_eval(args: &[&str]) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_cb-eval"))
        .args(args)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "cb-eval {args:?} failed\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    stdout
}

fn text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The value `numbers.tex` defines for a key.
fn number(numbers: &str, key: &str) -> String {
    let start = format!("\\expandafter\\newcommand\\csname cbnum/{key}\\endcsname{{");
    let line = numbers
        .lines()
        .find(|line| line.starts_with(&start))
        .unwrap_or_else(|| panic!("numbers.tex has no key {key}"));
    line[start.len()..line.len() - 1].to_string()
}

#[test]
fn rank_then_tune_then_report_gives_the_numbers_worked_out_by_hand() {
    let root = std::env::temp_dir().join(format!("cb-eval-end-to-end-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (data, run) = build(&root);
    let (data_arg, run_arg) = (data.to_str().unwrap(), run.to_str().unwrap());

    // The splits the comments above rely on.
    let queries: Vec<Query> = questions().into_iter().map(|(q, ..)| q).collect();
    let splits: Vec<(&str, Split)> = queries.iter().map(|q| (q.id.as_str(), split_of(q))).collect();
    assert_eq!(
        splits,
        [
            ("inc-a-q1", Split::Validation),
            ("inc-a-q2", Split::Validation),
            ("inc-b-q1", Split::Test),
            ("inc-b-q2", Split::Test),
            ("inc-c-q1", Split::Test),
            ("inc-c-q2", Split::Test),
            ("q-ord-1", Split::Test),
            ("q-ord-2", Split::Test),
            ("q-ord-6", Split::Validation),
        ]
    );

    let ranked = cb_eval(&[
        "rank",
        "--data",
        data_arg,
        "--run",
        run_arg,
        "dense",
        "batch(period=3600)",
        "rrf(k=60)",
        "rrf_td(k=10|60,h=60|1800)",
        "product",
        "product_decay(h=60|1800)",
        "product_newer_wins(tau=0.9)",
        "keyed_latest",
        "oracle",
    ]);
    // dense, batch, rrf, four rrf_td, product, two product_decay, newer_wins, keyed_latest, oracle.
    assert!(ranked.contains("wrote 13 methods"), "{ranked}");
    let rankings: Vec<Ranking> = cb_types::read_jsonl(
        fs::File::open(run.join("rankings/product.jsonl"))
            .map(std::io::BufReader::new)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(rankings.len(), 9);
    // inc-b-q2 under the product: the merge of D and S holds every listed record; doc-1.1 scores
    // 0.05 < 0.12 and is cut; the rest by relevance score.
    let ids: Vec<&str> = rankings[3].ranked.iter().map(|s| s.record.as_str()).collect();
    assert_eq!(ids, ["inc-b.s2", "inc-b.s1", "inc-b.s3", "inc-b.d1"]);

    // On validation (inc-a): product_decay(h=60) puts the young wrong statement first on q1
    // (0.55 * 2^-5 = 0.0172 against 0.70 * 2^-11.67 = 0.0002) and gets q2 right: 50%.
    // h=1800 gets both: q1 0.70 * 2^(-700/1800) = 0.535 > 0.55 * 2^(-300/1800) = 0.490; q2 s3 first.
    let tuned = cb_eval(&[
        "tune",
        "--run",
        run_arg,
        "--label",
        "product_decay",
        "--measure",
        "current@1",
        "--group",
        "changing",
        "product_decay(h=60|1800)",
    ]);
    assert!(tuned.contains("chose product_decay-h1800"), "{tuned}");

    let reported = cb_eval(&[
        "report",
        "--run",
        run_arg,
        "--name",
        "tiny",
        "--draws",
        "500",
        "--threads",
        "2",
    ]);
    assert!(reported.contains("applied the choices of"), "{reported}");

    let summary: serde_json::Value = serde_json::from_str(&text(&run.join("summary.json"))).unwrap();
    assert_eq!(summary["split"], "test");
    assert_eq!(summary["seed"], 20_261_119);
    assert_eq!(summary["draws"], 500);
    let groups: Vec<&str> = summary["groups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| g["group"].as_str().unwrap())
        .collect();
    assert_eq!(groups, ["changing", "ordinary", "ordinary_all"]);
    assert_eq!(summary["groups"][0]["questions"], serde_json::json!([4]));
    assert_eq!(summary["groups"][0]["clusters"], serde_json::json!([2]));

    let numbers = text(&run.join("numbers.tex"));
    let current = |method: &str| number(&numbers, &format!("tiny/{method}/changing/current@1"));
    // Test questions: inc-b and inc-c, q1 and q2 each.
    // dense: q1 has d1 first, q2 has s1 first: 0 of 4.
    assert_eq!(current("dense"), "0.0");
    // batch_3600: on q1 the index refreshed at the start of the incident and holds only doc-1.1;
    // on q2 it holds everything: s1 first. 0 of 4.
    assert_eq!(current("batch-period3600"), "0.0");
    // product: q1 s1 (0.70) before d1 (0.55); q2 s2 (0.64) first. 2 of 4.
    assert_eq!(current("product"), "50.0");
    // product_decay(h=1800): right on all four (see the validation comment above).
    assert_eq!(current("product_decay-h1800"), "100.0");
    assert_eq!(
        number(
            &numbers,
            "tiny/product_decay-h1800/changing/current@1/minus/product"
        ),
        "+50.0"
    );
    // keyed_latest: on q1 the newest record of the thread is the wrong statement d1, which is all
    // that is left of the incident; on q2 it is s3. 2 of 4.
    assert_eq!(current("keyed_latest"), "50.0");
    // product_newer_wins: on q2 s3 says the same as s2 and s1 and is newer, so it takes first
    // place; d1 says something else and stays. 4 of 4.
    assert_eq!(current("product_newer_wins-tau0.9"), "100.0");
    // The paper's formula: on q1 d1 leads with (1/3)(1/61 + 1/62 + 1/61) * 2^(-300/1800) = 0.01453
    // against s1's (1/3)(1/62 + 1/61 + 1/62) * 2^(-700/1800) = 0.01238; on q2 s3 leads. 2 of 4.
    assert_eq!(current("rrf_td-k60-h1800"), "50.0");
    assert_eq!(current("oracle"), "100.0");
    // The two help questions: the product has doc-1.1 first on q-ord-1 and inc-b.s3 (0.55) before
    // doc-2.1 (0.50) on q-ord-2, so hit@4 is 100% and current@1 50%.
    assert_eq!(number(&numbers, "tiny/product/ordinary_all/hit@4"), "100.0");
    assert_eq!(number(&numbers, "tiny/product/ordinary_all/current@1"), "50.0");
    assert_eq!(
        number(&numbers, "tiny/tuned/product_decay"),
        "\\texttt{product\\_decay(h=1800)}"
    );
    assert_eq!(number(&numbers, "tiny/tuned/product_decay/h"), "1800");
    assert_eq!(number(&numbers, "tiny/changing/questions"), "4");

    // The main table shows the chosen half-life with its mark and leaves the other out.
    let main = text(&run.join("tables/changing.tex"));
    assert!(
        main.contains("\\texttt{product\\_decay(h=1800)}$^{\\dagger}$ & "),
        "{main}"
    );
    assert!(!main.contains("product\\_decay(h=60)"), "{main}");
    assert!(run.join("tables/changing-minus-product.tex").exists());
    assert!(run.join("tables/rrf_td-changing-current@1.tex").exists());
    assert!(run.join("figures/results.csv").exists());
    assert!(run.join("figures/rrf_td.csv").exists());

    // The same files again, byte for byte, with another number of threads.
    let first = fs::read(run.join("summary.json")).unwrap();
    let first_numbers = fs::read(run.join("numbers.tex")).unwrap();
    cb_eval(&[
        "report",
        "--run",
        run_arg,
        "--name",
        "tiny",
        "--draws",
        "500",
        "--threads",
        "1",
    ]);
    assert_eq!(fs::read(run.join("summary.json")).unwrap(), first);
    assert_eq!(fs::read(run.join("numbers.tex")).unwrap(), first_numbers);

    if std::env::var_os("CB_EVAL_E2E_KEEP").is_some() {
        println!("kept {}", root.display());
    } else {
        fs::remove_dir_all(&root).unwrap();
    }
}

/// Write JSON Lines.
fn write_lines<T: serde::Serialize>(path: &Path, items: &[T]) {
    let mut bytes = Vec::new();
    write_jsonl(&mut bytes, items).unwrap();
    fs::write(path, bytes).unwrap();
}

/// Two incidents, inc-0006 (validation) and inc-0001 (test), each with a true state s1
/// (key term "fibre cut") and a planted wrong statement d1 ("your own cable"), and four calls
/// ten minutes apart by agents a, b, a and c. Calls 0 to 2 state s1; call 3 states d1; call 2
/// is extracted with confidence 0.5. Then 25 trips.
#[test]
fn gate_and_trips_run_from_the_command_line() {
    let root = std::env::temp_dir().join(format!("cb-eval-end-to-end-gate-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    let (data, run) = (root.join("data"), root.join("run"));
    fs::create_dir_all(&data).unwrap();
    fs::create_dir_all(&run).unwrap();
    let mut records = Vec::new();
    let mut transcripts = Vec::new();
    let mut extractions = Vec::new();
    let mut ids = Vec::new();
    let mut values: Vec<f32> = Vec::new();
    for (thread, start) in [("inc-0006", T0), ("inc-0001", T0 + 50_000)] {
        let planted = |correct: bool, term: &str| {
            Some(RecordTruth {
                correct,
                phase: Phase::Investigating,
                key_terms: vec![term.to_string()],
                superseded_at: None,
            })
        };
        records.push(record(
            &format!("{thread}.s1"),
            start,
            Some(thread),
            planted(true, "fibre cut"),
        ));
        records.push(record(
            &format!("{thread}.d1"),
            start + 600,
            Some(thread),
            planted(false, "your own cable"),
        ));
        for (i, agent) in ["a", "b", "a", "c"].into_iter().enumerate() {
            let id = format!("{thread}-call{i}");
            let wrong = i == 3;
            let said = if wrong {
                "It is your own cable."
            } else {
                "There is a fibre cut."
            };
            transcripts.push(Transcript {
                id: id.clone(),
                thread: thread.to_string(),
                started_at: start + 600 * i as i64,
                agent: agent.to_string(),
                persona: String::new(),
                turns: vec![
                    Turn {
                        who: Speaker::Customer,
                        text: "Is it down?".to_string(),
                        ends_at_s: 5.0,
                    },
                    Turn {
                        who: Speaker::Agent,
                        text: said.to_string(),
                        ends_at_s: 30.0,
                    },
                    Turn {
                        who: Speaker::Customer,
                        text: "Thanks.".to_string(),
                        ends_at_s: 60.0,
                    },
                ],
                states: vec![format!("{thread}.{}", if wrong { "d1" } else { "s1" })],
                asr_error_rate: 0.0,
            });
            extractions.push(Extraction {
                transcript: id.clone(),
                problem: "Internet down".to_string(),
                solution: if wrong {
                    "check your own cable"
                } else {
                    "a fibre cut in the area"
                }
                .to_string(),
                t_start: None,
                t_eta: None,
                times_unread: vec![],
                confidence: Confidence {
                    problem: if i == 2 { 0.5 } else { 0.95 },
                    solution: 1.0,
                    time: 1.0,
                },
                model: "test".to_string(),
                prompt_version: "1".to_string(),
                latency_ms: 2_000.0,
                tokens_in: 0,
                tokens_out: 0,
            });
            ids.push(id);
            values.extend(if wrong { [0.0, 1.0] } else { [1.0, 0.02 * i as f32] });
        }
    }
    write_lines(&data.join("records.jsonl"), &records);
    write_lines(&data.join("transcripts.jsonl"), &transcripts);
    write_lines(&run.join("extractions.jsonl"), &extractions);
    write_vectors(
        &run.join("extraction_vectors"),
        &VectorIndex { dim: 2, ids },
        &values,
    )
    .unwrap();
    // Hand-off to first hit of 2, 4, …, 50 ms.
    let trips: Vec<PipelineTrip> = (1..=25)
        .map(|i| PipelineTrip {
            record: format!("r{i}"),
            transcript_ms: Some(100.0),
            masked_ms: Some(110.0),
            extracted_ms: Some(600.0),
            written_ms: Some(600.0 + f64::from(i)),
            first_hit_ms: Some(600.0 + 2.0 * f64::from(i)),
        })
        .collect();
    write_lines(&run.join("trips.jsonl"), &trips);
    let run_arg = run.to_str().unwrap();

    let gated = cb_eval(&[
        "gate",
        "--data",
        data.to_str().unwrap(),
        "--run",
        run_arg,
        "--name",
        "tiny",
        "--judge-sample",
        "3",
    ]);
    assert!(gated.contains("n2-pairs-calls-cut0.85-bar0.92"), "{gated}");
    let numbers = text(&run.join("gate/numbers.tex"));
    // Test split (inc-0001): on its own every extraction is a record and 3 of 4 are correct.
    assert_eq!(number(&numbers, "tiny/gate/n1-nocut/precision"), "75.0");
    // The paper's setting: call 2 is below the cut; call 1 agrees with call 0 (cosine 0.9998) and
    // both are promoted when call 1's extraction arrives, 600 + 60 + 2 s after call 0 began; s1
    // was first stated 30 s into call 0: a wait of 632 s, 10.5 minutes. Call 3 waits alone.
    let paper = "tiny/gate/n2-pairs-calls-cut0.85-bar0.92";
    assert_eq!(number(&numbers, &format!("{paper}/precision")), "100.0");
    assert_eq!(number(&numbers, &format!("{paper}/validated")), "100.0");
    assert_eq!(number(&numbers, &format!("{paper}/wait-median-min")), "10.5");
    assert_eq!(number(&numbers, "tiny/gate/judge/sample"), "3");
    assert_eq!(
        fs::read_to_string(run.join("gate/judge_sample.jsonl"))
            .unwrap()
            .lines()
            .count(),
        3
    );

    let tripped = cb_eval(&["trips", "--run", run_arg, "--name", "tiny"]);
    assert!(tripped.contains("not met"), "{tripped}");
    let numbers = text(&run.join("trips/numbers.tex"));
    // 25 trips: P95 is the 24th smallest (ceil(0.95 * 25) = 24): 48 ms. P99 needs 100 trips.
    assert_eq!(number(&numbers, "tiny/trips/handoff_to_first_hit/p95-ms"), "48");
    assert!(!numbers.contains("handoff_to_first_hit/p99-ms"));
    // Masking takes 10 ms on every trip.
    assert_eq!(number(&numbers, "tiny/trips/stage/masking/median-ms"), "10.0");
    fs::remove_dir_all(&root).unwrap();
}
