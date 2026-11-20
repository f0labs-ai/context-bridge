//! `cb-data check <dir>`: checks that a data set directory keeps the contract of
//! DESIGN.md section 2 and, for a generated corpus, the rules of
//! docs/SYNTH_SPEC.md section 6. It prints one line per rule and fails when any
//! rule fails.

use crate::files::{self, DataSet};
use crate::synth::spec;
use crate::synth::text::{collapse_whitespace, contains_folded, fold, tokens_of_words, word_count};
use cb_types::{Manifest, QueryClass, Record, Tier, Transcript, ids, read_jsonl};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

/// `cb-data check` arguments.
#[derive(clap::Args, Debug)]
pub struct Args {
    /// The data set directory to check.
    pub dir: PathBuf,
}

/// The names of the rules, as the report prints them.
pub mod rule {
    pub const IDS: &str = "ids are valid and unique";
    pub const TEXTS: &str = "no two records share a text";
    pub const REFERENCES: &str = "every gold, harmful and stated id exists";
    pub const GRADE_3_KNOWN: &str = "a grade-3 record was written at or before its question";
    pub const SUPERSEDED: &str = "superseded_at equals the next state's written_at";
    pub const KEY_TERMS: &str = "every stated record's key terms appear in its transcript";
    pub const COUNTS: &str = "the counts match the specification";
    pub const LENGTH: &str = "the mean transcript length is within range";
    pub const HASHES: &str = "the manifest's hashes match the files";
    pub const SPEECH_COPIES: &str = "speech-error copies line up with the transcripts";
}

/// The outcome of one rule: what was looked at and what was wrong.
#[derive(Debug)]
pub struct RuleResult {
    pub rule: &'static str,
    pub checked: String,
    pub problems: Vec<String>,
}

/// The outcome of every rule that applied.
#[derive(Debug, Default)]
pub struct Report {
    pub results: Vec<RuleResult>,
}

/// Problems shown per failed rule; the rest are counted.
const SHOWN_PROBLEMS: usize = 5;

impl Report {
    fn add(&mut self, rule: &'static str, checked: impl Into<String>, problems: Vec<String>) {
        self.results.push(RuleResult {
            rule,
            checked: checked.into(),
            problems,
        });
    }

    pub fn passed(&self) -> bool {
        self.results.iter().all(|r| r.problems.is_empty())
    }

    /// The rules that failed.
    pub fn failed(&self) -> Vec<&'static str> {
        self.results
            .iter()
            .filter(|r| !r.problems.is_empty())
            .map(|r| r.rule)
            .collect()
    }

    /// The report as printed: one line per rule, a few problems under each failure.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for r in &self.results {
            let mark = if r.problems.is_empty() { "ok  " } else { "FAIL" };
            out.push_str(&format!("{mark}  {} ({})\n", r.rule, r.checked));
            for p in r.problems.iter().take(SHOWN_PROBLEMS) {
                out.push_str(&format!("        {p}\n"));
            }
            if r.problems.len() > SHOWN_PROBLEMS {
                out.push_str(&format!(
                    "        and {} more\n",
                    r.problems.len() - SHOWN_PROBLEMS
                ));
            }
        }
        let passed = self.results.iter().filter(|r| r.problems.is_empty()).count();
        out.push_str(&format!("{passed} of {} rules passed\n", self.results.len()));
        out
    }
}

pub fn run(args: Args) -> anyhow::Result<()> {
    let report = check_dir(&args.dir)?;
    println!("cb-data check {}", args.dir.display());
    print!("{}", report.render());
    if report.passed() {
        Ok(())
    } else {
        anyhow::bail!(
            "{} rule(s) failed: {}",
            report.failed().len(),
            report.failed().join("; ")
        )
    }
}

/// Check a data set directory: every rule on its contents, the manifest's
/// hashes and any speech-error copies of the transcripts.
pub fn check_dir(dir: &Path) -> anyhow::Result<Report> {
    let data = files::read_data_set(dir)?;
    let manifest = files::read_manifest(dir).ok();
    let mut report = check_data_set(&data, manifest.as_ref());
    match &manifest {
        None => report.add(
            rule::HASHES,
            "manifest.json",
            vec!["manifest.json is missing or unreadable".into()],
        ),
        Some(m) => {
            let mut problems = Vec::new();
            let mut expected = vec![files::RECORDS, files::QUERIES];
            if !data.transcripts.is_empty() {
                expected.push(files::TRANSCRIPTS);
            }
            for name in expected {
                if !m.sha256.contains_key(name) {
                    problems.push(format!("the manifest does not hash {name}"));
                }
            }
            let mut present = true;
            for name in m.sha256.keys() {
                if !dir.join(name).exists() {
                    problems.push(format!("{name} is in the manifest but not in the directory"));
                    present = false;
                }
            }
            if present && !files::hashes_match(dir, m)? {
                problems.push("a file's sha256 differs from the manifest".into());
            }
            report.add(rule::HASHES, format!("{} files", m.sha256.len()), problems);
            check_speech_copies(dir, m, &data.transcripts, &mut report)?;
        }
    }
    Ok(report)
}

/// The copies with speech-recognition errors keep every id, thread and state.
fn check_speech_copies(
    dir: &Path,
    manifest: &Manifest,
    transcripts: &[Transcript],
    report: &mut Report,
) -> anyhow::Result<()> {
    let copies: Vec<&String> = manifest
        .sha256
        .keys()
        .filter(|n| n.starts_with("transcripts.asr") && n.ends_with(".jsonl"))
        .collect();
    if copies.is_empty() {
        return Ok(());
    }
    let mut problems = Vec::new();
    for name in &copies {
        let path = dir.join(name);
        if !path.exists() {
            continue;
        }
        let copy: Vec<Transcript> = read_jsonl(BufReader::new(File::open(path)?))?;
        if copy.len() != transcripts.len() {
            problems.push(format!(
                "{name}: {} transcripts, expected {}",
                copy.len(),
                transcripts.len()
            ));
            continue;
        }
        for (a, b) in transcripts.iter().zip(&copy) {
            if a.id != b.id || a.thread != b.thread || a.states != b.states || a.turns.len() != b.turns.len()
            {
                problems.push(format!("{name}: {} does not line up with {}", b.id, a.id));
            }
        }
    }
    report.add(rule::SPEECH_COPIES, format!("{} copies", copies.len()), problems);
    Ok(())
}

/// Whether the data set is a generated corpus (docs/SYNTH_SPEC.md).
fn is_generated(data: &DataSet, manifest: Option<&Manifest>) -> bool {
    manifest.is_some_and(|m| m.made_by.starts_with(spec::MADE_BY))
        || data.records.iter().any(|r| r.source == spec::SOURCE)
}

/// Check every rule that needs only the files' contents.
pub fn check_data_set(data: &DataSet, manifest: Option<&Manifest>) -> Report {
    let mut report = Report::default();
    let records: BTreeMap<&str, &Record> = data.records.iter().map(|r| (r.id.as_str(), r)).collect();
    check_ids(data, &mut report);
    check_texts(data, &mut report);
    check_references(data, &records, &mut report);
    check_grade_3_known(data, &records, &mut report);
    check_superseded(data, &mut report);
    if !data.transcripts.is_empty() {
        check_key_terms(data, &records, &mut report);
    }
    if is_generated(data, manifest) {
        check_counts(data, manifest, &mut report);
        check_length(data, &mut report);
    }
    report
}

fn check_ids_of<'a>(kind: &str, all: impl Iterator<Item = &'a str>, problems: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    for id in all {
        if ids::check(id).is_err() {
            problems.push(format!("{kind} id {id:?} is not a valid id"));
        }
        if !seen.insert(id) {
            problems.push(format!("{kind} id {id} repeats"));
        }
    }
}

fn check_ids(data: &DataSet, report: &mut Report) {
    let mut problems = Vec::new();
    check_ids_of(
        "record",
        data.records.iter().map(|r| r.id.as_str()),
        &mut problems,
    );
    check_ids_of(
        "question",
        data.queries.iter().map(|q| q.id.as_str()),
        &mut problems,
    );
    check_ids_of(
        "transcript",
        data.transcripts.iter().map(|t| t.id.as_str()),
        &mut problems,
    );
    let checked = format!(
        "{} records, {} questions, {} transcripts",
        data.records.len(),
        data.queries.len(),
        data.transcripts.len()
    );
    report.add(rule::IDS, checked, problems);
}

/// The search index files a passage under a hash of its text, so two records
/// with the same text (whitespace collapsed) would become one entry.
fn check_texts(data: &DataSet, report: &mut Report) {
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    let mut problems = Vec::new();
    for r in &data.records {
        if let Some(first) = seen.insert(collapse_whitespace(&r.text), &r.id) {
            problems.push(format!("{first} and {} have the same text", r.id));
        }
    }
    report.add(rule::TEXTS, format!("{} texts", data.records.len()), problems);
}

fn check_references(data: &DataSet, records: &BTreeMap<&str, &Record>, report: &mut Report) {
    let mut problems = Vec::new();
    let threads: BTreeSet<&str> = data.records.iter().filter_map(|r| r.thread.as_deref()).collect();
    for q in &data.queries {
        for g in &q.gold {
            if !records.contains_key(g.record.as_str()) {
                problems.push(format!("{}: gold record {} does not exist", q.id, g.record));
            }
        }
        for h in &q.harmful {
            if !records.contains_key(h.as_str()) {
                problems.push(format!("{}: harmful record {h} does not exist", q.id));
            }
        }
        if q.gold.iter().all(|g| g.grade != 3) {
            problems.push(format!("{}: no grade-3 record", q.id));
        }
    }
    for t in &data.transcripts {
        for s in &t.states {
            if !records.contains_key(s.as_str()) {
                problems.push(format!("{}: stated record {s} does not exist", t.id));
            }
        }
        if !threads.contains(t.thread.as_str()) {
            problems.push(format!("{}: thread {} has no records", t.id, t.thread));
        }
    }
    let checked = format!(
        "{} questions, {} transcripts",
        data.queries.len(),
        data.transcripts.len()
    );
    report.add(rule::REFERENCES, checked, problems);
}

fn check_grade_3_known(data: &DataSet, records: &BTreeMap<&str, &Record>, report: &mut Report) {
    let mut problems = Vec::new();
    for q in &data.queries {
        for g in q.gold.iter().filter(|g| g.grade == 3) {
            if let Some(r) = records.get(g.record.as_str())
                && r.written_at > q.asked_at
            {
                problems.push(format!(
                    "{}: {} is written at {}, after the question at {}",
                    q.id, r.id, r.written_at, q.asked_at
                ));
            }
        }
    }
    report.add(
        rule::GRADE_3_KNOWN,
        format!("{} questions", data.queries.len()),
        problems,
    );
}

/// An id's place among its thread's ids: the part after the last dot as a
/// number when it is one (".9" before ".10"), else as written (".s2").
fn id_order(id: &str) -> (&str, Option<u64>, &str) {
    let (head, tail) = id.rsplit_once('.').unwrap_or((id, ""));
    (head, tail.parse().ok(), tail)
}

/// Within a thread, each true record is replaced exactly when the next one is
/// written, and the last is never replaced. A wrong statement, when it says it
/// was replaced, says so no earlier than it was written. A stated end comes
/// after the record is written.
fn check_superseded(data: &DataSet, report: &mut Report) {
    let mut problems = Vec::new();
    let mut by_thread: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in &data.records {
        if let Some(until) = r.valid_until
            && until <= r.written_at
        {
            problems.push(format!(
                "{}: valid_until {until} is not after written_at {}",
                r.id, r.written_at
            ));
        }
        let Some(truth) = &r.truth else {
            continue;
        };
        if truth.correct {
            if let Some(thread) = r.thread.as_deref() {
                by_thread.entry(thread).or_default().push(r);
            }
        } else if let Some(at) = truth.superseded_at
            && at < r.written_at
        {
            problems.push(format!(
                "{}: replaced at {at}, before it was written at {}",
                r.id, r.written_at
            ));
        }
    }
    let threads = by_thread.len();
    for (thread, mut states) in by_thread {
        // Several updates can share one second (a status page posting four at
        // once): those keep the order of their ids, read as numbers ".9" < ".10".
        states.sort_by(|a, b| (a.written_at, id_order(&a.id)).cmp(&(b.written_at, id_order(&b.id))));
        for pair in states.windows(2) {
            let (now, next) = (pair[0], pair[1]);
            let replaced = now.truth.as_ref().and_then(|t| t.superseded_at);
            if replaced != Some(next.written_at) {
                problems.push(format!(
                    "{thread}: {} is replaced at {replaced:?}, but {} is written at {}",
                    now.id, next.id, next.written_at
                ));
            }
        }
        if let Some(last) = states.last()
            && let Some(at) = last.truth.as_ref().and_then(|t| t.superseded_at)
        {
            problems.push(format!(
                "{thread}: the latest record {} says it was replaced at {at}",
                last.id
            ));
        }
    }
    report.add(rule::SUPERSEDED, format!("{threads} threads"), problems);
}

fn check_key_terms(data: &DataSet, records: &BTreeMap<&str, &Record>, report: &mut Report) {
    let mut problems = Vec::new();
    let mut stated = 0usize;
    for t in &data.transcripts {
        let text = fold(
            &t.turns
                .iter()
                .map(|turn| turn.text.as_str())
                .collect::<Vec<_>>()
                .join(" "),
        );
        for id in &t.states {
            stated += 1;
            let Some(truth) = records.get(id.as_str()).and_then(|r| r.truth.as_ref()) else {
                continue;
            };
            for term in &truth.key_terms {
                if !contains_folded(&text, term) {
                    problems.push(format!("{}: key term {term:?} of {id} is missing", t.id));
                }
            }
        }
    }
    let checked = format!("{stated} statements in {} transcripts", data.transcripts.len());
    report.add(rule::KEY_TERMS, checked, problems);
}

fn tally<'a>(counts: &mut BTreeMap<&'a str, usize>, ids: &[&'a str]) {
    for id in ids {
        *counts.entry(*id).or_default() += 1;
    }
}

fn expect_count(problems: &mut Vec<String>, what: &str, got: usize, want: usize) {
    if got != want {
        problems.push(format!("{what}: {got}, expected {want}"));
    }
}

/// The counts of docs/SYNTH_SPEC.md sections 2 to 5.
fn check_counts(data: &DataSet, manifest: Option<&Manifest>, report: &mut Report) {
    let mut p = Vec::new();
    let changing: Vec<&Record> = data.records.iter().filter(|r| r.tier == Tier::Changing).collect();
    let threads: BTreeSet<&str> = changing.iter().filter_map(|r| r.thread.as_deref()).collect();
    expect_count(&mut p, "incidents", threads.len(), spec::INCIDENTS);

    let mut true_per_thread: BTreeMap<&str, usize> = BTreeMap::new();
    let mut wrong_per_thread: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &changing {
        let (Some(truth), Some(thread)) = (&r.truth, r.thread.as_deref()) else {
            p.push(format!("{}: a changing record without truth or thread", r.id));
            continue;
        };
        let counts = if truth.correct {
            &mut true_per_thread
        } else {
            &mut wrong_per_thread
        };
        *counts.entry(thread).or_default() += 1;
        if truth.correct && !(1..=spec::STATES).any(|k| r.id == format!("{thread}.s{k}")) {
            p.push(format!("{}: not a state id of {thread}", r.id));
        }
    }
    expect_count(
        &mut p,
        "true records",
        true_per_thread.values().sum(),
        spec::TRUE_RECORDS,
    );
    if let Some((thread, n)) = true_per_thread.iter().find(|(_, n)| **n != spec::STATES) {
        p.push(format!("{thread} has {n} states, expected {}", spec::STATES));
    }
    expect_count(
        &mut p,
        "wrong statements",
        wrong_per_thread.values().sum(),
        spec::WRONG_RECORDS,
    );
    let with_three = wrong_per_thread.values().filter(|n| **n == 3).count();
    let with_two = wrong_per_thread.values().filter(|n| **n == 2).count();
    let want_two = spec::INCIDENTS - spec::INCIDENTS_WITH_THREE_WRONG;
    expect_count(
        &mut p,
        "incidents with three wrong statements",
        with_three,
        spec::INCIDENTS_WITH_THREE_WRONG,
    );
    expect_count(&mut p, "incidents with two wrong statements", with_two, want_two);
    let standing = data
        .records
        .iter()
        .filter(|r| r.tier == Tier::Standing && r.thread.is_none())
        .count();
    expect_count(&mut p, "standing records", standing, spec::STANDING_RECORDS);

    expect_count(&mut p, "transcripts", data.transcripts.len(), spec::TRANSCRIPTS);
    let mut calls: BTreeMap<&str, usize> = BTreeMap::new();
    for t in &data.transcripts {
        *calls.entry(t.thread.as_str()).or_default() += 1;
    }
    let few = threads
        .iter()
        .filter(|t| calls.get(*t).copied().unwrap_or(0) < spec::MIN_CALLS_PER_INCIDENT)
        .count();
    if few > 0 {
        p.push(format!(
            "{few} incidents have fewer than {} calls",
            spec::MIN_CALLS_PER_INCIDENT
        ));
    }

    let class_count = |class: QueryClass| data.queries.iter().filter(|q| q.class == class).count();
    expect_count(
        &mut p,
        "changing questions",
        class_count(QueryClass::Changing),
        spec::CHANGING_QUESTIONS,
    );
    let post_count = class_count(QueryClass::PostIncident);
    expect_count(
        &mut p,
        "post-incident questions",
        post_count,
        spec::POST_INCIDENT_QUESTIONS,
    );
    expect_count(
        &mut p,
        "ambiguous questions",
        class_count(QueryClass::Ambiguous),
        spec::AMBIGUOUS_QUESTIONS,
    );
    expect_count(
        &mut p,
        "ordinary questions",
        class_count(QueryClass::Ordinary),
        spec::ORDINARY_QUESTIONS,
    );
    let near_count = class_count(QueryClass::OrdinaryNear);
    expect_count(
        &mut p,
        "near-ordinary questions",
        near_count,
        spec::ORDINARY_NEAR_QUESTIONS,
    );

    // Two changing questions per state 1–4 of each incident, one post-incident
    // question per incident, one ordinary question per standing record, and two
    // incidents behind every ambiguous question.
    let mut per_state: BTreeMap<&str, usize> = BTreeMap::new();
    let mut post: BTreeMap<&str, usize> = BTreeMap::new();
    let mut ordinary: BTreeMap<&str, usize> = BTreeMap::new();
    for q in &data.queries {
        let top: Vec<&str> = q
            .gold
            .iter()
            .filter(|g| g.grade == 3)
            .map(|g| g.record.as_str())
            .collect();
        match q.class {
            QueryClass::Changing => tally(&mut per_state, &top),
            QueryClass::PostIncident => tally(&mut post, &top),
            QueryClass::Ordinary => tally(&mut ordinary, &top),
            QueryClass::Ambiguous => {
                let incidents: BTreeSet<&str> = top.iter().filter_map(|r| r.split('.').next()).collect();
                if incidents.len() < 2 {
                    p.push(format!(
                        "{}: an ambiguous question about fewer than two incidents",
                        q.id
                    ));
                }
            }
            QueryClass::OrdinaryNear => {}
        }
    }
    for thread in &threads {
        for k in 1..spec::STATES {
            let n = per_state
                .get(format!("{thread}.s{k}").as_str())
                .copied()
                .unwrap_or(0);
            if n != 2 {
                p.push(format!("{thread}.s{k}: {n} changing questions, expected 2"));
            }
        }
        if post.get(format!("{thread}.s{}", spec::STATES).as_str()).copied() != Some(1) {
            p.push(format!("{thread}: not exactly one post-incident question"));
        }
    }
    for r in data.records.iter().filter(|r| r.tier == Tier::Standing) {
        if ordinary.get(r.id.as_str()).copied() != Some(1) {
            p.push(format!("{}: not exactly one ordinary question", r.id));
        }
    }

    // What only the generator knows, as its manifest records it.
    if let Some(m) = manifest {
        for (name, want) in [
            ("incidents_short", spec::SHORT),
            ("incidents_medium", spec::MEDIUM),
            ("incidents_long", spec::LONG),
            ("planted_pairs", spec::PLANTED_PAIRS),
        ] {
            match m.counts.get(name) {
                Some(got) => expect_count(&mut p, &format!("manifest {name}"), *got as usize, want),
                None => p.push(format!("the manifest has no count {name}")),
            }
        }
    }
    report.add(rule::COUNTS, "docs/SYNTH_SPEC.md sections 2 to 5", p);
}

/// Mean transcript length, in tokens counted as words × 1.33.
pub fn mean_tokens(transcripts: &[Transcript]) -> f64 {
    if transcripts.is_empty() {
        return 0.0;
    }
    let words: usize = transcripts
        .iter()
        .map(|t| t.turns.iter().map(|turn| word_count(&turn.text)).sum::<usize>())
        .sum();
    tokens_of_words(words) / transcripts.len() as f64
}

fn check_length(data: &DataSet, report: &mut Report) {
    let mean = mean_tokens(&data.transcripts);
    let mut problems = Vec::new();
    if (mean - spec::MEAN_TOKENS).abs() > spec::MEAN_TOKENS_TOLERANCE {
        problems.push(format!(
            "mean {mean:.1} tokens, expected {} ± {}",
            spec::MEAN_TOKENS,
            spec::MEAN_TOKENS_TOLERANCE
        ));
    }
    report.add(rule::LENGTH, format!("mean {mean:.1} tokens"), problems);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::tests_support::shared_corpus;

    fn copy_of(data: &DataSet) -> DataSet {
        DataSet {
            records: data.records.clone(),
            queries: data.queries.clone(),
            transcripts: data.transcripts.clone(),
        }
    }

    fn temp_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("cb-data-check-{name}-{}", std::process::id()))
    }

    #[test]
    fn a_generated_data_set_passes_every_rule() {
        let corpus = shared_corpus();
        let dir = temp_dir("pass");
        crate::synth::write_generated(corpus, &dir, &[0.05]).unwrap();
        let report = check_dir(&dir).unwrap();
        assert!(report.passed(), "{}", report.render());
        assert!(report.results.iter().any(|r| r.rule == rule::SPEECH_COPIES));
        assert_eq!(report.results.len(), 10);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn each_kind_of_damage_fails_its_rule() {
        let good = &shared_corpus().data;
        type Damage = fn(&mut DataSet);
        let cases: [(&str, Damage); 8] = [
            (rule::TEXTS, |d| {
                // Same words as another record, laid out with other whitespace.
                let text = format!("{}  ", d.records[0].text.replace(' ', "\n"));
                d.records[7].text = text;
            }),
            (rule::IDS, |d| d.records[3].id = "Not An Id".into()),
            (rule::REFERENCES, |d| {
                d.queries[0].gold[0].record = "inc-9999.s1".into()
            }),
            (rule::GRADE_3_KNOWN, |d| d.queries[2].asked_at = 0),
            (rule::SUPERSEDED, |d| {
                if let Some(t) = d.records[1].truth.as_mut() {
                    t.superseded_at = t.superseded_at.map(|s| s + 1);
                }
            }),
            (rule::KEY_TERMS, |d| {
                let stating = d
                    .transcripts
                    .iter()
                    .position(|t| !t.states.is_empty())
                    .unwrap_or(0);
                for turn in &mut d.transcripts[stating].turns {
                    turn.text = "We could not say anything useful.".into();
                }
            }),
            (rule::COUNTS, |d| {
                d.transcripts.pop();
            }),
            (rule::LENGTH, |d| {
                for t in &mut d.transcripts {
                    t.turns.truncate(2);
                }
            }),
        ];
        assert!(check_data_set(good, None).passed());
        for (expected, damage) in cases {
            let mut broken = copy_of(good);
            damage(&mut broken);
            let report = check_data_set(&broken, None);
            assert!(
                report.failed().contains(&expected),
                "{expected} did not fail:\n{}",
                report.render()
            );
        }
    }

    #[test]
    fn a_changed_file_fails_the_hash_rule_and_the_command() {
        let corpus = shared_corpus();
        let dir = temp_dir("hash");
        crate::synth::write_generated(corpus, &dir, &[]).unwrap();
        let path = dir.join(files::QUERIES);
        let text = std::fs::read_to_string(&path)
            .unwrap()
            .replacen("Is ", "Is   ", 1);
        std::fs::write(&path, text).unwrap();
        let report = check_dir(&dir).unwrap();
        assert_eq!(report.failed(), vec![rule::HASHES]);
        assert!(run(Args { dir: dir.clone() }).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
