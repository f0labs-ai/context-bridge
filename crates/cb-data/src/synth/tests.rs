//! Corpus-level tests of the generator, on one shared corpus (seed 42).

use super::incidents::SAME_PLACE_GAP;
use super::records::first_shared_text;
use super::tests_support::shared_corpus;
use super::text::{contains_folded, fold};
use super::*;
use cb_types::{Query, Record, Speaker, Tier};
use std::collections::{BTreeMap, BTreeSet, HashSet};

fn records_by_id(corpus: &Corpus) -> BTreeMap<&str, &Record> {
    corpus.data.records.iter().map(|r| (r.id.as_str(), r)).collect()
}

/// The state number of a state id (`inc-0001.s3` → 3).
fn state_number(id: &str) -> Option<usize> {
    id.rsplit_once(".s").and_then(|(_, k)| k.parse().ok())
}

fn incident_of<'a>(corpus: &'a Corpus, record_id: &str) -> &'a Incident {
    let thread = record_id.split('.').next().unwrap();
    corpus.incidents.iter().find(|i| i.id == thread).unwrap()
}

fn queries_of(corpus: &Corpus, class: QueryClass) -> Vec<&Query> {
    corpus.data.queries.iter().filter(|q| q.class == class).collect()
}

#[test]
fn the_same_seed_reproduces_every_file_byte_for_byte() {
    let first = shared_corpus();
    let second = generate(42).unwrap();
    assert_eq!(first.data.records, second.data.records);
    assert_eq!(first.data.queries, second.data.queries);
    assert_eq!(first.data.transcripts, second.data.transcripts);

    let base = std::env::temp_dir().join(format!("cb-data-seed-{}", std::process::id()));
    let (a, b) = (base.join("a").join("synth-42"), base.join("b").join("synth-42"));
    write_generated(first, &a, &[0.05]).unwrap();
    write_generated(&second, &b, &[0.05]).unwrap();
    let names = [
        "records.jsonl",
        "queries.jsonl",
        "transcripts.jsonl",
        "transcripts.asr05.jsonl",
        "manifest.json",
    ];
    for name in names {
        let (x, y) = (
            std::fs::read(a.join(name)).unwrap(),
            std::fs::read(b.join(name)).unwrap(),
        );
        assert!(
            !x.is_empty() && x == y,
            "{name} differs between two runs of one seed"
        );
    }
    std::fs::remove_dir_all(&base).unwrap();

    let other = generate(43).unwrap();
    assert_ne!(other.data.records, first.data.records);
}

#[test]
fn the_counts_are_exact() {
    let c = shared_corpus();
    assert_eq!(c.incidents.len(), 500);
    let class = |k: incidents::LengthClass| c.incidents.iter().filter(|i| i.class == k).count();
    assert_eq!(class(incidents::LengthClass::Short), 200);
    assert_eq!(class(incidents::LengthClass::Medium), 200);
    assert_eq!(class(incidents::LengthClass::Long), 100);
    for i in &c.incidents {
        let (lo, hi) = i.class.bounds();
        assert!((lo..=hi).contains(&i.length), "{} has length {}", i.id, i.length);
    }
    let truth = |correct: bool| {
        c.data
            .records
            .iter()
            .filter(|r| r.truth.as_ref().is_some_and(|t| t.correct == correct))
            .count()
    };
    assert_eq!(truth(true), 2_500);
    assert_eq!(truth(false), 1_250);
    assert_eq!(
        c.data.records.iter().filter(|r| r.tier == Tier::Standing).count(),
        320
    );
    assert_eq!(c.incidents.iter().filter(|i| i.wrong.len() == 3).count(), 250);
    assert_eq!(c.incidents.iter().filter(|i| i.wrong.len() == 2).count(), 250);
    assert_eq!(c.data.transcripts.len(), 10_000);
    let mut calls: BTreeMap<&str, usize> = BTreeMap::new();
    for t in &c.data.transcripts {
        *calls.entry(&t.thread).or_default() += 1;
    }
    assert!(
        c.incidents
            .iter()
            .all(|i| calls.get(i.id.as_str()).copied().unwrap_or(0) >= 6)
    );
    assert_eq!(queries_of(c, QueryClass::Changing).len(), 4_000);
    assert_eq!(queries_of(c, QueryClass::PostIncident).len(), 500);
    assert_eq!(queries_of(c, QueryClass::Ambiguous).len(), 60);
    assert_eq!(queries_of(c, QueryClass::Ordinary).len(), 320);
    assert_eq!(queries_of(c, QueryClass::OrdinaryNear).len(), 100);
    assert_eq!(c.pairs.len(), 60);
    assert_eq!(
        c.incidents.iter().filter(|i| i.next_update.is_some()).count(),
        250
    );
    let counts = c.manifest_counts();
    assert_eq!(counts.get("records_true"), Some(&2_500));
    assert_eq!(counts.get("queries_ordinary_near"), Some(&100));
}

#[test]
fn every_state_is_replaced_when_the_next_one_is_written() {
    let c = shared_corpus();
    let records = records_by_id(c);
    for i in &c.incidents {
        for k in 1..=5 {
            let r = records[i.state_id(k).as_str()];
            let truth = r.truth.as_ref().unwrap();
            assert_eq!(r.written_at, i.moments[k - 1]);
            let next = (k < 5).then(|| records[i.state_id(k + 1).as_str()].written_at);
            assert_eq!(truth.superseded_at, next, "{}", r.id);
            if k > 1 {
                assert!(r.written_at > records[i.state_id(k - 1).as_str()].written_at);
            }
        }
        assert_eq!(records[i.state_id(1).as_str()].valid_until, i.next_update);
        assert_eq!(records[i.state_id(4).as_str()].valid_until, Some(i.eta));
        assert!(i.moments[0] > i.start && i.moments[4] == i.end());
    }
}

#[test]
fn no_agent_states_a_record_before_they_could_know_it() {
    let c = shared_corpus();
    for (t, call) in c.data.transcripts.iter().zip(&c.calls) {
        assert_eq!(t.started_at, call.started_at);
        let incident = &c.incidents[call.incident];
        let latest = c.knowledge.latest_known(call.incident, call.agent, t.started_at);
        for id in &t.states {
            if let Some(k) = state_number(id) {
                let known_at = c.knowledge.known_at(call.incident, call.agent, k).unwrap();
                assert!(known_at <= t.started_at, "{}: {id} is learnt at {known_at}", t.id);
                assert!(known_at >= incident.start);
                assert_eq!(k, latest, "{}: states {id} but knows state {latest}", t.id);
                // Never more than five minutes ahead of the written update.
                assert!(
                    incident.moments[k - 1] - 300 <= t.started_at,
                    "{}: {id} too early",
                    t.id
                );
            } else {
                assert!(latest < 2, "{}: a wrong statement after the cause is known", t.id);
                assert!(t.started_at < incident.end());
            }
        }
        if latest >= 1 {
            assert!(t.states.contains(&incident.state_id(latest)));
        }
    }
}

#[test]
fn every_stated_key_term_is_in_the_agent_turns() {
    let c = shared_corpus();
    let records = records_by_id(c);
    let mut checked = 0;
    for t in &c.data.transcripts {
        let agent_text: Vec<&str> = t
            .turns
            .iter()
            .filter(|x| x.who == Speaker::Agent)
            .map(|x| x.text.as_str())
            .collect();
        let agent_text = fold(&agent_text.join(" "));
        for id in &t.states {
            let truth = records[id.as_str()].truth.as_ref().unwrap();
            assert!(!truth.key_terms.is_empty());
            for term in &truth.key_terms {
                assert!(
                    contains_folded(&agent_text, term),
                    "{}: {term:?} of {id} missing",
                    t.id
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 10_000);
}

#[test]
fn no_question_contains_a_key_term_of_its_gold_records() {
    let c = shared_corpus();
    let records = records_by_id(c);
    for q in &c.data.queries {
        let text = fold(&q.text);
        let related = q
            .gold
            .iter()
            .map(|g| g.record.as_str())
            .chain(q.harmful.iter().map(String::as_str));
        for id in related {
            if let Some(truth) = &records[id].truth {
                for term in &truth.key_terms {
                    assert!(
                        !contains_folded(&text, term),
                        "{}: {:?} copies {term:?} of {id}",
                        q.id,
                        q.text
                    );
                }
            }
        }
    }
}

#[test]
fn the_mean_transcript_length_is_850_tokens_give_or_take_50() {
    let (mean, sd, min, max) = transcript_token_stats(&shared_corpus().data.transcripts);
    assert!((800.0..=900.0).contains(&mean), "mean {mean}");
    assert!(sd > 60.0 && min < max, "lengths should vary: sd {sd}");
}

#[test]
fn the_planted_pairs_overlap_and_feed_the_ambiguous_questions() {
    let c = shared_corpus();
    let ambiguous = queries_of(c, QueryClass::Ambiguous);
    assert_eq!(ambiguous.len(), c.pairs.len());
    for (pair, q) in c.pairs.iter().zip(&ambiguous) {
        let (a, b) = (&c.incidents[pair.a], &c.incidents[pair.b]);
        assert_eq!(a.service, b.service);
        assert_ne!(a.region, b.region);
        assert!(
            a.start < b.end() && b.start < a.end(),
            "{} and {} do not overlap",
            a.id,
            b.id
        );
        assert!((pair.ask_from..pair.ask_until).contains(&q.asked_at));
        let (la, lb) = (a.latest_state_at(q.asked_at), b.latest_state_at(q.asked_at));
        assert!(
            (1..=4).contains(&la) && (1..=4).contains(&lb),
            "{}: not both in progress",
            q.id
        );
        let top: BTreeSet<&str> = q
            .gold
            .iter()
            .filter(|g| g.grade == 3)
            .map(|g| g.record.as_str())
            .collect();
        assert_eq!(
            top,
            BTreeSet::from([a.state_id(la).as_str(), b.state_id(lb).as_str()])
        );
        let others = c.incidents.iter().filter(|i| {
            i.service == a.service
                && i.id != a.id
                && i.id != b.id
                && i.start <= q.asked_at
                && q.asked_at < i.end()
        });
        assert_eq!(
            others.count(),
            0,
            "{}: a third incident of the service is in progress",
            q.id
        );
        assert!(
            c.pools.regions.iter().all(|r| !q.text.contains(r.as_str())),
            "{} names a region",
            q.id
        );
    }
}

#[test]
fn no_two_records_share_a_text() {
    let c = shared_corpus();
    assert_eq!(first_shared_text(&c.data.records), None);
    let mut copy = c.data.records[..3].to_vec();
    copy[2].text = format!(" {}\n", copy[0].text.replace(' ', "  "));
    assert_eq!(
        first_shared_text(&copy),
        Some((copy[0].id.clone(), copy[2].id.clone()))
    );
}

#[test]
fn a_wrong_statement_never_shares_a_full_key_term_set_with_a_true_state() {
    let c = shared_corpus();
    let mut by_thread: BTreeMap<&str, Vec<&Record>> = BTreeMap::new();
    for r in c.data.records.iter().filter(|r| r.thread.is_some()) {
        by_thread.entry(r.thread.as_deref().unwrap()).or_default().push(r);
    }
    for records in by_thread.values() {
        let (good, bad): (Vec<&Record>, Vec<&Record>) =
            records.iter().partition(|r| r.truth.as_ref().unwrap().correct);
        assert_eq!(good.len(), 5);
        for w in &bad {
            let w_terms = &w.truth.as_ref().unwrap().key_terms;
            let w_text = fold(&w.text);
            for t in &good {
                let t_terms = &t.truth.as_ref().unwrap().key_terms;
                let t_text = fold(&t.text);
                assert!(
                    !w_terms.iter().all(|k| contains_folded(&t_text, k)),
                    "{} inside {}",
                    w.id,
                    t.id
                );
                assert!(
                    !t_terms.iter().all(|k| contains_folded(&w_text, k)),
                    "{} inside {}",
                    t.id,
                    w.id
                );
            }
        }
    }
}

#[test]
fn incidents_on_one_service_and_region_never_overlap() {
    let c = shared_corpus();
    for (n, a) in c.incidents.iter().enumerate() {
        for b in &c.incidents[n + 1..] {
            if a.service == b.service && a.region == b.region {
                let apart = b.start >= a.end() + SAME_PLACE_GAP || a.start >= b.end() + SAME_PLACE_GAP;
                assert!(
                    apart,
                    "{} and {} share a service and region too closely",
                    a.id, b.id
                );
            }
        }
    }
}

fn shingles(t: &Transcript) -> HashSet<String> {
    let words: Vec<String> = t
        .turns
        .iter()
        .flat_map(|turn| turn.text.split_whitespace())
        .map(str::to_lowercase)
        .collect();
    words.windows(3).map(|w| w.join(" ")).collect()
}

#[test]
fn transcripts_of_one_incident_are_not_near_copies() {
    let c = shared_corpus();
    let mut by_thread: BTreeMap<&str, Vec<HashSet<String>>> = BTreeMap::new();
    let first_hundred: BTreeSet<&str> = c.incidents.iter().take(100).map(|i| i.id.as_str()).collect();
    for t in c
        .data
        .transcripts
        .iter()
        .filter(|t| first_hundred.contains(t.thread.as_str()))
    {
        by_thread.entry(&t.thread).or_default().push(shingles(t));
    }
    let mut worst: f64 = 0.0;
    for sets in by_thread.values() {
        for (n, a) in sets.iter().enumerate() {
            for b in &sets[n + 1..] {
                let shared = a.intersection(b).count();
                let jaccard = shared as f64 / (a.len() + b.len() - shared) as f64;
                worst = worst.max(jaccard);
            }
        }
    }
    assert!(
        worst < 0.5,
        "two calls of one incident share {worst:.2} of their word triples"
    );
}

#[test]
fn speech_errors_keep_ids_and_alter_more_words_at_a_higher_rate() {
    let c = shared_corpus();
    let original = &c.data.transcripts;
    let none = c.transcripts_with_speech_errors(0.0);
    assert_eq!(&none, original);
    let changed_turns = |copy: &[Transcript]| -> usize {
        original
            .iter()
            .zip(copy)
            .map(|(a, b)| {
                a.turns
                    .iter()
                    .zip(&b.turns)
                    .filter(|(x, y)| x.text != y.text)
                    .count()
            })
            .sum()
    };
    let low = c.transcripts_with_speech_errors(0.05);
    let high = c.transcripts_with_speech_errors(0.15);
    assert_eq!(high, c.transcripts_with_speech_errors(0.15));
    for copy in [&low, &high] {
        for (a, b) in original.iter().zip(copy.iter()) {
            assert_eq!(
                (&a.id, &a.thread, &a.states, a.turns.len()),
                (&b.id, &b.thread, &b.states, b.turns.len())
            );
            assert!(
                b.turns
                    .iter()
                    .zip(&a.turns)
                    .all(|(x, y)| x.ends_at_s == y.ends_at_s && !x.text.is_empty())
            );
        }
    }
    assert!((low[0].asr_error_rate - 0.05).abs() < 1e-6);
    let (l, h) = (changed_turns(&low), changed_turns(&high));
    assert!(l > 0 && h > l, "changed turns: {l} at 0.05, {h} at 0.15");
}

#[test]
fn every_wrong_statement_is_said_and_written_when_first_said() {
    let c = shared_corpus();
    for r in c
        .data
        .records
        .iter()
        .filter(|r| r.truth.as_ref().is_some_and(|t| !t.correct))
    {
        let stating: Vec<&Transcript> = c
            .data
            .transcripts
            .iter()
            .filter(|t| t.states.contains(&r.id))
            .collect();
        assert!(!stating.is_empty(), "{} is never said", r.id);
        let first = stating.iter().map(|t| t.started_at).min().unwrap();
        let last_end = stating
            .iter()
            .map(|t| t.started_at + t.turns.last().map_or(0, |x| x.ends_at_s.ceil() as i64))
            .max()
            .unwrap();
        assert!(
            first <= r.written_at && r.written_at <= last_end,
            "{} written at {}",
            r.id,
            r.written_at
        );
        let incident = incident_of(c, &r.id);
        let truth = r.truth.as_ref().unwrap();
        assert_eq!(truth.superseded_at, Some(incident.moments[1].max(r.written_at)));
    }
    // About one call in five before the agent knows the cause carries one.
    let share = c.stats.wrong_calls as f64 / c.stats.eligible_calls as f64;
    assert!((0.17..=0.25).contains(&share), "share {share}");
}

#[test]
fn questions_are_asked_when_the_specification_says() {
    let c = shared_corpus();
    for q in &c.data.queries {
        let top = q.gold.iter().find(|g| g.grade == 3).unwrap();
        match q.class {
            QueryClass::Changing => {
                let i = incident_of(c, &top.record);
                let k = state_number(&top.record).unwrap();
                assert!((1..=4).contains(&k));
                assert!(
                    i.moments[k - 1] <= q.asked_at && q.asked_at < i.moments[k],
                    "{}",
                    q.id
                );
                assert_eq!(q.gold.len(), k);
                assert_eq!(q.harmful.len(), i.wrong.len());
            }
            QueryClass::PostIncident => {
                let i = incident_of(c, &top.record);
                assert_eq!(state_number(&top.record), Some(5));
                assert!((i.end() + 3_600..=i.end() + 48 * 3_600).contains(&q.asked_at));
            }
            QueryClass::Ordinary | QueryClass::OrdinaryNear => {
                assert!(top.record.starts_with("doc-"));
                assert_eq!(q.gold.len(), 4);
                let during = |i: &&Incident| i.moments[0] <= q.asked_at && q.asked_at < i.end();
                assert!(
                    c.incidents.iter().any(|i| during(&i)),
                    "{} is not asked during an incident",
                    q.id
                );
            }
            QueryClass::Ambiguous => {}
        }
    }
}
