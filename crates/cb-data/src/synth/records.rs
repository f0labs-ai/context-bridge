//! The written records: five cumulative status updates per incident, the
//! planted wrong statements as call notes, and the help-article sections.

use super::SynthError;
use super::clock::{self, DAY, WINDOW_START};
use super::draw::{between, pick};
use super::incidents::{Incident, WrongClaim, WrongStatement};
use super::pools::Pools;
use super::spec;
use super::text::{Vars, capitalize_first, collapse_whitespace, fill, key_form};
use cb_types::{Phase, Record, RecordTruth, Tier};
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;

const PHASES: [Phase; 5] = [
    Phase::Investigating,
    Phase::RootCause,
    Phase::Workaround,
    Phase::FixInProgress,
    Phase::Resolved,
];
const STATUS: [&str; 5] = [
    "Investigating",
    "Identified",
    "Workaround",
    "Fix in progress",
    "Resolved",
];
/// The key term of a third state that has no workaround.
pub const NO_WORKAROUND: &str = "no workaround";

/// "Home internet in Quillhaven: No connection".
pub fn incident_title(pools: &Pools, incident: &Incident) -> String {
    let service = &pools.services[incident.service];
    let mode = &service.modes[incident.mode];
    format!(
        "{} in {}: {}",
        service.label, pools.regions[incident.region], mode.title
    )
}

/// The words state `k` (1–5) adds, which a statement of it must contain:
/// 1 the symptom and since when; 2 the cause; 3 the workaround (or "no
/// workaround"); 4 the fix and the expected time back; 5 the time it was fixed.
pub fn state_key_terms(pools: &Pools, incident: &Incident, k: usize) -> Result<Vec<String>, SynthError> {
    let service = &pools.services[incident.service];
    let mode = &service.modes[incident.mode];
    let cause = &mode.causes[incident.cause];
    Ok(match k {
        1 => vec![key_form(&mode.symptom), clock::clock(incident.start)?],
        2 => vec![key_form(&cause.cause)],
        3 => vec![match incident.workaround {
            Some(w) => key_form(&mode.workarounds[w]),
            None => NO_WORKAROUND.to_string(),
        }],
        4 => vec![key_form(&cause.fixes[incident.fix]), clock::clock(incident.eta)?],
        5 => vec![clock::clock(incident.end())?],
        _ => return Err(SynthError::Plan(format!("no state {k}"))),
    })
}

/// The words a wrong statement uses that no true state of its incident uses.
pub fn wrong_key_terms(
    pools: &Pools,
    incident: &Incident,
    statement: &WrongStatement,
) -> Result<Vec<String>, SynthError> {
    let service = &pools.services[incident.service];
    Ok(match statement.claim {
        WrongClaim::CustomerSide(i) => vec![key_form(&service.wrong_customer_side[i])],
        WrongClaim::ProviderSide(i) => vec![key_form(&service.wrong_provider_side[i])],
        WrongClaim::TimeBack => {
            let t = statement
                .time_back
                .ok_or_else(|| SynthError::Plan(format!("{}: wrong time back not set", incident.id)))?;
            vec![clock::clock(t)?]
        }
    })
}

fn incident_vars(pools: &Pools, incident: &Incident) -> Result<Vars, SynthError> {
    let service = &pools.services[incident.service];
    let mode = &service.modes[incident.mode];
    let cause = &mode.causes[incident.cause];
    let mut v = Vars::default()
        .with("service", service.name.as_str())
        .with("Service", capitalize_first(&service.name))
        .with("region", pools.regions[incident.region].as_str())
        .with("symptom", mode.symptom.as_str())
        .with("since", clock::clock(incident.start)?)
        .with("since_date", clock::date(incident.start)?)
        .with("cause", cause.cause.as_str())
        .with("fix", cause.fixes[incident.fix].as_str())
        .with("eta", clock::clock(incident.eta)?)
        .with("eta_date", clock::date(incident.eta)?)
        .with("resolved_at", clock::clock(incident.end())?)
        .with("resolved_date", clock::date(incident.end())?);
    if let Some(w) = incident.workaround {
        v.set("workaround", mode.workarounds[w].as_str());
    }
    if let Some(a) = incident.aftercare {
        v.set("aftercare", mode.aftercare[a].as_str());
    }
    if let Some(t) = incident.next_update {
        v.set("next_update", clock::clock(t)?);
        v.set("next_update_date", clock::date(t)?);
    }
    Ok(v)
}

/// The five status updates of an incident. Each restates everything known so
/// far, so the latest one is always the whole answer.
pub fn state_records(
    pools: &Pools,
    incident: &Incident,
    rng: &mut ChaCha20Rng,
) -> Result<Vec<Record>, SynthError> {
    let t = &pools.records;
    let vars = incident_vars(pools, incident)?;
    let title = incident_title(pools, incident);
    let mut say =
        |pool: &[String], what: &str| -> Result<String, SynthError> { fill(pick(rng, pool, what)?, &vars) };
    let workaround_pool = if incident.workaround.is_some() {
        &t.workaround
    } else {
        &t.no_workaround
    };
    let mut bodies: Vec<Vec<String>> = Vec::with_capacity(5);
    let mut first = vec![
        say(&t.problem, "problem")?,
        say(&t.investigating, "investigating")?,
    ];
    if incident.next_update.is_some() {
        first.push(say(&t.next_update, "next update")?);
    }
    bodies.push(first);
    bodies.push(vec![
        say(&t.problem, "problem")?,
        say(&t.cause, "cause")?,
        say(&t.fix_pending, "fix pending")?,
    ]);
    bodies.push(vec![
        say(&t.problem, "problem")?,
        say(&t.cause, "cause")?,
        say(workaround_pool, "workaround")?,
    ]);
    bodies.push(vec![
        say(&t.problem, "problem")?,
        say(&t.cause, "cause")?,
        say(workaround_pool, "workaround")?,
        say(&t.fix, "fix")?,
    ]);
    let mut last = vec![
        say(&t.problem_past, "problem past")?,
        say(&t.cause_was, "cause was")?,
    ];
    if incident.workaround.is_some() {
        last.push(say(&t.workaround_was, "workaround was")?);
    }
    last.push(say(&t.resolved, "resolved")?);
    if incident.aftercare.is_some() {
        last.push(say(&t.aftercare, "aftercare")?);
    }
    bodies.push(last);

    let mut records = Vec::with_capacity(5);
    for (n, body) in bodies.into_iter().enumerate() {
        let k = n + 1;
        records.push(Record {
            id: incident.state_id(k),
            text: format!("{title}. {}: {}", STATUS[n], body.join(" ")),
            written_at: incident.moments[n],
            valid_until: match k {
                1 => incident.next_update,
                4 => Some(incident.eta),
                _ => None,
            },
            thread: Some(incident.id.clone()),
            tier: Tier::Changing,
            title: Some(title.clone()),
            source: spec::SOURCE.to_string(),
            truth: Some(RecordTruth {
                correct: true,
                phase: PHASES[n],
                key_terms: state_key_terms(pools, incident, k)?,
                superseded_at: incident.moments.get(k).copied(),
            }),
        });
    }
    Ok(records)
}

/// A planted wrong statement as the note of the call that first stated it.
/// The note names its service, region and date, so no two notes read alike.
pub fn wrong_record(
    pools: &Pools,
    incident: &Incident,
    statement: &WrongStatement,
    id: String,
    written_at: i64,
    rng: &mut ChaCha20Rng,
) -> Result<Record, SynthError> {
    let service = &pools.services[incident.service];
    let t = &pools.records;
    let mut vars = Vars::default()
        .with("service", service.name.as_str())
        .with("Service", capitalize_first(&service.name))
        .with("region", pools.regions[incident.region].as_str())
        .with("call_date", clock::date(written_at)?);
    let pool = match statement.claim {
        WrongClaim::CustomerSide(i) => {
            vars.set("wrong", service.wrong_customer_side[i].as_str());
            &t.wrong_customer_side
        }
        WrongClaim::ProviderSide(i) => {
            vars.set("wrong", service.wrong_provider_side[i].as_str());
            &t.wrong_provider_side
        }
        WrongClaim::TimeBack => {
            let at = statement
                .time_back
                .ok_or_else(|| SynthError::Plan(format!("{}: wrong time back not set", incident.id)))?;
            vars.set("wrong_time", clock::clock(at)?);
            vars.set("wrong_date", clock::date(at)?);
            &t.wrong_time_back
        }
    };
    // A wrong statement belongs to the investigating phase: the root-cause
    // update replaces it. One first stated after that update is never current.
    let superseded_at = incident.moments[1].max(written_at);
    Ok(Record {
        id,
        text: fill(pick(rng, pool, "wrong statement note")?, &vars)?,
        written_at,
        valid_until: None,
        thread: Some(incident.id.clone()),
        tier: Tier::Changing,
        title: None,
        source: spec::SOURCE.to_string(),
        truth: Some(RecordTruth {
            correct: false,
            phase: Phase::Investigating,
            key_terms: wrong_key_terms(pools, incident, statement)?,
            superseded_at: Some(superseded_at),
        }),
    })
}

/// `doc-012.3`: section 3 of article 12 (both counted from 1).
pub fn doc_id(article: usize, section: usize) -> String {
    format!("doc-{:03}.{}", article + 1, section + 1)
}

/// One standing record per help-article section, published 30 to 700 days
/// before the window (all sections of an article at the same moment).
pub fn standing_records(pools: &Pools, rng: &mut ChaCha20Rng) -> Vec<Record> {
    let mut records = Vec::with_capacity(spec::STANDING_RECORDS);
    for (a, article) in pools.articles.iter().enumerate() {
        let written_at = WINDOW_START - between(rng, 30 * DAY, 700 * DAY);
        for (s, section) in article.sections.iter().enumerate() {
            records.push(Record {
                id: doc_id(a, s),
                text: format!("{}. {}", section.heading, section.text),
                written_at,
                valid_until: None,
                thread: None,
                tier: Tier::Standing,
                title: Some(article.title.clone()),
                source: spec::SOURCE.to_string(),
                truth: None,
            });
        }
    }
    records
}

/// The first pair of records whose texts are the same once whitespace is
/// collapsed (the search index files a passage under a hash of its text, so
/// two such records would become one).
pub fn first_shared_text(records: &[Record]) -> Option<(String, String)> {
    let mut seen: BTreeMap<String, &str> = BTreeMap::new();
    for record in records {
        if let Some(first) = seen.insert(collapse_whitespace(&record.text), &record.id) {
            return Some((first.to_string(), record.id.clone()));
        }
    }
    None
}
