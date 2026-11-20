//! The five classes of questions with graded gold (docs/SYNTH_SPEC.md section 5).
//! Question wording names the service (and, except for ambiguous questions, the
//! region) and nothing else of the incident, so it never copies a key term.

use super::SynthError;
use super::clock::HOUR;
use super::draw::{between, index, pick, pick_distinct};
use super::incidents::{Incident, PlantedPair};
use super::pools::{Intent, Pools, Service};
use super::records::doc_id;
use super::spec;
use super::text::{Vars, capitalize_first, fill};
use cb_types::{Gold, Query, QueryClass};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;

fn question_vars(service: &Service, region: Option<&str>) -> Vars {
    let plural = service.plural;
    let pick_form = |one: &'static str, many: &'static str| if plural { many } else { one };
    let mut v = Vars::default()
        .with("service", service.name.as_str())
        .with("Service", capitalize_first(&service.name))
        .with("is", pick_form("is", "are"))
        .with("Is", pick_form("Is", "Are"))
        .with("has", pick_form("has", "have"))
        .with("Has", pick_form("Has", "Have"))
        .with("was", pick_form("was", "were"));
    if let Some(region) = region {
        v.set("region", region);
    }
    v
}

/// Grade 3 for state `latest`, grade 1 for every earlier state.
fn gold_for_latest(incident: &Incident, latest: usize) -> Vec<Gold> {
    let mut gold = vec![Gold {
        record: incident.state_id(latest),
        grade: 3,
    }];
    gold.extend((1..latest).map(|k| Gold {
        record: incident.state_id(k),
        grade: 1,
    }));
    gold
}

/// Grade 3 for the section, grade 1 for its article's other sections.
fn gold_for_section(article: usize, section: usize, sections: usize) -> Vec<Gold> {
    let mut gold = vec![Gold {
        record: doc_id(article, section),
        grade: 3,
    }];
    gold.extend((0..sections).filter(|s| *s != section).map(|s| Gold {
        record: doc_id(article, s),
        grade: 1,
    }));
    gold
}

/// Every question of the data set, class by class.
pub fn write_questions(
    pools: &Pools,
    incidents: &[Incident],
    pairs: &[PlantedPair],
    wrong_ids: &[Vec<String>],
    rng: &mut ChaCha20Rng,
) -> Result<Vec<Query>, SynthError> {
    let mut queries = Vec::with_capacity(
        spec::CHANGING_QUESTIONS
            + spec::POST_INCIDENT_QUESTIONS
            + spec::AMBIGUOUS_QUESTIONS
            + spec::ORDINARY_QUESTIONS
            + spec::ORDINARY_NEAR_QUESTIONS,
    );
    let harmful = |i: usize| wrong_ids.get(i).cloned().unwrap_or_default();

    // Changing: two per state interval of states 1–4, asked while it is the latest.
    for (i, incident) in incidents.iter().enumerate() {
        let service = &pools.services[incident.service];
        let vars = question_vars(service, Some(&pools.regions[incident.region]));
        for k in 1..=4 {
            for (n, intent) in pick_distinct(rng, &Intent::ALL, 2).into_iter().enumerate() {
                let asked_at = between(rng, incident.moments[k - 1], incident.moments[k] - 1);
                let template = pick(rng, &pools.questions.phrasings(*intent).with_region, "question")?;
                queries.push(Query {
                    id: format!("q-{}-s{k}-{}", incident.id, n + 1),
                    text: fill(template, &vars)?,
                    asked_at,
                    class: QueryClass::Changing,
                    gold: gold_for_latest(incident, k),
                    thread: Some(incident.id.clone()),
                    harmful: harmful(i),
                });
            }
        }
    }

    // After the incident: one each, 1 to 48 hours after it is resolved.
    for (i, incident) in incidents.iter().enumerate() {
        let service = &pools.services[incident.service];
        let vars = question_vars(service, Some(&pools.regions[incident.region]));
        let template = pick(rng, &pools.questions.after, "question after")?;
        queries.push(Query {
            id: format!("q-{}-post", incident.id),
            text: fill(template, &vars)?,
            asked_at: incident.end() + between(rng, HOUR, 48 * HOUR),
            class: QueryClass::PostIncident,
            gold: gold_for_latest(incident, 5),
            thread: Some(incident.id.clone()),
            harmful: harmful(i),
        });
    }

    // Ambiguous: one per planted pair, inside its asking window, naming the
    // service and no region.
    for (n, pair) in pairs.iter().enumerate() {
        let (a, b) = (&incidents[pair.a], &incidents[pair.b]);
        let asked_at = between(rng, pair.ask_from, pair.ask_until - 1);
        let service = &pools.services[a.service];
        let intent = Intent::ALL[index(rng, Intent::ALL.len())];
        let template = pick(rng, &pools.questions.phrasings(intent).without_region, "question")?;
        let mut gold = Vec::new();
        for incident in [a, b] {
            let latest = incident.latest_state_at(asked_at);
            if latest == 0 || latest == 5 {
                return Err(SynthError::Plan(format!(
                    "pair {n}: {} is not in progress",
                    incident.id
                )));
            }
            gold.extend(gold_for_latest(incident, latest));
        }
        let mut harmful_ids = harmful(pair.a);
        harmful_ids.extend(harmful(pair.b));
        queries.push(Query {
            id: format!("q-amb-{:02}", n + 1),
            text: fill(template, &question_vars(service, None))?,
            asked_at,
            class: QueryClass::Ambiguous,
            gold,
            thread: None,
            harmful: harmful_ids,
        });
    }

    write_documentation_questions(pools, incidents, rng, &mut queries)?;
    Ok(queries)
}

/// A moment inside an incident once its first record is out.
fn moment_inside(incident: &Incident, rng: &mut ChaCha20Rng) -> i64 {
    between(rng, incident.moments[0], incident.end() - 1)
}

/// Ordinary questions (one per section, asked during an incident on another
/// service) and near-ordinary ones (100 sections of articles about a service,
/// asked during an incident on that same service, in the other phrasing).
fn write_documentation_questions(
    pools: &Pools,
    incidents: &[Incident],
    rng: &mut ChaCha20Rng,
    queries: &mut Vec<Query>,
) -> Result<(), SynthError> {
    let mut near_candidates = Vec::new();
    for (a, article) in pools.articles.iter().enumerate() {
        let service = article.service.as_deref().and_then(|id| pools.service_index(id));
        let elsewhere: Vec<&Incident> = incidents
            .iter()
            .filter(|i| service.is_none_or(|s| i.service != s))
            .collect();
        for (s, section) in article.sections.iter().enumerate() {
            let incident = *pick(rng, &elsewhere, "incidents on another service")?;
            let q = index(rng, section.questions.len());
            queries.push(Query {
                id: format!("q-{}", doc_id(a, s)),
                text: section.questions[q].clone(),
                asked_at: moment_inside(incident, rng),
                class: QueryClass::Ordinary,
                gold: gold_for_section(a, s, article.sections.len()),
                thread: None,
                harmful: Vec::new(),
            });
            if let Some(service) = service {
                near_candidates.push((a, s, service, 1 - q));
            }
        }
    }
    near_candidates.shuffle(rng);
    if near_candidates.len() < spec::ORDINARY_NEAR_QUESTIONS {
        return Err(SynthError::Pool(
            "too few sections about a service for near-ordinary questions".into(),
        ));
    }
    for (a, s, service, q) in near_candidates.into_iter().take(spec::ORDINARY_NEAR_QUESTIONS) {
        let same: Vec<&Incident> = incidents.iter().filter(|i| i.service == service).collect();
        let incident = *pick(rng, &same, "incidents on the article's service")?;
        let article = &pools.articles[a];
        queries.push(Query {
            id: format!("q-near-{}", doc_id(a, s)),
            text: article.sections[s].questions[q].clone(),
            asked_at: moment_inside(incident, rng),
            class: QueryClass::OrdinaryNear,
            gold: gold_for_section(a, s, article.sections.len()),
            thread: None,
            harmful: Vec::new(),
        });
    }
    Ok(())
}
