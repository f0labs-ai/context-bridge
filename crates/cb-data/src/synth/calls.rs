//! Who calls when, which agent answers, what that agent knows at that moment,
//! and which early calls carry a wrong statement (docs/SYNTH_SPEC.md section 3).

use super::SynthError;
use super::clock::{HOUR, MINUTE};
use super::draw::{between, chance, index};
use super::incidents::{Incident, WrongClaim, wrong_time_back};
use super::pools::{AGENTS, Pools, Style};
use super::spec;
use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand_chacha::ChaCha20Rng;

/// Share of calls made after the incident is resolved.
const AFTER_RESOLUTION_SHARE: f64 = 0.06;
/// How long after the resolution those calls come, in seconds.
const AFTER_RESOLUTION: (i64, i64) = (5 * MINUTE, 12 * HOUR);
/// Calls during an incident come in an early burst (this share of them, spread
/// over the first `EARLY_BURST_SPAN` of the incident) and a steady trickle
/// over the whole incident (the rest).
const EARLY_BURST_SHARE: f64 = 0.62;
const EARLY_BURST_SPAN: f64 = 0.15;
/// Chance that an agent who does not yet know the cause states a wrong one.
/// Every wrong statement is also stated at least once, which lifts the share
/// of such calls that carry one to about one in five.
pub const WRONG_STATEMENT_CHANCE: f64 = 0.1;
/// How far an agent's moment of learning a state lies from the state's own
/// moment, in seconds, before or after.
const KNOWLEDGE_NOISE: (i64, i64) = (2 * MINUTE, 5 * MINUTE);

/// A support agent.
#[derive(Clone, Debug)]
pub struct Agent {
    /// `agent-07`: opaque, written into `Transcript.agent`.
    pub id: String,
    /// The first name the agent gives on the phone.
    pub name: String,
    pub style: Style,
}

/// Eighty agents, twenty for each way of speaking.
pub fn hire_agents(pools: &Pools, rng: &mut ChaCha20Rng) -> Vec<Agent> {
    let mut styles: Vec<Style> = Style::ALL
        .iter()
        .flat_map(|s| std::iter::repeat_n(*s, AGENTS / Style::ALL.len()))
        .collect();
    styles.shuffle(rng);
    styles
        .into_iter()
        .zip(&pools.people.agent_names)
        .enumerate()
        .map(|(i, (style, name))| Agent {
            id: format!("agent-{:02}", i + 1),
            name: name.clone(),
            style,
        })
        .collect()
}

/// When each agent learns each state of each incident: the state's moment
/// plus or minus 2 to 5 minutes, never before the incident starts, and never
/// before the resolution for the resolved state (nobody can report a repair
/// before it is finished).
pub struct Knowledge {
    agents: usize,
    known_at: Vec<[i64; 5]>,
}

impl Knowledge {
    pub fn draw(incidents: &[Incident], agents: usize, rng: &mut ChaCha20Rng) -> Knowledge {
        let mut known_at = Vec::with_capacity(incidents.len() * agents);
        for incident in incidents {
            for _ in 0..agents {
                let mut row = [0i64; 5];
                for (k, slot) in row.iter_mut().enumerate() {
                    let size = between(rng, KNOWLEDGE_NOISE.0, KNOWLEDGE_NOISE.1);
                    let signed = if rng.random::<bool>() { size } else { -size };
                    let mut t = (incident.moments[k] + signed).max(incident.start);
                    if k == 4 {
                        t = t.max(incident.moments[4]);
                    }
                    *slot = t;
                }
                known_at.push(row);
            }
        }
        Knowledge { agents, known_at }
    }

    fn row(&self, incident: usize, agent: usize) -> Option<&[i64; 5]> {
        self.known_at.get(incident * self.agents + agent)
    }

    /// When `agent` learns state `k` (1–5) of `incident`.
    pub fn known_at(&self, incident: usize, agent: usize, k: usize) -> Option<i64> {
        self.row(incident, agent)
            .and_then(|row| row.get(k.wrapping_sub(1)).copied())
    }

    /// The latest state (1–5) `agent` knows of `incident` at `t`; 0 for none.
    pub fn latest_known(&self, incident: usize, agent: usize, t: i64) -> usize {
        self.row(incident, agent)
            .and_then(|row| (1..=5).rev().find(|k| row[k - 1] <= t))
            .unwrap_or(0)
    }
}

/// One call to plan: about which incident, with which agent, when, and which
/// of the incident's wrong statements (by index) the agent states, if any.
#[derive(Clone, Debug)]
pub struct Call {
    pub incident: usize,
    pub agent: usize,
    pub started_at: i64,
    pub wrong: Option<usize>,
}

/// The calls, in order of start, with what it took to make them.
pub struct Schedule {
    pub calls: Vec<Call>,
    /// Calls in which the agent did not yet know the cause.
    pub eligible_calls: usize,
    /// Calls in which the agent stated a wrong statement.
    pub wrong_calls: usize,
    /// Calls moved early so every wrong statement could be stated.
    pub moved_calls: usize,
}

/// A call time: a few after the resolution; the rest front-loaded, most in an
/// early burst when the service fails and the others spread evenly over the
/// incident.
fn draw_call_time(incident: &Incident, rng: &mut ChaCha20Rng) -> i64 {
    if chance(rng, AFTER_RESOLUTION_SHARE) {
        return incident.end() + between(rng, AFTER_RESOLUTION.0, AFTER_RESOLUTION.1);
    }
    let span = if chance(rng, EARLY_BURST_SHARE) {
        EARLY_BURST_SPAN
    } else {
        1.0
    };
    let share: f64 = rng.random::<f64>() * span;
    incident.start + (share * incident.length as f64).floor() as i64
}

/// How many calls each incident gets: six each, and the rest shared out at
/// random with long incidents weighted three times, medium twice, short once.
fn calls_per_incident(incidents: &[Incident], rng: &mut ChaCha20Rng) -> Result<Vec<usize>, SynthError> {
    let base = spec::MIN_CALLS_PER_INCIDENT;
    let mut counts = vec![base; incidents.len()];
    let extra = spec::TRANSCRIPTS
        .checked_sub(base * incidents.len())
        .ok_or_else(|| SynthError::Plan("more base calls than calls".into()))?;
    let mut cumulative = Vec::with_capacity(incidents.len());
    let mut total = 0u64;
    for incident in incidents {
        total += incident.class.call_weight();
        cumulative.push(total);
    }
    for _ in 0..extra {
        let x = rng.random_range(0..total);
        let i = cumulative.partition_point(|c| *c <= x);
        let slot = counts
            .get_mut(i)
            .ok_or_else(|| SynthError::Plan("weighted draw out of range".into()))?;
        *slot += 1;
    }
    Ok(counts)
}

/// Plan every call, choose the calls that carry a wrong statement (each wrong
/// statement at least once), and set each wrong time back after the last call
/// that states it.
pub fn schedule_calls(
    incidents: &mut [Incident],
    knowledge: &Knowledge,
    agents: usize,
    rng: &mut ChaCha20Rng,
) -> Result<Schedule, SynthError> {
    let counts = calls_per_incident(incidents, rng)?;
    let mut all = Vec::with_capacity(spec::TRANSCRIPTS);
    let (mut eligible_total, mut wrong_total, mut moved_total) = (0, 0, 0);
    for (i, count) in counts.iter().enumerate() {
        let incident = &incidents[i];
        let mut calls: Vec<Call> = (0..*count)
            .map(|_| {
                let started_at = draw_call_time(incident, rng);
                Call {
                    incident: i,
                    agent: index(rng, agents),
                    started_at,
                    wrong: None,
                }
            })
            .collect();
        let eligible =
            |c: &Call| c.started_at < incident.end() && knowledge.latest_known(i, c.agent, c.started_at) < 2;
        let needed = incident.wrong.len();
        // Too few calls before the cause is known: move the earliest other call
        // into the first half of the time before the cause, with an agent who
        // learns the cause late.
        while calls.iter().filter(|c| eligible(c)).count() < needed {
            let pos = calls
                .iter()
                .enumerate()
                .filter(|(_, c)| !eligible(c))
                .min_by_key(|(n, c)| (c.started_at, *n))
                .map(|(n, _)| n)
                .ok_or_else(|| SynthError::Plan(format!("{}: no call to move", incident.id)))?;
            let t = incident.start
                + index(rng, ((incident.moments[1] - incident.start) / 2).max(1) as usize) as i64;
            let late: Vec<usize> = (0..agents)
                .filter(|a| knowledge.known_at(i, *a, 2).is_some_and(|k| k > t))
                .collect();
            let agent = *late
                .choose(rng)
                .ok_or_else(|| SynthError::Plan(format!("{}: every agent knows the cause", incident.id)))?;
            calls[pos].started_at = t;
            calls[pos].agent = agent;
            moved_total += 1;
        }
        let candidates: Vec<usize> = (0..calls.len()).filter(|n| eligible(&calls[*n])).collect();
        let mut chosen: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|_| chance(rng, WRONG_STATEMENT_CHANCE))
            .collect();
        if chosen.len() < needed {
            let mut rest: Vec<usize> = candidates
                .iter()
                .copied()
                .filter(|n| !chosen.contains(n))
                .collect();
            rest.shuffle(rng);
            let missing = needed - chosen.len();
            chosen.extend(rest.into_iter().take(missing));
        }
        chosen.shuffle(rng);
        for (n, call) in chosen.iter().enumerate() {
            calls[*call].wrong = Some(if n < needed { n } else { index(rng, needed) });
        }
        eligible_total += candidates.len();
        wrong_total += chosen.len();
        let mut times = Vec::new();
        for (j, statement) in incident.wrong.iter().enumerate() {
            if statement.claim == WrongClaim::TimeBack {
                let last = calls
                    .iter()
                    .filter(|c| c.wrong == Some(j))
                    .map(|c| c.started_at)
                    .max()
                    .ok_or_else(|| SynthError::Plan(format!("{}: wrong time never stated", incident.id)))?;
                times.push((j, wrong_time_back(incident, last, rng)?));
            }
        }
        for (j, t) in times {
            incidents[i].wrong[j].time_back = Some(t);
        }
        all.extend(calls);
    }
    all.sort_by_key(|c| (c.started_at, c.incident));
    Ok(Schedule {
        calls: all,
        eligible_calls: eligible_total,
        wrong_calls: wrong_total,
        moved_calls: moved_total,
    })
}
