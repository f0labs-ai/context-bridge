//! One call written out turn by turn (docs/SYNTH_SPEC.md section 3): greeting,
//! the customer's problem, identity checks, the agent looking into it, the
//! agent's statement, follow-up questions, sometimes troubleshooting, a close.
//!
//! Every part is drawn from the pools. Optional parts (a probe, small talk, a
//! recap, extra checks…) are drawn too, then kept in random order while they
//! fit the call's target length, so calls of one incident differ in phrasing,
//! in which parts they have and in length.

use super::SynthError;
use super::calls::Agent;
use super::clock::{self, elapsed_phrase};
use super::draw::{between, chance, index, pick, pick_distinct};
use super::incidents::{Incident, WrongClaim, WrongStatement};
use super::pools::{Exchange, FollowUp, FollowUpWhen, Pools, ServiceKind, Style};
use super::text::{Vars, capitalize_first, fill, word_count};
use cb_types::{Speaker, Turn};
use rand::Rng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;

/// Speaking rate the turn times assume.
const WORDS_PER_SECOND: f64 = 2.5;
/// The short pause before each turn, in seconds.
const PAUSE_S: (f64, f64) = (0.3, 1.2);
/// Mean words a call aims at for wordy and for brief agents, and how far one
/// call's aim may lie from it (a triangular spread). Chosen so the corpus mean
/// is close to 640 words, 850 tokens at 1.33 tokens a word.
const TARGET_WORDY: f64 = 772.0;
const TARGET_BRIEF: f64 = 570.0;
const TARGET_SPREAD: f64 = 180.0;

/// Everything that decides what is said in one call.
pub struct CallSetting<'a> {
    pub pools: &'a Pools,
    pub incident: &'a Incident,
    pub agent: &'a Agent,
    pub started_at: i64,
    /// The latest state the agent knows when the call starts; 0 for none.
    pub known: usize,
    /// The wrong statement the agent makes in this call, if any.
    pub wrong: Option<&'a WrongStatement>,
}

/// A written call: its turns, and when the wrong statement was said (the end
/// of its turn, in seconds from the start of the call).
pub struct WrittenCall {
    pub turns: Vec<Turn>,
    pub wrong_said_at_s: Option<f32>,
}

/// Where a part goes in the call. Parts are ordered by slot, then by their
/// position within the slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Slot {
    Greeting,
    Opening,
    Empathy,
    Reassurance,
    Identity,
    Clarify,
    TroubleshootFirst,
    Lookup,
    SmallTalk,
    Statement,
    TroubleshootAfter,
    Diagnosis,
    FollowUp,
    SmallTalkLate,
    Recap,
    Close,
}

struct Utterance {
    who: Speaker,
    text: String,
    /// Carries the planted wrong statement.
    wrong: bool,
}

fn agent(text: String) -> Utterance {
    Utterance {
        who: Speaker::Agent,
        text,
        wrong: false,
    }
}

fn customer(text: String) -> Utterance {
    Utterance {
        who: Speaker::Customer,
        text,
        wrong: false,
    }
}

fn scripted(exchange: &Exchange) -> Vec<Utterance> {
    exchange
        .iter()
        .map(|line| Utterance {
            who: line.who,
            text: line.text.clone(),
            wrong: false,
        })
        .collect()
}

struct Part {
    at: (Slot, u8),
    seq: usize,
    lines: Vec<Utterance>,
    words: usize,
}

/// The parts of one call before the optional ones are chosen.
#[derive(Default)]
struct Script {
    required: Vec<Part>,
    optional: Vec<Part>,
    seq: usize,
}

impl Script {
    fn part(&mut self, slot: Slot, sub: u8, lines: Vec<Utterance>) -> Part {
        self.seq += 1;
        let words = lines.iter().map(|l| word_count(&l.text)).sum();
        Part {
            at: (slot, sub),
            seq: self.seq,
            lines,
            words,
        }
    }

    fn must(&mut self, slot: Slot, sub: u8, lines: Vec<Utterance>) {
        let part = self.part(slot, sub, lines);
        self.required.push(part);
    }

    fn may(&mut self, slot: Slot, sub: u8, lines: Vec<Utterance>) {
        let part = self.part(slot, sub, lines);
        self.optional.push(part);
    }

    /// Keep the optional parts, taken in random order, that still fit under
    /// `target` words; put every part in its place; join consecutive lines of
    /// one speaker into one turn.
    fn assemble(self, target: usize, rng: &mut ChaCha20Rng) -> Vec<Utterance> {
        let Script {
            required, optional, ..
        } = self;
        let mut total: usize = required.iter().map(|p| p.words).sum();
        let mut order: Vec<usize> = (0..optional.len()).collect();
        order.shuffle(rng);
        let mut keep = vec![false; optional.len()];
        for i in order {
            if total + optional[i].words <= target {
                keep[i] = true;
                total += optional[i].words;
            }
        }
        let mut parts = required;
        parts.extend(optional.into_iter().zip(keep).filter(|(_, k)| *k).map(|(p, _)| p));
        parts.sort_by_key(|p| (p.at, p.seq));
        let mut out: Vec<Utterance> = Vec::new();
        for line in parts.into_iter().flat_map(|p| p.lines) {
            match out.last_mut() {
                Some(last) if last.who == line.who => {
                    last.text.push(' ');
                    last.text.push_str(&line.text);
                    last.wrong |= line.wrong;
                }
                _ => out.push(line),
            }
        }
        out
    }
}

fn say(rng: &mut ChaCha20Rng, pool: &[String], vars: &Vars, what: &str) -> Result<String, SynthError> {
    fill(pick(rng, pool, what)?, vars)
}

/// An invented caller.
struct Caller {
    name: String,
    first: String,
    account: String,
    phone: String,
    address: String,
    email: String,
}

fn invent_caller(pools: &Pools, region: &str, rng: &mut ChaCha20Rng) -> Result<Caller, SynthError> {
    let people = &pools.people;
    let first = pick(rng, &people.first_names, "first names")?.clone();
    let surname = pick(rng, &people.surnames, "surnames")?.clone();
    let account = format!("MC{:08}", between(rng, 10_000_000, 99_999_999));
    // Phone numbers from the ranges set aside for drama, so none is real.
    let phone = if chance(rng, 0.7) {
        format!("07700 900{:03}", between(rng, 0, 999))
    } else {
        format!("01632 960{:03}", between(rng, 0, 999))
    };
    let street = pick(rng, &people.streets, "streets")?;
    let address = format!("{} {street}, {region}", between(rng, 1, 180));
    let domain = pick(rng, &people.email_domains, "email domains")?;
    let email = format!("{}.{}@{domain}", first.to_lowercase(), surname.to_lowercase());
    Ok(Caller {
        name: format!("{first} {surname}"),
        first,
        account,
        phone,
        address,
        email,
    })
}

/// The values the templates of this call may use: only what the agent knows.
fn call_vars(set: &CallSetting, caller: &Caller, rng: &mut ChaCha20Rng) -> Result<Vars, SynthError> {
    let pools = set.pools;
    let incident = set.incident;
    let service = &pools.services[incident.service];
    let mode = &service.modes[incident.mode];
    let now = set.started_at;
    let say_today = chance(rng, 0.5);
    let day = |t: i64| clock::day_phrase(t, now, say_today);
    let mut v = Vars::default()
        .with("agent", set.agent.name.as_str())
        .with(
            "my_service",
            pick(rng, &service.customer_names, "customer names")?.as_str(),
        )
        .with("elapsed", elapsed_phrase(now - incident.start, index(rng, 3)))
        .with("name", caller.name.as_str())
        .with("first", caller.first.as_str())
        .with("account", caller.account.as_str())
        .with("phone", caller.phone.as_str())
        .with("address", caller.address.as_str())
        .with("email", caller.email.as_str())
        .with("service", service.name.as_str())
        .with("Service", capitalize_first(&service.name))
        .with("region", pools.regions[incident.region].as_str());
    let known = set.known;
    if known >= 1 {
        v.set("symptom", mode.symptom.as_str());
        v.set("since", clock::clock(incident.start)?);
        v.set("since_day", day(incident.start)?);
        if let Some(t) = incident.next_update {
            v.set("next_update", clock::clock(t)?);
            v.set("next_update_day", day(t)?);
        }
    }
    let cause = &mode.causes[incident.cause];
    if known >= 2 {
        v.set("cause", cause.cause.as_str());
    }
    if known >= 3
        && let Some(w) = incident.workaround
    {
        v.set("workaround", mode.workarounds[w].as_str());
    }
    if known >= 4 {
        v.set("fix", cause.fixes[incident.fix].as_str());
        v.set("eta", clock::clock(incident.eta)?);
        v.set("eta_day", day(incident.eta)?);
    }
    if known >= 5 {
        v.set("resolved_at", clock::clock(incident.end())?);
        v.set("resolved_day", day(incident.end())?);
        if let Some(a) = incident.aftercare {
            v.set("aftercare", mode.aftercare[a].as_str());
        }
    }
    if let Some(statement) = set.wrong {
        match statement.claim {
            WrongClaim::CustomerSide(i) => v.set("wrong", service.wrong_customer_side[i].as_str()),
            WrongClaim::ProviderSide(i) => v.set("wrong", service.wrong_provider_side[i].as_str()),
            WrongClaim::TimeBack => {
                let t = statement
                    .time_back
                    .ok_or_else(|| SynthError::Plan(format!("{}: wrong time back not set", incident.id)))?;
                v.set("wrong_time", clock::clock(t)?);
                v.set("wrong_day", day(t)?);
            }
        }
    }
    Ok(v)
}

/// Whether a follow-up question fits what the agent knows and the service.
fn follow_up_fits(follow_up: &FollowUp, known: usize, kind: ServiceKind) -> bool {
    let when = match follow_up.when {
        FollowUpWhen::Any => true,
        FollowUpWhen::Ongoing => (1..=4).contains(&known),
        FollowUpWhen::BeforeWorkaround => (1..=2).contains(&known),
        FollowUpWhen::AreaKnown => known >= 1,
        FollowUpWhen::NothingKnown => known == 0,
        FollowUpWhen::NoEta => (1..=3).contains(&known),
        FollowUpWhen::NoCause => known == 1,
        FollowUpWhen::Resolved => known == 5,
    };
    when && (follow_up.kinds.is_empty() || follow_up.kinds.contains(&kind))
}

/// One identity question and its answer, sometimes after a thank-you.
fn identity_step(
    rng: &mut ChaCha20Rng,
    ack: Option<&String>,
    ask: &[String],
    give: &[String],
    vars: &Vars,
) -> Result<Vec<Utterance>, SynthError> {
    let question = say(rng, ask, vars, "identity question")?;
    let question = match ack {
        Some(ack) if chance(rng, 0.5) => format!("{} {question}", fill(ack, vars)?),
        _ => question,
    };
    Ok(vec![
        agent(question),
        customer(say(rng, give, vars, "identity answer")?),
    ])
}

fn target_words(style: Style, rng: &mut ChaCha20Rng) -> usize {
    let mean = if style.is_wordy() {
        TARGET_WORDY
    } else {
        TARGET_BRIEF
    };
    let spread = (rng.random::<f64>() + rng.random::<f64>() - 1.0) * TARGET_SPREAD;
    (mean + spread).round().max(0.0) as usize
}

/// Write one call.
pub fn write_call(set: &CallSetting, rng: &mut ChaCha20Rng) -> Result<WrittenCall, SynthError> {
    let pools = set.pools;
    let incident = set.incident;
    let service = &pools.services[incident.service];
    let mode = &service.modes[incident.mode];
    let style = set.agent.style;
    let frame = &pools.frame;
    let known = set.known;
    if set.wrong.is_some() && known >= 2 {
        return Err(SynthError::Plan(format!(
            "{}: a wrong statement after the cause is known",
            incident.id
        )));
    }
    let caller = invent_caller(pools, &pools.regions[incident.region], rng)?;
    let vars = call_vars(set, &caller, rng)?;
    let mut s = Script::default();

    // Greeting and the caller's account of the problem.
    s.must(
        Slot::Greeting,
        0,
        vec![agent(say(rng, frame.greeting.get(style), &vars, "greeting")?)],
    );
    s.must(
        Slot::Opening,
        0,
        vec![customer(say(rng, &frame.opener, &vars, "opener")?)],
    );
    s.may(
        Slot::Opening,
        1,
        vec![agent(say(rng, frame.probe.get(style), &vars, "probe")?)],
    );
    s.must(
        Slot::Opening,
        2,
        vec![customer(pick(rng, &mode.customer_says, "customer says")?.clone())],
    );
    s.may(
        Slot::Opening,
        3,
        vec![customer(say(rng, &frame.elapsed_sentence, &vars, "elapsed")?)],
    );
    s.may(
        Slot::Opening,
        4,
        vec![customer(say(
            rng,
            frame.impact.get(service.kind),
            &vars,
            "impact",
        )?)],
    );
    s.must(
        Slot::Empathy,
        0,
        vec![agent(say(rng, frame.empathy.get(style), &vars, "empathy")?)],
    );
    s.may(
        Slot::Reassurance,
        0,
        scripted(pick(rng, &pools.reassurance, "reassurance")?),
    );

    // Identity checks: the name first, the rest in random order.
    s.must(
        Slot::Identity,
        0,
        vec![
            agent(say(rng, &frame.ask_name, &vars, "ask name")?),
            customer(say(rng, &frame.give_name, &vars, "give name")?),
        ],
    );
    let mut order: [u8; 4] = [1, 2, 3, 4];
    order.shuffle(rng);
    // A different thank-you each time, so a call never repeats one.
    let acks = pick_distinct(rng, &frame.ack, 5);
    let account = identity_step(
        rng,
        acks.first().copied(),
        &frame.ask_account,
        &frame.give_account,
        &vars,
    )?;
    s.must(Slot::Identity, order[0], account);
    let phone = identity_step(
        rng,
        acks.get(1).copied(),
        &frame.ask_phone,
        &frame.give_phone,
        &vars,
    )?;
    s.must(Slot::Identity, order[1], phone);
    let address = identity_step(
        rng,
        acks.get(2).copied(),
        &frame.ask_address,
        &frame.give_address,
        &vars,
    )?;
    s.may(Slot::Identity, order[2], address);
    let email = identity_step(
        rng,
        acks.get(3).copied(),
        &frame.ask_email,
        &frame.give_email,
        &vars,
    )?;
    s.may(Slot::Identity, order[3], email);
    let closing_ack = acks
        .get(4)
        .copied()
        .ok_or_else(|| SynthError::Pool("need five acknowledgements".into()))?;
    s.must(Slot::Identity, 9, vec![agent(fill(closing_ack, &vars)?)]);

    // Clarifying questions, then generic troubleshooting: before the lookup
    // when the agent will find the incident, after it when nothing is known.
    for (n, exchange) in pick_distinct(rng, pools.clarify.get(service.kind), 3)
        .into_iter()
        .enumerate()
    {
        if n == 0 {
            s.must(Slot::Clarify, 0, scripted(exchange));
        } else {
            s.may(Slot::Clarify, n as u8, scripted(exchange));
        }
    }
    let steps = pick_distinct(rng, pools.troubleshooting.get(service.kind), 2);
    for (n, exchange) in steps.into_iter().enumerate() {
        if known == 0 && n == 0 {
            s.must(Slot::TroubleshootAfter, 0, scripted(exchange));
        } else if known == 0 {
            s.may(Slot::TroubleshootAfter, n as u8, scripted(exchange));
        } else {
            s.may(Slot::TroubleshootFirst, n as u8, scripted(exchange));
        }
    }

    // The agent looks into it.
    s.must(
        Slot::Lookup,
        0,
        vec![
            agent(say(rng, frame.lookup.get(style), &vars, "lookup")?),
            customer(say(rng, &frame.hold_ok, &vars, "hold ok")?),
        ],
    );
    // Small talk while the agent waits may be about the wait; later it may not.
    let talk = pick_distinct(rng, &pools.small_talk, 2);
    let waiting = pick(rng, &pools.small_talk_waiting, "small talk while waiting")?;
    let (early, late) = if chance(rng, 0.5) {
        (Some(waiting), talk.first().copied())
    } else {
        (talk.first().copied(), talk.get(1).copied())
    };
    if let Some(exchange) = early {
        s.may(Slot::SmallTalk, 0, scripted(exchange));
    }

    write_statement(&mut s, set, &vars, rng)?;

    // Follow-up questions, small talk, a recap and the close.
    let fitting: Vec<&FollowUp> = pools
        .follow_ups
        .iter()
        .filter(|f| follow_up_fits(f, known, service.kind))
        .collect();
    for (n, follow_up) in pick_distinct(rng, &fitting, 2).into_iter().enumerate() {
        let lines = vec![
            customer(say(rng, &follow_up.customer, &vars, "follow-up question")?),
            agent(say(rng, &follow_up.agent, &vars, "follow-up answer")?),
        ];
        if n == 0 {
            s.must(Slot::FollowUp, 0, lines);
        } else {
            s.may(Slot::FollowUp, n as u8, lines);
        }
    }
    if let Some(exchange) = late {
        s.may(Slot::SmallTalkLate, 0, scripted(exchange));
    }
    s.may(
        Slot::Recap,
        0,
        vec![agent(say(rng, &pools.recap, &vars, "recap")?)],
    );
    s.must(
        Slot::Close,
        0,
        vec![
            agent(say(rng, &frame.anything_else, &vars, "anything else")?),
            customer(say(rng, &frame.nothing_else, &vars, "nothing else")?),
            agent(say(rng, frame.farewell.get(style), &vars, "farewell")?),
            customer(say(rng, &frame.bye, &vars, "bye")?),
        ],
    );

    let target = target_words(style, rng);
    let lines = s.assemble(target, rng);
    Ok(time_turns(lines, rng))
}

/// The wrong statement the agent makes, as said on the phone.
fn wrong_line(
    set: &CallSetting,
    vars: &Vars,
    rng: &mut ChaCha20Rng,
) -> Result<Option<Utterance>, SynthError> {
    let Some(statement) = set.wrong else {
        return Ok(None);
    };
    let st = &set.pools.statements;
    let pool = match statement.claim {
        WrongClaim::CustomerSide(_) => &st.wrong_customer_side,
        WrongClaim::ProviderSide(_) => &st.wrong_provider_side,
        // An agent who has just found nothing on the board can only guess.
        WrongClaim::TimeBack if set.known == 0 => &st.wrong_time_back_unconfirmed,
        WrongClaim::TimeBack => &st.wrong_time_back,
    };
    Ok(Some(Utterance {
        who: Speaker::Agent,
        text: say(rng, pool, vars, "wrong statement")?,
        wrong: true,
    }))
}

/// The agent's statement of the latest state they know (or that they know of
/// nothing), with the customer's replies.
fn write_statement(
    s: &mut Script,
    set: &CallSetting,
    vars: &Vars,
    rng: &mut ChaCha20Rng,
) -> Result<(), SynthError> {
    let st = &set.pools.statements;
    let style = set.agent.style;
    let incident = set.incident;
    let back = say(rng, &set.pools.frame.back, vars, "back")?;
    let wrong = wrong_line(set, vars, rng)?;
    // Different backchannels within one call, so a caller never repeats one.
    let mut backchannels = pick_distinct(rng, &st.backchannel, 4).into_iter();
    let mut bc = || -> Result<Vec<Utterance>, SynthError> {
        let line = backchannels
            .next()
            .ok_or_else(|| SynthError::Pool("need four different backchannels".into()))?;
        Ok(vec![customer(fill(line, vars)?)])
    };
    use Slot::Statement as At;
    match set.known {
        0 => {
            let nothing = say(rng, &st.no_known_issue, vars, "no known issue")?;
            s.must(
                At,
                0,
                vec![
                    agent(format!("{back} {nothing}")),
                    customer(say(rng, &st.reaction_nothing_known, vars, "reaction")?),
                ],
            );
            if let Some(line) = wrong {
                let mut lines = vec![line];
                lines.extend(bc()?);
                s.must(Slot::Diagnosis, 0, lines);
            }
            s.must(
                Slot::Diagnosis,
                1,
                vec![
                    agent(say(rng, &st.raise_fault, vars, "raise fault")?),
                    customer(say(rng, &st.customer_thanks, vars, "customer thanks")?),
                ],
            );
        }
        5 => {
            let resolved = say(rng, st.resolved.get(style), vars, "resolved")?;
            s.must(At, 0, vec![agent(format!("{back} {resolved}"))]);
            s.may(
                At,
                1,
                vec![agent(say(rng, st.cause_was.get(style), vars, "cause was")?)],
            );
            if incident.aftercare.is_some() {
                s.may(
                    At,
                    2,
                    vec![agent(say(rng, st.aftercare.get(style), vars, "aftercare")?)],
                );
            }
            s.must(
                At,
                3,
                vec![customer(say(rng, &st.reaction_resolved, vars, "reaction")?)],
            );
        }
        known => {
            s.must(At, 0, vec![agent(back)]);
            s.may(
                At,
                1,
                vec![agent(say(rng, st.lead_in.get(style), vars, "lead in")?)],
            );
            let problem = vec![agent(say(rng, st.problem.get(style), vars, "problem")?)];
            if known == 1 {
                s.must(At, 2, problem);
            } else {
                s.may(At, 2, problem);
            }
            if known == 1 && incident.next_update.is_some_and(|t| t > set.started_at) {
                s.may(
                    At,
                    3,
                    vec![agent(say(rng, st.next_update.get(style), vars, "next update")?)],
                );
            }
            s.may(At, 4, bc()?);
            let cause = |rng: &mut ChaCha20Rng| -> Result<Vec<Utterance>, SynthError> {
                Ok(vec![agent(say(rng, st.cause.get(style), vars, "cause")?)])
            };
            let waiting = vec![customer(say(rng, &st.reaction_waiting, vars, "reaction")?)];
            match known {
                1 => {
                    let mut lines = vec![agent(say(rng, st.no_cause_yet.get(style), vars, "no cause yet")?)];
                    lines.extend(wrong);
                    s.must(At, 5, lines);
                    s.must(At, 11, waiting);
                }
                2 => {
                    s.must(At, 5, cause(rng)?);
                    s.must(At, 11, waiting);
                }
                3 => {
                    s.may(At, 5, cause(rng)?);
                    s.may(At, 7, bc()?);
                    if incident.workaround.is_some() {
                        s.must(
                            At,
                            8,
                            vec![agent(say(rng, st.workaround.get(style), vars, "workaround")?)],
                        );
                        s.must(
                            At,
                            11,
                            vec![customer(say(rng, &st.reaction_workaround, vars, "reaction")?)],
                        );
                    } else {
                        s.must(
                            At,
                            8,
                            vec![agent(say(
                                rng,
                                st.no_workaround.get(style),
                                vars,
                                "no workaround",
                            )?)],
                        );
                        s.must(At, 11, waiting);
                    }
                }
                _ => {
                    s.may(At, 5, cause(rng)?);
                    let restate = if incident.workaround.is_some() {
                        st.workaround.get(style)
                    } else {
                        st.no_workaround.get(style)
                    };
                    s.may(At, 8, vec![agent(say(rng, restate, vars, "workaround")?)]);
                    s.may(At, 9, bc()?);
                    let fix = if set.started_at >= incident.eta {
                        st.fix_overdue.get(style)
                    } else {
                        st.fix.get(style)
                    };
                    s.must(At, 10, vec![agent(say(rng, fix, vars, "fix")?)]);
                    s.must(At, 11, waiting);
                }
            }
        }
    }
    Ok(())
}

/// Turn end times: 2.5 words a second, with a short pause before each turn.
fn time_turns(lines: Vec<Utterance>, rng: &mut ChaCha20Rng) -> WrittenCall {
    let mut t = 0.0f64;
    let mut turns = Vec::with_capacity(lines.len());
    let mut wrong_said_at_s = None;
    for line in lines {
        let pause = rng.random_range(PAUSE_S.0..=PAUSE_S.1);
        t += pause + word_count(&line.text) as f64 / WORDS_PER_SECOND;
        let ends_at_s = ((t * 100.0).round() / 100.0) as f32;
        if line.wrong && wrong_said_at_s.is_none() {
            wrong_said_at_s = Some(ends_at_s);
        }
        turns.push(Turn {
            who: line.who,
            text: line.text,
            ends_at_s,
        });
    }
    WrittenCall {
        turns,
        wrong_said_at_s,
    }
}
