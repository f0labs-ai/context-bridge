//! Incidents: what fails, where, when, and the five moments at which the
//! support floor learns more (docs/SYNTH_SPEC.md section 2).

use super::SynthError;
use super::clock::{self, DAY, HOUR, MINUTE, WINDOW_DAYS, WINDOW_START};
use super::draw::{between, chance, exact_flags, index};
use super::pools::Pools;
use super::spec;
use rand::Rng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;

/// Where each of the states 1–4 falls, as a share of the incident's length.
const STATE_SHARES: [(f64, f64); 4] = [(0.02, 0.10), (0.15, 0.35), (0.30, 0.50), (0.45, 0.70)];
/// Two incidents on the same service and region keep at least this far apart,
/// so a question naming the service and region, asked up to 48 hours after one
/// of them ends, is about one incident only.
pub const SAME_PLACE_GAP: i64 = 49 * HOUR;
/// The planted pairs' asking window: at most this long, at least this short.
const ASK_WINDOW_MAX: i64 = 2 * HOUR;
const ASK_WINDOW_MIN: i64 = 2 * MINUTE;
/// Share of incidents whose resolved state gives aftercare advice.
const AFTERCARE_SHARE: f64 = 0.7;
/// "Next update by" offsets after the first state, in minutes.
const NEXT_UPDATE_MINUTES: [i64; 7] = [15, 20, 30, 45, 60, 90, 120];
const MAX_TRIES: usize = 200_000;

/// The three lengths of incident the supplement names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LengthClass {
    /// 10 to 59 minutes.
    Short,
    /// 1 to 8 hours.
    Medium,
    /// 8 to 72 hours.
    Long,
}

impl LengthClass {
    /// Shortest and longest length, in seconds, both included.
    pub fn bounds(self) -> (i64, i64) {
        match self {
            LengthClass::Short => (10 * MINUTE, 59 * MINUTE),
            LengthClass::Medium => (HOUR, 8 * HOUR - 1),
            LengthClass::Long => (8 * HOUR, 72 * HOUR),
        }
    }

    /// Weight of the class when the calls beyond six per incident are shared out.
    pub fn call_weight(self) -> u64 {
        match self {
            LengthClass::Short => 1,
            LengthClass::Medium => 2,
            LengthClass::Long => 3,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            LengthClass::Short => "short",
            LengthClass::Medium => "medium",
            LengthClass::Long => "long",
        }
    }
}

/// What a planted wrong statement claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrongClaim {
    /// A cause on the customer's side; the index is into the service's pool.
    CustomerSide(usize),
    /// A cause on the provider's side; the index is into the service's pool.
    ProviderSide(usize),
    /// A time the service will be back that is not the true one.
    TimeBack,
}

/// One planted wrong statement of an incident.
#[derive(Clone, Debug)]
pub struct WrongStatement {
    pub claim: WrongClaim,
    /// For a wrong time back, the moment it names (set once the calls are known).
    pub time_back: Option<i64>,
}

/// One service failing in one region in one way.
#[derive(Clone, Debug)]
pub struct Incident {
    /// `inc-0001`, numbered in order of start.
    pub id: String,
    pub service: usize,
    pub mode: usize,
    pub region: usize,
    pub class: LengthClass,
    pub start: i64,
    pub length: i64,
    /// When each of the five states became known: `written_at` of `.s1` … `.s5`.
    pub moments: [i64; 5],
    pub cause: usize,
    pub fix: usize,
    /// The workaround, or `None` when nothing helps.
    pub workaround: Option<usize>,
    /// Aftercare advice given when it is resolved, if any.
    pub aftercare: Option<usize>,
    /// The "next update by" time of the first state, for half of the incidents.
    pub next_update: Option<i64>,
    /// The expected time back given by the fourth state.
    pub eta: i64,
    pub wrong: Vec<WrongStatement>,
    /// The planted pair it belongs to, if any.
    pub pair: Option<usize>,
}

impl Incident {
    pub fn end(&self) -> i64 {
        self.start + self.length
    }

    /// The id of state `k` (1–5): `inc-0001.s3`.
    pub fn state_id(&self, k: usize) -> String {
        format!("{}.s{k}", self.id)
    }

    /// The latest state written at or before `t` (1–5), or 0 before the first.
    pub fn latest_state_at(&self, t: i64) -> usize {
        self.moments.iter().filter(|m| **m <= t).count()
    }
}

/// Two incidents of one service in two regions at the same time, and the
/// window in which their ambiguous question is asked. In that window no other
/// incident of the service is in progress.
#[derive(Clone, Debug)]
pub struct PlantedPair {
    pub a: usize,
    pub b: usize,
    pub ask_from: i64,
    pub ask_until: i64,
}

pub struct IncidentPlan {
    pub incidents: Vec<Incident>,
    pub pairs: Vec<PlantedPair>,
}

#[derive(Clone, Copy)]
struct Timeline {
    start: i64,
    length: i64,
    moments: [i64; 5],
}

impl Timeline {
    fn end(&self) -> i64 {
        self.start + self.length
    }
}

#[derive(Clone, Copy)]
struct Draft {
    service: usize,
    mode: usize,
    region: usize,
    class: LengthClass,
    timeline: Timeline,
    pair: Option<usize>,
}

/// The minute of the day a moment shows on a clock.
pub fn minute_of_day(t: i64) -> i64 {
    t.div_euclid(MINUTE).rem_euclid(DAY / MINUTE)
}

/// Whether two moments show the same clock time ("09:40").
pub fn same_clock(a: i64, b: i64) -> bool {
    minute_of_day(a) == minute_of_day(b)
}

fn window_end() -> i64 {
    WINDOW_START + WINDOW_DAYS * DAY
}

/// A length and the five state moments for an incident starting at `start`,
/// or `None` when the draw is rejected: states closer than a minimum gap, or a
/// resolution that shows the same clock time as the start.
fn draw_timeline(class: LengthClass, start: i64, rng: &mut ChaCha20Rng) -> Option<Timeline> {
    let (lo, hi) = class.bounds();
    let length = between(rng, lo, hi);
    let mut shares = STATE_SHARES.map(|(a, b)| rng.random_range(a..=b));
    // Sorting keeps each order statistic inside its own state's range,
    // because the ranges' lower and upper ends both increase.
    shares.sort_by(f64::total_cmp);
    let mut moments = [start + length; 5];
    for (k, share) in shares.iter().enumerate() {
        moments[k] = start + (share * length as f64).round() as i64;
    }
    let min_gap = (length / 50).max(30);
    if moments[0] <= start || moments.windows(2).any(|w| w[1] - w[0] < min_gap) {
        return None;
    }
    if same_clock(start, start + length) {
        return None;
    }
    Some(Timeline {
        start,
        length,
        moments,
    })
}

fn too_close_to_same_place(drafts: &[Draft], service: usize, region: usize, t: &Timeline) -> bool {
    drafts.iter().any(|d| {
        d.service == service
            && d.region == region
            && t.start < d.timeline.end() + SAME_PLACE_GAP
            && d.timeline.start < t.end() + SAME_PLACE_GAP
    })
}

fn overlaps_reserved(reserved: &[(usize, i64, i64)], service: usize, t: &Timeline) -> bool {
    reserved
        .iter()
        .any(|(s, from, until)| *s == service && t.start < *until && *from < t.end())
}

fn service_in_progress(drafts: &[Draft], service: usize, from: i64, until: i64) -> bool {
    drafts
        .iter()
        .any(|d| d.service == service && d.timeline.start < until && from < d.timeline.end())
}

/// Place one planted pair: one service, two regions, overlapping in time.
fn place_pair(
    pools: &Pools,
    classes: (LengthClass, LengthClass),
    pair: usize,
    drafts: &mut Vec<Draft>,
    reserved: &mut Vec<(usize, i64, i64)>,
    rng: &mut ChaCha20Rng,
) -> Result<(usize, usize, i64, i64), SynthError> {
    for _ in 0..MAX_TRIES {
        let service = index(rng, pools.services.len());
        let modes = pools.services[service].modes.len();
        let (region_a, region_b) = (index(rng, pools.regions.len()), index(rng, pools.regions.len()));
        let (mode_a, mode_b) = (index(rng, modes), index(rng, modes));
        let start_a = between(rng, WINDOW_START, window_end() - 1);
        let Some(a) = draw_timeline(classes.0, start_a, rng) else {
            continue;
        };
        let start_b = start_a + between(rng, -(a.length / 4), a.length / 2);
        let Some(b) = draw_timeline(classes.1, start_b, rng) else {
            continue;
        };
        if region_a == region_b || !(WINDOW_START..window_end()).contains(&start_b) {
            continue;
        }
        let from = a.moments[0].max(b.moments[0]);
        let until = a.end().min(b.end());
        if until - from < ASK_WINDOW_MIN {
            continue;
        }
        let span = (until - from).min(ASK_WINDOW_MAX);
        let ask_from = between(rng, from, until - span);
        let ask_until = ask_from + span;
        if too_close_to_same_place(drafts, service, region_a, &a)
            || too_close_to_same_place(drafts, service, region_b, &b)
            || overlaps_reserved(reserved, service, &a)
            || overlaps_reserved(reserved, service, &b)
            || service_in_progress(drafts, service, ask_from, ask_until)
        {
            continue;
        }
        let draft = |mode, region, class, timeline| Draft {
            service,
            mode,
            region,
            class,
            timeline,
            pair: Some(pair),
        };
        drafts.push(draft(mode_a, region_a, classes.0, a));
        drafts.push(draft(mode_b, region_b, classes.1, b));
        reserved.push((service, ask_from, ask_until));
        return Ok((drafts.len() - 2, drafts.len() - 1, ask_from, ask_until));
    }
    Err(SynthError::Plan(format!("could not place planted pair {pair}")))
}

/// Place one incident that belongs to no pair.
fn place_single(
    pools: &Pools,
    class: LengthClass,
    drafts: &mut Vec<Draft>,
    reserved: &[(usize, i64, i64)],
    rng: &mut ChaCha20Rng,
) -> Result<(), SynthError> {
    for _ in 0..MAX_TRIES {
        let service = index(rng, pools.services.len());
        let mode = index(rng, pools.services[service].modes.len());
        let region = index(rng, pools.regions.len());
        let start = between(rng, WINDOW_START, window_end() - 1);
        let Some(timeline) = draw_timeline(class, start, rng) else {
            continue;
        };
        if too_close_to_same_place(drafts, service, region, &timeline)
            || overlaps_reserved(reserved, service, &timeline)
        {
            continue;
        }
        drafts.push(Draft {
            service,
            mode,
            region,
            class,
            timeline,
            pair: None,
        });
        return Ok(());
    }
    Err(SynthError::Plan("could not place an incident".into()))
}

/// The moment `base` moved forward in five-minute steps until it shows a
/// clock time none of `taken` shows.
fn distinct_clock(base: i64, taken: &[i64]) -> Result<i64, SynthError> {
    let mut t = clock::ceil_to(base, 5 * MINUTE);
    for _ in 0..(DAY / (5 * MINUTE)) {
        if !taken.iter().any(|x| same_clock(*x, t)) {
            return Ok(t);
        }
        t += 5 * MINUTE;
    }
    Err(SynthError::Plan("no free clock time".into()))
}

/// Plan all 500 incidents: lengths, places, times, the planted pairs, and what
/// each incident's records will say.
pub fn plan_incidents(pools: &Pools, rng: &mut ChaCha20Rng) -> Result<IncidentPlan, SynthError> {
    let mut classes: Vec<LengthClass> = std::iter::repeat_n(LengthClass::Short, spec::SHORT)
        .chain(std::iter::repeat_n(LengthClass::Medium, spec::MEDIUM))
        .chain(std::iter::repeat_n(LengthClass::Long, spec::LONG))
        .collect();
    classes.shuffle(rng);

    let mut drafts: Vec<Draft> = Vec::with_capacity(spec::INCIDENTS);
    let mut reserved: Vec<(usize, i64, i64)> = Vec::new();
    let mut raw_pairs = Vec::with_capacity(spec::PLANTED_PAIRS);
    for pair in 0..spec::PLANTED_PAIRS {
        let these = (classes[2 * pair], classes[2 * pair + 1]);
        raw_pairs.push(place_pair(pools, these, pair, &mut drafts, &mut reserved, rng)?);
    }
    for class in classes.iter().skip(2 * spec::PLANTED_PAIRS) {
        place_single(pools, *class, &mut drafts, &reserved, rng)?;
    }

    // Number incidents in order of start.
    let mut order: Vec<usize> = (0..drafts.len()).collect();
    order.sort_by_key(|i| (drafts[*i].timeline.start, *i));
    let mut new_index = vec![0usize; drafts.len()];
    for (new, old) in order.iter().enumerate() {
        new_index[*old] = new;
    }

    let three_wrong = exact_flags(rng, spec::INCIDENTS, spec::INCIDENTS_WITH_THREE_WRONG);
    let next_update = exact_flags(rng, spec::INCIDENTS, spec::INCIDENTS_WITH_NEXT_UPDATE);
    let no_workaround = exact_flags(rng, spec::INCIDENTS, spec::INCIDENTS_WITHOUT_WORKAROUND);
    let mut incidents = Vec::with_capacity(drafts.len());
    for (n, old) in order.iter().enumerate() {
        let d = drafts[*old];
        incidents.push(fill_in_incident(
            pools,
            n,
            &d,
            (three_wrong[n], next_update[n], no_workaround[n]),
            rng,
        )?);
    }
    let pairs = raw_pairs
        .into_iter()
        .map(|(a, b, ask_from, ask_until)| PlantedPair {
            a: new_index[a],
            b: new_index[b],
            ask_from,
            ask_until,
        })
        .collect();
    Ok(IncidentPlan { incidents, pairs })
}

/// Everything an incident's records say beyond its place and times.
fn fill_in_incident(
    pools: &Pools,
    n: usize,
    d: &Draft,
    (three_wrong, gives_next_update, no_workaround): (bool, bool, bool),
    rng: &mut ChaCha20Rng,
) -> Result<Incident, SynthError> {
    let service = &pools.services[d.service];
    let mode = &service.modes[d.mode];
    let t = d.timeline;
    let cause = index(rng, mode.causes.len());
    let fix = index(rng, mode.causes[cause].fixes.len());
    let workaround = (!no_workaround).then(|| index(rng, mode.workarounds.len()));
    let aftercare = chance(rng, AFTERCARE_SHARE).then(|| index(rng, mode.aftercare.len()));

    let mut taken = vec![t.start, t.end()];
    let next_update = if gives_next_update {
        let minutes = NEXT_UPDATE_MINUTES[index(rng, NEXT_UPDATE_MINUTES.len())];
        let at = distinct_clock(t.moments[0] + minutes * MINUTE, &taken)?;
        taken.push(at);
        Some(at)
    } else {
        None
    };
    // The expected time back lies around the true end (15% before to 25% after
    // it) and always after the fourth state is written.
    let share: f64 = rng.random_range(-0.15..0.25);
    let around_end = t.end() + (share * t.length as f64).round() as i64;
    let eta = distinct_clock(around_end.max(t.moments[3] + 5 * MINUTE), &taken)?;

    let mut kinds = [0usize, 1, 2];
    kinds.shuffle(rng);
    let count = if three_wrong { 3 } else { 2 };
    let mut chosen: Vec<usize> = kinds[..count].to_vec();
    chosen.sort_unstable();
    let wrong = chosen
        .into_iter()
        .map(|kind| WrongStatement {
            claim: match kind {
                0 => WrongClaim::CustomerSide(index(rng, service.wrong_customer_side.len())),
                1 => WrongClaim::ProviderSide(index(rng, service.wrong_provider_side.len())),
                _ => WrongClaim::TimeBack,
            },
            time_back: None,
        })
        .collect();

    Ok(Incident {
        id: format!("inc-{:04}", n + 1),
        service: d.service,
        mode: d.mode,
        region: d.region,
        class: d.class,
        start: t.start,
        length: t.length,
        moments: t.moments,
        cause,
        fix,
        workaround,
        aftercare,
        next_update,
        eta,
        wrong,
        pair: d.pair,
    })
}

/// A wrong time back for the calls that state it: 15 minutes to 2 hours after
/// the last of them starts, on a five-minute mark, showing a clock time no true
/// record of the incident shows, and at least 20 minutes from the true end and
/// from the expected time back.
pub fn wrong_time_back(
    incident: &Incident,
    last_call: i64,
    rng: &mut ChaCha20Rng,
) -> Result<i64, SynthError> {
    let mut taken = vec![incident.start, incident.end(), incident.eta];
    taken.extend(incident.next_update);
    let mut t = clock::ceil_to(last_call + between(rng, 15, 120) * MINUTE, 5 * MINUTE);
    for _ in 0..2_000 {
        let clash = taken.iter().any(|x| same_clock(*x, t));
        let near = (t - incident.end()).abs() < 20 * MINUTE || (t - incident.eta).abs() < 20 * MINUTE;
        if !clash && !near {
            return Ok(t);
        }
        t += 5 * MINUTE;
    }
    Err(SynthError::Plan(format!(
        "{}: no free wrong time back",
        incident.id
    )))
}
