//! The content pools: typed views of the JSON files under `assets/synth/`,
//! embedded in the binary at build time and checked once when they are loaded.

use super::SynthError;
use super::text::{fold, key_form, placeholders, word_count};
use cb_types::Speaker;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet};

macro_rules! asset {
    ($path:literal) => {
        ($path, include_str!(concat!("../../assets/synth/", $path)))
    };
}

const SERVICE_FILES: [(&str, &str); 12] = [
    asset!("services/home_internet.json"),
    asset!("services/home_wifi.json"),
    asset!("services/mobile_data.json"),
    asset!("services/mobile_calls.json"),
    asset!("services/voicemail.json"),
    asset!("services/tv_app.json"),
    asset!("services/tv_box.json"),
    asset!("services/email.json"),
    asset!("services/web_portal.json"),
    asset!("services/mobile_app.json"),
    asset!("services/online_payments.json"),
    asset!("services/number_porting.json"),
];

const ARTICLE_FILES: [(&str, &str); 20] = [
    asset!("articles/01_billing_a.json"),
    asset!("articles/02_billing_b.json"),
    asset!("articles/03_plans_a.json"),
    asset!("articles/04_plans_moving.json"),
    asset!("articles/05_moving.json"),
    asset!("articles/06_moving_equipment.json"),
    asset!("articles/07_equipment.json"),
    asset!("articles/08_equipment_wifi.json"),
    asset!("articles/09_wifi.json"),
    asset!("articles/10_wifi_account.json"),
    asset!("articles/11_account.json"),
    asset!("articles/12_account_mobile.json"),
    asset!("articles/13_mobile.json"),
    asset!("articles/14_mobile_tv.json"),
    asset!("articles/15_tv.json"),
    asset!("articles/16_tv_email.json"),
    asset!("articles/17_email.json"),
    asset!("articles/18_security.json"),
    asset!("articles/19_security_returns.json"),
    asset!("articles/20_returns.json"),
];

const REGIONS: (&str, &str) = asset!("regions.json");
const PEOPLE: (&str, &str) = asset!("people.json");
const FRAME: (&str, &str) = asset!("dialogue_frame.json");
const STATEMENTS: (&str, &str) = asset!("dialogue_statements.json");
const FOLLOW_UPS: (&str, &str) = asset!("dialogue_followups.json");
const TROUBLESHOOTING: (&str, &str) = asset!("dialogue_troubleshooting.json");
const QUESTIONS: (&str, &str) = asset!("questions.json");
const RECORDS: (&str, &str) = asset!("records.json");
const ASR: (&str, &str) = asset!("asr.json");

/// Counts the specification fixes for the pools.
pub const SERVICES: usize = 12;
pub const REGION_COUNT: usize = 24;
pub const AGENTS: usize = 80;
pub const ARTICLES: usize = 80;
pub const SECTIONS_PER_ARTICLE: usize = 4;
const SECTION_WORDS: std::ops::RangeInclusive<usize> = 60..=120;
const MIN_PHRASINGS_PER_INTENT: usize = 8;

fn parse<T: DeserializeOwned>((file, text): (&'static str, &str)) -> Result<T, SynthError> {
    serde_json::from_str(text).map_err(|source| SynthError::Asset { file, source })
}

fn pool_error(message: String) -> SynthError {
    SynthError::Pool(message)
}

/// How a service reaches the customer. It picks the clarifying questions and
/// the troubleshooting steps an agent can use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceKind {
    Line,
    Mobile,
    Tv,
    Online,
}

/// One of the four ways an agent speaks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Style {
    WordyTechnical,
    WordyEveryday,
    BriefTechnical,
    BriefEveryday,
}

impl Style {
    pub const ALL: [Style; 4] = [
        Style::WordyTechnical,
        Style::WordyEveryday,
        Style::BriefTechnical,
        Style::BriefEveryday,
    ];

    /// The name written into `Transcript.persona`.
    pub fn name(self) -> &'static str {
        match self {
            Style::WordyTechnical => "wordy_technical",
            Style::WordyEveryday => "wordy_everyday",
            Style::BriefTechnical => "brief_technical",
            Style::BriefEveryday => "brief_everyday",
        }
    }

    pub fn is_wordy(self) -> bool {
        matches!(self, Style::WordyTechnical | Style::WordyEveryday)
    }
}

/// One pool per way of speaking.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByStyle<T> {
    pub wordy_technical: T,
    pub wordy_everyday: T,
    pub brief_technical: T,
    pub brief_everyday: T,
}

impl<T> ByStyle<T> {
    pub fn get(&self, style: Style) -> &T {
        match style {
            Style::WordyTechnical => &self.wordy_technical,
            Style::WordyEveryday => &self.wordy_everyday,
            Style::BriefTechnical => &self.brief_technical,
            Style::BriefEveryday => &self.brief_everyday,
        }
    }

    fn each(&self) -> [&T; 4] {
        [
            &self.wordy_technical,
            &self.wordy_everyday,
            &self.brief_technical,
            &self.brief_everyday,
        ]
    }
}

/// One pool per kind of service.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByKind<T> {
    pub line: T,
    pub mobile: T,
    pub tv: T,
    pub online: T,
}

impl<T> ByKind<T> {
    pub fn get(&self, kind: ServiceKind) -> &T {
        match kind {
            ServiceKind::Line => &self.line,
            ServiceKind::Mobile => &self.mobile,
            ServiceKind::Tv => &self.tv,
            ServiceKind::Online => &self.online,
        }
    }

    fn map<U>(self, f: impl Fn(T, &str) -> Result<U, SynthError>) -> Result<ByKind<U>, SynthError> {
        Ok(ByKind {
            line: f(self.line, "line")?,
            mobile: f(self.mobile, "mobile")?,
            tv: f(self.tv, "tv")?,
            online: f(self.online, "online")?,
        })
    }

    fn each(&self) -> [&T; 4] {
        [&self.line, &self.mobile, &self.tv, &self.online]
    }
}

/// A cause of a failure and the fixes that repair it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cause {
    pub cause: String,
    pub fixes: Vec<String>,
}

/// One way a service fails, with everything an incident of this kind can say.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FailureMode {
    pub id: String,
    pub title: String,
    /// The symptom as a status update names it.
    pub symptom: String,
    /// The symptom as a customer says it.
    pub customer_says: Vec<String>,
    pub causes: Vec<Cause>,
    pub workarounds: Vec<String>,
    pub aftercare: Vec<String>,
}

/// One of the twelve services of the fictional provider.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Service {
    pub id: String,
    /// How agents and records name it in running text ("home internet").
    pub name: String,
    /// The label at the head of an incident title ("Home internet").
    pub label: String,
    /// Whether the name takes a plural verb ("TV set-top boxes are").
    pub plural: bool,
    pub kind: ServiceKind,
    /// How customers refer to it ("my broadband").
    pub customer_names: Vec<String>,
    pub modes: Vec<FailureMode>,
    /// Wrong causes on the customer's side (equipment, account, wiring).
    pub wrong_customer_side: Vec<String>,
    /// Wrong causes on the provider's side.
    pub wrong_provider_side: Vec<String>,
}

/// Invented people and places for identity checks.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct People {
    pub agent_names: Vec<String>,
    pub first_names: Vec<String>,
    pub surnames: Vec<String>,
    pub streets: Vec<String>,
    pub email_domains: Vec<String>,
}

/// The parts of a call around the agent's statement.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    pub greeting: ByStyle<Vec<String>>,
    pub probe: ByStyle<Vec<String>>,
    pub opener: Vec<String>,
    pub elapsed_sentence: Vec<String>,
    /// Why the fault matters to the caller, by kind of service.
    pub impact: ByKind<Vec<String>>,
    pub empathy: ByStyle<Vec<String>>,
    pub ask_name: Vec<String>,
    pub give_name: Vec<String>,
    pub ask_account: Vec<String>,
    pub give_account: Vec<String>,
    pub ask_phone: Vec<String>,
    pub give_phone: Vec<String>,
    pub ask_address: Vec<String>,
    pub give_address: Vec<String>,
    pub ask_email: Vec<String>,
    pub give_email: Vec<String>,
    pub ack: Vec<String>,
    pub lookup: ByStyle<Vec<String>>,
    pub hold_ok: Vec<String>,
    pub back: Vec<String>,
    pub anything_else: Vec<String>,
    pub nothing_else: Vec<String>,
    pub farewell: ByStyle<Vec<String>>,
    pub bye: Vec<String>,
}

/// How an agent states what is known, in each way of speaking, and what the
/// customer says back.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statements {
    pub lead_in: ByStyle<Vec<String>>,
    pub problem: ByStyle<Vec<String>>,
    pub no_cause_yet: ByStyle<Vec<String>>,
    pub next_update: ByStyle<Vec<String>>,
    pub cause: ByStyle<Vec<String>>,
    pub workaround: ByStyle<Vec<String>>,
    pub no_workaround: ByStyle<Vec<String>>,
    pub fix: ByStyle<Vec<String>>,
    pub fix_overdue: ByStyle<Vec<String>>,
    pub resolved: ByStyle<Vec<String>>,
    pub cause_was: ByStyle<Vec<String>>,
    pub aftercare: ByStyle<Vec<String>>,
    pub no_known_issue: Vec<String>,
    pub raise_fault: Vec<String>,
    pub wrong_customer_side: Vec<String>,
    pub wrong_provider_side: Vec<String>,
    pub wrong_time_back: Vec<String>,
    /// A wrong time back from an agent who knows of no incident.
    pub wrong_time_back_unconfirmed: Vec<String>,
    /// What a customer says after the agent logs a fault.
    pub customer_thanks: Vec<String>,
    pub backchannel: Vec<String>,
    pub reaction_waiting: Vec<String>,
    pub reaction_workaround: Vec<String>,
    pub reaction_resolved: Vec<String>,
    pub reaction_nothing_known: Vec<String>,
}

/// When a follow-up question fits the call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FollowUpWhen {
    Any,
    /// The agent knows of the incident and it is not over.
    Ongoing,
    /// The agent knows of the incident but has no workaround to give.
    BeforeWorkaround,
    /// The agent knows of the incident.
    AreaKnown,
    /// The agent knows of no incident.
    NothingKnown,
    /// The agent has no time back to give.
    NoEta,
    /// The agent has no cause to give.
    NoCause,
    /// The incident is over.
    Resolved,
}

/// A customer's follow-up question with the answers an agent can give.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FollowUp {
    pub when: FollowUpWhen,
    /// Kinds of service it fits; empty means every kind.
    #[serde(default)]
    pub kinds: Vec<ServiceKind>,
    pub customer: Vec<String>,
    pub agent: Vec<String>,
}

/// One line of a scripted exchange.
#[derive(Clone, Debug)]
pub struct Line {
    pub who: Speaker,
    pub text: String,
}

/// A few lines that belong together (a question and its answer, small talk).
pub type Exchange = Vec<Line>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFollowUps {
    follow_ups: Vec<FollowUp>,
    reassurance: Vec<Vec<String>>,
    recap: Vec<String>,
    small_talk: Vec<Vec<String>>,
    small_talk_waiting: Vec<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTroubleshooting {
    clarify: ByKind<Vec<Vec<String>>>,
    troubleshooting: ByKind<Vec<Vec<String>>>,
}

/// What a question asks about an incident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    IsItDown,
    Cause,
    Workaround,
    WhenBack,
    IsItFixed,
}

impl Intent {
    pub const ALL: [Intent; 5] = [
        Intent::IsItDown,
        Intent::Cause,
        Intent::Workaround,
        Intent::WhenBack,
        Intent::IsItFixed,
    ];
}

/// Phrasings of one intent, naming the region or not.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Phrasings {
    pub with_region: Vec<String>,
    pub without_region: Vec<String>,
}

/// Question phrasings per intent, and for questions after an incident.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QuestionPools {
    pub is_it_down: Phrasings,
    pub cause: Phrasings,
    pub workaround: Phrasings,
    pub when_back: Phrasings,
    pub is_it_fixed: Phrasings,
    pub after: Vec<String>,
}

impl QuestionPools {
    pub fn phrasings(&self, intent: Intent) -> &Phrasings {
        match intent {
            Intent::IsItDown => &self.is_it_down,
            Intent::Cause => &self.cause,
            Intent::Workaround => &self.workaround,
            Intent::WhenBack => &self.when_back,
            Intent::IsItFixed => &self.is_it_fixed,
        }
    }
}

/// Sentences of the written records: status updates and call notes.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordTemplates {
    pub problem: Vec<String>,
    pub investigating: Vec<String>,
    pub next_update: Vec<String>,
    pub cause: Vec<String>,
    pub fix_pending: Vec<String>,
    pub workaround: Vec<String>,
    pub no_workaround: Vec<String>,
    pub fix: Vec<String>,
    pub problem_past: Vec<String>,
    pub cause_was: Vec<String>,
    pub workaround_was: Vec<String>,
    pub resolved: Vec<String>,
    pub aftercare: Vec<String>,
    pub wrong_customer_side: Vec<String>,
    pub wrong_provider_side: Vec<String>,
    pub wrong_time_back: Vec<String>,
}

/// The fixed sound-alike table and fillers of the speech-recognition errors.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AsrTable {
    pub sound_alikes: BTreeMap<String, Vec<String>>,
    pub fillers: Vec<String>,
}

/// What a help article is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Topic {
    Billing,
    Plans,
    MovingHome,
    EquipmentSetup,
    Wifi,
    Account,
    Mobile,
    Tv,
    Email,
    Security,
    Returns,
}

/// One section of a help article, with two ways a customer asks for it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub heading: String,
    pub text: String,
    pub questions: Vec<String>,
}

/// One help article of the standing documentation.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Article {
    pub title: String,
    pub topic: Topic,
    /// The id of the service it is about, when it is about one.
    pub service: Option<String>,
    pub sections: Vec<Section>,
}

/// Every content pool, loaded and checked.
#[derive(Debug)]
pub struct Pools {
    pub services: Vec<Service>,
    pub regions: Vec<String>,
    pub people: People,
    pub frame: Frame,
    pub statements: Statements,
    pub follow_ups: Vec<FollowUp>,
    pub reassurance: Vec<Exchange>,
    pub recap: Vec<String>,
    pub small_talk: Vec<Exchange>,
    /// Small talk that fits only while the agent waits for a screen.
    pub small_talk_waiting: Vec<Exchange>,
    pub clarify: ByKind<Vec<Exchange>>,
    pub troubleshooting: ByKind<Vec<Exchange>>,
    pub questions: QuestionPools,
    pub records: RecordTemplates,
    pub asr: AsrTable,
    pub articles: Vec<Article>,
}

fn parse_exchange(raw: Vec<String>, pool: &str) -> Result<Exchange, SynthError> {
    raw.into_iter()
        .map(|line| {
            if let Some(text) = line.strip_prefix("A: ") {
                Ok(Line {
                    who: Speaker::Agent,
                    text: text.to_string(),
                })
            } else if let Some(text) = line.strip_prefix("C: ") {
                Ok(Line {
                    who: Speaker::Customer,
                    text: text.to_string(),
                })
            } else {
                Err(pool_error(format!(
                    "{pool}: a line must start with \"A: \" or \"C: \": {line:?}"
                )))
            }
        })
        .collect()
}

fn parse_exchanges(raw: Vec<Vec<String>>, pool: &str) -> Result<Vec<Exchange>, SynthError> {
    raw.into_iter().map(|lines| parse_exchange(lines, pool)).collect()
}

impl Pools {
    /// Parse every embedded pool and check it against the specification.
    pub fn load() -> Result<Pools, SynthError> {
        let pools = Pools::parse_all()?;
        pools.check()?;
        Ok(pools)
    }

    /// Parse every embedded pool without checking it.
    pub(crate) fn parse_all() -> Result<Pools, SynthError> {
        let services = SERVICE_FILES
            .into_iter()
            .map(parse)
            .collect::<Result<Vec<Service>, _>>()?;
        let mut articles = Vec::new();
        for file in ARTICLE_FILES {
            let mut part: Vec<Article> = parse(file)?;
            articles.append(&mut part);
        }
        let follow_ups: RawFollowUps = parse(FOLLOW_UPS)?;
        let troubleshooting: RawTroubleshooting = parse(TROUBLESHOOTING)?;
        let pools = Pools {
            services,
            regions: parse(REGIONS)?,
            people: parse(PEOPLE)?,
            frame: parse(FRAME)?,
            statements: parse(STATEMENTS)?,
            follow_ups: follow_ups.follow_ups,
            reassurance: parse_exchanges(follow_ups.reassurance, "reassurance")?,
            recap: follow_ups.recap,
            small_talk: parse_exchanges(follow_ups.small_talk, "small_talk")?,
            small_talk_waiting: parse_exchanges(follow_ups.small_talk_waiting, "small_talk_waiting")?,
            clarify: troubleshooting.clarify.map(parse_exchanges)?,
            troubleshooting: troubleshooting.troubleshooting.map(parse_exchanges)?,
            questions: parse(QUESTIONS)?,
            records: parse(RECORDS)?,
            asr: parse(ASR)?,
            articles,
        };
        Ok(pools)
    }

    /// The index of a service by its id.
    pub fn service_index(&self, id: &str) -> Option<usize> {
        self.services.iter().position(|s| s.id == id)
    }

    fn check(&self) -> Result<(), SynthError> {
        self.check_services()?;
        self.check_people_and_places()?;
        self.check_dialogue()?;
        self.check_questions()?;
        self.check_records()?;
        self.check_articles()?;
        if self.asr.fillers.is_empty() || self.asr.sound_alikes.values().any(Vec::is_empty) {
            return Err(pool_error(
                "the sound-alike table needs fillers and a sound-alike for every entry".into(),
            ));
        }
        Ok(())
    }

    fn check_services(&self) -> Result<(), SynthError> {
        if self.services.len() != SERVICES {
            return Err(pool_error(format!(
                "{} services, expected {SERVICES}",
                self.services.len()
            )));
        }
        let ids: BTreeSet<&str> = self.services.iter().map(|s| s.id.as_str()).collect();
        if ids.len() != SERVICES {
            return Err(pool_error("service ids repeat".into()));
        }
        for service in &self.services {
            let at = |what: &str| format!("service {}: {what}", service.id);
            if service.modes.len() < 4 {
                return Err(pool_error(at("needs at least four failure modes")));
            }
            if service.customer_names.is_empty()
                || service.wrong_customer_side.len() < 2
                || service.wrong_provider_side.len() < 2
            {
                return Err(pool_error(at(
                    "needs customer names and at least two wrong causes per side",
                )));
            }
            for mode in &service.modes {
                if mode.causes.len() < 3 || mode.workarounds.len() < 2 {
                    return Err(pool_error(at(&format!(
                        "mode {} needs at least three causes and two workarounds",
                        mode.id
                    ))));
                }
                if mode.customer_says.len() < 2 || mode.aftercare.is_empty() {
                    return Err(pool_error(at(&format!(
                        "mode {} needs customer phrasings and aftercare",
                        mode.id
                    ))));
                }
                if mode.causes.iter().any(|c| c.fixes.is_empty()) {
                    return Err(pool_error(at(&format!(
                        "mode {}: every cause needs a fix",
                        mode.id
                    ))));
                }
            }
            check_wrong_never_overlaps_true(service)?;
        }
        Ok(())
    }

    fn check_people_and_places(&self) -> Result<(), SynthError> {
        let regions: BTreeSet<&String> = self.regions.iter().collect();
        if self.regions.len() != REGION_COUNT || regions.len() != REGION_COUNT {
            return Err(pool_error(format!("need {REGION_COUNT} distinct regions")));
        }
        let agents: BTreeSet<&String> = self.people.agent_names.iter().collect();
        if self.people.agent_names.len() != AGENTS || agents.len() != AGENTS {
            return Err(pool_error(format!("need {AGENTS} distinct agent names")));
        }
        let p = &self.people;
        if p.first_names.len() < 20
            || p.surnames.len() < 20
            || p.streets.len() < 10
            || p.email_domains.is_empty()
        {
            return Err(pool_error(
                "too few invented names, streets or email domains".into(),
            ));
        }
        Ok(())
    }

    fn check_dialogue(&self) -> Result<(), SynthError> {
        let f = &self.frame;
        let s = &self.statements;
        let styled: [PoolRule<ByStyle<Vec<String>>>; 16] = [
            ("greeting", &f.greeting, &["agent"], &[]),
            ("probe", &f.probe, &[], &[]),
            ("empathy", &f.empathy, &[], &[]),
            ("lookup", &f.lookup, &["service", "Service", "region"], &[]),
            ("farewell", &f.farewell, &["first"], &[]),
            ("lead_in", &s.lead_in, &[], &[]),
            (
                "problem",
                &s.problem,
                &["service", "Service", "region", "symptom", "since", "since_day"],
                &["symptom", "since"],
            ),
            ("no_cause_yet", &s.no_cause_yet, &[], &[]),
            (
                "next_update",
                &s.next_update,
                &["next_update", "next_update_day"],
                &["next_update"],
            ),
            ("cause", &s.cause, &["cause"], &["cause"]),
            ("workaround", &s.workaround, &["workaround"], &["workaround"]),
            ("no_workaround", &s.no_workaround, &[], &[]),
            ("fix", &s.fix, &["fix", "eta", "eta_day"], &["fix", "eta"]),
            (
                "fix_overdue",
                &s.fix_overdue,
                &["fix", "eta", "eta_day"],
                &["fix", "eta"],
            ),
            (
                "resolved",
                &s.resolved,
                &["service", "Service", "region", "resolved_at", "resolved_day"],
                &["resolved_at"],
            ),
            ("cause_was", &s.cause_was, &["cause"], &["cause"]),
        ];
        for (name, pool, allowed, required) in styled {
            for templates in pool.each() {
                check_templates(name, templates, allowed, required)?;
            }
        }
        for templates in s.aftercare.each() {
            check_templates("aftercare", templates, &["aftercare"], &["aftercare"])?;
        }
        for templates in f.impact.each() {
            check_templates("impact", templates, &[], &[])?;
        }
        for templates in s.no_workaround.each() {
            if templates.iter().any(|t| !fold(t).contains("no workaround")) {
                return Err(pool_error(
                    "every no_workaround line must say \"no workaround\"".into(),
                ));
            }
        }
        let flat: [PoolRule<Vec<String>>; 31] = [
            ("opener", &f.opener, &["my_service", "agent"], &["my_service"]),
            (
                "elapsed_sentence",
                &f.elapsed_sentence,
                &["elapsed"],
                &["elapsed"],
            ),
            ("ask_name", &f.ask_name, &[], &[]),
            ("give_name", &f.give_name, &["name", "first"], &["name"]),
            ("ask_account", &f.ask_account, &[], &[]),
            ("give_account", &f.give_account, &["account"], &["account"]),
            ("ask_phone", &f.ask_phone, &[], &[]),
            ("give_phone", &f.give_phone, &["phone"], &["phone"]),
            ("ask_address", &f.ask_address, &[], &[]),
            ("give_address", &f.give_address, &["address"], &["address"]),
            ("ask_email", &f.ask_email, &[], &[]),
            ("give_email", &f.give_email, &["email"], &["email"]),
            ("ack", &f.ack, &["first"], &[]),
            ("hold_ok", &f.hold_ok, &[], &[]),
            ("back", &f.back, &[], &[]),
            ("anything_else", &f.anything_else, &[], &[]),
            ("nothing_else", &f.nothing_else, &[], &[]),
            ("bye", &f.bye, &[], &[]),
            (
                "no_known_issue",
                &s.no_known_issue,
                &["service", "Service", "region"],
                &[],
            ),
            ("raise_fault", &s.raise_fault, &[], &[]),
            (
                "wrong_customer_side",
                &s.wrong_customer_side,
                &["wrong"],
                &["wrong"],
            ),
            (
                "wrong_provider_side",
                &s.wrong_provider_side,
                &["wrong"],
                &["wrong"],
            ),
            (
                "wrong_time_back",
                &s.wrong_time_back,
                &["wrong_time", "wrong_day"],
                &["wrong_time"],
            ),
            (
                "wrong_time_back_unconfirmed",
                &s.wrong_time_back_unconfirmed,
                &["wrong_time", "wrong_day"],
                &["wrong_time"],
            ),
            ("customer_thanks", &s.customer_thanks, &[], &[]),
            ("backchannel", &s.backchannel, &[], &[]),
            ("reaction_waiting", &s.reaction_waiting, &[], &[]),
            ("reaction_workaround", &s.reaction_workaround, &[], &[]),
            ("reaction_resolved", &s.reaction_resolved, &[], &[]),
            ("reaction_nothing_known", &s.reaction_nothing_known, &[], &[]),
            ("recap", &self.recap, &[], &[]),
        ];
        for (name, templates, allowed, required) in flat {
            check_templates(name, templates, allowed, required)?;
        }
        for follow_up in &self.follow_ups {
            check_templates("follow-up question", &follow_up.customer, &[], &[])?;
            check_templates(
                "follow-up answer",
                &follow_up.agent,
                &["service", "Service", "region"],
                &[],
            )?;
        }
        if self
            .follow_ups
            .iter()
            .filter(|f| f.when == FollowUpWhen::Any && f.kinds.is_empty())
            .count()
            < 2
        {
            return Err(pool_error(
                "need at least two follow-ups that fit every call".into(),
            ));
        }
        let exchanges = self
            .reassurance
            .iter()
            .chain(&self.small_talk)
            .chain(&self.small_talk_waiting)
            .chain(self.clarify.each().into_iter().flatten())
            .chain(self.troubleshooting.each().into_iter().flatten());
        for exchange in exchanges {
            if exchange.is_empty() || exchange.iter().any(|line| !placeholders(&line.text).is_empty()) {
                return Err(pool_error(
                    "scripted exchanges must be non-empty and take no placeholders".into(),
                ));
            }
        }
        if self.small_talk.len() < 2
            || self.clarify.each().iter().any(|pool| pool.len() < 3)
            || self.troubleshooting.each().iter().any(|pool| pool.len() < 2)
        {
            return Err(pool_error(
                "too few small-talk, clarifying or troubleshooting exchanges".into(),
            ));
        }
        Ok(())
    }

    fn check_questions(&self) -> Result<(), SynthError> {
        let allowed = ["service", "Service", "region", "is", "Is", "has", "Has", "was"];
        for intent in Intent::ALL {
            let p = self.questions.phrasings(intent);
            if p.with_region.len() < MIN_PHRASINGS_PER_INTENT
                || p.without_region.len() < MIN_PHRASINGS_PER_INTENT
            {
                return Err(pool_error(format!(
                    "{intent:?}: need {MIN_PHRASINGS_PER_INTENT} phrasings each way"
                )));
            }
            check_templates("question", &p.with_region, &allowed, &["region"])?;
            check_templates("question", &p.without_region, &allowed, &[])?;
            if p.without_region
                .iter()
                .any(|t| placeholders(t).contains(&"region"))
            {
                return Err(pool_error("a question without region names one".into()));
            }
        }
        if self.questions.after.len() < MIN_PHRASINGS_PER_INTENT {
            return Err(pool_error(
                "too few phrasings for questions after an incident".into(),
            ));
        }
        check_templates("after", &self.questions.after, &allowed, &["region"])
    }

    fn check_records(&self) -> Result<(), SynthError> {
        let r = &self.records;
        let place = ["region", "service", "Service"];
        let problem = ["region", "service", "Service", "symptom", "since", "since_date"];
        let pools: [PoolRule<Vec<String>>; 16] = [
            (
                "problem",
                &r.problem,
                &problem,
                &["symptom", "since", "since_date", "region"],
            ),
            ("investigating", &r.investigating, &[], &[]),
            (
                "next_update",
                &r.next_update,
                &["next_update", "next_update_date"],
                &["next_update"],
            ),
            ("cause", &r.cause, &["cause"], &["cause"]),
            ("fix_pending", &r.fix_pending, &[], &[]),
            ("workaround", &r.workaround, &["workaround"], &["workaround"]),
            ("no_workaround", &r.no_workaround, &[], &[]),
            ("fix", &r.fix, &["fix", "eta", "eta_date"], &["fix", "eta"]),
            (
                "problem_past",
                &r.problem_past,
                &problem,
                &["symptom", "since", "since_date", "region"],
            ),
            ("cause_was", &r.cause_was, &["cause"], &["cause"]),
            (
                "workaround_was",
                &r.workaround_was,
                &["workaround"],
                &["workaround"],
            ),
            (
                "resolved",
                &r.resolved,
                &["fix", "resolved_at", "resolved_date"],
                &["fix", "resolved_at"],
            ),
            ("aftercare", &r.aftercare, &["aftercare"], &["aftercare"]),
            (
                "wrong_customer_side",
                &r.wrong_customer_side,
                &[&place[..], &["wrong", "call_date"]].concat(),
                &["wrong", "call_date", "service", "region"],
            ),
            (
                "wrong_provider_side",
                &r.wrong_provider_side,
                &[&place[..], &["wrong", "call_date"]].concat(),
                &["wrong", "call_date", "service", "region"],
            ),
            (
                "wrong_time_back",
                &r.wrong_time_back,
                &[&place[..], &["wrong_time", "wrong_date", "call_date"]].concat(),
                &["wrong_time", "call_date", "service", "region"],
            ),
        ];
        for (name, templates, allowed, required) in pools {
            check_templates(name, templates, allowed, required)?;
        }
        if r.no_workaround.iter().any(|t| !fold(t).contains("no workaround")) {
            return Err(pool_error(
                "every written no_workaround sentence must say \"no workaround\"".into(),
            ));
        }
        Ok(())
    }

    fn check_articles(&self) -> Result<(), SynthError> {
        if self.articles.len() != ARTICLES {
            return Err(pool_error(format!(
                "{} help articles, expected {ARTICLES}",
                self.articles.len()
            )));
        }
        let mut titles = BTreeSet::new();
        let mut texts = BTreeSet::new();
        let mut questions = BTreeSet::new();
        for article in &self.articles {
            let at = |what: String| pool_error(format!("article {:?}: {what}", article.title));
            if !titles.insert(article.title.as_str()) {
                return Err(at("title repeats".into()));
            }
            if let Some(service) = &article.service
                && self.service_index(service).is_none()
            {
                return Err(at(format!("unknown service {service:?}")));
            }
            if article.sections.len() != SECTIONS_PER_ARTICLE {
                return Err(at(format!(
                    "{} sections, expected {SECTIONS_PER_ARTICLE}",
                    article.sections.len()
                )));
            }
            for section in &article.sections {
                let words = word_count(&section.text);
                if !SECTION_WORDS.contains(&words) {
                    return Err(at(format!("section {:?} has {words} words", section.heading)));
                }
                if section.questions.len() != 2 || !texts.insert(section.text.as_str()) {
                    return Err(at(format!(
                        "section {:?}: two questions and a unique text",
                        section.heading
                    )));
                }
                for q in &section.questions {
                    if !questions.insert(q.as_str()) || !placeholders(q).is_empty() {
                        return Err(at(format!("question {q:?} repeats or has a placeholder")));
                    }
                }
            }
        }
        Ok(())
    }
}

/// A pool to check: its name, its templates, the placeholders it may use and
/// those it must use.
type PoolRule<'a, T> = (&'a str, &'a T, &'a [&'a str], &'a [&'a str]);

/// Every template of a pool uses only `allowed` placeholders and all of `required`.
fn check_templates(
    pool: &str,
    templates: &[String],
    allowed: &[&str],
    required: &[&str],
) -> Result<(), SynthError> {
    if templates.is_empty() {
        return Err(pool_error(format!("pool {pool} is empty")));
    }
    for template in templates {
        let used = placeholders(template);
        if let Some(bad) = used.iter().find(|name| !allowed.contains(name)) {
            return Err(pool_error(format!(
                "pool {pool}: {{{bad}}} is not allowed in {template:?}"
            )));
        }
        if let Some(missing) = required
            .iter()
            .find(|name| !used.contains(name) && !(**name == "service" && used.contains(&"Service")))
        {
            return Err(pool_error(format!(
                "pool {pool}: {{{missing}}} is missing from {template:?}"
            )));
        }
    }
    Ok(())
}

/// A wrong statement must never use a key term of a true one, and the other way round.
fn check_wrong_never_overlaps_true(service: &Service) -> Result<(), SynthError> {
    let mut true_phrases: Vec<String> = Vec::new();
    for mode in &service.modes {
        true_phrases.push(fold(&key_form(&mode.symptom)));
        true_phrases.extend(mode.workarounds.iter().map(|w| fold(&key_form(w))));
        true_phrases.extend(mode.aftercare.iter().map(|a| fold(a)));
        for cause in &mode.causes {
            true_phrases.push(fold(&key_form(&cause.cause)));
            true_phrases.extend(cause.fixes.iter().map(|f| fold(&key_form(f))));
        }
    }
    let wrong = service
        .wrong_customer_side
        .iter()
        .chain(&service.wrong_provider_side);
    for phrase in wrong {
        let w = fold(&key_form(phrase));
        if let Some(t) = true_phrases
            .iter()
            .find(|t| t.contains(&w) || w.contains(t.as_str()))
        {
            return Err(pool_error(format!(
                "service {}: wrong cause {w:?} overlaps the true phrase {t:?}",
                service.id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pools_load_and_meet_the_specification() {
        let pools = Pools::load().unwrap();
        assert_eq!(pools.services.len(), SERVICES);
        assert!(pools.services.iter().all(|s| s.modes.len() >= 4));
        assert_eq!(pools.articles.len(), ARTICLES);
        assert_eq!(pools.regions.len(), REGION_COUNT);
        assert_eq!(pools.people.agent_names.len(), AGENTS);
    }

    #[test]
    fn a_template_check_names_a_bad_placeholder() {
        let templates = vec!["Is {service} down in {area}?".to_string()];
        assert!(check_templates("q", &templates, &["service", "region"], &[]).is_err());
        let templates = vec!["Is {Service} down?".to_string()];
        assert!(check_templates("q", &templates, &["service", "Service"], &["service"]).is_ok());
    }
}
