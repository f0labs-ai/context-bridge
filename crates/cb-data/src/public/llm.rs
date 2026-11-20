//! LLM service outages (docs/PUBLIC_DATA.md section 3), Zenodo
//! 10.5281/zenodo.14018219.
//!
//! The archive holds the status-page history of OpenAI, Anthropic and
//! Character.AI up to 2024-08-31 as CSV files, one row per incident, with the
//! incident's updates as a JSON list in one column, newest first. An update's
//! time is what the page showed the compiler: US Pacific time, to the minute
//! ("Posted 3 days ago. Aug 28, 2024 - 13:36 PDT").

use super::Converted;
use super::csv::{CsvError, Table};
use super::text;
use super::time::{self, TimeError};
use super::timeline::{
    self, Incident, InvalidDataSet, Life, Rules, SourceSpec, Stats, TimelineError, Update,
};
use super::zenodo::{Archive, ZenodoError, ZenodoFile};
use cb_types::Phase;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

pub const ARCHIVE: ZenodoFile = ZenodoFile {
    doi: "10.5281/zenodo.14018219",
    page: "https://zenodo.org/records/14018219",
    title: "LLM Service Outages and Incident Reports",
    author: "Chu, Xiaoyu",
    published: "2024-10-31",
    url: "https://zenodo.org/api/records/14018219/files/LLM%20Service%20Outages%20and%20Incident%20Reports.zip/content",
    file_name: "llm-service-outages.zip",
    bytes: 250_498,
    sha256: "ee4c6856477e26344b5fc68c9e55012caf2b0c65a366235493547b9fa3648747",
};

pub const SPEC: SourceSpec = SourceSpec {
    name: "llm",
    record_source: "llm-status",
    id_prefix: "llm",
    // Timed to the minute.
    time_precision_s: 60,
    // Questions name the component whenever the incident names one.
    service_question_share: 0.0,
};

/// The folder of the archive that holds the incident files, one sub-folder per provider.
const INCIDENT_DIR: &str = "LLM Service Outages and Incident Reports/raw_data/incident/2024-08-31/";

#[derive(Debug, Error)]
pub enum LlmError {
    #[error(transparent)]
    Archive(#[from] ZenodoError),
    #[error("{file}: {source}")]
    Csv {
        file: String,
        #[source]
        source: CsvError,
    },
    #[error("{file}: not UTF-8")]
    NotText { file: String },
    #[error("{file}, {link}: the updates column is not the expected JSON list: {source}")]
    Updates {
        file: String,
        link: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("{file}: {link:?} is not a link to one incident of a status page")]
    Link { file: String, link: String },
    #[error("{link}: {source}")]
    Time {
        link: String,
        #[source]
        source: TimeError,
    },
    #[error("{0}: a provider folder this loader does not know")]
    Provider(String),
    #[error("the archive has no incident files under {INCIDENT_DIR:?}")]
    Empty,
    #[error(transparent)]
    Timeline(#[from] TimelineError),
    #[error(transparent)]
    Invalid(#[from] InvalidDataSet),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Provider {
    OpenAi,
    Anthropic,
    CharacterAi,
}

impl Provider {
    fn from_folder(folder: &str) -> Option<Self> {
        match folder {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "characterai" => Some(Self::CharacterAi),
            _ => None,
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::CharacterAi => "characterai",
        }
    }

    /// How a caller names the service.
    pub fn service(self) -> &'static str {
        match self {
            Self::OpenAi => "OpenAI",
            Self::Anthropic => "Anthropic",
            Self::CharacterAi => "Character.AI",
        }
    }

    /// How a caller names one of the components the page lists.
    fn spoken(self, component: &str) -> String {
        let name = match (self, component) {
            (Self::OpenAi, "API") => "the OpenAI API",
            (Self::OpenAi, "Labs") => "OpenAI Labs",
            (Self::OpenAi, "Playground") => "the OpenAI Playground",
            (Self::Anthropic, "api.anthropic.com") => "the Anthropic API",
            (Self::Anthropic, "console.anthropic.com") => "the Anthropic Console",
            (_, other) => other,
        };
        name.to_string()
    }
}

/// The status an update was filed under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
pub enum Stage {
    Investigating,
    Identified,
    Monitoring,
    /// A further update with no change of status.
    Update,
    Resolved,
    Postmortem,
}

impl Stage {
    fn label(self) -> &'static str {
        match self {
            Self::Investigating => "Investigating",
            Self::Identified => "Identified",
            Self::Monitoring => "Monitoring",
            Self::Update => "Update",
            Self::Resolved => "Resolved",
            Self::Postmortem => "Postmortem",
        }
    }

    fn closes(self) -> bool {
        matches!(self, Self::Resolved | Self::Postmortem)
    }
}

/// One update as the `Updates` column lists it.
#[derive(Clone, Debug, Deserialize)]
pub struct RawUpdate {
    #[serde(rename = "Update_Title")]
    pub stage: Stage,
    #[serde(rename = "Update_Body")]
    pub body: String,
    #[serde(rename = "Update_Timestamp")]
    pub timestamp: String,
}

/// One row of an incident file.
#[derive(Clone, Debug)]
pub struct RawIncident {
    pub provider: Provider,
    pub file: String,
    pub title: String,
    pub link: String,
    /// Newest first, as the file has them.
    pub updates: Vec<RawUpdate>,
    /// "This incident affected: API and ChatGPT."
    pub affected: String,
}

/// The rows of one incident file.
pub fn rows_of(provider: Provider, file: &str, csv_text: &str) -> Result<Vec<RawIncident>, LlmError> {
    let csv = |source| LlmError::Csv {
        file: file.to_string(),
        source,
    };
    let table = Table::parse(csv_text).map_err(csv)?;
    let (title, link, updates, service) = (
        table.column("Incident_Title").map_err(csv)?,
        table.column("Incident_Link").map_err(csv)?,
        table.column("Updates").map_err(csv)?,
        table.column("Service").map_err(csv)?,
    );
    table
        .rows
        .iter()
        .map(|row| {
            let parsed: Vec<RawUpdate> =
                serde_json::from_str(&row[updates]).map_err(|source| LlmError::Updates {
                    file: file.to_string(),
                    link: row[link].clone(),
                    source,
                })?;
            Ok(RawIncident {
                provider,
                file: file.to_string(),
                title: row[title].clone(),
                link: row[link].clone(),
                updates: parsed,
                affected: row[service].clone(),
            })
        })
        .collect()
}

/// `"This incident affected: API, ChatGPT, and Labs."` → the components' own names.
pub fn affected_components(text: &str) -> Vec<String> {
    let Some(list) = text.trim().strip_prefix("This incident affected:") else {
        return Vec::new();
    };
    let list = list.trim();
    let list = list.strip_suffix('.').unwrap_or(list);
    list.split(", ")
        .flat_map(|part| part.strip_prefix("and ").unwrap_or(part).split(" and "))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}

/// The page's own id of the incident: the last part of its link.
fn code_of(raw: &RawIncident) -> Result<String, LlmError> {
    let bad = || LlmError::Link {
        file: raw.file.clone(),
        link: raw.link.clone(),
    };
    let rest = raw.link.strip_prefix("https://status.").ok_or_else(bad)?;
    let (_, code) = rest.split_once("/incidents/").ok_or_else(bad)?;
    let ok =
        (4..=40).contains(&code.len()) && code.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
    if ok { Ok(code.to_string()) } else { Err(bad()) }
}

/// One row as the common shape.
pub fn incident_of(raw: &RawIncident) -> Result<Incident, LlmError> {
    let components: Vec<String> = affected_components(&raw.affected)
        .iter()
        .map(|c| raw.provider.spoken(c))
        .collect();
    let mut updates = Vec::with_capacity(raw.updates.len());
    let mut phase = Phase::Investigating;
    let mut over = false;
    for update in raw.updates.iter().rev() {
        let at = time::parse_status_page(&update.timestamp).map_err(|source| LlmError::Time {
            link: raw.link.clone(),
            source,
        })?;
        phase = match update.stage {
            Stage::Investigating => Phase::Investigating,
            Stage::Identified => Phase::RootCause,
            Stage::Monitoring => Phase::FixInProgress,
            Stage::Resolved | Stage::Postmortem => Phase::Resolved,
            // No change of status: the status stays what it was.
            Stage::Update => phase,
        };
        over = over || update.stage.closes();
        updates.push(Update {
            at,
            edited_at: None,
            phase: if over { Phase::Resolved } else { phase },
            life: if over { Life::Over } else { Life::Open },
            title: raw.title.clone(),
            label: Some(update.stage.label().to_string()),
            body: text::status_text(&update.body),
            components: components.clone(),
            workaround: false,
        });
    }
    Ok(Incident {
        code: format!("{}-{}", raw.provider.slug(), code_of(raw)?),
        service: raw.provider.service().to_string(),
        updates,
    })
}

/// What reading the archive did.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct LlmStats {
    /// Incident files read, by provider.
    pub files: BTreeMap<String, u64>,
    /// Rows (incidents) read, by provider.
    pub incidents: BTreeMap<String, u64>,
    pub timeline: Stats,
}

/// Every incident in the archive under `raw_dir`, in file and row order.
pub fn read(raw_dir: &Path) -> Result<(Vec<Incident>, LlmStats), LlmError> {
    read_archive(&mut ARCHIVE.open(raw_dir)?)
}

/// Every incident in an opened archive, in file and row order.
pub fn read_archive(archive: &mut Archive) -> Result<(Vec<Incident>, LlmStats), LlmError> {
    let mut stats = LlmStats::default();
    let mut incidents = Vec::new();
    for name in archive.names()? {
        let Some(rest) = name.strip_prefix(INCIDENT_DIR) else {
            continue;
        };
        let Some((folder, file)) = rest.split_once('/') else {
            continue;
        };
        if !file.ends_with(".csv") {
            continue;
        }
        let provider = Provider::from_folder(folder).ok_or_else(|| LlmError::Provider(folder.to_string()))?;
        let bytes = archive.member(&name)?;
        let csv_text = String::from_utf8(bytes).map_err(|_| LlmError::NotText { file: name.clone() })?;
        *stats.files.entry(provider.slug().to_string()).or_insert(0) += 1;
        for raw in rows_of(provider, &name, &csv_text)? {
            *stats.incidents.entry(provider.slug().to_string()).or_insert(0) += 1;
            incidents.push(incident_of(&raw)?);
        }
    }
    if incidents.is_empty() {
        return Err(LlmError::Empty);
    }
    Ok((incidents, stats))
}

/// Records and questions from the archive under `raw_dir`.
pub fn convert(raw_dir: &Path, seed: u64) -> Result<Converted<LlmStats>, LlmError> {
    let (incidents, mut stats) = read(raw_dir)?;
    let horizon = timeline::latest_update(&incidents);
    let built = timeline::build(&SPEC, &Rules::default(), incidents, horizon, seed)?;
    timeline::validate(&built.records, &built.queries)?;
    stats.timeline = built.stats;
    Ok(Converted {
        name: SPEC.name,
        records: built.records,
        queries: built.queries,
        sources: ARCHIVE.sources(),
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public::zenodo::testing::zip_of;
    use cb_types::QueryClass;

    /// Real rows of the archive's incident files, one file per provider.
    const OPENAI: &str = include_str!("testdata/llm/openai.csv");
    const ANTHROPIC: &str = include_str!("testdata/llm/anthropic.csv");
    const CHARACTERAI: &str = include_str!("testdata/llm/characterai.csv");

    fn all() -> Vec<Incident> {
        [
            (Provider::OpenAi, OPENAI),
            (Provider::Anthropic, ANTHROPIC),
            (Provider::CharacterAi, CHARACTERAI),
        ]
        .into_iter()
        .flat_map(|(provider, text)| rows_of(provider, "fixture.csv", text).unwrap())
        .map(|raw| incident_of(&raw).unwrap())
        .collect()
    }

    fn find<'a>(incidents: &'a [Incident], code: &str) -> &'a Incident {
        incidents.iter().find(|i| i.code == code).unwrap()
    }

    #[test]
    fn a_row_becomes_an_incident_with_utc_times_oldest_first() {
        let incidents = all();
        let incident = find(&incidents, "openai-8kb29fwgbp25");
        assert_eq!(incident.service, "OpenAI");
        let times: Vec<i64> = incident.updates.iter().map(|u| u.at).collect();
        // 11:16, 12:01, 12:50 and 13:36 Pacific daylight time.
        assert_eq!(
            times,
            [1_724_868_960, 1_724_871_660, 1_724_874_600, 1_724_877_360]
        );
        let labels: Vec<&str> = incident
            .updates
            .iter()
            .filter_map(|u| u.label.as_deref())
            .collect();
        assert_eq!(labels, ["Investigating", "Update", "Monitoring", "Resolved"]);
        let phases: Vec<Phase> = incident.updates.iter().map(|u| u.phase).collect();
        assert_eq!(
            phases,
            [
                Phase::Investigating,
                Phase::Investigating,
                Phase::FixInProgress,
                Phase::Resolved
            ]
        );
        let lives: Vec<Life> = incident.updates.iter().map(|u| u.life).collect();
        assert_eq!(lives, [Life::Open, Life::Open, Life::Open, Life::Over]);
        assert!(
            incident
                .updates
                .iter()
                .all(|u| u.components == ["the OpenAI API", "ChatGPT"])
        );
        assert_eq!(
            incident.updates[0].title,
            "Elevated errors rates in Assistants API and ChatGPT"
        );

        let identified = find(&incidents, "openai-shqybftjc7mc");
        let phases: Vec<Phase> = identified.updates.iter().map(|u| u.phase).collect();
        assert_eq!(
            phases,
            [
                Phase::RootCause,
                Phase::RootCause,
                Phase::FixInProgress,
                Phase::FixInProgress,
                Phase::Resolved
            ]
        );
        // No component listed: the incident may concern anything of Anthropic.
        assert!(
            find(&incidents, "anthropic-svk9pskgp576")
                .updates
                .iter()
                .all(|u| u.components.is_empty())
        );
    }

    #[test]
    fn the_components_line_is_split_into_names() {
        assert_eq!(
            affected_components("This incident affected: API and ChatGPT."),
            ["API", "ChatGPT"]
        );
        assert_eq!(
            affected_components("This incident affected: API, ChatGPT, Labs, and Playground."),
            ["API", "ChatGPT", "Labs", "Playground"]
        );
        assert_eq!(
            affected_components(
                "This incident affected: claude.ai, api.anthropic.com, and Claude on Vertex AI."
            ),
            ["claude.ai", "api.anthropic.com", "Claude on Vertex AI"]
        );
        assert_eq!(
            affected_components("This incident affected: Character.AI."),
            ["Character.AI"]
        );
        assert!(affected_components("").is_empty());
        assert_eq!(
            Provider::Anthropic.spoken("console.anthropic.com"),
            "the Anthropic Console"
        );
    }

    #[test]
    fn the_fixture_rows_give_unique_texts_a_merged_repeat_and_an_ambiguous_question() {
        let incidents = all();
        let horizon = timeline::latest_update(&incidents);
        let built = timeline::build(&SPEC, &Rules::default(), incidents, horizon, 42).unwrap();
        // gm6vlw4gc377 posted "We are continuing to investigate this issue." twice in one minute.
        assert_eq!(built.stats.repeats_merged, 1);
        // f8qfyzkbs439 posted two different updates in one minute: two records.
        let haiku: Vec<&str> = built
            .records
            .iter()
            .filter(|r| r.thread.as_deref() == Some("llm-anthropic-f8qfyzkbs439"))
            .map(|r| r.text.as_str())
            .collect();
        assert_eq!(haiku.len(), 3);
        assert!(
            haiku[0].contains("Update 2024-04-18 23:20 UTC, Investigating:"),
            "{}",
            haiku[0]
        );
        assert!(
            haiku[1].contains("Update 2024-04-18 23:20 UTC, Update:"),
            "{}",
            haiku[1]
        );
        let ambiguous = built
            .queries
            .iter()
            .find(|q| q.id == "q-llm-amb-openai-9hrykvh16skp-openai-pfjlxg6d6rgq")
            .unwrap();
        assert_eq!(ambiguous.class, QueryClass::Ambiguous);
        assert_eq!(ambiguous.gold.iter().filter(|g| g.grade == 3).count(), 2);
        // Questions are asked a minute or more after the update they depend on was stamped.
        for q in built.queries.iter().filter(|q| q.class == QueryClass::Changing) {
            let answer = q.gold.iter().find(|g| g.grade == 3).unwrap();
            let record = built.records.iter().find(|r| r.id == answer.record).unwrap();
            assert!(q.asked_at >= record.written_at + 60, "{}", q.id);
        }
    }

    #[test]
    fn the_archive_is_read_from_its_incident_folder_only() {
        let member = |folder: &str| format!("{INCIDENT_DIR}{folder}/incident_history_202406_202408.csv");
        let bytes = zip_of(&[
            (&member("openai"), OPENAI.as_bytes()),
            (&member("anthropic"), ANTHROPIC.as_bytes()),
            (&member("characterai"), CHARACTERAI.as_bytes()),
            (
                "LLM Service Outages and Incident Reports/raw_data/uptime/2024-08-31/api/uptime_history.csv",
                b"x,y\n",
            ),
        ]);
        let mut archive = Archive::from_bytes("fixture.zip".into(), bytes).unwrap();
        let (incidents, stats) = read_archive(&mut archive).unwrap();
        assert_eq!(incidents.len(), 8);
        assert_eq!(stats.incidents.get("openai"), Some(&4));
        assert_eq!(stats.files.values().sum::<u64>(), 3);

        let stranger = zip_of(&[(&member("mistral"), OPENAI.as_bytes())]);
        let mut archive = Archive::from_bytes("fixture.zip".into(), stranger).unwrap();
        assert!(matches!(read_archive(&mut archive), Err(LlmError::Provider(p)) if p == "mistral"));
    }
}
