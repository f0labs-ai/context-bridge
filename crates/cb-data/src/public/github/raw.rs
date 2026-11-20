//! The raw files of the GitHub source as they come off the wire, and where they
//! are kept under `--raw`.
//!
//! ```text
//! history/page-0001.json      https://www.githubstatus.com/history.json?page=1
//! incidents/<code>.json       https://www.githubstatus.com/incidents/<code>.json
//! docs/pagelist.txt           https://docs.github.com/api/pagelist/en/free-pro-team@latest
//! docs/articles/<name>.json   https://docs.github.com/api/article?pathname=/en/...
//! fetched.json                when the download ran
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const STATUS_BASE: &str = "https://www.githubstatus.com";
pub const DOCS_BASE: &str = "https://docs.github.com";
/// The list of every English page of the current documentation, one path a line.
pub const PAGELIST_URL: &str = "https://docs.github.com/api/pagelist/en/free-pro-team@latest";

#[derive(Debug, Error)]
pub enum RawError {
    #[error("not an incident code: {0:?}")]
    Code(String),
    #[error("not a documentation path: {0:?}")]
    DocPath(String),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

/// Read and parse one raw JSON file, naming the file when it fails.
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, RawError> {
    let bytes = std::fs::read(path).map_err(|source| RawError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_slice(&bytes).map_err(|source| RawError::Json {
        path: path.to_path_buf(),
        source,
    })
}

// ---- status history ----------------------------------------------------------------

/// A status page's own id of an incident: lower-case letters and digits.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IncidentCode(String);

impl IncidentCode {
    pub fn parse(code: &str) -> Result<Self, RawError> {
        let ok = (4..=40).contains(&code.len())
            && code.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit());
        if ok {
            Ok(Self(code.to_string()))
        } else {
            Err(RawError::Code(code.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IncidentCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn history_url(page: u32) -> String {
    format!("{STATUS_BASE}/history.json?page={page}")
}

pub fn incident_url(code: &IncidentCode) -> String {
    format!("{STATUS_BASE}/incidents/{code}.json")
}

pub fn history_file(raw: &Path, page: u32) -> PathBuf {
    raw.join("history").join(format!("page-{page:04}.json"))
}

pub fn incident_file(raw: &Path, code: &IncidentCode) -> PathBuf {
    raw.join("incidents").join(format!("{code}.json"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
pub enum MonthName {
    January = 1,
    February,
    March,
    April,
    May,
    June,
    July,
    August,
    September,
    October,
    November,
    December,
}

/// One page of `history.json`: three calendar months, newest first.
#[derive(Debug, Deserialize)]
pub struct HistoryPage {
    pub months: Vec<HistoryMonth>,
    /// Start of the oldest month on the page, RFC 3339.
    pub start_time: String,
    /// End of the newest month on the page, RFC 3339.
    pub end_time: String,
}

#[derive(Debug, Deserialize)]
pub struct HistoryMonth {
    pub name: MonthName,
    pub year: i32,
    pub incidents: Vec<HistoryEntry>,
}

#[derive(Debug, Deserialize)]
pub struct HistoryEntry {
    pub code: String,
    pub name: String,
    pub impact: Impact,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Impact {
    None,
    Minor,
    Major,
    Critical,
    /// Planned work, not an incident.
    Maintenance,
    #[serde(other)]
    Unknown,
}

/// The status of an incident or of one of its updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Investigating,
    Identified,
    Monitoring,
    Resolved,
    Postmortem,
    // Planned work.
    Scheduled,
    InProgress,
    Verifying,
    Completed,
    #[serde(other)]
    Unknown,
}

/// `incidents/<code>.json`: one incident with every update.
#[derive(Debug, Deserialize)]
pub struct IncidentFile {
    pub id: String,
    pub name: String,
    pub status: Status,
    pub impact: Impact,
    pub created_at: String,
    #[serde(default)]
    pub resolved_at: Option<String>,
    /// Set for planned work.
    #[serde(default)]
    pub scheduled_for: Option<String>,
    /// Newest first.
    pub incident_updates: Vec<IncidentUpdate>,
    /// The components attached to the incident when the file was fetched.
    #[serde(default)]
    pub components: Option<Vec<Component>>,
}

#[derive(Debug, Deserialize)]
pub struct IncidentUpdate {
    pub id: String,
    pub status: Status,
    #[serde(default)]
    pub body: Option<String>,
    /// When the update was posted.
    pub created_at: String,
    /// When it was last changed; later than `created_at` when it was edited.
    #[serde(default)]
    pub updated_at: Option<String>,
    /// The time the page shows for it; can be set back by its author.
    #[serde(default)]
    pub display_at: Option<String>,
    /// The components attached to the incident as of this update.
    #[serde(default)]
    pub affected_components: Option<Vec<AffectedComponent>>,
}

#[derive(Debug, Deserialize)]
pub struct Component {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct AffectedComponent {
    pub name: String,
}

// ---- documentation -----------------------------------------------------------------

/// A page path of the documentation site, as the page list prints it (`/en/...`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DocPath(String);

impl DocPath {
    pub fn parse(path: &str) -> Result<Self, RawError> {
        let ok = path.starts_with("/en/")
            && path.len() <= 400
            && !path.contains("..")
            && !path.contains("//")
            && path
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.' | b'@'));
        if ok {
            Ok(Self(path.to_string()))
        } else {
            Err(RawError::DocPath(path.to_string()))
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The last part of the path: the article's own name.
    pub fn last_segment(&self) -> &str {
        self.0.rsplit('/').next().unwrap_or("")
    }

    /// The file an article is saved under: its path with `/` written as `__`.
    pub fn file_name(&self) -> String {
        format!("{}.json", self.0.trim_start_matches("/en/").replace('/', "__"))
    }
}

impl fmt::Display for DocPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

pub fn article_url(path: &DocPath) -> String {
    format!("{DOCS_BASE}/api/article?pathname={path}")
}

pub fn pagelist_file(raw: &Path) -> PathBuf {
    raw.join("docs").join("pagelist.txt")
}

pub fn article_file(raw: &Path, path: &DocPath) -> PathBuf {
    raw.join("docs").join("articles").join(path.file_name())
}

/// A product area of the documentation that is kept as standing knowledge.
#[derive(Debug, PartialEq, Eq)]
pub struct DocArea {
    /// Short name, used in record ids.
    pub slug: &'static str,
    /// Pages whose path starts with one of these belong to the area.
    pub prefixes: &'static [&'static str],
}

/// The areas that appear as components on the status page (docs/PUBLIC_DATA.md 1.1).
/// The REST reference pages are generated tables of endpoints and are left out;
/// the guides to using the API are kept.
pub const DOC_AREAS: &[DocArea] = &[
    DocArea {
        slug: "pull-requests",
        prefixes: &["/en/pull-requests/"],
    },
    DocArea {
        slug: "actions",
        prefixes: &["/en/actions/"],
    },
    DocArea {
        slug: "issues",
        prefixes: &["/en/issues/"],
    },
    DocArea {
        slug: "codespaces",
        prefixes: &["/en/codespaces/"],
    },
    DocArea {
        slug: "pages",
        prefixes: &["/en/pages/"],
    },
    DocArea {
        slug: "packages",
        prefixes: &["/en/packages/"],
    },
    DocArea {
        slug: "copilot",
        prefixes: &["/en/copilot/"],
    },
    DocArea {
        slug: "repositories",
        prefixes: &["/en/repositories/"],
    },
    DocArea {
        slug: "git",
        prefixes: &["/en/get-started/git-basics/", "/en/get-started/using-git/"],
    },
    DocArea {
        slug: "webhooks",
        prefixes: &["/en/webhooks/"],
    },
    DocArea {
        slug: "api",
        prefixes: &[
            "/en/rest/about-the-rest-api/",
            "/en/rest/using-the-rest-api/",
            "/en/rest/authentication/",
            "/en/rest/guides/",
            "/en/rest/quickstart",
        ],
    },
    DocArea {
        slug: "authentication",
        prefixes: &["/en/authentication/"],
    },
];

fn area_of_str(path: &str) -> Option<&'static DocArea> {
    DOC_AREAS
        .iter()
        .find(|area| area.prefixes.iter().any(|p| path.starts_with(p)))
}

/// The area a page belongs to, if it is one of the kept areas.
pub fn area_of(path: &DocPath) -> Option<&'static DocArea> {
    area_of_str(path.as_str())
}

/// The pages of the kept areas, in the order the page list gives them. A line
/// of a kept area that is not a clean path is an error; other lines are not
/// looked at.
pub fn kept_paths(pagelist: &str) -> Result<Vec<DocPath>, RawError> {
    pagelist
        .lines()
        .map(str::trim)
        .filter(|line| area_of_str(line).is_some())
        .map(DocPath::parse)
        .collect()
}

/// `docs/articles/<name>.json`: the article API's answer.
#[derive(Debug, Deserialize)]
pub struct ArticleFile {
    pub meta: ArticleMeta,
    /// The article as Markdown, templating already resolved by the site.
    pub body: String,
}

#[derive(Debug, Deserialize)]
pub struct ArticleMeta {
    pub title: String,
    #[serde(default)]
    pub intro: Option<String>,
    #[serde(default)]
    pub product: Option<String>,
    #[serde(rename = "documentType", default)]
    pub document_type: Option<String>,
}

/// `fetched.json`: when a download ran, for the manifest.
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fetched {
    /// Unix seconds, UTC.
    pub started_at: i64,
    pub finished_at: i64,
    pub user_agent: String,
    pub requests: u64,
}

pub fn fetched_file(raw: &Path) -> PathBuf {
    raw.join("fetched.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_and_paths_are_checked_before_they_reach_a_url() {
        assert!(IncidentCode::parse("2zfqmghmg3s3").is_ok());
        assert!(IncidentCode::parse("../etc").is_err());
        assert!(IncidentCode::parse("abc?x=1").is_err());
        assert!(IncidentCode::parse("").is_err());
        let code = IncidentCode::parse("2zfqmghmg3s3").unwrap();
        assert_eq!(
            incident_url(&code),
            "https://www.githubstatus.com/incidents/2zfqmghmg3s3.json"
        );
        assert_eq!(history_url(7), "https://www.githubstatus.com/history.json?page=7");

        let path = DocPath::parse("/en/actions/how-tos/write-workflows").unwrap();
        assert_eq!(
            article_url(&path),
            "https://docs.github.com/api/article?pathname=/en/actions/how-tos/write-workflows"
        );
        assert_eq!(path.file_name(), "actions__how-tos__write-workflows.json");
        assert_eq!(path.last_segment(), "write-workflows");
        assert!(DocPath::parse("/fr/actions").is_err());
        assert!(DocPath::parse("/en/actions/../../x").is_err());
        assert!(DocPath::parse("/en/actions?x=1&y=2").is_err());
        assert!(DocPath::parse("/en/a b").is_err());
    }

    #[test]
    fn only_the_listed_areas_are_kept_from_the_page_list() {
        let list = "/en\n/en/actions\n/en/actions/how-tos/write-workflows\n/en/rest/issues/comments\n\
                    /en/rest/guides/scripting-with-the-rest-api-and-javascript\n/en/billing/concepts\n\
                    /en/get-started/using-git/about-git-rebase\n/en/get-started/onboarding/x\n";
        let kept = kept_paths(list).unwrap();
        let names: Vec<&str> = kept.iter().map(DocPath::as_str).collect();
        assert_eq!(
            names,
            [
                "/en/actions/how-tos/write-workflows",
                "/en/rest/guides/scripting-with-the-rest-api-and-javascript",
                "/en/get-started/using-git/about-git-rebase",
            ]
        );
        assert_eq!(area_of(&kept[2]).map(|a| a.slug), Some("git"));
        // A line outside the kept areas is never turned into a request, whatever it holds.
        assert_eq!(kept_paths("/en/ok\nhttps://elsewhere.example/x\n").unwrap(), []);
        // A line inside a kept area must be a clean path.
        assert!(kept_paths("/en/actions/x?pathname=/en/y\n").is_err());
        assert!(kept_paths("/en/actions/../../site-policy\n").is_err());
    }

    #[test]
    fn statuses_and_impacts_parse_and_unknown_ones_do_not_break_the_file() {
        let update: IncidentUpdate = serde_json::from_str(
            r#"{"id":"u1","status":"a_new_status","body":"x","created_at":"2026-11-16T16:26:26.775Z"}"#,
        )
        .unwrap();
        assert_eq!(update.status, Status::Unknown);
        assert_eq!(update.affected_components.map(|c| c.len()), None);
        let s: Status = serde_json::from_str("\"in_progress\"").unwrap();
        assert_eq!(s, Status::InProgress);
        let i: Impact = serde_json::from_str("\"maintenance\"").unwrap();
        assert_eq!(i, Impact::Maintenance);
        let m: MonthName = serde_json::from_str("\"February\"").unwrap();
        assert_eq!(m as u32, 2);
    }
}
