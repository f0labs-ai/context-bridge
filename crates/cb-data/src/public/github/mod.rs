//! GitHub: the status history as changing knowledge, the documentation as
//! standing knowledge (docs/PUBLIC_DATA.md section 1).

pub mod docs;
pub mod download;
pub mod raw;
pub mod status;

use crate::public::Converted;
use crate::public::time;
use crate::public::timeline::{self, InvalidDataSet, Rules, Stats, TimelineError};
use cb_types::UnixSeconds;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

/// Incidents created before 2019-01-01T00:00:00Z were back-filled as a single update.
pub const SINCE: UnixSeconds = 1_546_300_800;

#[derive(Debug, Error)]
pub enum GithubError {
    #[error(transparent)]
    Status(#[from] status::StatusError),
    #[error(transparent)]
    Docs(#[from] docs::DocsError),
    #[error(transparent)]
    Timeline(#[from] TimelineError),
    #[error(transparent)]
    Invalid(#[from] InvalidDataSet),
    #[error("{0} (run with --fetch first)")]
    Raw(#[from] raw::RawError),
}

/// What the conversion did.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GithubStats {
    /// Incident files read.
    pub incident_files: u64,
    /// Incidents not handed to the timeline, by reason.
    pub incidents_not_used: BTreeMap<String, u64>,
    pub timeline: Stats,
    pub docs: docs::DocsStats,
}

const TERMS: &str = "Terms: GitHub Acceptable Use Policies, section 7 \"Information Usage Restrictions\" \
    (https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies\
    #7-information-usage-restrictions), binding through the GitHub Terms of Service \
    (https://docs.github.com/en/site-policy/github-terms/github-terms-of-service, linked from the footer of \
    https://www.githubstatus.com): \"Researchers may use public, non-personal information from the Service for \
    research purposes, only if any publications resulting from that research are open access.\"";

const DOCS_LICENCE: &str = "Licence of the documentation, https://github.com/github/docs#license: \"This \
    project is dual-licensed under: Creative Commons Attribution 4.0 - for documentation and content in the \
    assets, content, and data folders (see LICENSE)\"; licence text https://github.com/github/docs/blob/main/LICENSE \
    (Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/legalcode). \
    Changes made: articles were cut into sections of 40 to 300 words; tables, code blocks and markup were \
    removed.";

fn sources(raw_dir: &Path, history_pages: usize) -> Vec<String> {
    let fetched: Option<raw::Fetched> = raw::read_json(&raw::fetched_file(raw_dir)).ok();
    let when = match fetched {
        Some(f) => format!(
            "; fetched {} to {} by \"{}\" in {} requests",
            time::rfc3339(f.started_at),
            time::rfc3339(f.finished_at),
            f.user_agent,
            f.requests
        ),
        None => String::new(),
    };
    vec![
        format!(
            "GitHub status history: https://www.githubstatus.com/history.json?page=1 to page={history_pages}, \
             and https://www.githubstatus.com/incidents/<code>.json for every incident listed from January 2019 \
             on{when}."
        ),
        TERMS.to_string(),
        "GitHub Docs (the content of https://github.com/github/docs): the English articles of the product areas \
         listed by https://docs.github.com/api/pagelist/en/free-pro-team@latest, each fetched as Markdown from \
         https://docs.github.com/api/article?pathname=<path>."
            .to_string(),
        DOCS_LICENCE.to_string(),
    ]
}

/// Records and questions from what is saved under `raw_dir`.
pub fn convert(raw_dir: &Path, seed: u64) -> Result<Converted<GithubStats>, GithubError> {
    let fetched: raw::Fetched = raw::read_json(&raw::fetched_file(raw_dir))?;
    let read = status::read(raw_dir, SINCE)?;
    // Nothing is known about the time after the download began.
    let built = timeline::build(
        &status::SPEC,
        &Rules::default(),
        read.incidents,
        fetched.started_at,
        seed,
    )?;
    let docs = docs::read(raw_dir, &docs::DocRules::default())?;
    let mut records = built.records;
    records.extend(docs.records);
    timeline::validate(&records, &built.queries)?;
    let history_pages = std::fs::read_dir(raw_dir.join("history"))
        .map(|dir| dir.filter_map(Result::ok).count())
        .unwrap_or(0);
    Ok(Converted {
        name: status::SPEC.name,
        records,
        queries: built.queries,
        sources: sources(raw_dir, history_pages),
        stats: GithubStats {
            incident_files: read.files,
            incidents_not_used: read.left_out,
            timeline: built.stats,
            docs: docs.stats,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::Tier;

    const INCIDENTS: [(&str, &str); 7] = [
        (
            "2zfqmghmg3s3",
            include_str!("../testdata/github/incidents/2zfqmghmg3s3.json"),
        ),
        (
            "lc8s9rtyjmgq",
            include_str!("../testdata/github/incidents/lc8s9rtyjmgq.json"),
        ),
        (
            "xlm9bd5xq5bk",
            include_str!("../testdata/github/incidents/xlm9bd5xq5bk.json"),
        ),
        (
            "rp7669f45fjw",
            include_str!("../testdata/github/incidents/rp7669f45fjw.json"),
        ),
        (
            "1lnqb2vk25vn",
            include_str!("../testdata/github/incidents/1lnqb2vk25vn.json"),
        ),
        (
            "kq75qp7try8t",
            include_str!("../testdata/github/incidents/kq75qp7try8t.json"),
        ),
        (
            "kz4khcgdsfdv",
            include_str!("../testdata/github/incidents/kz4khcgdsfdv.json"),
        ),
    ];
    const ARTICLE_PATH: &str =
        "/en/copilot/tutorials/roll-out-at-scale/assign-licenses/track-usage-and-adoption";
    const ARTICLE: &str = include_str!("../testdata/github/docs/track-usage-and-adoption.json");

    /// A `--raw` directory as the download leaves it, made from the fixtures.
    fn raw_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("cb-data-github-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("incidents")).unwrap();
        std::fs::create_dir_all(dir.join("history")).unwrap();
        for (code, text) in INCIDENTS {
            std::fs::write(dir.join("incidents").join(format!("{code}.json")), text).unwrap();
        }
        let path = raw::DocPath::parse(ARTICLE_PATH).unwrap();
        std::fs::create_dir_all(raw::article_file(&dir, &path).parent().unwrap()).unwrap();
        std::fs::write(raw::article_file(&dir, &path), ARTICLE).unwrap();
        let list = format!("/en\n/en/copilot\n{ARTICLE_PATH}\n/en/billing/concepts\n");
        std::fs::write(raw::pagelist_file(&dir), list).unwrap();
        let fetched = raw::Fetched {
            started_at: 1_795_046_400,
            finished_at: 1_795_047_000,
            user_agent: "test".into(),
            requests: 3,
        };
        std::fs::write(raw::fetched_file(&dir), serde_json::to_vec(&fetched).unwrap()).unwrap();
        dir
    }

    #[test]
    fn the_raw_files_convert_into_one_valid_data_set_with_its_terms() {
        let dir = raw_dir("convert");
        let converted = convert(&dir, 42).unwrap();
        let standing = converted
            .records
            .iter()
            .filter(|r| r.tier == Tier::Standing)
            .count();
        let changing = converted.records.len() - standing;
        assert_eq!(standing as u64, converted.stats.docs.records);
        assert!(standing >= 5, "{standing}");
        assert_eq!(changing as u64, converted.stats.timeline.records);
        assert_eq!(
            converted
                .stats
                .incidents_not_used
                .get(status::LEFT_OUT_MAINTENANCE),
            Some(&1)
        );
        assert!(timeline::validate(&converted.records, &converted.queries).is_ok());
        assert!(converted.queries.iter().all(|q| q.asked_at <= 1_795_046_400));
        assert!(converted.sources.iter().any(|s| s.contains(
            "\"Researchers may use public, non-personal information from the Service for research purposes, only \
             if any publications resulting from that research are open access.\""
        )));
        assert!(
            converted
                .sources
                .iter()
                .any(|s| s.contains("Creative Commons Attribution 4.0"))
        );

        let again = convert(&dir, 42).unwrap();
        assert_eq!(again.records, converted.records);
        assert_eq!(again.queries, converted.queries);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_raw_directory_that_was_never_fetched_is_an_error() {
        let dir = raw_dir("unfetched");
        std::fs::remove_file(raw::fetched_file(&dir)).unwrap();
        let error = convert(&dir, 42).err().unwrap();
        assert!(error.to_string().contains("fetched.json"), "{error}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
