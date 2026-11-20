//! The download step of the GitHub source: the status history from a cut-off
//! month on, every incident it lists, and the documentation articles of the
//! kept product areas.

use super::raw::{
    self, DocPath, Fetched, HistoryPage, IncidentCode, PAGELIST_URL, RawError, article_file, article_url,
    fetched_file, history_file, history_url, incident_file, incident_url, pagelist_file,
};
use crate::public::fetch::{FetchError, Fetcher, Outcome, write_whole};
use crate::public::time::{self, TimeError};
use cb_types::UnixSeconds;
use std::collections::BTreeSet;
use std::path::Path;
use thiserror::Error;

/// The history has about 70 pages back to 2010 and never answers 404: a page
/// past the beginning is an empty page. This bounds the walk whatever happens.
const MAX_HISTORY_PAGES: u32 = 200;

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error(transparent)]
    Fetch(#[from] FetchError),
    #[error(transparent)]
    Raw(#[from] RawError),
    #[error(transparent)]
    Time(#[from] TimeError),
    #[error("the status history did not reach {cutoff} within {pages} pages")]
    HistoryTooLong { cutoff: String, pages: u32 },
    #[error("{path}: the page list is not text")]
    PagelistNotText { path: String },
}

/// The URL prefixes the GitHub download may request.
pub fn allowed_urls() -> Vec<String> {
    vec![
        format!("{}/history.json?page=", raw::STATUS_BASE),
        format!("{}/incidents/", raw::STATUS_BASE),
        PAGELIST_URL.to_string(),
        format!("{}/api/article?pathname=/en/", raw::DOCS_BASE),
    ]
}

/// What a download did.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub history_pages: u32,
    pub incidents_listed: usize,
    pub incidents_downloaded: usize,
    pub articles_listed: usize,
    pub articles_downloaded: usize,
    /// Listed items the server no longer has (404 or 410).
    pub gone: Vec<String>,
}

fn is_gone(error: &FetchError) -> bool {
    matches!(
        error,
        FetchError::Status {
            status: 404 | 410,
            ..
        }
    )
}

/// The incident codes a history page lists in months from `(year, month)` on.
pub fn codes_from(page: &HistoryPage, since: (i32, u32)) -> Result<Vec<IncidentCode>, RawError> {
    let mut codes = Vec::new();
    for month in &page.months {
        if (month.year, month.name as u32) >= since {
            for entry in &month.incidents {
                codes.push(IncidentCode::parse(&entry.code)?);
            }
        }
    }
    Ok(codes)
}

/// Download what is missing under `raw`. Incidents listed in months before the
/// month of `since` are not requested.
pub async fn download(
    raw: &Path,
    since: UnixSeconds,
    fetcher: &mut Fetcher,
    progress: &mut dyn FnMut(String),
) -> Result<Report, DownloadError> {
    let started_at = time::now();
    let since_month = time::year_month(since)?;
    let mut report = Report::default();

    // 1. The history, newest page first, until a page lies wholly before the cut-off.
    let mut codes: Vec<IncidentCode> = Vec::new();
    let mut seen = BTreeSet::new();
    let mut reached_cutoff = false;
    for page_no in 1..=MAX_HISTORY_PAGES {
        let file = history_file(raw, page_no);
        fetcher.get_to_file(&history_url(page_no), &file).await?;
        let page: HistoryPage = raw::read_json(&file)?;
        report.history_pages = page_no;
        for code in codes_from(&page, since_month)? {
            if seen.insert(code.clone()) {
                codes.push(code);
            }
        }
        if time::parse_rfc3339(&page.start_time)? <= since {
            reached_cutoff = true;
            break;
        }
    }
    if !reached_cutoff {
        return Err(DownloadError::HistoryTooLong {
            cutoff: time::rfc3339(since),
            pages: MAX_HISTORY_PAGES,
        });
    }
    report.incidents_listed = codes.len();
    progress(format!(
        "history: {} pages, {} incidents listed since {}-{:02}",
        report.history_pages,
        codes.len(),
        since_month.0,
        since_month.1
    ));

    // 2. Every listed incident.
    for (i, code) in codes.iter().enumerate() {
        match fetcher
            .get_to_file(&incident_url(code), &incident_file(raw, code))
            .await
        {
            Ok(Outcome::Downloaded { .. }) => report.incidents_downloaded += 1,
            Ok(Outcome::AlreadyThere) => {}
            Err(error) if is_gone(&error) => report.gone.push(incident_url(code)),
            Err(error) => return Err(error.into()),
        }
        if (i + 1) % 100 == 0 {
            progress(format!("incidents: {} of {}", i + 1, codes.len()));
        }
    }

    // 3. The documentation's page list, then the articles of the kept areas.
    let list_file = pagelist_file(raw);
    fetcher.get_to_file(PAGELIST_URL, &list_file).await?;
    let list = std::fs::read_to_string(&list_file).map_err(|_| DownloadError::PagelistNotText {
        path: list_file.display().to_string(),
    })?;
    let paths: Vec<DocPath> = raw::kept_paths(&list)?;
    report.articles_listed = paths.len();
    for (i, path) in paths.iter().enumerate() {
        match fetcher
            .get_to_file(&article_url(path), &article_file(raw, path))
            .await
        {
            Ok(Outcome::Downloaded { .. }) => report.articles_downloaded += 1,
            Ok(Outcome::AlreadyThere) => {}
            Err(error) if is_gone(&error) => report.gone.push(article_url(path)),
            Err(error) => return Err(error.into()),
        }
        if (i + 1) % 100 == 0 {
            progress(format!("articles: {} of {}", i + 1, paths.len()));
        }
    }

    // 4. When it ran. An earlier, interrupted run keeps its start.
    let stamp_file = fetched_file(raw);
    let earlier: Option<Fetched> = raw::read_json(&stamp_file).ok();
    let stamp = Fetched {
        started_at: earlier.map_or(started_at, |e| e.started_at.min(started_at)),
        finished_at: time::now(),
        user_agent: crate::public::fetch::USER_AGENT.to_string(),
        requests: fetcher.requests(),
    };
    let body = serde_json::to_vec_pretty(&stamp).map_err(|source| RawError::Json {
        path: stamp_file.clone(),
        source,
    })?;
    write_whole(&stamp_file, &body).await?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_come_only_from_months_at_or_after_the_cut_off() {
        let page: HistoryPage = serde_json::from_str(
            r#"{"start_time":"2018-12-01T00:00:00Z","end_time":"2019-02-28T23:59:59Z","months":[
                {"name":"February","year":2019,"incidents":[{"code":"zjwpj2dlt1g4","name":"a","impact":"none"}]},
                {"name":"January","year":2019,"incidents":[{"code":"6rbyhd6vrc0k","name":"b","impact":"minor"},
                                                          {"code":"kq75qp7try8t","name":"c","impact":"maintenance"}]},
                {"name":"December","year":2018,"incidents":[{"code":"1bvl46bpr0bl","name":"d","impact":"minor"}]}]}"#,
        )
        .unwrap();
        let codes = codes_from(&page, (2019, 1)).unwrap();
        let codes: Vec<&str> = codes.iter().map(IncidentCode::as_str).collect();
        assert_eq!(codes, ["zjwpj2dlt1g4", "6rbyhd6vrc0k", "kq75qp7try8t"]);
        assert!(codes_from(&page, (2019, 3)).unwrap().is_empty());
    }

    #[test]
    fn the_download_may_only_use_the_four_listed_addresses() {
        let allowed = allowed_urls();
        let listed = |url: &str| allowed.iter().any(|p| url.starts_with(p.as_str()));
        assert!(listed(&history_url(3)));
        assert!(listed(&incident_url(
            &IncidentCode::parse("2zfqmghmg3s3").unwrap()
        )));
        assert!(listed(PAGELIST_URL));
        assert!(listed(&article_url(&DocPath::parse("/en/actions/x").unwrap())));
        assert!(!listed("https://www.githubstatus.com/api/v2/summary.json"));
        assert!(!listed("https://docs.github.com/en/actions"));
        assert!(!listed("https://github.com/github/docs"));
    }
}
