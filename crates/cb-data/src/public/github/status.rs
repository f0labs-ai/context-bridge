//! GitHub's status history as incidents (docs/PUBLIC_DATA.md section 1).

use super::raw::{self, AffectedComponent, Impact, IncidentFile, IncidentUpdate, RawError, Status};
use crate::public::text;
use crate::public::time::{self, TimeError};
use crate::public::timeline::{Incident, Life, SourceSpec, Update};
use cb_types::{Phase, UnixSeconds};
use chrono::DateTime;
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;

pub const SPEC: SourceSpec = SourceSpec {
    name: "github",
    record_source: "github-status",
    id_prefix: "gh",
    // Updates are timed to the millisecond; a record keeps the second.
    time_precision_s: 1,
    // A question about an incident that names a component names it, as the
    // Google Cloud and LLM sets do; only incidents that name none are asked
    // about as "GitHub".
    service_question_share: 0.0,
};

/// How a caller names the service as a whole.
pub const SERVICE: &str = "GitHub";

/// A component the status page attached to incidents that named none.
const PLACEHOLDER_COMPONENT: &str = "Visit www.githubstatus.com for more information";

pub const LEFT_OUT_MAINTENANCE: &str = "planned maintenance";
pub const LEFT_OUT_BEFORE_CUTOFF: &str = "created before the cut-off";
pub const LEFT_OUT_UNKNOWN_STATUS: &str = "an update has a status this loader does not know";

#[derive(Debug, Error)]
pub enum StatusError {
    #[error(transparent)]
    Raw(#[from] RawError),
    #[error("incident {code}: {source}")]
    Time {
        code: String,
        #[source]
        source: TimeError,
    },
    #[error("{path}: {source}")]
    List {
        path: String,
        #[source]
        source: std::io::Error,
    },
}

/// What becomes of one raw incident.
#[derive(Debug, PartialEq)]
pub enum Decision {
    Use(Incident),
    LeaveOut(&'static str),
}

/// How a caller names a component of the status page. The page renamed some
/// components over the years ("GitHub Actions" became "Actions"); both names
/// lead to the same words. One old component covered two products.
pub fn spoken(component: &str) -> Vec<String> {
    let names: &[&str] = match component {
        PLACEHOLDER_COMPONENT => &[],
        "Actions" | "GitHub Actions" => &["GitHub Actions"],
        "Packages" | "GitHub Packages" => &["GitHub Packages"],
        "Pages" | "GitHub Pages" => &["GitHub Pages"],
        "Issues" => &["GitHub Issues"],
        "Pull Requests" => &["GitHub pull requests"],
        "Issues, PRs, Projects" | "Issues, Pull Requests, Projects" => {
            &["GitHub Issues", "GitHub pull requests"]
        }
        "Git Operations" => &["Git operations on GitHub"],
        "Webhooks" => &["GitHub webhooks"],
        "API Requests" => &["the GitHub API"],
        "Copilot" => &["GitHub Copilot"],
        "Codespaces" => &["GitHub Codespaces"],
        "Copilot AI Model Providers" => &["the AI models in GitHub Copilot"],
        "Notifications" => &["GitHub notifications"],
        other => {
            let other = text::squash(other);
            return if other.contains("GitHub") {
                vec![other]
            } else {
                vec![format!("{other} on GitHub")]
            };
        }
    };
    names.iter().map(|n| (*n).to_string()).collect()
}

/// The components a name of the status page's own form "Incident with X, Y
/// and Z" lists. Other names are free text and say nothing reliable.
pub fn components_in_name(name: &str) -> Vec<String> {
    let Some(list) = name.trim().strip_prefix("Incident with ") else {
        return Vec::new();
    };
    let mut names: Vec<String> = Vec::new();
    let parts = list
        .split(", ")
        .flat_map(|part| part.strip_prefix("and ").unwrap_or(part).split(" and "));
    for part in parts.map(str::trim).filter(|p| !p.is_empty()) {
        for name in spoken(part) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

fn spoken_all(components: &[AffectedComponent]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for component in components {
        for name in spoken(&component.name) {
            if !names.contains(&name) {
                names.push(name);
            }
        }
    }
    names
}

fn phase_of(status: Status) -> Option<Phase> {
    match status {
        Status::Investigating => Some(Phase::Investigating),
        Status::Identified => Some(Phase::RootCause),
        Status::Monitoring => Some(Phase::FixInProgress),
        Status::Resolved | Status::Postmortem => Some(Phase::Resolved),
        Status::Scheduled | Status::InProgress | Status::Verifying | Status::Completed | Status::Unknown => {
            None
        }
    }
}

fn label_of(status: Status) -> &'static str {
    match status {
        Status::Investigating => "Investigating",
        Status::Identified => "Identified",
        Status::Monitoring => "Monitoring",
        Status::Resolved => "Resolved",
        Status::Postmortem => "Postmortem",
        Status::Scheduled => "Scheduled",
        Status::InProgress => "In progress",
        Status::Verifying => "Verifying",
        Status::Completed => "Completed",
        Status::Unknown => "Update",
    }
}

fn is_maintenance(file: &IncidentFile) -> bool {
    file.impact == Impact::Maintenance
        || file.scheduled_for.is_some()
        || file.incident_updates.iter().any(|u| {
            matches!(
                u.status,
                Status::Scheduled | Status::InProgress | Status::Verifying | Status::Completed
            )
        })
}

/// Milliseconds since the epoch, to put updates posted in the same second in order.
fn millis(code: &str, stamp: &str) -> Result<i64, StatusError> {
    DateTime::parse_from_rfc3339(stamp.trim())
        .map(|t| t.timestamp_millis())
        .map_err(|_| StatusError::Time {
            code: code.to_string(),
            source: TimeError::Rfc3339(stamp.to_string()),
        })
}

/// One raw incident as the common shape, or the reason it is not used.
/// `since` is the cut-off: incidents created before it are left out.
pub fn decide(file: &IncidentFile, since: UnixSeconds) -> Result<Decision, StatusError> {
    let code = raw::IncidentCode::parse(&file.id)?;
    let seconds = |stamp: &str| {
        time::parse_rfc3339(stamp).map_err(|source| StatusError::Time {
            code: code.to_string(),
            source,
        })
    };
    if is_maintenance(file) {
        return Ok(Decision::LeaveOut(LEFT_OUT_MAINTENANCE));
    }
    if seconds(&file.created_at)? < since {
        return Ok(Decision::LeaveOut(LEFT_OUT_BEFORE_CUTOFF));
    }
    if file.incident_updates.iter().any(|u| u.status == Status::Unknown) {
        return Ok(Decision::LeaveOut(LEFT_OUT_UNKNOWN_STATUS));
    }

    // Oldest first, by the millisecond.
    let mut ordered: Vec<(i64, &IncidentUpdate)> = Vec::with_capacity(file.incident_updates.len());
    for update in file.incident_updates.iter().rev() {
        ordered.push((millis(code.as_str(), &update.created_at)?, update));
    }
    ordered.sort_by_key(|(at, _)| *at);

    let title = text::squash(&file.name);
    // An incident none of whose updates lists a component may still name what it
    // affects in its title ("Incident with Dependabot").
    let lists_any = file.incident_updates.iter().any(|u| {
        u.affected_components
            .as_deref()
            .is_some_and(|c| !spoken_all(c).is_empty())
    });
    let named = if lists_any {
        Vec::new()
    } else {
        components_in_name(&title)
    };
    let mut updates = Vec::with_capacity(ordered.len());
    let mut components: Vec<String> = named;
    let mut over = false;
    for (_, raw_update) in ordered {
        let at = seconds(&raw_update.created_at)?;
        let edited_at = match &raw_update.updated_at {
            Some(stamp) => Some(seconds(stamp)?).filter(|edited| *edited > at),
            None => None,
        };
        // The components attached as of this update: the latest list an update carried.
        if let Some(listed) = raw_update
            .affected_components
            .as_deref()
            .filter(|c| !c.is_empty())
        {
            components = spoken_all(listed);
        }
        let closes = matches!(raw_update.status, Status::Resolved | Status::Postmortem);
        // Once the incident is resolved it stays so: a later update is a postscript,
        // whatever status the page filed it under.
        let (phase, label) = if over && !closes {
            (Phase::Resolved, label_of(Status::Resolved))
        } else {
            (
                phase_of(raw_update.status).unwrap_or(Phase::Investigating),
                label_of(raw_update.status),
            )
        };
        over = over || closes;
        updates.push(Update {
            at,
            edited_at,
            phase,
            life: if over { Life::Over } else { Life::Open },
            title: title.clone(),
            label: Some(label.to_string()),
            body: text::status_text(raw_update.body.as_deref().unwrap_or("")),
            components: components.clone(),
            workaround: false,
        });
    }
    Ok(Decision::Use(Incident {
        code: code.to_string(),
        service: SERVICE.to_string(),
        updates,
    }))
}

/// What reading the raw incident files gave.
#[derive(Debug, Default)]
pub struct Read {
    pub incidents: Vec<Incident>,
    /// Raw incidents that were not used, by reason.
    pub left_out: BTreeMap<String, u64>,
    pub files: u64,
}

/// Every incident file under `raw/incidents`, in file-name order.
pub fn read(raw_dir: &Path, since: UnixSeconds) -> Result<Read, StatusError> {
    let dir = raw_dir.join("incidents");
    let list = |source| StatusError::List {
        path: dir.display().to_string(),
        source,
    };
    let mut files = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(list)? {
        let path = entry.map_err(list)?.path();
        if path.extension().is_some_and(|e| e == "json") {
            files.push(path);
        }
    }
    files.sort();
    let mut read = Read::default();
    for path in files {
        let file: IncidentFile = raw::read_json(&path)?;
        read.files += 1;
        match decide(&file, since)? {
            Decision::Use(incident) => read.incidents.push(incident),
            Decision::LeaveOut(why) => *read.left_out.entry(why.to_string()).or_insert(0) += 1,
        }
    }
    Ok(read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public::timeline::{self, Rules};
    use cb_types::{Query, QueryClass, Record};

    /// Real incidents from the status history, trimmed to the fields the loader reads.
    const FIXTURES: [&str; 7] = [
        include_str!("../testdata/github/incidents/2zfqmghmg3s3.json"),
        include_str!("../testdata/github/incidents/lc8s9rtyjmgq.json"),
        include_str!("../testdata/github/incidents/xlm9bd5xq5bk.json"),
        include_str!("../testdata/github/incidents/rp7669f45fjw.json"),
        include_str!("../testdata/github/incidents/1lnqb2vk25vn.json"),
        include_str!("../testdata/github/incidents/kq75qp7try8t.json"),
        include_str!("../testdata/github/incidents/kz4khcgdsfdv.json"),
    ];
    /// 2019-01-01T00:00:00Z
    const SINCE: UnixSeconds = 1_546_300_800;
    /// 2026-11-19T00:00:00Z
    const HORIZON: UnixSeconds = 1_795_046_400;

    fn file(code: &str) -> IncidentFile {
        FIXTURES
            .iter()
            .map(|text| serde_json::from_str::<IncidentFile>(text).unwrap())
            .find(|f| f.id == code)
            .unwrap()
    }

    fn used(code: &str) -> Incident {
        match decide(&file(code), SINCE).unwrap() {
            Decision::Use(incident) => incident,
            Decision::LeaveOut(why) => panic!("{code} left out: {why}"),
        }
    }

    fn all_used() -> Vec<Incident> {
        FIXTURES
            .iter()
            .map(|text| serde_json::from_str::<IncidentFile>(text).unwrap())
            .filter_map(|f| match decide(&f, SINCE).unwrap() {
                Decision::Use(incident) => Some(incident),
                Decision::LeaveOut(_) => None,
            })
            .collect()
    }

    fn of_thread<'a>(records: &'a [Record], thread: &str) -> Vec<&'a Record> {
        records
            .iter()
            .filter(|r| r.thread.as_deref() == Some(thread))
            .collect()
    }

    #[test]
    fn planned_work_and_incidents_before_the_cut_off_are_not_used() {
        assert_eq!(
            decide(&file("kq75qp7try8t"), SINCE).unwrap(),
            Decision::LeaveOut(LEFT_OUT_MAINTENANCE)
        );
        // 2022-01-01 as the cut-off leaves out the 2021 incident.
        assert_eq!(
            decide(&file("lc8s9rtyjmgq"), 1_640_995_200).unwrap(),
            Decision::LeaveOut(LEFT_OUT_BEFORE_CUTOFF)
        );
    }

    #[test]
    fn updates_come_oldest_first_with_their_status_mapped_to_a_phase() {
        let incident = used("lc8s9rtyjmgq");
        let phases: Vec<Phase> = incident.updates.iter().map(|u| u.phase).collect();
        assert_eq!(
            phases,
            [
                Phase::Investigating,
                Phase::Investigating,
                Phase::RootCause,
                Phase::FixInProgress,
                Phase::Resolved
            ]
        );
        let labels: Vec<&str> = incident
            .updates
            .iter()
            .filter_map(|u| u.label.as_deref())
            .collect();
        assert_eq!(
            labels,
            [
                "Investigating",
                "Investigating",
                "Identified",
                "Monitoring",
                "Resolved"
            ]
        );
        let lives: Vec<Life> = incident.updates.iter().map(|u| u.life).collect();
        assert_eq!(
            lives,
            [Life::Open, Life::Open, Life::Open, Life::Open, Life::Over]
        );
        // 2021-08-10T22:24:05.186Z, to the second.
        assert_eq!(incident.updates[2].at, 1_628_634_245);
        // The second update lists no component: the incident still affects Actions.
        assert!(
            incident
                .updates
                .iter()
                .all(|u| u.components == ["GitHub Actions"])
        );
        assert_eq!(incident.code, "lc8s9rtyjmgq");
        assert_eq!(incident.service, "GitHub");

        let pull_requests = used("2zfqmghmg3s3");
        assert_eq!(pull_requests.updates.len(), 12);
        assert!(pull_requests.updates.windows(2).all(|w| w[0].at <= w[1].at));
        assert_eq!(pull_requests.updates[0].at, 1_794_830_381);
        assert_eq!(pull_requests.updates[0].components, ["GitHub pull requests"]);
        // Actions joined the incident at its seventh update.
        assert_eq!(
            pull_requests.updates[6].components,
            ["GitHub Actions", "GitHub pull requests"]
        );
        assert_eq!(pull_requests.updates[0].title, "Incident with Pull Requests");
    }

    #[test]
    fn component_names_are_what_a_caller_would_say() {
        assert_eq!(spoken("Pull Requests"), ["GitHub pull requests"]);
        assert_eq!(spoken("Actions"), spoken("GitHub Actions"));
        assert_eq!(spoken("API Requests"), ["the GitHub API"]);
        assert_eq!(
            spoken("Issues, PRs, Projects"),
            ["GitHub Issues", "GitHub pull requests"]
        );
        assert!(spoken(PLACEHOLDER_COMPONENT).is_empty());
        assert_eq!(spoken("Discussions"), ["Discussions on GitHub"]);
        assert_eq!(
            spoken("GitHub Community Discussions"),
            ["GitHub Community Discussions"]
        );
        assert_eq!(
            components_in_name("Incident with Dependabot"),
            ["Dependabot on GitHub"]
        );
        assert_eq!(
            components_in_name("Incident with GitHub Actions, GitHub Pages, and Webhooks"),
            ["GitHub Actions", "GitHub Pages", "GitHub webhooks"]
        );
        assert_eq!(
            components_in_name("Incident with Codespaces and API Requests"),
            ["GitHub Codespaces", "the GitHub API"]
        );
        assert!(components_in_name("Disruption with some GitHub services").is_empty());
        assert!(components_in_name("Run start delays in Actions").is_empty());
    }

    #[test]
    fn the_fixture_timelines_give_the_expected_records_and_questions() {
        let built = timeline::build(&SPEC, &Rules::default(), all_used(), HORIZON, 42).unwrap();
        assert_eq!(built.stats.incidents_kept, 6);
        // Every rewritten closing update is two records; a same-second repeat is one.
        assert_eq!(built.stats.closing_rewrites, 4);
        assert_eq!(built.stats.repeats_merged, 1);

        let records = of_thread(&built.records, "gh-2zfqmghmg3s3");
        let ids: Vec<String> = (1..=12).map(|n| format!("gh-2zfqmghmg3s3.{n}")).collect();
        assert_eq!(records.iter().map(|r| r.id.clone()).collect::<Vec<_>>(), ids);
        for pair in records.windows(2) {
            assert_eq!(
                pair[0].truth.as_ref().and_then(|t| t.superseded_at),
                Some(pair[1].written_at)
            );
        }
        assert_eq!(records[11].truth.as_ref().and_then(|t| t.superseded_at), None);
        assert_eq!(
            records[0].text,
            "Incident with Pull Requests. Update 2026-11-16 11:59:41 UTC, Investigating: We are investigating \
             reports of degraded performance for Pull Requests"
        );
        // The first two updates are 44 seconds apart: no question in between.
        let changing: Vec<&str> = built
            .queries
            .iter()
            .filter(|q| q.thread.as_deref() == Some("gh-2zfqmghmg3s3") && q.class == QueryClass::Changing)
            .map(|q| q.id.as_str())
            .collect();
        let expected: Vec<String> = (2..=11).map(|n| format!("q-gh-2zfqmghmg3s3.c{n}")).collect();
        assert_eq!(changing, expected.iter().map(String::as_str).collect::<Vec<_>>());

        // The rewritten summary of 1lnqb2vk25vn appears only when it was written.
        let copilot = of_thread(&built.records, "gh-1lnqb2vk25vn");
        let bare = copilot[copilot.len() - 2];
        assert!(bare.text.ends_with("UTC, Resolved."), "{}", bare.text);
        assert!(
            copilot[copilot.len() - 1]
                .text
                .contains("between 09:25 UTC and 10:11 UTC")
        );

        // The two Codespaces incidents overlap: one question fits both.
        let ambiguous = built
            .queries
            .iter()
            .find(|q| q.id == "q-gh-amb-xlm9bd5xq5bk-rp7669f45fjw")
            .unwrap();
        assert!(ambiguous.text.contains("GitHub Codespaces"), "{}", ambiguous.text);
        let mut threads: Vec<&str> = ambiguous
            .gold
            .iter()
            .filter(|g| g.grade == 3)
            .map(|g| g.record.split('.').next().unwrap_or(""))
            .collect();
        threads.sort_unstable();
        assert_eq!(threads, ["gh-rp7669f45fjw", "gh-xlm9bd5xq5bk"]);
        assert!((1_703_645_583..1_703_646_416).contains(&ambiguous.asked_at));
    }

    /// For every question about one incident: its answer is the incident's latest
    /// record at the moment it is asked, and that record was there in time.
    #[test]
    fn every_answer_is_the_latest_record_of_its_incident_when_asked() {
        let built = timeline::build(&SPEC, &Rules::default(), all_used(), HORIZON, 42).unwrap();
        let check = |q: &Query| {
            let thread = q.thread.as_deref().unwrap();
            let records = of_thread(&built.records, thread);
            let latest = records.iter().rev().find(|r| r.written_at <= q.asked_at).unwrap();
            let answer: Vec<&str> = q
                .gold
                .iter()
                .filter(|g| g.grade == 3)
                .map(|g| g.record.as_str())
                .collect();
            assert_eq!(answer, [latest.id.as_str()], "{}", q.id);
            let earlier = records
                .iter()
                .filter(|r| r.written_at < latest.written_at)
                .count();
            assert_eq!(q.gold.len(), 1 + earlier, "{}", q.id);
        };
        let about_one: Vec<&Query> = built.queries.iter().filter(|q| q.thread.is_some()).collect();
        assert!(about_one.len() > 20);
        about_one.into_iter().for_each(check);
        assert!(built.queries.iter().any(|q| q.class == QueryClass::PostIncident));
    }
}
