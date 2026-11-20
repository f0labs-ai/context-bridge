//! Google Cloud incident reports (docs/PUBLIC_DATA.md section 2), Zenodo
//! 10.5281/zenodo.14010282.
//!
//! The archive's `1_raw_data/gcp_provider.parquet` holds 2,186 rows, but a row
//! is not an incident: the compiler saved Google's status feed again and
//! again, so one incident appears in up to 16 rows, each holding the updates
//! known when it was saved. Two feed formats are mixed: the old one (to 2021,
//! no incident id, no update status) and the new one (from 2021, with both).
//! The loader joins the rows of one incident and takes each update once, in
//! the earliest form a row saved it.

use super::Converted;
use super::text::{self, squash};
use super::time::{self, TimeError};
use super::timeline::{
    self, Incident, InvalidDataSet, Life, Rules, SourceSpec, Stats, TimelineError, Update,
};
use super::zenodo::{ZenodoError, ZenodoFile};
use cb_types::{Phase, UnixSeconds};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::Field;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use thiserror::Error;

pub const ARCHIVE: ZenodoFile = ZenodoFile {
    doi: "10.5281/zenodo.14010282",
    page: "https://zenodo.org/records/14010282",
    title: "Cloud Incident Reports (2016-2024)",
    author: "Chu, Xiaoyu",
    published: "2024-10-29",
    url: "https://zenodo.org/api/records/14010282/files/data.zip/content",
    file_name: "data.zip",
    bytes: 5_802_271,
    sha256: "31b4ff81f78252b6c027019fbd4c1c625ba0c0b9097395216559f64930c5ddbe",
};

/// The Google Cloud part of the archive, as the compiler collected it.
pub const MEMBER: &str = "1_raw_data/gcp_provider.parquet";

pub const SPEC: SourceSpec = SourceSpec {
    name: "gcp",
    record_source: "gcp-status",
    id_prefix: "gcp",
    // Timed to the second.
    time_precision_s: 1,
    // Questions name the product whenever the incident names one.
    service_question_share: 0.0,
};

/// How a caller names the service as a whole.
pub const SERVICE: &str = "Google Cloud";

/// An update this close to the stated end of the incident counts as after it.
const END_SLACK_S: i64 = 2;

/// The columns the loader reads.
const COLUMNS: [&str; 8] = [
    "id",
    "service_id",
    "service_name",
    "external_desc",
    "event_start_time",
    "event_end_time",
    "affected_products",
    "updates",
];

#[derive(Debug, Error)]
pub enum GcpError {
    #[error(transparent)]
    Archive(#[from] ZenodoError),
    #[error("{MEMBER}: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("{MEMBER}: no column {0:?}")]
    NoColumn(&'static str),
    #[error("{MEMBER}, row {row}, {column}: expected {expected}, found {found}")]
    Shape {
        row: usize,
        column: String,
        expected: &'static str,
        found: String,
    },
    #[error("{MEMBER}, row {row}: an update without its creation time")]
    NoCreated { row: usize },
    #[error("{MEMBER}, row {row}: {source}")]
    Time {
        row: usize,
        #[source]
        source: TimeError,
    },
    #[error("{MEMBER}, row {row}: update status {0:?} is not one this loader knows", .status)]
    Status { row: usize, status: String },
    #[error("{MEMBER}: rows {first} and {second} carry different incident ids but the same first update")]
    Ids { first: usize, second: usize },
    #[error(transparent)]
    Timeline(#[from] TimelineError),
    #[error(transparent)]
    Invalid(#[from] InvalidDataSet),
}

// ---- the rows, typed -----------------------------------------------------------------

/// One row of the file: the incident as one saving of the feed showed it.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct Row {
    /// Set in the new format only.
    pub id: Option<String>,
    pub service_id: Option<String>,
    pub service_name: Option<String>,
    /// The incident's title on the status page.
    pub external_desc: Option<String>,
    pub event_start_time: Option<String>,
    pub event_end_time: Option<String>,
    /// New format only: the products the incident affects.
    pub affected_products: Option<Vec<Product>>,
    /// Newest first.
    pub updates: Option<Vec<RowUpdate>>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct Product {
    pub title: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
pub struct RowUpdate {
    pub created: Option<String>,
    pub modified: Option<String>,
    /// New format only.
    pub status: Option<String>,
    pub text: Option<String>,
}

/// The status the new feed gives an update: how the product was doing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FeedStatus {
    Available,
    ServiceInformation,
    ServiceDisruption,
    ServiceOutage,
}

impl FeedStatus {
    fn parse(text: &str) -> Option<Self> {
        match text {
            "AVAILABLE" => Some(Self::Available),
            "SERVICE_INFORMATION" => Some(Self::ServiceInformation),
            "SERVICE_DISRUPTION" => Some(Self::ServiceDisruption),
            "SERVICE_OUTAGE" => Some(Self::ServiceOutage),
            _ => None,
        }
    }
}

fn shape(row: usize, column: &str, expected: &'static str, found: &Field) -> GcpError {
    let mut found = format!("{found:?}");
    found.truncate(80);
    GcpError::Shape {
        row,
        column: column.to_string(),
        expected,
        found,
    }
}

fn string(field: &Field, row: usize, column: &str) -> Result<Option<String>, GcpError> {
    match field {
        Field::Null => Ok(None),
        Field::Str(s) => Ok(Some(s.clone())),
        other => Err(shape(row, column, "text", other)),
    }
}

fn list<'a>(field: &'a Field, row: usize, column: &str) -> Result<&'a [Field], GcpError> {
    match field {
        Field::Null => Ok(&[]),
        Field::ListInternal(list) => Ok(list.elements()),
        other => Err(shape(row, column, "a list", other)),
    }
}

fn group<'a>(
    field: &'a Field,
    row: usize,
    column: &str,
) -> Result<Option<&'a parquet::record::Row>, GcpError> {
    match field {
        Field::Null => Ok(None),
        Field::Group(group) => Ok(Some(group)),
        other => Err(shape(row, column, "a group", other)),
    }
}

/// One record of the Parquet file as a typed row. A null column stays `None`;
/// columns the loader does not use are not looked at.
pub fn decode(record: &parquet::record::Row, row: usize) -> Result<Row, GcpError> {
    let mut out = Row::default();
    for (name, field) in record.get_column_iter() {
        if matches!(field, Field::Null) {
            continue;
        }
        match name.as_str() {
            "id" => out.id = string(field, row, name)?,
            "service_id" => out.service_id = string(field, row, name)?,
            "service_name" => out.service_name = string(field, row, name)?,
            "external_desc" => out.external_desc = string(field, row, name)?,
            "event_start_time" => out.event_start_time = string(field, row, name)?,
            "event_end_time" => out.event_end_time = string(field, row, name)?,
            "affected_products" => {
                let mut products = Vec::new();
                for element in list(field, row, name)? {
                    let mut product = Product::default();
                    if let Some(g) = group(element, row, name)? {
                        for (key, value) in g.get_column_iter() {
                            if key == "title" {
                                product.title = string(value, row, "affected_products.title")?;
                            }
                        }
                    }
                    products.push(product);
                }
                out.affected_products = Some(products);
            }
            "updates" => {
                let mut updates = Vec::new();
                for element in list(field, row, name)? {
                    let mut update = RowUpdate::default();
                    if let Some(g) = group(element, row, name)? {
                        for (key, value) in g.get_column_iter() {
                            let column = format!("updates.{key}");
                            match key.as_str() {
                                "created" => update.created = string(value, row, &column)?,
                                "modified" => update.modified = string(value, row, &column)?,
                                "status" => update.status = string(value, row, &column)?,
                                "text" => update.text = string(value, row, &column)?,
                                _ => {}
                            }
                        }
                    }
                    updates.push(update);
                }
                out.updates = Some(updates);
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Every row of the Parquet file in the archive under `raw_dir`.
pub fn read_rows(raw_dir: &Path) -> Result<Vec<Row>, GcpError> {
    let mut archive = ARCHIVE.open(raw_dir)?;
    rows_from_parquet(archive.member(MEMBER)?)
}

/// Every row of a Parquet file with the columns the loader reads.
pub fn rows_from_parquet(file: Vec<u8>) -> Result<Vec<Row>, GcpError> {
    let reader = SerializedFileReader::new(bytes::Bytes::from(file))?;
    let schema = reader.metadata().file_metadata().schema_descr();
    let names: BTreeSet<&str> = schema
        .root_schema()
        .get_fields()
        .iter()
        .map(|f| f.name())
        .collect();
    if let Some(missing) = COLUMNS.iter().find(|c| !names.contains(*c)) {
        return Err(GcpError::NoColumn(missing));
    }
    let expected = usize::try_from(reader.metadata().file_metadata().num_rows()).unwrap_or(0);
    let mut rows = Vec::with_capacity(expected);
    for (index, record) in reader.get_row_iter(None)?.enumerate() {
        rows.push(decode(&record?, index)?);
    }
    Ok(rows)
}

// ---- joining the saved copies of one incident ---------------------------------------

/// One update as one row saved it.
#[derive(Clone, Debug)]
struct Version {
    created: UnixSeconds,
    modified: Option<UnixSeconds>,
    status: Option<FeedStatus>,
    text: String,
    /// The row that saved it, and how many updates that row holds.
    row: usize,
    row_size: usize,
}

/// Minimal union–find over row numbers.
struct Groups {
    parent: Vec<usize>,
}

impl Groups {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }

    fn root(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    fn join(&mut self, a: usize, b: usize) {
        let (a, b) = (self.root(a), self.root(b));
        if a != b {
            self.parent[a.max(b)] = a.min(b);
        }
    }
}

/// What joining the rows did.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct JoinStats {
    pub rows: u64,
    pub rows_without_updates: u64,
    /// Incidents after joining.
    pub incidents: u64,
    pub most_rows_for_one_incident: u64,
    /// Rows joined to an incident because they share its id or its service and
    /// stated start, not its first update (its first update was taken down).
    pub rows_joined_without_the_first_update: u64,
    /// Updates as the rows hold them, counting every saved copy.
    pub updates_in_rows: u64,
    /// Different updates (by creation time within an incident).
    pub updates: u64,
    /// Updates saved with more than one wording; the earliest saved wording is used.
    pub updates_with_several_wordings: u64,
}

fn seconds(stamp: &str, row: usize) -> Result<UnixSeconds, GcpError> {
    time::parse_rfc3339(stamp).map_err(|source| GcpError::Time { row, source })
}

fn versions_of(row: &Row, index: usize) -> Result<Vec<Version>, GcpError> {
    let updates = row.updates.as_deref().unwrap_or(&[]);
    updates
        .iter()
        .map(|u| {
            let created = u.created.as_deref().ok_or(GcpError::NoCreated { row: index })?;
            let status = match u.status.as_deref() {
                None => None,
                Some(s) => Some(FeedStatus::parse(s).ok_or_else(|| GcpError::Status {
                    row: index,
                    status: s.to_string(),
                })?),
            };
            Ok(Version {
                created: seconds(created, index)?,
                modified: u.modified.as_deref().map(|m| seconds(m, index)).transpose()?,
                status,
                text: u.text.clone().unwrap_or_default(),
                row: index,
                row_size: updates.len(),
            })
        })
        .collect()
}

/// The rows of each incident, joined.
fn join(rows: &[Row], stats: &mut JoinStats) -> Result<Vec<Vec<usize>>, GcpError> {
    stats.rows = rows.len() as u64;
    let mut groups = Groups::new(rows.len());
    let mut by_first: BTreeMap<UnixSeconds, usize> = BTreeMap::new();
    let mut by_id: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_start: BTreeMap<(&str, &str), usize> = BTreeMap::new();
    let mut id_of_first: BTreeMap<UnixSeconds, (usize, &str)> = BTreeMap::new();
    let mut first_of: Vec<Option<UnixSeconds>> = vec![None; rows.len()];
    for (index, row) in rows.iter().enumerate() {
        let versions = versions_of(row, index)?;
        let Some(first) = versions.iter().map(|v| v.created).min() else {
            stats.rows_without_updates += 1;
            continue;
        };
        first_of[index] = Some(first);
        stats.updates_in_rows += versions.len() as u64;
        groups.join(index, *by_first.entry(first).or_insert(index));
        match (
            row.id.as_deref(),
            row.service_id.as_deref(),
            row.event_start_time.as_deref(),
        ) {
            (Some(id), _, _) => {
                groups.join(index, *by_id.entry(id).or_insert(index));
                // One first update, one id: the two ways of telling incidents apart agree.
                let (other, other_id) = *id_of_first.entry(first).or_insert((index, id));
                if other_id != id {
                    return Err(GcpError::Ids {
                        first: other,
                        second: index,
                    });
                }
            }
            (None, Some(service), Some(start)) => {
                groups.join(index, *by_start.entry((service, start)).or_insert(index));
            }
            (None, _, _) => {}
        }
    }
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, first) in first_of.iter().enumerate() {
        if first.is_some() {
            let root = groups.root(index);
            members.entry(root).or_default().push(index);
        }
    }
    let joined: Vec<Vec<usize>> = members.into_values().collect();
    for group in &joined {
        let firsts: BTreeSet<UnixSeconds> = group.iter().filter_map(|&i| first_of[i]).collect();
        if let Some(&earliest) = firsts.first() {
            stats.rows_joined_without_the_first_update +=
                group.iter().filter(|&&i| first_of[i] != Some(earliest)).count() as u64;
        }
        stats.most_rows_for_one_incident = stats.most_rows_for_one_incident.max(group.len() as u64);
    }
    stats.incidents = joined.len() as u64;
    Ok(joined)
}

// ---- reading an update -----------------------------------------------------------------

/// Words that say the issue is over.
const RESOLVED: &[&str] = &[
    "has been resolved",
    "have been resolved",
    "has been fully resolved",
    "have been fully resolved",
    "has now been resolved",
    "is resolved",
    "are resolved",
    "is now resolved",
    "are now resolved",
    "is fully resolved",
    "was resolved",
    "were resolved",
    "believed to be resolved",
];

/// Words that say a fix or mitigation is under way.
const FIX_UNDER_WAY: &[&str] = &[
    "mitigation work is",
    "mitigation is underway",
    "mitigation is in progress",
    "mitigation is being",
    "mitigation has been",
    "rolling out a fix",
    "rolling out the fix",
    "rollout of the fix",
    "rollout of a fix",
    "fix is being",
    "fix has been",
    "deploying a fix",
    "deploying the fix",
    "implementing a fix",
    "rolling back",
];

/// Words that say the cause is known.
const CAUSE_KNOWN: &[&str] = &[
    "root cause",
    "has been identified",
    "have identified",
    "we identified",
    "identified the cause",
    "identified the issue",
];

/// How the workaround field reads when there is none.
const NO_WORKAROUND: &[&str] = &[
    "none",
    "no workaround",
    "n/a",
    "na",
    "nil",
    "unknown",
    "not applicable",
    "there is no workaround",
    "there are no workaround",
    "no known workaround",
    "not available",
];

fn says(text: &str, phrases: &[&str]) -> bool {
    let lower = text.to_lowercase();
    phrases.iter().any(|p| lower.contains(p))
}

/// The value of the `Workaround:` field of an update, when it has one: the rest
/// of its line, or the lines after it up to the next field.
pub fn workaround_field(body: &str) -> Option<String> {
    let lines: Vec<&str> = body.lines().map(str::trim).collect();
    let at = lines
        .iter()
        .position(|l| l.to_lowercase().starts_with("workaround:"))?;
    let first = lines[at]["workaround:".len()..].trim();
    if !first.is_empty() {
        return Some(first.to_string());
    }
    let is_field = |l: &str| {
        l.split_once(':')
            .is_some_and(|(name, rest)| rest.trim().is_empty() || name.split_whitespace().count() <= 3)
    };
    let value: Vec<&str> = lines[at + 1..]
        .iter()
        .copied()
        .take_while(|l| !l.is_empty() && !is_field(l))
        .collect();
    (!value.is_empty()).then(|| value.join(" "))
}

/// Whether an update states a way around the problem.
pub fn states_workaround(body: &str) -> bool {
    workaround_field(body).is_some_and(|value| {
        let value = value.to_lowercase();
        let value = value.trim_start_matches(|c: char| !c.is_alphanumeric());
        !NO_WORKAROUND.iter().any(|none| {
            value == *none
                || value
                    .strip_prefix(none)
                    .is_some_and(|rest| rest.starts_with(|c: char| !c.is_alphanumeric()))
        })
    })
}

/// Names that stand for many products at once: an incident filed under one of
/// them names no product in particular.
fn is_catch_all(name: &str) -> bool {
    matches!(
        name.to_lowercase().as_str(),
        "multiple products" | "google cloud infrastructure components"
    )
}

/// How a caller names a product.
fn spoken(product: &str) -> String {
    let product = squash(product);
    match product.as_str() {
        "Operations" => "Google Cloud Operations".to_string(),
        _ => product,
    }
}

/// The products a row names: its affected products when it lists them, else
/// the product it is filed under.
fn components_of(row: &Row) -> Vec<String> {
    let listed: Vec<String> = row
        .affected_products
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter_map(|p| p.title.as_deref())
        .filter(|t| !t.trim().is_empty() && !is_catch_all(t))
        .map(spoken)
        .collect();
    if !listed.is_empty() {
        let unique: BTreeSet<String> = listed.into_iter().collect();
        return unique.into_iter().collect();
    }
    match row.service_name.as_deref() {
        Some(name) if !name.trim().is_empty() && !is_catch_all(name) => vec![spoken(name)],
        _ => Vec::new(),
    }
}

/// The incident's name as a row shows it: its product, then its title.
fn title_of(row: &Row) -> String {
    let service = row.service_name.as_deref().map(squash).filter(|s| !s.is_empty());
    let desc = row.external_desc.as_deref().map(squash).filter(|s| !s.is_empty());
    match (service, desc) {
        (Some(service), Some(desc)) => format!("{service}: {desc}"),
        (Some(service), None) => service,
        (None, Some(desc)) => desc,
        (None, None) => SERVICE.to_string(),
    }
}

/// `20200109-015543`: the time of an incident's first update, as an id.
fn code_of(first: UnixSeconds) -> String {
    time::rfc3339(first)
        .replace(['-', ':'], "")
        .replace('T', "-")
        .trim_end_matches('Z')
        .to_lowercase()
}

/// One incident from its rows.
fn incident_of(rows: &[Row], members: &[usize], stats: &mut JoinStats) -> Result<Incident, GcpError> {
    let mut versions: BTreeMap<UnixSeconds, Vec<Version>> = BTreeMap::new();
    for &index in members {
        for version in versions_of(&rows[index], index)? {
            versions.entry(version.created).or_default().push(version);
        }
    }
    // The end the latest saving states.
    let mut end: Option<UnixSeconds> = None;
    let mut latest = (0usize, i64::MIN, 0usize);
    for &index in members {
        let size = rows[index].updates.as_deref().map_or(0, <[RowUpdate]>::len);
        let last = versions_of(&rows[index], index)?
            .iter()
            .map(|v| v.created)
            .max()
            .unwrap_or(i64::MIN);
        if let Some(stated) = rows[index].event_end_time.as_deref()
            && (size, last, index) >= latest
        {
            latest = (size, last, index);
            end = Some(seconds(stated, index)?);
        }
    }

    stats.updates += versions.len() as u64;
    let mut chosen: Vec<Version> = Vec::with_capacity(versions.len());
    for (_, mut saved) in versions {
        let wordings: BTreeSet<String> = saved.iter().map(|v| squash(&v.text)).collect();
        if wordings.len() > 1 {
            stats.updates_with_several_wordings += 1;
        }
        // The earliest saving, then the earliest edit, then the text itself.
        saved.sort_by(|a, b| {
            (a.row_size, a.modified.unwrap_or(a.created), &a.text, a.row).cmp(&(
                b.row_size,
                b.modified.unwrap_or(b.created),
                &b.text,
                b.row,
            ))
        });
        if let Some(first) = saved.into_iter().next() {
            chosen.push(first);
        }
    }

    let last = chosen.len().saturating_sub(1);
    let mut signalled = false;
    let mut updates = Vec::with_capacity(chosen.len());
    for (i, version) in chosen.iter().enumerate() {
        let row = &rows[version.row];
        let body = text::status_text(&version.text);
        let available = version.status == Some(FeedStatus::Available);
        let resolved_words = says(&body, RESOLVED);
        let after_end = end.is_some_and(|e| version.created >= e - END_SLACK_S);
        signalled = signalled || available || resolved_words || after_end;
        let life = if !signalled {
            Life::Open
        } else if after_end || i == last {
            Life::Over
        } else {
            Life::Unclear
        };
        let workaround = states_workaround(&body);
        let phase = if life == Life::Over || available || resolved_words {
            Phase::Resolved
        } else if says(&body, FIX_UNDER_WAY) {
            Phase::FixInProgress
        } else if workaround {
            Phase::Workaround
        } else if says(&body, CAUSE_KNOWN) {
            Phase::RootCause
        } else {
            Phase::Investigating
        };
        updates.push(Update {
            at: version.created,
            edited_at: version.modified.filter(|m| *m > version.created),
            phase,
            life,
            title: title_of(row),
            label: None,
            body,
            components: components_of(row),
            workaround,
        });
    }
    let first = chosen.first().map_or(0, |v| v.created);
    Ok(Incident {
        code: code_of(first),
        service: SERVICE.to_string(),
        updates,
    })
}

/// The incidents of the rows.
pub fn incidents(rows: &[Row]) -> Result<(Vec<Incident>, JoinStats), GcpError> {
    let mut stats = JoinStats::default();
    let joined = join(rows, &mut stats)?;
    let mut out = Vec::with_capacity(joined.len());
    for members in &joined {
        out.push(incident_of(rows, members, &mut stats)?);
    }
    // Two incidents whose first updates share a second keep apart by a suffix.
    out.sort_by(|a, b| {
        (&a.code, &a.updates.first().map(|u| u.body.clone()))
            .cmp(&(&b.code, &b.updates.first().map(|u| u.body.clone())))
    });
    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    for incident in &mut out {
        let n = seen.entry(incident.code.clone()).or_insert(0);
        *n += 1;
        if *n > 1 {
            incident.code = format!("{}-{n}", incident.code);
        }
    }
    Ok((out, stats))
}

/// What the conversion did.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GcpStats {
    pub join: JoinStats,
    /// Kept incidents that state a workaround at some point.
    pub incidents_with_a_workaround: u64,
    pub timeline: Stats,
}

/// Records and questions from the archive under `raw_dir`.
pub fn convert(raw_dir: &Path, seed: u64) -> Result<Converted<GcpStats>, GcpError> {
    let rows = read_rows(raw_dir)?;
    convert_rows(&rows, seed)
}

/// Records and questions from rows already read.
pub fn convert_rows(rows: &[Row], seed: u64) -> Result<Converted<GcpStats>, GcpError> {
    let (incidents, join) = incidents(rows)?;
    let rules = Rules::default();
    let with_workaround = incidents
        .iter()
        .filter(|i| i.updates.len() >= rules.min_updates && i.updates.iter().any(|u| u.workaround))
        .count() as u64;
    let horizon = timeline::latest_update(&incidents);
    let built = timeline::build(&SPEC, &rules, incidents, horizon, seed)?;
    timeline::validate(&built.records, &built.queries)?;
    Ok(Converted {
        name: SPEC.name,
        records: built.records,
        queries: built.queries,
        sources: ARCHIVE.sources(),
        stats: GcpStats {
            join,
            incidents_with_a_workaround: with_workaround,
            timeline: built.stats,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use cb_types::QueryClass;

    /// Real rows of the Parquet file, trimmed to the columns the loader reads:
    /// seven incidents saved in 20 rows, in both feed formats.
    const ROWS: &str = include_str!("testdata/gcp/rows.json");

    fn rows() -> Vec<Row> {
        serde_json::from_str(ROWS).unwrap()
    }

    fn by_code<'a>(incidents: &'a [Incident], code: &str) -> &'a Incident {
        incidents.iter().find(|i| i.code == code).unwrap()
    }

    /// A two-row Parquet file laid out like the compiler's: optional strings and
    /// lists of groups. The first row is in the new feed format, the second in
    /// the old one with most columns null.
    fn parquet_file(with_updates: bool) -> Vec<u8> {
        use parquet::data_type::{ByteArray, ByteArrayType};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;
        use std::sync::Arc;

        let updates = if with_updates {
            "OPTIONAL group updates (LIST) { REPEATED group list { OPTIONAL group element {
               OPTIONAL BYTE_ARRAY created (STRING); OPTIONAL BYTE_ARRAY modified (STRING);
               OPTIONAL BYTE_ARRAY status (STRING); OPTIONAL BYTE_ARRAY text (STRING); } } }"
        } else {
            ""
        };
        let schema = format!(
            "message schema {{
               OPTIONAL BYTE_ARRAY id (STRING); OPTIONAL BYTE_ARRAY service_id (STRING);
               OPTIONAL BYTE_ARRAY service_name (STRING); OPTIONAL BYTE_ARRAY external_desc (STRING);
               OPTIONAL BYTE_ARRAY event_start_time (STRING); OPTIONAL BYTE_ARRAY event_end_time (STRING);
               {updates}
               OPTIONAL group affected_products (LIST) {{ REPEATED group list {{ OPTIONAL group element {{
                 OPTIONAL BYTE_ARRAY title (STRING); }} }} }}
             }}"
        );
        let schema = Arc::new(parse_message_type(&schema).unwrap());
        let strings = |values: &[&str]| -> Vec<ByteArray> {
            values
                .iter()
                .map(|v| ByteArray::from(v.as_bytes().to_vec()))
                .collect()
        };
        /// One leaf column: its values, definition levels and repetition levels.
        type Column = (Vec<ByteArray>, Vec<i16>, Option<Vec<i16>>);
        // In schema order.
        let mut columns: Vec<Column> = vec![
            (strings(&["Lvd7FBrPJmL7oxhqrDwt"]), vec![1, 0], None),
            (strings(&["jog4nyYkquiLeSK5s26q", "cloud-run"]), vec![1, 1], None),
            (strings(&["Cloud Filestore", "Cloud Run"]), vec![1, 1], None),
            (strings(&["Filestore instances unavailable"]), vec![1, 0], None),
            (strings(&["2022-06-14T18:40:51+00:00"]), vec![1, 0], None),
            (strings(&["2022-06-15T00:22:31+00:00"]), vec![1, 0], None),
        ];
        if with_updates {
            let both = Some(vec![0, 1, 0]);
            columns.extend([
                (
                    strings(&["2022-06-15T00:22:31+00:00", "2022-06-14T18:55:08+00:00"]),
                    vec![4, 4, 0],
                    both.clone(),
                ),
                (
                    strings(&["2022-06-14T18:55:09+00:00"]),
                    vec![3, 4, 0],
                    both.clone(),
                ),
                (
                    strings(&["AVAILABLE", "SERVICE_DISRUPTION"]),
                    vec![4, 4, 0],
                    both.clone(),
                ),
                (strings(&["Resolved.", "Investigating."]), vec![4, 4, 0], both),
            ]);
        }
        columns.push((strings(&["Cloud Filestore"]), vec![4, 0], Some(vec![0, 0])));

        let mut buffer = Vec::new();
        let mut writer =
            SerializedFileWriter::new(&mut buffer, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut group = writer.next_row_group().unwrap();
        for (values, definitions, repetitions) in columns {
            let mut column = group.next_column().unwrap().unwrap();
            column
                .typed::<ByteArrayType>()
                .write_batch(&values, Some(&definitions), repetitions.as_deref())
                .unwrap();
            column.close().unwrap();
        }
        group.close().unwrap();
        writer.close().unwrap();
        buffer
    }

    #[test]
    fn a_parquet_file_reads_into_typed_rows() {
        let rows = rows_from_parquet(parquet_file(true)).unwrap();
        assert_eq!(rows.len(), 2);
        let new = &rows[0];
        assert_eq!(new.id.as_deref(), Some("Lvd7FBrPJmL7oxhqrDwt"));
        assert_eq!(new.service_name.as_deref(), Some("Cloud Filestore"));
        assert_eq!(new.event_end_time.as_deref(), Some("2022-06-15T00:22:31+00:00"));
        assert_eq!(
            new.affected_products,
            Some(vec![Product {
                title: Some("Cloud Filestore".into())
            }])
        );
        let updates = new.updates.as_deref().unwrap();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].status.as_deref(), Some("AVAILABLE"));
        assert_eq!(updates[0].modified, None);
        assert_eq!(updates[1].modified.as_deref(), Some("2022-06-14T18:55:09+00:00"));
        assert_eq!(updates[1].text.as_deref(), Some("Investigating."));
        let old = &rows[1];
        assert_eq!(old.id, None);
        assert_eq!(old.service_id.as_deref(), Some("cloud-run"));
        assert_eq!(old.updates, None);
        assert_eq!(old.affected_products, None);

        assert!(matches!(
            rows_from_parquet(parquet_file(false)),
            Err(GcpError::NoColumn("updates"))
        ));
        assert!(matches!(
            rows_from_parquet(b"PAR1 not parquet".to_vec()),
            Err(GcpError::Parquet(_))
        ));
    }

    #[test]
    fn the_saved_copies_of_one_incident_join_and_a_shared_update_does_not_join_two() {
        let rows = rows();
        let (incidents, stats) = incidents(&rows).unwrap();
        let mut sizes: Vec<(&str, usize)> = incidents
            .iter()
            .map(|i| (i.code.as_str(), i.updates.len()))
            .collect();
        sizes.sort_unstable();
        assert_eq!(
            sizes,
            [
                ("20190311-173337", 15),
                ("20190607-023011", 4),
                ("20200109-015543", 4),
                ("20200715-142852", 6),
                ("20210218-221756", 6),
                ("20210218-224201", 3),
                ("20220428-151817", 5),
            ]
        );
        assert_eq!(stats.rows, 20);
        assert_eq!(stats.incidents, 7);
        assert_eq!(stats.updates, 43);
        let in_rows: usize = rows.iter().map(|r| r.updates.as_ref().map_or(0, Vec::len)).sum();
        assert_eq!(stats.updates_in_rows, in_rows as u64);
        // The console incident's first update was taken down in later savings; its
        // rows still join through the product and the stated start.
        assert_eq!(stats.rows_joined_without_the_first_update, 2);
        // Compute Engine and Cloud Functions share an update at 23:11:33 but stay apart.
        let shared = 1_613_689_893;
        assert!(
            by_code(&incidents, "20210218-221756")
                .updates
                .iter()
                .any(|u| u.at == shared)
        );
        assert!(
            by_code(&incidents, "20210218-224201")
                .updates
                .iter()
                .any(|u| u.at == shared)
        );
    }

    #[test]
    fn the_earliest_saved_wording_of_an_update_is_kept() {
        let update = |created: &str, modified: &str, text: &str| RowUpdate {
            created: Some(created.into()),
            modified: Some(modified.into()),
            status: None,
            text: Some(text.into()),
        };
        let first = "2019-01-01T10:00:00Z";
        let early = Row {
            service_name: Some("Cloud Run".into()),
            updates: Some(vec![update(
                first,
                first,
                "We are investigating an issue with Cloud Run DAYOFWEEK.",
            )]),
            ..Row::default()
        };
        let late = Row {
            service_name: Some("Cloud Run".into()),
            updates: Some(vec![
                update(
                    "2019-01-01T11:00:00Z",
                    "2019-01-01T11:00:00Z",
                    "The issue has been resolved.",
                ),
                update(
                    first,
                    "2019-01-01T10:30:00Z",
                    "We are investigating an issue with Cloud Run on Tuesday.",
                ),
            ]),
            ..Row::default()
        };
        let (incidents, stats) = incidents(&[late, early]).unwrap();
        assert_eq!(incidents.len(), 1);
        assert_eq!(stats.updates_with_several_wordings, 1);
        assert_eq!(
            incidents[0].updates[0].body,
            "We are investigating an issue with Cloud Run DAYOFWEEK."
        );
        assert_eq!(incidents[0].updates[0].edited_at, None);
        assert_eq!(incidents[0].updates[0].components, ["Cloud Run"]);
    }

    #[test]
    fn life_phase_and_workaround_come_from_the_text_the_feed_status_and_the_stated_end() {
        let (incidents, _) = incidents(&rows()).unwrap();
        let spanner = by_code(&incidents, "20200109-015543");
        let lives: Vec<Life> = spanner.updates.iter().map(|u| u.life).collect();
        assert_eq!(lives, [Life::Open, Life::Open, Life::Open, Life::Over]);
        let phases: Vec<Phase> = spanner.updates.iter().map(|u| u.phase).collect();
        assert_eq!(
            phases,
            [
                Phase::Investigating,
                Phase::Investigating,
                Phase::FixInProgress,
                Phase::Resolved
            ]
        );
        assert!(
            spanner
                .updates
                .iter()
                .all(|u| !u.workaround && u.components == ["Cloud Spanner"])
        );
        assert!(spanner.updates[0].title.starts_with("Cloud Spanner: "));
        assert_eq!(spanner.updates[0].label, None);

        let support = by_code(&incidents, "20220428-151817");
        let workarounds: Vec<bool> = support.updates.iter().map(|u| u.workaround).collect();
        assert_eq!(workarounds, [false, true, false, false, false]);
        let gke = by_code(&incidents, "20200715-142852");
        let workarounds: Vec<bool> = gke.updates.iter().map(|u| u.workaround).collect();
        assert_eq!(workarounds, [false, false, false, true, true, false]);
        assert_eq!(gke.updates.last().map(|u| u.life), Some(Life::Over));
    }

    #[test]
    fn the_workaround_field_is_read_and_none_is_not_a_workaround() {
        let says = |body: &str| states_workaround(&text::status_text(body));
        assert!(!says("Diagnosis: Errors.\nWorkaround: None at this time."));
        assert!(!says("Workaround: No workaround is available at the moment"));
        assert!(!says("Workaround: N/A"));
        assert!(!says("Workaround:\nNone"));
        assert!(!says("Description: nothing about it."));
        assert!(says("Workaround: Retry the deployment."));
        assert!(says("**Workaround:** Customers can enable KTD in SCC settings."));
        assert!(says(
            "Workaround:\n1. Get in touch with your administrator\n2. Visit the admin console"
        ));
        assert!(says("Workaround: Navigate to another region."));
        assert_eq!(
            workaround_field("Workaround:\n- Run jobs in us-east1\n- Retry\nDiagnosis: x").as_deref(),
            Some("- Run jobs in us-east1 - Retry")
        );
    }

    #[test]
    fn the_fixture_rows_convert_into_a_valid_data_set() {
        let converted = convert_rows(&rows(), 42).unwrap();
        let stats = &converted.stats;
        assert_eq!(stats.timeline.incidents_kept, 7);
        assert_eq!(
            converted.records.first().map(|r| r.id.as_str()),
            Some("gcp-20190311-173337.1")
        );
        for record in &converted.records {
            assert!(
                record.text.contains(" UTC: ") || record.text.ends_with(" UTC."),
                "{}",
                record.text
            );
            assert_eq!(record.source, "gcp-status");
        }
        let workaround: Vec<&cb_types::Query> =
            converted.queries.iter().filter(|q| q.id.contains(".w")).collect();
        assert_eq!(workaround.len() as u64, stats.timeline.workaround_questions);
        assert!(workaround.len() >= 2);
        for q in workaround {
            assert_eq!(q.class, QueryClass::Changing);
            let answer = q.gold.iter().find(|g| g.grade == 3).unwrap();
            let record = converted.records.iter().find(|r| r.id == answer.record).unwrap();
            assert!(record.text.contains("Workaround:"), "{}", record.text);
            assert!(q.asked_at > record.written_at && q.asked_at <= record.written_at + 600);
        }
        assert!(converted.sources[1].contains("Creative Commons Attribution 4.0 International"));
    }
}
