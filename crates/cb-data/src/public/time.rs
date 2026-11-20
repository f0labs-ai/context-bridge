//! The time formats of the sources, all turned into Unix seconds, UTC.

use cb_types::UnixSeconds;
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum TimeError {
    #[error("not an RFC 3339 time: {0:?}")]
    Rfc3339(String),
    #[error("not a status-page time such as \"Aug 28, 2024 - 13:36 PDT\": {0:?}")]
    StatusPage(String),
    #[error("time zone {zone:?} in {text:?} is not one this loader knows")]
    Zone { zone: String, text: String },
    #[error("{0} is outside the range of dates")]
    OutOfRange(i64),
}

/// The current time.
pub fn now() -> UnixSeconds {
    Utc::now().timestamp()
}

/// `2026-11-16T16:26:26.775Z` or `2022-06-14T18:40:51+00:00` as whole seconds
/// (a fraction of a second is dropped).
pub fn parse_rfc3339(text: &str) -> Result<UnixSeconds, TimeError> {
    DateTime::parse_from_rfc3339(text.trim())
        .map(|t| t.timestamp())
        .map_err(|_| TimeError::Rfc3339(text.to_string()))
}

fn date_time(t: UnixSeconds) -> Result<DateTime<Utc>, TimeError> {
    DateTime::<Utc>::from_timestamp(t, 0).ok_or(TimeError::OutOfRange(t))
}

/// `2019-01-01T00:00:00Z`; a time outside chrono's range prints as its number.
pub fn rfc3339(t: UnixSeconds) -> String {
    match date_time(t) {
        Ok(dt) => dt.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        Err(_) => t.to_string(),
    }
}

/// A moment the way a record's text prints it: `2026-11-16 12:35:07 UTC`, or
/// `2026-11-16 12:35 UTC` for a source that times updates to the minute.
pub fn printed_utc(t: UnixSeconds, to_the_minute: bool) -> String {
    let format = if to_the_minute {
        "%Y-%m-%d %H:%M UTC"
    } else {
        "%Y-%m-%d %H:%M:%S UTC"
    };
    match date_time(t) {
        Ok(dt) => dt.format(format).to_string(),
        Err(_) => format!("{t} UTC"),
    }
}

/// The calendar month a moment falls in, UTC.
pub fn year_month(t: UnixSeconds) -> Result<(i32, u32), TimeError> {
    let dt = date_time(t)?;
    Ok((dt.year(), dt.month()))
}

/// Midnight UTC at the start of a calendar day.
pub fn utc_midnight(year: i32, month: u32, day: u32) -> Option<UnixSeconds> {
    Some(
        NaiveDate::from_ymd_opt(year, month, day)?
            .and_hms_opt(0, 0, 0)?
            .and_utc()
            .timestamp(),
    )
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// Hours to add to a local time in the zone to get UTC. The LLM status pages
/// were saved by a browser on US Pacific time, so those are the two zones met.
fn hours_behind_utc(zone: &str) -> Option<i64> {
    match zone {
        "PDT" => Some(7),
        "PST" => Some(8),
        "UTC" | "GMT" => Some(0),
        _ => None,
    }
}

/// The time a saved status page prints under an update, to the minute:
/// `Aug 28, 2024 - 13:36 PDT`, optionally after `Posted 3 days ago. `.
pub fn parse_status_page(text: &str) -> Result<UnixSeconds, TimeError> {
    let bad = || TimeError::StatusPage(text.to_string());
    let stamp = match text.rfind(" ago. ") {
        Some(at) => &text[at + " ago. ".len()..],
        None => text,
    }
    .trim();
    let (date, clock_zone) = stamp.split_once(" - ").ok_or_else(bad)?;
    let (month_day, year) = date.split_once(", ").ok_or_else(bad)?;
    let (month, day) = month_day.split_once(' ').ok_or_else(bad)?;
    let (clock, zone) = clock_zone.split_once(' ').ok_or_else(bad)?;
    let (hour, minute) = clock.split_once(':').ok_or_else(bad)?;
    let month = MONTHS.iter().position(|m| *m == month).ok_or_else(bad)? as u32 + 1;
    let number = |s: &str| s.trim().parse::<u32>().map_err(|_| bad());
    let local = NaiveDate::from_ymd_opt(year.trim().parse().map_err(|_| bad())?, month, number(day)?)
        .and_then(|d| d.and_hms_opt(number(hour).ok()?, number(minute).ok()?, 0))
        .ok_or_else(bad)?;
    let behind = hours_behind_utc(zone.trim()).ok_or_else(|| TimeError::Zone {
        zone: zone.trim().to_string(),
        text: text.to_string(),
    })?;
    Ok(local.and_utc().timestamp() + behind * 3600)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_times_drop_the_fraction_and_accept_both_utc_spellings() {
        assert_eq!(parse_rfc3339("2026-11-16T16:26:26.775Z"), Ok(1_794_846_386));
        assert_eq!(parse_rfc3339("2026-11-16T16:26:26Z"), Ok(1_794_846_386));
        assert_eq!(
            parse_rfc3339("2022-06-14T18:40:51+00:00"),
            parse_rfc3339("2022-06-14T18:40:51Z")
        );
        assert_eq!(parse_rfc3339("2020-01-09T03:10:49Z"), Ok(1_578_539_449));
        assert!(parse_rfc3339("Nov 16, 16:26 UTC").is_err());
        assert_eq!(rfc3339(1_578_539_449), "2020-01-09T03:10:49Z");
        assert_eq!(printed_utc(1_578_539_449, false), "2020-01-09 03:10:49 UTC");
        assert_eq!(printed_utc(1_578_539_449, true), "2020-01-09 03:10 UTC");
        assert_eq!(utc_midnight(2019, 1, 1), Some(1_546_300_800));
        assert_eq!(utc_midnight(2019, 2, 30), None);
        assert_eq!(year_month(1_546_300_800), Ok((2019, 1)));
        assert_eq!(year_month(1_546_300_799), Ok((2018, 12)));
    }

    #[test]
    fn status_page_times_are_pacific_and_become_utc() {
        // Checked against the data set's own UTC column: 14:00 PST is 22:00 UTC.
        assert_eq!(
            parse_status_page("Posted 3 years ago. Feb 11, 2021 - 14:00 PST"),
            Ok(1_613_080_800)
        );
        // Daylight time is seven hours behind.
        assert_eq!(parse_status_page("Apr 20, 2021 - 14:27 PDT"), Ok(1_618_954_020));
        // Late evening rolls over to the next UTC day.
        assert_eq!(
            parse_status_page("Posted 1 year ago. Mar 01, 2021 - 18:17 PST"),
            parse_rfc3339("2021-03-02T02:17:00Z")
        );
        assert!(matches!(
            parse_status_page("Aug 28, 2024 - 13:36 CEST"),
            Err(TimeError::Zone { .. })
        ));
        assert!(matches!(
            parse_status_page("28 August 2024, 13:36"),
            Err(TimeError::StatusPage(_))
        ));
        assert!(matches!(
            parse_status_page("Feb 30, 2021 - 14:00 PST"),
            Err(TimeError::StatusPage(_))
        ));
    }
}
