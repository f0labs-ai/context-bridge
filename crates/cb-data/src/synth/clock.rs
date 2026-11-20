//! Moments and how people say them. Every moment is Unix seconds, UTC, and the
//! fictional company keeps UTC as its local time, so "today" means the UTC date.

use super::SynthError;
use chrono::{DateTime, Utc};

pub const MINUTE: i64 = 60;
pub const HOUR: i64 = 3_600;
pub const DAY: i64 = 86_400;

/// 2026-06-01T00:00:00Z, the start of the 90-day window incidents begin in.
pub const WINDOW_START: i64 = 1_780_272_000;
/// Length of the window incidents begin in, in days.
pub const WINDOW_DAYS: i64 = 90;

fn at(t: i64) -> Result<DateTime<Utc>, SynthError> {
    DateTime::from_timestamp(t, 0).ok_or(SynthError::Time(t))
}

/// "09:40": the time of day, to the minute.
pub fn clock(t: i64) -> Result<String, SynthError> {
    Ok(at(t)?.format("%H:%M").to_string())
}

/// "3 June".
pub fn date(t: i64) -> Result<String, SynthError> {
    Ok(at(t)?.format("%-d %B").to_string())
}

/// "Wednesday 3 June".
pub fn weekday_date(t: i64) -> Result<String, SynthError> {
    Ok(at(t)?.format("%A %-d %B").to_string())
}

/// The day number of a moment (days since the epoch).
pub fn day_of(t: i64) -> i64 {
    t.div_euclid(DAY)
}

/// The moment rounded down to a whole minute.
pub fn floor_minute(t: i64) -> i64 {
    t - t.rem_euclid(MINUTE)
}

/// The moment rounded up to the next multiple of `step` seconds.
pub fn ceil_to(t: i64, step: i64) -> i64 {
    let r = t.rem_euclid(step);
    if r == 0 { t } else { t + step - r }
}

/// How someone on a call at `now` names the day of `event`, as a suffix for a
/// clock time: "" or " today" on the same day, " yesterday", " tomorrow", or
/// " on Wednesday 3 June" further away.
pub fn day_phrase(event: i64, now: i64, say_today: bool) -> Result<String, SynthError> {
    let gap = day_of(event) - day_of(now);
    Ok(match gap {
        0 if say_today => " today".to_string(),
        0 => String::new(),
        -1 => " yesterday".to_string(),
        1 => " tomorrow".to_string(),
        _ => format!(" on {}", weekday_date(event)?),
    })
}

/// How a caller says how long something has been going on, in words.
pub fn elapsed_phrase(seconds: i64, variant: usize) -> &'static str {
    const BANDS: [(i64, [&str; 3]); 8] = [
        (
            10 * MINUTE,
            [
                "for the last few minutes",
                "for a few minutes now",
                "since just now, really",
            ],
        ),
        (
            25 * MINUTE,
            [
                "for about a quarter of an hour",
                "for the last twenty minutes or so",
                "for about fifteen minutes",
            ],
        ),
        (
            50 * MINUTE,
            [
                "for about half an hour",
                "for the last half hour or so",
                "for forty minutes or so",
            ],
        ),
        (
            100 * MINUTE,
            ["for about an hour", "for an hour or more", "for the last hour"],
        ),
        (
            4 * HOUR,
            [
                "for a couple of hours",
                "for two or three hours now",
                "since earlier on today",
            ],
        ),
        (
            12 * HOUR,
            [
                "for most of the day",
                "for several hours now",
                "for the best part of the day",
            ],
        ),
        (
            36 * HOUR,
            [
                "for the best part of a day",
                "for about a day now",
                "for well over twelve hours",
            ],
        ),
        (
            i64::MAX,
            [
                "for a couple of days now",
                "for days now",
                "since the day before yesterday",
            ],
        ),
    ];
    let phrases = BANDS
        .iter()
        .find(|(limit, _)| seconds <= *limit)
        .map_or(BANDS[7].1, |(_, phrases)| *phrases);
    phrases[variant % phrases.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_starts_on_the_first_of_june() {
        assert_eq!(clock(WINDOW_START).unwrap(), "00:00");
        assert_eq!(date(WINDOW_START).unwrap(), "1 June");
        assert_eq!(weekday_date(WINDOW_START).unwrap(), "Monday 1 June");
        assert_eq!(
            clock(WINDOW_START + 9 * HOUR + 40 * MINUTE + 59).unwrap(),
            "09:40"
        );
    }

    #[test]
    fn days_are_named_from_the_caller_side() {
        let now = WINDOW_START + 2 * DAY + 10 * HOUR;
        assert_eq!(day_phrase(now - HOUR, now, false).unwrap(), "");
        assert_eq!(day_phrase(now - HOUR, now, true).unwrap(), " today");
        assert_eq!(day_phrase(now - DAY, now, false).unwrap(), " yesterday");
        assert_eq!(day_phrase(now + DAY, now, false).unwrap(), " tomorrow");
        assert_eq!(
            day_phrase(now - 2 * DAY, now, false).unwrap(),
            " on Monday 1 June"
        );
        assert_eq!(ceil_to(61, 300), 300);
        assert_eq!(ceil_to(600, 300), 600);
        assert_eq!(floor_minute(119), 60);
        assert_eq!(elapsed_phrase(5 * MINUTE, 0), "for the last few minutes");
        assert_eq!(elapsed_phrase(3 * DAY, 1), "for days now");
    }
}
