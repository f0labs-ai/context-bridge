//! `cb-eval speech-trips`: trips from the end of speech to the first hit, put together
//! per call from stage times measured on that call (DESIGN.md section 7).
//!
//! The stages run apart on one machine, each one item at a time on an idle GPU:
//! speech-to-text of a voice line (`transcribe_many`), masking of
//! the line's call (`mask_transcripts`, the median of its repeats), extraction of the
//! masked call (`cb-run extract --concurrency 1`), and one upload timed to its first hit
//! (`cb-run trips`). A trip is the voice line's speech-to-text, then its call's masking
//! and extraction, then the next upload in the order the upload run took its records:
//! the i-th voice line that has an extraction takes the i-th upload. Every mark is the
//! sum of the stage times before it, so `cb-eval trips` takes percentiles over whole
//! trips; nothing adds up percentiles. Putting the stages together assumes one stage's
//! time does not depend on another's.
//!
//! An upload the search never returned (`never_found` in the upload run's
//! `failures.jsonl`) keeps its place: the voice line that takes it gets a trip without a
//! first hit and a `never_found` line in `<out>/failures.jsonl`, which `cb-eval trips`
//! counts as slower than every timed trip.

use crate::error::EvalError;
use crate::files::{TRIPS, read_jsonl_file, sha256_of_file, write_json_file, write_jsonl_file};
use cb_types::PipelineTrip;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Where the trips were put together from, and how: `<out>/composed.json`.
pub const COMPOSED: &str = "composed.json";
/// A run's failures, one per line; `kind` `never_found` marks a record never returned.
pub const FAILURES: &str = "failures.jsonl";
pub const NEVER_FOUND: &str = "never_found";

#[derive(Clone, Debug)]
pub struct SpeechTripsArgs {
    /// `transcribe_many`'s output: `{"id", "decode_ms", …}` per voice line.
    pub stt: PathBuf,
    /// The voice lines: `{"id", "transcript", …}` per line.
    pub voice: PathBuf,
    /// `mask_transcripts`'s timings: `{"id", "mask_us", …}` per call.
    pub mask: PathBuf,
    /// `cb-run extract`'s extractions, made one call at a time.
    pub extract: PathBuf,
    /// `cb-run trips`'s trips: `written_ms` and `first_hit_ms` from the hand-over.
    pub upload: PathBuf,
    /// The records the upload run timed, in the order it took them (its data set's
    /// `records.jsonl`). Needed when the run's `failures.jsonl`, beside `upload`, lists
    /// records never found: it gives them their places among the uploads.
    pub upload_records: Option<PathBuf>,
    /// Where `trips.jsonl`, `failures.jsonl` and `composed.json` are written.
    pub out: PathBuf,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FailureLine {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct RecordId {
    id: String,
}

/// One upload, in the order the upload run took them.
enum Upload<'a> {
    Timed(&'a UploadLine),
    NeverFound(&'a str),
}

#[derive(Debug, Deserialize)]
struct SttLine {
    id: String,
    decode_ms: f64,
}

#[derive(Debug, Deserialize)]
struct VoiceLine {
    id: String,
    transcript: String,
}

#[derive(Debug, Deserialize)]
struct MaskLine {
    id: String,
    mask_us: f64,
}

#[derive(Debug, Deserialize)]
struct ExtractLine {
    transcript: String,
    latency_ms: f64,
}

#[derive(Debug, Deserialize)]
struct UploadLine {
    record: String,
    written_ms: Option<f64>,
    first_hit_ms: Option<f64>,
}

/// What `composed.json` holds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Composed {
    /// File name → sha256 of every input.
    pub inputs: BTreeMap<String, String>,
    /// How a trip is put together, in words.
    pub rule: String,
    pub trips: usize,
    /// Of `trips`, the ones whose upload the search never returned (no first hit).
    #[serde(default)]
    pub trips_never_found: usize,
    /// Voice lines without an extraction of their call (not made into trips).
    pub voice_lines_without_extraction: usize,
    /// Uploads left over once every voice line had one.
    pub uploads_left_over: usize,
}

/// One trip, from the stages of one call and one upload; without an upload that was
/// timed (one never found), it ends at the hand-over. The record is the voice line's id.
fn trip(
    voice: &str,
    stt_ms: f64,
    mask_ms: f64,
    extract_ms: f64,
    upload: Option<&UploadLine>,
) -> PipelineTrip {
    let transcript = stt_ms;
    let masked = transcript + mask_ms;
    let extracted = masked + extract_ms;
    PipelineTrip {
        record: voice.to_string(),
        transcript_ms: Some(transcript),
        masked_ms: Some(masked),
        extracted_ms: Some(extracted),
        written_ms: upload.and_then(|u| u.written_ms).map(|w| extracted + w),
        first_hit_ms: upload.and_then(|u| u.first_hit_ms).map(|f| extracted + f),
    }
}

/// The uploads in the order the run took them: its timed trips and, placed by
/// `upload_records`, the records it never found.
fn uploads_in_order<'a>(
    usable: &[&'a UploadLine],
    never_found: &'a [String],
    upload_records: Option<&Path>,
    inputs: &mut BTreeMap<String, String>,
) -> Result<Vec<Upload<'a>>, EvalError> {
    if never_found.is_empty() {
        return Ok(usable.iter().map(|u| Upload::Timed(u)).collect());
    }
    let order = upload_records.ok_or_else(|| {
        EvalError::invalid(format!(
            "speech-trips: the upload run never found {} record(s); --upload-records must name the records it \
             timed, in its order, to give them their places",
            never_found.len()
        ))
    })?;
    inputs.insert("upload_records".to_string(), sha256_of_file(order)?);
    let timed: HashMap<&str, &UploadLine> = usable.iter().map(|u| (u.record.as_str(), *u)).collect();
    let lost: HashSet<&str> = never_found.iter().map(String::as_str).collect();
    let mut placed = HashSet::new();
    let mut out = Vec::with_capacity(timed.len() + lost.len());
    for record in read_jsonl_file::<RecordId>(order)? {
        let id = record.id.as_str();
        if let Some(upload) = timed.get(id) {
            out.push(Upload::Timed(upload));
        } else if let Some(lost_id) = lost.get(id) {
            out.push(Upload::NeverFound(lost_id));
        } else {
            continue;
        }
        if !placed.insert(id.to_string()) {
            return Err(EvalError::invalid(format!(
                "{}: record {id:?} appears twice",
                order.display()
            )));
        }
    }
    if let Some(missing) = timed.keys().chain(lost.iter()).find(|id| !placed.contains(**id)) {
        return Err(EvalError::invalid(format!(
            "{}: the upload run timed record {missing:?}, which it does not list",
            order.display()
        )));
    }
    Ok(out)
}

fn by_id<T>(items: Vec<T>, id: impl Fn(&T) -> &str, file: &Path) -> Result<HashMap<String, T>, EvalError> {
    let mut map = HashMap::with_capacity(items.len());
    for item in items {
        let key = id(&item).to_string();
        if map.contains_key(&key) {
            return Err(EvalError::invalid(format!(
                "{}: {key:?} appears twice",
                file.display()
            )));
        }
        map.insert(key, item);
    }
    Ok(map)
}

/// Put the trips together and write them.
pub fn run(args: &SpeechTripsArgs) -> Result<Composed, EvalError> {
    let mut inputs = BTreeMap::new();
    for (name, path) in [
        ("stt", &args.stt),
        ("voice", &args.voice),
        ("mask", &args.mask),
        ("extract", &args.extract),
        ("upload", &args.upload),
    ] {
        inputs.insert(name.to_string(), sha256_of_file(path)?);
    }
    let stt = by_id(read_jsonl_file::<SttLine>(&args.stt)?, |l| &l.id, &args.stt)?;
    let mask = by_id(read_jsonl_file::<MaskLine>(&args.mask)?, |l| &l.id, &args.mask)?;
    let extract = by_id(
        read_jsonl_file::<ExtractLine>(&args.extract)?,
        |l| &l.transcript,
        &args.extract,
    )?;
    let voice: Vec<VoiceLine> = read_jsonl_file(&args.voice)?;
    let uploads: Vec<UploadLine> = read_jsonl_file(&args.upload)?;
    let usable: Vec<&UploadLine> = uploads
        .iter()
        .filter(|u| u.written_ms.is_some() && u.first_hit_ms.is_some())
        .collect();
    let upload_failures = args.upload.with_file_name(FAILURES);
    let never_found: Vec<String> = if upload_failures.exists() {
        inputs.insert("upload_failures".to_string(), sha256_of_file(&upload_failures)?);
        read_jsonl_file::<FailureLine>(&upload_failures)?
            .into_iter()
            .filter(|f| f.kind == NEVER_FOUND)
            .map(|f| f.id)
            .collect()
    } else {
        Vec::new()
    };
    let ordered = uploads_in_order(&usable, &never_found, args.upload_records.as_deref(), &mut inputs)?;

    let mut trips = Vec::new();
    let mut failures = Vec::new();
    let mut without_extraction = 0;
    for line in &voice {
        let Some(extracted) = extract.get(&line.transcript) else {
            without_extraction += 1;
            continue;
        };
        let wrong = |what: &str| EvalError::invalid(format!("voice line {}: {what}", line.id));
        let spoken = stt.get(&line.id).ok_or_else(|| wrong("no speech-to-text time"))?;
        let masked = mask
            .get(&line.transcript)
            .ok_or_else(|| wrong(&format!("no masking time for call {}", line.transcript)))?;
        let upload = ordered.get(trips.len()).ok_or_else(|| {
            wrong(&format!(
                "fewer uploads ({}) than voice lines with an extraction",
                ordered.len()
            ))
        })?;
        let times = [spoken.decode_ms, masked.mask_us / 1000.0, extracted.latency_ms];
        if times.iter().any(|t| !t.is_finite() || *t < 0.0) {
            return Err(wrong("a stage time is not a number of milliseconds"));
        }
        let timed = match upload {
            Upload::Timed(u) => Some(*u),
            Upload::NeverFound(record) => {
                failures.push(FailureLine {
                    id: line.id.clone(),
                    kind: NEVER_FOUND.to_string(),
                    message: format!("its upload, record {record}, was never found by the search"),
                });
                None
            }
        };
        trips.push(trip(&line.id, times[0], times[1], times[2], timed));
    }
    write_jsonl_file(&args.out.join(TRIPS), &trips)?;
    write_jsonl_file(&args.out.join(FAILURES), &failures)?;
    let composed = Composed {
        inputs,
        rule: "The i-th voice line whose call has an extraction: its speech-to-text time, its call's masking \
               time (median of repeats), its call's extraction time (one call at a time), then the i-th upload \
               in the order the upload run took them: its hand-over to written and to first hit, or, for an \
               upload never found, no first hit. Marks are sums of the stage times before them."
            .to_string(),
        trips: trips.len(),
        trips_never_found: failures.len(),
        voice_lines_without_extraction: without_extraction,
        uploads_left_over: ordered.len() - trips.len(),
    };
    write_json_file(&args.out.join(COMPOSED), &composed)?;
    Ok(composed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::test_dir;
    use std::fs;

    fn write(path: &Path, lines: &[&str]) {
        fs::write(path, lines.join("\n") + "\n").unwrap();
    }

    #[test]
    fn a_trip_is_its_calls_stage_times_in_order_then_one_upload() {
        let dir = test_dir("speech-trips");
        let at = |name: &str| dir.join(name);
        write(
            &at("stt.jsonl"),
            &[
                r#"{"id":"vl-0","decode_ms":80.0}"#,
                r#"{"id":"vl-1","decode_ms":90.0}"#,
                r#"{"id":"vl-2","decode_ms":70.0}"#,
            ],
        );
        write(
            &at("voice.jsonl"),
            &[
                r#"{"id":"vl-0","transcript":"c0"}"#,
                r#"{"id":"vl-1","transcript":"c1"}"#,
                r#"{"id":"vl-2","transcript":"c2"}"#,
            ],
        );
        write(
            &at("mask.jsonl"),
            &[
                r#"{"id":"c0","mask_us":500.0}"#,
                r#"{"id":"c1","mask_us":250.0}"#,
                r#"{"id":"c2","mask_us":100.0}"#,
            ],
        );
        // c1 was not extracted one at a time: vl-1 makes no trip and takes no upload.
        write(
            &at("extract.jsonl"),
            &[
                r#"{"transcript":"c0","latency_ms":2000.0}"#,
                r#"{"transcript":"c2","latency_ms":1500.0}"#,
            ],
        );
        write(
            &at("upload.jsonl"),
            &[
                r#"{"record":"u0","written_ms":40.0,"first_hit_ms":90.0}"#,
                r#"{"record":"u-lost","written_ms":40.0}"#,
                r#"{"record":"u1","written_ms":45.0,"first_hit_ms":100.0}"#,
                r#"{"record":"u2","written_ms":50.0,"first_hit_ms":110.0}"#,
            ],
        );
        let args = SpeechTripsArgs {
            stt: at("stt.jsonl"),
            voice: at("voice.jsonl"),
            mask: at("mask.jsonl"),
            extract: at("extract.jsonl"),
            upload: at("upload.jsonl"),
            upload_records: None,
            out: at("out"),
        };
        let composed = run(&args).unwrap();
        assert_eq!(
            (
                composed.trips,
                composed.voice_lines_without_extraction,
                composed.uploads_left_over
            ),
            (2, 1, 1)
        );
        let trips: Vec<PipelineTrip> = read_jsonl_file(&at("out").join(TRIPS)).unwrap();
        // vl-0: 80 + 0.5 = 80.5 masked, + 2000 = 2080.5 extracted, + 40 written, + 90 first hit.
        assert_eq!(trips[0].record, "vl-0");
        assert_eq!(
            (
                trips[0].masked_ms,
                trips[0].extracted_ms,
                trips[0].written_ms,
                trips[0].first_hit_ms
            ),
            (Some(80.5), Some(2_080.5), Some(2_120.5), Some(2_170.5))
        );
        // vl-2 takes the second usable upload (u1): the one without a first hit is passed over.
        assert_eq!(
            (trips[1].record.as_str(), trips[1].first_hit_ms),
            ("vl-2", Some(70.0 + 0.1 + 1_500.0 + 100.0))
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn an_upload_never_found_keeps_its_place_and_its_trip_has_no_first_hit() {
        let dir = test_dir("speech-trips-never-found");
        let at = |name: &str| dir.join(name);
        write(
            &at("stt.jsonl"),
            &[
                r#"{"id":"vl-0","decode_ms":80.0}"#,
                r#"{"id":"vl-1","decode_ms":90.0}"#,
            ],
        );
        write(
            &at("voice.jsonl"),
            &[
                r#"{"id":"vl-0","transcript":"c0"}"#,
                r#"{"id":"vl-1","transcript":"c1"}"#,
            ],
        );
        write(
            &at("mask.jsonl"),
            &[r#"{"id":"c0","mask_us":500.0}"#, r#"{"id":"c1","mask_us":250.0}"#],
        );
        write(
            &at("extract.jsonl"),
            &[
                r#"{"transcript":"c0","latency_ms":2000.0}"#,
                r#"{"transcript":"c1","latency_ms":1500.0}"#,
            ],
        );
        // The run took u0, then u-lost (never found), then u1.
        write(
            &at("upload.jsonl"),
            &[
                r#"{"record":"u0","written_ms":40.0,"first_hit_ms":90.0}"#,
                r#"{"record":"u1","written_ms":45.0,"first_hit_ms":100.0}"#,
            ],
        );
        write(
            &at("failures.jsonl"),
            &[r#"{"id":"u-lost","kind":"never_found","message":"not found within 30000 ms"}"#],
        );
        write(
            &at("records.jsonl"),
            &[r#"{"id":"u0"}"#, r#"{"id":"u-lost"}"#, r#"{"id":"u1"}"#],
        );
        let mut args = SpeechTripsArgs {
            stt: at("stt.jsonl"),
            voice: at("voice.jsonl"),
            mask: at("mask.jsonl"),
            extract: at("extract.jsonl"),
            upload: at("upload.jsonl"),
            upload_records: None,
            out: at("out"),
        };
        // Without the order, the records never found have no place: refused.
        let error = run(&args).unwrap_err().to_string();
        assert!(error.contains("--upload-records"), "{error}");
        args.upload_records = Some(at("records.jsonl"));
        let composed = run(&args).unwrap();
        assert_eq!(
            (
                composed.trips,
                composed.trips_never_found,
                composed.uploads_left_over
            ),
            (2, 1, 1)
        );
        let trips: Vec<PipelineTrip> = read_jsonl_file(&at("out").join(TRIPS)).unwrap();
        // vl-1 takes u-lost: its stages up to the hand-over, no written or first-hit mark.
        assert_eq!(
            (
                trips[1].record.as_str(),
                trips[1].extracted_ms,
                trips[1].first_hit_ms
            ),
            ("vl-1", Some(90.0 + 0.25 + 1_500.0), None)
        );
        let failures: Vec<FailureLine> = read_jsonl_file(&at("out").join(FAILURES)).unwrap();
        assert_eq!(failures.len(), 1);
        assert_eq!(
            (failures[0].id.as_str(), failures[0].kind.as_str()),
            ("vl-1", NEVER_FOUND)
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
