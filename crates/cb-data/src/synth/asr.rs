//! Speech-recognition-style errors (docs/SYNTH_SPEC.md section 3.1): each word
//! is altered with probability `p`. A word with an entry in the fixed
//! sound-alike table is replaced by a sound-alike; any other word is dropped
//! (30%), doubled or preceded by a filler (20%), or left as it was (50%).
//! Ids, states and turn times stay the same, so results line up.

use super::draw::chance;
use super::pools::AsrTable;
use cb_types::Transcript;
use rand::Rng;
use rand::seq::IndexedRandom;
use rand_chacha::ChaCha20Rng;

/// The file name for rate `p`: `transcripts.asr05.jsonl` for 0.05.
pub fn file_name(p: f64) -> String {
    format!("transcripts.asr{:02}.jsonl", (p * 100.0).round() as u32)
}

/// A copy of the transcripts with speech-recognition-style errors at rate `p`.
pub fn with_speech_errors(
    transcripts: &[Transcript],
    table: &AsrTable,
    p: f64,
    rng: &mut ChaCha20Rng,
) -> Vec<Transcript> {
    transcripts
        .iter()
        .map(|t| {
            let mut out = t.clone();
            out.asr_error_rate = p as f32;
            for turn in &mut out.turns {
                turn.text = alter_words(&turn.text, table, p, rng);
            }
            out
        })
        .collect()
}

/// Split a word into leading punctuation, its letters and trailing punctuation.
fn split_word(word: &str) -> (&str, &str, &str) {
    let start = word.find(|c: char| c.is_alphanumeric()).unwrap_or(word.len());
    let end = word
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_alphanumeric())
        .map_or(start, |(i, c)| i + c.len_utf8());
    if end <= start {
        return (word, "", "");
    }
    (&word[..start], &word[start..end], &word[end..])
}

/// `replacement` written with the capital first letter of `original`, if it had one.
fn match_case(original: &str, replacement: &str) -> String {
    if original.chars().next().is_some_and(char::is_uppercase) {
        super::text::capitalize_first(replacement)
    } else {
        replacement.to_string()
    }
}

fn alter_words(text: &str, table: &AsrTable, p: f64, rng: &mut ChaCha20Rng) -> String {
    let mut out: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        if !chance(rng, p) {
            out.push(word.to_string());
            continue;
        }
        let (lead, core, trail) = split_word(word);
        if let Some(alike) = table
            .sound_alikes
            .get(&core.to_lowercase())
            .and_then(|alikes| alikes.choose(rng))
        {
            out.push(format!("{lead}{}{trail}", match_case(core, alike)));
            continue;
        }
        let roll: f64 = rng.random();
        if roll < 0.3 {
            // Dropped.
        } else if roll < 0.5 {
            if rng.random::<bool>() {
                out.push(format!("{lead}{core}"));
                out.push(word.to_string());
            } else {
                let filler = table.fillers.choose(rng).map_or("um", String::as_str);
                out.push(filler.to_string());
                out.push(word.to_string());
            }
        } else {
            out.push(word.to_string());
        }
    }
    if out.is_empty() {
        return text.split_whitespace().next().unwrap_or_default().to_string();
    }
    out.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use std::collections::BTreeMap;

    fn table() -> AsrTable {
        let mut sound_alikes = BTreeMap::new();
        sound_alikes.insert("there".to_string(), vec!["their".to_string()]);
        AsrTable {
            sound_alikes,
            fillers: vec!["um".to_string()],
        }
    }

    #[test]
    fn words_are_split_from_their_punctuation_and_keep_their_case() {
        assert_eq!(split_word("\"There,"), ("\"", "There", ","));
        assert_eq!(split_word("..."), ("...", "", ""));
        assert_eq!(match_case("There", "their"), "Their");
        assert_eq!(file_name(0.05), "transcripts.asr05.jsonl");
        assert_eq!(file_name(0.15), "transcripts.asr15.jsonl");
    }

    #[test]
    fn rate_zero_changes_nothing_and_rate_one_changes_every_table_word() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let text = "There, it is over there.";
        assert_eq!(alter_words(text, &table(), 0.0, &mut rng), text);
        let altered = alter_words("There there there.", &table(), 1.0, &mut rng);
        assert_eq!(altered, "Their their their.");
    }
}
