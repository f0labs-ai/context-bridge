//! Small text helpers: word counts, whitespace and case folding, key terms and
//! the `{name}` templates every content pool is written in.

use super::SynthError;

/// Words as the corpus counts them: runs of non-whitespace characters.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Tokens as the specification counts them: words × 1.33.
pub fn tokens_of_words(words: usize) -> f64 {
    words as f64 * 1.33
}

/// The text with every run of whitespace turned into one space, ends trimmed.
pub fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Case-folded and whitespace-normalised: the form key terms are matched in.
pub fn fold(text: &str) -> String {
    collapse_whitespace(&text.to_lowercase())
}

/// Whether `term`, folded, occurs in `folded_text` (which the caller folded once).
pub fn contains_folded(folded_text: &str, term: &str) -> bool {
    let term = fold(term);
    !term.is_empty() && folded_text.contains(&term)
}

/// A phrase as a key term: the phrase without a leading article, so that "a" or
/// "the" in front of it in running speech does not matter.
pub fn key_form(phrase: &str) -> String {
    let phrase = phrase.trim();
    for article in ["a ", "an ", "the "] {
        if let Some(rest) = phrase.strip_prefix(article) {
            return rest.to_string();
        }
    }
    phrase.to_string()
}

/// The text with its first letter in upper case.
pub fn capitalize_first(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// Values for the `{name}` placeholders of a template.
#[derive(Clone, Debug, Default)]
pub struct Vars {
    items: Vec<(&'static str, String)>,
}

impl Vars {
    /// Set `name` to `value`, replacing an earlier value.
    pub fn set(&mut self, name: &'static str, value: impl Into<String>) {
        let value = value.into();
        match self.items.iter_mut().find(|(k, _)| *k == name) {
            Some(slot) => slot.1 = value,
            None => self.items.push((name, value)),
        }
    }

    /// This set of values with `name` set to `value`.
    pub fn with(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.set(name, value);
        self
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.items
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
}

fn template_error(template: &str, problem: String) -> SynthError {
    SynthError::Template {
        template: template.to_string(),
        problem,
    }
}

/// Replace every `{name}` of `template` with its value. A placeholder without a
/// value, or an unclosed brace, is an error rather than a silent gap.
pub fn fill(template: &str, vars: &Vars) -> Result<String, SynthError> {
    let mut out = String::with_capacity(template.len() + 48);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        let after = &rest[open + 1..];
        let close = after
            .find('}')
            .ok_or_else(|| template_error(template, "unclosed placeholder".to_string()))?;
        let name = &after[..close];
        let value = vars
            .get(name)
            .ok_or_else(|| template_error(template, format!("no value for {{{name}}}")))?;
        out.push_str(value);
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// The placeholder names a template uses, in order of appearance.
pub fn placeholders(template: &str) -> Vec<&str> {
    let mut names = Vec::new();
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        let after = &rest[open + 1..];
        match after.find('}') {
            Some(close) => {
                names.push(&after[..close]);
                rest = &after[close + 1..];
            }
            None => break,
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_template_is_filled_and_a_missing_value_is_an_error() {
        let vars = Vars::default()
            .with("service", "home internet")
            .with("region", "Quillhaven");
        assert_eq!(
            fill("Is {service} down in {region}?", &vars).unwrap(),
            "Is home internet down in Quillhaven?"
        );
        assert!(fill("Is {service} down in {area}?", &vars).is_err());
        assert!(fill("Is {service down", &vars).is_err());
        assert_eq!(placeholders("{a} and {b}."), vec!["a", "b"]);
    }

    #[test]
    fn key_terms_match_whatever_the_case_and_spacing() {
        assert_eq!(key_form("a failed line card"), "failed line card");
        assert_eq!(key_form("the hub"), "hub");
        assert_eq!(key_form("an outage"), "outage");
        let folded = fold("The cause is a Failed   line\ncard.");
        assert!(contains_folded(&folded, "failed line card"));
        assert!(!contains_folded(&folded, "line cards"));
        assert_eq!(collapse_whitespace("  a \t b\n"), "a b");
        assert_eq!(capitalize_first("the app"), "The app");
        assert_eq!(word_count(" one two  three "), 3);
    }
}
