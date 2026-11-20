//! What the paper reads: numbers as LaTeX macros, tables, and CSV for figures.
//!
//! A number is printed once, here, in one format per unit, so the paper never
//! holds a number typed by hand.

use crate::error::EvalError;
use crate::measures::Unit;
use std::collections::BTreeSet;
use std::fmt::Write;

/// A value in its unit: a fraction with three decimals (`0.712`) or a share in
/// percent with one decimal (`41.2`). A value that rounds to zero has no sign.
pub fn format_value(value: f64, unit: Unit) -> String {
    let text = match unit {
        Unit::Fraction => format!("{value:.3}"),
        Unit::Percent => format!("{:.1}", 100.0 * value),
    };
    match text.strip_prefix('-') {
        Some(digits) if digits.bytes().all(|b| b == b'0' || b == b'.') => digits.to_string(),
        _ => text,
    }
}

/// A difference in its unit with its sign written out: `+3.1`, `-0.012`, `0.0`.
/// For a share the unit is percentage points.
pub fn format_difference(value: f64, unit: Unit) -> String {
    let text = format_value(value, unit);
    if text.starts_with('-') || text.bytes().all(|b| b == b'0' || b == b'.') {
        text
    } else {
        format!("+{text}")
    }
}

/// A p-value with four decimals.
pub fn format_p(p: f64) -> String {
    format!("{p:.4}")
}

/// Text made safe for LaTeX text mode.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '_' | '%' | '&' | '#' | '$' | '{' | '}' => {
                out.push('\\');
                out.push(c);
            }
            '\\' => out.push_str("\\textbackslash{}"),
            '~' => out.push_str("\\textasciitilde{}"),
            '^' => out.push_str("\\textasciicircum{}"),
            _ => out.push(c),
        }
    }
    out
}

/// A `tabular` in the booktabs style, with comment lines above it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Table {
    /// Lines written as `% …` above the table: where the numbers come from.
    pub comments: Vec<String>,
    /// LaTeX lines written above the `tabular`, such as the `\providecommand`s its cells use.
    pub preamble: Vec<String>,
    /// The column specification, e.g. `lrrr`.
    pub columns: String,
    /// Header cells, already LaTeX.
    pub header: Vec<String>,
    /// Body rows of cells, already LaTeX.
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn render(&self) -> String {
        let mut out = String::new();
        for comment in &self.comments {
            let _ = writeln!(out, "% {comment}");
        }
        for line in &self.preamble {
            let _ = writeln!(out, "{line}");
        }
        let _ = writeln!(out, "\\begin{{tabular}}{{{}}}", self.columns);
        let _ = writeln!(out, "\\toprule");
        let _ = writeln!(out, "{} \\\\", self.header.join(" & "));
        let _ = writeln!(out, "\\midrule");
        for row in &self.rows {
            let _ = writeln!(out, "{} \\\\", row.join(" & "));
        }
        let _ = writeln!(out, "\\bottomrule");
        let _ = writeln!(out, "\\end{{tabular}}");
        out
    }
}

/// Whether a name can be one part of a number key or a file name: letters,
/// digits, `-`, `_` and `.` only (no `/`, which separates the parts of a key).
pub fn is_plain_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Whether a key can be used inside `\csname … \endcsname` in any common
/// LaTeX set-up: letters, digits and `. _ @ / + -` only.
pub fn is_number_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'@' | b'/' | b'+' | b'-'))
}

/// A file of numbers for the paper: each is defined once with `\newcommand`
/// under a key and read with `\cbnum{key}`; a key that does not exist stops
/// the LaTeX run instead of printing nothing.
#[derive(Clone, Debug, Default)]
pub struct Numbers {
    comments: Vec<String>,
    entries: Vec<(String, String)>,
    keys: BTreeSet<String>,
}

impl Numbers {
    pub fn new(comments: Vec<String>) -> Self {
        Self {
            comments,
            ..Self::default()
        }
    }

    /// Add a number. A key outside the safe characters, or used twice, is refused.
    pub fn push(&mut self, key: String, value: String) -> Result<(), EvalError> {
        if !is_number_key(&key) {
            return Err(EvalError::invalid(format!(
                "number key {key:?} may hold letters, digits and . _ @ / + - only"
            )));
        }
        if !self.keys.insert(key.clone()) {
            return Err(EvalError::invalid(format!("number key {key:?} is used twice")));
        }
        self.entries.push((key, value));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        for comment in &self.comments {
            let _ = writeln!(out, "% {comment}");
        }
        out.push_str(concat!(
            "% Use \\cbnum{<key>}. A key that is not defined below stops the LaTeX run.\n",
            "\\providecommand{\\cbnum}[1]{\\ifcsname cbnum/#1\\endcsname\\csname cbnum/#1\\endcsname",
            "\\else\\errmessage{cb-eval: no number is named #1}\\fi}\n",
        ));
        for (key, value) in &self.entries {
            let _ = writeln!(
                out,
                "\\expandafter\\newcommand\\csname cbnum/{key}\\endcsname{{{value}}}"
            );
        }
        out
    }
}

/// One field of a CSV line: quoted when it holds a comma, a quote or a line break.
pub fn csv_field(text: &str) -> String {
    if text.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", text.replace('"', "\"\""))
    } else {
        text.to_string()
    }
}

/// One CSV line with its line break.
pub fn csv_line<S: AsRef<str>>(fields: &[S]) -> String {
    let mut line = fields
        .iter()
        .map(|f| csv_field(f.as_ref()))
        .collect::<Vec<_>>()
        .join(",");
    line.push('\n');
    line
}

/// A number for a CSV cell: the shortest decimal that reads back to the same value.
pub fn csv_number(value: f64) -> String {
    value.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_are_printed_in_their_unit() {
        assert_eq!(format_value(0.7123, Unit::Fraction), "0.712");
        assert_eq!(format_value(0.7126, Unit::Fraction), "0.713");
        assert_eq!(format_value(1.0, Unit::Fraction), "1.000");
        assert_eq!(format_value(0.4124, Unit::Percent), "41.2");
        assert_eq!(format_value(0.4126, Unit::Percent), "41.3");
        assert_eq!(format_value(1.0, Unit::Percent), "100.0");
        assert_eq!(format_value(0.0, Unit::Percent), "0.0");
        // A tiny negative value does not print as "-0.0".
        assert_eq!(format_value(-0.0001, Unit::Percent), "0.0");
        assert_eq!(format_value(-0.0001, Unit::Fraction), "0.000");
        assert_eq!(format_value(-0.25, Unit::Percent), "-25.0");
    }

    #[test]
    fn differences_carry_their_sign() {
        assert_eq!(format_difference(0.031, Unit::Percent), "+3.1");
        assert_eq!(format_difference(-0.012, Unit::Fraction), "-0.012");
        assert_eq!(format_difference(0.0, Unit::Percent), "0.0");
        assert_eq!(format_difference(-0.00001, Unit::Fraction), "0.000");
        assert_eq!(format_difference(0.1234, Unit::Fraction), "+0.123");
        assert_eq!(format_p(1.0 / 10_001.0), "0.0001");
        assert_eq!(format_p(0.04321), "0.0432");
        assert_eq!(format_p(1.0), "1.0000");
    }

    #[test]
    fn text_is_escaped_for_latex() {
        assert_eq!(escape("rrf_td-k60-h1800"), "rrf\\_td-k60-h1800");
        assert_eq!(escape("hit@4 (%)"), "hit@4 (\\%)");
        assert_eq!(escape("a&b#c$d{e}"), "a\\&b\\#c\\$d\\{e\\}");
        assert_eq!(
            escape("x\\y~z^"),
            "x\\textbackslash{}y\\textasciitilde{}z\\textasciicircum{}"
        );
    }

    #[test]
    fn a_table_renders_as_a_booktabs_tabular() {
        let table = Table {
            comments: vec!["test split".to_string()],
            preamble: vec!["\\providecommand{\\x}{y}".to_string()],
            columns: "lr".to_string(),
            header: vec!["Method".to_string(), "hit@4 (\\%)".to_string()],
            rows: vec![
                vec!["\\texttt{dense}".to_string(), "68.5".to_string()],
                vec!["\\texttt{product}".to_string(), "71.0".to_string()],
            ],
        };
        assert_eq!(
            table.render(),
            "% test split\n\\providecommand{\\x}{y}\n\\begin{tabular}{lr}\n\\toprule\nMethod & hit@4 (\\%) \\\\\n\\midrule\n\
             \\texttt{dense} & 68.5 \\\\\n\\texttt{product} & 71.0 \\\\\n\\bottomrule\n\\end{tabular}\n"
        );
    }

    #[test]
    fn numbers_are_defined_once_under_safe_keys() {
        let mut numbers = Numbers::new(vec!["Written by cb-eval.".to_string()]);
        numbers
            .push(
                "synth-42/product/changing/current@1".to_string(),
                "41.2".to_string(),
            )
            .unwrap();
        numbers
            .push(
                "synth-42/rrf_td-k60-h1800/changing/ndcg@10".to_string(),
                "0.512".to_string(),
            )
            .unwrap();
        assert!(
            numbers
                .push("synth-42/product/changing/current@1".to_string(), "9".to_string())
                .is_err()
        );
        assert!(numbers.push("has space".to_string(), "1".to_string()).is_err());
        assert!(numbers.push("back\\slash".to_string(), "1".to_string()).is_err());
        assert!(numbers.push("colon:key".to_string(), "1".to_string()).is_err());
        assert!(numbers.push(String::new(), "1".to_string()).is_err());
        assert_eq!(numbers.len(), 2);
        assert_eq!(numbers.get("synth-42/product/changing/current@1"), Some("41.2"));
        let text = numbers.render();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "% Written by cb-eval.");
        assert!(lines[2].starts_with("\\providecommand{\\cbnum}[1]{\\ifcsname cbnum/#1\\endcsname"));
        assert_eq!(
            lines[3],
            "\\expandafter\\newcommand\\csname cbnum/synth-42/product/changing/current@1\\endcsname{41.2}"
        );
        assert_eq!(lines.len(), 5);
    }

    #[test]
    fn csv_fields_are_quoted_when_they_need_to_be() {
        assert_eq!(csv_field("dense"), "dense");
        assert_eq!(csv_field("rrf_td(k=60,h=1800)"), "\"rrf_td(k=60,h=1800)\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
        assert_eq!(csv_line(&["a", "b,c", ""]), "a,\"b,c\",\n");
        assert_eq!(csv_number(0.1), "0.1");
        assert_eq!(csv_number(1.0 / 3.0), "0.3333333333333333");
    }
}
