//! Comma-separated values as RFC 4180 writes them: a field may be quoted, a
//! quote inside a quoted field is doubled, and a quoted field may hold commas
//! and line breaks.

use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum CsvError {
    #[error("line {line}: a quoted field is never closed")]
    Unclosed { line: usize },
    #[error("line {line}: text after the closing quote of a field")]
    AfterQuote { line: usize },
    #[error("no column named {0:?}")]
    NoColumn(String),
    #[error("row {row}: {found} fields where the header has {expected}")]
    Width {
        row: usize,
        found: usize,
        expected: usize,
    },
}

/// Every row, the header included. A line with nothing on it is not a row.
pub fn parse(text: &str) -> Result<Vec<Vec<String>>, CsvError> {
    let mut rows = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let (mut line, mut opened_on) = (1, 1);
    let (mut in_quotes, mut was_quoted) = (false, false);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => {
                    in_quotes = false;
                    was_quoted = true;
                }
                _ => {
                    line += usize::from(c == '\n');
                    field.push(c);
                }
            }
            continue;
        }
        match c {
            '"' if field.is_empty() && !was_quoted => {
                in_quotes = true;
                opened_on = line;
            }
            ',' => {
                row.push(std::mem::take(&mut field));
                was_quoted = false;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                row.push(std::mem::take(&mut field));
                if !(row.len() == 1 && row[0].is_empty() && !was_quoted) {
                    rows.push(std::mem::take(&mut row));
                }
                row.clear();
                was_quoted = false;
                line += 1;
            }
            _ if was_quoted => return Err(CsvError::AfterQuote { line }),
            _ => field.push(c),
        }
    }
    if in_quotes {
        return Err(CsvError::Unclosed { line: opened_on });
    }
    if !field.is_empty() || !row.is_empty() || was_quoted {
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

/// A parsed file whose first row names the columns.
pub struct Table {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl Table {
    pub fn parse(text: &str) -> Result<Self, CsvError> {
        let mut rows = parse(text.strip_prefix('\u{feff}').unwrap_or(text))?.into_iter();
        let header = rows.next().unwrap_or_default();
        let rows: Vec<Vec<String>> = rows.collect();
        for (i, row) in rows.iter().enumerate() {
            if row.len() != header.len() {
                return Err(CsvError::Width {
                    row: i + 2,
                    found: row.len(),
                    expected: header.len(),
                });
            }
        }
        Ok(Self { header, rows })
    }

    /// Where the column named `name` is.
    pub fn column(&self, name: &str) -> Result<usize, CsvError> {
        self.header
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| CsvError::NoColumn(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_fields_hold_commas_quotes_and_line_breaks() {
        let text = "a,b,c\r\n1,\"x, \"\"y\"\"\",\"two\nlines\"\n\n2,,\"\"\n";
        let rows = parse(text).unwrap();
        assert_eq!(
            rows,
            vec![
                vec!["a", "b", "c"],
                vec!["1", "x, \"y\"", "two\nlines"],
                vec!["2", "", ""],
            ]
        );
        assert_eq!(parse("last,row").unwrap(), vec![vec!["last", "row"]]);
    }

    #[test]
    fn broken_quoting_is_an_error_with_its_line() {
        assert_eq!(
            parse("a\n\"open,\nnever closed"),
            Err(CsvError::Unclosed { line: 2 })
        );
        assert_eq!(parse("a\n\"x\"y,z"), Err(CsvError::AfterQuote { line: 2 }));
    }

    #[test]
    fn a_table_finds_its_columns_and_checks_every_row_width() {
        let table = Table::parse("\u{feff}Title,Link\nOne,https://x/1\n").unwrap();
        assert_eq!(table.column("Link"), Ok(1));
        assert_eq!(table.column("Missing"), Err(CsvError::NoColumn("Missing".into())));
        assert_eq!(table.rows, vec![vec!["One", "https://x/1"]]);
        assert!(matches!(
            Table::parse("a,b\n1\n"),
            Err(CsvError::Width {
                row: 2,
                found: 1,
                expected: 2
            })
        ));
    }
}
