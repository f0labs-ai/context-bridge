//! GitHub's documentation as standing knowledge (docs/PUBLIC_DATA.md section 1.1).
//!
//! An article arrives from the site's article API as Markdown with the
//! templating already resolved. It is cut at its second-level headings; each
//! section becomes one record of plain prose, or several when it is long.
//! Tables and code blocks are dropped: they are not something to say aloud.

use super::raw::{self, ArticleFile, DocArea, DocPath, RawError};
use crate::public::text::{plain, squash, word_count};
use cb_types::{Record, Tier, UnixSeconds, fnv1a64, ids};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;
use thiserror::Error;

pub const RECORD_SOURCE: &str = "github-docs";

/// `written_at` of every documentation record: 2018-06-01T00:00:00Z, before
/// the first kept incident, so the documentation is always there.
pub const WRITTEN_AT: UnixSeconds = 1_527_811_200;

/// The only kind of page that holds prose; the others are lists of links.
const ARTICLE: &str = "article";

/// Sections that only list links to other articles.
const LINK_LIST_HEADINGS: &[&str] = &["further reading"];

pub const LEFT_OUT_NOT_ARTICLE: &str = "a page that only lists other pages";
pub const LEFT_OUT_NO_PROSE: &str = "an article with no section of the least length";

/// The size of a record's text, in words.
#[derive(Clone, Copy, Debug)]
pub struct DocRules {
    pub min_words: usize,
    pub max_words: usize,
}

impl Default for DocRules {
    fn default() -> Self {
        Self {
            min_words: 40,
            max_words: 300,
        }
    }
}

#[derive(Debug, Error)]
pub enum DocsError {
    #[error(transparent)]
    Raw(#[from] RawError),
    #[error(transparent)]
    Id(#[from] cb_types::ContractError),
    #[error("{path}: {source}")]
    List {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("articles {first} and {second} get the same record id {id:?}")]
    SameId {
        id: String,
        first: String,
        second: String,
    },
    #[error("{0}: not in one of the kept areas of the documentation")]
    NoArea(String),
}

/// One second-level section of an article.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    /// Empty for the part before the first second-level heading.
    pub heading: String,
    /// Paragraphs, list items and lower headings, each as one line of plain text.
    pub blocks: Vec<String>,
}

/// A line with its quote marks (`> `) removed.
fn unquote(line: &str) -> &str {
    let mut rest = line.trim_start();
    while let Some(inner) = rest.strip_prefix('>') {
        rest = inner.trim_start();
    }
    rest
}

/// `(level, text)` of a heading line.
fn heading(line: &str) -> Option<(usize, &str)> {
    let level = line.bytes().take_while(|b| *b == b'#').count();
    let rest = &line[level..];
    ((1..=6).contains(&level) && rest.starts_with(' '))
        .then(|| (level, rest.trim().trim_end_matches('#').trim()))
}

/// The text of a list item line, without its marker.
fn list_item(line: &str) -> Option<&str> {
    for marker in ["* ", "- ", "+ "] {
        if let Some(item) = line.strip_prefix(marker) {
            return Some(item);
        }
    }
    let digits = line.bytes().take_while(u8::is_ascii_digit).count();
    if (1..=3).contains(&digits) {
        let rest = &line[digits..];
        return rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "));
    }
    None
}

fn is_fence(line: &str) -> bool {
    line.starts_with("```") || line.starts_with("~~~")
}

fn is_rule(line: &str) -> bool {
    line.len() >= 3 && (line.bytes().all(|b| b == b'-') || line.bytes().all(|b| b == b'*'))
}

/// `[!NOTE]` and its kin: the mark of a call-out box.
fn is_callout_mark(line: &str) -> bool {
    line.starts_with("[!") && line.ends_with(']')
}

/// `[label]: https://…`: where a reference link points, not text.
fn is_link_definition(line: &str) -> bool {
    line.starts_with('[')
        && line.split_once("]: ").is_some_and(|(_, target)| {
            !target.contains(' ') && (target.starts_with("http") || target.starts_with('/'))
        })
}

fn has_words(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

/// Cut an article's Markdown into its second-level sections, as plain text.
/// The first-level heading is the article's title and is not a block.
pub fn sections(markdown: &str) -> Vec<Section> {
    let mut sections = vec![Section {
        heading: String::new(),
        blocks: Vec::new(),
    }];
    let mut paragraph = String::new();
    let mut in_code = false;
    let mut in_table = false;

    fn flush(paragraph: &mut String, sections: &mut [Section]) {
        let block = squash(&plain(paragraph));
        paragraph.clear();
        if has_words(&block)
            && let Some(section) = sections.last_mut()
        {
            section.blocks.push(block);
        }
    }

    for raw_line in markdown.lines() {
        let line = unquote(raw_line);
        if is_fence(line) {
            in_code = !in_code;
            flush(&mut paragraph, &mut sections);
            continue;
        }
        if in_code {
            continue;
        }
        // An HTML table can span lines; a Markdown table row starts with a bar.
        if in_table {
            in_table = !line.contains("</table");
            continue;
        }
        if line.contains("<table") && !line.contains("</table") {
            in_table = true;
            flush(&mut paragraph, &mut sections);
            continue;
        }
        if line.starts_with('|') {
            flush(&mut paragraph, &mut sections);
            continue;
        }
        if line.is_empty() || is_rule(line) || is_callout_mark(line) || is_link_definition(line) {
            flush(&mut paragraph, &mut sections);
            continue;
        }
        if let Some((level, text)) = heading(line) {
            flush(&mut paragraph, &mut sections);
            let text = squash(&plain(text));
            match level {
                1 => {}
                2 => sections.push(Section {
                    heading: text,
                    blocks: Vec::new(),
                }),
                _ => {
                    if has_words(&text)
                        && let Some(section) = sections.last_mut()
                    {
                        section.blocks.push(text);
                    }
                }
            }
            continue;
        }
        if let Some(item) = list_item(line) {
            flush(&mut paragraph, &mut sections);
            paragraph.push_str(item);
            continue;
        }
        if !paragraph.is_empty() {
            paragraph.push(' ');
        }
        paragraph.push_str(line);
    }
    flush(&mut paragraph, &mut sections);
    sections
}

/// Cut a block that is longer than `budget` words at sentence ends, and inside
/// a sentence only when the sentence itself is too long.
fn split_block(block: &str, budget: usize) -> Vec<String> {
    if word_count(block) <= budget {
        return vec![block.to_string()];
    }
    let mut sentences: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in block.split_whitespace() {
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(word);
        if word.ends_with(['.', '?', '!']) {
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }
    let mut parts: Vec<String> = Vec::new();
    let mut part = String::new();
    for sentence in sentences {
        let pieces: Vec<String> = if word_count(&sentence) > budget {
            let words: Vec<&str> = sentence.split_whitespace().collect();
            words.chunks(budget.max(1)).map(|chunk| chunk.join(" ")).collect()
        } else {
            vec![sentence]
        };
        for piece in pieces {
            if !part.is_empty() && word_count(&part) + word_count(&piece) > budget {
                parts.push(std::mem::take(&mut part));
            }
            if !part.is_empty() {
                part.push(' ');
            }
            part.push_str(&piece);
        }
    }
    if !part.is_empty() {
        parts.push(part);
    }
    parts
}

/// The texts of a section's records: the article's title and the section's
/// heading on the first line, then the section's prose. A section shorter than
/// the least length gives none; a long one is cut at block ends into parts of
/// about equal length, each within the limits.
pub fn passages(section: &Section, title: &str, rules: &DocRules) -> Vec<String> {
    let first_line = if section.heading.is_empty() || section.heading == title {
        title.to_string()
    } else {
        format!("{title}: {}", section.heading)
    };
    let head = word_count(&first_line);
    let budget = rules.max_words.saturating_sub(head).max(1);
    let units: Vec<String> = section
        .blocks
        .iter()
        .flat_map(|b| split_block(b, budget))
        .collect();
    let total: usize = units.iter().map(|u| word_count(u)).sum();
    if head + total < rules.min_words {
        return Vec::new();
    }
    let parts_wanted = total.div_ceil(budget).max(1);
    let target = total.div_ceil(parts_wanted);

    let mut parts: Vec<Vec<String>> = vec![Vec::new()];
    let mut words_in_part = 0;
    for unit in units {
        let words = word_count(&unit);
        if words_in_part > 0 && (words_in_part + words > budget || words_in_part >= target) {
            parts.push(Vec::new());
            words_in_part = 0;
        }
        words_in_part += words;
        if let Some(part) = parts.last_mut() {
            part.push(unit);
        }
    }

    // A short tail takes blocks from the part before it, or is left out.
    let size = |part: &[String]| part.iter().map(|u| word_count(u)).sum::<usize>();
    while parts.len() >= 2 && head + parts.last().map_or(0, |p| size(p)) < rules.min_words {
        let last = parts.len() - 1;
        let (before, tail) = parts.split_at_mut(last);
        let (previous, tail) = (&mut before[last - 1], &mut tail[0]);
        let movable = previous.last().map(|u| word_count(u)).unwrap_or(0);
        let fits = previous.len() > 1
            && size(tail) + movable <= budget
            && head + size(previous) - movable >= rules.min_words;
        if fits {
            if let Some(unit) = previous.pop() {
                tail.insert(0, unit);
            }
        } else {
            parts.pop();
            break;
        }
    }

    parts
        .into_iter()
        .filter(|part| head + size(part) >= rules.min_words)
        .map(|part| format!("{first_line}\n{}", part.join("\n")))
        .collect()
}

/// `ghdoc-<area>-<article>-<hash>`: the area, the article's own name (cut to
/// 48 characters) and six hex digits of the path's hash, which keep two
/// articles with one name apart.
pub fn slug(path: &DocPath, area: &DocArea) -> String {
    let name: String = path
        .last_segment()
        .chars()
        .map(|c| c.to_ascii_lowercase())
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '-')
        .take(48)
        .collect();
    let hash = fnv1a64(path.as_str().as_bytes()) & 0xff_ffff;
    format!("ghdoc-{}-{}-{hash:06x}", area.slug, name.trim_matches('-'))
}

/// The texts of an article's records, in reading order. "Further reading"
/// sections are lists of links and give none.
pub fn passages_of(file: &ArticleFile, rules: &DocRules) -> Vec<String> {
    let title = squash(&plain(&file.meta.title));
    sections(&file.body)
        .iter()
        .filter(|s| !LINK_LIST_HEADINGS.contains(&s.heading.to_lowercase().as_str()))
        .flat_map(|s| passages(s, &title, rules))
        .collect()
}

/// The `n`th record (from 1) of the article whose id stem is `slug`.
pub fn record(slug: &str, n: usize, title: &str, text: String) -> Record {
    Record {
        id: format!("{slug}.{n}"),
        text,
        written_at: WRITTEN_AT,
        valid_until: None,
        thread: None,
        tier: Tier::Standing,
        title: Some(title.to_string()),
        source: RECORD_SOURCE.to_string(),
        truth: None,
    }
}

/// What reading the saved articles did.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct DocsStats {
    /// Pages of the kept areas found under `raw`.
    pub pages: u64,
    /// Articles that gave at least one record.
    pub articles_used: u64,
    /// Pages that gave none, by reason.
    pub left_out: BTreeMap<String, u64>,
    pub records: u64,
    /// Records by product area.
    pub records_by_area: BTreeMap<String, u64>,
    /// Passages that repeat an earlier passage word for word (reused text):
    /// the search index would keep one of them, so only the first is a record.
    pub repeated_passages_dropped: u64,
}

/// What reading the saved articles gave.
#[derive(Debug, Default)]
pub struct Read {
    pub records: Vec<Record>,
    pub stats: DocsStats,
}

/// Every saved article under `raw/docs/articles`, in the page list's order.
/// A page of the list that was not saved (the site no longer had it) is skipped.
pub fn read(raw_dir: &Path, rules: &DocRules) -> Result<Read, DocsError> {
    let list_file = raw::pagelist_file(raw_dir);
    let list = std::fs::read_to_string(&list_file).map_err(|source| DocsError::List {
        path: list_file.display().to_string(),
        source,
    })?;
    let mut read = Read::default();
    let mut slugs: BTreeMap<String, String> = BTreeMap::new();
    let mut texts: HashSet<String> = HashSet::new();
    for path in raw::kept_paths(&list)? {
        let file = raw::article_file(raw_dir, &path);
        if !file.exists() {
            continue;
        }
        let area = raw::area_of(&path).ok_or_else(|| DocsError::NoArea(path.to_string()))?;
        let article: ArticleFile = raw::read_json(&file)?;
        read.stats.pages += 1;
        if article.meta.document_type.as_deref() != Some(ARTICLE) {
            *read
                .stats
                .left_out
                .entry(LEFT_OUT_NOT_ARTICLE.to_string())
                .or_insert(0) += 1;
            continue;
        }
        let slug = slug(&path, area);
        if let Some(first) = slugs.insert(slug.clone(), path.to_string()) {
            return Err(DocsError::SameId {
                id: slug,
                first,
                second: path.to_string(),
            });
        }
        let title = squash(&plain(&article.meta.title));
        let mut kept = Vec::new();
        for text in passages_of(&article, rules) {
            if texts.insert(squash(&text)) {
                kept.push(text);
            } else {
                read.stats.repeated_passages_dropped += 1;
            }
        }
        if kept.is_empty() {
            *read
                .stats
                .left_out
                .entry(LEFT_OUT_NO_PROSE.to_string())
                .or_insert(0) += 1;
            continue;
        }
        read.stats.articles_used += 1;
        *read
            .stats
            .records_by_area
            .entry(area.slug.to_string())
            .or_insert(0) += kept.len() as u64;
        for (i, text) in kept.into_iter().enumerate() {
            let record = record(&slug, i + 1, &title, text);
            ids::check(&record.id)?;
            read.records.push(record);
        }
    }
    read.stats.records = read.records.len() as u64;
    Ok(read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public::text::words;

    /// A real article from the article API, with a call-out, tables, code blocks,
    /// an icon in a sentence and a "Further reading" list.
    const ARTICLE: &str = include_str!("../testdata/github/docs/track-usage-and-adoption.json");

    fn article() -> ArticleFile {
        serde_json::from_str(ARTICLE).unwrap()
    }

    #[test]
    fn an_article_is_cut_at_its_second_level_headings() {
        let sections = sections(&article().body);
        let headings: Vec<&str> = sections.iter().map(|s| s.heading.as_str()).collect();
        assert_eq!(
            headings,
            [
                "",
                "Prerequisite",
                "Step 1: Access the usage metrics dashboard",
                "Step 2: Track license activation",
                "Step 3: Identify early adoption signals",
                "Step 4: Act on limited adoption signals",
                "Step 5: Track activation programmatically",
                "Step 6: Export user-level data for deeper analysis",
                "Further reading",
            ]
        );
    }

    #[test]
    fn a_passage_is_plain_prose_without_tables_code_or_markup() {
        let file = article();
        let passages = passages_of(&file, &DocRules::default());
        assert!(!passages.is_empty());
        let all = passages.join("\n");
        for markup in ["```", "| ", "<svg", "](", "[!NOTE]", "[!TIP]", "**", "<div", "{%"] {
            assert!(!all.contains(markup), "{markup:?} left in:\n{all}");
        }
        // A table's cells and a code block's lines are not prose.
        assert!(!all.contains("Rolling 7-day total of active users"));
        assert!(!all.contains("What patterns might explain low adoption"));
        // A call-out's text is prose and stays; an icon reads as its label.
        assert!(all.contains("download"), "{all}");
        // "Further reading" lists links only.
        assert!(!all.contains("Further reading"));
        let title = "Tracking license activation and initial usage with Copilot usage metrics";
        for passage in &passages {
            let n = passage.split_whitespace().count();
            assert!((40..=300).contains(&n), "{n} words: {passage}");
            assert!(passage.starts_with(title), "{passage}");
        }
        assert!(
            passages
                .iter()
                .any(|p| p.starts_with(&format!("{title}: Step 2: Track license activation\n")))
        );
    }

    #[test]
    fn a_long_section_is_split_at_paragraph_ends_and_a_short_one_gives_nothing() {
        let paragraph = |n: usize| {
            let mut words: Vec<String> = (0..n).map(|i| format!("word{i}")).collect();
            if let Some(last) = words.last_mut() {
                last.push('.');
            }
            words.join(" ")
        };
        let long = Section {
            heading: "Configuring runners".into(),
            blocks: vec![
                paragraph(120),
                paragraph(150),
                paragraph(130),
                paragraph(110),
                paragraph(90),
            ],
        };
        let parts = passages(&long, "Self-hosted runners", &DocRules::default());
        assert!(parts.len() >= 2, "{}", parts.len());
        for part in &parts {
            let n = part.split_whitespace().count();
            assert!((40..=300).contains(&n), "{n}");
            assert!(part.starts_with("Self-hosted runners: Configuring runners\n"));
        }
        // No paragraph is cut in two, and none is lost.
        // Each part starts with the four words of "Self-hosted runners: Configuring runners".
        let total: usize = parts.iter().map(|p| words(p).len() - 4).sum();
        assert_eq!(total, 120 + 150 + 130 + 110 + 90);
        let short = Section {
            heading: "Note".into(),
            blocks: vec!["Too short to say anything.".into()],
        };
        assert!(passages(&short, "Self-hosted runners", &DocRules::default()).is_empty());
    }

    #[test]
    fn ids_name_the_area_and_the_article_and_stay_apart() {
        let a = DocPath::parse("/en/actions/how-tos/manage-runners/about-runners").unwrap();
        let b = DocPath::parse("/en/actions/reference/about-runners").unwrap();
        let area = raw::area_of(&a).unwrap();
        let (sa, sb) = (slug(&a, area), slug(&b, area));
        assert!(sa.starts_with("ghdoc-actions-about-runners-"), "{sa}");
        assert_ne!(sa, sb);
        let record = record(&sa, 3, "About runners", "About runners\nText.".into());
        assert_eq!(record.id, format!("{sa}.3"));
        assert!(ids::check(&record.id).is_ok());
        assert_eq!(record.tier, Tier::Standing);
        assert_eq!(record.thread, None);
        assert_eq!(record.written_at, WRITTEN_AT);
    }
}
