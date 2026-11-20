//! Turning the sources' marked-up text into plain text.
//!
//! Status updates arrive with a little HTML or Markdown in them, and the
//! documentation arrives as Markdown. A record's text is what a person would
//! read: no tags, no link targets, no emphasis marks.

/// Number of whitespace-separated words.
pub fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Every run of whitespace becomes one space; none at either end.
pub fn squash(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for word in text.split_whitespace() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// Like [`squash`] within each line; blank lines are dropped.
pub fn squash_lines(text: &str) -> String {
    let lines: Vec<String> = text.lines().map(squash).filter(|l| !l.is_empty()).collect();
    lines.join("\n")
}

const NAMED_ENTITIES: &[(&str, &str)] = &[
    ("amp", "&"),
    ("lt", "<"),
    ("gt", ">"),
    ("quot", "\""),
    ("apos", "'"),
    ("nbsp", " "),
    ("ndash", "–"),
    ("mdash", "—"),
    ("hellip", "…"),
    ("lsquo", "‘"),
    ("rsquo", "’"),
    ("ldquo", "“"),
    ("rdquo", "”"),
    ("copy", "©"),
    ("reg", "®"),
    ("trade", "™"),
    ("times", "×"),
    ("rarr", "→"),
    ("larr", "←"),
];

fn entity(name: &str) -> Option<String> {
    if let Some(number) = name.strip_prefix('#') {
        let code = match number.strip_prefix(['x', 'X']) {
            Some(hex) => u32::from_str_radix(hex, 16).ok()?,
            None => number.parse::<u32>().ok()?,
        };
        return char::from_u32(code).map(String::from);
    }
    NAMED_ENTITIES
        .iter()
        .find(|(known, _)| *known == name)
        .map(|(_, text)| (*text).to_string())
}

/// `&amp;` → `&`, `&#39;` → `'`, `&#x2014;` → `—`. A reference that is not known stays as written.
pub fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let after = &rest[at + 1..];
        let decoded = after
            .find(';')
            .filter(|end| (1..=10).contains(end))
            .and_then(|end| entity(&after[..end]).map(|text| (text, end)));
        match decoded {
            Some((text, end)) => {
                out.push_str(&text);
                rest = &after[end + 1..];
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Elements whose tags are removed. `<` followed by anything else is ordinary
/// text (`a < b`, `--zone=<zone>`), so it is left alone.
const HTML_ELEMENTS: &[&str] = &[
    "a",
    "abbr",
    "article",
    "b",
    "blockquote",
    "br",
    "button",
    "caption",
    "center",
    "cite",
    "code",
    "col",
    "colgroup",
    "dd",
    "del",
    "details",
    "div",
    "dl",
    "dt",
    "em",
    "figcaption",
    "figure",
    "font",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "i",
    "iframe",
    "img",
    "input",
    "ins",
    "kbd",
    "label",
    "li",
    "link",
    "mark",
    "meta",
    "nav",
    "ol",
    "p",
    "path",
    "picture",
    "pre",
    "q",
    "s",
    "script",
    "section",
    "small",
    "source",
    "span",
    "strong",
    "style",
    "sub",
    "summary",
    "sup",
    "svg",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "time",
    "tr",
    "u",
    "ul",
    "var",
    "video",
];

/// Elements whose whole content is dropped with them.
const DROPPED_WITH_CONTENT: &[&str] = &["svg", "style", "script", "table"];

/// Tags that end a line of text.
const LINE_ENDING: &[&str] = &[
    "br",
    "p",
    "div",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "ul",
    "ol",
    "hr",
];

/// The tag starting at `text` (which begins with `<`): its lower-cased element
/// name, whether it is a closing tag, and its length in bytes.
fn tag_at(text: &str) -> Option<(String, bool, usize)> {
    let inner = text.strip_prefix('<')?;
    let (closing, inner) = match inner.strip_prefix('/') {
        Some(rest) => (true, rest),
        None => (false, inner),
    };
    let name_len = inner.bytes().take_while(u8::is_ascii_alphanumeric).count();
    let name = inner[..name_len].to_ascii_lowercase();
    if !HTML_ELEMENTS.contains(&name.as_str()) {
        return None;
    }
    // The name must end the tag or be followed by attributes.
    let after_name = inner[name_len..].chars().next()?;
    if !(after_name == '>' || after_name == '/' || after_name.is_whitespace()) {
        return None;
    }
    // The tag ends at the first `>` outside a quoted attribute value.
    let mut quote: Option<char> = None;
    for (i, c) in text.char_indices() {
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '"' || c == '\'' => quote = Some(c),
            None if c == '>' => return Some((name, closing, i + 1)),
            None => {}
        }
    }
    None
}

/// The value of a double-quoted attribute of a tag.
fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let key = format!(" {name}=\"");
    let start = tag.find(&key)? + key.len();
    let end = tag[start..].find('"')?;
    Some(&tag[start..start + end])
}

/// Remove HTML tags and comments, keeping the text between them. The content of
/// drawings, style sheets, scripts and tables goes with their tags (a drawing
/// with a label leaves the label); a tag that ends a block ends the line.
pub fn strip_html(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('<') {
        out.push_str(&rest[..at]);
        let from_tag = &rest[at..];
        if let Some(comment) = from_tag.strip_prefix("<!--") {
            rest = comment.find("-->").map_or("", |end| &comment[end + 3..]);
            continue;
        }
        let Some((name, closing, len)) = tag_at(from_tag) else {
            out.push('<');
            rest = &from_tag[1..];
            continue;
        };
        let tag = &from_tag[..len];
        rest = &from_tag[len..];
        let self_closed = tag.ends_with("/>");
        if !closing && !self_closed && DROPPED_WITH_CONTENT.contains(&name.as_str()) {
            let close = format!("</{name}");
            let lower = rest.to_ascii_lowercase();
            rest = match lower.find(&close) {
                Some(end) => {
                    let after = &rest[end..];
                    after.find('>').map_or("", |gt| &after[gt + 1..])
                }
                None => "",
            };
            // An icon in a sentence ("click <svg aria-label="Edit">") reads as its label.
            match attribute(tag, "aria-label").filter(|_| name == "svg") {
                Some(label) => out.push_str(label),
                None => out.push('\n'),
            }
        } else if LINE_ENDING.contains(&name.as_str()) {
            out.push('\n');
        }
    }
    out.push_str(rest);
    out
}

/// The position of the `]` that closes the `[` at the start of `text`.
fn closing_bracket(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in text.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            '\n' => return None,
            _ => {}
        }
    }
    None
}

/// The position of the `)` that closes the `(` at the start of `text`.
fn closing_paren(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in text.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            '\n' => return None,
            _ => {}
        }
    }
    None
}

/// `[text](target)` → `text`; `![alt](target)` → nothing.
fn unlink(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('[') {
        let image = at > 0 && rest.as_bytes()[at - 1] == b'!';
        let before = if image { &rest[..at - 1] } else { &rest[..at] };
        let from_bracket = &rest[at..];
        let link = closing_bracket(from_bracket).and_then(|close| {
            let after = &from_bracket[close + 1..];
            if !after.starts_with('(') {
                return None;
            }
            closing_paren(after).map(|end| (&from_bracket[1..close], &after[end + 1..]))
        });
        match link {
            Some((label, after)) => {
                out.push_str(before);
                if !image {
                    out.push_str(&unlink(label));
                }
                rest = after;
            }
            None => {
                out.push_str(&rest[..=at]);
                rest = &rest[at + 1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Remove templating tags (`{% ... %}`) that a source left unresolved.
fn strip_template_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("{%") {
        out.push_str(&rest[..at]);
        match rest[at..].find("%}") {
            Some(end) => rest = &rest[at + end + 2..],
            None => {
                out.push_str(&rest[at..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// Remove emphasis marks, code ticks and backslash escapes.
fn unmark(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        match c {
            // A backslash before punctuation only protects it.
            '\\' if next.is_some_and(|n| n.is_ascii_punctuation()) => {
                out.extend(next);
                i += 2;
                continue;
            }
            '`' => {}
            // `**bold**`, `__bold__`, and `*emphasis*` when the star touches a word.
            '*' => {
                let prev = if i == 0 { None } else { chars.get(i - 1).copied() };
                let touches_word =
                    next.is_some_and(|n| !n.is_whitespace()) || prev.is_some_and(|p| !p.is_whitespace());
                if !touches_word {
                    out.push(c);
                }
            }
            '_' if next == Some('_') => {
                i += 2;
                continue;
            }
            _ => out.push(c),
        }
        i += 1;
    }
    out
}

/// Marks that stand for an inline code span while the text around it is
/// cleaned. They are private-use characters, which no cleaning step touches.
const CODE_OPEN: char = '\u{E000}';
const CODE_CLOSE: char = '\u{E001}';

/// Where a run of exactly `ticks` backticks starts in `text`.
fn backtick_run(text: &str, ticks: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let run = bytes[i..].iter().take_while(|b| **b == b'`').count();
            if run == ticks {
                return Some(i);
            }
            i += run;
        } else {
            i += 1;
        }
    }
    None
}

/// Take the inline code spans out of `text`, each replaced by a numbered mark;
/// their contents come back to be put in again unchanged.
fn take_code_spans(text: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(text.len());
    let mut spans = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find('`') {
        let ticks = rest[at..].bytes().take_while(|b| *b == b'`').count();
        let after = &rest[at + ticks..];
        // A span closes with as many backticks, on the same line.
        let line = &after[..after.find('\n').unwrap_or(after.len())];
        match backtick_run(line, ticks) {
            Some(end) => {
                out.push_str(&rest[..at]);
                out.push(CODE_OPEN);
                out.push_str(&spans.len().to_string());
                out.push(CODE_CLOSE);
                spans.push(squash(&after[..end]));
                rest = &after[end + ticks..];
            }
            None => {
                out.push_str(&rest[..at + ticks]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    (out, spans)
}

fn put_code_spans(text: &str, spans: &[String]) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(CODE_OPEN) {
        out.push_str(&rest[..at]);
        let after = &rest[at + CODE_OPEN.len_utf8()..];
        let span = after.find(CODE_CLOSE).and_then(|end| {
            let n = after[..end].parse::<usize>().ok()?;
            spans.get(n).map(|span| (span, end))
        });
        match span {
            Some((span, end)) => {
                out.push_str(span);
                rest = &after[end + CODE_CLOSE.len_utf8()..];
            }
            None => rest = after,
        }
    }
    out.push_str(rest);
    out
}

/// One line or paragraph of Markdown or lightly marked-up text as plain text:
/// templating tags, HTML tags, link targets, images, emphasis marks and code
/// ticks removed; character references decoded; whitespace squashed per line.
/// What is inside an inline code span is kept as written.
pub fn plain(text: &str) -> String {
    let text = text.replace([CODE_OPEN, CODE_CLOSE], "");
    let (text, spans) = take_code_spans(&text);
    let text = strip_template_tags(&text);
    let text = strip_html(&text);
    let text = unlink(&text);
    let text = unmark(&text);
    let text = decode_entities(&text);
    put_code_spans(&squash_lines(&text), &spans)
}

/// A line that only draws a rule (`---`, `***`, `___`, `===`).
fn is_rule_line(line: &str) -> bool {
    line.len() >= 3 && line.bytes().all(|b| matches!(b, b'-' | b'*' | b'_' | b'='))
}

/// A status update as plain text. Status pages write updates in light HTML or
/// Markdown: quote, heading and list marks go, then [`plain`] does the rest.
/// One line per paragraph or list item.
pub fn status_text(text: &str) -> String {
    let lines: Vec<&str> = text
        .lines()
        .map(|line| {
            let mut line = line.trim();
            while let Some(rest) = line.strip_prefix('>') {
                line = rest.trim_start();
            }
            let hashes = line.bytes().take_while(|b| *b == b'#').count();
            if hashes > 0 && line[hashes..].starts_with(' ') {
                line = line[hashes..].trim_start();
            }
            for marker in ["* ", "- ", "+ "] {
                if let Some(rest) = line.strip_prefix(marker) {
                    line = rest;
                    break;
                }
            }
            line
        })
        .filter(|line| !is_rule_line(line))
        .collect();
    plain(&lines.join("\n"))
}

/// Lower-cased words with the punctuation at their edges removed.
pub fn words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Whether `needle` occurs in `haystack` as a run of whole words.
pub fn contains_run(haystack: &[String], needle: &[String]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_is_squashed_and_words_are_counted() {
        assert_eq!(squash("  a \t b\r\n\r\n c  "), "a b c");
        assert_eq!(squash_lines("  a  b \r\n\r\n\n c \n"), "a b\nc");
        assert_eq!(word_count("one  two\nthree"), 3);
        assert_eq!(word_count("   "), 0);
    }

    #[test]
    fn character_references_are_decoded_and_unknown_ones_kept() {
        assert_eq!(
            decode_entities("a &amp; b &lt;c&gt; &quot;d&quot;"),
            "a & b <c> \"d\""
        );
        assert_eq!(
            decode_entities("it&#39;s &#x2014; fine&nbsp;now"),
            "it's — fine now"
        );
        assert_eq!(
            decode_entities("AT&T &unknown; & alone &"),
            "AT&T &unknown; & alone &"
        );
        // Decoded once, not twice.
        assert_eq!(decode_entities("&amp;lt;"), "&lt;");
    }

    #[test]
    fn html_tags_go_and_their_text_stays() {
        assert_eq!(
            squash_lines(&strip_html(
                "We are investigating.<br>Next update in <b>30 minutes</b>."
            )),
            "We are investigating.\nNext update in 30 minutes."
        );
        assert_eq!(
            squash(&strip_html(
                "See <a href=\"https://x.example/?a=1>2\">the post</a> for more."
            )),
            "See the post for more."
        );
        // A button with a drawing in it keeps its label and loses the drawing.
        let button = "<a class=\"btn\"><span>View your pull requests</span> <svg viewBox=\"0 0 16 16\">\
                      <path d=\"M3.75 2h3.5\"></path><title>link external icon</title></svg></a>";
        assert_eq!(squash(&strip_html(button)), "View your pull requests");
        assert_eq!(
            squash(&strip_html("before<!-- markdownlint-disable -->after")),
            "beforeafter"
        );
        assert_eq!(
            squash_lines(&strip_html("x<table><tr><td>cell</td></tr></table>y")),
            "x\ny"
        );
    }

    #[test]
    fn angle_brackets_that_are_not_html_are_text() {
        let text = "Run gcloud compute ssh <instance> --zone=<zone> when a < b and b > c.";
        assert_eq!(strip_html(text), text);
        assert_eq!(
            strip_html("<your_project> <instance1,instance2,...>"),
            "<your_project> <instance1,instance2,...>"
        );
    }

    #[test]
    fn markdown_marks_and_link_targets_go() {
        assert_eq!(
            plain(
                "See [Creating a pull request](/en/pull-requests/creating-a-pull-request) and **merge** it."
            ),
            "See Creating a pull request and merge it."
        );
        assert_eq!(
            plain("An image ![Screenshot of the tab](/assets/tab.png) here."),
            "An image here."
        );
        assert_eq!(
            plain("Use `on: push` or *maybe* `schedule`."),
            "Use on: push or maybe schedule."
        );
        assert_eq!(
            plain("[`check_run`](/en/webhooks#check_run) event"),
            "check_run event"
        );
        assert_eq!(
            plain("Keep snake_case_names and 2 * 3 = 6."),
            "Keep snake_case_names and 2 * 3 = 6."
        );
        assert_eq!(
            plain("A \\*literal\\* star and __bold__."),
            "A *literal* star and bold."
        );
        assert_eq!(
            plain("Brackets [not a link] stay."),
            "Brackets [not a link] stay."
        );
        assert_eq!(
            plain("[https://status.example/i/1](https://status.example/i/1)."),
            "https://status.example/i/1."
        );
    }

    #[test]
    fn inline_code_is_kept_as_written() {
        assert_eq!(
            plain("Add a `<details>` block, or set `${{ secrets.TOKEN }}` and `**not bold**`."),
            "Add a <details> block, or set ${{ secrets.TOKEN }} and **not bold**."
        );
        assert_eq!(plain("Run ``a `quoted` word`` now"), "Run a `quoted` word now");
        // An unclosed tick is dropped like any other mark.
        assert_eq!(plain("a stray ` tick"), "a stray tick");
        // The private-use marks cannot be smuggled in.
        assert_eq!(plain("x \u{E000}0\u{E001} y"), "x 0 y");
    }

    #[test]
    fn an_icon_in_a_sentence_reads_as_its_label() {
        let line = "Next to the release, click <svg version=\"1.1\" width=\"16\" class=\"octicon octicon-pencil\" \
                    aria-label=\"Edit\" role=\"img\"><path d=\"M11.013 1.427a1.75\"></path></svg>.";
        assert_eq!(plain(line), "Next to the release, click Edit.");
        assert_eq!(
            plain("x<svg viewBox=\"0 0 1 1\"><path d=\"M0\"></path></svg>y"),
            "x\ny"
        );
    }

    #[test]
    fn status_texts_lose_their_markdown_structure() {
        let text = "# Mini Incident Report\n\n**Summary:**\nThe issue is mitigated.\n\n---\n\
                    * Retry the job.\n- Or move to [another zone](https://cloud.example/zones).\n> quoted";
        assert_eq!(
            status_text(text),
            "Mini Incident Report\nSummary:\nThe issue is mitigated.\nRetry the job.\nOr move to another zone.\nquoted"
        );
        assert_eq!(
            status_text("We are investigating.<br>Next update in 30 minutes."),
            "We are investigating.\nNext update in 30 minutes."
        );
    }

    #[test]
    fn templating_tags_are_removed_and_expression_braces_are_kept() {
        assert_eq!(
            plain("On {% data variables.product.prodname_dotcom %}, use {% ifversion fpt %}this{% endif %}."),
            "On , use this."
        );
        assert_eq!(
            plain("Set `${{ secrets.TOKEN }}` in the step."),
            "Set ${{ secrets.TOKEN }} in the step."
        );
    }

    #[test]
    fn a_run_of_words_is_found_whatever_the_case_and_punctuation() {
        let body = words("We are investigating reports of degraded performance for Pull Requests.");
        assert!(contains_run(&body, &words("reports of degraded performance")));
        assert!(contains_run(&body, &words("pull requests")));
        assert!(!contains_run(&body, &words("performance of reports")));
        assert!(!contains_run(&body, &[]));
    }
}
