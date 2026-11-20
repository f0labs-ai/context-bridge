//! Record ids and the document path a record is filed under in the search service.
//!
//! A record is uploaded to the knowledge service as its own small document.
//! The service answers searches with that document's path, so the path is how
//! a search hit is joined back to its record.

use crate::ContractError;

/// Longest id, in bytes.
pub const MAX_ID_LEN: usize = 96;

/// An id is 1–96 characters of `a-z`, `0-9`, `-`, `_` and `.`.
pub fn check(id: &str) -> Result<(), ContractError> {
    let ok = !id.is_empty()
        && id.len() <= MAX_ID_LEN
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.'));
    if ok {
        Ok(())
    } else {
        Err(ContractError::Invalid(format!("not a record id: {id:?}")))
    }
}

/// The path a record's document is uploaded under.
pub fn source_path(record_id: &str) -> String {
    format!("cb/{record_id}.md")
}

/// The record a search hit's document path belongs to; `None` for any other document.
pub fn record_of(source_path: &str) -> Option<&str> {
    let id = source_path.strip_prefix("cb/")?.strip_suffix(".md")?;
    check(id).ok().map(|()| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_leads_back_to_its_record() {
        assert!(check("gh-2026-11-18-abc.3").is_ok());
        assert_eq!(
            record_of(&source_path("gh-2026-11-18-abc.3")),
            Some("gh-2026-11-18-abc.3")
        );
    }

    #[test]
    fn other_documents_and_bad_ids_are_not_records() {
        assert_eq!(record_of("https://docs.example.com/page"), None);
        assert_eq!(record_of("cb/Has Space.md"), None);
        assert!(check("").is_err());
        assert!(check("UPPER").is_err());
        assert!(check(&"x".repeat(MAX_ID_LEN + 1)).is_err());
    }
}
