//! A token or key that is sent to a service and shown nowhere else.

use std::convert::Infallible;
use std::fmt;
use std::str::FromStr;

/// A token or key. It can be sent in a request header; it cannot be printed, logged or
/// written to a file: it has no `Display`, its `Debug` hides the value, and it is not
/// serialisable.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// The value, for the one place it belongs: a request header.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// An unset variable and an empty one mean the same: no token.
    pub fn is_blank(&self) -> bool {
        self.0.trim().is_empty()
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(hidden)")
    }
}

impl FromStr for Secret {
    type Err = Infallible;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Ok(Self(value.to_string()))
    }
}

/// The token to send, or none when it is unset or empty.
pub fn usable(secret: Option<&Secret>) -> Option<&Secret> {
    secret.filter(|s| !s.is_blank())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_secret_never_shows_its_value() {
        let secret: Secret = "sk-very-private".parse().unwrap();
        let shown = format!("{secret:?} {:?}", Some(&secret));
        assert!(!shown.contains("very-private"), "{shown}");
        assert_eq!(secret.expose(), "sk-very-private");
    }

    #[test]
    fn an_empty_token_is_no_token() {
        let empty: Secret = "  ".parse().unwrap();
        assert!(usable(Some(&empty)).is_none());
        assert!(usable(None).is_none());
        let set: Secret = "t".parse().unwrap();
        assert!(usable(Some(&set)).is_some());
    }
}
