//! The one way this tool talks HTTP.
//!
//! Every request has a connect timeout and a total timeout. Redirects are not followed
//! (a service that moved is a failure to look at, not something to follow with a token).
//! An answer that is not a success is an error that keeps the start of its body. An
//! address is only ever shown without its user, password and query string.

use crate::error::{RunError, excerpt};
use reqwest::Url;
use serde::de::DeserializeOwned;
use std::time::{Duration, Instant};

/// The time limits every request gets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HttpSettings {
    pub connect_timeout: Duration,
    pub total_timeout: Duration,
    /// Whether the proxy variables of the environment apply. Tests turn this off so a
    /// developer's proxy never sits between a test and its local stand-in.
    pub proxy_from_env: bool,
}

/// A client that keeps the limits of [`HttpSettings`] on every request.
#[derive(Clone, Debug)]
pub struct Http {
    client: reqwest::Client,
    total_timeout: Duration,
}

/// A successful answer: its whole body, and how long the exchange took as this side saw
/// it (from sending the request to the last byte of the body).
#[derive(Debug)]
pub struct Answered {
    pub body: Vec<u8>,
    pub wall: Duration,
}

impl Http {
    pub fn new(settings: &HttpSettings) -> Result<Self, RunError> {
        let mut builder = reqwest::Client::builder()
            .connect_timeout(settings.connect_timeout)
            .timeout(settings.total_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .user_agent(concat!("cb-run/", env!("CARGO_PKG_VERSION")));
        if !settings.proxy_from_env {
            builder = builder.no_proxy();
        }
        let client = builder.build().map_err(|e| RunError::Setting {
            name: "http client",
            reason: chain(&e),
        })?;
        Ok(Self {
            client,
            total_timeout: settings.total_timeout,
        })
    }

    /// The client requests are built with.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// One exchange: sends `request` to `url` and reads the whole answer. One try; the
    /// caller decides about retries ([`crate::retry`]).
    pub async fn exchange(&self, url: &Url, request: reqwest::RequestBuilder) -> Result<Answered, RunError> {
        let started = Instant::now();
        let response = request.send().await.map_err(|e| self.failed(url, e))?;
        let status = response.status();
        let retry_after_s = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok());
        let body = response.bytes().await.map_err(|e| self.failed(url, e))?;
        let wall = started.elapsed();
        if !status.is_success() {
            return Err(RunError::Status {
                url: shown(url),
                status: status.as_u16(),
                body: excerpt(&body),
                retry_after_s,
            });
        }
        Ok(Answered {
            body: Vec::from(body),
            wall,
        })
    }

    fn failed(&self, url: &Url, error: reqwest::Error) -> RunError {
        if error.is_timeout() {
            RunError::Timeout {
                url: shown(url),
                limit_ms: self.total_timeout.as_millis() as u64,
            }
        } else {
            // reqwest prints the full address in its message; it is replaced by the shown form.
            RunError::Transport {
                url: shown(url),
                reason: chain(&error.without_url()),
            }
        }
    }
}

/// An error with everything that caused it, in one line.
fn chain(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut source = error.source();
    while let Some(inner) = source {
        let inner_text = inner.to_string();
        if !text.contains(&inner_text) {
            text.push_str(": ");
            text.push_str(&inner_text);
        }
        source = inner.source();
    }
    text
}

/// An address as it may be logged and stored: no user, no password, no query, no fragment.
pub fn shown(url: &Url) -> String {
    let mut plain = url.clone();
    // Neither setter can fail for an http(s) address with a host, which is all `parse_address` lets in.
    let _ = plain.set_username("");
    let _ = plain.set_password(None);
    plain.set_query(None);
    plain.set_fragment(None);
    plain.to_string()
}

/// A service's base address from a flag or the environment: http or https, with a host.
pub fn parse_address(name: &'static str, raw: &str) -> Result<Url, RunError> {
    let bad = |reason: &str| RunError::Setting {
        name,
        reason: reason.to_string(),
    };
    let url = Url::parse(raw.trim()).map_err(|e| bad(&format!("not an address: {e}")))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(bad("an address starts with http:// or https://"));
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(bad("an address names a host"));
    }
    Ok(url)
}

/// `base` followed by `segments`, each escaped as one path segment, whatever characters it
/// holds. A base with a path of its own (`http://host/kb`) keeps it.
pub fn endpoint(base: &Url, segments: &[&str]) -> Result<Url, RunError> {
    let mut url = base.clone();
    url.set_query(None);
    url.set_fragment(None);
    {
        let mut path = url.path_segments_mut().map_err(|()| RunError::Setting {
            name: "address",
            reason: format!("{} cannot carry a path", shown(base)),
        })?;
        path.pop_if_empty();
        path.extend(segments);
    }
    Ok(url)
}

/// The body of a successful answer as the contract's type. A body of another shape is an
/// error that keeps the start of the body.
pub fn parse_json<T: DeserializeOwned>(url: &Url, body: &[u8]) -> Result<T, RunError> {
    serde_json::from_slice(body).map_err(|e| RunError::Malformed {
        url: shown(url),
        reason: e.to_string(),
        body: excerpt(body),
    })
}

/// Milliseconds as run files carry them.
pub fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_address_is_shown_without_its_user_password_or_query() {
        let url = Url::parse("https://user:hunter2@model.example:8443/v1?api_key=sk-123#frag").unwrap();
        let plain = shown(&url);
        assert_eq!(plain, "https://model.example:8443/v1");
        assert!(!plain.contains("hunter2") && !plain.contains("sk-123"));
    }

    #[test]
    fn an_endpoint_keeps_the_base_path_and_escapes_each_segment() {
        let base = parse_address("CB_KB_URL", "http://127.0.0.1:8082/kb/").unwrap();
        let url = endpoint(&base, &["v1", "tenants", "a b/c", "ingest"]).unwrap();
        assert_eq!(
            url.as_str(),
            "http://127.0.0.1:8082/kb/v1/tenants/a%20b%2Fc/ingest"
        );
        let bare = parse_address("CB_KB_URL", "http://127.0.0.1:8082").unwrap();
        assert_eq!(
            endpoint(&bare, &["v1", "info"]).unwrap().as_str(),
            "http://127.0.0.1:8082/v1/info"
        );
    }

    #[test]
    fn only_http_addresses_with_a_host_are_taken() {
        assert!(parse_address("CB_KB_URL", "http://localhost:8082").is_ok());
        assert!(parse_address("CB_KB_URL", " https://kb.example/ ").is_ok());
        for bad in [
            "",
            "localhost:8082",
            "ftp://kb.example",
            "file:///etc/passwd",
            "http://",
        ] {
            assert!(parse_address("CB_KB_URL", bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn a_body_of_another_shape_is_an_error_that_keeps_its_start() {
        #[derive(Debug, serde::Deserialize)]
        struct Wanted {
            #[allow(dead_code)]
            count: u64,
        }
        let url = Url::parse("http://kb/v1/x?token=abc").unwrap();
        let error = parse_json::<Wanted>(&url, b"<html>busy</html>").unwrap_err();
        assert!(error.is_retryable());
        assert_eq!(error.raw_answer(), Some("<html>busy</html>"));
        assert_eq!(error.url(), Some("http://kb/v1/x"));
    }
}
