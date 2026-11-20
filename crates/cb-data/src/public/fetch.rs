//! A polite, resumable downloader for the public sources.
//!
//! One request at a time and never faster than the configured pace. Every
//! request has a timeout, a 429 or 5xx answer is retried a bounded number of
//! times with a growing wait (or the wait the server asks for), and nothing is
//! ever followed: a URL is requested only when it starts with one of the
//! prefixes the caller listed, and a redirect is an error, not a hop.
//!
//! A file that is already on disk is not requested again, so a fetch that was
//! interrupted picks up where it stopped. A download is written beside its
//! final name and renamed when complete, so a file that exists is whole.

use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;
use tokio::time::Instant;

/// Sent with every request: who is asking and why.
pub const USER_AGENT: &str = concat!(
    "cb-data/",
    env!("CARGO_PKG_VERSION"),
    " (ContextBridge proof; academic research on time-aware retrieval over public incident timelines;",
    " results published open access)"
);

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("{url}: not one of the addresses this fetch was told to use")]
    NotListed { url: String },
    #[error("{url}: answered {status}")]
    Status { url: String, status: u16 },
    #[error("{url}: answered {status} pointing to {location:?}; redirects are not followed")]
    Redirect {
        url: String,
        status: u16,
        location: Option<String>,
    },
    #[error("{url}: gave up after {attempts} attempts; last: {last}")]
    GaveUp {
        url: String,
        attempts: u32,
        last: String,
    },
    #[error("{url}: the server asked to wait {asked_s} s, longer than the {limit_s} s this fetch will wait")]
    WaitTooLong { url: String, asked_s: u64, limit_s: u64 },
    #[error("{url}: {source}")]
    Transport {
        url: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("building the HTTP client: {0}")]
    Client(#[source] reqwest::Error),
}

/// How the fetcher behaves towards a server.
#[derive(Clone, Debug)]
pub struct Politeness {
    pub user_agent: String,
    /// Least time between the starts of two requests.
    pub min_interval: Duration,
    /// Longest a single request may take, connection to last byte.
    pub timeout: Duration,
    /// Requests made for one URL before giving up.
    pub max_attempts: u32,
    /// Wait after the first failed attempt; doubles each time.
    pub backoff_base: Duration,
    /// Longest wait between two attempts when the server names none.
    pub backoff_cap: Duration,
    /// Longest wait a server may ask for (`Retry-After`) before the fetch gives up.
    pub max_retry_after: Duration,
}

impl Default for Politeness {
    /// At most two requests a second.
    fn default() -> Self {
        Self {
            user_agent: USER_AGENT.to_string(),
            min_interval: Duration::from_millis(500),
            timeout: Duration::from_secs(60),
            max_attempts: 5,
            backoff_base: Duration::from_secs(2),
            backoff_cap: Duration::from_secs(60),
            max_retry_after: Duration::from_secs(600),
        }
    }
}

impl Politeness {
    /// The wait before attempt `attempt + 1` when the server names none.
    pub fn backoff(&self, attempt: u32) -> Duration {
        let doubled = self
            .backoff_base
            .saturating_mul(1u32 << attempt.saturating_sub(1).min(16));
        doubled.min(self.backoff_cap)
    }
}

/// What `get_to_file` did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The file was already there; no request was made.
    AlreadyThere,
    Downloaded {
        bytes: u64,
    },
}

/// What one attempt came to.
enum Attempt {
    Body(Vec<u8>),
    /// Worth another try, after `wait` when the server named one.
    Again {
        why: String,
        wait: Option<Duration>,
    },
}

pub struct Fetcher {
    client: reqwest::Client,
    rules: Politeness,
    allowed: Vec<String>,
    next_slot: Option<Instant>,
    requests: u64,
}

impl Fetcher {
    /// `allowed` lists the URL prefixes this fetcher may request. Anything else is refused.
    pub fn new(rules: Politeness, allowed: impl IntoIterator<Item = String>) -> Result<Self, FetchError> {
        let client = reqwest::Client::builder()
            .user_agent(rules.user_agent.clone())
            .timeout(rules.timeout)
            .connect_timeout(rules.timeout.min(Duration::from_secs(15)))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(FetchError::Client)?;
        Ok(Self {
            client,
            rules,
            allowed: allowed.into_iter().collect(),
            next_slot: None,
            requests: 0,
        })
    }

    /// Requests sent so far, retries included.
    pub fn requests(&self) -> u64 {
        self.requests
    }

    fn check_listed(&self, url: &str) -> Result<(), FetchError> {
        if self.allowed.iter().any(|prefix| url.starts_with(prefix.as_str())) {
            Ok(())
        } else {
            Err(FetchError::NotListed { url: url.to_string() })
        }
    }

    /// Wait for the next free slot, then claim the one after it.
    async fn pace(&mut self) {
        if let Some(slot) = self.next_slot {
            tokio::time::sleep_until(slot).await;
        }
        self.next_slot = Some(Instant::now() + self.rules.min_interval);
    }

    /// The body of a 200 answer to `GET url`.
    pub async fn get(&mut self, url: &str) -> Result<Vec<u8>, FetchError> {
        self.check_listed(url)?;
        let mut last = String::new();
        for attempt in 1..=self.rules.max_attempts {
            self.pace().await;
            self.requests += 1;
            match self.attempt(url).await? {
                Attempt::Body(body) => return Ok(body),
                Attempt::Again { why, wait } => {
                    last = why;
                    if attempt == self.rules.max_attempts {
                        break;
                    }
                    let wait = match wait {
                        Some(asked) if asked > self.rules.max_retry_after => {
                            return Err(FetchError::WaitTooLong {
                                url: url.to_string(),
                                asked_s: asked.as_secs(),
                                limit_s: self.rules.max_retry_after.as_secs(),
                            });
                        }
                        Some(asked) => asked,
                        None => self.rules.backoff(attempt),
                    };
                    // Back-off: the wait is the point, nothing is being synchronised.
                    tokio::time::sleep(wait).await;
                }
            }
        }
        Err(FetchError::GaveUp {
            url: url.to_string(),
            attempts: self.rules.max_attempts,
            last,
        })
    }

    async fn attempt(&self, url: &str) -> Result<Attempt, FetchError> {
        let response = match self.client.get(url).send().await {
            Ok(response) => response,
            Err(source) if source.is_timeout() || source.is_connect() || source.is_request() => {
                return Ok(Attempt::Again {
                    why: describe(&source),
                    wait: None,
                });
            }
            Err(source) => {
                return Err(FetchError::Transport {
                    url: url.to_string(),
                    source,
                });
            }
        };
        let status = response.status();
        if status == reqwest::StatusCode::OK {
            return match response.bytes().await {
                Ok(body) => Ok(Attempt::Body(body.to_vec())),
                // The connection dropped or timed out part-way through the body.
                Err(source) => Ok(Attempt::Again {
                    why: describe(&source),
                    wait: None,
                }),
            };
        }
        if status.is_redirection() {
            let location = response
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            return Err(FetchError::Redirect {
                url: url.to_string(),
                status: status.as_u16(),
                location,
            });
        }
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error() {
            let wait = response
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.trim().parse::<u64>().ok())
                .map(Duration::from_secs);
            return Ok(Attempt::Again {
                why: format!("answered {}", status.as_u16()),
                wait,
            });
        }
        Err(FetchError::Status {
            url: url.to_string(),
            status: status.as_u16(),
        })
    }

    /// Download `url` to `path` unless the file is already there.
    pub async fn get_to_file(&mut self, url: &str, path: &Path) -> Result<Outcome, FetchError> {
        if is_whole_file(path).await {
            return Ok(Outcome::AlreadyThere);
        }
        let body = self.get(url).await?;
        write_whole(path, &body).await?;
        Ok(Outcome::Downloaded {
            bytes: body.len() as u64,
        })
    }
}

fn describe(error: &reqwest::Error) -> String {
    if error.is_timeout() {
        "no answer in time".to_string()
    } else if error.is_connect() {
        "could not connect".to_string()
    } else {
        // reqwest's own text can hold the URL; keep the kind only.
        "the connection failed".to_string()
    }
}

/// Whether a finished download is at `path`.
pub async fn is_whole_file(path: &Path) -> bool {
    match tokio::fs::metadata(path).await {
        Ok(meta) => meta.is_file() && meta.len() > 0,
        Err(_) => false,
    }
}

/// The name a download has while it is still being written.
fn part_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".part");
    path.with_file_name(name)
}

/// Write `body` to `path` so that the file either holds all of it or does not exist.
pub async fn write_whole(path: &Path, body: &[u8]) -> Result<(), FetchError> {
    let io = |p: &Path| {
        let p = p.to_path_buf();
        move |source| FetchError::Io { path: p, source }
    };
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(io(parent))?;
    }
    let part = part_path(path);
    tokio::fs::write(&part, body).await.map_err(io(&part))?;
    tokio::fs::rename(&part, path).await.map_err(io(path))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A scripted answer of the stub server.
    #[derive(Clone)]
    enum Reply {
        Status(u16, Vec<(&'static str, String)>, &'static str),
        /// Accept the request and never answer.
        Silence,
    }

    struct Stub {
        base: String,
        seen: Arc<Mutex<Vec<String>>>,
    }

    impl Stub {
        fn seen(&self) -> Vec<String> {
            self.seen.lock().unwrap().clone()
        }
    }

    /// A server on a local socket that answers each request with the next reply of `script`
    /// (the last reply repeats) and records the request paths.
    async fn stub(script: Vec<Reply>) -> Stub {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen_by_server = Arc::clone(&seen);
        let script = Arc::new(Mutex::new(VecDeque::from(script)));
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let seen = Arc::clone(&seen_by_server);
                let script = Arc::clone(&script);
                tokio::spawn(async move {
                    let mut head = Vec::new();
                    let mut buf = [0u8; 1024];
                    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
                        match socket.read(&mut buf).await {
                            Ok(0) | Err(_) => return,
                            Ok(n) => head.extend_from_slice(&buf[..n]),
                        }
                    }
                    let head = String::from_utf8_lossy(&head).to_string();
                    let path = head.split_whitespace().nth(1).unwrap_or("").to_string();
                    let agent = head
                        .lines()
                        .find_map(|l| {
                            l.strip_prefix("user-agent: ")
                                .or_else(|| l.strip_prefix("User-Agent: "))
                        })
                        .unwrap_or("")
                        .to_string();
                    seen.lock().unwrap().push(format!("{path} | {agent}"));
                    let reply = {
                        let mut script = script.lock().unwrap();
                        if script.len() > 1 {
                            script.pop_front().unwrap()
                        } else {
                            script.front().cloned().unwrap()
                        }
                    };
                    match reply {
                        Reply::Silence => {
                            // Hold the connection open until the client hangs up.
                            let _ = socket.read(&mut buf).await;
                        }
                        Reply::Status(code, headers, body) => {
                            let mut out = format!("HTTP/1.1 {code} X\r\ncontent-length: {}\r\n", body.len());
                            for (name, value) in headers {
                                out.push_str(&format!("{name}: {value}\r\n"));
                            }
                            out.push_str("connection: close\r\n\r\n");
                            out.push_str(body);
                            let _ = socket.write_all(out.as_bytes()).await;
                            let _ = socket.shutdown().await;
                        }
                    }
                });
            }
        });
        Stub { base, seen }
    }

    fn quick() -> Politeness {
        Politeness {
            user_agent: "cb-data-test".into(),
            min_interval: Duration::from_millis(1),
            timeout: Duration::from_secs(5),
            max_attempts: 4,
            backoff_base: Duration::from_millis(5),
            backoff_cap: Duration::from_millis(20),
            max_retry_after: Duration::from_secs(2),
        }
    }

    fn ok(body: &'static str) -> Reply {
        Reply::Status(200, vec![], body)
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("cb-data-fetch-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[tokio::test]
    async fn a_503_then_a_429_are_retried_until_the_answer_comes() {
        let server = stub(vec![
            Reply::Status(503, vec![], "busy"),
            Reply::Status(429, vec![("retry-after", "0".into())], "slow down"),
            ok("{\"ok\":true}"),
        ])
        .await;
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let body = fetcher
            .get(&format!("{}/history.json?page=1", server.base))
            .await
            .unwrap();
        assert_eq!(body, b"{\"ok\":true}");
        assert_eq!(fetcher.requests(), 3);
        assert_eq!(server.seen().len(), 3);
        assert!(server.seen()[0].starts_with("/history.json?page=1 | cb-data-test"));
    }

    #[tokio::test]
    async fn a_server_that_stays_down_ends_in_an_error_naming_the_url() {
        let server = stub(vec![Reply::Status(500, vec![], "down")]).await;
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let url = format!("{}/incidents/abc.json", server.base);
        let error = fetcher.get(&url).await.unwrap_err();
        assert!(
            matches!(&error, FetchError::GaveUp { attempts: 4, last, .. } if last == "answered 500"),
            "{error}"
        );
        assert!(error.to_string().contains(&url));
        assert_eq!(server.seen().len(), 4);
    }

    #[tokio::test]
    async fn a_404_is_not_retried() {
        let server = stub(vec![Reply::Status(404, vec![], "gone")]).await;
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let error = fetcher
            .get(&format!("{}/missing", server.base))
            .await
            .unwrap_err();
        assert!(matches!(error, FetchError::Status { status: 404, .. }));
        assert_eq!(server.seen().len(), 1);
    }

    #[tokio::test]
    async fn a_redirect_is_an_error_and_is_never_followed() {
        let server = stub(vec![Reply::Status(
            302,
            vec![("location", "http://127.0.0.1:9/elsewhere".into())],
            "",
        )])
        .await;
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let error = fetcher.get(&format!("{}/moved", server.base)).await.unwrap_err();
        assert!(matches!(
            &error,
            FetchError::Redirect { status: 302, location: Some(l), .. } if l == "http://127.0.0.1:9/elsewhere"
        ));
        assert_eq!(server.seen().len(), 1);
    }

    #[tokio::test]
    async fn a_url_that_was_not_listed_is_refused_without_a_request() {
        let server = stub(vec![ok("x")]).await;
        let mut fetcher = Fetcher::new(quick(), [format!("{}/incidents/", server.base)]).unwrap();
        let error = fetcher
            .get(&format!("{}/api/v2/summary.json", server.base))
            .await
            .unwrap_err();
        assert!(matches!(error, FetchError::NotListed { .. }));
        let error = fetcher
            .get("https://example.org/incidents/x.json")
            .await
            .unwrap_err();
        assert!(matches!(error, FetchError::NotListed { .. }));
        assert!(server.seen().is_empty());
        assert_eq!(fetcher.requests(), 0);
    }

    #[tokio::test]
    async fn a_request_that_gets_no_answer_times_out_and_is_retried() {
        let server = stub(vec![Reply::Silence, ok("late but here")]).await;
        let rules = Politeness {
            timeout: Duration::from_millis(150),
            ..quick()
        };
        let mut fetcher = Fetcher::new(rules, [server.base.clone()]).unwrap();
        let body = fetcher.get(&format!("{}/slow", server.base)).await.unwrap();
        assert_eq!(body, b"late but here");
        assert_eq!(fetcher.requests(), 2);
    }

    #[tokio::test]
    async fn a_wait_longer_than_the_limit_stops_the_fetch() {
        let server = stub(vec![Reply::Status(429, vec![("retry-after", "3600".into())], "")]).await;
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let error = fetcher.get(&format!("{}/x", server.base)).await.unwrap_err();
        assert!(matches!(
            error,
            FetchError::WaitTooLong {
                asked_s: 3600,
                limit_s: 2,
                ..
            }
        ));
        assert_eq!(server.seen().len(), 1);
    }

    #[tokio::test]
    async fn requests_keep_the_pace() {
        let server = stub(vec![ok("x")]).await;
        let rules = Politeness {
            min_interval: Duration::from_millis(60),
            ..quick()
        };
        let mut fetcher = Fetcher::new(rules, [server.base.clone()]).unwrap();
        let started = std::time::Instant::now();
        for i in 0..4 {
            fetcher.get(&format!("{}/p{i}", server.base)).await.unwrap();
        }
        // Four requests leave three gaps of at least 60 ms.
        assert!(
            started.elapsed() >= Duration::from_millis(180),
            "{:?}",
            started.elapsed()
        );
    }

    #[tokio::test]
    async fn a_file_already_on_disk_is_not_requested_again() {
        let server = stub(vec![ok("first")]).await;
        let dir = temp_dir("resume");
        let path = dir.join("incidents").join("abc.json");
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let url = format!("{}/incidents/abc.json", server.base);
        assert_eq!(
            fetcher.get_to_file(&url, &path).await.unwrap(),
            Outcome::Downloaded { bytes: 5 }
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"first");
        assert!(!part_path(&path).exists());
        assert_eq!(
            fetcher.get_to_file(&url, &path).await.unwrap(),
            Outcome::AlreadyThere
        );
        assert_eq!(server.seen().len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn a_half_written_download_does_not_count_as_the_file() {
        let server = stub(vec![ok("whole")]).await;
        let dir = temp_dir("part");
        let path = dir.join("page-0001.json");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(part_path(&path), b"wh").unwrap();
        let mut fetcher = Fetcher::new(quick(), [server.base.clone()]).unwrap();
        let outcome = fetcher
            .get_to_file(&format!("{}/h", server.base), &path)
            .await
            .unwrap();
        assert_eq!(outcome, Outcome::Downloaded { bytes: 5 });
        assert_eq!(std::fs::read(&path).unwrap(), b"whole");
        assert!(!part_path(&path).exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_wait_doubles_up_to_the_cap() {
        let rules = Politeness::default();
        assert_eq!(rules.backoff(1), Duration::from_secs(2));
        assert_eq!(rules.backoff(2), Duration::from_secs(4));
        assert_eq!(rules.backoff(3), Duration::from_secs(8));
        assert_eq!(rules.backoff(9), Duration::from_secs(60));
        assert_eq!(rules.min_interval, Duration::from_millis(500));
        assert!(USER_AGENT.contains("research"));
    }
}
