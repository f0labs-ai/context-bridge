//! Bounded retries with exponential back-off and jitter.
//!
//! One rule for every call this tool makes: a try that got no answer, a broken answer,
//! "too many requests" or a server-side failure is made again after a growing, randomised
//! wait, at most [`RetryPolicy::max_tries`] times in all. Any other refusal is final at once.

use crate::error::RunError;
use rand::Rng;
use std::future::Future;
use std::time::Duration;

/// How often a call is tried and how long it waits in between.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RetryPolicy {
    /// Tries in all, the first one included.
    pub max_tries: u32,
    /// The wait before the second try, before jitter; it doubles for each try after.
    pub base: Duration,
    /// The longest wait, whatever the doubling or the service asks for.
    pub cap: Duration,
}

impl RetryPolicy {
    /// The rule of docs/RUN_TOOLS.md: at most three tries.
    pub const STANDARD_TRIES: u32 = 3;

    pub fn standard(base: Duration) -> Self {
        Self {
            max_tries: Self::STANDARD_TRIES,
            base,
            cap: Duration::from_secs(30),
        }
    }

    /// One try and no more: for a call whose time is being measured.
    pub fn single() -> Self {
        Self {
            max_tries: 1,
            base: Duration::ZERO,
            cap: Duration::ZERO,
        }
    }

    /// The wait after `failed_tries` failed tries. `jitter` is a draw in `[0, 1)`: the wait
    /// lies between half and the whole of the doubled base, so two callers that failed
    /// together do not come back together. A service's own `Retry-After` is honoured when
    /// it is longer, up to the cap.
    pub fn wait_after(&self, failed_tries: u32, jitter: f64, retry_after: Option<Duration>) -> Duration {
        let doubled = self
            .base
            .saturating_mul(2u32.saturating_pow(failed_tries.saturating_sub(1)));
        let bounded = doubled.min(self.cap);
        let jittered = bounded.mul_f64(0.5 + 0.5 * jitter.clamp(0.0, 1.0));
        match retry_after {
            Some(asked) => jittered.max(asked.min(self.cap)),
            None => jittered,
        }
    }
}

/// What a call produced, and how many tries it took.
#[derive(Debug)]
pub struct Tried<T> {
    pub value: T,
    pub tries: u32,
}

/// A call that failed for good, and how many tries were made.
#[derive(Debug)]
pub struct GaveUp {
    pub error: RunError,
    pub tries: u32,
}

/// Makes `one_try` until it succeeds, fails in a way a retry cannot help, or the tries
/// run out. `one_try` is given the number of the try, starting at 1.
pub async fn with_retries<T, F, Fut>(policy: &RetryPolicy, mut one_try: F) -> Result<Tried<T>, GaveUp>
where
    F: FnMut(u32) -> Fut,
    Fut: Future<Output = Result<T, RunError>>,
{
    let mut tries = 0;
    loop {
        tries += 1;
        match one_try(tries).await {
            Ok(value) => return Ok(Tried { value, tries }),
            Err(error) if error.is_retryable() && tries < policy.max_tries => {
                let asked = error.retry_after_s().map(Duration::from_secs);
                let wait = policy.wait_after(tries, rand::rng().random::<f64>(), asked);
                tracing::debug!(
                    tries,
                    wait_ms = wait.as_millis() as u64,
                    kind = error.kind().as_str(),
                    status = error.status(),
                    "try_failed_will_retry"
                );
                // A back-off wait between two tries, not a way of waiting for another task.
                tokio::time::sleep(wait).await;
            }
            Err(error) => return Err(GaveUp { error, tries }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn status(code: u16) -> RunError {
        RunError::Status {
            url: "http://kb".into(),
            status: code,
            body: String::new(),
            retry_after_s: None,
        }
    }

    fn fast() -> RetryPolicy {
        RetryPolicy {
            max_tries: 3,
            base: Duration::from_millis(1),
            cap: Duration::from_millis(4),
        }
    }

    #[test]
    fn the_wait_doubles_stays_within_half_and_whole_and_never_passes_the_cap() {
        let policy = RetryPolicy {
            max_tries: 5,
            base: Duration::from_millis(100),
            cap: Duration::from_millis(350),
        };
        assert_eq!(policy.wait_after(1, 0.0, None), Duration::from_millis(50));
        assert_eq!(policy.wait_after(1, 1.0, None), Duration::from_millis(100));
        assert_eq!(policy.wait_after(2, 1.0, None), Duration::from_millis(200));
        assert_eq!(policy.wait_after(3, 1.0, None), Duration::from_millis(350));
        assert_eq!(policy.wait_after(30, 1.0, None), Duration::from_millis(350));
        // The service's own request wins when it is longer, up to the cap.
        assert_eq!(
            policy.wait_after(1, 0.0, Some(Duration::from_millis(300))),
            Duration::from_millis(300)
        );
        assert_eq!(
            policy.wait_after(1, 0.0, Some(Duration::from_secs(3600))),
            Duration::from_millis(350)
        );
    }

    #[tokio::test]
    async fn a_call_that_fails_twice_then_works_takes_three_tries() {
        let calls = Cell::new(0u32);
        let result = with_retries(&fast(), |_| {
            calls.set(calls.get() + 1);
            let n = calls.get();
            async move { if n < 3 { Err(status(503)) } else { Ok("done") } }
        })
        .await
        .unwrap();
        assert_eq!((result.value, result.tries), ("done", 3));
    }

    #[tokio::test]
    async fn tries_are_bounded_and_the_last_error_is_kept() {
        let calls = Cell::new(0u32);
        let gave_up = with_retries(&fast(), |_| {
            calls.set(calls.get() + 1);
            async { Err::<(), _>(status(429)) }
        })
        .await
        .unwrap_err();
        assert_eq!((calls.get(), gave_up.tries), (3, 3));
        assert_eq!(gave_up.error.status(), Some(429));
    }

    #[tokio::test]
    async fn a_refusal_is_never_tried_again() {
        let calls = Cell::new(0u32);
        let gave_up = with_retries(&fast(), |_| {
            calls.set(calls.get() + 1);
            async { Err::<(), _>(status(400)) }
        })
        .await
        .unwrap_err();
        assert_eq!((calls.get(), gave_up.tries), (1, 1));
    }

    #[tokio::test]
    async fn a_single_try_policy_makes_one_try() {
        let calls = Cell::new(0u32);
        let gave_up = with_retries(&RetryPolicy::single(), |_| {
            calls.set(calls.get() + 1);
            async { Err::<(), _>(status(503)) }
        })
        .await
        .unwrap_err();
        assert_eq!((calls.get(), gave_up.tries), (1, 1));
    }
}
