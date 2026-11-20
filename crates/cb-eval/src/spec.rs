//! The ranking methods under test, as typed values (DESIGN.md section 4).
//!
//! A [`MethodSpec`] has three spellings:
//!
//! - its serde form, used inside files this crate writes;
//! - a compact text form for the command line, `rrf_td(k=60,h=1800)`
//!   ([`std::fmt::Display`] and [`std::str::FromStr`]);
//! - a [`MethodSpec::name`] that is safe as a file name, `rrf_td-k60-h1800`.
//!
//! A parameter that DESIGN.md gives a default for is left out of the text form
//! and the name while it has that default, so the paper's own setting is always
//! `rrf_td(k=60,h=1800)`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;
use thiserror::Error;

/// How many records each ranker contributes in the paper's formula.
pub const PAPER_LIST: u32 = 20;
/// The floor under the decay in the paper's formula.
pub const PAPER_EPS: f64 = 1e-6;
/// The cold-start threshold of the paper's formula, seconds.
pub const PAPER_THETA: f64 = 60.0;
/// The paper's grid of the fusion constant.
pub const PAPER_K_GRID: [u32; 4] = [10, 30, 60, 100];
/// The paper's grid of half-lives, seconds (1, 5, 30, 120, 720 and 1440 minutes).
pub const PAPER_H_GRID: [u64; 6] = [60, 300, 1800, 7200, 43_200, 86_400];
/// The fusion constant where DESIGN.md names a default.
pub const DEFAULT_K: u32 = 60;
/// How many merged records the product sends to its relevance model, as deployed.
pub const DEFAULT_DEPTH: u32 = 10;
/// The published weight of the score against recency in `tw_cs_rrf`.
pub const DEFAULT_LAMBDA: f64 = 0.9;

/// One third each: the paper's weights for the dense, keyword and time rankers.
pub fn equal_thirds() -> [f64; 3] {
    [1.0 / 3.0; 3]
}

/// The parameters of the paper's formula (Algorithm 1) and its reorder variant.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RrfTdParams {
    /// The fusion constant.
    pub k: u32,
    /// The half-life of the decay, seconds.
    pub h: u64,
    /// Weights of the dense, keyword and time rankers.
    pub weights: [f64; 3],
    /// How many records each ranker contributes.
    pub list: u32,
    /// The floor under the decay.
    pub eps: f64,
    /// Decay is applied only when the candidates' `written_at` spread exceeds this, seconds.
    pub theta: f64,
}

impl RrfTdParams {
    /// The paper's defaults around a chosen `k` and half-life.
    pub fn paper(k: u32, h: u64) -> Self {
        Self {
            k,
            h,
            weights: equal_thirds(),
            list: PAPER_LIST,
            eps: PAPER_EPS,
            theta: PAPER_THETA,
        }
    }
}

/// A ranking method with its parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MethodSpec {
    /// The dense list as given.
    Dense,
    /// Reciprocal rank fusion of the dense and keyword lists, weights one half each.
    Rrf { k: u32 },
    /// Dense records scored `cos * decay_h`.
    DecayOnly { h: u64 },
    /// `dense` on an index that refreshes every `period` seconds.
    Batch { period: i64 },
    /// `dense` on records at most `window` seconds old; `dense` when fewer than 5 are left.
    Cutoff { window: i64 },
    /// `alpha * mm(cos) + (1 - alpha) * mm(bm25)`.
    Cc { alpha: f64 },
    /// The paper's formula, Algorithm 1 as printed.
    RrfTd(RrfTdParams),
    /// The paper's formula where the time ranker only reorders the relevance candidates.
    RrfTdReorder(RrfTdParams),
    /// The paper's three rankers without the decay.
    Rrf3 { k: u32, weights: [f64; 3], list: u32 },
    /// `rrf_k([D, S]) + beta * (1 / (k + 1)) * 2^(-age / h)`.
    RrfAdd { beta: f64, h: u64, k: u32 },
    /// Per list, `(lambda * mm(score) + (1 - lambda) * recency) / (k + rank)`.
    TwCsRrf { lambda: f64, k: u32 },
    /// The first `n` dense records, newest first.
    NewestFirst { n: u32 },
    /// What the service returns today. Every `product*` method also takes `recent`: the
    /// first `recent` records of the newest list join its candidates (DESIGN.md
    /// section 4.4). With 0, the default, it is left out of every spelling.
    Product {
        depth: u32,
        #[serde(default, skip_serializing_if = "is_zero")]
        recent: u32,
    },
    /// As `product`, scored `rho * decay_h`.
    ProductDecay {
        h: u64,
        depth: u32,
        #[serde(default, skip_serializing_if = "is_zero")]
        recent: u32,
    },
    /// As `product`, scored `rho * (1 + beta * 2^(-age / h))`.
    ProductBonus {
        beta: f64,
        h: u64,
        depth: u32,
        #[serde(default, skip_serializing_if = "is_zero")]
        recent: u32,
    },
    /// As `product`, without records whose stated end has passed.
    ProductEnds {
        depth: u32,
        #[serde(default, skip_serializing_if = "is_zero")]
        recent: u32,
    },
    /// As `product`, where a newer record that says the same thing replaces the older one.
    ProductNewerWins {
        tau: f64,
        depth: u32,
        #[serde(default, skip_serializing_if = "is_zero")]
        recent: u32,
    },
    /// **Keyed.** Only the newest visible record of each thread is a candidate, then `product`.
    KeyedLatest {
        depth: u32,
        #[serde(default, skip_serializing_if = "is_zero")]
        recent: u32,
    },
    /// The gold order. Reads the answers; a ceiling, not a method.
    Oracle,
}

fn is_zero(n: &u32) -> bool {
    *n == 0
}

#[derive(Debug, Error, PartialEq)]
pub enum SpecError {
    #[error("\"{0}\" is not a method (known: {known})", known = KINDS.join(", "))]
    UnknownMethod(String),
    #[error("\"{text}\" is not a method: {reason}")]
    Syntax { text: String, reason: &'static str },
    #[error("{method}: \"{key}\" is not one of its parameters")]
    UnknownParameter { method: String, key: String },
    #[error("{method}: parameter \"{key}\" is given twice")]
    RepeatedParameter { method: String, key: String },
    #[error("{method}: parameter \"{key}\" is required")]
    MissingParameter { method: String, key: &'static str },
    #[error("{method}: \"{value}\" is not a value for \"{key}\"")]
    BadValue {
        method: String,
        key: String,
        value: String,
    },
    #[error("{method}: {key} must be {rule}")]
    OutOfRange {
        method: &'static str,
        key: &'static str,
        rule: &'static str,
    },
}

/// Every method kind, in the order of DESIGN.md section 4.
pub const KINDS: [&str; 19] = [
    "dense",
    "rrf",
    "decay_only",
    "batch",
    "cutoff",
    "cc",
    "rrf_td",
    "rrf_td_reorder",
    "rrf3",
    "rrf_add",
    "tw_cs_rrf",
    "newest_first",
    "product",
    "product_decay",
    "product_bonus",
    "product_ends",
    "product_newer_wins",
    "keyed_latest",
    "oracle",
];

impl MethodSpec {
    /// The method without its parameters, e.g. `rrf_td`.
    pub fn kind(&self) -> &'static str {
        match self {
            MethodSpec::Dense => "dense",
            MethodSpec::Rrf { .. } => "rrf",
            MethodSpec::DecayOnly { .. } => "decay_only",
            MethodSpec::Batch { .. } => "batch",
            MethodSpec::Cutoff { .. } => "cutoff",
            MethodSpec::Cc { .. } => "cc",
            MethodSpec::RrfTd(_) => "rrf_td",
            MethodSpec::RrfTdReorder(_) => "rrf_td_reorder",
            MethodSpec::Rrf3 { .. } => "rrf3",
            MethodSpec::RrfAdd { .. } => "rrf_add",
            MethodSpec::TwCsRrf { .. } => "tw_cs_rrf",
            MethodSpec::NewestFirst { .. } => "newest_first",
            MethodSpec::Product { .. } => "product",
            MethodSpec::ProductDecay { .. } => "product_decay",
            MethodSpec::ProductBonus { .. } => "product_bonus",
            MethodSpec::ProductEnds { .. } => "product_ends",
            MethodSpec::ProductNewerWins { .. } => "product_newer_wins",
            MethodSpec::KeyedLatest { .. } => "keyed_latest",
            MethodSpec::Oracle => "oracle",
        }
    }

    /// Whether the method reads `Record.thread` (DESIGN.md: "keyed").
    pub fn is_keyed(&self) -> bool {
        matches!(self, MethodSpec::KeyedLatest { .. })
    }

    /// Whether the method reads the graded answers. Only the ceiling does.
    pub fn reads_gold(&self) -> bool {
        matches!(self, MethodSpec::Oracle)
    }

    /// Whether the method compares passage vectors.
    pub fn needs_vectors(&self) -> bool {
        matches!(self, MethodSpec::ProductNewerWins { .. })
    }

    /// How many records of the newest list join the method's candidates; 0 for a
    /// method that never reads the list.
    pub fn recent(&self) -> u32 {
        match *self {
            MethodSpec::Product { recent, .. }
            | MethodSpec::ProductDecay { recent, .. }
            | MethodSpec::ProductBonus { recent, .. }
            | MethodSpec::ProductEnds { recent, .. }
            | MethodSpec::ProductNewerWins { recent, .. }
            | MethodSpec::KeyedLatest { recent, .. } => recent,
            _ => 0,
        }
    }

    /// Whether the method reads the newest list (`newest.jsonl`).
    pub fn needs_newest(&self) -> bool {
        self.recent() > 0
    }

    /// The parameters that appear in the text form and the name, in a fixed order.
    fn shown(&self) -> Vec<(&'static str, String)> {
        let mut out: Vec<(&'static str, String)> = Vec::new();
        // A product method's pool: `depth` merged records, and `recent` newest ones.
        let depth = |out: &mut Vec<(&'static str, String)>, depth: u32, recent: u32| {
            if depth != DEFAULT_DEPTH {
                out.push(("depth", depth.to_string()));
            }
            if recent != 0 {
                out.push(("recent", recent.to_string()));
            }
        };
        let fusion_k = |out: &mut Vec<(&'static str, String)>, k: u32| {
            if k != DEFAULT_K {
                out.push(("k", k.to_string()));
            }
        };
        let weights_and_list = |out: &mut Vec<(&'static str, String)>, weights: &[f64; 3], list: u32| {
            if *weights != equal_thirds() {
                out.push(("w", format!("{}:{}:{}", weights[0], weights[1], weights[2])));
            }
            if list != PAPER_LIST {
                out.push(("list", list.to_string()));
            }
        };
        match self {
            MethodSpec::Dense | MethodSpec::Oracle => {}
            MethodSpec::Rrf { k } => out.push(("k", k.to_string())),
            MethodSpec::DecayOnly { h } => out.push(("h", h.to_string())),
            MethodSpec::Batch { period } => out.push(("period", period.to_string())),
            MethodSpec::Cutoff { window } => out.push(("window", window.to_string())),
            MethodSpec::Cc { alpha } => out.push(("alpha", alpha.to_string())),
            MethodSpec::RrfTd(p) | MethodSpec::RrfTdReorder(p) => {
                out.push(("k", p.k.to_string()));
                out.push(("h", p.h.to_string()));
                weights_and_list(&mut out, &p.weights, p.list);
                if p.eps != PAPER_EPS {
                    out.push(("eps", p.eps.to_string()));
                }
                if p.theta != PAPER_THETA {
                    out.push(("theta", p.theta.to_string()));
                }
            }
            MethodSpec::Rrf3 { k, weights, list } => {
                out.push(("k", k.to_string()));
                weights_and_list(&mut out, weights, *list);
            }
            MethodSpec::RrfAdd { beta, h, k } => {
                out.push(("beta", beta.to_string()));
                out.push(("h", h.to_string()));
                fusion_k(&mut out, *k);
            }
            MethodSpec::TwCsRrf { lambda, k } => {
                out.push(("lambda", lambda.to_string()));
                fusion_k(&mut out, *k);
            }
            MethodSpec::NewestFirst { n } => out.push(("n", n.to_string())),
            MethodSpec::Product { depth: d, recent: r }
            | MethodSpec::ProductEnds { depth: d, recent: r }
            | MethodSpec::KeyedLatest { depth: d, recent: r } => depth(&mut out, *d, *r),
            MethodSpec::ProductDecay {
                h,
                depth: d,
                recent: r,
            } => {
                out.push(("h", h.to_string()));
                depth(&mut out, *d, *r);
            }
            MethodSpec::ProductBonus {
                beta,
                h,
                depth: d,
                recent: r,
            } => {
                out.push(("beta", beta.to_string()));
                out.push(("h", h.to_string()));
                depth(&mut out, *d, *r);
            }
            MethodSpec::ProductNewerWins {
                tau,
                depth: d,
                recent: r,
            } => {
                out.push(("tau", tau.to_string()));
                depth(&mut out, *d, *r);
            }
        }
        out
    }

    /// A stable name made of `a-z`, `0-9`, `_`, `-` and `.` only: the method
    /// kind, then `-<parameter><value>` for each shown parameter. Used for
    /// file names, table rows and the keys of `numbers.tex`.
    pub fn name(&self) -> String {
        let mut name = self.kind().to_string();
        for (key, value) in self.shown() {
            name.push('-');
            name.push_str(key);
            name.push_str(&value.replace(':', "_"));
        }
        name
    }

    /// Every parameter of the method, defaults included, in a fixed order, as
    /// numbers: the parameter columns of the figure files. The three weights
    /// of the paper's formula are `w_dense`, `w_sparse` and `w_time`.
    pub fn parameters(&self) -> Vec<(&'static str, f64)> {
        let weights = |w: [f64; 3]| [("w_dense", w[0]), ("w_sparse", w[1]), ("w_time", w[2])];
        match *self {
            MethodSpec::Dense | MethodSpec::Oracle => Vec::new(),
            MethodSpec::Rrf { k } => vec![("k", f64::from(k))],
            MethodSpec::DecayOnly { h } => vec![("h", h as f64)],
            MethodSpec::Batch { period } => vec![("period", period as f64)],
            MethodSpec::Cutoff { window } => vec![("window", window as f64)],
            MethodSpec::Cc { alpha } => vec![("alpha", alpha)],
            MethodSpec::RrfTd(p) | MethodSpec::RrfTdReorder(p) => {
                let mut out = vec![("k", f64::from(p.k)), ("h", p.h as f64)];
                out.extend(weights(p.weights));
                out.extend([("list", f64::from(p.list)), ("eps", p.eps), ("theta", p.theta)]);
                out
            }
            MethodSpec::Rrf3 { k, weights: w, list } => {
                let mut out = vec![("k", f64::from(k))];
                out.extend(weights(w));
                out.push(("list", f64::from(list)));
                out
            }
            MethodSpec::RrfAdd { beta, h, k } => vec![("beta", beta), ("h", h as f64), ("k", f64::from(k))],
            MethodSpec::TwCsRrf { lambda, k } => vec![("lambda", lambda), ("k", f64::from(k))],
            MethodSpec::NewestFirst { n } => vec![("n", f64::from(n))],
            MethodSpec::Product { depth, recent }
            | MethodSpec::ProductEnds { depth, recent }
            | MethodSpec::KeyedLatest { depth, recent } => {
                vec![("depth", f64::from(depth)), ("recent", f64::from(recent))]
            }
            MethodSpec::ProductDecay { h, depth, recent } => {
                vec![
                    ("h", h as f64),
                    ("depth", f64::from(depth)),
                    ("recent", f64::from(recent)),
                ]
            }
            MethodSpec::ProductBonus {
                beta,
                h,
                depth,
                recent,
            } => vec![
                ("beta", beta),
                ("h", h as f64),
                ("depth", f64::from(depth)),
                ("recent", f64::from(recent)),
            ],
            MethodSpec::ProductNewerWins { tau, depth, recent } => {
                vec![
                    ("tau", tau),
                    ("depth", f64::from(depth)),
                    ("recent", f64::from(recent)),
                ]
            }
        }
    }

    /// The order methods are listed in: by kind in the order of DESIGN.md
    /// section 4, then by parameter values, first parameter first.
    pub fn cmp_listing(&self, other: &MethodSpec) -> std::cmp::Ordering {
        let place = |spec: &MethodSpec| KINDS.iter().position(|kind| *kind == spec.kind());
        place(self).cmp(&place(other)).then_with(|| {
            self.parameters()
                .iter()
                .zip(other.parameters())
                .map(|((_, a), (_, b))| a.total_cmp(&b))
                .find(|order| order.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Check every parameter against its range.
    pub fn validate(&self) -> Result<(), SpecError> {
        let method = self.kind();
        let out_of_range = |key, rule| SpecError::OutOfRange { method, key, rule };
        let half_life = |h: u64| {
            if h >= 1 {
                Ok(())
            } else {
                Err(out_of_range("h", "at least 1 second"))
            }
        };
        let depth = |d: u32| {
            if d >= 1 {
                Ok(())
            } else {
                Err(out_of_range("depth", "at least 1"))
            }
        };
        let unit = |key, x: f64| {
            if (0.0..=1.0).contains(&x) {
                Ok(())
            } else {
                Err(out_of_range(key, "between 0 and 1"))
            }
        };
        let not_negative = |key, x: f64| {
            if x.is_finite() && x >= 0.0 {
                Ok(())
            } else {
                Err(out_of_range(key, "a number that is not negative"))
            }
        };
        let weights_and_list = |weights: &[f64; 3], list: u32| {
            for w in weights {
                not_negative("w", *w)?;
            }
            if weights.iter().sum::<f64>() <= 0.0 {
                return Err(out_of_range("w", "above zero for at least one ranker"));
            }
            if list == 0 {
                return Err(out_of_range("list", "at least 1"));
            }
            Ok(())
        };
        match self {
            MethodSpec::Dense | MethodSpec::Oracle | MethodSpec::Rrf { .. } => Ok(()),
            MethodSpec::DecayOnly { h } => half_life(*h),
            MethodSpec::Batch { period } if *period < 1 => Err(out_of_range("period", "at least 1 second")),
            MethodSpec::Cutoff { window } if *window < 1 => Err(out_of_range("window", "at least 1 second")),
            MethodSpec::Batch { .. } | MethodSpec::Cutoff { .. } => Ok(()),
            MethodSpec::Cc { alpha } => unit("alpha", *alpha),
            MethodSpec::RrfTd(p) | MethodSpec::RrfTdReorder(p) => {
                half_life(p.h)?;
                weights_and_list(&p.weights, p.list)?;
                unit("eps", p.eps)?;
                not_negative("theta", p.theta)
            }
            MethodSpec::Rrf3 { weights, list, .. } => weights_and_list(weights, *list),
            MethodSpec::RrfAdd { beta, h, .. } => {
                not_negative("beta", *beta)?;
                half_life(*h)
            }
            MethodSpec::TwCsRrf { lambda, .. } => unit("lambda", *lambda),
            MethodSpec::NewestFirst { n } if *n == 0 => Err(out_of_range("n", "at least 1")),
            MethodSpec::NewestFirst { .. } => Ok(()),
            MethodSpec::Product { depth: d, .. }
            | MethodSpec::ProductEnds { depth: d, .. }
            | MethodSpec::KeyedLatest { depth: d, .. } => depth(*d),
            MethodSpec::ProductDecay { h, depth: d, .. } => {
                half_life(*h)?;
                depth(*d)
            }
            MethodSpec::ProductBonus {
                beta, h, depth: d, ..
            } => {
                not_negative("beta", *beta)?;
                half_life(*h)?;
                depth(*d)
            }
            MethodSpec::ProductNewerWins { tau, depth: d, .. } => {
                if !(-1.0..=1.0).contains(tau) {
                    return Err(out_of_range("tau", "between -1 and 1"));
                }
                depth(*d)
            }
        }
    }

    /// Read one text form in which a parameter may list alternatives with `|`,
    /// and return every combination, the first parameter varying slowest:
    /// `rrf_td(k=10|60,h=300|1800)` is four methods.
    pub fn parse_grid(text: &str) -> Result<Vec<MethodSpec>, SpecError> {
        let (kind, pairs) = split_call(text)?;
        let mut combinations: Vec<Vec<(&str, &str)>> = vec![Vec::new()];
        for (key, value) in pairs {
            let mut next = Vec::new();
            for combination in &combinations {
                for alternative in value.split('|') {
                    let mut extended = combination.clone();
                    extended.push((key, alternative.trim()));
                    next.push(extended);
                }
            }
            combinations = next;
        }
        combinations
            .into_iter()
            .map(|pairs| from_parts(kind, pairs))
            .collect()
    }
}

impl fmt::Display for MethodSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind())?;
        let shown = self.shown();
        if shown.is_empty() {
            return Ok(());
        }
        f.write_str("(")?;
        for (i, (key, value)) in shown.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{key}={value}")?;
        }
        f.write_str(")")
    }
}

impl FromStr for MethodSpec {
    type Err = SpecError;

    fn from_str(text: &str) -> Result<Self, SpecError> {
        let (kind, pairs) = split_call(text)?;
        from_parts(kind, pairs)
    }
}

/// A method kind and the `key=value` pairs written after it.
type Call<'a> = (&'a str, Vec<(&'a str, &'a str)>);

/// Split `kind(key=value,key=value)` into its kind and its pairs.
fn split_call(text: &str) -> Result<Call<'_>, SpecError> {
    let syntax = |reason| SpecError::Syntax {
        text: text.to_string(),
        reason,
    };
    let text_trimmed = text.trim();
    let Some((kind, rest)) = text_trimmed.split_once('(') else {
        if text_trimmed.is_empty() {
            return Err(syntax("it is empty"));
        }
        return Ok((text_trimmed, Vec::new()));
    };
    let inner = rest
        .trim_end()
        .strip_suffix(')')
        .ok_or_else(|| syntax("the closing bracket is missing"))?;
    let mut pairs = Vec::new();
    if !inner.trim().is_empty() {
        for part in inner.split(',') {
            let (key, value) = part
                .split_once('=')
                .ok_or_else(|| syntax("a parameter is written key=value"))?;
            pairs.push((key.trim(), value.trim()));
        }
    }
    Ok((kind.trim(), pairs))
}

/// The parameters given for one method, each taken at most once.
struct Given<'a> {
    method: &'a str,
    values: BTreeMap<&'a str, &'a str>,
}

impl<'a> Given<'a> {
    fn new(method: &'a str, pairs: Vec<(&'a str, &'a str)>) -> Result<Self, SpecError> {
        let mut values = BTreeMap::new();
        for (key, value) in pairs {
            if values.insert(key, value).is_some() {
                return Err(SpecError::RepeatedParameter {
                    method: method.to_string(),
                    key: key.to_string(),
                });
            }
        }
        Ok(Self { method, values })
    }

    fn parse<T: FromStr>(&self, key: &str, value: &str) -> Result<T, SpecError> {
        value.parse().map_err(|_| SpecError::BadValue {
            method: self.method.to_string(),
            key: key.to_string(),
            value: value.to_string(),
        })
    }

    fn required<T: FromStr>(&mut self, key: &'static str) -> Result<T, SpecError> {
        match self.values.remove(key) {
            Some(value) => self.parse(key, value),
            None => Err(SpecError::MissingParameter {
                method: self.method.to_string(),
                key,
            }),
        }
    }

    fn optional<T: FromStr>(&mut self, key: &'static str, default: T) -> Result<T, SpecError> {
        match self.values.remove(key) {
            Some(value) => self.parse(key, value),
            None => Ok(default),
        }
    }

    /// `w=a:b:c`, the weights of the dense, keyword and time rankers.
    fn weights(&mut self) -> Result<[f64; 3], SpecError> {
        let Some(value) = self.values.remove("w") else {
            return Ok(equal_thirds());
        };
        let parts: Vec<&str> = value.split(':').collect();
        let [a, b, c] = parts.as_slice() else {
            return Err(SpecError::BadValue {
                method: self.method.to_string(),
                key: "w".to_string(),
                value: value.to_string(),
            });
        };
        Ok([self.parse("w", a)?, self.parse("w", b)?, self.parse("w", c)?])
    }

    fn paper(&mut self) -> Result<RrfTdParams, SpecError> {
        Ok(RrfTdParams {
            k: self.required("k")?,
            h: self.required("h")?,
            weights: self.weights()?,
            list: self.optional("list", PAPER_LIST)?,
            eps: self.optional("eps", PAPER_EPS)?,
            theta: self.optional("theta", PAPER_THETA)?,
        })
    }

    fn depth(&mut self) -> Result<u32, SpecError> {
        self.optional("depth", DEFAULT_DEPTH)
    }

    fn recent(&mut self) -> Result<u32, SpecError> {
        self.optional("recent", 0)
    }

    fn nothing_left(self) -> Result<(), SpecError> {
        match self.values.into_keys().next() {
            Some(key) => Err(SpecError::UnknownParameter {
                method: self.method.to_string(),
                key: key.to_string(),
            }),
            None => Ok(()),
        }
    }
}

fn from_parts(kind: &str, pairs: Vec<(&str, &str)>) -> Result<MethodSpec, SpecError> {
    let mut given = Given::new(kind, pairs)?;
    let spec = match kind {
        "dense" => MethodSpec::Dense,
        "rrf" => MethodSpec::Rrf {
            k: given.optional("k", DEFAULT_K)?,
        },
        "decay_only" => MethodSpec::DecayOnly {
            h: given.required("h")?,
        },
        "batch" => MethodSpec::Batch {
            period: given.required("period")?,
        },
        "cutoff" => MethodSpec::Cutoff {
            window: given.required("window")?,
        },
        "cc" => MethodSpec::Cc {
            alpha: given.required("alpha")?,
        },
        "rrf_td" => MethodSpec::RrfTd(given.paper()?),
        "rrf_td_reorder" => MethodSpec::RrfTdReorder(given.paper()?),
        "rrf3" => MethodSpec::Rrf3 {
            k: given.required("k")?,
            weights: given.weights()?,
            list: given.optional("list", PAPER_LIST)?,
        },
        "rrf_add" => MethodSpec::RrfAdd {
            beta: given.required("beta")?,
            h: given.required("h")?,
            k: given.optional("k", DEFAULT_K)?,
        },
        "tw_cs_rrf" => MethodSpec::TwCsRrf {
            lambda: given.optional("lambda", DEFAULT_LAMBDA)?,
            k: given.optional("k", DEFAULT_K)?,
        },
        "newest_first" => MethodSpec::NewestFirst {
            n: given.required("n")?,
        },
        "product" => MethodSpec::Product {
            depth: given.depth()?,
            recent: given.recent()?,
        },
        "product_decay" => MethodSpec::ProductDecay {
            h: given.required("h")?,
            depth: given.depth()?,
            recent: given.recent()?,
        },
        "product_bonus" => MethodSpec::ProductBonus {
            beta: given.required("beta")?,
            h: given.required("h")?,
            depth: given.depth()?,
            recent: given.recent()?,
        },
        "product_ends" => MethodSpec::ProductEnds {
            depth: given.depth()?,
            recent: given.recent()?,
        },
        "product_newer_wins" => MethodSpec::ProductNewerWins {
            tau: given.required("tau")?,
            depth: given.depth()?,
            recent: given.recent()?,
        },
        "keyed_latest" => MethodSpec::KeyedLatest {
            depth: given.depth()?,
            recent: given.recent()?,
        },
        "oracle" => MethodSpec::Oracle,
        other => return Err(SpecError::UnknownMethod(other.to_string())),
    };
    given.nothing_left()?;
    spec.validate()?;
    Ok(spec)
}

/// Read a list of text forms, each of which may be a grid (`k=10|60`). A
/// method named more than once is kept once, at its first place.
pub fn parse_methods<S: AsRef<str>>(texts: &[S]) -> Result<Vec<MethodSpec>, SpecError> {
    let mut methods: Vec<MethodSpec> = Vec::new();
    for text in texts {
        for spec in MethodSpec::parse_grid(text.as_ref())? {
            if !methods.contains(&spec) {
                methods.push(spec);
            }
        }
    }
    Ok(methods)
}

/// Read methods written one per line. Blank lines and lines that start with `#` are skipped.
pub fn parse_methods_file(text: &str) -> Result<Vec<MethodSpec>, SpecError> {
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect();
    parse_methods(&lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameters_list_every_value_with_defaults_and_order_the_listing() {
        let paper = MethodSpec::RrfTd(RrfTdParams::paper(60, 1800));
        let third = 1.0 / 3.0;
        assert_eq!(
            paper.parameters(),
            [
                ("k", 60.0),
                ("h", 1800.0),
                ("w_dense", third),
                ("w_sparse", third),
                ("w_time", third),
                ("list", 20.0),
                ("eps", 1e-6),
                ("theta", 60.0)
            ]
        );
        assert_eq!(
            MethodSpec::Product { depth: 10, recent: 0 }.parameters(),
            [("depth", 10.0), ("recent", 0.0)]
        );
        assert!(MethodSpec::Oracle.parameters().is_empty());
        // Every example names each of its parameters once.
        for spec in examples() {
            let mut names: Vec<&str> = spec.parameters().iter().map(|(name, _)| *name).collect();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), spec.parameters().len(), "{spec}");
        }
        // By kind first (dense before rrf_td before product), then numerically by parameter:
        // k = 10 before k = 60 before k = 100, although "100" sorts before "60" as text.
        let mut listed = parse_methods(&[
            "product",
            "rrf_td(k=100,h=60)",
            "rrf_td(k=60,h=1800)",
            "rrf_td(k=10,h=86400)",
            "rrf_td(k=60,h=300)",
            "dense",
        ])
        .unwrap();
        listed.sort_by(MethodSpec::cmp_listing);
        let texts: Vec<String> = listed.iter().map(MethodSpec::to_string).collect();
        assert_eq!(
            texts,
            [
                "dense",
                "rrf_td(k=10,h=86400)",
                "rrf_td(k=60,h=300)",
                "rrf_td(k=60,h=1800)",
                "rrf_td(k=100,h=60)",
                "product"
            ]
        );
    }

    #[test]
    fn a_list_of_methods_expands_grids_and_keeps_each_method_once() {
        let methods = parse_methods(&["dense", "rrf(k=10|60)", "rrf", "dense"]).unwrap();
        let texts: Vec<String> = methods.iter().map(MethodSpec::to_string).collect();
        assert_eq!(texts, ["dense", "rrf(k=10)", "rrf(k=60)"]);
        let file = "# the paper's setting\nrrf_td(k=60,h=1800)\n\n  product(depth=32|64)  \n";
        let texts: Vec<String> = parse_methods_file(file)
            .unwrap()
            .iter()
            .map(MethodSpec::to_string)
            .collect();
        assert_eq!(
            texts,
            ["rrf_td(k=60,h=1800)", "product(depth=32)", "product(depth=64)"]
        );
        assert!(parse_methods(&["dense", "nonsense"]).is_err());
    }

    /// One spec of every kind, with and without defaults.
    fn examples() -> Vec<MethodSpec> {
        vec![
            MethodSpec::Dense,
            MethodSpec::Rrf { k: 60 },
            MethodSpec::Rrf { k: 10 },
            MethodSpec::DecayOnly { h: 1800 },
            MethodSpec::Batch { period: 300 },
            MethodSpec::Cutoff { window: 14_400 },
            MethodSpec::Cc { alpha: 0.3 },
            MethodSpec::RrfTd(RrfTdParams::paper(60, 1800)),
            MethodSpec::RrfTd(RrfTdParams {
                k: 10,
                h: 60,
                weights: [0.45, 0.45, 0.1],
                list: 64,
                eps: 1e-9,
                theta: 0.0,
            }),
            MethodSpec::RrfTdReorder(RrfTdParams::paper(100, 86_400)),
            MethodSpec::Rrf3 {
                k: 60,
                weights: equal_thirds(),
                list: PAPER_LIST,
            },
            MethodSpec::RrfAdd {
                beta: 0.1,
                h: 1800,
                k: 60,
            },
            MethodSpec::RrfAdd {
                beta: 2.0,
                h: 60,
                k: 10,
            },
            MethodSpec::TwCsRrf { lambda: 0.9, k: 60 },
            MethodSpec::NewestFirst { n: 10 },
            MethodSpec::Product { depth: 10, recent: 0 },
            MethodSpec::Product { depth: 64, recent: 0 },
            MethodSpec::Product { depth: 10, recent: 8 },
            MethodSpec::ProductDecay {
                h: 7200,
                depth: 32,
                recent: 0,
            },
            MethodSpec::ProductBonus {
                beta: 0.5,
                h: 300,
                depth: 10,
                recent: 0,
            },
            MethodSpec::ProductBonus {
                beta: 0.5,
                h: 1800,
                depth: 10,
                recent: 16,
            },
            MethodSpec::ProductEnds { depth: 10, recent: 0 },
            MethodSpec::ProductNewerWins {
                tau: 0.9,
                depth: 10,
                recent: 32,
            },
            MethodSpec::KeyedLatest { depth: 10, recent: 0 },
            MethodSpec::Oracle,
        ]
    }

    #[test]
    fn recent_is_shown_only_when_above_zero_and_old_files_read_as_zero() {
        let bonus = MethodSpec::ProductBonus {
            beta: 0.5,
            h: 1800,
            depth: 10,
            recent: 16,
        };
        assert_eq!(bonus.to_string(), "product_bonus(beta=0.5,h=1800,recent=16)");
        assert_eq!(bonus.name(), "product_bonus-beta0.5-h1800-recent16");
        assert_eq!(
            "product_bonus(beta=0.5,h=1800,recent=16)".parse::<MethodSpec>(),
            Ok(bonus)
        );
        assert!(bonus.needs_newest());
        let plain = MethodSpec::Product { depth: 10, recent: 0 };
        assert_eq!(
            (plain.to_string(), plain.name()),
            ("product".into(), "product".into())
        );
        assert!(!plain.needs_newest() && !MethodSpec::Dense.needs_newest());
        // A file written before `recent` existed reads back with recent 0, and a method
        // with recent 0 is written exactly as before.
        let old: MethodSpec = serde_json::from_str("{\"product\":{\"depth\":10}}").unwrap();
        assert_eq!(old, plain);
        assert_eq!(
            serde_json::to_string(&plain).unwrap(),
            "{\"product\":{\"depth\":10}}"
        );
        assert_eq!(
            serde_json::to_string(&bonus).unwrap(),
            "{\"product_bonus\":{\"beta\":0.5,\"h\":1800,\"depth\":10,\"recent\":16}}"
        );
        // A grid over recent.
        let grid = MethodSpec::parse_grid("product(recent=0|8)").unwrap();
        assert_eq!(grid.iter().map(MethodSpec::recent).collect::<Vec<_>>(), [0, 8]);
    }

    #[test]
    fn the_text_form_reads_back_to_the_same_method() {
        for spec in examples() {
            let text = spec.to_string();
            assert_eq!(text.parse::<MethodSpec>(), Ok(spec), "{text}");
        }
    }

    #[test]
    fn the_serde_form_reads_back_to_the_same_method() {
        for spec in examples() {
            let json = serde_json::to_string(&spec).unwrap();
            assert_eq!(serde_json::from_str::<MethodSpec>(&json).unwrap(), spec, "{json}");
        }
        assert_eq!(serde_json::to_string(&MethodSpec::Dense).unwrap(), "\"dense\"");
        assert_eq!(
            serde_json::to_string(&MethodSpec::ProductDecay {
                h: 60,
                depth: 10,
                recent: 0
            })
            .unwrap(),
            "{\"product_decay\":{\"h\":60,\"depth\":10}}"
        );
    }

    #[test]
    fn the_paper_setting_has_the_expected_text_and_name() {
        let paper = MethodSpec::RrfTd(RrfTdParams::paper(60, 1800));
        assert_eq!(paper.to_string(), "rrf_td(k=60,h=1800)");
        assert_eq!(paper.name(), "rrf_td-k60-h1800");
        assert_eq!("rrf_td(k=60,h=1800)".parse::<MethodSpec>(), Ok(paper));
        // The defaults written out are the same method.
        assert_eq!(
            " rrf_td( h = 1800, k=60, list=20, eps=1e-6, theta=60 ) ".parse::<MethodSpec>(),
            Ok(paper)
        );
    }

    #[test]
    fn defaults_are_left_out_and_everything_else_is_shown() {
        let text = |s: MethodSpec| s.to_string();
        assert_eq!(text(MethodSpec::Dense), "dense");
        assert_eq!(text(MethodSpec::Rrf { k: 60 }), "rrf(k=60)");
        assert_eq!(text(MethodSpec::Product { depth: 10, recent: 0 }), "product");
        assert_eq!(
            text(MethodSpec::Product { depth: 32, recent: 0 }),
            "product(depth=32)"
        );
        assert_eq!(
            text(MethodSpec::Product { depth: 32, recent: 8 }),
            "product(depth=32,recent=8)"
        );
        assert_eq!(
            text(MethodSpec::TwCsRrf { lambda: 0.9, k: 60 }),
            "tw_cs_rrf(lambda=0.9)"
        );
        assert_eq!(
            text(MethodSpec::RrfAdd {
                beta: 0.1,
                h: 300,
                k: 10
            }),
            "rrf_add(beta=0.1,h=300,k=10)"
        );
        let wide = MethodSpec::RrfTd(RrfTdParams {
            k: 10,
            h: 60,
            weights: [0.45, 0.45, 0.1],
            list: 64,
            eps: 1e-9,
            theta: 0.0,
        });
        assert_eq!(
            wide.to_string(),
            "rrf_td(k=10,h=60,w=0.45:0.45:0.1,list=64,eps=0.000000001,theta=0)"
        );
        assert_eq!(
            wide.name(),
            "rrf_td-k10-h60-w0.45_0.45_0.1-list64-eps0.000000001-theta0"
        );
        assert_eq!("rrf".parse::<MethodSpec>(), Ok(MethodSpec::Rrf { k: 60 }));
        assert_eq!(
            "tw_cs_rrf".parse::<MethodSpec>(),
            Ok(MethodSpec::TwCsRrf { lambda: 0.9, k: 60 })
        );
    }

    #[test]
    fn names_are_distinct_and_safe_as_file_names() {
        let names: Vec<String> = examples().iter().map(MethodSpec::name).collect();
        for name in &names {
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.')),
                "{name}"
            );
        }
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), names.len());
    }

    #[test]
    fn every_kind_is_listed_once_and_has_an_example() {
        let mut kinds: Vec<&str> = examples().iter().map(MethodSpec::kind).collect();
        kinds.dedup();
        assert_eq!(kinds, KINDS);
    }

    #[test]
    fn a_grid_expands_to_every_combination_in_order() {
        let grid = MethodSpec::parse_grid("rrf_td(k=10|60,h=300|1800)").unwrap();
        let texts: Vec<String> = grid.iter().map(MethodSpec::to_string).collect();
        assert_eq!(
            texts,
            [
                "rrf_td(k=10,h=300)",
                "rrf_td(k=10,h=1800)",
                "rrf_td(k=60,h=300)",
                "rrf_td(k=60,h=1800)"
            ]
        );
        assert_eq!(MethodSpec::parse_grid("dense").unwrap(), [MethodSpec::Dense]);
        let paper = format!(
            "rrf_td(k={},h={})",
            PAPER_K_GRID.map(|k| k.to_string()).join("|"),
            PAPER_H_GRID.map(|h| h.to_string()).join("|")
        );
        assert_eq!(MethodSpec::parse_grid(&paper).unwrap().len(), 24);
    }

    #[test]
    fn mistakes_are_named() {
        let parse = |t: &str| t.parse::<MethodSpec>();
        assert_eq!(
            parse("rrf_tf(k=60)"),
            Err(SpecError::UnknownMethod("rrf_tf".into()))
        );
        assert!(matches!(
            parse("rrf_td(k=60)"),
            Err(SpecError::MissingParameter { key: "h", .. })
        ));
        assert!(matches!(
            parse("rrf(k=60,k=10)"),
            Err(SpecError::RepeatedParameter { .. })
        ));
        assert!(matches!(
            parse("dense(k=60)"),
            Err(SpecError::UnknownParameter { .. })
        ));
        assert!(matches!(parse("rrf(k=sixty)"), Err(SpecError::BadValue { .. })));
        assert!(matches!(parse("rrf(k=60"), Err(SpecError::Syntax { .. })));
        assert!(matches!(parse("rrf(k)"), Err(SpecError::Syntax { .. })));
        assert!(matches!(parse(""), Err(SpecError::Syntax { .. })));
        assert!(matches!(
            parse("rrf_td(k=60,h=1800,w=1:2)"),
            Err(SpecError::BadValue { .. })
        ));
    }

    #[test]
    fn values_outside_their_range_are_refused() {
        let parse = |t: &str| t.parse::<MethodSpec>();
        for text in [
            "decay_only(h=0)",
            "batch(period=0)",
            "cutoff(window=-5)",
            "cc(alpha=1.5)",
            "cc(alpha=NaN)",
            "rrf_td(k=60,h=1800,w=0:0:0)",
            "rrf_td(k=60,h=1800,w=1:-1:1)",
            "rrf_td(k=60,h=1800,list=0)",
            "rrf_td(k=60,h=1800,eps=2)",
            "rrf_td(k=60,h=1800,theta=-1)",
            "rrf_add(beta=-0.1,h=60)",
            "tw_cs_rrf(lambda=1.1)",
            "newest_first(n=0)",
            "product(depth=0)",
            "product_newer_wins(tau=1.5)",
        ] {
            assert!(matches!(parse(text), Err(SpecError::OutOfRange { .. })), "{text}");
        }
        // The supplement's own example uses one ranker with unit weight.
        assert!(parse("rrf_td(k=60,h=1800,w=1:0:0)").is_ok());
    }
}
