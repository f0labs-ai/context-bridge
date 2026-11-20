# ContextBridge proof — design and contracts

This repository holds everything needed to check the claims of the ContextBridge paper
on a real voice-agent stack: the data tools, the service drivers, the evaluator, the
paper source, and (outside git) the run files.

The services under test (knowledge search, speech-to-text, masking) are a private
product. They are driven over HTTP by `cb-run`, and everything they return is written to
files in the shapes of `cb-types`. **All ranking results are computed offline by
`cb-eval` from those files**, so anyone holding the files can reproduce every number
without the services.

## 1. Crates

| Crate | What it is | Talks to a service? |
|---|---|---|
| `cb-types` | The file contracts (serde types, JSON Lines helpers). | no |
| `cb-data` | Builds data sets: the generated incident corpus, the public incident timelines. | only to download public data |
| `cb-run` | Uploads records, collects candidate lists, runs extraction, times the pipeline. | yes |
| `cb-eval` | Ranking rules, the two-conversation rule, metrics, statistics, tables. | no |

Rust edition 2024, toolchain 1.99.0, `cargo clippy --all-targets -- -D warnings` clean,
no `unwrap`/`expect`/`panic` outside tests (workspace lints), `thiserror` in libraries
and `anyhow` in binaries. No Python anywhere.

## 2. Files of a data set

A data set is a directory under `data/<name>/`:

| File | Type per line | Notes |
|---|---|---|
| `records.jsonl` | `cb_types::Record` | every piece of knowledge, changing and standing |
| `queries.jsonl` | `cb_types::Query` | every question with its moment and graded answers |
| `transcripts.jsonl` | `cb_types::Transcript` | generated corpus only |
| `manifest.json` | `cb_types::Manifest` | counts, seed, sha256 of each file, sources and terms |

Rules:

- Time is Unix seconds, UTC.
- A record id satisfies `cb_types::ids::check`. A record is uploaded to the knowledge
  service as its own document at path `cb/<id>.md` (`ids::source_path`).
- `Record.truth` and `Query.gold` are for scoring only. No method under test reads them.
- `Record.thread` is the source's own grouping (an incident id). Only methods marked
  **keyed** may read it.
- Graded relevance: 3 = the answer (for a changing question, the record that was current
  when it was asked); 1 = about the right matter but replaced since; anything else 0.
  `Query.harmful` lists planted wrong statements on the same matter.
- Splits are by thread, never by question: `split(thread) = validation` when
  `fnv1a64(thread) % 5 == 0`, else `test`. Questions without a thread hash their own id.
  Parameters are chosen on validation; only test is reported.

## 3. Files of a run

A run is a directory under `runs/<name>/`:

| File | Type per line | Written by |
|---|---|---|
| `candidates.jsonl` | `cb_types::Candidates` (one per question and view) | `cb-run candidates` |
| `newest.jsonl` | `cb_types::Newest` (one per question: the `now` view's newest records, scored) | `cb-run newest` |
| `vectors.f32` + `vectors.json` | passage vectors, row order in the JSON | `cb-run vectors` |
| `extractions.jsonl` | `cb_types::Extraction` | `cb-run extract` |
| `trips.jsonl` | `cb_types::PipelineTrip` | `cb-run trips` |
| `rankings/<method>.jsonl` | `cb_types::Ranking` | `cb-eval rank` |
| `summary.json`, `tables/*.tex` | metrics with intervals | `cb-eval report` |
| `manifest.json` | what ran: git commits, service build, model names, machine, seeds | every tool appends |

### 3.1 Views

A candidate list is taken "as of" the moment the question was asked: the service may
only see records with `written_at <= as_of`. Baselines that model a slower or narrower
index need other views of the same question, so one question yields several
`Candidates` lines, one per view, each with the view's name in `Candidates.view`:

| View | Records the index may see |
|---|---|
| `now` | `written_at <= asked_at` |
| `batch_300`, `batch_1800`, `batch_3600` | `written_at <= floor(asked_at / Δ) * Δ` for Δ = 300, 1800, 3600 s |
| `window_3600`, `window_14400`, `window_86400` | `asked_at - W < written_at <= asked_at` |

Only the `now` view carries relevance-model scores (`rerank`), for the union of the
first 64 records of each of its two lists.

## 4. Methods (`cb-eval`)

For a question asked at `t`, the `now` view gives two lists: `D` (by meaning, best
first, with cosine scores) and `S` (by keywords, best first, with BM25 scores), and a
relevance score `rho(r)` in [0, 1] for records in either list. `age(r) = t - written_at(r)`
in seconds. `decay_h(r) = max(2^(-age(r)/h), eps)`. `rank_L(r)` is 1-based.

`rrf_k(lists, weights)(r) = sum over lists L containing r of w_L / (k + rank_L(r))`.

Ties are broken by newer `written_at` first, then by record id, so every method is
deterministic.

### 4.1 The paper's baselines

| Name | Definition |
|---|---|
| `dense` | `D` as given. (The paper's "static RAG".) |
| `rrf(k)` | `rrf_k([D, S], [0.5, 0.5])`, default k = 60. |
| `decay_only(h)` | records of `D`, score `cos * decay_h`. |
| `batch(Δ)` | `dense` on view `batch_Δ`. |
| `cutoff(W)` | `dense` on view `window_W`; when that list has fewer than 5 records, `dense` on `now`. |
| `cc(alpha)` | `alpha * mm(cos) + (1 - alpha) * mm(bm25)`, each score min–max scaled within its own list, a missing score counting 0. `alpha` chosen on validation. |

Self-RAG is not run: it needs its own fine-tuned language model and a Python code base.
The paper says so.

### 4.2 The paper's formula

`rrf_td(k, h, weights, list, eps, theta)` — Algorithm 1 of the paper, as printed:

1. `T` = the `list` newest records visible at `t` (from the whole data set), newest first.
2. `C = D[..list] ∪ S[..list] ∪ T`.
3. `s(r) = rrf_k([D[..list], S[..list], T], weights)(r)`.
4. `sigma` = standard deviation of `written_at` over `C`. If `sigma > theta` then
   `score(r) = s(r) * decay_h(r)`, else `score(r) = s(r)`.
5. Order by `score` descending, then `written_at` descending, then `s` descending.

Defaults are the paper's: `list = 20`, `eps = 1e-6`, `theta = 60`, weights one third
each. The grid is k in {10, 30, 60, 100} and h in {60, 300, 1800, 7200, 43200, 86400}
seconds. Variants: `rrf_td_reorder` (T is `D ∪ S` ordered by recency, nothing is added
to the candidates) and `rrf3` (step 4 skipped: three rankers, no decay).

### 4.3 Published alternatives

| Name | Definition | Source |
|---|---|---|
| `rrf_add(beta, h)` | `rrf_k([D, S]) + beta * (1/(k+1)) * 2^(-age/h)` | the additive pattern in Qdrant's documentation |
| `tw_cs_rrf(lambda)` | per list, `(lambda * mm(score) + (1 - lambda) * recency) / (k + rank)`, recency scaled linearly to [0, 1] over the candidates; lambda = 0.9 | Rollings, Rankel and Oard, TREC 2022 |
| `newest_first(n)` | the first `n` records of `D`, ordered by recency | Li and Croft, 2003 |

### 4.4 The product's path and the corrected rules

The product's candidate set is the first `depth` records (default 10, as deployed) of
`rrf_2([D[..64], S[..64]])` with rank counted from 0 (Qdrant's built-in merge: a record's
score is the sum of `1 / (position + 2)` over the lists that hold it), scored by the
relevance model. Every `product*` method takes `depth`; 32 and 64 are reported as
variants. `D[..64]` and `S[..64]` are the records the service listed in its first 64
places, in the list's own order (equal scores newer first). A record there that the
relevance pass never scored is not a candidate: equal scores across place 64, or a
record of several passages whose first passage came after the 64th. `cb-eval rank`
reports how many there were.

| Name | Definition |
|---|---|
| `product` | product candidates with `rho >= 0.12`, ordered by `rho`. What the service returns today. |
| `product_decay(h)` | as `product`, score `rho * decay_h`. The paper's idea on a calibrated score. |
| `product_bonus(beta, h)` | as `product`, score `rho * (1 + beta * 2^(-age/h))`. The cut-off applies to `rho`. |
| `product_ends` | as `product`, after dropping records with `valid_until <= t`. |
| `product_newer_wins(tau)` | as `product`; then, walking down the list, a record is dropped when a kept record says the same thing (`cos(passage, passage) >= tau`) and is newer; when the newer one comes later in the list it takes the older one's place. |
| `keyed_latest` (**keyed**) | only the newest visible record of each thread is a candidate, then `product`. Models "one document per incident, edited in place". |
| `oracle` | gold order. The ceiling. |

`tau`, `beta` and `h` are chosen on validation.

**The recency list.** `N` is the list of the newest
records the `now` view could show, newest first (records of one second by id): the
records of the service's 32 newest passages, each scored by the relevance model
(`newest.jsonl`). Every `product*` method takes `recent` (default 0, chosen on validation
from 0, 8, 16, 32): the first `recent` records of `N` join the `depth` merged records,
each record once, before the cut-off and the ordering. A record's `rho` is then the
higher of its two passes' scores, because the two passes may have scored different
passages of one record. A method with `recent` 0 never reads `N` and never sees its
scores, so `product` is the same whether or not a run has the file.

## 5. Measures

Per question, then averaged within a class:

- `ndcg@5`, `ndcg@10` with gain `2^grade - 1`.
- `mrr`: reciprocal rank of the first grade-3 record within the first 20.
- `recall@20`: share of grade-3 records within the first 20.
- `current@1`: 1 when the first record has grade 3.
- `hit@4`: 1 when a grade-3 record is within the first 4 (the agent receives 4 passages).
- `stale@4`: share of the first 4 that are grade-1 records.
- `harmful@4`: share of the first 4 that are in `Query.harmful`.
- `changing@4`: share of the first 4 that are `Tier::Changing` records (reported for
  ordinary questions: how much fresh material intrudes).
- `temporal_precision@10`: share of the first 10 that were valid when the question was
  asked. A changing record is valid from `written_at` until `truth.superseded_at`;
  a standing record is always valid.

Statistics: a 95% interval from 10,000 bootstrap resamples **of threads** (questions
about one incident are not independent), and a two-sided paired randomization test with
10,000 permutations against the named baseline. Seeds are fixed and recorded.

## 6. The two-conversation rule (`cb-eval gate`)

Input: `extractions.jsonl`, the transcripts they came from, one vector per extraction
(the service's embedding of `problem + " " + solution`, `extraction_vectors.f32` and
`.json`), and the planted truth. An extraction that found nothing (no problem and no
solution) writes no record: it is counted, never grouped or judged, and needs no vector.

An extraction is **correct** when it states a true record of its transcript and no
planted wrong statement: every key term of one true state appears (case-folded,
whitespace-normalised) and no key term set of a wrong statement appears in full. An
independent language-model judge re-scores a random sample of 500, and the agreement
between the two is reported.

Variants, all reported as precision against delay:

- how many conversations must agree: 1, 2, 3;
- what "agree" means: every pair above a bar (the paper: 0.92), or the newcomer's
  average likeness to the group above a bar;
- what "independent" means: different calls, or different agents;
- with and without the paper's confidence cut at 0.85.

Outputs: precision of validated records, share of true states that were ever validated,
the wait from a state's first statement to its validation (median, P90), false merges
(validated groups mixing two states).

Beside them, on the test split:

- every variant's precision minus that of single extractions (`n1-nocut`), with a 95%
  interval from 10,000 bootstrap resamples of the incidents, all variants on the same
  resamples; a group counts under the incident of its first member;
- the inputs of the first paper's error model at the headline variant's bar: over every
  pair of extractions of one incident from two different calls, the share that clears
  the bar when both are correct with one state, both wrong, or one of each, and the
  precision the model predicts from them.

## 7. Timing (`cb-run trips`)

One `PipelineTrip` per record: every mark is taken per record on one machine, and
percentiles are taken over whole trips. Stage summaries are medians of per-record stage
times. Sums of stage percentiles are never reported.

The upload's return and the first hit both follow the hand-over, in either order: the
record becomes searchable inside the upload call, and the search may return it before
the call has returned. A record the search never returns before the run stops waiting
(`never_found` in the run's `failures.jsonl`) counts on the hand-over → first hit path
as slower than every timed trip; a percentile that falls among those records is
reported as beyond the wait.

## 8. What stays outside this repository

- The experiment builds of the private services: the knowledge service, the voice
  server's speech-to-text and the product's masker.
- The data sets, the run files, the results and the paper. `data/` and `runs/` are
  git-ignored: the tools write them, and none is published here.
