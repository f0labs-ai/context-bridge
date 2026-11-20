# cb-eval — decisions taken where DESIGN.md leaves room

Each entry says what DESIGN.md (or the brief) says, what cb-eval does, and why.
Where a choice follows the first paper's printed algorithm or the product's deployed
behaviour, that is named. Every rule here is pinned by a unit test with the arithmetic
written out.

## 1. Inputs and the order of equal scores

1. **Ties in service lists.** DESIGN §4: "Ties are broken by newer `written_at` first,
   then by record id". cb-eval applies this once, where a run's lists are read
   (`QuestionContext::new`): records the service scored *exactly* alike are put newer
   first, then by id; records with different scores keep the service's order. So
   `dense` ("D as given"), `batch`, `cutoff` and the ranks every fusion reads never depend
   on how the service happened to return a tie. *Changed my mind:* the first version kept
   the service's order for ties in `dense`; the brief says "everywhere".
2. **Every ordering ends in the record id**, including the paper's step 5 (score,
   `written_at`, pre-decay score), so every answer is a total order.
3. **A run file is checked before any method sees it.** Refused, with the item named: a
   record not in the data set, a record listed twice, a record the view could not have
   seen (`written_at` against the view of DESIGN §3.1), a score that is not a number, a
   list whose scores rise, an `as_of` other than the question's `asked_at`, a relevance
   score outside 0–1, a question without candidate lists.
4. **A candidate without a relevance score is an error, never 0.** The product methods
   need `rho` for the first `depth` merged records. Such a record lies within the first
   2·`depth` positions of one of the lists (a record at position m in both lists scores
   at most 2/(m+2), which the `depth` leaders of one list beat once m > 2·`depth`), and the
   merge reads only the first 64 of each list. Scoring the first 64 of each list, as
   DESIGN §3.1 says, therefore always suffices; KB_EXPERIMENT.md's example request
   (`per_list: 32`) suffices only up to `depth = 15`, and `product(depth=32|64)` would stop
   with this error.
5. **Rankings files hold the first 20 records** of each answer by default (`--keep`, 0 =
   all); the measures are always computed on the whole answer.

## 2. Methods (DESIGN §4)

6. **`decay_h` has the floor `eps = 1e-6` everywhere it is used** (`decay_only`,
   `rrf_td`, `product_decay`), as DESIGN §4 defines `decay_h`. `rrf_add` and
   `product_bonus` use `2^(-age/h)` without a floor, as written.
7. **`rrf_td`, cold start.** `sigma` is the population standard deviation (divide by
   n), as numpy's default and the first check (`rrf_td_first_check.mjs`) compute it;
   decay applies only when `sigma > theta` (strict, as Algorithm 1 prints it). The time
   ranker's records belong to `C` even when its weight is 0.
8. **`rrf_td_reorder`.** "T is D ∪ S ordered by recency": T is `D[..list] ∪ S[..list]`,
   newest first (ties by id), not cut to `list` — as in the first check's
   "time only reorders" variant.
9. **`rrf3`** keeps Algorithm 1's time ranker from the whole data set and skips step 4;
   its half-life, floor and threshold are never read.
10. **Weights** default to one third each; `w=a:b:c` sets them (the supplement's own
    example, one ranker with unit weight, is `w=1:0:0`).
11. **`cc(alpha)`.** Min–max scaling of a list whose scores are all equal (a list of one
    included) gives 1, so being listed never counts the same as being absent.
12. **`rrf_add(beta, h)`.** DESIGN writes `rrf_k([D, S])` without weights. cb-eval uses
    unit weights (Qdrant's documented RRF), so `1/(k+1)` is what first place in one list
    is worth and `beta` counts first places. Its order at `beta = 0` equals `rrf(k)`. `k`
    defaults to 60.
13. **`tw_cs_rrf(lambda)`.** Lists D and S whole; recency is linear over the records of
    either list (newest 1, oldest 0, all 1 when they share a second); `k = 60`.
14. **`newest_first(n)`** reports `written_at` as its score.
15. **The product merge is computed in `f32`**, as Qdrant computes it
    (`1 / (position + 2)`, positions from 0). Equal merged scores go newer first, then by
    id; Qdrant's own order for such ties is not recoverable from the files. The cut-off
    keeps `rho >= 0.12`.
16. **`product_ends` and `keyed_latest` remove records before the merge**, from
    `D[..64]` and `S[..64]`, the way the deployed design applies expiry as a query filter
    (KB_EXPERIMENT §2). Removed records are not replaced by deeper ones (the run has no
    `rho` beyond 64). A stated end at the moment asked (`valid_until == t`) is ended.
17. **`product_newer_wins(tau)`.** `cos >= tau` (inclusive, as DESIGN writes it). Two
    records written in the same second are both kept. A newcomer that says the same as
    several kept older records takes the first one's place and the others are dropped.
    Each record keeps its own `rho` as its score, so that answer's scores need not fall.
18. **`keyed_latest`** keeps a record without a thread; only it reads `thread`, through
    one accessor named `keyed_…`. A test relabels threads and checks that only
    `keyed_latest` changes; another changes the answers and checks only `oracle` does.
19. **`oracle`** lists the graded records with grade > 0, highest first.

## 3. Measures (DESIGN §5)

20. **An empty place is a miss.** Shares of "the first n" divide by n even when a
    method returns fewer records (e.g. `product` after its cut-off). `filled@4` and
    `filled@10` report how many places were filled so this is visible.
21. **NDCG's ideal** is the question's own grades from highest to lowest.
22. **`temporal_precision@10`** follows DESIGN literally: standing records are always
    valid; changing records from `written_at` until `truth.superseded_at` (exclusive), or
    for good when nothing replaced them. A planted wrong statement is never superseded
    and so counts as valid; `harmful@4` measures it. `valid_until` is not used here.
23. **A question is refused** when an answer or harmful record is not in the data set, a
    record is graded twice, a grade is above 3, a harmful record also has a grade above
    0, or it has no grade-3 answer.
24. **Groups.** Results are per `QueryClass` and split, plus `ordinary_all`
    (`ordinary` + `ordinary_near`), because claim 2's guard is on "ordinary and
    near-ordinary questions".

## 4. Statistics

25. **The statistic** is the mean over the questions of each run, then the plain mean
    over runs (the five seeds of corpus (a) are five runs, and their mean counts). Each run is resampled on its own.
26. **Bootstrap.** 10,000 resamples of clusters (a thread, or a question without one),
    drawn with replacement, as many as the run has; the interval is the 2.5th and 97.5th
    percentile by nearest rank (the 250th and 9,750th of 10,000). The same resamples serve
    every method of a group, so a difference's interval is paired.
27. **Randomization test.** Each cluster's two answers are swapped with probability ½;
    two-sided; `p = (1 + b) / (1 + N)` with `b` the permutations at least as far from 0
    (equal within a relative 1e-12 counts). Computed from each method's signed sums (the
    statistic is linear), checked against a direct per-cluster computation by a test.
28. **Percentiles everywhere are nearest rank** with the rank `ceil(q·n)` computed in
    whole numbers; a median is the middle value or the mean of the two middles.
29. **Random streams.** ChaCha20 keyed by the seed (8 bytes LE), FNV-1a 64 of a label
    (8 bytes LE), zeros. Labels: `bootstrap/<group>/<run>`, `randomization/<group>/<run>`,
    `gate/judge-sample`. A run's draws do not change when other runs join an analysis.
    *Changed my mind:* streams were first keyed by run position. Default seed 20261119;
    every output records seed and draws; a test pins the generator against the published
    ChaCha20 keystream and pins the first draws.
30. **Byte-for-byte outputs.** Runs are sorted by name; work is spread over threads in
    fixed chunks; a test checks that a report is the same with 1 or 3 threads and with
    the runs given in another order.

## 5. Parameter choice and the report

31. **`tune`** takes the mean of a named measure over a named group of validation
    questions (several runs: the mean of the runs' means) and keeps the best candidate in
    `tuned.json` under a label. Ties go to the candidate named first. A measure counting
    unwanted records (`stale@4`, `harmful@4`, `changing@4`) is minimised unless told
    otherwise. Optional guard: a second measure on a second group that may fall at most a
    margin below a baseline (claim 2: `hit@4`, ordinary questions, `product`, 1 point);
    only candidates within it can win, and when none is, the one that misses by least.
32. **`report`** reads only the test split. A candidate of a choice that was not chosen is
    left out of the main tables (marked † when chosen) but is in `summary.json`, the
    CSVs, the family tables and `numbers.tex`. Methods ranked from different input files
    are refused rather than compared. A group that only some runs have is left out with a
    note.
33. **Number keys** are `<name>/<method>/<group>/<measure>` with `/lo`, `/hi`, and
    `/minus/<baseline>` with `/lo`, `/hi`, `/p`; shares print in percent with one decimal
    (differences in points), ndcg and mrr with three decimals, p with four. Keys use
    letters, digits and `. _ @ / + -`; a babel language that makes one of these active
    would need `\shorthandoff`. `\cbnum{key}` stops the LaTeX run on an unknown key.

## 6. The two-conversation rule (DESIGN §6)

34. **"Appears".** Key terms and the text are case-folded (Unicode lowercase) and their
    whitespace collapsed; a term appears when it occurs as whole words (no letter or digit
    just before or after). The text is `problem + " " + solution`. cb-data's `check`
    ("every stated record's key terms appear in its transcript") should use the very same
    function; see §9.
35. **Which truth.** The true states are the records in `Transcript.states` with
    `truth.correct`; the wrong statements are every planted wrong record of the call's
    incident that has key terms. A wrong statement without key terms can never be
    recognised; a true record without key terms is stated by any extraction of a call
    that states it (vacuously). Both are counted in `gate/summary.json`. A text that
    holds a wrong statement in full is wrong even beside a true one; of several true
    records the newest is named.
36. **One extraction per call**, refused otherwise; extraction vectors are
    `extraction_vectors.f32` + `.json` keyed by the transcript id.
37. **Replay order.** An extraction becomes available at the end of its call plus the
    model's own time (`latency_ms`); ties by transcript id. Each split (by incident, as
    for questions) is replayed on its own. Pending extractions never expire.
38. **Agreement.** "Every pair": the paper's Algorithm 2 generalised to n — the newcomer
    looks for n − 1 pending extractions that, with it, are pair by pair more alike than
    the bar and independent; the most alike first, earlier first among equals; all are
    promoted and leave the pool. "Average": pending extractions form groups; a newcomer
    joins the group to which its average likeness is highest when above the bar, and a
    group is promoted when it holds n independent members. The bar is strict (the paper:
    "cosine similarity > 0.92"); the confidence cut keeps `c >= 0.85` (the paper: below
    0.85 is discarded). With one extraction per call, "different calls" never refuses a
    pair; "different agents" does.
39. **What a promoted record says** is the paper's merge (problem of the member most
    confident of its problem, solution of the member most confident of its solution). It
    is **correct** when that text states a true record of a member's call and no wrong
    statement of the members' incidents, and the members do not state two different
    records. A group whose members state two different records is a **false merge**.
40. **Validated states and wait.** A true state is validated when a correct promoted
    record states it. The denominator is the true states stated in calls that have an
    extraction (calls without one are counted). A state's first statement is the end of
    the agent turn by which all its key terms have been said in that call (agent words
    only, accumulated); when they never all appear (altered words) the end of the call is
    used, and this is counted. Wait = promotion − first statement; median and P90.
41. **The bar chosen on validation.** For every combination of conversations, agreement,
    independence and cut: try 0.80, 0.81 … 0.98 on the validation split; take the most
    precise bar that validates at least 70% of true states (claim 3's pass line); if none
    does, the bar that validates the most; ties to the higher share, then the lower bar.
    The paper's 0.92 is always reported beside it.
42. **Likeness** is the cosine of the stored `f32` vectors computed in `f64` and kept as
    `f32` once per pair; at most 20,000 extractions per split (about 800 MB).
43. **The judge.** cb-eval draws 500 extractions (seeded, uniform, without replacement,
    from both splits) into `gate/judge_sample.jsonl` without the rule's verdicts, and reads
    the judge's verdicts (`{"transcript", "correct"}` per line) with `--judge` to report
    agreement and Cohen's kappa. It does not call a language model.

## 7. Trips (DESIGN §7)

44. **Paths over whole trips:** hand-off (the `extracted_ms` mark) to first hit — claim
    1's measure — and to index write; utterance (0) to first hit and to index write, only
    for trips that carry a `transcript_ms` mark (they began with speech). A record handed
    straight to the knowledge service should carry `extracted_ms = 0`.
45. **A percentile is reported only when at least one trip lies above it**
    (n ≥ 1/(1 − q): P90 needs 10 trips, P95 20, P99 100).
46. **Stages** are medians of per-record differences of consecutive marks; nothing adds
    stage figures. The pass line is met when at least 1,000 trips have the hand-off path
    and its P95 is at or under 2,000 ms.
47. **A trip is refused** when its record has another trip, a mark is negative or not a
    number, or marks run backwards.

## 8. Not built here

- Claim 1's "time including the wait for a second conversation at three call rates":
  it needs a model of call arrivals that no contract defines.
- The run's `manifest.json` ("every tool appends") has no defined shape; cb-eval writes
  the sha256 of its inputs, its version, seeds and draws into each output instead and
  leaves the manifest alone.
- The language-model judge itself (§6, item 43).

## 9. Suggested changes outside this crate (not made)

- `cb-types`: expose the split rule for a bare key (`split_of_key(&str)`); cb-eval now
  builds a `Query` to reuse `split_of` for an incident's transcripts.
- `cb-types`: own the key-term rule (item 34) so `cb-data check` and `cb-eval gate` cannot
  disagree, plus the judge-verdict line and the extraction-vector file names.
- KB_EXPERIMENT.md: the example request's `per_list: 32` should be 64 (item 4).
- DESIGN §4.3: write `rrf_add`'s fusion weights (item 12).
- DESIGN §5: say whether a planted wrong statement counts as temporally valid (item 22).
- `Cargo.lock` gains `rand_chacha` and `sha2` under `cb-eval`; it is outside this crate's
  path and was not committed.
