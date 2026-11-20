# The knowledge service's experiment build

The knowledge service is part of a private product. For this work it runs an experiment
build that is never released and adds four things, each by extending the part of the
service that already owns that concern. This file is the contract between that build
and `cb-run`; the service's code is not part of this repository.

## 1. A document can say when it was written and when it ends

Two optional attributes of an upload, stamped on every passage, Unix seconds UTC:

```json
{ "written_at": 1780272000, "valid_until": 1780279200 }
```

- `written_at` must be positive and no later than one day after the server's clock.
- `valid_until` needs `written_at` and must be later than it.
- Both are stamped on every passage of the document (payload keys `written_at`,
  `valid_until`, integer-indexed) and echoed in the upload's `attributes`.

## 2. A search is asked "as of" a moment

A search request takes `as_of` (optional, Unix seconds; default: the server's clock),
applied by the same filter as every other condition of a search. With it:

- a passage with `written_at > as_of` is not returned (it was not known yet);
- a passage with `valid_until <= as_of` is not returned (it has ended);
- a passage with neither attribute behaves exactly as today.

Each hit carries `written_at` and `valid_until` when its passage has them.

## 3. The candidate lists behind a search (experiment only)

The route exists only in the experiment build and is guarded by the same internal token
as every `/v1` route.

`POST /v1/tenants/{tenant}/eval/candidates`

```json
{
  "query": "Is home internet down in Northgate?",
  "as_of": 1780275000,
  "n": 100,
  "exact": true,
  "collections": ["cb"],
  "views": [
    { "name": "now" },
    { "name": "batch_300",  "index_as_of": 1780274700 },
    { "name": "window_3600", "written_from": 1780271400 }
  ],
  "rerank": { "view": "now", "per_list": 32 }
}
```

- `as_of` is required. A view sees passages with
  `written_from < written_at <= min(as_of, index_as_of)`; a passage without
  `written_at` is seen by every view that has no `written_from`.
- Expiry (`valid_until`) is **not** applied here: it is a rule under test, applied by
  the evaluator. `valid_until` is returned for every passage.
- `exact: true` searches the dense vectors exhaustively (no quantised shortlist).
- At most 8 views, `n` at most 200, `per_list` at most 64.
- Optional `"newest": k` (1 to 200): each view also lists its `k` newest passages,
  newest first, each scored with its `written_at`; a passage without `written_at` is
  never in it, and passages of one second are settled by point id. Optional
  `"rerank": {…, "newest": m}` (0 to 64) scores the first `m` of the named view's
  newest list too; `per_list` may then be 0. This serves the recency list.

```json
{
  "views": {
    "now":        { "dense": [{"id": "…", "score": 0.83}], "sparse": [{"id": "…", "score": 11.2}] },
    "batch_300":  { "dense": [], "sparse": [] }
  },
  "rerank": [{"id": "…", "score": 0.41}],
  "passages": {
    "<point id>": { "source_path": "cb/inc-0001.s2.md", "kind": "prose",
                    "written_at": 1780273000, "valid_until": null, "chars": 212 }
  },
  "timings_ms": { "embed": 7.1, "search": 9.4, "rerank": 31.0 }
}
```

- The query is embedded once. Each view runs one dense search and one keyword search
  with the service's own query encoding. Lists are best first.
- `rerank` holds the relevance model's raw score for the union of the first
  `per_list` passages of the named view's two lists (and of the first `newest` of its
  newest list, when asked), scored against the passage text exactly as the live
  search scores it.
- A view asked for its newest list carries `"newest": [{"id": "…", "score": 1780273000}]`
  beside `dense` and `sparse`.
- `passages` describes every point id that appears anywhere in the answer.
- A failure of the embedder, the index or the relevance model is a 503 naming the
  stage. Nothing is silently degraded: a list from a degraded search would be
  evaluated as if it were real.

## 4. An upload reports where its time went

Each entry of an upload's `sources[]` gains

```json
"timings_ms": { "parse": 0.2, "mask": 0.1, "models": 0.0, "embed": 8.0,
                "write": 12.4, "searchable": 21.3, "after": 4.0, "total": 25.3 }
```

- `models` is the time spent in language-model calls before the write (the rewrite, the
  status check, how-to steps, spoken answers); zero when the service runs without one.
- `searchable` is the time from the start of the ingest to the return of the index
  write that makes the passages searchable. `after` is everything after that (archive,
  quality check). `total` is the whole call.

## 5. What does not change

The live search path and its response are untouched except for `as_of` and the two
echoed fields. Whichever time rule the evaluation selects is added to the live path in
a second step, as one setting beside the two adjustments that already follow the
relevance model (a passage tied to the caller's screen is raised; passages flagged as
contradicting each other are lowered).
