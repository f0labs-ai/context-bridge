# `cb-run` — the service drivers

`cb-run` is the only tool that talks to a service. It uploads records, collects
candidate lists, runs extraction and times uploads, and writes everything it gets back
in the shapes of `cb-types` (DESIGN.md section 3). The evaluator never needs a service.

Every command:

- reads its addresses and tokens from flags or the environment (`CB_KB_URL`,
  `CB_KB_TOKEN`, `CB_QDRANT_URL`, `CB_MODEL_URL`, `CB_MODEL_KEY`); nothing is hard-coded
  and no token is ever printed or written to a file;
- gives every request a connect timeout and a total timeout, retries 429, 5xx and
  timeouts with exponential back-off and jitter (at most three tries) and never retries
  a 4xx;
- is resumable: it appends to its output file and skips what is already there;
- writes its failures to `<run>/failures.jsonl` and exits non-zero if any remain, so a
  short run can never pass for a complete one;
- appends to `<run>/manifest.json` what it ran: its own version and git commit, the
  service's `/v1/info`, the model name, the flags, start and end time, counts;
- logs one structured line per batch (count, elapsed, failures), never a record's text.

## upload

```
cb-run upload --data data/synth-42 --tenant <uuid> --collection cb --run runs/synth-42
```

Each record becomes one document: a multipart upload to
`POST /v1/tenants/{tenant}/ingest` with the file named `cb/<record id>.md` (the record's
text), the collection, and attributes `written_at`, `valid_until` (when set), `title`
(when set) and `document_ref` (the record id). Field names are the service's own
(read them from the service's source, not from this page).

Writes `<run>/uploads.jsonl`: one line per record with the record id, passages
indexed, the service's `timings_ms`, and the call's wall time as the client saw it.
`--concurrency` (default 4), `--only changing|standing`.

## candidates

```
cb-run candidates --data data/synth-42 --tenant <uuid> --collection cb --run runs/synth-42
```

For each question one request to `POST /v1/tenants/{tenant}/eval/candidates`
(docs/KB_EXPERIMENT.md section 3) with `as_of` the question's moment, the seven views of
DESIGN.md section 3.1, `n` 100, `exact` true, and relevance scores for the `now` view
with `per_list` 64. Point ids are joined back to records through `passages[].source_path`
(`cb_types::ids::record_of`; a passage of any other document keeps its point id as its
record id). When a record has several passages in one list it keeps its best position.

Writes `<run>/candidates.jsonl`: seven `Candidates` lines per question.

## vectors

```
cb-run vectors --collection <name> --tenant <uuid> --run runs/synth-42
```

Reads every passage of the tenant from the search index (`CB_QDRANT_URL`, read-only
scroll with vectors) and writes one vector per record (its first passage):
`<run>/vectors.f32` (little-endian f32 rows) and `<run>/vectors.json`
(`{"dim": 1024, "ids": [...]}`).

## extract

```
cb-run extract --transcripts data/synth-42/transcripts.jsonl --model <name> \
               --prompt prompts/extract.v1.md --run runs/synth-42-extract
```

One chat-completion call per transcript to an OpenAI-compatible endpoint
(`CB_MODEL_URL`), temperature 0, JSON answer. The transcript is rendered as
`Customer: …` / `Agent: …` lines after the prompt; `{{started_at}}` is replaced by the
call's start as an ISO 8601 UTC timestamp. The answer is parsed strictly; an answer
that is not the asked JSON counts as a failure after the retries, and is kept.

Writes `<run>/extractions.jsonl` (`cb_types::Extraction`; `{"found": false}` answers are
written with empty problem and solution so they are counted), with the wall time of the
successful call and the token counts the endpoint reported. `--concurrency`
(default 8), `--limit`, `--masked-with <file>` to read already-masked text for the
same ids.

## embed

```
cb-run embed --texts <jsonl of {"id","text"}> --run <dir> --name extractions
```

Embeds texts through the knowledge service's `/v1/embed` (batches of 64) and writes
`<run>/<name>.f32` + `<run>/<name>.json`.

## trips

```
cb-run trips --data data/trips --tenant <uuid> --collection cb --run runs/trips --n 1000
```

Times how long a new record takes to become searchable. For each record, one at a
time: note the clock, start the upload, and from the same moment ask the live search
route (`POST /v1/tenants/{tenant}/search`, 4 passages, cut-off 0.12) the record's
question every 25 ms until an answer contains the record. Writes one
`cb_types::PipelineTrip` per record with `written_ms` (the upload call returned) and
`first_hit_ms` (the first answer containing the record arrived), plus the service's
own `timings_ms`, to `<run>/trips.jsonl`. `--poll-ms`, `--give-up-ms` (default 30,000;
a record never found is a failure).

## questions

```
cb-run questions --data data/github --model <name> --per-section 1 --sample 800 --seed 42
```

Writes ordinary questions for a data set's standing records with a language model:
for a seeded sample of standing records, "write one question a customer could ask on a
support call that this passage answers, without copying its wording". Each question
gets the record as grade-3 gold and an `asked_at` drawn inside a randomly chosen
incident of the data set (`ordinary_near` when that incident's title shares a product
word with the record's title, `ordinary` otherwise). Appends to `queries.jsonl` and
re-writes the manifest.

## control

```
cb-run control export --from-collection <collection> --from-tenant <uuid> \
                      --questions helpq.json --window-start <unix> --out data/helpcentre
cb-run control import --data data/helpcentre --tenant <uuid> --collection <name>
```

`export` reads one tenant's passages (ids, both vectors, payload) from a search index
(`CB_QDRANT_URL`, read-only scroll) and writes them as a data set: one standing
`Record` per passage (the point id as the record id; `written_at` drawn 30 to 700 days
before `--window-start`, seeded), one ordinary `Query` per held-out question (gold: the
passage it was written from, grade 3; passages sharing its parent section, grade 1;
`asked_at` assigned later by the run that pairs it with incidents), and the points
themselves in `points.jsonl` (id, payload, dense vector, sparse indices and values).

`import` writes those points into the experiment's index under the experiment tenant
and collection label, adding `written_at` to each payload, through the index's own
upsert route. This is the one place a record enters the index without the upload
route: the passages keep the ids, vectors and enrichment they have in the product.
