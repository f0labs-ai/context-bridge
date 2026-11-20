# cb-run — decisions where the specification was open

`docs/RUN_TOOLS.md` is the specification and `docs/KB_EXPERIMENT.md` the service contract;
the service's experiment build is private and documented with it.
Where a reading had to be chosen, `cb-run` takes the one that loses no data. Each choice
is listed here with its reason.

## 1. Every command

1. **Exit status.** 0 when the run directory holds no failure line at all; 2 when any
   line remains in `<run>/failures.jsonl`, whatever command wrote it; 1 when the command
   could not run (a missing setting, a run directory that belongs to something else, a
   service that is not the experiment build). So `candidates` exits 2 while an upload
   failure of the same run directory is open: a run with a hole never passes for
   complete.
2. **`failures.jsonl` holds the failures that remain.** A failure is appended the moment
   it happens, one line per item: `command`, `id`, `kind` (one word), `message`, `url`
   (shown without user, password or query), `status`, `tries`, `raw` (what came back: the
   first 2,000 bytes of a service's body, or the model's whole answer) and `at`. When a
   command ends, its own earlier lines for items that are now done are removed, a newer
   line replaces an older one for the same item, and every other line (other commands,
   lines this tool cannot read) is kept as it was.
3. **The run manifest** is `<run>/manifest.json` = `{"entries": [...]}`. A command adds its
   entry when it starts (`"ending": "running"`) and completes it when it ends
   (`complete`, `incomplete`, `stopped`), so a killed run shows as one that never ended.
   An entry holds: `tool`, `version`, `git_commit` (with `-dirty` when tracked files
   differed from the commit at build time), `command`, `flags`, start and end, the
   knowledge service's address and its `/v1/info` exactly as answered (and the embedding
   model it named), the index's address and collection, the model's address, the name
   asked for, the names the endpoint reported, the prompt's version and sha256, whether
   JSON mode stayed on, the extra body, the machine (os, arch, CPUs, and `--machine` /
   `CB_MACHINE`), and counts. Entries written by other tools are kept byte for byte. A
   `manifest.json` of another shape is left untouched and the command does not run.
   `cb-eval` should append its own entries to the same `entries` array.
4. **A run file belongs to one thing.** Every command skips what its output file already
   holds. Pointed at another tenant or another build of a data set, the same run
   directory would skip everything and look complete. So the manifest entry records
   what the output file belongs to (service address, tenant, label, sha256 of
   `records.jsonl`; for `extract`: endpoint, model, prompt sha256, transcripts sha256,
   masked file sha256, extra body; for `questions`: endpoint, model, prompt sha256,
   records sha256, seed, product-word list sha256, extra body; for `embed`: the service;
   for `control import`: index, collection, tenant, label, points sha256), and a later
   run that differs is refused while the file holds lines: "use another `--run`".
5. **One `cb-run` per run directory.** An operating-system lock on `<run>/.cb-run.lock`,
   released when the process ends, so two resumes started by mistake cannot write the
   same item twice and a crash leaves no stale lock.
6. **Retries.** At most three tries in all. Back-off: `--retry-base-ms` (default 500 ms)
   before the second try, doubled before the third, each wait drawn between half and the
   whole of it (jitter), never over 30 s; a `Retry-After` in seconds is honoured up to
   that cap. Tried again: 429, 5xx, no answer in time, a connection that failed or broke
   off, a 200 whose body is not the contract's shape (a truncated or rewritten answer is
   as transient as a 503), and a model answer that is not the asked JSON (the
   specification: "counts as a failure after the retries"). Never tried again: any
   other 4xx, a well-formed answer that cannot be used (a contract failure), a bad input.
7. **Timeouts.** Connect 5 s (`CB_CONNECT_TIMEOUT_MS`), whole request 120 s
   (`CB_TIMEOUT_MS`). Redirects are not followed: a token would follow them.
8. **Logs** go to standard error, as text or JSON (`--log-format`, `CB_LOG_FORMAT`):
   `command_started`, one `batch` line per 100 items (count, failures, elapsed, totals),
   one `item_failed` line per failure (id, kind, status, tries — never the message,
   which may quote a service), `command_ended`. Never a record's text, never a token.
9. **Tokens and keys** (`CB_KB_TOKEN`, `CB_MODEL_KEY`) are held in a type with no
   `Display`, a hidden `Debug` and no serialisation; `--help` hides environment values;
   flags recorded in a manifest are the command's own, which never include a token.
   Tested by scanning every run file and the binary's output.
10. **Order.** Results are handed to the output file in input order whatever the
    concurrency (a bounded, ordered stream), so a run without failures is the same file
    every time; a resume appends what was missing, in input order.
11. **Crash safety.** A line is written whole in one write. Reading one of this tool's run
    files back removes a cut-off last line (and only that); a broken line anywhere else
    is an error, never skipped. Whole files (vector files, data sets, manifests) are
    written to a temporary file beside them, flushed and renamed.

## 2. `upload`

12. The service's own multipart fields (`kb-api` `ingest_ep`): `files` (file name
    `cb/<record id>.md`, `text/markdown`), `collection`, `attributes` (a JSON string:
    `written_at`; `valid_until` and `title` when set; `document_ref` = the record id).
    `wait_ms` and `new_add` are not sent.
13. The answer is read strictly: exactly one `sources[]` entry, for the document's path;
    no `jobs[]` (a queued upload means the service runs the bulk tier, which the
    experiment must not); `chunks_indexed` above 0; `attributes` echoed with the same
    `written_at` and `valid_until`; `timings_ms` present. Anything else is a contract
    failure for that record.
14. `uploads.jsonl`: `record`, `passages_indexed` (`chunks_indexed`), `timings_ms`,
    `wall_ms` (the successful try only, from sending to the last byte), `tries`, and what
    else the service said (`source_id`, `bytes_processed`, `refined`, `task_extracted`,
    `symptom_chunks`, `spoken_chunks`, `redactions`).
15. **Identical texts.** The index keys a passage by its text (finding 1 of the service's
    document). A record whose text is byte for byte an earlier record's is not uploaded:
    a `duplicate_text` failure. The first in file order keeps the points on every run.
16. **Every record is checked in the index after the uploads** (`GET /v1/tenants/{t}/
    sources/count?source_path=cb/<id>.md&collection=<label>`, the check the service's
    document advises). A count different from `passages_indexed` — a shared paragraph
    taken over by a later upload — is a `not_whole_in_index` failure, found again on
    every run until the data set is fixed. All uploaded records are checked, whatever
    `--only` selects.
17. Needs `/v1/info` to list `document_times` and `ingest_timings`.

## 3. `candidates`

18. One request per question with the seven views of DESIGN.md 3.1: `now`;
    `batch_Δ` with `index_as_of = floor(asked_at / Δ) * Δ` (`cb_types::views`); `window_W`
    with `written_from = asked_at - W`. `n` 100, `exact` true, `collections` the label,
    `rerank` `{now, 64}`.
19. **The join.** A point is its document's record (`ids::record_of(source_path)`), or —
    for any other document — its own point id. In a list a record keeps its best
    position and that position's score; in `rerank` its best score (the service sorts
    best first, so this is also its best position). A point in a list that `passages`
    does not describe is a contract failure, never a guess.
20. **The index must hold the data set's times.** For every passage of a data-set record
    in an answer, `written_at` and `valid_until` must equal the data set's; else the
    question fails. Views are cut by these times inside the service.
21. A view missing from the answer, or one not asked for, fails the question. A
    question's seven lines are written in one write. On a resume only the missing
    (question, view) lines are asked for and written. A file that holds a (question,
    view) twice is refused, and so is a line whose `as_of` differs from the data set's
    `asked_at` (another build of the data set).
22. The service reports one `timings_ms` per request; it is written on the `now` line
    only, so adding up a file's lines counts each request once.
23. `--concurrency` defaults to 1; lines stay in question order at any concurrency within
    one run (decision 10).
24. The manifest says how many of the data set's records have an upload line in the same
    run directory: keyword scores use the whole index's word counts (finding 3), so lists
    are comparable only when every record was loaded first.
25. Needs `eval_candidates` and `document_times`.

## 4. `vectors`

26. `--collection` is the search index's own collection (as `/v1/info` names it under
    `collection`), not a label. Every passage of the tenant is read, whatever its label.
27. The tenant's points are counted (exactly) and then scrolled in pages of 256 with
    `source_path` and `seq` and the `dense` vector only, never the text. A read that
    returns another number than the count is a failure and nothing is written.
28. A record's "first passage" is its lowest `seq` (ties: the lowest point id), so the
    choice never depends on the order the index answers in. Rows are in record-id
    order. A passage without a dense vector is a failure naming the point.
29. The file pair is written whole on every run; this run's outcome replaces the
    command's earlier failures.

## 5. `extract`

30. The message is the prompt with `{{started_at}}` replaced by the call's start
    (`2026-06-01T00:00:00Z`), trailing white space trimmed, an empty line, then one line
    per turn, `Customer: …` / `Agent: …`. A line break inside a turn becomes a space, so
    no speaker's words can pass for another speaker's line.
31. The request: `model`, one user message, `temperature` 0, `response_format`
    `{"type": "json_object"}`; `max_tokens` only with `--max-tokens`. When the endpoint
    answers 400 or 422, the same request goes once more without `response_format`; if
    that one is answered, JSON mode stays off for the rest of the run (the manifest says
    so); if it is refused too, the first refusal stands.
32. **`--extra-body <JSON object>`** (also on `questions`): its fields are added at the
    top level of every request body, after the tool's own. Refused: `model`, `messages`,
    `temperature`, `response_format`, `max_tokens` (set by the tool), and `stream` and
    `n` (they would change the answer's shape: pieces, or several answers of which one
    would be read and the others' tokens counted). Recorded in the manifest (so: no
    secrets in it) and part of what the run file belongs to.
33. The answer is read from `choices[0].message.content` only. `reasoning`,
    `reasoning_content` and everything else are not read; a null content is an answer of
    another shape (tried again, then a failure).
34. **Strict reading.** Exactly one JSON object in the answer (a code fence or words
    around it are put up with; none, or two, is a failure kept with the raw answer).
    `{"found": false}` is written with empty problem and solution, zero confidences and no
    times. With `found` true: problem and solution not empty, a confidence object with
    three numbers in [0, 1], `t_start` / `t_eta` null or ISO 8601: `Z`, an offset
    (`+02:00`, `+0200`, `+02`) converted to UTC, no zone read as UTC (the prompt asks for
    UTC), seconds optional, a fraction dropped, a space for `T` allowed; a date without a
    time, or anything else, fails the whole answer (never turned into "no time"). Keys
    beyond the asked ones are ignored.
35. `latency_ms` is the wall time of the call that gave the written answer only (sending
    to the last byte), never the retries before it. `tokens_in` / `tokens_out` are the
    endpoint's `usage.prompt_tokens` / `usage.completion_tokens`, 0 when it reported
    none; such answers are counted in the manifest (`answers_without_token_counts`), and
    the manifest sums the tokens.
36. `--limit N` reads the first N transcripts of the file. `--masked-with` must hold the
    same transcript with the same number of turns and the same speaker per turn;
    otherwise that transcript fails — its unmasked text is never sent instead.
37. `prompt_version` is the prompt file's name without its extension (`extract.v1`).

## 6. `embed`

38. Batches of 64, `kind` `"query"` (the only kind the service takes; at most 128 texts,
    each 1 to 1,000 characters). A batch the service refuses (a 4xx) is sent again text
    by text, so one text it will not take costs only its own row. Ids must be unique.
    `--name` is a file name, never a path.
39. Rows are appended in input order and the pair is rewritten every 16 batches and at
    the end; a resume skips the ids already in the file.

## 7. `trips`

40. The records timed are the first `--n` of the data set (file order) that have a
    question; a record's question is the first one naming it as grade-3 gold. Fewer than
    `--n` is a `shortfall` failure; the ones that exist are timed all the same.
41. Before a trip, one search (not timed, retried): a record the search already returns
    cannot be timed (its first hit would be the old copy) — an `already_searchable`
    failure. A record whose `written_at` is in the future or whose `valid_until` has
    passed is a failure before any request: the live search, asked as of now, would
    never return it.
42. The trip: one clock is read, the upload and the first search start at once, then a
    search every `--poll-ms` (missed ticks skipped), at most `--max-polls-in-flight`
    (default 8) under way; one not sent because of the bound is counted. Each search's
    answer is stamped on the trip's clock when the whole answer is in; the earliest
    arrival that holds the record is `first_hit_ms`, whatever order the searches were
    sent in. `written_ms` is when the upload's answer was in. A record found before its
    upload returns (the search sees the write before the archive and the quality check
    finish) is kept as found. `--give-up-ms` counts from the start for both.
43. In a trip the upload is tried once (a retry would be another trip) and so is each
    search; a failed search is counted (`polls_failed`) and decides nothing.
44. The search is a call's: `top_k` 4, `score_threshold` 0.12, `collections` the label, no
    `as_of` (a live call asks as of now).
45. `trips.jsonl` lines are `cb_types::PipelineTrip` (`record`, `written_ms`,
    `first_hit_ms`) with `timings_ms`, `passages_indexed`, `polls`, `polls_failed`,
    `polls_held_back` and `first_hit_mode` beside them.
46. The first upload after a service start is slower (finding 4); every trip is kept and
    the analysis decides (its `timings_ms.mask` shows it).

## 8. `questions`

47. The prompt is `crates/cb-run/prompts/questions.v1.md` (built in), or `--prompt`;
    `{{count}}`, `{{title}}`, `{{text}}` are replaced in one pass (a passage that contains
    a placeholder stays as it is). The answer is `{"questions": [...]}` with exactly as
    many questions as asked for, none empty.
48. **The sample** is the first `--sample` picks of one seeded shuffle of the standing
    records sorted by id, so a larger sample holds a smaller one. Draws are ChaCha20 with
    the ranges and the shuffle written here, so a seed gives the same sample on any
    machine and any library version.
49. **A question's moment** is drawn from a stream of the seed and the record's id, so it
    never moves when the sample changes: an incident chosen uniformly among the incidents
    that end at or after the record was written, and a moment uniform in
    `[max(incident start, record written), incident end]`. An incident is the changing
    records of one thread, from its first to its last `written_at`, titled by its first
    record. A record written after every incident ended is a failure.
50. **"Shares a product word."** With `--product-words <file>` (one word or phrase a
    line), the two titles must both hold one of them as whole words; without it, any
    title word of three letters or more that is not a common word counts. Shared:
    `ordinary_near`; otherwise `ordinary`.
51. Ids are `oq-<fnv1a64 of the record id, hex>-<k>`; gold the record (grade 3); no thread.
52. Questions go to `<run>/questions.jsonl` as they come (default run directory
    `<data>/run-questions`); at the end `queries.jsonl` is replaced in one step by its old
    bytes plus the new lines, and the data set's `manifest.json` gets the new count, the
    file's sha256 and one provenance line (model, prompt and its sha256, seed).
53. A planned question the data set already holds with another moment or class (another
    seed, another word list) is refused: two plans are never mixed in one data set.
54. `--sample` larger than the standing records is a `shortfall` failure; every standing
    record still gets its questions.

## 9. `control export` and `control import`

55. `export` is given a reader of the index and nothing else: it cannot write. It counts
    and scrolls one tenant (whole payload, `dense` and `sparse`), and the two numbers
    must agree.
56. One standing record per passage: the point id as the record id (an id that cannot be
    a record id is a failure), the payload's text (a passage without text or without a
    dense vector is a failure), title = the payload's `title`, else its `heading`;
    `source` = the output directory's name. `written_at` is drawn 30 to 700 days before
    `--window-start` from a stream of `--seed` (default 42) and the point id, so it does
    not depend on which other passages are exported.
57. One ordinary question per held-out question: id `hq-<qid>`, gold the passage it was
    written from (grade 3) and the passages with the same `parent_id` (grade 1).
    `asked_at` is `--window-start`, a placeholder until the run that pairs questions with
    incidents sets it (every record is written by then). A question whose passage is not
    exported is a failure, never dropped in silence. The held-out file is
    `{"tenant", "questions": [{"qid", "question", "gold_id", …}]}`; its `tenant` must be
    `--from-tenant`.
58. The data set is `records.jsonl`, `queries.jsonl`, `points.jsonl` (id, the payload as
    the product holds it, `dense`, `sparse`), `export.json` (where it was read from) and,
    last, `manifest.json` with every file's sha256. `--leave-out-label` (repeatable)
    leaves passages of a label out: pass `atlas` for the product's help centre, whose
    Site Atlas the live search never answers from, because the import gives every passage
    the experiment's label.
59. `import` refuses when `--collection` equals the export's collection, or when the
    index's server (host and port; `localhost`, `127.0.0.1`, `::1` and `0.0.0.0` counted
    as one) is the export's. It checks the data set's hashes first. It writes in batches
    of 64 with `PUT /collections/{c}/points?wait=true` and needs `"status": "completed"`:
    the ids and vectors as exported, the product's payload with `tenant_id`, `collection`
    (= `--label`) and `written_at` set. It never creates a collection: the service's
    `ensure_collection` owns its shape and indexes. `<run>/imported.jsonl` lists what is
    written; a resume skips it.

## 10. Proposed changes outside this crate

- `cb_types::Extraction.tokens_in` / `tokens_out` as `Option<u32>`: an endpoint that
  reports no usage now writes 0, which reads like a measurement (the manifest counts
  such answers).
- DESIGN.md section 3: document the run manifest's shape (`{"entries": [...]}`, one
  entry per command run) so `cb-eval` appends to it the same way.
- The question-writing prompt lives in `crates/cb-run/prompts/`; move it beside
  `prompts/extract.v1.md` if prompts should live in one place.
