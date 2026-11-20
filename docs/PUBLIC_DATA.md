# Public incident timelines

Real incident updates are written by people while an incident is in progress, and each
update replaces the one before it. That is the situation the paper describes, without
any template. `cb-data public` turns three public sources into data sets in the shape
of DESIGN.md section 2.

```
cb-data public github --raw data/raw/github --out data/github
cb-data public gcp    --raw data/raw/gcp    --out data/gcp
cb-data public llm    --raw data/raw/llm    --out data/llm
cb-data check data/github
```

Each command has a `--fetch` step that downloads into `--raw` (kept out of git) and a
convert step that reads only from `--raw`, so a conversion can be repeated offline.
Fetching is polite: at most two requests a second, a descriptive User-Agent, retries
with back-off on 429 and 5xx, and it resumes where it stopped.

## 1. GitHub status history (`github`)

- **Source.** `https://www.githubstatus.com/history.json?page=N` lists incidents page
  by page (about 1,950 back to 2010). `https://www.githubstatus.com/incidents/<code>.json`
  gives one incident with every update and its timestamp.
- **Terms.** GitHub's terms of service allow research use of public, non-personal
  information when the resulting publication is open access. Record this sentence and
  its source URL in the manifest.
- **Kept.** Incidents created on or after 2019-01-01 with at least three updates
  (earlier ones were back-filled as a single update).
- **Records.** One per update, oldest first: id `gh-<code>.<n>`, thread `gh-<code>`,
  tier changing, `written_at` the update's `created_at`, title the incident's name,
  text `"<incident name>. <Status>: <update body>"`, `truth.phase` from the update's
  status (investigating, identified → root_cause, monitoring → fix_in_progress,
  resolved, postmortem → resolved), `truth.superseded_at` the next update's time.
  No `valid_until`.
- **Changing questions.** For each gap between two updates that lasts at least two
  minutes, one question asked at a random moment in the gap; phrasings name the
  affected component or the incident's subject, never the update's own wording. Gold:
  grade 3 the latest update at that moment, grade 1 the earlier ones.
- **Post-incident questions.** One per incident, 1 to 48 hours after its last update.
- **Ambiguous questions.** Where two kept incidents overlap in time and share a
  component, one question that names the component only.

### 1.1 GitHub documentation as standing knowledge

- **Source.** GitHub's public documentation (`github/docs`, content licensed CC BY 4.0;
  record the licence text's URL). Take the English articles of the product areas that
  appear as components on the status page (pull requests, Actions, issues, Codespaces,
  Pages, Packages, Copilot, repositories and Git, webhooks and API, authentication),
  about 400 articles.
- **Records.** One per second-level section of an article: id `ghdoc-<slug>.<n>`,
  tier standing, no thread, title the article's title, text the section's heading and
  body as plain text (templating tags resolved or removed, tables and code blocks
  dropped, 40 to 300 words; longer sections are split at paragraph ends).
  `written_at` 2018-06-01T00:00:00Z for all.
- **Ordinary questions** are not written here. `cb-run questions` writes them with a
  language model from sampled sections and appends them to `queries.jsonl`.

## 2. Google Cloud incident reports (`gcp`)

- **Source.** Zenodo 10.5281/zenodo.14010282, "Cloud Incident Reports (2016-2024)",
  CC BY 4.0. Use the Google Cloud part: about 2,180 incidents with about 4,690 updates
  timed to the second; 1,165 carry a workaround field.
- **Records and questions.** As for GitHub: one record per update, thread per incident,
  changing questions in the gaps. Where the workaround field first becomes non-empty,
  add one question asking for a workaround shortly after that update.

## 3. LLM service outages (`llm`)

- **Source.** Zenodo 10.5281/zenodo.14018219, "LLM Service Outages and Incident
  Reports", CC BY 4.0: 542 incidents with investigating / identified / monitoring /
  resolved updates timed to the minute.
- **Records and questions.** As for GitHub.

## 4. Not used

Cloudflare's status feed (its site terms forbid automated collection for
machine-learning work), the Kaggle "Customer Support on Twitter" set (non-commercial,
share-alike), CRAG (non-commercial).
