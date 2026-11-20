# Notes on the three public sources

Looked at on 2026-11-19 with `curl` and `jq` before the loaders were written, then
checked again against the full downloads. Where this differs from
`docs/PUBLIC_DATA.md`, the loaders follow what is written here.

## 1. GitHub status — www.githubstatus.com

**Endpoints.**

- `/history.json?page=N`: three calendar months per page, newest first (page 1 is
  Sep–Nov 2026). `months[].{name, year, incidents[]}` with `code`, `name`, `impact`
  (`none|minor|major|critical|maintenance`), `message`, `timestamp` (an HTML
  snippet). Page 32 covers Dec 2018–Feb 2019. Pages past the beginning still answer
  200 with empty months (page 80 is 2006–2007); there is never a 404. The fetch stops
  at the first page whose `start_time` is before the cut-off.
- `/incidents/<code>.json`: one incident. `incident_updates[]` is newest first, each
  with `status`, `body`, `created_at`, `updated_at`, `display_at` (millisecond
  times) and `affected_components[]` (`name`, `old_status`, `new_status`; or null).
  `components[]` is the incident-level list.
- `robots.txt` disallows only `/api/` and `/embed/`; the fetch uses neither.

**What the data is like.**

- 1,120 incidents are listed from January 2019 on; 33 of them are planned
  maintenance (`impact: maintenance`, `scheduled_for`, statuses
  scheduled/in_progress/completed). The loader leaves those out.
- Timelines with several updates begin in Dec 2018/Jan 2019. Before that, incidents
  were back-filled as one update whose body is a `<ul>` of times
  (1bvl46bpr0bl, 2018-12-06).
- Update statuses: investigating 4,487, monitoring 216, identified **2**, resolved
  1,087. GitHub almost never uses "identified", so `root_cause` is rare.
- `created_at` is when an update was posted. `display_at` (the time the page shows)
  was set back in 94 updates, by up to 16 days. Records use `created_at`.
- `updated_at`: since 2023 the resolved update's body is replaced, days later, by a
  root-cause summary (2024–2026: almost every resolved update; 587 such edits, median
  101 h after posting).
  The text shown at resolution is not kept. **Decision:** such a closing update
  becomes two records — at its posting time only the name and "Resolved", and the
  summary from its `updated_at` on (562 of these in the kept set). A non-closing
  update edited more than two minutes after posting keeps its posting time, but no
  question depends on it before the edit.
- After resolution some incidents get more updates filed as "investigating"
  (component notices, summaries). They are labelled Resolved, as part of the closed
  incident.
- Automated component notices can share a second. kz4khcgdsfdv (2024-08-15) posted
  "Actions is operating normally." twice 59 ms apart: folded into one record.
- Components were renamed ("GitHub Actions" became "Actions").
  "Visit www.githubstatus.com for more information" is a placeholder, not a
  component. 205 of the multi-update incidents name no component (mostly
  "Disruption with some GitHub services" and the 2019–2020 "Incident on <date>").
  Four that list none are named "Incident with X" (CodeQL, Dependabot, Slack
  Integration, GitHub Community Discussions); X is used as their component.
- Bodies hold `<br>`, a few `<a>`, `<p>`, `<i>`, `<code>`; no character references.

## 2. GitHub Docs — docs.github.com

- `/api/pagelist/en/free-pro-team@latest`: plain text, one path a line (3,251 pages).
- `/api/article?pathname=<path>`: JSON `{meta: {title, intro, product, breadcrumbs,
  documentType, redirectedFrom}, body}`. The body is Markdown with the templating
  already resolved (no `{%` left) plus some HTML: `<div class="ghd-tool ...">`
  platform blocks, `<svg aria-label=...>` icons inside sentences, a few `<table>`s,
  `<kbd>`, `<code>`. `documentType` is `article`, `category`, `subcategory` or
  `product`. An unknown path answers 404 with `{error}`. A moved page answers with
  the new article and `redirectedFrom`.
- `robots.txt` allows everything.
- **Chosen over a sparse checkout of github/docs:** the API returns the rendered
  text with versioning and reusable snippets resolved. The repository would need a
  Liquid engine and its `data/` folder to give the same text.
- The kept areas (the status page's components, with only the REST guides,
  authentication and quickstart for "webhooks and API"; the generated endpoint
  reference is left out) hold **1,404 pages, 1,189 of them articles with prose**,
  not "about 400". Copilot alone has 573 pages and gives 3,044 of the 6,730 passages.
- A passage is "<title>: <section heading>" on its first line, then the section's
  prose; the article's title is added for context. 58 passages repeating an earlier
  one word for word (reused snippets) are dropped.

## 3. Zenodo 10.5281/zenodo.14010282 — "Cloud Incident Reports (2016-2024)"

- One file, `data.zip`, 5,802,271 bytes (md5 1bb311c9d08c4853a1373e5370231d54,
  sha256 31b4ff81f78252b6c027019fbd4c1c625ba0c0b9097395216559f64930c5ddbe). It
  holds `1_raw_data/{aws,azure,gcp}_provider.parquet`, CSV exports in
  `2_clean_data` (the `description` column is a Python repr of the updates),
  `3_sample_data`, `4_label_data` and `data_process.py` (not run).
- `gcp_provider.parquet` has 2,186 rows (parquet-cpp-arrow 16.1.0, Snappy).
  Columns: event_start_time, first_notification, event_end_time, external_desc,
  last_notification, most-recent-update, public, service_id, service_name,
  severity, updates (list of {affected_locations, created, modified, status, text,
  when}), year, vendor, id, most_recent_update, status_impact, affected_products,
  currently_affected_locations, previously_affected_locations.
- **A row is a saved copy of an incident, not an incident.** The compiler saved the
  feed again and again: one incident appears in up to 17 rows, each with the
  updates known when it was saved. 2,186 rows are **819 incidents**, and the 9,450
  saved updates are **3,585 different updates**. Rows are joined on the first
  update's time, on the new format's incident id, and (old format) on product plus
  stated start. That last link catches developers-console 2019-06-06, whose first
  update was taken down between savings.
- Two feed formats are mixed:
  - old (2016–2021): no id, no update status, `service_id` like `cloud-spanner`,
    times ending in `Z`;
  - new (2021–2022): an id, update status `AVAILABLE|SERVICE_INFORMATION|
    SERVICE_DISRUPTION|SERVICE_OUTAGE`, times ending in `+00:00`, `affected_products`.

  130 incidents appear in both.
- Linked incidents share an update (same text, same second: 2021-02-13 GKE and
  Data Fusion, 2021-02-18 Compute Engine and Cloud Functions) but are separate
  incidents; a shared update never joins them.
- 92 updates were saved with more than one wording (typo fixes, "DAYOFWEEK"
  filled in). The earliest saved wording is used.
- `created` is when posted. `when` (the time shown) differs in 281 old-format
  updates, from 12 days earlier to 59 days later. `modified` is when edited.
- Texts carry `Description:`, `Diagnosis:`, `Workaround:`, `Summary:` fields; the
  new format writes Markdown (`#` headings, `**bold**`, links).
- The workaround field appears in 447 of the 819 incidents and mostly says "None at
  this time". 127 kept incidents state a real workaround at some point.
- Covers 2016-11-02 to 2022-12-29 (the record's own table says 2016–2021).
- Old updates have no status, so an incident's course is read from the AVAILABLE
  status, resolution words ("has been resolved", ...) and the stated end time:
  - **open** until any of these appears;
  - **over** from the stated end, or at the last update once one has appeared;
  - **unclear** in between.

  Phases come from short word lists in `gcp.rs`.
- `robots.txt` disallows `/api` except `/api/records/*/files` and asks for ten
  seconds between requests. The fetch makes one request, to the file URL, at that
  pace. While looking, the record pages and `/api/records/<id>` (metadata) were
  opened once by hand.

## 4. Zenodo 10.5281/zenodo.14018219 — "LLM Service Outages and Incident Reports"

- One file, `LLM Service Outages and Incident Reports.zip`, 250,498 bytes (md5
  36b1ffe4a57d7976de605b805a464793, sha256
  ee4c6856477e26344b5fc68c9e55012caf2b0c65a366235493547b9fa3648747).
  `raw_data/incident/2024-08-31/<provider>/incident_history_YYYYMM_YYYYMM.csv` (25
  files), `raw_data/uptime`, `clean_data/incident/incident_stages.csv` (one UTC time
  per stage), `clean_data/outage`.
- Incident CSV columns: Incident_Title, Incident_Link, Incident_color,
  Incident_Impact, Updates (a JSON list, newest first, of Update_Title, Update_Body,
  Update_Timestamp), Service ("This incident affected: API and ChatGPT.").
- 542 incidents (OpenAI 365, Anthropic 141, Character.AI 36) with 1,754 updates,
  2021-02-09 to 2024-08-30. Stages: Resolved 542, Investigating 393, Monitoring 304,
  **Update 271** (no status change), Identified 211, **Postmortem 33** (always after
  Resolved).
- Times read "Posted 3 days ago. Aug 28, 2024 - 13:36 PDT": US Pacific with
  PDT/PST, to the minute. They are converted to UTC and checked against
  `incident_stages.csv`: all 542 resolved times, and the first investigating (381),
  monitoring (291) and identified (205) times, agree.
- 30 consecutive pairs share a minute. One pair repeats word for word (Anthropic
  gm6vlw4gc377, 2024-06-04 08:40 PDT) and is folded. The others differ in their
  status and keep the source's order.
- 60 incidents name no component (an empty Service).

## 5. Terms (pages opened on 2026-11-19)

- GitHub status and docs pages: GitHub Acceptable Use Policies, section 7,
  https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies#7-information-usage-restrictions
  — "Researchers may use public, non-personal information from the Service for
  research purposes, only if any publications resulting from that research are open
  access." The status page's footer links the GitHub Terms of Service, which make
  the policies binding ("You agree that you will not under any circumstances violate
  our Acceptable Use Policies").
- GitHub Docs content: https://github.com/github/docs#license — "This project is
  dual-licensed under: Creative Commons Attribution 4.0 - for documentation and
  content in the assets, content, and data folders (see LICENSE)". The licence text
  is https://github.com/github/docs/blob/main/LICENSE ("Attribution 4.0
  International").
- Both Zenodo records, https://zenodo.org/records/14010282 and
  https://zenodo.org/records/14018219: "Creative Commons Attribution 4.0
  International" — "The Creative Commons Attribution license allows
  re-distribution and re-use of a licensed work on the condition that the creator
  is appropriately credited."
  (https://creativecommons.org/licenses/by/4.0/legalcode). The licence is the
  compiler's; the incident texts were written by Google, OpenAI, Anthropic and
  Character.AI on their status pages.

None of the terms forbids this use: research, published open access, with credit.

## 6. Rules the loaders add to docs/PUBLIC_DATA.md

- **Unique texts.** A record's text carries its update's UTC time ("Update
  2026-11-16 12:35:29 UTC, Investigating: …"), to the second where the source has
  seconds and to the minute for the LLM set. Status pages repeat updates word for
  word; without the time, 843 GitHub, 47 Google Cloud and 109 LLM updates would
  share a text with an earlier one (264, 43 and 77 distinct texts). The conversion
  refuses to write a data set in which two records share a text after collapsing
  whitespace.
- **Who a question is about.** A question names a component of the affected
  service, or the service as a whole. It is asked only when every incident in
  progress that its wording fits is that one incident; for an ambiguous question,
  only the pair. An incident that names no component may concern any component of
  its service. Incidents that were not kept (fewer than three updates, never
  closed) have no records, but they block questions while they are in progress.
  The source never says when an unclosed incident ended, so it counts as in
  progress until 48 hours after its last update.
- **Wording.** A question names the component or product whenever its incident
  names one, in all three sets; only incidents that name none are asked about as
  the service as a whole ("GitHub"). A share of component-naming questions can be
  switched to the service-wide form (`service_question_share`); it is 0 in all
  three sets. (A share of 0.25 was tried for GitHub on 19 Nov and reverted: the
  request for it rested on a miscount.)
- **Horizon.** No question is asked after the source ends: the moment the GitHub
  download began, or the latest update in an archive.
- **Post-incident questions** are asked 1 to 48 hours after the last update, naming
  something no other incident may concern from the end of this one to the moment
  asked.
- Questions are not asked within the time span an update's stamp covers (one
  second, or one minute for the LLM set), nor before an edited update's final text
  existed.
