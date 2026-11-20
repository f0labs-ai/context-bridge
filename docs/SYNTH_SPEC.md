# The generated incident corpus

The first paper's supplement (section 3) describes a synthetic contact-centre corpus:
10,000 transcripts over 500 incidents, 2,500 true records, 1,250 planted wrong ones,
average transcript 850 tokens, English, built from templates with personas, timestamp
noise, early wrong hypotheses and speech-recognition-style errors. No code for it
exists. This document turns that description into a generator: `cb-data synth`.

Everything is drawn from one seeded generator (ChaCha20), so a seed reproduces a data
set byte for byte. The paper's seeds are 42, 123, 456, 789 and 2024.

```
cb-data synth --seed 42 --out data/synth-42
cb-data synth --seed 42 --out data/synth-42 --asr 0.05     # also writes transcripts.asr05.jsonl
cb-data check data/synth-42
```

## 1. The company

A fictional provider, **Meridian Connect**: home internet, mobile, TV and the accounts
around them. Twelve services (home internet, home Wi-Fi equipment, mobile data, mobile
calls and text, voicemail, TV streaming app, TV set-top box, email, the customer web
portal, the mobile app, online payments, number porting). Twenty-four service regions
with invented names. No real company, place or person is named.

## 2. Incidents (500)

- Start times are spread uniformly over a 90-day window that begins
  2026-06-01T00:00:00Z.
- Lengths, as in the supplement: 200 short (10 to 59 minutes), 200 medium (1 to 8
  hours), 100 long (8 to 72 hours).
- An incident is one service failing in one region in one way, with one true cause, one
  workaround (or none) and one fix. At least four failure modes per service, at least
  three possible causes and two possible workarounds per mode.
- Two incidents never share service, mode and region while they overlap. Sixty pairs
  are planted on purpose with the same service in two different regions at the same
  time; they feed the ambiguous questions.

### 2.1 What is known, and when (2,500 true records)

Each incident has five states. A state is what the support floor knows from that
moment; its text restates everything known so far, the way a status update does, so
the latest state is always the whole answer.

| State | Becomes known at | Adds |
|---|---|---|
| 1 investigating | 2–10% of the incident's length | the symptom, where, since when; no cause yet |
| 2 root cause | 15–35% | the cause |
| 3 workaround | 30–50% | what customers can do meanwhile (or that nothing helps) |
| 4 fix in progress | 45–70% | the fix under way and the expected time back |
| 5 resolved | 100% | fixed, at what time, and any aftercare |

One `Record` per state: id `inc-0001.s1` … `.s5`, `thread` `inc-0001`, `tier`
changing, `written_at` the state's moment, `truth.phase`, `truth.correct = true`,
`truth.superseded_at` the next state's moment, and `truth.key_terms` = the words that
state adds (the cause phrase, the workaround phrase, the fix phrase and expected time,
the resolution time). `valid_until` is set only where the text states an end: state 4
ends at its expected time back; state 1 ends at its "next update by" time when it gives
one (half of the incidents).

### 2.2 Wrong statements (1,250 planted)

Early in an incident some agents guess. Each incident has two or three wrong
statements (250 incidents with two, 250 with three): a wrong cause on the customer's
side (their own equipment, their account, their wiring), a wrong cause on the
provider's side, or a wrong time back. One `Record` each: id `inc-0001.d1` …,
`truth.correct = false`, `truth.phase = investigating`, key terms no true state uses,
`written_at` the first call that states it.

## 3. Calls (10,000 transcripts)

- Calls per incident: six for every incident, and the remaining 7,000 shared out at
  random with long incidents weighted three times, medium twice, short once. Exactly
  10,000 in all.
- Call times are front-loaded: most calls come early in the incident, a few after it
  is resolved.
- Eighty agents. Each has one of four ways of speaking (wordy or brief, technical or
  everyday). An agent learns each state at its moment plus or minus 2 to 5 minutes
  (never before the incident starts), so two agents can disagree for a few minutes.
- In a call the agent states the latest state they know. Before they know the cause,
  an agent states one of the incident's wrong statements in about one call in five.
  `Transcript.states` lists the record ids stated.
- A call has a greeting, the customer's description of the problem, identity checks
  (name, account number, callback phone, sometimes an address or an email, all
  invented), the agent looking into it, the agent's statement, one or two follow-up
  questions, sometimes generic troubleshooting, and a close. Mean length 850 tokens
  (counted as words × 1.33), within ±50.
- Every key term of a stated record appears word for word in the agent's turns.
- Turn end times assume 2.5 words a second with short pauses.

### 3.1 Speech-recognition-style errors

`--asr p` writes a second transcript file in which each word is altered with
probability `p`: replaced by a sound-alike from a fixed table when one exists,
otherwise dropped (30%), doubled or joined by a filler (20%), or left. The supplement's
rates are 0, 0.05 and 0.15. Ids stay the same so results line up.

## 4. Standing documentation (320 records)

Eighty help articles for the same company (billing, plans, moving home, equipment
set-up, Wi-Fi, account and sign-in, mobile, TV, email, security, returns and
cancellation), four sections each, 60 to 120 words a section. One `Record` per section:
id `doc-012.3`, `tier` standing, no thread, `written_at` 30 to 700 days before the
window. The text is written by hand for this corpus and lives in the crate's assets.

## 5. Questions

| Class | How many | Asked | Gold |
|---|---|---|---|
| changing | two per state interval of states 1–4, per incident (4,000) | at a random moment while that state is the latest | grade 3 the latest state, grade 1 the earlier ones; `harmful` the incident's wrong statements |
| post_incident | one per incident (500) | 1 to 48 hours after it is resolved | grade 3 state 5, grade 1 the others |
| ambiguous | one per planted pair (60) | while both incidents are in progress; the question names the service and no region | grade 3 the latest state of each |
| ordinary | one per standing record (320) | at a random moment inside some incident | grade 3 the record, grade 1 its article's other sections |
| ordinary_near | 100 | during an incident on the same service as the article | as ordinary |

Question wording comes from pools of at least eight phrasings per intent (is it down,
what is the cause, is there a workaround, when is it back, is it fixed) and never
copies a record's key terms.

## 6. What `cb-data check` verifies

Ids are valid and unique; every gold and harmful id exists; a grade-3 record was
written at or before its question; `superseded_at` equals the next state's
`written_at`; the counts match section 2 to 5; every stated record's key terms appear
in its transcript; the mean transcript length is within range; the manifest's hashes
match the files.
