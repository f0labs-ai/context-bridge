# cb-data synth — decisions the specification leaves open

Each entry names the question, the reading chosen (the one closest to
docs/SYNTH_SPEC.md and the supplement it quotes) and where it lives in the code.

## Time

1. **Clock and calendar.** Every moment is Unix seconds, UTC, and the fictional
   company keeps UTC as its local time, so "today", "yesterday" and "09:40" in
   records and calls are UTC (`clock.rs`).
2. **Lengths.** Drawn uniformly in seconds within each class: short 10–59 min,
   medium 1 h to just under 8 h, long 8–72 h (`LengthClass::bounds`).
3. **State moments.** One draw per state from its share range (2–10%, 15–35%,
   30–50%, 45–70%), then sorted; sorting keeps each state inside its own range
   because both ends of the ranges increase. A draw is rejected when two states
   are closer than max(30 s, 2% of the length), so every state is the latest for
   long enough to be asked about, or when the start and the resolution show the
   same clock minute (`draw_timeline`).
4. **Distinct clock times.** Within one incident the start, the "next update by"
   time, the expected time back, the resolution time and the wrong time back
   never show the same HH:MM, so a clock-time key term belongs to one statement
   only (`distinct_clock`, `wrong_time_back`).
5. **"Next update by".** Exactly 250 incidents give one, 15 to 120 minutes after
   the first state, on a five-minute mark; it is `valid_until` of `.s1`.
6. **Expected time back.** Drawn around the true end (15% of the length before
   it to 25% after), on a five-minute mark, at least five minutes after the
   fourth state; it is `valid_until` of `.s4`. When it falls before the true end,
   `.s4` expires before `.s5` is written (as real estimates slip), and a call in
   that gap uses "running late" phrasing that still carries the fix and the time.

## Incidents

7. **Overlap rule, made stronger.** The specification forbids two overlapping
   incidents with the same service, mode and region. The generator forbids the
   same service and region (any mode) within 49 hours of each other, so a
   question naming a service and region, including a post-incident question
   asked up to 48 hours later, is about one incident only (`SAME_PLACE_GAP`).
8. **Planted pairs.** 60 pairs: one service, two different regions, modes drawn
   independently, the second incident starting between a quarter of the first's
   length before it and half its length after it. Each pair reserves an asking
   window (up to two hours) inside the time both have a first state and neither
   is resolved; no other incident of that service is in progress during it, so
   the ambiguous question has exactly two right answers (`place_pair`).
9. **Exact shares.** 250 incidents with three wrong statements and 250 with two;
   250 with a "next update by"; 100 with no workaround; aftercare advice in the
   resolved state of 70% (random).
10. **Wrong statement kinds.** Three wrong statements are one of each kind
    (customer side, provider side, time back); two are two different kinds.

## Records

11. **Text shape.** `"<incident title>. <Status>: <body>"`, the shape
    docs/PUBLIC_DATA.md uses for public status pages; status words are
    Investigating, Identified, Workaround, Fix in progress, Resolved. The title is
    "<Service> in <Region>: <failure mode>". Every state restates the symptom,
    where and since when, and everything learnt so far.
12. **Key terms.** Phrases are stored without a leading article.
    `.s1` symptom phrase + start time; `.s2` cause phrase; `.s3` workaround
    phrase, or "no workaround"; `.s4` fix phrase + expected time back; `.s5`
    resolution time. The "next update by" time is not a key term. Region and
    service names are never key terms, because questions name them.
13. **Wrong statements as records.** A call note, not a status update: it names
    the service, the region and the date of the call that first said it, then the
    claim, so no two notes can read alike (two incidents on one service and region
    are always more than a day apart). `title` is empty (a call note has no
    incident title). `written_at` is the end of the turn in which it was first
    said. `superseded_at` is the moment of the root-cause state (the
    investigating phase ends), or `written_at` itself when first said after that.
    Ids `.d1`, `.d2`, … follow the order in which they were first said.
14. **No two records share a text.** Collapsing whitespace, every record text is
    unique by construction (see 13); the generator also checks and refuses to
    produce a corpus with a shared text (`SynthError::SharedText`), and
    `cb-data check` fails on one.
15. **Standing records.** Text "<heading>. <section text>", `title` the article's
    title, no `truth`, `written_at` one draw per article (30 to 700 days before
    the window) shared by its four sections.

## Calls

16. **When calls come.** 6% after the resolution (5 minutes to 12 hours after);
    of the rest, 62% in an early burst over the first 15% of the incident and
    38% spread evenly over it. See "Tension" below for why.
17. **What an agent knows.** Each agent learns each state of each incident at the
    state's moment plus or minus 2–5 minutes, never before the incident starts,
    and never before the resolution for the resolved state (nobody can report a
    finished repair early). Knowledge is taken at the start of the call.
18. **What is stated.** The latest state known; `Transcript.states` holds that
    state's id, plus the wrong statement's id when one is made. An agent who
    knows nothing says so, troubleshoots and raises a fault.
19. **Wrong statements in calls.** A call is eligible when the agent does not yet
    know the cause and the incident is not over. Each eligible call carries one
    with chance 0.1; every wrong statement is also said at least once (more
    eligible calls are drawn when needed, and when an incident has fewer eligible
    calls than wrong statements its earliest other calls are moved early and given
    agents who learn the cause late). The realized share of eligible calls with a
    wrong statement is about one in five and is written to the manifest
    (`calls_with_wrong_statement` / `calls_before_cause_known`). A wrong time back
    lies 15 minutes to 2 hours after the last call that says it; an agent who has
    found nothing on the board words it as a guess.
20. **Agents and personas.** 80 agents, 20 per way of speaking;
    `Transcript.persona` is the agent's way of speaking (`wordy_technical`, …).
    The caller is invented per call: name, an `MC` account number, a phone number
    from the ranges set aside for drama (07700 900xxx, 01632 960xxx), an address
    in the incident's region, an email address at a reserved `.example` domain.
21. **Length.** Tokens are whitespace-separated words of all turns × 1.33. Each
    call draws a target (wordy agents around 772 words, brief around 570, ±180,
    triangular) and keeps optional parts, in random order, while they fit. The
    corpus mean lands near 640 words, about 850 tokens.
22. **Turn times.** Words / 2.5 per second plus a pause of 0.3–1.2 s before each
    turn; no hold music or long silences.
23. **Ids.** Calls are numbered `call-00001` … in order of start; agents
    `agent-01` … `agent-80`.

## Questions

24. **Wording.** Questions use the canonical service name; changing and
    post-incident questions also name the region; ambiguous questions name only
    the service. The two changing questions of one state use two different
    intents.
25. **Harmful.** Every question about an incident (changing, post-incident and
    ambiguous) lists that incident's wrong statements.
26. **Ambiguous gold.** Grade 3 for the latest state of each incident, grade 1 for
    their earlier states (DESIGN.md section 2's general rule). `thread` is empty,
    because the question is about two threads.
27. **Ordinary and near-ordinary.** Asked between the first state and the end of
    an incident chosen at random. An ordinary question avoids incidents on its
    article's service ("a subject no incident touches"); a near-ordinary question
    uses one on the same service. Near-ordinary questions are 100 sections drawn
    from the 61 articles about a service, in the other of the section's two
    phrasings. `thread` is empty for both.

## Speech-recognition errors

28. **Words and punctuation.** A word is a run of non-whitespace; its leading and
    trailing punctuation are kept, a sound-alike keeps a leading capital, a
    "joined by a filler" word gets the filler before it. A turn never becomes
    empty. Ids, states and turn times are unchanged.
29. **Several rates.** `--asr` may be repeated. Each copy starts from the
    generator's state after the corpus was drawn, so the copy for one rate does
    not depend on which other rates were asked for, and the main files do not
    depend on `--asr` at all. File name: `transcripts.asrNN.jsonl`, NN the rate in
    hundredths.

## Check

30. **Which rules apply.** Generic rules run on any data set (public ones too).
    The counts and the mean length run when the records' source is `synthetic`
    or the manifest was made by `cb-data synth`. The length classes and the
    planted pairs cannot be read from the records, so `check` reads them from the
    manifest's counts, which the generator writes.

## Tension in the specification

31. "About one call in five" before the cause, with all 1,250 wrong statements
    said at least once, needs at least 6,250 calls (62.5% of 10,000) made before
    the agent knows the cause. That forces strong front-loading and leaves fewer
    calls for the later states. The call model (16) keeps the share near 0.2 while
    keeping the fix-in-progress state reasonably covered; the narrow root-cause
    and workaround intervals and the few calls after resolution get fewer
    statements, which bounds how many true states two independent calls can ever
    confirm.
