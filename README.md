# ContextBridge tools

Rust tools for measuring how fresh a voice agent's knowledge is: how quickly a record
taken from a conversation becomes searchable, how a search ranks newer records against
standing documentation, and whether agreement between conversations makes an extracted
record trustworthy. The tools build data sets, drive the services under test, and
compute every measure from the files those runs write.

| crate | what |
|---|---|
| `cb-types` | The file contracts as Rust types: records, questions, transcripts, candidate lists, extractions, timed trips. |
| `cb-data` | Builds data sets: a generated incident corpus ([`docs/SYNTH_SPEC.md`](docs/SYNTH_SPEC.md)) and incident timelines from public status histories ([`docs/PUBLIC_DATA.md`](docs/PUBLIC_DATA.md)). |
| `cb-run` | Drives the services under test and writes what they return: uploads, candidate lists, newest lists, extractions, judgements, timed trips ([`docs/RUN_TOOLS.md`](docs/RUN_TOOLS.md), [`docs/KB_EXPERIMENT.md`](docs/KB_EXPERIMENT.md)). |
| `cb-eval` | Computes measures from run files: ranking methods and their scores, parameter choice on a validation split, reports with bootstrap intervals and paired tests, the two-conversation rule, and timing percentiles over whole trips ([`DESIGN.md`](DESIGN.md)). |

`prompts/` holds the extraction and judge prompts by version.

## Build and test

Rust 1.99 or newer:

```sh
cargo build --release
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

## What is not here

The services the tools drive are parts of a private product. The data sets, run files
and results are not published.

## Licence

Apache License 2.0.
