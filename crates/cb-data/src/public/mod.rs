//! Loaders for public incident timelines (docs/PUBLIC_DATA.md).
//!
//! Each source has a fetch step that downloads into `--raw` and a convert step
//! that reads only from `--raw`, so a conversion can be repeated offline. All
//! three sources are read into one shape ([`timeline::Incident`]) and turned
//! into records and questions by one set of rules ([`timeline::build`]).

pub mod csv;
pub mod fetch;
pub mod gcp;
pub mod github;
pub mod llm;
pub mod text;
pub mod time;
pub mod timeline;
pub mod wording;
pub mod zenodo;

use crate::files::{self, DataSet, Provenance};
use anyhow::Context;
use cb_types::{Query, Record};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// `cb-data public` arguments.
#[derive(clap::Args, Debug)]
pub struct Args {
    #[command(subcommand)]
    pub source: Source,
}

#[derive(clap::Subcommand, Debug)]
pub enum Source {
    /// GitHub's status history, with GitHub's documentation as standing knowledge.
    Github(SourceArgs),
    /// Google Cloud incident reports (Zenodo 10.5281/zenodo.14010282).
    Gcp(SourceArgs),
    /// LLM service outages (Zenodo 10.5281/zenodo.14018219).
    Llm(SourceArgs),
}

#[derive(clap::Args, Debug)]
pub struct SourceArgs {
    /// Directory of the downloaded source files.
    #[arg(long)]
    pub raw: PathBuf,
    /// Directory the data set is written to.
    #[arg(long)]
    pub out: PathBuf,
    /// Download what is missing from --raw before converting.
    #[arg(long)]
    pub fetch: bool,
    /// Seed of the generator that draws each question's moment and wording.
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// A converted data set, before it is written.
pub struct Converted<S> {
    pub name: &'static str,
    pub records: Vec<Record>,
    pub queries: Vec<Query>,
    /// Where the material came from and under what terms, for the manifest.
    pub sources: Vec<String>,
    /// What the conversion did; written beside the data set as [`STATS`].
    pub stats: S,
}

/// The file that records what a conversion did. The manifest holds its hash.
pub const STATS: &str = "stats.json";

fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?)
}

fn fetch_github(raw: &Path) -> anyhow::Result<()> {
    let report = runtime()?.block_on(async {
        let mut fetcher =
            fetch::Fetcher::new(fetch::Politeness::default(), github::download::allowed_urls())?;
        github::download::download(raw, github::SINCE, &mut fetcher, &mut |line| eprintln!("{line}")).await
    })?;
    eprintln!(
        "fetched: {} history pages, {} incidents ({} new), {} articles ({} new), {} gone",
        report.history_pages,
        report.incidents_listed,
        report.incidents_downloaded,
        report.articles_listed,
        report.articles_downloaded,
        report.gone.len()
    );
    for url in &report.gone {
        eprintln!("gone: {url}");
    }
    Ok(())
}

fn fetch_zenodo(file: &zenodo::ZenodoFile, raw: &Path) -> anyhow::Result<()> {
    let outcome = runtime()?.block_on(async {
        let mut fetcher = fetch::Fetcher::new(zenodo::politeness(), file.allowed_urls())?;
        file.download(raw, &mut fetcher).await
    })?;
    eprintln!(
        "{}: {outcome:?}, checked against SHA-256 {}",
        file.path(raw).display(),
        file.sha256
    );
    Ok(())
}

/// Write a converted data set to `out`, with its stats and manifest.
fn write<S: Serialize>(out: &Path, seed: u64, converted: Converted<S>) -> anyhow::Result<()> {
    std::fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let mut stats = serde_json::to_vec_pretty(&converted.stats)?;
    stats.push(b'\n');
    std::fs::write(out.join(STATS), stats)
        .with_context(|| format!("writing {STATS} in {}", out.display()))?;
    let provenance = Provenance {
        name: converted.name.to_string(),
        made_by: format!("cb-data {} public {}", env!("CARGO_PKG_VERSION"), converted.name),
        seed: Some(seed),
        sources: converted.sources,
    };
    let data = DataSet {
        records: converted.records,
        queries: converted.queries,
        transcripts: Vec::new(),
    };
    let manifest = files::write_data_set(out, &data, &provenance, &[STATS])
        .with_context(|| format!("writing the data set to {}", out.display()))?;
    eprintln!(
        "{}: {} records, {} questions -> {}",
        manifest.name,
        data.records.len(),
        data.queries.len(),
        out.display()
    );
    Ok(())
}

pub fn run(args: Args) -> anyhow::Result<()> {
    match args.source {
        Source::Github(a) => {
            if a.fetch {
                fetch_github(&a.raw)?;
            }
            let converted =
                github::convert(&a.raw, a.seed).with_context(|| format!("converting {}", a.raw.display()))?;
            write(&a.out, a.seed, converted)
        }
        Source::Gcp(a) => {
            if a.fetch {
                fetch_zenodo(&gcp::ARCHIVE, &a.raw)?;
            }
            let converted =
                gcp::convert(&a.raw, a.seed).with_context(|| format!("converting {}", a.raw.display()))?;
            write(&a.out, a.seed, converted)
        }
        Source::Llm(a) => {
            if a.fetch {
                fetch_zenodo(&llm::ARCHIVE, &a.raw)?;
            }
            let converted =
                llm::convert(&a.raw, a.seed).with_context(|| format!("converting {}", a.raw.display()))?;
            write(&a.out, a.seed, converted)
        }
    }
}
