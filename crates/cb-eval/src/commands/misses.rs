//! `cb-eval misses`: on changing questions of the test split, where a product method
//! that reads the newest list misses the record that was current. A question counts
//! once, in the first of: its first record is a current one; no current record is a
//! candidate of the method (neither among the merged records nor among its `recent`
//! newest); a current record is a candidate but every one scores under the product's
//! cut-off; or one passes the cut-off and another record is put first.
//!
//! The candidates and their scores are the method's own (`product_candidates`), so the
//! split is exact for `product`, `product_decay` and `product_bonus`, whose candidates
//! no rule removes. Written to `<run>/misses/<method>.json`.

use crate::context::QuestionContext;
use crate::error::EvalError;
use crate::files::{
    CANDIDATES, CandidateFile, NEWEST, QUERIES, RECORDS, read_jsonl_file, sha256_of_file, write_json_file,
};
use crate::measures::QuestionKey;
use crate::methods;
use crate::methods::product::{PRODUCT_CUTOFF, PRODUCT_PREFETCH, Pool, product_candidates};
use crate::records::RecordTable;
use crate::spec::MethodSpec;
use cb_types::{Candidates, Newest, Query, QueryClass, Record, Split, split_of};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

pub const MISSES_DIR: &str = "misses";

#[derive(Clone, Debug)]
pub struct MissesArgs {
    pub data: PathBuf,
    pub run: PathBuf,
    pub method: MethodSpec,
}

/// `<run>/misses/<method>.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Misses {
    pub method: String,
    pub text: String,
    /// File name → sha256 of every input.
    pub inputs: BTreeMap<String, String>,
    /// Changing questions of the test split.
    pub questions: usize,
    pub first: usize,
    pub not_a_candidate: usize,
    pub under_the_cut: usize,
    pub outranked: usize,
    /// Questions whose current record was among the newest records the method read.
    pub current_among_newest: usize,
    /// Without the newest list (the product as deployed): questions whose current record
    /// the service listed in the first 64 places by meaning, by keywords, and questions
    /// where it was one of the product's candidates at the method's depth.
    pub current_in_dense_64: usize,
    pub current_in_sparse_64: usize,
    pub current_a_product_candidate: usize,
}

fn which_methods_can(spec: &MethodSpec) -> Result<(Pool, f64), EvalError> {
    match *spec {
        MethodSpec::Product { depth, recent }
        | MethodSpec::ProductDecay { depth, recent, .. }
        | MethodSpec::ProductBonus { depth, recent, .. } => Ok((Pool::new(depth, recent), PRODUCT_CUTOFF)),
        _ => Err(EvalError::invalid(format!(
            "misses: {spec} is not product, product_decay or product_bonus, whose candidates no rule removes"
        ))),
    }
}

pub fn run(args: &MissesArgs) -> Result<Misses, EvalError> {
    let (pool, cut) = which_methods_can(&args.method)?;
    let records_path = args.data.join(RECORDS);
    let queries_path = args.data.join(QUERIES);
    let candidates_path = args.run.join(CANDIDATES);
    let newest_path = args.run.join(NEWEST);
    let mut inputs = BTreeMap::new();
    for (name, path) in [
        (RECORDS, &records_path),
        (QUERIES, &queries_path),
        (CANDIDATES, &candidates_path),
        (NEWEST, &newest_path),
    ] {
        inputs.insert(name.to_string(), sha256_of_file(path)?);
    }
    let records: Vec<Record> = read_jsonl_file(&records_path)?;
    let queries: Vec<Query> = read_jsonl_file(&queries_path)?;
    let table = RecordTable::from_records(&records)?;
    drop(records);
    let newest: HashMap<String, Newest> = read_jsonl_file::<Newest>(&newest_path)?
        .into_iter()
        .map(|line| (line.query.clone(), line))
        .collect();
    let mut candidates = CandidateFile::open(&candidates_path)?;

    let mut out = Misses {
        method: args.method.name(),
        text: args.method.to_string(),
        inputs,
        questions: 0,
        first: 0,
        not_a_candidate: 0,
        under_the_cut: 0,
        outranked: 0,
        current_among_newest: 0,
        current_in_dense_64: 0,
        current_in_sparse_64: 0,
        current_a_product_candidate: 0,
    };
    for query in queries
        .iter()
        .filter(|q| q.class == QueryClass::Changing && split_of(q) == Split::Test)
    {
        let key = QuestionKey::new(query, &table)?;
        let current: Vec<_> = key
            .gold()
            .iter()
            .filter(|g| g.grade == 3)
            .map(|g| g.record)
            .collect();
        let lines = candidates.lines_of(&query.id)?;
        let lines: Vec<&Candidates> = lines.iter().collect();
        let ctx = QuestionContext::new(query, &lines, newest.get(&query.id), &table, None, key.gold())?;
        let ranked = methods::rank(&args.method, &ctx)?;
        let candidates_of_method = product_candidates(&ctx, pool, |_| true)?;
        out.questions += 1;
        let now = ctx.now()?;
        let in_first_64 = |list: &[crate::context::Listed]| {
            list.iter()
                .any(|l| l.served < PRODUCT_PREFETCH && current.contains(&l.record))
        };
        out.current_in_dense_64 += usize::from(in_first_64(&now.dense));
        out.current_in_sparse_64 += usize::from(in_first_64(&now.sparse));
        let deployed = Pool { recent: 0, ..pool };
        if product_candidates(&ctx, deployed, |_| true)?
            .iter()
            .any(|(record, _)| current.contains(record))
        {
            out.current_a_product_candidate += 1;
        }
        if pool.recent > 0
            && ctx
                .newest()?
                .iter()
                .filter(|l| l.served < pool.recent)
                .any(|l| current.contains(&l.record))
        {
            out.current_among_newest += 1;
        }
        let scored: Vec<f64> = candidates_of_method
            .iter()
            .filter(|(record, _)| current.contains(record))
            .map(|(_, rho)| *rho)
            .collect();
        if ranked.first().is_some_and(|r| current.contains(&r.record)) {
            out.first += 1;
        } else if scored.is_empty() {
            out.not_a_candidate += 1;
        } else if scored.iter().all(|rho| *rho < cut) {
            out.under_the_cut += 1;
        } else {
            out.outranked += 1;
        }
    }
    write_json_file(
        &args.run.join(MISSES_DIR).join(format!("{}.json", out.method)),
        &out,
    )?;
    Ok(out)
}
