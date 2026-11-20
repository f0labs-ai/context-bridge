//! Seeded statistics: bootstrap intervals over threads, a paired randomization
//! test over threads, and the one percentile rule.
//!
//! Questions about one incident are not independent, so the unit that is
//! resampled, and the unit whose labels are swapped, is the cluster: a thread,
//! or a single question that has none. A [`Stratum`] is one run; with several
//! runs (the five seeds of the generated corpus) the statistic is the plain
//! mean of the runs' means and each run is resampled on its own.
//!
//! Every random draw comes from ChaCha20 keyed by the seed and a label naming
//! its purpose, its group of questions and its run
//! (`bootstrap/<group>/<run>`, `randomization/<group>/<run>`), so a result
//! does not depend on which other methods, measures or runs were computed.

use crate::error::EvalError;
use cb_types::fnv1a64;
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use serde::{Deserialize, Serialize};

/// The seed used when none is given: the date the pass lines were written.
pub const DEFAULT_SEED: u64 = 20_261_119;
/// Bootstrap resamples and permutations, as DESIGN.md section 5 fixes them.
pub const DEFAULT_DRAWS: usize = 10_000;

/// The ChaCha20 stream for one purpose. The 256-bit key is the seed (8 bytes,
/// little-endian), then FNV-1a 64 of the label (8 bytes, little-endian), then zeros.
pub fn stream(seed: u64, label: &str) -> ChaCha20Rng {
    let mut key = [0u8; 32];
    key[..8].copy_from_slice(&seed.to_le_bytes());
    key[8..16].copy_from_slice(&fnv1a64(label.as_bytes()).to_le_bytes());
    ChaCha20Rng::from_seed(key)
}

/// A whole number from 0 to `n - 1`, each equally likely (Lemire's
/// multiply-and-reject method on 64-bit draws). `n` must be at least 1.
pub fn uniform_below(rng: &mut impl RngCore, n: u64) -> u64 {
    let mut product = u128::from(rng.next_u64()) * u128::from(n);
    if (product as u64) < n {
        let threshold = n.wrapping_neg() % n;
        while (product as u64) < threshold {
            product = u128::from(rng.next_u64()) * u128::from(n);
        }
    }
    (product >> 64) as u64
}

/// The 1-based rank of the nearest-rank percentile: `ceil(per_mille * n / 1000)`,
/// at least 1 and at most `n`, in whole numbers so no rounding can move it.
fn nearest_rank_position(per_mille: u32, n: usize) -> usize {
    let n = n as u128;
    (u128::from(per_mille) * n).div_ceil(1000).clamp(1, n) as usize
}

/// The percentile by nearest rank: the smallest value such that at least
/// `per_mille / 1000` of the values are at or below it. `sorted` must be in
/// ascending order. `None` for no values.
pub fn nearest_rank(sorted: &[f64], per_mille: u32) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    sorted
        .get(nearest_rank_position(per_mille, sorted.len()) - 1)
        .copied()
}

/// [`nearest_rank`] of values in any order. Reorders `values`.
pub fn nearest_rank_unsorted(values: &mut [f64], per_mille: u32) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let position = nearest_rank_position(per_mille, values.len()) - 1;
    let (_, value, _) = values.select_nth_unstable_by(position, f64::total_cmp);
    Some(*value)
}

/// The median: the middle value, or the mean of the two middle values.
/// `sorted` must be in ascending order. `None` for no values.
pub fn median(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        None
    } else if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    }
}

/// Sort ascending. Values must be finite.
pub fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    values
}

/// A 95% interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

/// The percentile interval of a set of resampled statistics: its 2.5th and
/// 97.5th percentiles by nearest rank (of 10 000, the 250th and the 9 750th
/// smallest). `None` for no statistics.
pub fn percentile_interval(statistics: &[f64]) -> Option<Interval> {
    let mut values = statistics.to_vec();
    Some(Interval {
        lo: nearest_rank_unsorted(&mut values, 25)?,
        hi: nearest_rank_unsorted(&mut values, 975)?,
    })
}

/// One run's clusters within a group of questions.
#[derive(Clone, Debug, PartialEq)]
pub struct Stratum {
    /// The run's name. It keys the run's random stream, so a run's draws do
    /// not change when other runs are added to an analysis.
    pub label: String,
    /// How many questions each cluster holds, in a fixed cluster order.
    pub questions_per_cluster: Vec<u32>,
}

fn check_labels(what: &str, strata: &[Stratum]) -> Result<(), EvalError> {
    for (i, stratum) in strata.iter().enumerate() {
        if strata[..i].iter().any(|earlier| earlier.label == stratum.label) {
            return Err(EvalError::invalid(format!(
                "{what}: two runs are both called {:?}",
                stratum.label
            )));
        }
        if stratum.questions_per_cluster.is_empty()
            || stratum.questions_per_cluster.len() > usize::from(u16::MAX)
            || stratum.questions_per_cluster.contains(&0)
        {
            return Err(EvalError::invalid(format!(
                "{what}: run {:?} needs between 1 and 65 535 clusters, each with a question; it has {}",
                stratum.label,
                stratum.questions_per_cluster.len()
            )));
        }
    }
    if strata.is_empty() {
        return Err(EvalError::invalid(format!("{what}: no run was given")));
    }
    Ok(())
}

fn check_shape<T>(what: &str, clusters: &[usize], given: &[Vec<T>]) -> Result<(), EvalError> {
    let shapes: Vec<usize> = given.iter().map(Vec::len).collect();
    if shapes == clusters {
        Ok(())
    } else {
        Err(EvalError::invalid(format!(
            "{what}: values for clusters {shapes:?} were given, the draws are for {clusters:?}"
        )))
    }
}

/// Bootstrap resamples of clusters, drawn once and shared by every method and
/// measure of a group so that their intervals and differences use the same resamples.
#[derive(Debug)]
pub struct Resamples {
    count: usize,
    strata: Vec<ResampledStratum>,
}

#[derive(Debug)]
struct ResampledStratum {
    clusters: usize,
    /// For each resample, how often each cluster was drawn (`count * clusters` values).
    times_drawn: Vec<u16>,
    /// For each resample, how many questions it holds.
    questions: Vec<f64>,
}

impl Resamples {
    /// Draw `count` resamples. A resample of a stratum takes as many clusters
    /// as the stratum has, uniformly with replacement, from the stream
    /// `bootstrap/<label>/<stratum label>`.
    pub fn draw(seed: u64, label: &str, strata: &[Stratum], count: usize) -> Result<Self, EvalError> {
        check_labels(&format!("bootstrap {label}"), strata)?;
        let mut drawn = Vec::with_capacity(strata.len());
        for stratum in strata {
            let sizes = &stratum.questions_per_cluster;
            let clusters = sizes.len();
            let mut rng = stream(seed, &format!("bootstrap/{label}/{}", stratum.label));
            let mut times_drawn = vec![0u16; count * clusters];
            let mut questions = Vec::with_capacity(count);
            for resample in times_drawn.chunks_exact_mut(clusters) {
                for _ in 0..clusters {
                    resample[uniform_below(&mut rng, clusters as u64) as usize] += 1;
                }
                questions.push(
                    resample
                        .iter()
                        .zip(sizes)
                        .map(|(times, size)| f64::from(*times) * f64::from(*size))
                        .sum(),
                );
            }
            drawn.push(ResampledStratum {
                clusters,
                times_drawn,
                questions,
            });
        }
        Ok(Self { count, strata: drawn })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    fn clusters(&self) -> Vec<usize> {
        self.strata.iter().map(|s| s.clusters).collect()
    }

    /// The statistic of every resample: per stratum, the sum of the drawn
    /// clusters' totals over the number of questions drawn (the mean over
    /// questions); then the plain mean over strata.
    /// `totals_per_cluster[s][c]` is a measure summed over the questions of
    /// cluster `c` of stratum `s`.
    pub fn means(&self, totals_per_cluster: &[Vec<f64>]) -> Result<Vec<f64>, EvalError> {
        check_shape("bootstrap", &self.clusters(), totals_per_cluster)?;
        let mut statistics = vec![0.0; self.count];
        for (stratum, totals) in self.strata.iter().zip(totals_per_cluster) {
            let resamples = stratum.times_drawn.chunks_exact(stratum.clusters);
            for ((statistic, resample), questions) in
                statistics.iter_mut().zip(resamples).zip(&stratum.questions)
            {
                let total: f64 = resample
                    .iter()
                    .zip(totals)
                    .map(|(times, total)| f64::from(*times) * total)
                    .sum();
                *statistic += total / questions;
            }
        }
        let strata = self.strata.len() as f64;
        for statistic in &mut statistics {
            *statistic /= strata;
        }
        Ok(statistics)
    }
}

/// The mean over questions per stratum, then the plain mean over strata: the
/// statistic every interval and test here is about.
pub fn mean_over_strata(totals_per_cluster: &[Vec<f64>], questions_per_cluster: &[Vec<u32>]) -> f64 {
    let per_stratum: f64 = totals_per_cluster
        .iter()
        .zip(questions_per_cluster)
        .map(|(totals, sizes)| totals.iter().sum::<f64>() / sizes.iter().map(|s| f64::from(*s)).sum::<f64>())
        .sum();
    per_stratum / totals_per_cluster.len() as f64
}

/// Random swaps of the two methods' labels, cluster by cluster, drawn once
/// and shared by every method and measure of a group.
#[derive(Debug)]
pub struct LabelSwaps {
    count: usize,
    strata: Vec<SwappedStratum>,
}

#[derive(Debug)]
struct SwappedStratum {
    clusters: usize,
    questions: f64,
    /// For each permutation, one bit per cluster (set = swapped), in 64-bit
    /// words; bits past the last cluster are never read.
    swapped: Vec<u64>,
}

impl SwappedStratum {
    fn words(&self) -> usize {
        self.clusters.div_ceil(64)
    }
}

/// Whether a permuted statistic is at least as far from zero as the observed
/// one. Equal up to rounding counts as "at least as far".
fn at_least_as_far(permuted: f64, observed: f64) -> bool {
    permuted.abs() >= observed.abs() * (1.0 - 1e-12)
}

/// `p = (1 + b) / (1 + n)`, where `b` of `n` permutations were at least as far from zero.
fn p_from_count(at_least_as_far: usize, permutations: usize) -> f64 {
    (1 + at_least_as_far) as f64 / (1 + permutations) as f64
}

impl LabelSwaps {
    /// Draw `count` permutations: each cluster's labels are swapped with
    /// probability one half, from the stream `randomization/<label>/<stratum label>`.
    pub fn draw(seed: u64, label: &str, strata: &[Stratum], count: usize) -> Result<Self, EvalError> {
        check_labels(&format!("randomization {label}"), strata)?;
        Ok(Self {
            count,
            strata: strata
                .iter()
                .map(|stratum| {
                    let mut rng = stream(seed, &format!("randomization/{label}/{}", stratum.label));
                    let words = stratum.questions_per_cluster.len().div_ceil(64);
                    SwappedStratum {
                        clusters: stratum.questions_per_cluster.len(),
                        questions: stratum.questions_per_cluster.iter().map(|s| f64::from(*s)).sum(),
                        swapped: (0..count * words).map(|_| rng.next_u64()).collect(),
                    }
                })
                .collect(),
        })
    }

    pub fn count(&self) -> usize {
        self.count
    }

    fn clusters(&self) -> Vec<usize> {
        self.strata.iter().map(|s| s.clusters).collect()
    }

    /// The two-sided p-value of the paired randomization test, computed
    /// directly from the differences (the reference for [`two_sided_p_of_difference`]).
    ///
    /// `differences_per_cluster[s][c]` is the sum over the questions of
    /// cluster `c` of (method − baseline). The statistic is the mean
    /// difference over questions per stratum, then the mean over strata.
    /// Swapping a cluster's labels flips the sign of its difference.
    /// `p = (1 + b) / (1 + count)`, where `b` permutations gave a statistic at
    /// least as far from zero as the observed one; the observed labelling
    /// counts as one more, so `p` is never 0.
    pub fn two_sided_p(&self, differences_per_cluster: &[Vec<f64>]) -> Result<f64, EvalError> {
        check_shape("randomization", &self.clusters(), differences_per_cluster)?;
        let strata = self.strata.len() as f64;
        let observed: f64 = self
            .strata
            .iter()
            .zip(differences_per_cluster)
            .map(|(stratum, differences)| differences.iter().sum::<f64>() / stratum.questions)
            .sum::<f64>()
            / strata;
        let mut permuted = vec![0.0; self.count];
        for (stratum, differences) in self.strata.iter().zip(differences_per_cluster) {
            for (statistic, bits) in permuted
                .iter_mut()
                .zip(stratum.swapped.chunks_exact(stratum.words()))
            {
                let total: f64 = differences
                    .iter()
                    .enumerate()
                    .map(|(c, d)| {
                        if (bits[c / 64] >> (c % 64)) & 1 == 1 {
                            -d
                        } else {
                            *d
                        }
                    })
                    .sum();
                *statistic += total / stratum.questions;
            }
        }
        let count = permuted
            .iter()
            .filter(|t| at_least_as_far(**t / strata, observed))
            .count();
        Ok(p_from_count(count, self.count))
    }

    /// For each permutation, the statistic of one method's totals with every
    /// swapped cluster's sign flipped: per stratum
    /// `(sum of totals - 2 * sum of the swapped clusters' totals) / questions`,
    /// then the mean over strata.
    ///
    /// The statistic is linear in the totals, so the permuted statistic of a
    /// difference (method − baseline) is the method's minus the baseline's;
    /// see [`two_sided_p_of_difference`].
    pub fn signed_means(&self, totals_per_cluster: &[Vec<f64>]) -> Result<Vec<f64>, EvalError> {
        check_shape("randomization", &self.clusters(), totals_per_cluster)?;
        let mut statistics = vec![0.0; self.count];
        for (stratum, totals) in self.strata.iter().zip(totals_per_cluster) {
            let all: f64 = totals.iter().sum();
            let words = stratum.words();
            let last_word_mask = match stratum.clusters % 64 {
                0 => u64::MAX,
                used => (1u64 << used) - 1,
            };
            for (statistic, bits) in statistics.iter_mut().zip(stratum.swapped.chunks_exact(words)) {
                let mut swapped = 0.0;
                for (word, chunk) in bits.iter().zip(totals.chunks(64)) {
                    let mut set = if chunk.len() == 64 {
                        *word
                    } else {
                        *word & last_word_mask
                    };
                    while set != 0 {
                        swapped += chunk[set.trailing_zeros() as usize];
                        set &= set - 1;
                    }
                }
                *statistic += (all - 2.0 * swapped) / stratum.questions;
            }
        }
        let strata = self.strata.len() as f64;
        for statistic in &mut statistics {
            *statistic /= strata;
        }
        Ok(statistics)
    }
}

/// The two-sided p-value of a method against a baseline from their
/// [`LabelSwaps::signed_means`] under the same swaps and the observed
/// difference of their statistics (method − baseline).
pub fn two_sided_p_of_difference(observed: f64, permuted_method: &[f64], permuted_baseline: &[f64]) -> f64 {
    let count = permuted_method
        .iter()
        .zip(permuted_baseline)
        .filter(|(m, b)| at_least_as_far(*m - *b, observed))
        .count();
    p_from_count(count, permuted_method.len().min(permuted_baseline.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strata named run-0, run-1, … with the given cluster sizes.
    fn strata(sizes: &[&[u32]]) -> Vec<Stratum> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, sizes)| Stratum {
                label: format!("run-{i}"),
                questions_per_cluster: sizes.to_vec(),
            })
            .collect()
    }

    #[test]
    fn nearest_rank_is_the_smallest_value_with_enough_values_at_or_below_it() {
        let ten: Vec<f64> = (1..=10).map(f64::from).collect();
        // rank = ceil(per_mille * 10 / 1000)
        assert_eq!(nearest_rank(&ten, 500), Some(5.0)); // ceil(5.0) = 5
        assert_eq!(nearest_rank(&ten, 900), Some(9.0)); // ceil(9.0) = 9
        assert_eq!(nearest_rank(&ten, 950), Some(10.0)); // ceil(9.5) = 10
        assert_eq!(nearest_rank(&ten, 25), Some(1.0)); // ceil(0.25) = 1
        assert_eq!(nearest_rank(&ten, 101), Some(2.0)); // ceil(1.01) = 2
        assert_eq!(nearest_rank(&ten, 1000), Some(10.0));
        assert_eq!(nearest_rank(&ten, 0), Some(1.0));
        assert_eq!(nearest_rank(&[], 500), None);
        // 1 000 trips: the 95th percentile is the 950th smallest, the 99th the 990th.
        let thousand: Vec<f64> = (1..=1000).map(f64::from).collect();
        assert_eq!(nearest_rank(&thousand, 950), Some(950.0));
        assert_eq!(nearest_rank(&thousand, 990), Some(990.0));
        // 0.07 * 100 is 7.000000000000001 in floating point; whole numbers keep the rank at 7.
        let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(nearest_rank(&hundred, 70), Some(7.0));
    }

    #[test]
    fn nearest_rank_without_sorting_finds_the_same_value() {
        let mut rng = stream(3, "shuffle");
        let values: Vec<f64> = (0..1_001)
            .map(|_| uniform_below(&mut rng, 50) as f64 / 7.0)
            .collect();
        let ordered = sorted(values.clone());
        for per_mille in [0, 1, 25, 333, 500, 950, 975, 999, 1000] {
            let mut scratch = values.clone();
            assert_eq!(
                nearest_rank_unsorted(&mut scratch, per_mille),
                nearest_rank(&ordered, per_mille),
                "{per_mille}"
            );
        }
        assert_eq!(nearest_rank_unsorted(&mut [], 500), None);
    }

    #[test]
    fn the_median_is_the_middle_or_the_mean_of_the_two_middles() {
        assert_eq!(median(&[1.0, 2.0, 9.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 4.0, 9.0]), Some(3.0));
        assert_eq!(median(&[5.0]), Some(5.0));
        assert_eq!(median(&[]), None);
        assert_eq!(sorted(vec![3.0, -1.0, 2.0]), [-1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_interval_is_the_250th_and_9750th_of_ten_thousand() {
        let statistics: Vec<f64> = (1..=10_000).rev().map(f64::from).collect();
        // ceil(0.025 * 10 000) = 250 and ceil(0.975 * 10 000) = 9 750
        assert_eq!(
            percentile_interval(&statistics),
            Some(Interval {
                lo: 250.0,
                hi: 9_750.0
            })
        );
        assert_eq!(percentile_interval(&[]), None);
    }

    #[test]
    fn a_stream_depends_on_the_seed_and_the_label_and_nothing_else() {
        let first = |seed, label| stream(seed, label).next_u64();
        assert_eq!(first(7, "a"), first(7, "a"));
        assert_ne!(first(7, "a"), first(8, "a"));
        assert_ne!(first(7, "a"), first(7, "b"));
        // Fixed values: a change in the generator or in how the key is built would move every
        // reported interval and p-value, and must show up here first.
        let mut rng = stream(DEFAULT_SEED, "bootstrap/changing/0");
        let draws: Vec<u64> = (0..4).map(|_| uniform_below(&mut rng, 1000)).collect();
        assert_eq!(draws, PINNED_DRAWS);
    }

    /// The first four draws below 1000 of the stream "bootstrap/changing/0" under the default seed.
    const PINNED_DRAWS: [u64; 4] = [178, 125, 536, 925];

    #[test]
    fn the_generator_is_the_published_chacha20() {
        // The ChaCha20 keystream for an all-zero key and nonce begins
        // 76 b8 e0 ad a0 f1 3d 90 40 5d 6a e5 53 86 bd 28 (the standard test vector).
        // Read as little-endian 64-bit words: 0x903df1a0ade0b876, 0x28bd8653e56a5d40.
        let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
        assert_eq!(rng.next_u64(), 0x903d_f1a0_ade0_b876);
        assert_eq!(rng.next_u64(), 0x28bd_8653_e56a_5d40);
    }

    #[test]
    fn uniform_below_stays_in_range_and_is_even() {
        let mut rng = stream(1, "even");
        assert!((0..100).all(|_| uniform_below(&mut rng, 1) == 0));
        let mut seen = [0u32; 5];
        for _ in 0..50_000 {
            seen[uniform_below(&mut rng, 5) as usize] += 1;
        }
        // Each of the five values is expected 10 000 times; the standard deviation is 89.
        assert!(
            seen.iter().all(|count| (9_500..=10_500).contains(count)),
            "{seen:?}"
        );
        // A bound just under 2^64 exercises the rejection branch.
        assert!((0..100).all(|_| uniform_below(&mut rng, u64::MAX - 1) < u64::MAX - 1));
    }

    #[test]
    fn a_resample_is_drawn_from_the_stream_named_after_the_group_and_the_run() {
        // Three clusters of one question each with totals 1, 10 and 100: a resample's statistic
        // tells exactly which clusters it drew. The first resample takes three draws below 3
        // from "bootstrap/changing/synth-42".
        let one_run = vec![Stratum {
            label: "synth-42".to_string(),
            questions_per_cluster: vec![1, 1, 1],
        }];
        let resamples = Resamples::draw(DEFAULT_SEED, "changing", &one_run, 2).unwrap();
        let statistics = resamples.means(&[vec![1.0, 10.0, 100.0]]).unwrap();
        let mut rng = stream(DEFAULT_SEED, "bootstrap/changing/synth-42");
        let value = [1.0, 10.0, 100.0];
        let first: f64 = (0..3)
            .map(|_| value[uniform_below(&mut rng, 3) as usize])
            .sum::<f64>()
            / 3.0;
        assert_eq!(statistics[0], first);
    }

    #[test]
    fn the_bootstrap_is_the_same_under_one_seed_and_differs_under_another() {
        let sizes = strata(&[&[2, 1, 3, 1, 1]]);
        let totals = vec![vec![1.0, 0.0, 2.0, 1.0, 0.0]];
        let run = |seed| {
            Resamples::draw(seed, "changing", &sizes, 2_000)
                .unwrap()
                .means(&totals)
                .unwrap()
        };
        assert_eq!(run(11), run(11));
        assert_ne!(run(11), run(12));
        let other_label = Resamples::draw(11, "ordinary", &sizes, 2_000)
            .unwrap()
            .means(&totals)
            .unwrap();
        assert_ne!(run(11), other_label);
    }

    #[test]
    fn a_runs_resamples_do_not_change_when_another_run_is_added() {
        let alone = strata(&[&[1, 1, 1, 1]]);
        let mut with_another = alone.clone();
        with_another.push(Stratum {
            label: "other".to_string(),
            questions_per_cluster: vec![1, 1],
        });
        let totals = vec![vec![0.0, 1.0, 2.0, 3.0]];
        let first = Resamples::draw(4, "g", &alone, 500)
            .unwrap()
            .means(&totals)
            .unwrap();
        // The second run contributes a constant 5 (both clusters total 5): the mean over the two
        // runs is (first + 5) / 2 resample by resample.
        let mut both_totals = totals.clone();
        both_totals.push(vec![5.0, 5.0]);
        let both = Resamples::draw(4, "g", &with_another, 500)
            .unwrap()
            .means(&both_totals)
            .unwrap();
        for (alone, both) in first.iter().zip(&both) {
            assert!((both - (alone + 5.0) / 2.0).abs() < 1e-12);
        }
        // Two runs with one name would share a stream; they are refused.
        let twice = vec![alone[0].clone(), alone[0].clone()];
        assert!(Resamples::draw(4, "g", &twice, 10).is_err());
        assert!(LabelSwaps::draw(4, "g", &twice, 10).is_err());
        assert!(Resamples::draw(4, "g", &[], 10).is_err());
    }

    #[test]
    fn a_resample_draws_whole_threads_and_averages_over_their_questions() {
        // Thread A has three questions, all scoring 1 (total 3). Thread B has one question scoring 0.
        // A resample is two threads: A,A → 6/6 = 1; A,B → 3/4 = 0.75; B,B → 0/2 = 0.
        // Resampling single questions would also give 0.25 and 0.5; they must never appear.
        let resamples = Resamples::draw(3, "threads", &strata(&[&[3, 1]]), 4_000).unwrap();
        let statistics = resamples.means(&[vec![3.0, 0.0]]).unwrap();
        assert_eq!(statistics.len(), 4_000);
        let count = |value: f64| statistics.iter().filter(|s| **s == value).count();
        assert_eq!(count(1.0) + count(0.75) + count(0.0), 4_000);
        // A,B in either order is half of all resamples; A,A and B,B a quarter each.
        assert!((1_800..=2_200).contains(&count(0.75)), "{}", count(0.75));
        assert!((850..=1_150).contains(&count(1.0)), "{}", count(1.0));
        assert!((850..=1_150).contains(&count(0.0)), "{}", count(0.0));
    }

    #[test]
    fn equal_threads_give_an_interval_of_no_width_and_spread_gives_one_around_the_mean() {
        let sizes = vec![vec![2; 30]];
        let flat = Resamples::draw(5, "flat", &strata(&[&[2; 30]]), 2_000).unwrap();
        // Every thread totals 1 over 2 questions: every resample's mean is 0.5.
        let interval = percentile_interval(&flat.means(&[vec![1.0; 30]]).unwrap()).unwrap();
        assert_eq!((interval.lo, interval.hi), (0.5, 0.5));
        // Half the threads total 2, half 0: the mean is 0.5 and the interval lies around it.
        let mixed: Vec<f64> = (0..30).map(|c| if c % 2 == 0 { 2.0 } else { 0.0 }).collect();
        let totals = vec![mixed];
        assert_eq!(mean_over_strata(&totals, &sizes), 0.5);
        let interval = percentile_interval(&flat.means(&totals).unwrap()).unwrap();
        // The standard error of the mean of 30 values of 0 or 1 is sqrt(0.25/30) = 0.091;
        // 1.96 of those is 0.18, so the ends lie near 0.32 and 0.68.
        assert!((0.25..0.40).contains(&interval.lo), "{interval:?}");
        assert!((0.60..0.75).contains(&interval.hi), "{interval:?}");
    }

    #[test]
    fn several_runs_are_resampled_apart_and_their_means_averaged() {
        // Run 1: one thread, mean 1. Run 2: one thread, mean 0. Each resample keeps both means.
        let sizes = vec![vec![4], vec![2]];
        let totals = vec![vec![4.0], vec![0.0]];
        assert_eq!(mean_over_strata(&totals, &sizes), 0.5);
        let resamples = Resamples::draw(9, "runs", &strata(&[&[4], &[2]]), 100).unwrap();
        assert!(resamples.means(&totals).unwrap().iter().all(|s| *s == 0.5));
        // Pooling the questions instead would give 4/6.
        assert!(resamples.means(&[vec![4.0]]).is_err());
        assert!(Resamples::draw(9, "runs", &strata(&[&[]]), 10).is_err());
        assert!(Resamples::draw(9, "runs", &strata(&[&[1, 0]]), 10).is_err());
    }

    fn p_value(seed: u64, differences: Vec<f64>, sizes: Vec<u32>) -> f64 {
        LabelSwaps::draw(seed, "test", &strata(&[&sizes]), DEFAULT_DRAWS)
            .unwrap()
            .two_sided_p(&[differences])
            .unwrap()
    }

    #[test]
    fn the_randomization_test_is_the_same_under_one_seed() {
        let differences = vec![0.5, -0.25, 1.0, 0.0, 0.75, -0.5];
        let sizes = vec![2, 1, 3, 1, 2, 1];
        assert_eq!(
            p_value(4, differences.clone(), sizes.clone()),
            p_value(4, differences.clone(), sizes.clone())
        );
        assert_ne!(
            p_value(4, differences.clone(), sizes.clone()),
            p_value(5, differences, sizes)
        );
    }

    #[test]
    fn the_p_value_matches_cases_counted_by_hand() {
        // One cluster: either labelling gives the same distance from zero. p = 1.
        assert_eq!(p_value(1, vec![2.0], vec![3]), 1.0);
        // No difference anywhere: every permutation ties with the observed zero. p = 1.
        assert_eq!(p_value(1, vec![0.0, 0.0, 0.0], vec![1, 1, 1]), 1.0);
        // Two clusters, +1 each. The four labellings give 2, 0, 0, -2: half are as far as the observed 2.
        let half = p_value(1, vec![1.0, 1.0], vec![1, 1]);
        assert!((half - 0.5).abs() < 0.02, "{half}");
        // Three clusters +3, +1, +1 (sum 5). The eight labellings give ±5, ±3, ±3, ±1: only +5 and -5
        // reach 5, a quarter of them.
        let quarter = p_value(1, vec![3.0, 1.0, 1.0], vec![1, 1, 1]);
        assert!((quarter - 0.25).abs() < 0.02, "{quarter}");
        // Twenty clusters, +1 each: only all-kept and all-swapped reach 20, two in 2^20.
        // With 10 000 permutations none is expected: p = (1 + 0) / (1 + 10 000).
        assert_eq!(p_value(1, vec![1.0; 20], vec![1; 20]), 1.0 / 10_001.0);
        // The sign of the difference does not matter to a two-sided test.
        assert_eq!(p_value(1, vec![-3.0, -1.0, -1.0], vec![1, 1, 1]), quarter);
    }

    #[test]
    fn labels_are_swapped_by_thread_not_by_question() {
        // One thread of 100 questions, each +0.01 (total +1), against nothing else: whichever way
        // the thread is labelled the distance is the same, so p = 1. Swapping question by
        // question would make a total of +1 from 100 small differences very unlikely.
        assert_eq!(p_value(2, vec![1.0], vec![100]), 1.0);
        // More than 64 clusters use more than one word of swap bits.
        let many = p_value(2, vec![1.0; 100], vec![1; 100]);
        assert_eq!(many, 1.0 / 10_001.0);
        let swaps = LabelSwaps::draw(2, "shape", &strata(&[&[1, 1]]), 10).unwrap();
        assert!(swaps.two_sided_p(&[vec![1.0]]).is_err());
        assert!(swaps.signed_means(&[vec![1.0]]).is_err());
        assert!(LabelSwaps::draw(2, "shape", &strata(&[&[]]), 10).is_err());
    }

    #[test]
    fn with_several_runs_the_statistic_is_the_mean_of_the_runs_mean_differences() {
        // Run 1: one thread, difference +2 over 2 questions (mean +1). Run 2: one thread, -1 over
        // 1 question (mean -1). Observed: (1 + -1)/2 = 0; nothing can be closer to zero. p = 1.
        let swaps = LabelSwaps::draw(6, "runs", &strata(&[&[2], &[1]]), 1_000).unwrap();
        assert_eq!(swaps.two_sided_p(&[vec![2.0], vec![-1.0]]).unwrap(), 1.0);
        // Run 2 at +1 over 1 question: observed (1 + 1)/2 = 1. Labellings give 1, 0, 0, -1: half.
        let half = swaps.two_sided_p(&[vec![2.0], vec![1.0]]).unwrap();
        assert!((half - 0.5).abs() < 0.05, "{half}");
    }

    #[test]
    fn signed_means_flip_the_swapped_clusters() {
        // Two clusters of 1 and 3 questions with totals 1 and 3 (4 questions). Each permutation
        // gives one of (1 + 3)/4 = 1, (-1 + 3)/4 = 0.5, (1 - 3)/4 = -0.5 or (-1 - 3)/4 = -1.
        let swaps = LabelSwaps::draw(8, "signs", &strata(&[&[1, 3]]), 4_000).unwrap();
        let means = swaps.signed_means(&[vec![1.0, 3.0]]).unwrap();
        for value in [1.0, 0.5, -0.5, -1.0] {
            let seen = means.iter().filter(|m| **m == value).count();
            assert!((850..=1_150).contains(&seen), "{value}: {seen}");
        }
        assert_eq!(means.len(), 4_000);
    }

    #[test]
    fn the_p_value_from_signed_means_equals_the_reference() {
        // Random differences over two runs, one with more than 64 clusters and a last word that
        // is only partly used (130 = 2 * 64 + 2).
        let mut rng = stream(10, "values");
        let sizes: [Vec<u32>; 2] = [
            (0..130).map(|_| 1 + uniform_below(&mut rng, 4) as u32).collect(),
            (0..7).map(|_| 1 + uniform_below(&mut rng, 4) as u32).collect(),
        ];
        let runs = strata(&[&sizes[0], &sizes[1]]);
        let swaps = LabelSwaps::draw(10, "equal", &runs, 3_000).unwrap();
        let mut random_totals = |shift: f64| -> Vec<Vec<f64>> {
            sizes
                .iter()
                .map(|s| {
                    s.iter()
                        .map(|q| f64::from(*q) * (uniform_below(&mut rng, 5) as f64 / 4.0 + shift))
                        .collect()
                })
                .collect()
        };
        for shift in [0.0, 0.05, 0.3] {
            let method = random_totals(shift);
            let baseline = random_totals(0.0);
            let differences: Vec<Vec<f64>> = method
                .iter()
                .zip(&baseline)
                .map(|(m, b)| m.iter().zip(b).map(|(m, b)| m - b).collect())
                .collect();
            let reference = swaps.two_sided_p(&differences).unwrap();
            let sizes_vec = sizes.to_vec();
            let observed = mean_over_strata(&method, &sizes_vec) - mean_over_strata(&baseline, &sizes_vec);
            let fast = two_sided_p_of_difference(
                observed,
                &swaps.signed_means(&method).unwrap(),
                &swaps.signed_means(&baseline).unwrap(),
            );
            assert_eq!(fast, reference, "shift {shift}");
        }
        // A method against itself: every permutation ties with the observed zero.
        let same = random_totals(0.0);
        let signed = swaps.signed_means(&same).unwrap();
        assert_eq!(two_sided_p_of_difference(0.0, &signed, &signed), 1.0);
    }
}
