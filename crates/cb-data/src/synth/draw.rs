//! Drawing from the one seeded generator. Only integer ranges, uniform floats,
//! shuffles and square roots are used, so a seed gives the same draws on every
//! platform.

use super::SynthError;
use rand::Rng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand_chacha::ChaCha20Rng;

/// One item of a pool, or an error naming the empty pool.
pub fn pick<'a, T>(rng: &mut ChaCha20Rng, items: &'a [T], what: &str) -> Result<&'a T, SynthError> {
    items
        .choose(rng)
        .ok_or_else(|| SynthError::Pool(format!("nothing to draw from: {what}")))
}

/// Up to `n` different items of a pool, in random order.
pub fn pick_distinct<'a, T>(rng: &mut ChaCha20Rng, items: &'a [T], n: usize) -> Vec<&'a T> {
    let mut order: Vec<usize> = (0..items.len()).collect();
    order.shuffle(rng);
    order.into_iter().take(n).map(|i| &items[i]).collect()
}

/// True with probability `p`.
pub fn chance(rng: &mut ChaCha20Rng, p: f64) -> bool {
    rng.random::<f64>() < p
}

/// A whole number in `lo..=hi`; `lo` when the range is empty.
pub fn between(rng: &mut ChaCha20Rng, lo: i64, hi: i64) -> i64 {
    if hi <= lo { lo } else { rng.random_range(lo..=hi) }
}

/// A position in `0..n`; 0 when `n` is 0.
pub fn index(rng: &mut ChaCha20Rng, n: usize) -> usize {
    if n == 0 { 0 } else { rng.random_range(0..n) }
}

/// `count` flags of which exactly `set` are true, in random order.
pub fn exact_flags(rng: &mut ChaCha20Rng, count: usize, set: usize) -> Vec<bool> {
    let mut flags: Vec<bool> = (0..count).map(|i| i < set).collect();
    flags.shuffle(rng);
    flags
}
